"""Step: the atomic unit of a flow.

Anything callable becomes a step. ``as_step`` also accepts registry names and
foreign objects (LangChain runnables, LlamaIndex engines, retrievers, agents)
via ``followers.adapters.wrap`` duck-typing. Composites (map_step, parallel,
branch) build fan-out/routing on top without turning Flow into a DAG engine.
"""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .context import RunContext
from .errors import FlowError


@dataclass
class Step:
    fn: Callable  # normalized to (payload, ctx) -> output
    name: str = "step"
    meta: dict = field(default_factory=dict)

    def __call__(self, payload: Any = None, ctx: RunContext | None = None) -> Any:
        """Steps are directly callable (ad-hoc context) — handy in tests/REPL."""
        return _run_fn(self.fn, payload, ctx or RunContext(flow=f"adhoc:{self.name}"))

    def rename(self, name: str) -> "Step":
        return Step(fn=self.fn, name=name, meta=dict(self.meta))

    @property
    def effects(self) -> tuple[str, ...] | None:
        """Declared side effects: None = unknown, () = declared pure,
        ("db.write", ...) = effectful. See ``step(effects=...)``."""
        return self.meta.get("effects")


def _normalize_effects(effects: Any) -> tuple[str, ...] | None:
    """Canonicalize the ``effects`` declaration.

    None -> unknown (no declaration); "pure" / () -> declared pure;
    True -> effectful without labels; a string or iterable -> effect labels.
    """
    if effects is None:
        return None
    if effects is True:
        return ("unlabeled",)
    if effects == "pure":
        return ()
    if isinstance(effects, str):
        return (effects,)
    return tuple(effects)


def _run_fn(fn: Callable, payload: Any, ctx: RunContext) -> Any:
    out = fn(payload, ctx)
    if inspect.iscoroutine(out):
        # Sync-first core with an async bridge: async steps just work as long
        # as the caller is not already inside a running event loop.
        try:
            out = asyncio.run(out)
        except RuntimeError as exc:
            raise FlowError(
                "async step called from a running event loop; "
                "run the flow from sync code or wrap the step yourself"
            ) from exc
    return out


def _adapt_callable(fn: Callable) -> Callable:
    """Normalize a user callable to the (payload, ctx) calling convention."""
    if inspect.ismethoddescriptor(fn) or inspect.isbuiltin(fn):
        # e.g. str.strip: signature reports (self, ...) but it is called fn(payload)
        return lambda payload, ctx: fn(payload)
    try:
        sig = inspect.signature(fn)
        positional = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        has_var = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
    except (ValueError, TypeError):  # builtins without signatures
        return lambda payload, ctx: fn(payload)
    if has_var or len(positional) >= 2:
        return fn
    return lambda payload, ctx: fn(payload)


def step(name: str | None = None,
         effects: Any = None) -> Callable[[Callable], Step]:
    """Decorator: turn a function into a named Step.

        @step("clean")
        def clean(text): ...

        @step("save", effects="db.write")     # declares a side effect
        def save(record): ...

        @step("format", effects="pure")       # declares purity
        def format(record): ...

    ``effects`` is a declaration, not an inference: None (default) = unknown,
    "pure" or () = pure, a label / iterable of labels / True = effectful.
    Cache uses it as a guard — a cached hit skips the step, and skipped side
    effects are silent data loss.
    """
    def decorate(fn: Callable) -> Step:
        meta = {} if effects is None else {"effects": _normalize_effects(effects)}
        return Step(fn=_adapt_callable(fn), meta=meta,
                    name=name or getattr(fn, "__name__", "step"))
    return decorate


def as_step(obj: Any, name: str | None = None, effects: Any = None) -> Step:
    """Coerce anything step-like into a Step.

    Accepts: Step | callable | registry name (str) | foreign object with a
    recognizable method (invoke/query/retrieve/run/...).
    ``effects`` declares side effects the same way ``step()`` does.
    """
    result = _as_step(obj, name)
    if effects is not None:  # fresh Step: never mutate a caller-owned one
        result = Step(fn=result.fn, name=result.name,
                      meta={**result.meta, "effects": _normalize_effects(effects)})
    return result


def _as_step(obj: Any, name: str | None = None) -> Step:
    if isinstance(obj, Step):
        return obj.rename(name) if name else obj
    if isinstance(obj, str):
        from .registry import resolve  # local import: registry depends on step
        resolved = resolve(obj)
        return _as_step(resolved, name or obj)
    from .adapters import METHOD_PRIORITY, wrap
    plain_callable = (inspect.isroutine(obj) or  # functions, methods, builtins, descriptors
                      isinstance(obj, type(lambda: 0)))
    if not plain_callable and any(callable(getattr(obj, m, None)) for m in METHOD_PRIORITY):
        # foreign framework object: map to its semantically right method
        return wrap(obj, name=name)
    if callable(obj):
        return Step(fn=_adapt_callable(obj), name=name or getattr(obj, "__name__", "step"))
    return wrap(obj, name=name)  # raises with a helpful message


# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------

def map_step(inner: Any, name: str | None = None, workers: int = 1) -> Step:
    """Apply ``inner`` to every item of an iterable payload; returns a list.

    workers > 1 fans out over threads (useful for I/O-bound LLM calls).
    """
    inner_step = as_step(inner)
    step_name = name or f"map({inner_step.name})"

    def fn(payload: Any, ctx: RunContext) -> list:
        items = list(payload)
        ctx.emit("map_started", step=step_name, items=len(items))
        if workers <= 1:
            return [_run_fn(inner_step.fn, item, ctx) for item in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda item: _run_fn(inner_step.fn, item, ctx), items))

    return Step(fn=fn, name=step_name, meta={"kind": "map", "workers": workers})


def parallel(steps: dict[str, Any] | list, name: str | None = None,
             workers: int | None = None) -> Step:
    """Run several steps on the same payload; gather outputs.

    dict in -> dict of outputs keyed the same; list in -> list of outputs.
    """
    if isinstance(steps, dict):
        named = {key: as_step(value, key) for key, value in steps.items()}
    else:
        named = {str(i): as_step(value) for i, value in enumerate(steps)}
    step_name = name or f"parallel({','.join(named)})"
    as_list = not isinstance(steps, dict)

    def fn(payload: Any, ctx: RunContext) -> Any:
        def run_one(item):
            key, sub = item
            ctx.emit("parallel_branch", step=step_name, branch=sub.name)
            return key, _run_fn(sub.fn, payload, ctx)
        if workers and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = dict(pool.map(run_one, named.items()))
        else:
            results = dict(run_one(item) for item in named.items())
        return [results[k] for k in named] if as_list else results

    return Step(fn=fn, name=step_name, meta={"kind": "parallel"})


def branch(selector: Callable[[Any], Any], routes: dict[Any, Any],
           default: Any | None = None, name: str | None = None) -> Step:
    """Route the payload to one of ``routes`` based on ``selector(payload)``."""
    compiled = {key: as_step(value) for key, value in routes.items()}
    default_step = as_step(default) if default is not None else None
    step_name = name or f"branch({','.join(map(str, compiled))})"

    def fn(payload: Any, ctx: RunContext) -> Any:
        key = selector(payload)
        chosen = compiled.get(key, default_step)
        if chosen is None:
            raise FlowError(f"branch '{step_name}': no route for {key!r} and no default")
        ctx.emit("branch_taken", step=step_name, route=str(key), target=chosen.name)
        return _run_fn(chosen.fn, payload, ctx)

    return Step(fn=fn, name=step_name, meta={"kind": "branch"})
