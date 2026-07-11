"""Adapters: onboard third-party objects as steps without importing frameworks.

`wrap(obj)` duck-types the object against known interfaces, in priority order:

    invoke(x)                    LangChain Runnable / LCEL chains / LangGraph apps
    query(x)                     LlamaIndex query engines
    retrieve(x)                  LlamaIndex/LangChain retrievers
    get_relevant_documents(x)    classic LangChain retrievers
    search(x)                    vector stores / search clients
    run(x)                       many agent frameworks (incl. legacy LangChain)
    complete(x) / generate(x)    LLM client objects
    __call__(x)                  anything callable

No framework is imported — only the object's own methods are used, so any
library (or your in-house code) onboards with one line:

    throughline.wrap(my_langchain_chain)
    throughline.wrap(my_llamaindex_engine, unwrap=lambda r: r.response)

Duck typing is implicit while it works and explicit when it does not:
`wrap` fails *at wrap time* with the full detection trace (what was tried,
what the object actually has, how to force a method), and `explain(obj)`
shows the decision — detected method, skipped candidates, call convention —
before anything runs. `throughline doctor <preset>` runs the same inspection
over a whole preset.
"""

from __future__ import annotations

from typing import Any, Callable

from ..errors import WrapError
from ..step import Step, _adapt_callable

METHOD_PRIORITY = (
    "invoke",
    "query",
    "retrieve",
    "get_relevant_documents",
    "search",
    "run",
    "complete",
    "generate",
)


def _public_callables(obj: Any) -> list[str]:
    names = []
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            if callable(getattr(obj, attr)):
                names.append(attr)
        except Exception:
            continue
    return names


def _detect(obj: Any, method: str | None = None) -> dict:
    """The detection decision, as data. Raises WrapError with the full trace."""
    chosen = method
    skipped: list[str] = []
    if chosen is None:
        for candidate in METHOD_PRIORITY:
            if callable(getattr(obj, candidate, None)):
                if chosen is None:
                    chosen = candidate
                else:
                    skipped.append(candidate)
    elif not callable(getattr(obj, chosen, None)) and chosen != "__call__":
        found = _public_callables(obj)
        raise WrapError(
            f"cannot adapt {type(obj).__name__}: forced method {chosen!r} "
            f"is not callable on it.\n"
            f"  Object has: {', '.join(found) or '(no public callables)'}",
            tried=(chosen,), found=found)
    if chosen is None and callable(obj):
        chosen = "__call__"
    if chosen is None:
        found = _public_callables(obj)
        hint = f"\n  Hint: tl.wrap(obj, method={found[0]!r})" if found else ""
        raise WrapError(
            f"cannot adapt {type(obj).__name__}: none of the known interfaces "
            f"found and the object is not callable.\n"
            f"  Tried (in order): {', '.join(METHOD_PRIORITY)}, __call__\n"
            f"  Object has: {', '.join(found) or '(no public callables)'}{hint}",
            tried=METHOD_PRIORITY + ("__call__",), found=found)
    return {"method": chosen, "skipped": skipped, "type": type(obj).__name__}


def explain(obj: Any, method: str | None = None) -> dict:
    """Show the wrap decision without wrapping: detected method, skipped
    candidates, object type. Raises WrapError (with the trace) if undetectable.
    """
    return _detect(obj, method)


def render_explain(obj: Any, method: str | None = None) -> str:
    """Human-readable form of ``explain`` — used by `throughline doctor`."""
    try:
        decision = _detect(obj, method)
    except WrapError as exc:
        return f"UNADAPTABLE: {exc}"
    parts = [f"detected: {decision['method']}()"]
    if decision["skipped"]:
        parts.append(f"skipped (lower priority): {', '.join(decision['skipped'])}")
    return f"{decision['type']}: " + "; ".join(parts)


def wrap(obj: Any, name: str | None = None, method: str | None = None,
         unwrap: Callable[[Any], Any] | None = None, **call_kwargs: Any) -> Step:
    """Adapt a foreign object into a Step. Fails fast, at wrap time.

    Args:
        obj:     the third-party object (chain, engine, retriever, agent, client).
        name:    step name (defaults to the object's class name).
        method:  force a specific method instead of duck-typed discovery.
        unwrap:  post-process the raw return value (e.g. ``lambda r: r.content``).
        call_kwargs: extra keyword arguments passed on every call.
    """
    decision = _detect(obj, method)
    chosen = decision["method"]
    bound = getattr(obj, chosen) if chosen != "__call__" else obj
    base = _adapt_callable(bound)

    def fn(payload, ctx):
        result = base(payload, ctx) if not call_kwargs else bound(payload, **call_kwargs)
        return unwrap(result) if unwrap else result

    step_name = name or type(obj).__name__
    return Step(fn=fn, name=step_name,
                meta={"adapter": chosen, "wrapped": type(obj).__name__,
                      "skipped": decision["skipped"], "unwrap": unwrap is not None})


from . import llm, rag, transcripts  # noqa: E402  (re-export convenience)

# NOTE: serving MCP is not an adapter — adapters bring components INTO flows;
# MCP *serving* lives in throughline.contrib.mcp (fully optional).
# adapters.mcp brings MCP *tools* into flows; adapters.transcripts normalizes
# Claude Code / Cursor / Codex session logs for agent-audit.

__all__ = [
    "wrap", "explain", "render_explain", "llm", "rag", "transcripts",
    "METHOD_PRIORITY",
]
