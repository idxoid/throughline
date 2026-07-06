"""Debugging aids: opt-in payload history and foreign-object leak detection.

Core invariant: stock middleware never retain references to payload versions
between steps — at any moment only one or two versions are alive, memory does
not grow with pipeline length. ``Snapshots`` is the single sanctioned way to
break that invariant, consciously, when you need the full version chain for
debugging. You pay with memory the moment you plug it in.

``StrictOutputs`` catches the classic distant failure: a forgotten
``unwrap=`` lets a framework object (LlamaIndex Response, LangChain message)
leak into the payload and blow up three steps later with an obscure error.
The point of failure is far from the point of cause; this middleware names
the cause at the step that produced it.
"""

from __future__ import annotations

import copy as _copy
from fnmatch import fnmatch
from typing import Any

from ..context import RunContext
from ..middleware import Middleware
from ..step import Step
from ..store import ArtifactRef
from .citations import EvidenceChunk

#: The formal "plain" contract — see StrictOutputs for the full definition.
_SCALARS = (str, bytes, int, float, bool, type(None))
_CONTRACTS = (ArtifactRef, EvidenceChunk)
_CONTAINERS = (dict, list, tuple)


class Snapshots(Middleware):
    """Opt-in: record every payload version, one per step.

    Args:
        deep:         deepcopy each version (safe against later in-place
                      mutation, costs more memory). Default False: keep refs.
        max_versions: ring-buffer cap; oldest snapshots are dropped.

    Snapshots land in ``ctx.artifacts["snapshots"]`` as (step_name, payload)
    pairs; ``result.ctx.artifacts["snapshots"]`` after the run.
    """

    name = "snapshots"

    def __init__(self, deep: bool = False, max_versions: int = 50):
        self.deep = deep
        self.max_versions = max_versions

    def _record(self, ctx: RunContext, label: str, payload: Any) -> None:
        trail = ctx.artifacts.setdefault("snapshots", [])
        trail.append((label, _copy.deepcopy(payload) if self.deep else payload))
        if len(trail) > self.max_versions:
            del trail[: len(trail) - self.max_versions]

    def on_run_start(self, ctx: RunContext, payload):
        self._record(ctx, "input", payload)
        return payload

    def on_step_end(self, ctx: RunContext, step: Step, payload, output):
        self._record(ctx, step.name, output)
        return output

    def on_run_end(self, ctx: RunContext, output):
        # EarlyReturn contract: the substituted output never passed through
        # on_step_end, so the trail would be missing the value the caller
        # actually received. on_run_start may not have run either (e.g.
        # Snapshots inside a run-level Cache) — the trail then starts here.
        if ctx.short_circuited:
            self._record(ctx, "early_return", output)
        return output


class StrictOutputs(Middleware):
    """Warn (or raise) when a step's output violates the *plain data* contract.

    The contract, formally — a value is plain iff every node of its object
    graph, reached through dict keys/values and list/tuple items, is:

      * a scalar:            str, bytes, int, float, bool, None;
      * a container:         dict, list, tuple (checked recursively);
      * a followers contract: ArtifactRef, EvidenceChunk;
      * one of your types:   anything passed in ``allow=``.

    Everything else — at any depth — is *foreign*: usually a framework object
    (LlamaIndex Response, LangChain message) that should have been
    ``unwrap``-ed. The violation names the offender's exact path
    (``$.answer[3].meta``) at the step that produced it — not three steps
    later where it finally crashes.

    Precise semantics:

      * checks OUTPUTS in ``on_step_end``. Onion position matters: this
        middleware sees the output after any transforms applied by middleware
        listed after it (inner layers run their on_step_end first).
      * a short-circuited run's substituted output never passes on_step_end
        (EarlyReturn contract), so it is checked in ``on_run_end`` instead,
        reported as step "early_return" — only when ``step=None``, since a
        step pattern means you scoped the check deliberately.
      * traversal is cycle-safe and budgeted: at most ``max_nodes`` nodes per
        output are inspected (a ``foreign_scan_truncated`` event tells you
        the guarantee became partial); the first offender wins.
      * dict keys are checked too — a non-scalar key is always a leak.

    Args:
        on_foreign: "warn" (default) — emit a ``foreign_output`` event and a
                    violation; "raise" — fail at the offending step.
        allow:      extra types that count as plain (your value objects).
        step:       fnmatch pattern of step names to check (None = all).
        max_nodes:  traversal budget per checked output.
    """

    name = "strict-outputs"

    def __init__(self, on_foreign: str = "warn", allow: tuple = (),
                 step: str | None = None, max_nodes: int = 10_000):
        if on_foreign not in ("warn", "raise"):
            raise ValueError(f"on_foreign must be 'warn' or 'raise', got {on_foreign!r}")
        self.on_foreign = on_foreign
        self.allow = tuple(allow)
        self.step = step
        self.max_nodes = max_nodes

    def on_step_end(self, ctx: RunContext, step: Step, payload, output):
        if self.step is None or fnmatch(step.name, self.step):
            self._check(ctx, step.name, output)
        return output

    def on_run_end(self, ctx: RunContext, output):
        # EarlyReturn contract: the substituted output skipped on_step_end.
        if ctx.short_circuited and self.step is None:
            self._check(ctx, "early_return", output)
        return output

    # -- the contract -----------------------------------------------------------
    def _check(self, ctx: RunContext, label: str, output: Any) -> None:
        offender = self._find_foreign(ctx, label, output)
        if offender is None:
            return
        path, type_name = offender
        message = (f"step {label!r} produced a foreign object: {type_name} "
                   f"at {path}; forgot unwrap= on a wrapped component?")
        ctx.emit("foreign_output", step=label, path=path, output_type=type_name)
        ctx.artifacts.setdefault("violations", []).append(message)
        if self.on_foreign == "raise":
            from ..errors import ValidationError
            raise ValidationError(message, step=label)

    def _plain_leaf(self, value: Any) -> bool:
        return isinstance(value, _SCALARS + _CONTRACTS) or isinstance(value, self.allow)

    def _find_foreign(self, ctx: RunContext, label: str,
                      root: Any) -> tuple[str, str] | None:
        """Iterative DFS over the object graph; returns (path, type) of the
        first foreign node, None if the graph is plain (within budget)."""
        stack: list[tuple[str, Any]] = [("$", root)]
        seen: set[int] = set()
        budget = self.max_nodes
        while stack:
            if budget <= 0:
                ctx.emit("foreign_scan_truncated", step=label,
                         max_nodes=self.max_nodes)
                return None
            budget -= 1
            path, value = stack.pop()
            if self._plain_leaf(value):
                continue
            if isinstance(value, dict):
                if id(value) in seen:
                    continue
                seen.add(id(value))
                for key, item in value.items():
                    if not isinstance(key, _SCALARS):
                        return (f"{path}.<key>", type(key).__name__)
                    stack.append((f"{path}.{key}", item))
                continue
            if isinstance(value, (list, tuple)):
                if id(value) in seen:
                    continue
                seen.add(id(value))
                for index, item in enumerate(value):
                    stack.append((f"{path}[{index}]", item))
                continue
            return (path, type(value).__name__)
        return None
