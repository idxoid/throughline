"""Middleware protocol: pluggable pre/post-processing around runs and steps.

The flow applies middleware as an onion: the FIRST middleware in the list is
the OUTERMOST layer. Hooks:

    on_run_start(ctx, payload)          -> payload   (pre-processing, in order)
    on_step_start(ctx, step, payload)   -> payload   (per-step pre, in order)
    wrap_step(invoke, ctx, step)        -> invoke    (full control: retry, tracing)
    on_step_end(ctx, step, payload, output) -> output (per-step post, reverse order)
    on_step_error(ctx, step, payload, exc)  -> None | Handled(output)
    on_run_end(ctx, output)             -> output    (post-processing, reverse order)

Every hook is optional; returning None from the payload/output hooks means
"unchanged" — so observers never have to remember to return values.

Phases (``Middleware.phase``) document stack position. Flow construction
rejects ``short_circuit`` (run-level Cache) placed before ``security_ingress``
(ManifestGate) or ``policy`` — those hooks must not be skipped on a cache hit.
Recommended order: Observe/Metrics → ManifestGate → Policy → Cache → …

The EarlyReturn contract (formal — see Flow.run for the full spec):

  * EarlyReturn skips the remaining on_run_start hooks, the remaining steps
    and (if raised mid-step) that step's on_step_end hooks. It bypasses
    on_step_error and must never be retried or counted as an error —
    wrap_step implementations re-raise it untouched (see Retry, Metrics).
  * on_run_end is a FINALIZER SWEEP, not stack unwinding: it runs for every
    middleware exactly once, in reverse order, on success and on EarlyReturn
    alike — even for middleware whose on_run_start never got to run (e.g.
    everything inside a run-level Cache on a hit). Consequences for authors:
      - on_run_end must not assume on_run_start ran: guard your own run
        state with ctx.artifacts.get(...) / setdefault (see Cache, Quota);
      - ctx.short_circuited tells you the output came from EarlyReturn —
        use it when your on_run_end should account for skipped work
        (see Lineage, Snapshots).
  * On a real failure (any other exception) on_run_end does NOT run; the
    FlowError carries the ctx with everything collected so far.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .context import RunContext
from .step import Step


# Declared stack phases. List order is outermost-first; short_circuit middleware
# may EarlyReturn from on_run_start and skip everything after it — so phases
# that must always run (security_ingress, policy) cannot follow short_circuit.
PHASE_ORDER: tuple[str, ...] = (
    "observe",
    "security_ingress",
    "policy",
    "short_circuit",
    "default",
)

# Phases that must not be skipped by a run-level EarlyReturn ahead of them.
_PROTECTED_FROM_SHORT_CIRCUIT = frozenset({"security_ingress", "policy"})


class Handled:
    """Return from on_step_error to swallow the error and substitute output."""

    __slots__ = ("output",)

    def __init__(self, output: Any):
        self.output = output


class Middleware:
    name: str = "middleware"
    phase: str = "default"

    def on_run_start(self, ctx: RunContext, payload: Any) -> Any:
        return payload

    def on_step_start(self, ctx: RunContext, step: Step, payload: Any) -> Any:
        return payload

    def wrap_step(self, invoke: Callable[[Any], Any], ctx: RunContext,
                  step: Step) -> Callable[[Any], Any]:
        return invoke

    def on_step_end(self, ctx: RunContext, step: Step, payload: Any, output: Any) -> Any:
        return output

    def on_step_error(self, ctx: RunContext, step: Step, payload: Any,
                      exc: Exception) -> Handled | None:
        return None

    def on_run_end(self, ctx: RunContext, output: Any) -> Any:
        return output

    def __repr__(self) -> str:  # helps in flow reprs / debugging
        return f"<{type(self).__name__}>"


def check_middleware_order(middleware: Sequence[Middleware]) -> None:
    """Raise ``MiddlewareOrderError`` if short-circuit can bypass protected ingress.

    Run-level Cache (phase ``short_circuit``) raises EarlyReturn from
    ``on_run_start``, which skips remaining start hooks. ManifestGate
    (``security_ingress``) and Policy (``policy``) must therefore appear
    earlier in the stack (outer / before the short-circuiter).
    """
    short_circuit: Middleware | None = None
    for mw in middleware:
        phase = getattr(mw, "phase", "default") or "default"
        if phase == "short_circuit":
            if short_circuit is None:
                short_circuit = mw
            continue
        if short_circuit is not None and phase in _PROTECTED_FROM_SHORT_CIRCUIT:
            from .errors import MiddlewareOrderError
            raise MiddlewareOrderError(
                f"{type(mw).__name__} (phase={phase!r}) cannot follow "
                f"{type(short_circuit).__name__} (phase='short_circuit'): "
                f"a run-level cache hit would skip {type(mw).__name__}. "
                f"Recommended order: Observe/Metrics → ManifestGate → "
                f"Policy → Cache.",
                earlier=type(short_circuit).__name__,
                later=type(mw).__name__,
            )
