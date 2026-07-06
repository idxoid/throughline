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

from typing import Any, Callable

from .context import RunContext
from .step import Step


class Handled:
    """Return from on_step_error to swallow the error and substitute output."""

    __slots__ = ("output",)

    def __init__(self, output: Any):
        self.output = output


class Middleware:
    name: str = "middleware"

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
