"""Flow: an ordered chain of steps executed under a middleware stack."""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

from .context import Result, RunContext
from .errors import EarlyReturn, FlowError
from .middleware import Handled, Middleware, check_middleware_order
from .step import Step, as_step, _run_fn


class Flow:
    def __init__(self, steps: Iterable[Any], middleware: Sequence[Middleware] = (),
                 name: str = "flow", config: dict | None = None):
        self.steps: list[Step] = [as_step(s) for s in steps]
        self.middleware: list[Middleware] = list(middleware)
        check_middleware_order(self.middleware)
        self.name = name
        self.config = dict(config or {})

    # -- composition --------------------------------------------------------
    def then(self, step: Any, name: str | None = None) -> "Flow":
        """Return a new Flow with one more step appended."""
        return Flow(self.steps + [as_step(step, name)], self.middleware,
                    name=self.name, config=self.config)

    def use(self, *middleware: Middleware) -> "Flow":
        """Return a new Flow with extra middleware appended (inner layers)."""
        return Flow(self.steps, self.middleware + list(middleware),
                    name=self.name, config=self.config)

    # -- execution -----------------------------------------------------------
    def run(self, payload: Any = None, *, run_id: str | None = None,
            config: dict | None = None, ctx: RunContext | None = None) -> Result:
        """Execute the flow. The middleware/EarlyReturn contract, formally:

        Normal run:
          1. ``on_run_start`` hooks, in list order (outermost first).
          2. Per step: ``on_step_start`` in order -> ``wrap_step`` onion
             (first middleware outermost) -> ``on_step_end`` in reverse order.
          3. ``on_run_end`` hooks, in reverse order (innermost first).

        EarlyReturn (from any step, on_run_start, on_step_* or wrap_step):
          * remaining ``on_run_start`` hooks and all remaining steps are
            SKIPPED; if raised mid-step, that step's ``on_step_end`` hooks
            are skipped too;
          * it is control flow, not an error: never retried, never counted,
            ``on_step_error`` hooks are bypassed;
          * ``ctx.short_circuited`` is set to True;
          * ``on_run_end`` hooks still run — for EVERY middleware, in reverse
            order, exactly once. This is a finalizer sweep, not stack
            unwinding: a middleware's ``on_run_end`` fires even if its
            ``on_run_start`` never ran, so it MUST NOT assume its own run
            state exists (guard artifacts with .get / setdefault).

        Failure (any other exception):
          * ``on_step_error`` hooks may recover the step (``Handled``);
          * unrecovered: NO ``on_run_end`` hooks run — the error propagates
            as FlowError with the ctx attached (metrics/events collected so
            far stay inspectable on it).
        """
        if ctx is None:
            merged = {**self.config, **(config or {})}
            ctx = RunContext(flow=self.name, config=merged)
            if run_id:
                ctx.run_id = run_id
        started = time.perf_counter()
        try:
            try:
                for mw in self.middleware:
                    result = mw.on_run_start(ctx, payload)
                    payload = payload if result is None else result
                # emitted after the on_run_start hooks so that sinks subscribed
                # by middleware (Observe) actually see it; a short-circuited run
                # (e.g. cache hit) announces itself as run_short_circuited instead
                ctx.emit("run_started", steps=[s.name for s in self.steps])
                for index, step in enumerate(self.steps):
                    payload = self._invoke(ctx, step, index, payload)
                output = payload
            except EarlyReturn as stop:
                # control flow: skip remaining steps, keep on_run_end hooks
                ctx.short_circuited = True
                ctx.emit("run_short_circuited")
                output = stop.output
            for mw in reversed(self.middleware):
                result = mw.on_run_end(ctx, output)
                output = output if result is None else result
        except Exception as exc:
            ctx.emit("run_failed", error=repr(exc),
                     duration=time.perf_counter() - started)
            if isinstance(exc, FlowError):
                exc.ctx = exc.ctx or ctx
                raise
            raise FlowError(f"flow '{self.name}' failed: {exc}", ctx=ctx) from exc
        ctx.emit("run_finished", duration=time.perf_counter() - started)
        return Result(output=output, ctx=ctx)

    def _invoke(self, ctx: RunContext, step: Step, index: int, payload: Any) -> Any:
        for mw in self.middleware:
            result = mw.on_step_start(ctx, step, payload)
            payload = payload if result is None else result

        invoke = lambda p: _run_fn(step.fn, p, ctx)  # noqa: E731
        # reversed() so the first middleware in the list is the outermost layer
        for mw in reversed(self.middleware):
            wrapped = mw.wrap_step(invoke, ctx, step)
            invoke = invoke if wrapped is None else wrapped

        ctx.emit("step_started", step=step.name, index=index)
        started = time.perf_counter()
        try:
            output = invoke(payload)
        except EarlyReturn:
            raise  # control flow — not an error, bypass on_step_error hooks
        except Exception as exc:
            for mw in reversed(self.middleware):
                handled = mw.on_step_error(ctx, step, payload, exc)
                if isinstance(handled, Handled):
                    ctx.emit("step_recovered", step=step.name, index=index,
                             error=repr(exc), by=type(mw).__name__)
                    output = handled.output
                    break
            else:
                ctx.emit("step_failed", step=step.name, index=index, error=repr(exc),
                         duration=time.perf_counter() - started)
                if isinstance(exc, FlowError):
                    exc.step = exc.step or step.name
                    exc.ctx = exc.ctx or ctx
                    raise
                raise FlowError(f"step '{step.name}' failed: {exc}",
                                step=step.name, ctx=ctx) from exc

        for mw in reversed(self.middleware):
            result = mw.on_step_end(ctx, step, payload, output)
            output = output if result is None else result
        ctx.emit("step_finished", step=step.name, index=index,
                 duration=time.perf_counter() - started)
        return output

    def __repr__(self) -> str:
        chain = " -> ".join(s.name for s in self.steps) or "(empty)"
        return f"<Flow {self.name!r}: {chain}; middleware={self.middleware}>"
