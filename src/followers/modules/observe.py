"""Observability: event sinks + optional OpenTelemetry bridge.

The flow core already emits structured events (run_started, step_finished,
step_failed, ...) into ctx.events. This middleware attaches sinks to that bus
and always keeps an in-memory record at ctx.artifacts["events"], so
`result.events` works out of the box.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, TextIO

from ..context import RunContext
from ..middleware import Middleware
from ..step import Step


class MemorySink:
    """Keeps the last ``limit`` events in memory."""

    def __init__(self, limit: int = 10_000):
        self.limit = limit
        self.events: list[dict] = []

    def __call__(self, event: dict) -> None:
        self.events.append(event)
        if len(self.events) > self.limit:
            del self.events[: len(self.events) - self.limit]


class ConsoleSink:
    """Compact one-line-per-event rendering for humans."""

    INTERESTING = {"run_started", "run_finished", "run_failed", "step_started",
                   "step_finished", "step_failed", "step_retry", "step_recovered",
                   "validation_failed", "branch_taken"}

    def __init__(self, stream: TextIO | None = None, verbose: bool = False):
        self.stream = stream or sys.stderr
        self.verbose = verbose

    def __call__(self, event: dict) -> None:
        if not self.verbose and event["type"] not in self.INTERESTING:
            return
        clock = time.strftime("%H:%M:%S", time.localtime(event["ts"]))
        extras = " ".join(
            f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in event.items()
            if key not in ("ts", "type", "run_id", "flow")
        )
        print(f"{clock} [{event.get('flow', '-')}] {event['type']} {extras}".rstrip(),
              file=self.stream)


class JsonlSink:
    """Appends every event as a JSON line — feed it to jq / your log shipper."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __call__(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


class Observe(Middleware):
    """Attach sinks to the run's event bus; optionally bridge to OpenTelemetry.

    Args:
        sinks: callables taking an event dict. Strings are shortcuts:
               "console", "console:verbose", "path/to/file.jsonl".
        otel:  if True, wrap each step in an OpenTelemetry span
               (needs `opentelemetry-api` installed; degrades gracefully).
    """

    name = "observe"

    def __init__(self, *sinks: Callable[[dict], None] | str,
                 sink: str | None = None, otel: bool = False, memory_limit: int = 10_000):
        raw: list[Any] = list(sinks) + ([sink] if sink else [])
        self.sinks = [self._coerce(s) for s in raw]
        self.memory_limit = memory_limit
        self.otel = otel
        self._tracer = None
        if otel:
            try:
                from opentelemetry import trace  # type: ignore
                self._tracer = trace.get_tracer("followers")
            except ImportError:
                self.otel = False

    @staticmethod
    def _coerce(sink: Any) -> Callable[[dict], None]:
        if callable(sink):
            return sink
        if sink == "console":
            return ConsoleSink()
        if sink == "console:verbose":
            return ConsoleSink(verbose=True)
        if isinstance(sink, str):
            return JsonlSink(sink)
        raise TypeError(f"cannot interpret sink {sink!r}")

    def on_run_start(self, ctx: RunContext, payload):
        memory = MemorySink(self.memory_limit)
        ctx.artifacts["events"] = memory.events
        ctx.events.subscribe(memory)
        for sink in self.sinks:
            ctx.events.subscribe(sink)
        # run_started is emitted by the flow *after* the on_run_start hooks,
        # so everything subscribed here sees the full run. Keep Observe early
        # in the middleware list — outer layers' events must have a listener.
        ctx.emit("observe_attached", sinks=len(self.sinks) + 1)
        return payload

    def wrap_step(self, invoke, ctx: RunContext, step: Step):
        if not self.otel or self._tracer is None:
            return invoke

        def traced(payload):
            with self._tracer.start_as_current_span(f"{ctx.flow}.{step.name}") as span:
                span.set_attribute("followers.run_id", ctx.run_id)
                span.set_attribute("followers.step", step.name)
                return invoke(payload)
        return traced
