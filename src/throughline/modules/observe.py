"""Observability: event sinks + optional OpenTelemetry bridge.

The flow core already emits structured events (run_started, step_finished,
step_failed, ...) into ctx.events. This middleware attaches sinks to that bus
and always keeps an in-memory record at ctx.artifacts["events"], so
`result.events` works out of the box.

Sinks never dump full step payloads — only the event bus. Console and JSONL
still truncate oversized *event field* values so a verbose trail cannot
blow up stderr or a log file when a step emits a large diagnostic string.
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

# Default caps for human/file sinks. MemorySink keeps full events (bounded
# by ``memory_limit`` count, not size) for ``result.events``.
_CONSOLE_MAX_VALUE_CHARS = 200
_JSONL_MAX_VALUE_CHARS = 2_000


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars == 1:
        return "…"
    return text[: max_chars - 1] + "…"


def _truncate_value(value: Any, max_chars: int) -> Any:
    """Bound string / nested JSON size for log sinks (not MemorySink)."""
    if max_chars <= 0:
        return value
    if isinstance(value, str):
        return _truncate_text(value, max_chars)
    if isinstance(value, float):
        return value
    if isinstance(value, (int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _truncate_value(v, max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate_value(v, max_chars) for v in value]
    return _truncate_text(str(value), max_chars)


def _truncate_event(event: dict, max_chars: int) -> dict:
    return {key: _truncate_value(value, max_chars) for key, value in event.items()}


class MemorySink:
    """Keeps the last ``limit`` events in memory."""

    def __init__(self, limit: int = 10_000):
        self.limit = limit
        self.events: list[dict] = []

    def __call__(self, event: dict) -> None:
        self.events.append(event)
        if len(self.events) > self.limit:
            del self.events[: len(self.events) - self.limit]


class NullSink:
    """Discard events — useful when only the in-memory bus is wanted."""

    def __call__(self, event: dict) -> None:
        return


class ConsoleSink:
    """Compact one-line-per-event rendering for humans."""

    INTERESTING = {"run_started", "run_finished", "run_failed", "step_started",
                   "step_finished", "step_failed", "step_retry", "step_recovered",
                   "validation_failed", "branch_taken"}

    def __init__(self, stream: TextIO | None = None, verbose: bool = False,
                 max_value_chars: int = _CONSOLE_MAX_VALUE_CHARS):
        self.stream = stream or sys.stderr
        self.verbose = verbose
        self.max_value_chars = max_value_chars

    def __call__(self, event: dict) -> None:
        if not self.verbose and event["type"] not in self.INTERESTING:
            return
        clock = time.strftime("%H:%M:%S", time.localtime(event["ts"]))
        extras = " ".join(
            f"{key}={self._fmt(value)}"
            for key, value in event.items()
            if key not in ("ts", "type", "run_id", "flow")
        )
        print(f"{clock} [{event.get('flow', '-')}] {event['type']} {extras}".rstrip(),
              file=self.stream)

    def _fmt(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return _truncate_text(str(value), self.max_value_chars)


class JsonlSink:
    """Appends every event as a JSON line — feed it to jq / your log shipper.

    Large string / nested values are truncated (``max_value_chars``) so an
    audit trail cannot grow with tool_result-sized diagnostics on the bus.
    """

    def __init__(self, path: str | Path,
                 max_value_chars: int = _JSONL_MAX_VALUE_CHARS):
        self.path = Path(path)
        self.max_value_chars = max_value_chars

    def __call__(self, event: dict) -> None:
        safe = _truncate_event(event, self.max_value_chars)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(safe, ensure_ascii=False, default=str) + "\n")


class Observe(Middleware):
    """Attach sinks to the run's event bus; optionally bridge to OpenTelemetry.

    Args:
        sinks: callables taking an event dict. Strings are shortcuts:
               "console", "console:verbose", "null", "path/to/file.jsonl".
        otel:  if True, wrap each step in an OpenTelemetry span
               (needs `opentelemetry-api` installed; degrades gracefully).
    """

    name = "observe"
    phase = "observe"

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
                self._tracer = trace.get_tracer("throughline")
            except ImportError:
                self.otel = False

    @staticmethod
    def _coerce(sink: Any) -> Callable[[dict], None]:
        if callable(sink):
            return sink
        if sink in ("null", "none"):
            return NullSink()
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
                span.set_attribute("throughline.run_id", ctx.run_id)
                span.set_attribute("throughline.step", step.name)
                return invoke(payload)
        return traced
