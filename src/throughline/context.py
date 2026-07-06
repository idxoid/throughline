"""Run context: the object every step and middleware receives.

Design: the context owns an EventBus (observability backbone), a free-form
``state`` dict for user code, and an ``artifacts`` dict where middleware
publish what they build (metrics, lineage ledger, captured events, ...).
Core stays decoupled from concrete modules: nothing here imports them.
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class EventBus:
    """Minimal synchronous pub/sub. Events are plain dicts."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[dict], None]] = []
        self._broken: set[int] = set()  # sinks already reported as failing

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._subscribers.append(fn)

    def unsubscribe(self, fn: Callable[[dict], None]) -> None:
        if fn in self._subscribers:
            self._subscribers.remove(fn)

    def emit(self, type: str, **fields: Any) -> dict:
        event = {"ts": time.time(), "type": type, **fields}
        for fn in list(self._subscribers):
            try:
                fn(event)
            except Exception as exc:  # a broken sink must never kill the run
                if id(fn) not in self._broken:  # ...but must not fail silently
                    self._broken.add(id(fn))
                    print(f"throughline: event sink {fn!r} raised {exc!r}; "
                          f"further errors from this sink are suppressed",
                          file=sys.stderr)
        return event


@dataclass
class RunContext:
    """Carried through a whole flow run and handed to every step/middleware."""

    flow: str = "flow"
    run_id: str = field(default_factory=new_run_id)
    config: dict = field(default_factory=dict)
    events: EventBus = field(default_factory=EventBus)
    state: dict = field(default_factory=dict)      # scratch space for user code
    artifacts: dict = field(default_factory=dict)  # published by middleware
    short_circuited: bool = False  # set by the flow when EarlyReturn ends the run

    # -- convenience -------------------------------------------------------
    def emit(self, type: str, **fields: Any) -> dict:
        return self.events.emit(type, run_id=self.run_id, flow=self.flow, **fields)

    def metric(self, name: str, value: float = 1, kind: str = "incr") -> None:
        """Record a metric if a metrics collector is attached; no-op otherwise.

        Lets steps report domain metrics (token counts, hit rates) without
        depending on whether MetricsMiddleware is installed.
        """
        metrics = self.artifacts.get("metrics")
        if metrics is None:
            return
        if kind == "incr":
            metrics.incr(name, value)
        else:
            metrics.observe(name, value)


@dataclass
class Result:
    """Returned by Flow.run: final output plus everything collected on the way."""

    output: Any
    ctx: RunContext

    @property
    def run_id(self) -> str:
        return self.ctx.run_id

    @property
    def metrics(self) -> dict:
        m = self.ctx.artifacts.get("metrics")
        return m.snapshot() if m is not None else {}

    @property
    def lineage(self):
        """LineageLedger if LineageMiddleware was installed, else None."""
        return self.ctx.artifacts.get("lineage")

    @property
    def events(self) -> list[dict]:
        return self.ctx.artifacts.get("events", [])

    @property
    def violations(self) -> list[str]:
        return self.ctx.artifacts.get("violations", [])
