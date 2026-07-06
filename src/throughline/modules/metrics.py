"""Metrics: counters + value observations with per-step timing built in.

Steps report domain metrics through the context without importing anything:

    def answer(payload, ctx):
        ctx.metric("llm.output_tokens", n)          # counter
        ctx.metric("retrieval.score", s, kind="observe")  # distribution
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

from ..context import RunContext
from ..errors import EarlyReturn
from ..middleware import Middleware
from ..step import Step


class Metrics:
    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.observations: dict[str, dict] = {}

    def incr(self, name: str, value: float = 1) -> None:
        self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        agg = self.observations.setdefault(
            name, {"count": 0, "total": 0.0, "min": float("inf"), "max": float("-inf")})
        agg["count"] += 1
        agg["total"] += value
        agg["min"] = min(agg["min"], value)
        agg["max"] = max(agg["max"], value)

    @contextmanager
    def timeit(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started)

    def snapshot(self) -> dict:
        observations = {
            name: {**agg, "mean": agg["total"] / agg["count"] if agg["count"] else 0.0}
            for name, agg in self.observations.items()
        }
        return {"counters": dict(self.counters), "observations": observations}


class MetricsMiddleware(Middleware):
    """Publishes a Metrics collector to ctx.artifacts["metrics"] and times steps."""

    name = "metrics"

    def __init__(self, collector: Metrics | None = None):
        self._external = collector  # share one collector across runs if given

    def on_run_start(self, ctx: RunContext, payload):
        ctx.artifacts["metrics"] = self._external or Metrics()
        ctx.artifacts["metrics"].incr("runs")
        return payload

    def wrap_step(self, invoke, ctx: RunContext, step: Step):
        metrics: Metrics = ctx.artifacts["metrics"]

        def timed(payload):
            metrics.incr("steps")
            metrics.incr(f"step.{step.name}.calls")
            started = time.perf_counter()
            try:
                return invoke(payload)
            except EarlyReturn:
                raise  # control flow, not an error
            except Exception:
                metrics.incr("errors")
                metrics.incr(f"step.{step.name}.errors")
                raise
            finally:
                metrics.observe(f"step.{step.name}.seconds", time.perf_counter() - started)
        return timed
