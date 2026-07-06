"""Quota / cost-tracking middleware: hard budgets with an explicit scope.

Checks fire before every step, so an exhausted budget stops the flow BEFORE
the next expensive LLM/RAG call. Budgets:

  * limits:      {"metric.counter": max} — any counter steps report via
                 ctx.metric() (llm.calls, llm.output_tokens, retrieval.docs...)
  * cost + max_cost: unit prices per counter -> running cost (also published
                 as the "quota.cost" counter — your CostTracker)
  * max_seconds: wall-clock budget
  * max_steps:   how many steps may complete

Scope is EXPLICIT — a property of this middleware, never a side effect of
how something else is configured:

  * ``scope="run"`` (default): every budget covers one run. Robust by
    construction: consumption is measured as a delta against a baseline
    snapshot taken at run start, so even a Metrics collector shared across
    runs cannot silently turn run limits into lifetime limits.
  * ``scope="global"``: budgets cover the lifetime of this middleware
    instance across all runs (finished-run consumption is folded in under a
    lock; in-flight concurrent runs are counted approximately). "seconds"
    means cumulative time spent inside runs; "steps" means total steps.
    The published "quota.cost" counter is the scoped (lifetime) figure.

Need both? Stack two instances — scope composes like everything else:

    Quota(limits={"llm.calls": 20}, scope="run"),      # per-request ceiling
    Quota(max_cost=50.0, cost=PRICES, scope="global"), # daily-ish kill switch

On exceed: ``on_exceed="raise"`` aborts with QuotaExceeded (wrapped in
FlowError, reachable via ``err.__cause__``); ``on_exceed="return"`` finishes
the run early via EarlyReturn with ``fallback`` (a value, a callable
``(payload, ctx) -> value``, or — by default — the current intermediate
payload, i.e. a degraded-but-useful partial result).

``warn_at=0.8`` emits a one-shot-per-run "quota_warning" event per budget.

Works standalone (installs its own metrics collector if none present) and
composes with MetricsMiddleware.

Preset usage:

    [middleware.quota]
    scope = "run"           # or "global"
    max_seconds = 30
    max_cost = 0.25
    [middleware.quota.limits]
    "llm.calls" = 20
    [middleware.quota.cost]
    "llm.input_tokens" = 5e-6
    "llm.output_tokens" = 25e-6
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Callable

from ..context import RunContext
from ..errors import EarlyReturn, QuotaExceeded
from ..middleware import Middleware
from ..step import Step
from .metrics import Metrics


class Quota(Middleware):
    name = "quota"

    def __init__(self, limits: dict[str, float] | None = None,
                 max_cost: float | None = None, cost: dict[str, float] | None = None,
                 max_seconds: float | None = None, max_steps: int | None = None,
                 on_exceed: str = "raise",
                 fallback: Any | Callable[[Any, RunContext], Any] = None,
                 warn_at: float | None = 0.8,
                 scope: str = "run"):
        if on_exceed not in ("raise", "return"):
            raise ValueError("on_exceed must be 'raise' or 'return'")
        if scope not in ("run", "global"):
            raise ValueError(f"scope must be 'run' or 'global', got {scope!r}")
        self.limits = dict(limits or {})
        self.max_cost = max_cost
        self.cost = dict(cost or {})
        self.max_seconds = max_seconds
        self.max_steps = max_steps
        self.on_exceed = on_exceed
        self.fallback = fallback
        self.warn_at = warn_at
        self.scope = scope
        # lifetime accounting for scope="global": folded in at run end
        self._lifetime_counters: dict[str, float] = defaultdict(float)
        self._lifetime_seconds = 0.0
        self._lifetime_steps = 0
        self._fold_lock = threading.Lock()

    def _tracked(self) -> set[str]:
        return set(self.limits) | set(self.cost)

    # -- lifecycle -------------------------------------------------------------
    def on_run_start(self, ctx: RunContext, payload):
        # per-run state lives on the context: one middleware instance may
        # serve many (even concurrent) runs. The baseline snapshot makes
        # scope="run" robust against Metrics collectors shared across runs.
        metrics = ctx.artifacts.setdefault("metrics", Metrics())
        baseline = {name: metrics.counters.get(name, 0.0) for name in self._tracked()}
        ctx.artifacts["quota"] = {"started": time.monotonic(), "steps": 0,
                                  "warned": set(), "baseline": baseline}
        return payload

    def on_step_start(self, ctx: RunContext, step: Step, payload):
        state = ctx.artifacts.get("quota")
        if state is None:  # defensive: run() called without on_run_start? never normally
            return payload
        state["steps"] += 1
        for budget, spent, limit in self._budgets(ctx, state):
            if spent >= limit:
                self._exceeded(ctx, payload, step, budget, spent, limit)
            elif (self.warn_at is not None and budget not in state["warned"]
                  and limit > 0 and spent >= self.warn_at * limit):
                state["warned"].add(budget)
                ctx.emit("quota_warning", budget=budget, spent=spent, limit=limit,
                         scope=self.scope, fraction=round(spent / limit, 3))
        return payload

    def on_run_end(self, ctx: RunContext, output):
        # EarlyReturn contract: this is a finalizer sweep — state may be
        # absent (e.g. Quota inside a run-level Cache on a hit).
        state = ctx.artifacts.get("quota")
        metrics = ctx.artifacts.get("metrics")
        if state is None or metrics is None:
            return output
        deltas = self._run_deltas(metrics, state)
        if self.scope == "global":
            with self._fold_lock:
                for name, delta in deltas.items():
                    self._lifetime_counters[name] += delta
                self._lifetime_seconds += time.monotonic() - state["started"]
                self._lifetime_steps += state["steps"]
        # checks run before steps, so fold the last step's consumption into
        # the published cost counter (no abort here — the work is done)
        if self.cost:
            metrics.counters["quota.cost"] = self._spent_cost(deltas)
        return output

    # -- accounting --------------------------------------------------------------
    def _run_deltas(self, metrics: Metrics, state: dict) -> dict[str, float]:
        baseline = state["baseline"]
        return {name: metrics.counters.get(name, 0.0) - baseline.get(name, 0.0)
                for name in self._tracked()}

    def _spent(self, name: str, deltas: dict[str, float]) -> float:
        run_spent = deltas.get(name, 0.0)
        if self.scope == "global":
            return self._lifetime_counters[name] + run_spent
        return run_spent

    def _spent_cost(self, deltas: dict[str, float]) -> float:
        return sum(self._spent(name, deltas) * price
                   for name, price in self.cost.items())

    def _budgets(self, ctx: RunContext, state: dict) -> list[tuple[str, float, float]]:
        metrics = ctx.artifacts.get("metrics")
        deltas = (self._run_deltas(metrics, state) if metrics is not None
                  else {name: 0.0 for name in self._tracked()})
        budgets: list[tuple[str, float, float]] = []
        for name, limit in self.limits.items():
            budgets.append((name, self._spent(name, deltas), float(limit)))
        if self.cost:
            spent_cost = self._spent_cost(deltas)
            if metrics is not None:
                metrics.counters["quota.cost"] = spent_cost
            if self.max_cost is not None:
                budgets.append(("cost", spent_cost, float(self.max_cost)))
        if self.max_seconds is not None:
            elapsed = time.monotonic() - state["started"]
            if self.scope == "global":
                elapsed += self._lifetime_seconds
            budgets.append(("seconds", elapsed, float(self.max_seconds)))
        if self.max_steps is not None:
            steps = float(state["steps"] - 1)
            if self.scope == "global":
                steps += self._lifetime_steps
            budgets.append(("steps", steps, float(self.max_steps)))
        return budgets

    def _exceeded(self, ctx: RunContext, payload, step: Step,
                  budget: str, spent: float, limit: float) -> None:
        ctx.metric("quota.exceeded")
        ctx.emit("quota_exceeded", budget=budget, spent=round(spent, 6),
                 limit=limit, before_step=step.name, policy=self.on_exceed,
                 scope=self.scope)
        if self.on_exceed == "return":
            if callable(self.fallback):
                value = self.fallback(payload, ctx)
            elif self.fallback is not None:
                value = self.fallback
            else:
                value = payload  # degraded mode: return what we have so far
            raise EarlyReturn(value)
        raise QuotaExceeded(
            f"{self.scope} budget {budget!r} exhausted before step {step.name!r}: "
            f"spent {spent:g} >= limit {limit:g}",
            budget=budget, spent=spent, limit=limit, scope=self.scope)
