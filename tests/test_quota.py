import time
import unittest

import throughline as tl
from throughline.modules import MetricsMiddleware, Observe, Quota


def llm_step(name="llm", tokens=100):
    def fn(payload, ctx):
        ctx.metric("llm.calls")
        ctx.metric("llm.output_tokens", tokens)
        return payload
    return tl.as_step(fn, name)


class QuotaLimits(unittest.TestCase):
    def test_counter_limit_aborts_before_next_step(self):
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2"), llm_step("llm3")],
                       middleware=[MetricsMiddleware(),
                                   Quota(limits={"llm.calls": 2})])
        with self.assertRaises(tl.FlowError) as caught:
            flow.run("x")
        cause = caught.exception.__cause__
        self.assertIsInstance(cause, tl.QuotaExceeded)
        self.assertEqual(cause.budget, "llm.calls")
        self.assertEqual(cause.spent, 2)
        # exactly two llm calls happened, the third was blocked
        metrics = caught.exception.ctx.artifacts["metrics"].snapshot()
        self.assertEqual(metrics["counters"]["llm.calls"], 2)

    def test_under_limit_passes(self):
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2")],
                       middleware=[MetricsMiddleware(),
                                   Quota(limits={"llm.calls": 5})])
        self.assertEqual(flow.run("x").output, "x")

    def test_works_without_metrics_middleware(self):
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2"), llm_step("llm3")],
                       middleware=[Quota(limits={"llm.calls": 2})])
        with self.assertRaises(tl.FlowError) as caught:
            flow.run("x")
        self.assertIsInstance(caught.exception.__cause__, tl.QuotaExceeded)

    def test_max_steps(self):
        flow = tl.Flow([lambda p: p, lambda p: p, lambda p: p],
                       middleware=[Quota(max_steps=2)])
        with self.assertRaises(tl.FlowError) as caught:
            flow.run("x")
        self.assertEqual(caught.exception.__cause__.budget, "steps")

    def test_max_seconds(self):
        def slow(payload, ctx):
            time.sleep(0.03)
            return payload
        flow = tl.Flow([tl.as_step(slow, "slow"), lambda p: p],
                       middleware=[Quota(max_seconds=0.01)])
        with self.assertRaises(tl.FlowError) as caught:
            flow.run("x")
        self.assertEqual(caught.exception.__cause__.budget, "seconds")


class QuotaCost(unittest.TestCase):
    def test_cost_budget_and_tracking(self):
        flow = tl.Flow([llm_step("llm1", tokens=1000), llm_step("llm2", tokens=1000),
                        llm_step("llm3", tokens=1000)],
                       middleware=[MetricsMiddleware(),
                                   Quota(cost={"llm.output_tokens": 0.0001},
                                         max_cost=0.15)])
        with self.assertRaises(tl.FlowError) as caught:
            flow.run("x")
        cause = caught.exception.__cause__
        self.assertEqual(cause.budget, "cost")
        self.assertAlmostEqual(cause.spent, 0.2)
        metrics = caught.exception.ctx.artifacts["metrics"].snapshot()
        self.assertAlmostEqual(metrics["counters"]["quota.cost"], 0.2)

    def test_cost_tracked_without_budget(self):
        flow = tl.Flow([llm_step(tokens=500), lambda p: p],
                       middleware=[MetricsMiddleware(),
                                   Quota(cost={"llm.output_tokens": 0.001})])
        result = flow.run("x")
        self.assertAlmostEqual(result.metrics["counters"]["quota.cost"], 0.5)

    def test_final_cost_includes_last_step(self):
        # a single-step flow: consumption happens after the only pre-step check
        flow = tl.Flow([llm_step(tokens=500)],
                       middleware=[MetricsMiddleware(),
                                   Quota(cost={"llm.output_tokens": 0.001})])
        result = flow.run("x")
        self.assertAlmostEqual(result.metrics["counters"]["quota.cost"], 0.5)


class QuotaReturnPolicy(unittest.TestCase):
    def test_return_gives_intermediate_payload(self):
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2"), llm_step("llm3")],
                       middleware=[MetricsMiddleware(),
                                   Quota(limits={"llm.calls": 1}, on_exceed="return")])
        result = flow.run("payload-so-far")
        self.assertEqual(result.output, "payload-so-far")

    def test_return_with_fallback_value(self):
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2")],
                       middleware=[MetricsMiddleware(),
                                   Quota(limits={"llm.calls": 1}, on_exceed="return",
                                         fallback={"answer": "quota hit"})])
        self.assertEqual(flow.run("x").output, {"answer": "quota hit"})

    def test_return_with_fallback_callable(self):
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2")],
                       middleware=[MetricsMiddleware(),
                                   Quota(limits={"llm.calls": 1}, on_exceed="return",
                                         fallback=lambda p, ctx: f"partial:{p}")])
        self.assertEqual(flow.run("x").output, "partial:x")


class QuotaWarnings(unittest.TestCase):
    def test_warning_emitted_once(self):
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2"), llm_step("llm3")],
                       middleware=[Observe(), MetricsMiddleware(),
                                   Quota(limits={"llm.calls": 10}, warn_at=0.1)])
        result = flow.run("x")
        warnings = [e for e in result.events if e["type"] == "quota_warning"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["budget"], "llm.calls")

    def test_exceeded_event_has_details(self):
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2")],
                       middleware=[Observe(), MetricsMiddleware(),
                                   Quota(limits={"llm.calls": 1}, on_exceed="return")])
        result = flow.run("x")
        exceeded = [e for e in result.events if e["type"] == "quota_exceeded"]
        self.assertEqual(len(exceeded), 1)
        self.assertEqual(exceeded[0]["before_step"], "llm2")


class QuotaScope(unittest.TestCase):
    """Scope is a property of the Quota itself, never a side effect of how
    Metrics is configured."""

    def test_run_scope_is_isolated_even_with_a_shared_collector(self):
        # Historically, sharing one Metrics collector across runs silently
        # turned per-run limits into lifetime limits. Baselines fix that:
        # scope="run" means THIS run, no matter how counters accumulate.
        from throughline.modules.metrics import Metrics
        shared = Metrics()
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2")],
                       middleware=[MetricsMiddleware(collector=shared),
                                   Quota(limits={"llm.calls": 2})])
        for _ in range(5):                      # would trip on run 2 before
            self.assertEqual(flow.run("x").output, "x")
        self.assertEqual(shared.counters["llm.calls"], 10)

    def test_global_scope_accumulates_across_runs(self):
        quota = Quota(limits={"llm.calls": 3}, scope="global")
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2")],
                       middleware=[MetricsMiddleware(), quota])
        self.assertEqual(flow.run("x").output, "x")     # lifetime: 2 calls
        with self.assertRaises(tl.FlowError) as caught:  # 2 + 1 >= 3 before llm2
            flow.run("x")
        cause = caught.exception.__cause__
        self.assertEqual(cause.budget, "llm.calls")
        self.assertEqual(cause.scope, "global")

    def test_global_cost_kill_switch(self):
        quota = Quota(cost={"llm.output_tokens": 0.001}, max_cost=0.15,
                      scope="global", on_exceed="return",
                      fallback={"answer": "global budget exhausted"})
        flow = tl.Flow([llm_step(tokens=100)],
                       middleware=[MetricsMiddleware(), quota])
        for _ in range(2):
            self.assertEqual(flow.run("x").output, "x")  # 0.1 then 0.2 total
        # pre-step check now sees lifetime 0.2 >= 0.15 -> degraded answer
        self.assertEqual(flow.run("x").output,
                         {"answer": "global budget exhausted"})

    def test_global_steps_accumulate(self):
        quota = Quota(max_steps=3, scope="global")
        flow = tl.Flow([lambda p: p, lambda p: p], middleware=[quota])
        flow.run("x")                                    # lifetime: 2 steps
        with self.assertRaises(tl.FlowError) as caught:
            flow.run("x")
        self.assertEqual(caught.exception.__cause__.budget, "steps")

    def test_mixed_scopes_compose_as_two_instances(self):
        per_run = Quota(limits={"llm.calls": 10})
        global_cap = Quota(limits={"llm.calls": 3}, scope="global",
                           on_exceed="return", fallback="capped")
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2")],
                       middleware=[MetricsMiddleware(), per_run, global_cap])
        self.assertEqual(flow.run("x").output, "x")      # run 1: fine everywhere
        self.assertEqual(flow.run("x").output, "capped")  # global cap trips first

    def test_scope_lands_in_events_and_exception(self):
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2")],
                       middleware=[Observe(), MetricsMiddleware(),
                                   Quota(limits={"llm.calls": 1}, on_exceed="return")])
        result = flow.run("x")
        exceeded = [e for e in result.events if e["type"] == "quota_exceeded"]
        self.assertEqual(exceeded[0]["scope"], "run")

    def test_invalid_scope_rejected(self):
        with self.assertRaises(ValueError):
            Quota(scope="daily")


class QuotaStacking(unittest.TestCase):
    """Stacked Quota instances (the documented run-ceiling + kill-switch
    pattern) must not share per-run state through the context."""

    def test_stacked_quotas_count_steps_once(self):
        # with a shared state dict both instances incremented "steps", so a
        # 3-step run was counted as 6 and max_steps tripped at half budget
        flow = tl.Flow([lambda p: p, lambda p: p, lambda p: p],
                       middleware=[Quota(max_steps=3),
                                   Quota(max_cost=1.0,
                                         cost={"llm.output_tokens": 1e-6},
                                         scope="global")])
        self.assertEqual(flow.run("x").output, "x")

    def test_stacked_quotas_keep_their_own_baselines(self):
        # the second instance tracks different counters; its baseline snapshot
        # must not clobber the first one's — otherwise, with a collector
        # shared across runs, per-run deltas silently become lifetime totals
        from throughline.modules.metrics import Metrics
        shared = Metrics()
        flow = tl.Flow([llm_step("llm1")],
                       middleware=[MetricsMiddleware(collector=shared),
                                   Quota(limits={"llm.calls": 3}),
                                   Quota(max_cost=100.0,
                                         cost={"llm.output_tokens": 1e-6},
                                         scope="global")])
        for _ in range(5):                       # tripped on run 4 before
            self.assertEqual(flow.run("x").output, "x")
        self.assertEqual(shared.counters["llm.calls"], 5)

    def test_stacked_quotas_warn_independently(self):
        # the one-shot warn flags were shared: whichever instance warned
        # first about a budget name suppressed the other's warning
        flow = tl.Flow([llm_step("llm1"), llm_step("llm2"), llm_step("llm3")],
                       middleware=[Observe(), MetricsMiddleware(),
                                   Quota(limits={"llm.calls": 10}, warn_at=0.1),
                                   Quota(limits={"llm.calls": 30}, warn_at=0.05,
                                         scope="global")])
        result = flow.run("x")
        warnings = [e for e in result.events if e["type"] == "quota_warning"]
        self.assertEqual(sorted(w["scope"] for w in warnings),
                         ["global", "run"])


class PresetIntegration(unittest.TestCase):
    def test_quota_from_preset(self):
        flow = tl.build_flow({
            "name": "budgeted",
            "steps": [{"uses": "throughline.contrib.demo:normalize"}],
            "middleware": {"metrics": {},
                           "quota": {"max_seconds": 30,
                                     "scope": "run",
                                     "limits": {"llm.calls": 5},
                                     "cost": {"llm.output_tokens": 1e-5},
                                     "max_cost": 1.0}},
        })
        self.assertEqual(type(flow.middleware[1]).__name__, "Quota")
        self.assertEqual(flow.run("x").output, {"question": "x"})


if __name__ == "__main__":
    unittest.main()
