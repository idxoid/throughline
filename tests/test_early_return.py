"""The EarlyReturn / on_run_end contract, pinned formally.

Spec (Flow.run docstring): EarlyReturn skips the remaining on_run_start
hooks, the remaining steps and the raising step's on_step_end hooks; it
bypasses on_step_error and Retry; ctx.short_circuited becomes True; and
on_run_end is a FINALIZER SWEEP — every middleware, reverse order, exactly
once, even if its on_run_start never ran. On real failures on_run_end does
not run at all. These tests are the contract: changing them is changing
the semantics for every cache/quota/debug-style module.
"""

import unittest

import followers as fl
from followers.errors import EarlyReturn


class Probe(fl.Middleware):
    """Records every hook invocation into a shared log."""

    def __init__(self, label: str, log: list):
        self.label = label
        self.log = log

    def on_run_start(self, ctx, payload):
        self.log.append(f"{self.label}.run_start")
        return payload

    def on_step_start(self, ctx, step, payload):
        self.log.append(f"{self.label}.step_start:{step.name}")
        return payload

    def on_step_end(self, ctx, step, payload, output):
        self.log.append(f"{self.label}.step_end:{step.name}")
        return output

    def on_step_error(self, ctx, step, payload, exc):
        self.log.append(f"{self.label}.step_error:{step.name}")
        return None

    def on_run_end(self, ctx, output):
        self.log.append(f"{self.label}.run_end")
        return output


class ShortCircuitInRunStart(fl.Middleware):
    def __init__(self, output):
        self.output = output

    def on_run_start(self, ctx, payload):
        raise EarlyReturn(self.output)


class EarlyReturnFromRunStartTests(unittest.TestCase):
    """The Cache-hit shape: EarlyReturn inside the on_run_start chain."""

    def _run(self):
        log = []
        flow = fl.Flow(
            [fl.as_step(lambda p: p, "never")],
            middleware=[Probe("outer", log),
                        ShortCircuitInRunStart("cached"),
                        Probe("inner", log)],
        )
        return flow.run("q"), log

    def test_remaining_run_start_hooks_are_skipped(self):
        _, log = self._run()
        self.assertIn("outer.run_start", log)
        self.assertNotIn("inner.run_start", log)

    def test_no_steps_run(self):
        result, log = self._run()
        self.assertEqual(result.output, "cached")
        self.assertFalse(any("step_start" in entry for entry in log))

    def test_on_run_end_is_a_full_sweep_in_reverse_order(self):
        _, log = self._run()
        # inner.run_end fires even though inner.run_start never ran
        run_ends = [e for e in log if e.endswith("run_end")]
        self.assertEqual(run_ends, ["inner.run_end", "outer.run_end"])

    def test_short_circuited_flag(self):
        result, _ = self._run()
        self.assertTrue(result.ctx.short_circuited)

    def test_normal_run_flag_is_false(self):
        result = fl.Flow([lambda p: p]).run("q")
        self.assertFalse(result.ctx.short_circuited)


class EarlyReturnFromStepTests(unittest.TestCase):
    """The Quota-fallback shape: EarlyReturn raised mid-pipeline."""

    def _run(self):
        log = []

        def bail(payload, ctx):
            raise EarlyReturn("partial")

        flow = fl.Flow(
            [fl.as_step(str.upper, "first"),
             fl.as_step(bail, "bail"),
             fl.as_step(str.lower, "after")],
            middleware=[Probe("probe", log)],
        )
        return flow.run("q"), log

    def test_output_and_skipped_steps(self):
        result, log = self._run()
        self.assertEqual(result.output, "partial")
        self.assertNotIn("probe.step_start:after", log)

    def test_raising_steps_on_step_end_is_skipped(self):
        _, log = self._run()
        self.assertIn("probe.step_end:first", log)
        self.assertNotIn("probe.step_end:bail", log)

    def test_on_step_error_is_bypassed(self):
        _, log = self._run()
        self.assertFalse(any("step_error" in entry for entry in log))

    def test_on_run_end_still_runs(self):
        result, log = self._run()
        self.assertIn("probe.run_end", log)
        self.assertTrue(result.ctx.short_circuited)

    def test_retry_never_retries_early_return(self):
        calls = {"n": 0}

        def bail(payload, ctx):
            calls["n"] += 1
            raise EarlyReturn("done")

        result = fl.Flow([bail], middleware=[fl.modules.Retry(attempts=5)]).run("q")
        self.assertEqual(result.output, "done")
        self.assertEqual(calls["n"], 1)


class FailureIsNotASweepTests(unittest.TestCase):
    def test_on_run_end_does_not_run_on_failure(self):
        log = []

        def boom(payload, ctx):
            raise ValueError("nope")

        with self.assertRaises(fl.FlowError):
            fl.Flow([boom], middleware=[Probe("probe", log)]).run("q")
        self.assertNotIn("probe.run_end", log)
        self.assertIn("probe.step_error:boom", log)  # error hooks did fire


class ModulesHonorTheContractTests(unittest.TestCase):
    """Lineage and Snapshots must account for the substituted output."""

    def test_lineage_attributes_early_output(self):
        def bail(payload, ctx):
            raise EarlyReturn({"answer": "substituted line"})

        flow = fl.Flow(
            [fl.as_step(lambda p: {"answer": "drafted line"}, "draft"),
             fl.as_step(bail, "bail")],
            middleware=[fl.modules.LineageMiddleware(extract="answer")],
        )
        result = flow.run({"question": "q"})
        blame = result.lineage.blame()
        self.assertEqual(blame[0]["text"], "substituted line")
        self.assertEqual(blame[0]["step"], "early_return")

    def test_lineage_untouched_on_normal_runs(self):
        flow = fl.Flow([fl.as_step(lambda p: {"answer": "x"}, "draft")],
                       middleware=[fl.modules.LineageMiddleware(extract="answer")])
        ledger = flow.run({"question": "q"}).lineage
        self.assertNotIn("early_return", ledger.steps)

    def test_lineage_survives_run_level_cache_hit(self):
        # Lineage inside Cache: on a hit its on_run_start never runs, so the
        # sweep sees no ledger — must be a no-op, not a KeyError.
        flow = fl.Flow(
            [lambda p: {"answer": p}],
            middleware=[fl.modules.Cache(),
                        fl.modules.LineageMiddleware(extract="answer")],
        )
        flow.run("q")
        hit = flow.run("q")                      # would raise before the fix
        self.assertEqual(hit.output, {"answer": "q"})
        self.assertIsNone(hit.lineage)           # no ledger on a hit — honest

    def test_snapshots_record_early_output(self):
        def bail(payload, ctx):
            raise EarlyReturn("substituted")

        flow = fl.Flow([fl.as_step(str.upper, "up"), bail],
                       middleware=[fl.modules.Snapshots()])
        trail = flow.run("q").ctx.artifacts["snapshots"]
        self.assertEqual(trail[-1], ("early_return", "substituted"))

    def test_quota_fallback_end_to_end(self):
        # The composed case the contract exists for: metrics outside, quota
        # inside; the fallback output is visible, attributed, accounted.
        flow = fl.Flow(
            [fl.as_step(lambda p, ctx: (ctx.metric("llm.calls"), p)[1], "llm")] * 3,
            middleware=[
                fl.modules.MetricsMiddleware(),
                fl.modules.Quota(limits={"llm.calls": 1}, on_exceed="return",
                                 fallback=lambda p, ctx: "budget exhausted"),
                fl.modules.Snapshots(),
            ],
        )
        result = flow.run("q")
        self.assertEqual(result.output, "budget exhausted")
        self.assertTrue(result.ctx.short_circuited)
        self.assertEqual(result.ctx.artifacts["snapshots"][-1],
                         ("early_return", "budget exhausted"))
        self.assertEqual(result.metrics["counters"]["quota.exceeded"], 1)


if __name__ == "__main__":
    unittest.main()
