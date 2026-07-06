"""Duck typing must explain itself: wrap errors, explain(), doctor, strict mode."""

import unittest
import weakref

import followers as fl
from followers.adapters import explain, render_explain
from followers.errors import WrapError
from followers.modules.debug import Snapshots, StrictOutputs


class Fetcher:
    """No known interface — only fetch()."""

    def fetch(self, query):
        return f"fetched:{query}"


class Ambiguous:
    """Both query() and run(): detection must pick query and report run skipped."""

    def query(self, q):
        return f"q:{q}"

    def run(self, q):
        return f"r:{q}"


class WrapDiagnosticsTests(unittest.TestCase):
    def test_wrap_fails_at_wrap_time_with_trace(self):
        with self.assertRaises(WrapError) as caught:
            fl.wrap(Fetcher())
        message = str(caught.exception)
        self.assertIn("Tried (in order)", message)
        self.assertIn("invoke", message)
        self.assertIn("Object has: fetch", message)
        self.assertIn("Hint: fl.wrap(obj, method='fetch')", message)
        self.assertIn("fetch", caught.exception.found)

    def test_forced_method_missing_is_loud(self):
        with self.assertRaises(WrapError) as caught:
            fl.wrap(Fetcher(), method="query")
        self.assertIn("forced method 'query'", str(caught.exception))

    def test_forced_method_works(self):
        step = fl.wrap(Fetcher(), method="fetch")
        self.assertEqual(fl.Flow([step]).run("x").output, "fetched:x")

    def test_explain_shows_decision_and_skipped(self):
        decision = explain(Ambiguous())
        self.assertEqual(decision["method"], "query")
        self.assertIn("run", decision["skipped"])
        rendered = render_explain(Ambiguous())
        self.assertIn("detected: query()", rendered)
        self.assertIn("skipped", rendered)

    def test_render_explain_on_unadaptable(self):
        self.assertIn("UNADAPTABLE", render_explain(Fetcher()))

    def test_step_meta_records_detection(self):
        step = fl.wrap(Ambiguous())
        self.assertEqual(step.meta["adapter"], "query")
        self.assertIn("run", step.meta["skipped"])
        self.assertFalse(step.meta["unwrap"])


class ForeignResponse:
    def __init__(self, text):
        self.response = text


class StrictOutputsTests(unittest.TestCase):
    def test_names_the_cause_at_the_offending_step(self):
        def leaky(payload, ctx):
            return ForeignResponse("oops")  # forgot unwrap=

        result = fl.Flow(
            [fl.as_step(leaky, "leaky"), lambda r: r],
            middleware=[StrictOutputs()],
        ).run("x")
        self.assertTrue(any("leaky" in v and "ForeignResponse" in v
                            for v in result.violations))

    def test_reports_the_exact_path(self):
        flow = fl.Flow(
            [fl.as_step(lambda p: {"answer": ForeignResponse("x")}, "gen")],
            middleware=[StrictOutputs()])
        violations = flow.run("q").violations
        self.assertTrue(any("$.answer" in v and "ForeignResponse" in v
                            for v in violations))

    def test_finds_leak_at_any_depth(self):
        """The precise contract: the whole object graph, not one level."""
        deep = {"results": [{"docs": [{"meta": ForeignResponse("x")}]}]}
        flow = fl.Flow([lambda p: deep], middleware=[StrictOutputs()])
        violations = flow.run("q").violations
        self.assertTrue(any("$.results[0].docs[0].meta" in v for v in violations))

    def test_non_scalar_dict_key_is_a_leak(self):
        flow = fl.Flow([lambda p: {("tuple", "key"): "v"}],
                       middleware=[StrictOutputs()])
        self.assertTrue(any("<key>" in v for v in flow.run("q").violations))

    def test_raise_mode(self):
        flow = fl.Flow([lambda p: ForeignResponse("x")],
                       middleware=[StrictOutputs(on_foreign="raise")])
        with self.assertRaises(fl.FlowError):
            flow.run("q")

    def test_plain_outputs_pass(self):
        flow = fl.Flow([lambda p: {"answer": "fine", "ref": fl.ArtifactRef(id="s/k"),
                                   "nested": [{"deep": [1, 2.5, None, True]}]}],
                       middleware=[StrictOutputs()])
        self.assertEqual(flow.run("q").violations, [])

    def test_allow_extends_the_contract(self):
        flow = fl.Flow([lambda p: {"custom": ForeignResponse("x")}],
                       middleware=[StrictOutputs(allow=(ForeignResponse,))])
        self.assertEqual(flow.run("q").violations, [])

    def test_step_scoping(self):
        flow = fl.Flow(
            [fl.as_step(lambda p: ForeignResponse("a"), "loader"),
             fl.as_step(lambda p: {"answer": "clean"}, "llm")],
            middleware=[StrictOutputs(step="llm*")])
        self.assertEqual(flow.run("q").violations, [])  # loader is out of scope

    def test_cycle_safety(self):
        cyclic: dict = {"name": "loop"}
        cyclic["self"] = cyclic
        flow = fl.Flow([lambda p: cyclic], middleware=[StrictOutputs()])
        self.assertEqual(flow.run("q").violations, [])  # terminates, no offender

    def test_budget_truncation_is_announced(self):
        wide = {"items": list(range(100))}
        flow = fl.Flow([lambda p: wide],
                       middleware=[fl.modules.Observe(), StrictOutputs(max_nodes=10)])
        result = flow.run("q")
        self.assertIn("foreign_scan_truncated",
                      [e["type"] for e in result.events])

    def test_early_returned_output_is_checked(self):
        from followers.errors import EarlyReturn

        def bail(payload, ctx):
            raise EarlyReturn({"answer": ForeignResponse("cached-wrong")})

        result = fl.Flow([bail], middleware=[StrictOutputs()]).run("q")
        self.assertTrue(any("early_return" in v and "$.answer" in v
                            for v in result.violations))


class SnapshotsTests(unittest.TestCase):
    def test_opt_in_records_every_version(self):
        flow = fl.Flow(
            [fl.as_step(str.upper, "up"), fl.as_step(str.strip, "strip")],
            middleware=[Snapshots()],
        )
        result = flow.run("  hi  ")
        trail = result.ctx.artifacts["snapshots"]
        self.assertEqual([label for label, _ in trail], ["input", "up", "strip"])
        self.assertEqual(trail[-1][1], "HI")

    def test_ring_buffer_cap(self):
        flow = fl.Flow([lambda p: p + 1] * 5, middleware=[Snapshots(max_versions=3)])
        result = flow.run(0)
        self.assertEqual(len(result.ctx.artifacts["snapshots"]), 3)

    def test_deep_copies_survive_mutation(self):
        def mutate(payload, ctx):
            payload["x"].append(1)
            return payload

        flow = fl.Flow([mutate, mutate], middleware=[Snapshots(deep=True)])
        result = flow.run({"x": []})
        trail = result.ctx.artifacts["snapshots"]
        self.assertEqual(trail[0][1], {"x": []})       # input untouched
        self.assertEqual(trail[1][1], {"x": [1]})      # after first step
        self.assertEqual(result.output, {"x": [1, 1]})


class PayloadRetentionInvariantTests(unittest.TestCase):
    """Stock middleware must not retain payload versions between steps.

    Snapshots is the single sanctioned violation; everything shipped by
    default must let intermediate versions die as soon as the next step ran.
    """

    def test_stock_stack_releases_intermediate_payloads(self):
        class Tracker:
            pass

        graveyard = []

        def make_garbage(payload, ctx):
            obj = Tracker()
            graveyard.append(weakref.ref(obj))
            return {"answer": "ok", "heavy": obj}

        def drop_garbage(payload, ctx):
            return {"answer": payload["answer"]}

        flow = fl.Flow(
            [make_garbage, drop_garbage],
            middleware=[
                fl.modules.Observe(),  # a bare string would mean a JSONL path
                fl.modules.MetricsMiddleware(),
                fl.modules.Retry(attempts=2),
                fl.modules.Validate(schema={"type": "object"}),
                fl.modules.LineageMiddleware(extract="answer"),
            ],
        )
        result = flow.run({"question": "q"})
        self.assertEqual(result.output, {"answer": "ok"})
        import gc
        gc.collect()
        self.assertTrue(all(ref() is None for ref in graveyard),
                        "a stock middleware retained an intermediate payload")

    def test_snapshots_is_the_sanctioned_exception(self):
        graveyard = []

        class Tracker:
            pass

        def make_garbage(payload, ctx):
            obj = Tracker()
            graveyard.append(weakref.ref(obj))
            return {"heavy": obj}

        flow = fl.Flow([make_garbage, lambda p: {"done": True}],
                       middleware=[Snapshots()])
        result = flow.run(None)
        import gc
        gc.collect()
        self.assertTrue(any(ref() is not None for ref in graveyard))
        del result
        gc.collect()
        self.assertTrue(all(ref() is None for ref in graveyard))


class DoctorTests(unittest.TestCase):
    def test_inspect_preset_reports_slots(self):
        from followers.presets import inspect_preset
        report = inspect_preset("demo")
        self.assertTrue(report["ok"])
        self.assertTrue(all(r["status"] == "ok" for r in report["steps"]))
        self.assertTrue(any(r["slot"].startswith("[middleware.")
                            for r in report["middleware"]))

    def test_doctor_cli(self):
        from followers.cli import main
        self.assertEqual(main(["doctor", "demo"]), 0)

    def test_components_cli(self):
        from followers.cli import main
        self.assertEqual(main(["components"]), 0)


class SlotKindCheckTests(unittest.TestCase):
    def test_wrong_kind_in_middleware_slot(self):
        config = {
            "name": "bad",
            "steps": [{"uses": "followers.contrib.demo:normalize"}],
            "middleware": {"custom": {"uses": "followers.store:MemoryArtifactStore"}},
        }
        with self.assertRaises(fl.PresetError) as caught:
            fl.build_flow(config)
        message = str(caught.exception)
        self.assertIn("[middleware.custom]", message)
        self.assertIn("'middleware' contract", message)
        self.assertIn("It satisfies: store", message)


if __name__ == "__main__":
    unittest.main()
