import unittest

import throughline as tl
from throughline.modules import MemorySink, Metrics, MetricsMiddleware, Observe, Retry, Validate


class MetricsTests(unittest.TestCase):
    def test_step_timing_and_counts(self):
        flow = tl.Flow([str.strip, str.upper], middleware=[MetricsMiddleware()])
        metrics = flow.run("  x  ").metrics
        self.assertEqual(metrics["counters"]["steps"], 2)
        self.assertEqual(metrics["counters"]["step.strip.calls"], 1)
        self.assertIn("step.upper.seconds", metrics["observations"])
        self.assertGreaterEqual(metrics["observations"]["step.upper.seconds"]["count"], 1)

    def test_ctx_metric_from_step(self):
        def counting(payload, ctx):
            ctx.metric("tokens", 7)
            ctx.metric("score", 0.5, kind="observe")
            return payload
        metrics = tl.Flow([counting], middleware=[MetricsMiddleware()]).run("x").metrics
        self.assertEqual(metrics["counters"]["tokens"], 7)
        self.assertEqual(metrics["observations"]["score"]["mean"], 0.5)

    def test_ctx_metric_noop_without_middleware(self):
        def counting(payload, ctx):
            ctx.metric("tokens", 7)  # must not raise
            return payload
        self.assertEqual(tl.Flow([counting]).run("x").output, "x")

    def test_shared_collector(self):
        shared = Metrics()
        flow = tl.Flow([str.strip], middleware=[MetricsMiddleware(shared)])
        flow.run(" a ")
        flow.run(" b ")
        self.assertEqual(shared.snapshot()["counters"]["runs"], 2)

    def test_errors_counted(self):
        def boom(payload):
            raise ValueError("x")
        flow = tl.Flow([boom], middleware=[MetricsMiddleware()])
        with self.assertRaises(tl.FlowError) as caught:
            flow.run("x")
        metrics = caught.exception.ctx.artifacts["metrics"].snapshot()
        self.assertEqual(metrics["counters"]["errors"], 1)


class ObserveTests(unittest.TestCase):
    def test_events_recorded(self):
        result = tl.Flow([str.strip], middleware=[Observe()]).run(" x ")
        types = [e["type"] for e in result.events]
        self.assertIn("step_finished", types)
        self.assertIn("run_finished", types)

    def test_custom_sink(self):
        sink = MemorySink()
        tl.Flow([str.strip], middleware=[Observe(sink)]).run(" x ")
        self.assertTrue(any(e["type"] == "step_finished" for e in sink.events))

    def test_null_sink_shortcut(self):
        from throughline.modules.observe import NullSink
        self.assertIsInstance(Observe._coerce("null"), NullSink)
        self.assertIsInstance(Observe._coerce("none"), NullSink)
        # Memory bus still records; null is a no-op extra sink.
        result = tl.Flow([str.strip], middleware=[Observe(sink="null")]).run(" x ")
        self.assertTrue(any(e["type"] == "step_finished" for e in result.events))

    def test_console_sink_truncates_large_values(self):
        import io
        from throughline.modules.observe import ConsoleSink
        buf = io.StringIO()
        sink = ConsoleSink(stream=buf, verbose=True, max_value_chars=20)
        sink({"ts": 0.0, "type": "diag", "run_id": "r", "flow": "f",
              "blob": "x" * 200})
        line = buf.getvalue()
        self.assertIn("blob=", line)
        self.assertIn("…", line)
        self.assertNotIn("x" * 50, line)

    def test_jsonl_sink(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            tl.Flow([str.strip], middleware=[Observe(str(path))]).run(" x ")
            lines = path.read_text().strip().splitlines()
            self.assertTrue(lines)
            self.assertIn("type", json.loads(lines[0]))

    def test_jsonl_sink_truncates_large_values(self):
        import json
        import tempfile
        from pathlib import Path
        from throughline.modules.observe import JsonlSink
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            sink = JsonlSink(path, max_value_chars=30)
            sink({"ts": 1.0, "type": "diag", "blob": "y" * 500,
                  "nested": {"note": "z" * 500}})
            row = json.loads(path.read_text().strip())
            self.assertLessEqual(len(row["blob"]), 30)
            self.assertTrue(row["blob"].endswith("…"))
            self.assertLessEqual(len(row["nested"]["note"]), 30)


class ValidateTests(unittest.TestCase):
    def test_schema_pass(self):
        flow = tl.Flow([lambda p: {"answer": p}],
                       middleware=[Validate(schema={"type": "object", "required": ["answer"]})])
        self.assertEqual(flow.run("x").output, {"answer": "x"})

    def test_default_scope_is_final_output_only(self):
        """Pinned semantics: the default scope validates ONLY the final run
        output. Intermediate payloads may violate the schema (here: no
        "answer" after normalize) — the run must still pass."""
        flow = tl.Flow(
            [tl.as_step(lambda p: {"question": p}, "normalize"),
             tl.as_step(lambda p: {**p, "answer": "42"}, "answer")],
            middleware=[Validate(schema={"type": "object", "required": ["answer"]})])
        self.assertEqual(flow.run("q").output, {"question": "q", "answer": "42"})

    def test_explicit_scope_final(self):
        flow = tl.Flow(
            [tl.as_step(lambda p: {"question": p}, "normalize"),
             tl.as_step(lambda p: {**p, "answer": "42"}, "answer")],
            middleware=[Validate(scope="final",
                                 schema={"type": "object", "required": ["answer"]})])
        self.assertEqual(flow.run("q").output["answer"], "42")

    def test_scope_step_checks_every_step(self):
        # scope="step" without step= means every step's output; the schema
        # violation after normalize (no "answer") must now be caught
        flow = tl.Flow(
            [tl.as_step(lambda p: {"question": p}, "normalize"),
             tl.as_step(lambda p: {**p, "answer": "42"}, "answer")],
            middleware=[Validate(scope="step", on_fail="warn",
                                 schema={"required": ["answer"]})])
        result = flow.run("q")
        self.assertEqual(len(result.violations), 1)
        self.assertIn("normalize.output", result.violations[0])

    def test_step_pattern_implies_step_scope(self):
        validate = Validate(step="retrieve", check=lambda out: True)
        self.assertEqual(validate.scope, "step")

    def test_conflicting_scope_combinations_rejected(self):
        with self.assertRaises(ValueError):
            Validate(scope="final", step="retrieve", check=lambda o: True)
        with self.assertRaises(ValueError):
            Validate(scope="final", at="input", check=lambda o: True)
        with self.assertRaises(ValueError):
            Validate(scope="everything", check=lambda o: True)

    def test_scope_from_preset(self):
        flow = tl.build_flow({
            "name": "validated",
            "steps": [{"uses": "throughline.contrib.demo:normalize"}],
            "middleware": {"validate": {"scope": "final", "on_fail": "warn",
                                        "schema": {"required": ["question"]}}},
        })
        self.assertEqual(flow.run("x").output, {"question": "x"})

    def test_schema_raise(self):
        flow = tl.Flow([lambda p: {"other": p}],
                       middleware=[Validate(schema={"type": "object", "required": ["answer"]})])
        with self.assertRaises(tl.FlowError) as caught:
            flow.run("x")
        self.assertIsInstance(caught.exception.__cause__, tl.ValidationError)

    def test_warn_policy_collects_violations(self):
        flow = tl.Flow([lambda p: {"other": p}],
                       middleware=[Validate(schema={"required": ["answer"]}, on_fail="warn")])
        result = flow.run("x")
        self.assertEqual(len(result.violations), 1)
        self.assertIn("answer", result.violations[0])

    def test_predicate_and_tuple_checks(self):
        flow = tl.Flow([str.upper], middleware=[Validate(check=lambda out: out.isupper())])
        self.assertEqual(flow.run("hey").output, "HEY")
        flow_msg = tl.Flow([str.upper],
                           middleware=[Validate(check=lambda out: (False, "always bad"),
                                                on_fail="warn")])
        self.assertIn("always bad", flow_msg.run("hey").violations[0])

    def test_step_scoped_validation(self):
        flow = tl.Flow(
            [tl.as_step(lambda p: {"context": []}, "retrieve"),
             tl.as_step(lambda p: {**p, "answer": "?"}, "answer")],
            middleware=[Validate(step="retrieve",
                                 check=lambda out: bool(out["context"]) or "empty context",
                                 on_fail="warn")])
        result = flow.run("q")
        self.assertIn("empty context", result.violations[0])

    def test_schema_type_checks(self):
        from throughline.modules.validate import check_schema
        self.assertEqual(check_schema("s", {"type": "string"}), [])
        self.assertTrue(check_schema(True, {"type": "integer"}))  # bool is not integer
        self.assertTrue(check_schema({"a": 1}, {"type": "object",
                                                "properties": {"a": {"type": "string"}}}))
        self.assertEqual(check_schema([1, 2], {"type": "array",
                                               "items": {"type": "integer"}}), [])
        self.assertTrue(check_schema({"x": 1}, {"type": "object", "properties": {},
                                                "additionalProperties": False}))
        self.assertTrue(check_schema("c", {"enum": ["a", "b"]}))


class RetryTests(unittest.TestCase):
    def test_succeeds_after_failures(self):
        calls = {"n": 0}

        def flaky(payload):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return "ok"
        flow = tl.Flow([flaky], middleware=[Retry(attempts=3, backoff=0.001)])
        self.assertEqual(flow.run(None).output, "ok")
        self.assertEqual(calls["n"], 3)

    def test_exhausted_reraises(self):
        def always(payload):
            raise ConnectionError("down")
        flow = tl.Flow([always], middleware=[Retry(attempts=2, backoff=0.001)])
        with self.assertRaises(tl.FlowError):
            flow.run(None)

    def test_step_pattern_scoping(self):
        calls = {"n": 0}

        def flaky(payload):
            calls["n"] += 1
            raise ConnectionError("x")
        flow = tl.Flow([tl.as_step(flaky, "other")],
                       middleware=[Retry(attempts=3, backoff=0.001, step="llm*")])
        with self.assertRaises(tl.FlowError):
            flow.run(None)
        self.assertEqual(calls["n"], 1)  # not retried: name does not match

    def test_retry_events_and_metrics(self):
        calls = {"n": 0}

        def flaky(payload):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ValueError("x")
            return "ok"
        result = tl.Flow([flaky], middleware=[MetricsMiddleware(),
                                              Observe(),
                                              Retry(attempts=2, backoff=0.001)]).run(None)
        self.assertEqual(result.metrics["counters"]["retries"], 1)
        self.assertIn("step_retry", [e["type"] for e in result.events])


if __name__ == "__main__":
    unittest.main()
