import unittest

import throughline as tl


class FlowBasics(unittest.TestCase):
    def test_linear_flow(self):
        flow = tl.Flow([str.strip, str.upper], name="clean")
        result = flow.run("  hello  ")
        self.assertEqual(result.output, "HELLO")
        self.assertEqual(result.ctx.flow, "clean")

    def test_two_arg_steps_receive_ctx(self):
        def with_ctx(payload, ctx):
            ctx.state["seen"] = True
            return payload + 1
        result = tl.Flow([with_ctx]).run(41)
        self.assertEqual(result.output, 42)
        self.assertTrue(result.ctx.state["seen"])

    def test_defaulted_second_param_is_not_mistaken_for_ctx(self):
        # f(text, strip=True) is a one-payload step with an option — the
        # RunContext must not land in `strip`
        seen = {}

        def clean(text, strip=True):
            seen["strip"] = strip
            return text.strip() if strip else text
        result = tl.Flow([clean]).run("  hello  ")
        self.assertEqual(result.output, "hello")
        self.assertIs(seen["strip"], True)

    def test_defaulted_ctx_param_receives_ctx(self):
        seen = {}

        def with_optional_ctx(payload, ctx=None):
            seen["ctx"] = ctx
            return payload
        tl.Flow([with_optional_ctx]).run("x")
        self.assertIsInstance(seen["ctx"], tl.RunContext)

    def test_step_decorator_and_direct_call(self):
        @tl.step("double")
        def double(x):
            return x * 2
        self.assertEqual(double.name, "double")
        self.assertEqual(double(21), 42)  # steps are callable stand-alone

    def test_then_and_use_are_immutable(self):
        base = tl.Flow([str.strip])
        extended = base.then(str.upper)
        self.assertEqual(len(base.steps), 1)
        self.assertEqual(len(extended.steps), 2)
        with_mw = base.use(tl.modules.MetricsMiddleware())
        self.assertEqual(len(base.middleware), 0)
        self.assertEqual(len(with_mw.middleware), 1)

    def test_error_wraps_into_flow_error_with_context(self):
        def boom(payload):
            raise ValueError("nope")
        flow = tl.Flow([str.strip, boom], name="failing")
        with self.assertRaises(tl.FlowError) as caught:
            flow.run(" x ")
        self.assertEqual(caught.exception.step, "boom")
        self.assertIsNotNone(caught.exception.ctx)
        types = [e["type"] for e in caught.exception.ctx.artifacts.get("events", [])]
        # events artifact only exists with Observe; check the error attrs instead
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_events_emitted(self):
        seen = []
        flow = tl.Flow([str.strip], name="observed")
        from throughline.context import RunContext
        ctx = RunContext(flow="observed")
        ctx.events.subscribe(lambda e: seen.append(e["type"]))
        flow.run("  x ", ctx=ctx)
        self.assertIn("run_started", seen)
        self.assertIn("step_started", seen)
        self.assertIn("step_finished", seen)
        self.assertIn("run_finished", seen)

    def test_async_step_bridge(self):
        async def astep(payload):
            return payload * 2
        result = tl.Flow([astep]).run(21)
        self.assertEqual(result.output, 42)

    def test_config_merging(self):
        def read_cfg(payload, ctx):
            return ctx.config["top_k"]
        flow = tl.Flow([read_cfg], config={"top_k": 3})
        self.assertEqual(flow.run(None).output, 3)
        self.assertEqual(flow.run(None, config={"top_k": 7}).output, 7)


class Composites(unittest.TestCase):
    def test_map_step(self):
        flow = tl.Flow([tl.map_step(str.upper)])
        self.assertEqual(flow.run(["a", "b"]).output, ["A", "B"])

    def test_map_step_threaded(self):
        flow = tl.Flow([tl.map_step(lambda x: x + 1, workers=4)])
        self.assertEqual(flow.run(range(5)).output, [1, 2, 3, 4, 5])

    def test_parallel_dict(self):
        result = tl.Flow([tl.parallel({"up": str.upper, "low": str.lower})]).run("MiXeD")
        self.assertEqual(result.output, {"up": "MIXED", "low": "mixed"})

    def test_parallel_list(self):
        result = tl.Flow([tl.parallel([str.upper, str.lower], workers=2)]).run("Ab")
        self.assertEqual(result.output, ["AB", "ab"])

    def test_branch_routes_and_default(self):
        route = tl.branch(lambda p: p["kind"], {
            "greet": lambda p: "hi",
            "farewell": lambda p: "bye",
        }, default=lambda p: "unknown")
        flow = tl.Flow([route])
        self.assertEqual(flow.run({"kind": "greet"}).output, "hi")
        self.assertEqual(flow.run({"kind": "other"}).output, "unknown")

    def test_branch_without_default_raises(self):
        route = tl.branch(lambda p: p, {"a": lambda p: p})
        with self.assertRaises(tl.FlowError):
            tl.Flow([route]).run("missing")


class MiddlewareOrdering(unittest.TestCase):
    def test_onion_order(self):
        trace = []

        class Probe(tl.Middleware):
            def __init__(self, tag):
                self.tag = tag

            def on_step_start(self, ctx, step, payload):
                trace.append(f"{self.tag}:start")
                return payload

            def on_step_end(self, ctx, step, payload, output):
                trace.append(f"{self.tag}:end")
                return output

            def wrap_step(self, invoke, ctx, step):
                def wrapped(payload):
                    trace.append(f"{self.tag}:wrap-in")
                    out = invoke(payload)
                    trace.append(f"{self.tag}:wrap-out")
                    return out
                return wrapped

        tl.Flow([lambda p: p], middleware=[Probe("A"), Probe("B")]).run(1)
        self.assertEqual(trace, [
            "A:start", "B:start",
            "A:wrap-in", "B:wrap-in", "B:wrap-out", "A:wrap-out",
            "B:end", "A:end",
        ])

    def test_handled_error_recovery(self):
        class Recover(tl.Middleware):
            def on_step_error(self, ctx, step, payload, exc):
                return tl.Handled("fallback")

        def boom(payload):
            raise RuntimeError("x")
        result = tl.Flow([boom], middleware=[Recover()]).run("in")
        self.assertEqual(result.output, "fallback")


class EarlyReturnTests(unittest.TestCase):
    def test_step_short_circuits_remaining_steps(self):
        calls = {"later": 0}

        def stop_now(payload, ctx):
            raise tl.EarlyReturn(f"early:{payload}")

        def later(payload):
            calls["later"] += 1
            return payload
        result = tl.Flow([stop_now, later]).run("x")
        self.assertEqual(result.output, "early:x")
        self.assertEqual(calls["later"], 0)

    def test_run_end_hooks_apply_to_early_output(self):
        class Stamp(tl.Middleware):
            def on_run_end(self, ctx, output):
                return f"{output}+stamped"

        def stop_now(payload, ctx):
            raise tl.EarlyReturn("early")
        result = tl.Flow([stop_now], middleware=[Stamp()]).run("x")
        self.assertEqual(result.output, "early+stamped")

    def test_not_retried_and_not_counted_as_error(self):
        from throughline.modules import MetricsMiddleware, Retry
        attempts = {"n": 0}

        def stop_now(payload, ctx):
            attempts["n"] += 1
            raise tl.EarlyReturn("early")
        result = tl.Flow([stop_now],
                         middleware=[MetricsMiddleware(),
                                     Retry(attempts=3, backoff=0.001)]).run("x")
        self.assertEqual(result.output, "early")
        self.assertEqual(attempts["n"], 1)
        metrics = result.ctx.artifacts["metrics"].snapshot()
        self.assertNotIn("errors", metrics["counters"])

    def test_short_circuit_event_emitted(self):
        from throughline.modules import Observe

        def stop_now(payload, ctx):
            raise tl.EarlyReturn("early")
        result = tl.Flow([stop_now], middleware=[Observe()]).run("x")
        self.assertIn("run_short_circuited", [e["type"] for e in result.events])


if __name__ == "__main__":
    unittest.main()
