"""Structured output: parse_json, json_step, structured_step and the
validate→retry composition semantics they exist for."""

import unittest

import throughline as tl
from throughline.errors import FlowError, ValidationError
from throughline.modules import (MetricsMiddleware, Retry, Validate, json_step,
                                 parse_json, structured_step)

SCHEMA = {"type": "object", "required": ["name", "age"]}


class ParseJsonTests(unittest.TestCase):
    def test_direct(self):
        self.assertEqual(parse_json('{"a": 1}'), {"a": 1})

    def test_code_fence(self):
        text = 'Here you go:\n```json\n{"a": [1, 2]}\n```\nHope that helps!'
        self.assertEqual(parse_json(text), {"a": [1, 2]})

    def test_fence_without_language_tag(self):
        self.assertEqual(parse_json('```\n[1, 2]\n```'), [1, 2])

    def test_prose_wrapped_braces_salvage(self):
        text = 'The result is {"name": "Ada"} as requested.'
        self.assertEqual(parse_json(text), {"name": "Ada"})

    def test_invalid_raises_with_text_head(self):
        with self.assertRaises(ValidationError) as caught:
            parse_json("I cannot answer that.")
        self.assertIn("not valid JSON", str(caught.exception))
        self.assertIn("I cannot answer", str(caught.exception))


class JsonStepTests(unittest.TestCase):
    def test_parses_in_place_by_default(self):
        flow = tl.Flow([json_step()], middleware=[MetricsMiddleware()])
        result = flow.run({"answer": '```json\n{"x": 1}\n```'})
        self.assertEqual(result.output["answer"], {"x": 1})
        self.assertEqual(result.metrics["counters"].get("json.parsed"), 1)

    def test_out_key_and_schema(self):
        flow = tl.Flow([json_step(out_key="data", schema=SCHEMA)])
        result = flow.run({"answer": '{"name": "Ada", "age": 36}'})
        self.assertEqual(result.output["data"]["name"], "Ada")
        self.assertEqual(result.output["answer"], '{"name": "Ada", "age": 36}')

    def test_schema_failure_raises(self):
        flow = tl.Flow([json_step(schema=SCHEMA)])
        with self.assertRaises(FlowError) as caught:
            flow.run({"answer": '{"name": "Ada"}'})
        self.assertIn("schema", str(caught.exception))

    def test_warn_records_violation_and_passes_through(self):
        flow = tl.Flow([json_step(on_fail="warn")], middleware=[MetricsMiddleware()])
        result = flow.run({"answer": "no json here"})
        self.assertEqual(result.output["answer"], "no json here")
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(result.metrics["counters"].get("json.invalid"), 1)


class FlakyLLM:
    """Bad JSON on the first call, valid on the next — retry fodder."""

    def __init__(self):
        self.calls = 0

    def __call__(self, payload):
        self.calls += 1
        if self.calls < 2:
            return {**payload, "answer": "Sorry, let me think..."}
        return {**payload, "answer": '{"name": "Ada", "age": 36}'}


class StructuredStepTests(unittest.TestCase):
    def test_retry_regenerates_until_valid(self):
        llm = FlakyLLM()
        flow = tl.Flow(
            [structured_step(llm, schema=SCHEMA, name="extract")],
            middleware=[MetricsMiddleware(), Retry(attempts=3, backoff=0, step="extract")],
        )
        result = flow.run({"question": "who?"})
        self.assertEqual(result.output["answer"], {"name": "Ada", "age": 36})
        self.assertEqual(llm.calls, 2)                      # regenerated once
        self.assertEqual(result.metrics["counters"].get("retries"), 1)

    def test_exhausted_attempts_raise(self):
        always_bad = lambda payload: {**payload, "answer": "nope"}  # noqa: E731
        flow = tl.Flow(
            [structured_step(always_bad, name="extract")],
            middleware=[Retry(attempts=2, backoff=0, step="extract")],
        )
        with self.assertRaises(FlowError) as caught:
            flow.run({})
        self.assertIn("not valid JSON", str(caught.exception))

    def test_non_dict_generator_output(self):
        step = structured_step(lambda p: '```json\n[1, 2, 3]\n```')
        self.assertEqual(tl.Flow([step]).run("x").output, [1, 2, 3])

    def test_validate_plus_retry_does_not_compose_the_pin(self):
        """The reason structured_step exists, pinned: per-step Validate
        raises in on_step_end, which runs AFTER the wrap_step onion — Retry
        never sees the raise, so the LLM is NOT regenerated. If this test
        ever fails, the flow contract changed and structured_step's
        docstring (and this recipe) must be revisited."""
        llm = FlakyLLM()
        flow = tl.Flow(
            [tl.as_step(llm, "llm")],
            middleware=[
                Retry(attempts=3, backoff=0, step="llm"),
                Validate(step="llm", scope="step", on_fail="raise",
                         schema={"type": "object", "required": ["missing-key"]}),
            ],
        )
        with self.assertRaises(FlowError):
            flow.run({"question": "who?"})
        self.assertEqual(llm.calls, 1)  # validation raise was never retried


if __name__ == "__main__":
    unittest.main()
