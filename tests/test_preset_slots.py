"""Preset slots: [slots] declarations, @refs, [fill], fill= and composites."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import throughline as tl
from throughline.errors import PresetError
from throughline.presets import inspect_preset, load_preset


# --- components referenced by the fixture presets ---------------------------

def upper(payload):
    return str(payload).upper()


def double(payload):
    return payload * 2


def negate(payload):
    return -payload


def make_prefixer(prefix: str = "> "):
    def fn(payload):
        return f"{prefix}{payload}"
    return fn


def make_wrapper(inner=None):
    """Factory whose kwarg should arrive as a LIVE callable via @slot."""
    def fn(payload):
        return inner(payload)
    return fn


def pick_kind(payload):
    return payload["kind"]


def double_value(payload):
    return payload["value"] * 2


def negate_value(payload):
    return -payload["value"]


class TagMiddleware(tl.Middleware):
    def __init__(self, tag: str = "x"):
        self.tag = tag

    def on_step_end(self, ctx, step, payload, output):
        ctx.metric(f"tag.{self.tag}")
        return output


NOT_A_STEP = object()


# --- harness ----------------------------------------------------------------

class SlotPresetBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._old_env = os.environ.get("THROUGHLINE_PRESETS")
        os.environ["THROUGHLINE_PRESETS"] = str(self.dir)

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop("THROUGHLINE_PRESETS", None)
        else:
            os.environ["THROUGHLINE_PRESETS"] = self._old_env
        self._tmp.cleanup()

    def write(self, name: str, content: str) -> Path:
        path = self.dir / f"{name}.toml"
        path.write_text(content, encoding="utf-8")
        return path


MOD = "tests.test_preset_slots"


class SlotTests(SlotPresetBase):
    def abstract_preset(self, extra: str = "") -> None:
        self.write("abstract", f"""
            name = "abstract"
            [slots.transform]
            kind = "step"
            description = "text transform of the payload"
            [[steps]]
            uses = "@transform"
            {extra}
        """)

    def test_default_fill(self):
        self.write("with-default", f"""
            [slots.transform]
            default = "{MOD}:upper"
            [[steps]]
            uses = "@transform"
        """)
        self.assertEqual(load_preset("with-default").run("hi").output, "HI")

    def test_fill_table_via_extends(self):
        self.abstract_preset()
        self.write("child", f"""
            extends = "abstract"
            [fill]
            transform = "{MOD}:upper"
        """)
        self.assertEqual(load_preset("child").run("hi").output, "HI")

    def test_fill_kwarg_overrides_fill_table(self):
        self.abstract_preset()
        self.write("child", f"""
            extends = "abstract"
            [fill]
            transform = "{MOD}:upper"
        """)
        flow = load_preset("child", fill={"transform": make_prefixer("! ")})
        self.assertEqual(flow.run("hi").output, "! hi")

    def test_missing_fill_lists_all_slots_with_description(self):
        self.write("two-holes", """
            [slots.first]
            description = "the first hole"
            [slots.second]
            kind = "step"
            [[steps]]
            uses = "@first"
            [[steps]]
            uses = "@second"
        """)
        with self.assertRaises(PresetError) as caught:
            load_preset("two-holes")
        message = str(caught.exception)
        self.assertIn("@first", message)
        self.assertIn("@second", message)
        self.assertIn("the first hole", message)
        self.assertIn("--fill", message)

    def test_unknown_fill_name_rejected(self):
        self.abstract_preset()
        with self.assertRaises(PresetError) as caught:
            load_preset("abstract", fill={"transformer": f"{MOD}:upper"})
        self.assertIn("unknown slot 'transformer'", str(caught.exception))
        self.assertIn("transform", str(caught.exception))

    def test_reference_to_undeclared_slot_rejected(self):
        self.write("typo", """
            [[steps]]
            uses = "@nope"
        """)
        with self.assertRaises(PresetError) as caught:
            load_preset("typo")
        self.assertIn("undeclared slot '@nope'", str(caught.exception))

    def test_slot_kind_checked_on_fill(self):
        self.abstract_preset()
        with self.assertRaises(PresetError) as caught:
            load_preset("abstract", fill={"transform": NOT_A_STEP})
        self.assertIn("@transform", str(caught.exception))

    def test_slot_kind_checked_after_factory_call(self):
        # the slot is filled with a FACTORY; kind applies to what it returns
        self.write("factory-slot", f"""
            [slots.transform]
            kind = "step"
            [fill]
            transform = "{MOD}:make_prefixer"
            [[steps]]
            uses = "@transform"
            [steps.with]
            prefix = "# "
        """)
        self.assertEqual(load_preset("factory-slot").run("t").output, "# t")

    def test_slot_inside_with_table_passes_live_object(self):
        self.write("with-slot", f"""
            [slots.inner]
            description = "callable the wrapper delegates to"
            [fill]
            inner = "{MOD}:upper"
            [[steps]]
            uses = "{MOD}:make_wrapper"
            [steps.with]
            inner = "@inner"
        """)
        self.assertEqual(load_preset("with-slot").run("hi").output, "HI")

    def test_double_at_escapes_literal(self):
        self.write("escape", f"""
            [[steps]]
            uses = "{MOD}:make_prefixer"
            [steps.with]
            prefix = "@@channel "
        """)
        self.assertEqual(load_preset("escape").run("x").output, "@channel x")

    def test_slot_in_middleware_uses_and_options(self):
        self.write("mw-slot", f"""
            [slots.audit]
            description = "middleware class"
            [fill]
            audit = "{MOD}:TagMiddleware"
            [[steps]]
            uses = "{MOD}:upper"
            [middleware.metrics]
            [middleware.audit]
            uses = "@audit"
            tag = "seen"
        """)
        result = load_preset("mw-slot").run("hi")
        self.assertEqual(result.metrics["counters"]["tag.seen"], 1)

    def test_doctor_reports_slot_status(self):
        self.abstract_preset()
        self.write("spare", f"""
            extends = "abstract"
            [slots.unused]
            description = "declared but never used"
        """)
        report = inspect_preset("spare")
        by_name = {row["slot"]: row for row in report["slots"]}
        self.assertEqual(by_name["[slots.transform]"]["status"], "missing")
        self.assertEqual(by_name["[slots.unused]"]["status"], "unreferenced")
        self.assertFalse(report["ok"])

        filled = inspect_preset("spare", fill={"transform": f"{MOD}:upper"})
        by_name = {row["slot"]: row for row in filled["slots"]}
        self.assertEqual(by_name["[slots.transform]"]["status"], "ok")
        self.assertEqual(by_name["[slots.transform]"]["source"], "fill=")
        self.assertTrue(filled["ok"])

    def test_cli_run_and_doctor_fill(self):
        from throughline.cli import main
        self.abstract_preset()
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["run", "abstract", "-i", "hi",
                         "--fill", f"transform={MOD}:upper"])
        self.assertEqual(code, 0)
        self.assertIn("HI", out.getvalue())

        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["doctor", "abstract", "--json",
                         "--fill", f"transform={MOD}:upper"])
        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertTrue(report["ok"])
        self.assertEqual(report["slots"][0]["fill"], f"{MOD}:upper")


class CompositeTests(SlotPresetBase):
    def test_map_over_payload(self):
        self.write("mapper", f"""
            [[steps]]
            map = "{MOD}:double"
            name = "double-all"
        """)
        self.assertEqual(load_preset("mapper").run([1, 2, 3]).output, [2, 4, 6])

    def test_map_with_workers_and_factory(self):
        self.write("mapper", f"""
            [[steps]]
            map = "{MOD}:make_prefixer"
            workers = 4
            [steps.with]
            prefix = "- "
        """)
        self.assertEqual(load_preset("mapper").run(["a", "b"]).output,
                         ["- a", "- b"])

    def test_parallel_gathers_dict(self):
        self.write("both", f"""
            [[steps]]
            name = "gather"
            [steps.parallel]
            twice = "{MOD}:double"
            minus = "{MOD}:negate"
        """)
        self.assertEqual(load_preset("both").run(3).output,
                         {"twice": 6, "minus": -3})

    def test_branch_on_payload_key_with_default(self):
        self.write("router", f"""
            [[steps]]
            [steps.branch]
            selector = "kind"
            default = "{MOD}:negate_value"
            [steps.branch.routes]
            twice = "{MOD}:double_value"
        """)
        flow = load_preset("router")
        self.assertEqual(flow.run({"kind": "twice", "value": 5}).output, 10)
        self.assertEqual(flow.run({"kind": "other", "value": 5}).output, -5)

    def test_branch_selector_import_path(self):
        self.write("router", f"""
            [[steps]]
            [steps.branch]
            selector = "{MOD}:pick_kind"
            [steps.branch.routes]
            neg = "{MOD}:negate_value"
        """)
        result = load_preset("router").run({"kind": "neg", "value": 7})
        self.assertEqual(result.output, -7)

    def test_composite_forms_are_exclusive(self):
        self.write("bad", f"""
            [[steps]]
            uses = "{MOD}:upper"
            map = "{MOD}:double"
        """)
        with self.assertRaises(PresetError) as caught:
            load_preset("bad")
        self.assertIn("exactly one of", str(caught.exception))

    def test_branch_requires_selector_and_routes(self):
        self.write("bad", f"""
            [[steps]]
            [steps.branch]
            default = "{MOD}:upper"
        """)
        with self.assertRaises(PresetError) as caught:
            load_preset("bad")
        self.assertIn("routes", str(caught.exception))

    def test_slot_as_composite_inner(self):
        self.write("mapper", """
            [slots.item_step]
            kind = "step"
            description = "applied to every item"
            [[steps]]
            map = "@item_step"
        """)
        flow = load_preset("mapper", fill={"item_step": f"{MOD}:double"})
        self.assertEqual(flow.run([2, 3]).output, [4, 6])

    def test_doctor_reports_composites(self):
        self.write("mixed", f"""
            [[steps]]
            map = "{MOD}:double"
            [[steps]]
            [steps.parallel]
            a = "{MOD}:double"
            b = "{MOD}:negate"
        """)
        report = inspect_preset("mixed")
        self.assertTrue(report["ok"])
        self.assertIn("map", report["steps"][0]["uses"])
        self.assertIn("composite map", report["steps"][0]["detail"])
        self.assertIn("parallel(a,b)", report["steps"][1]["uses"])


if __name__ == "__main__":
    unittest.main()
