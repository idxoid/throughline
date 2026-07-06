import os
import tempfile
import unittest
from pathlib import Path

import followers as fl
from followers.presets import build_flow, load_preset, load_preset_config


class PresetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._old_env = os.environ.get("FOLLOWERS_PRESETS")
        os.environ["FOLLOWERS_PRESETS"] = str(self.dir)

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop("FOLLOWERS_PRESETS", None)
        else:
            os.environ["FOLLOWERS_PRESETS"] = self._old_env
        self._tmp.cleanup()

    def write(self, name: str, content: str) -> Path:
        path = self.dir / f"{name}.toml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_and_run_by_path_and_name(self):
        self.write("upper", """
            name = "upper"
            [[steps]]
            uses = "followers.contrib.demo:normalize"
            [middleware.metrics]
        """)
        for ref in ("upper", str(self.dir / "upper.toml")):
            flow = load_preset(ref)
            result = flow.run("  Question  ")
            self.assertEqual(result.output["question"], "Question")
            self.assertEqual(result.metrics["counters"]["steps"], 1)

    def test_steps_with_factory_kwargs(self):
        self.write("rag", """
            [[steps]]
            uses = "followers.contrib.demo:retriever"
            name = "retrieve"
            [steps.with]
            top_k = 2
        """)
        flow = load_preset("rag")
        out = flow.run({"question": "lineage middleware"}).output
        self.assertLessEqual(len(out["context"]), 2)

    def test_extends_merging(self):
        self.write("base", """
            [config]
            top_k = 5
            temperature = 1
            [[steps]]
            uses = "followers.contrib.demo:normalize"
            [middleware.metrics]
            [middleware.validate]
            on_fail = "warn"
        """)
        self.write("child", """
            extends = "base"
            [config]
            top_k = 2
            [middleware.validate]
            on_fail = "raise"
        """)
        config = load_preset_config("child")
        self.assertEqual(config["config"], {"top_k": 2, "temperature": 1})
        self.assertEqual(config["middleware"]["validate"]["on_fail"], "raise")
        self.assertIn("metrics", config["middleware"])          # inherited
        self.assertEqual(len(config["steps"]), 1)               # inherited steps
        self.assertEqual(config["name"], "child")

    def test_extends_steps_replace_wholesale(self):
        self.write("base", """
            [[steps]]
            uses = "followers.contrib.demo:normalize"
            [[steps]]
            uses = "followers.contrib.demo:prompt"
        """)
        self.write("child", """
            extends = "base"
            [[steps]]
            uses = "followers.contrib.demo:normalize"
        """)
        self.assertEqual(len(load_preset_config("child")["steps"]), 1)

    def test_circular_extends_detected(self):
        self.write("a", 'extends = "b"')
        self.write("b", 'extends = "a"')
        with self.assertRaises(fl.PresetError):
            load_preset_config("a")

    def test_middleware_disable_and_custom_uses(self):
        self.write("custom", """
            [[steps]]
            uses = "followers.contrib.demo:normalize"
            [middleware.metrics]
            enabled = false
            [middleware.audit]
            uses = "followers.modules.observe:Observe"
        """)
        flow = load_preset("custom")
        self.assertEqual(len(flow.middleware), 1)
        self.assertEqual(type(flow.middleware[0]).__name__, "Observe")

    def test_unknown_middleware_and_bad_step_errors(self):
        self.write("bad-mw", """
            [[steps]]
            uses = "followers.contrib.demo:normalize"
            [middleware.nonsense]
        """)
        with self.assertRaises(fl.PresetError):
            load_preset("bad-mw")
        self.write("bad-step", """
            [[steps]]
            name = "no-uses"
        """)
        with self.assertRaises(fl.PresetError):
            load_preset("bad-step")

    def test_missing_preset_error_lists_search_dirs(self):
        with self.assertRaises(fl.PresetError) as caught:
            load_preset("does-not-exist")
        self.assertIn("not found", str(caught.exception))

    def test_builtin_demo_preset_end_to_end(self):
        flow = load_preset("demo")
        result = flow.run("how does line-level lineage work?")
        self.assertIn("answer", result.output)
        self.assertIn("Answer to:", result.output["answer"])
        self.assertGreater(result.metrics["counters"]["steps"], 2)
        self.assertIsNotNone(result.lineage)
        blame_steps = {entry["step"] for entry in result.lineage.blame()}
        self.assertIn("answer", blame_steps)
        self.assertEqual(result.violations, [])


class BuildFlowUnit(unittest.TestCase):
    def test_build_flow_minimal_dict(self):
        flow = build_flow({"name": "inline",
                           "steps": [{"uses": "followers.contrib.demo:normalize"}]})
        self.assertEqual(flow.name, "inline")
        self.assertEqual(flow.run("x").output, {"question": "x"})


if __name__ == "__main__":
    unittest.main()
