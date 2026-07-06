import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from followers.cli import main


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_run_demo_plain(self):
        code, out, err = self.run_cli("run", "demo", "--input", "what are presets?")
        self.assertEqual(code, 0)
        self.assertIn("Answer to: what are presets?", out)

    def test_run_demo_json_report(self):
        code, out, _ = self.run_cli("run", "demo", "--input", "lineage?", "--json")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertIn("answer", report["output"])
        self.assertIn("counters", report["metrics"])
        self.assertIn("blame", report["lineage"])
        self.assertTrue(report["lineage"]["blame"])

    def test_run_demo_blame_and_metrics_flags(self):
        code, _, err = self.run_cli("run", "demo", "--input", "middleware?",
                                    "--blame", "--metrics")
        self.assertEqual(code, 0)
        self.assertIn("lineage (blame)", err)
        self.assertIn("metrics", err)

    def test_presets_lists_demo(self):
        code, out, _ = self.run_cli("presets")
        self.assertEqual(code, 0)
        self.assertIn("demo", out)

    def test_missing_preset_is_clean_error(self):
        code, _, err = self.run_cli("run", "nope-not-here")
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_steps_command_runs(self):
        code, _, _ = self.run_cli("steps")
        self.assertEqual(code, 0)

    def test_contrib_demo_can_run_as_script(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [
                sys.executable,
                str(root / "src" / "followers" / "contrib" / "demo.py"),
                "--input",
                "what are presets?",
            ],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Answer to: what are presets?", proc.stdout)


if __name__ == "__main__":
    unittest.main()
