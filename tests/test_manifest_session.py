import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import throughline as tl
from throughline.manifest.session import (SessionRecorder, capture_drift,
                                          declared_config, effective_environment,
                                          preflight_session_start,
                                          session_start_event)

_REPO = Path(__file__).resolve().parents[1]
_LOCK = _REPO / "examples/data/agent.lock.json"
_HARNESS = json.loads(_LOCK.read_text(encoding="utf-8"))


class ManifestSessionTests(unittest.TestCase):
    def test_preflight_embeds_observed_and_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, result = preflight_session_start(
                _HARNESS, root=tmp, lockfile=str(_LOCK))
        self.assertIn("observed", config)
        self.assertIn("repository", config["observed"]["live"])
        self.assertEqual(config["verify"]["gate"], "pass")
        self.assertIsNotNone(result)
        self.assertEqual(declared_config(config)["model"], _HARNESS["model"])
        self.assertIn("model", config["observed"]["harness"])
        self.assertNotIn("model", config["observed"]["live"])

    def test_preflight_blocks_when_declared_lies_about_temperature(self):
        lied = json.loads(json.dumps(_HARNESS))
        lied["model"]["temperature"] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "throughline.manifest.session.capture_environment",
                return_value={"model": {"temperature": 0.9}},
            ):
                with self.assertRaises(tl.ManifestVerifyError):
                    preflight_session_start(
                        lied, root=tmp, lockfile=str(_LOCK), on_block="raise")

    def test_session_recorder_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            recorder = SessionRecorder(path)
            recorder.start("s-1", _HARNESS, root=tmp, lockfile=str(_LOCK))
            recorder.append({"type": "assistant", "text": "hi"})
            recorder.end(status="ok", usage={"input_tokens": 1, "output_tokens": 2})
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        start = json.loads(lines[0])
        self.assertEqual(start["type"], "session_start")
        self.assertIn("observed", start["config"])

    def test_session_start_drops_secrets_from_declared(self):
        dirty = json.loads(json.dumps(_HARNESS))
        dirty["api_key"] = "sk-live-secret123"
        dirty["authorization"] = "Bearer abcdefghi12345"
        dirty["model"]["api_key"] = "nested-secret"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            recorder = SessionRecorder(path)
            recorder.start("s-secret", dirty, root=tmp, lockfile=str(_LOCK))
            raw = path.read_text(encoding="utf-8")
        self.assertNotIn("sk-live-secret123", raw)
        self.assertNotIn("Bearer abcdefghi12345", raw)
        self.assertNotIn("nested-secret", raw)
        start = json.loads(raw.splitlines()[0])
        self.assertNotIn("api_key", start["config"])
        self.assertNotIn("authorization", start["config"])
        self.assertEqual(start["config"]["model"]["api_key"], "[redacted]")
        self.assertEqual(start["config"]["model"]["id"], _HARNESS["model"]["id"])

    def test_capture_drift_detects_declared_vs_observed(self):
        declared = {"repository": {"dirty": False, "commit": "abc"}}
        observed = {"repository": {"dirty": True, "commit": "abc"}}
        drift = capture_drift(declared, observed)
        self.assertEqual(drift[0]["field"], "repository.dirty")

    def test_effective_environment_prefers_observed_workspace_facts(self):
        manifest = {
            "config": {"repository": {"dirty": False}, "model": {"id": "x"}},
            "observed": {
                "live": {"repository": {"dirty": True}},
                "harness": {"model": {"id": "x"}},
            },
        }
        env = effective_environment(manifest)
        self.assertTrue(env["repository"]["dirty"])
        self.assertEqual(env["model"]["id"], "x")

    def test_effective_environment_accepts_flat_legacy_observed(self):
        manifest = {
            "config": {"repository": {"dirty": False}, "model": {"id": "x"}},
            "observed": {"repository": {"dirty": True}},
        }
        env = effective_environment(manifest)
        self.assertTrue(env["repository"]["dirty"])
        self.assertEqual(env["model"]["id"], "x")

    def test_agent_session_hook_start_stdout(self):
        from examples.agent_session_hook import main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main([
                "start",
                "--session-id", "s-hook",
                "--config", str(_LOCK),
                "--lockfile", str(_LOCK),
                "--root", tmp,
            ])
        self.assertEqual(rc, 0)


class ManifestSessionAuditTests(unittest.TestCase):
    def test_audit_uses_recorded_observed_for_readiness(self):
        from examples.agent_audit import _readiness_for

        manifest = {
            "config": {"repository": {"dirty": False, "commit": "abc"},
                       "workspace": {"merkle_root": "m-a"}},
            "observed": {
                "live": {
                    "repository": {"dirty": True, "commit": "abc"},
                    "workspace": {"merkle_root": "m-a"},
                },
                "harness": {},
            },
        }
        ready = _readiness_for(manifest, [], {"risky_calls": []}, None)
        self.assertFalse(ready["can_start"])
        self.assertIn("repository_dirty", {b["id"] for b in ready["blockers"]})


if __name__ == "__main__":
    unittest.main()
