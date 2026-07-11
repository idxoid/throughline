import json
import unittest
from pathlib import Path

import throughline as tl
from throughline.modules import ManifestGate
from throughline.presets import load_preset

_REPO = Path(__file__).resolve().parents[1]
_LOCKFILE = _REPO / "examples/data/agent.lock.json"

_HARNESS = json.loads(_LOCKFILE.read_text(encoding="utf-8"))


class ManifestGateTests(unittest.TestCase):
    def test_passes_when_harness_matches_lockfile(self):
        flow = tl.Flow(
            [lambda p: p],
            middleware=[tl.modules.MetricsMiddleware(),
                        ManifestGate(lockfile=str(_LOCKFILE), root=str(_REPO))],
        )
        result = flow.run({"harness_config": _HARNESS, "root": str(_REPO)})
        manifest = result.ctx.artifacts["manifest"]
        self.assertEqual(manifest["gate"], "pass")
        self.assertEqual(result.metrics["counters"]["manifest.verify.passed"], 1)

    def test_blocks_on_model_drift(self):
        drifted = json.loads(json.dumps(_HARNESS))
        drifted["model"]["temperature"] = 0.9
        flow = tl.Flow(
            [lambda p: p],
            middleware=[tl.modules.MetricsMiddleware(),
                        ManifestGate(lockfile=str(_LOCKFILE), root=str(_REPO))],
        )
        with self.assertRaises(tl.FlowError) as caught:
            flow.run({"harness_config": drifted, "root": str(_REPO)})
        cause = caught.exception.__cause__
        self.assertIsInstance(cause, tl.ManifestVerifyError)
        self.assertEqual(cause.gate, "block")
        self.assertTrue(cause.violations)

    def test_payload_cannot_override_expected_manifest(self):
        locked = {"model": {"temperature": 0.0}}
        injected = {"model": {"temperature": 0.9}}
        flow = tl.Flow(
            [lambda p: p],
            middleware=[ManifestGate(expected=locked, root=str(_REPO))],
        )
        with self.assertRaises(tl.FlowError) as caught:
            flow.run({"harness_config": injected, "expected": injected})
        self.assertIsInstance(caught.exception.__cause__, tl.ManifestVerifyError)

    def test_warn_mode_continues(self):
        drifted = json.loads(json.dumps(_HARNESS))
        drifted["model"]["temperature"] = 0.9
        policy = {"model.temperature": "warn", "model": "ignore",
                  "model.id": "ignore", "model.top_p": "ignore",
                  "model.reasoning_effort": "ignore", "model.max_tokens": "ignore"}
        flow = tl.Flow(
            [lambda p: {"ok": True}],
            middleware=[ManifestGate(
                lockfile=str(_LOCKFILE), root=str(_REPO),
                policy=policy, on_fail="raise")],
        )
        result = flow.run({"harness_config": drifted, "root": str(_REPO)})
        self.assertEqual(result.ctx.artifacts["manifest"]["gate"], "warn")
        self.assertEqual(result.output, {"ok": True})

    def test_agent_preflight_preset_end_to_end(self):
        flow = load_preset("examples/presets/agent-preflight.toml")
        result = flow.run({"harness_config": _HARNESS, "root": str(_REPO)})
        self.assertEqual(result.output["gate"], "pass")
        self.assertEqual(result.output["violations"], [])
        self.assertIn("Gate: pass", result.output["report"])
        self.assertIn("model", result.output["observed"])

    def test_agent_preflight_output_redacts_secret_like_manifest_values(self):
        from examples.agent_preflight import render_report

        declared = {
            "model": {"id": "test-model"},
            "credentials": {"api_key": "sk-live-secret123"},
            "auth": {"header": "Bearer abcdefghi12345"},
        }
        flow = tl.Flow(
            [render_report],
            middleware=[ManifestGate(expected=declared, root=str(_REPO))],
        )
        result = flow.run({"harness_config": declared})
        rendered = json.dumps(result.output, sort_keys=True)
        self.assertNotIn("sk-live-secret123", rendered)
        self.assertNotIn("Bearer abcdefghi12345", rendered)
        self.assertEqual(result.output["expected"]["credentials"], "[redacted]")


if __name__ == "__main__":
    unittest.main()
