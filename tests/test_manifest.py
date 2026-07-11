import json
import tempfile
import unittest
from pathlib import Path

from throughline.manifest import (DEFAULT_VERIFY_POLICY, SOURCE_HARNESS,
                                    SOURCE_LIVE_PROBE, capture_environment,
                                    env_hash, flatten_observed, observed_sources,
                                    short_digest, verify_manifest)
from throughline.manifest.capture import git_snapshot, workspace_merkle_root
from throughline.manifest.verify import load_lockfile, policy_action


class ManifestVerifyTests(unittest.TestCase):
    def test_identical_manifests_pass(self):
        lock = {
            "model": {"id": "claude-opus-4-8", "temperature": 0.0},
            "repository": {"commit": "abc", "dirty": False},
        }
        result = verify_manifest(lock, dict(lock))
        self.assertEqual(result.gate, "pass")
        self.assertEqual(result.violations, [])

    def test_observed_extras_outside_lockfile_are_ignored(self):
        expected = {"model": {"id": "a", "temperature": 0.0}}
        observed = {
            "model": {"id": "a", "temperature": 0.0},
            "repository": {"dirty": True},
            "workspace": {"merkle_root": "m-x"},
        }
        result = verify_manifest(expected, observed)
        self.assertEqual(result.gate, "pass")

    def test_dirty_tree_blocks(self):
        expected = {"repository": {"commit": "abc", "dirty": False}}
        observed = {"repository": {"commit": "abc", "dirty": True}}
        result = verify_manifest(expected, observed)
        self.assertEqual(result.gate, "block")
        self.assertEqual(result.violations[0].field, "repository.dirty")

    def test_same_commit_different_merkle_blocks(self):
        expected = {
            "repository": {"commit": "d4c3b2a1"},
            "workspace": {"merkle_root": "m-aaaa1111"},
        }
        observed = {
            "repository": {"commit": "d4c3b2a1"},
            "workspace": {"merkle_root": "m-bbbb2222"},
        }
        result = verify_manifest(expected, observed)
        self.assertEqual(result.gate, "block")
        self.assertEqual(result.violations[0].field, "workspace.merkle_root")

    def test_temperature_drift_blocks(self):
        expected = {"model": {"temperature": 0.0}}
        observed = {"model": {"temperature": 0.7}}
        result = verify_manifest(expected, observed)
        self.assertEqual(result.gate, "block")
        self.assertEqual(result.violations[0].field, "model.temperature")

    def test_runtime_patch_is_ignored_by_default(self):
        expected = {"runtime": {"python": "3.12.4"}}
        observed = {"runtime": {"python": "3.12.5"}}
        result = verify_manifest(expected, observed)
        self.assertEqual(result.gate, "pass")

    def test_commit_mismatch_warns(self):
        expected = {"repository": {"commit": "aaaa1111"}}
        observed = {"repository": {"commit": "bbbb2222"}}
        result = verify_manifest(expected, observed)
        self.assertEqual(result.gate, "warn")
        self.assertEqual(result.violations[0].action, "warn")

    def test_custom_policy_can_ignore_model(self):
        expected = {"model": {"id": "a"}}
        observed = {"model": {"id": "b"}}
        result = verify_manifest(expected, observed, {"model.id": "ignore"})
        self.assertEqual(result.gate, "pass")

    def test_policy_longest_prefix_wins(self):
        self.assertEqual(
            policy_action("model.temperature", DEFAULT_VERIFY_POLICY), "block")
        self.assertEqual(
            policy_action("runtime.python", DEFAULT_VERIFY_POLICY), "ignore")

    def test_zero_token_outcomes_silence_not_applicable_here(self):
        """Verify gate with block beats warn."""
        expected = {
            "repository": {"commit": "a", "dirty": False},
            "model": {"temperature": 0.0},
        }
        observed = {
            "repository": {"commit": "b", "dirty": True},
            "model": {"temperature": 0.7},
        }
        result = verify_manifest(expected, observed)
        self.assertEqual(result.gate, "block")
        actions = {v.action for v in result.violations}
        self.assertIn("block", actions)
        self.assertIn("warn", actions)


class ManifestCaptureTests(unittest.TestCase):
    def test_env_hash_is_full_sha256(self):
        digest = env_hash("secret-value")
        self.assertEqual(len(digest), 64)
        self.assertEqual(env_hash("x"), env_hash("x"))
        self.assertEqual(short_digest(digest), digest[:8])

    def test_capture_separates_live_and_harness_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            observed = capture_environment(
                tmp,
                harness={"model": {"id": "test-model", "temperature": 0.0}},
                env_allowlist=("HOME",),
                environ={"HOME": "/tmp/home"},
            )
        self.assertEqual(set(observed), {"live", "harness"})
        self.assertEqual(observed["harness"]["model"]["id"], "test-model")
        self.assertIn("HOME", observed["live"]["environment"])
        self.assertIn("dirty", observed["live"]["repository"])
        self.assertNotIn("model", observed["live"])
        self.assertNotIn("repository", observed["harness"])
        sources = observed_sources(observed)
        self.assertEqual(sources["repository"], SOURCE_LIVE_PROBE)
        self.assertEqual(sources["model"], SOURCE_HARNESS)

    def test_harness_cannot_supply_live_observed_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                capture_environment(
                    tmp,
                    harness={
                        "repository": {"commit": "fake", "dirty": False},
                        "model": {"id": "test-model"},
                    },
                )
        self.assertIn("live-observed fields", str(caught.exception))
        self.assertIn("repository", str(caught.exception))

    def test_flatten_rejects_harness_live_field_override(self):
        observed = {
            "live": {"repository": {"commit": "real", "dirty": True}},
            "harness": {
                "repository": {"commit": "fake", "dirty": False},
                "model": {"id": "test-model"},
            },
        }
        with self.assertRaises(ValueError) as caught:
            flatten_observed(observed)
        self.assertIn("live-observed fields", str(caught.exception))
        self.assertIn("repository", str(caught.exception))

    def test_observed_sources_rejects_harness_live_field_override(self):
        observed = {
            "live": {"workspace": {"merkle_root": "m-real"}},
            "harness": {
                "workspace": {"merkle_root": "m-fake"},
                "model": {"id": "test-model"},
            },
        }
        with self.assertRaises(ValueError):
            observed_sources(observed)

    def test_declared_config_cannot_spoof_live_workspace(self):
        from throughline.manifest import verify_live

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tracked.py").write_text("live-content", encoding="utf-8")
            declared = {
                "repository": {
                    "commit": "fake",
                    "dirty": False,
                },
                "workspace": {
                    "merkle_root": "m-fake",
                },
                "model": {
                    "id": "test-model",
                },
            }
            observed, _ = verify_live(declared, root=root)

        self.assertNotEqual(observed["live"]["repository"]["commit"], "fake")
        self.assertNotEqual(observed["live"]["workspace"]["merkle_root"], "m-fake")
        self.assertEqual(observed["harness"]["model"]["id"], "test-model")

    def test_workspace_merkle_changes_when_files_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = workspace_merkle_root(root)
            (root / "a.py").write_text("one", encoding="utf-8")
            after = workspace_merkle_root(root)
        self.assertNotEqual(before, after)

    def test_git_snapshot_on_repo(self):
        repo_root = Path(__file__).resolve().parents[1]
        snap = git_snapshot(repo_root)
        self.assertIsNotNone(snap["commit"])
        self.assertEqual(len(snap["commit"]), 40)
        self.assertIsInstance(snap["dirty"], bool)

    def test_workspace_merkle_uses_full_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("one", encoding="utf-8")
            digest = workspace_merkle_root(root)
        self.assertTrue(digest.startswith("m-"))
        self.assertEqual(len(digest), 2 + 64)

    def test_load_lockfile_json_and_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = {"model": {"id": "x"}}
            json_path = root / "agent.lock.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(load_lockfile(json_path), data)

            toml_path = root / "agent.lock.toml"
            toml_path.write_text('[model]\nid = "y"\n', encoding="utf-8")
            self.assertEqual(load_lockfile(toml_path)["model"]["id"], "y")

    def test_live_capture_then_verify_passes_on_self(self):
        repo_root = Path(__file__).resolve().parents[1]
        observed = capture_environment(
            repo_root,
            harness={
                "model": {"id": "claude-opus-4-8", "temperature": 0.0},
                "prompt": {"system_sha256": "beefcafe"},
            },
        )
        flat = flatten_observed(observed)
        result = verify_manifest(flat, flat)
        self.assertEqual(result.gate, "pass")
        self.assertEqual(flat["model"]["id"], "claude-opus-4-8")
        self.assertIn("repository", flat)


if __name__ == "__main__":
    unittest.main()
