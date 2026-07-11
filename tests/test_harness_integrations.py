import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from throughline.adapters.transcripts import (convert_events, convert_file,
                                               detect_format, read_jsonl)
from throughline.cli import main
from throughline.manifest import (capture_lockfile, extract_harness_config,
                                    update_lockfile, verify_lockfile)

_DATA = Path(__file__).resolve().parent / "data"
_TRANSCRIPTS = _DATA / "transcripts"
_HOMES = _DATA / "harness_homes"
_WORKSPACES = _DATA / "workspaces"


class TranscriptAdapterTests(unittest.TestCase):
    def test_detect_and_convert_claude_code(self):
        raw = read_jsonl(_TRANSCRIPTS / "claude_code_sample.jsonl")
        self.assertEqual(detect_format(raw), "claude-code")
        events = convert_events(raw)
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "session_start")
        self.assertIn("user", types)
        self.assertIn("tool_call", types)
        self.assertIn("tool_result", types)
        self.assertEqual(events[0]["config"]["model"]["id"], "claude-opus-4-8")
        call = next(e for e in events if e["type"] == "tool_call")
        self.assertEqual(call["name"], "Read")
        self.assertEqual(call["call_id"], "toolu_1")

    def test_detect_and_convert_cursor(self):
        raw = read_jsonl(_TRANSCRIPTS / "cursor_sample.jsonl")
        self.assertEqual(detect_format(raw), "cursor")
        events = convert_events(raw, session_id="cur-1")
        self.assertEqual(events[0]["session_id"], "cur-1")
        user = next(e for e in events if e["type"] == "user")
        self.assertEqual(user["text"], "add logging")
        calls = [e for e in events if e["type"] == "tool_call"]
        self.assertEqual({c["name"] for c in calls}, {"Read", "StrReplace"})

    def test_detect_and_convert_codex(self):
        raw = read_jsonl(_TRANSCRIPTS / "codex_sample.jsonl")
        self.assertEqual(detect_format(raw), "codex")
        events = convert_events(raw)
        self.assertEqual(events[0]["session_id"], "codex-thread-1")
        self.assertTrue(any(e["type"] == "tool_call" for e in events))
        self.assertTrue(any(e["type"] == "tool_result" for e in events))
        end = events[-1]
        self.assertEqual(end["type"], "session_end")
        self.assertEqual(end["usage"]["input_tokens"], 10)

    def test_detect_and_convert_codex_rollout(self):
        raw = read_jsonl(_TRANSCRIPTS / "codex_rollout_sample.jsonl")
        self.assertEqual(detect_format(raw), "codex")
        events = convert_events(raw)
        self.assertEqual(events[0]["session_id"], "019e74e8-rollout-demo")
        self.assertEqual(events[0]["config"]["harness"]["version"], "0.133.0-alpha.1")
        self.assertEqual(events[0]["config"]["model"]["id"], "gpt-5.5")
        self.assertEqual(events[0]["config"]["model"]["reasoning_effort"], "xhigh")
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "session_start")
        self.assertIn("user", types)
        self.assertIn("assistant", types)
        calls = [e for e in events if e["type"] == "tool_call"]
        results = [e for e in events if e["type"] == "tool_result"]
        self.assertEqual({c["name"] for c in calls}, {"exec_command", "apply_patch"})
        self.assertEqual(calls[0]["args"]["cmd"], "sed -n '1,20p' README.md")
        self.assertEqual({r["call_id"] for r in results}, {"call_demo_1", "call_demo_2"})
        end = events[-1]
        self.assertEqual(end["type"], "session_end")
        self.assertEqual(end["usage"]["input_tokens"], 100)
        self.assertEqual(end["usage"]["output_tokens"], 20)

    def test_convert_file_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out.jsonl"
            events = convert_file(_TRANSCRIPTS / "cursor_sample.jsonl", dest)
            self.assertTrue(dest.is_file())
            self.assertEqual(len(read_jsonl(dest)), len(events))


class HarnessExtractTests(unittest.TestCase):
    def test_extract_claude_code(self):
        cfg = extract_harness_config(
            _WORKSPACES / "claude_proj",
            kind="claude-code",
            home=_HOMES / "claude",
        )
        self.assertEqual(cfg["harness"]["name"], "claude-code")
        self.assertEqual(cfg["model"]["id"], "claude-test-model")
        self.assertEqual(cfg["model"]["reasoning_effort"], "medium")
        self.assertIn("CLAUDE.md", cfg["prompt"]["instructions"])
        self.assertIn("git", cfg["mcp"])
        self.assertIn("args_sha256", cfg["mcp"]["git"])

    def test_extract_codex(self):
        cfg = extract_harness_config(
            _WORKSPACES / "claude_proj",
            kind="codex",
            home=_HOMES / "codex",
        )
        self.assertEqual(cfg["harness"]["name"], "codex")
        self.assertEqual(cfg["model"]["id"], "gpt-test")
        self.assertIn("demo", cfg["mcp"])

    def test_extract_cursor_hashes_env(self):
        cfg = extract_harness_config(
            _WORKSPACES / "cursor_proj",
            kind="cursor",
            home=_HOMES / "cursor",
        )
        self.assertEqual(cfg["harness"]["name"], "cursor")
        self.assertIn(".cursorrules", cfg["prompt"]["instructions"])
        self.assertNotIn("secret-value", json.dumps(cfg))
        self.assertIn("demo", cfg["mcp"])


class LockfileCliTests(unittest.TestCase):
    def test_capture_update_verify_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "agent.lock.json"
            cfg = capture_lockfile(
                lock,
                root=_WORKSPACES / "claude_proj",
                harness="claude-code",
                home=_HOMES / "claude",
            )
            self.assertTrue(lock.is_file())
            self.assertEqual(cfg["model"]["id"], "claude-test-model")

            updated = update_lockfile(
                lock,
                root=_WORKSPACES / "claude_proj",
                harness="claude-code",
                home=_HOMES / "claude",
            )
            self.assertEqual(updated["model"]["id"], "claude-test-model")

            # Verify using the lockfile itself as declared harness attestation.
            observed, result = verify_lockfile(
                lock,
                root=_WORKSPACES / "claude_proj",
                harness=None,
                declared=cfg,
            )
            self.assertEqual(result.gate, "pass")
            self.assertIn("live", observed)
            self.assertIn("harness", observed)

    def test_lockfile_capture_and_verify_redact_declared_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "agent.lock.json"
            declared = {
                "model": {"id": "test-model", "api_key": "sk-live-secret123"},
                "authorization": "Bearer abcdefghi12345",
            }
            cfg = capture_lockfile(lock, root=tmp, declared=declared)
            raw = lock.read_text(encoding="utf-8")
            self.assertNotIn("sk-live-secret123", raw)
            self.assertNotIn("Bearer abcdefghi12345", raw)
            self.assertEqual(cfg["model"]["api_key"], "[redacted]")

            observed, result = verify_lockfile(
                lock,
                root=tmp,
                harness=None,
                declared=declared,
            )
            rendered = json.dumps({
                "observed": observed,
                "violations": [asdict(v) for v in result.violations],
            }, sort_keys=True)
            self.assertNotIn("sk-live-secret123", rendered)
            self.assertEqual(observed["harness"]["model"]["api_key"], "[redacted]")

    def test_cli_lockfile_capture_and_transcript_convert(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "lock.json"
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = main([
                    "lockfile", "capture",
                    "--out", str(lock),
                    "--root", str(_WORKSPACES / "claude_proj"),
                    "--harness", "claude-code",
                    "--config", str(_DATA.parent.parent / "examples/data/agent.lock.json"),
                    "--json",
                ])
            self.assertEqual(code, 0)
            self.assertTrue(lock.is_file())

            dest = Path(tmp) / "neutral.jsonl"
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = main([
                    "transcript", "convert",
                    "-i", str(_TRANSCRIPTS / "codex_sample.jsonl"),
                    "-o", str(dest),
                    "--json",
                ])
            self.assertEqual(code, 0)
            summary = json.loads(out.getvalue())
            self.assertEqual(summary["format"], "codex")
            self.assertGreater(summary["events"], 0)


if __name__ == "__main__":
    unittest.main()
