#!/usr/bin/env python3
"""Harness hook: live preflight + session_start JSONL line for agent harnesses.

Usage (from repository root):

    PYTHONPATH=src python3 -m examples.agent_session_hook start \\
        --session-id s-demo \\
        --config examples/data/agent.lock.json \\
        --lockfile examples/data/agent.lock.json \\
        --out /tmp/demo.jsonl

Prints the session_start event to stdout when ``--out`` is omitted.
Exits 1 when verify gate is ``block``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from throughline.manifest.session import (SessionRecorder, preflight_session_start,
                                          session_start_event)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_start(args: argparse.Namespace) -> int:
    declared = _load_json(Path(args.config))
    kwargs = {
        "root": args.root,
        "env_allowlist": args.env or [],
        "on_block": "raise" if args.strict else "return",
    }
    if args.lockfile:
        kwargs["lockfile"] = args.lockfile

    if args.out:
        recorder = SessionRecorder(args.out)
        result = recorder.start(args.session_id, declared, **kwargs)
        if result and result.gate == "block":
            return 1
        return 0 if not result or result.gate != "block" else 1

    config, result = preflight_session_start(declared, **kwargs)
    event = session_start_event(args.session_id, config)
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    if result and result.gate == "block":
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent session harness hook")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="preflight + emit session_start")
    start.add_argument("--session-id", required=True)
    start.add_argument("--config", required=True,
                       help="declared manifest JSON (harness effective config)")
    start.add_argument("--lockfile", help="expected lockfile for verify")
    start.add_argument("--root", default=".", help="workspace root to capture")
    start.add_argument("--env", action="append", default=[],
                       help="env var name to hash (repeatable)")
    start.add_argument("--out", help="append session_start to JSONL path")
    start.add_argument("--strict", action="store_true",
                       help="exit 1 on verify block (default when --out set)")
    start.set_defaults(func=cmd_start)

    args = parser.parse_args(argv)
    if args.out and not args.strict:
        args.strict = True
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
