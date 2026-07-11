#!/usr/bin/env python3
"""User workflow: two harness transcripts → agent-audit report.

Packages the path operators actually run after recording two agent sessions:

    transcript convert (if needed) → agent-audit → report

Default inputs are the checked-in real Codex pair under
``examples/data/agent_sessions/codex-diverged/`` (same surgical_context
task, two runs, first behavioral divergence at tool event 1).

From the repository root::

    PYTHONPATH=src:. THROUGHLINE_PRESETS=examples/presets \\
      python3 examples/audit_diverged_runs.py

    PYTHONPATH=src:. THROUGHLINE_PRESETS=examples/presets \\
      python3 examples/audit_diverged_runs.py \\
        --baseline ~/.codex/sessions/.../rollout-a.jsonl \\
        --candidate ~/.codex/sessions/.../rollout-b.jsonl \\
        --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = (
    ROOT / "examples/data/agent_sessions/codex-diverged/baseline.jsonl"
)
DEFAULT_CAND = (
    ROOT / "examples/data/agent_sessions/codex-diverged/candidate.jsonl"
)


def _ensure_presets_path() -> None:
    presets = str(ROOT / "examples/presets")
    existing = os.environ.get("THROUGHLINE_PRESETS", "")
    if presets not in existing.split(os.pathsep):
        os.environ["THROUGHLINE_PRESETS"] = (
            presets if not existing else presets + os.pathsep + existing
        )


def _to_neutral(path: Path, work: Path) -> Path:
    """Return a neutral JSONL path, converting harness dialects when needed."""
    from throughline.adapters.transcripts import (
        convert_file,
        detect_format,
        read_jsonl,
    )

    raw = read_jsonl(path)
    kind = detect_format(raw)
    if kind == "neutral":
        return path
    dest = work / f"{path.stem}.neutral.jsonl"
    convert_file(path, dest, format=kind)
    print(f"converted {kind} {path.name} -> {dest.name} "
          f"({sum(1 for _ in dest.open())} events)", file=sys.stderr)
    return dest


def run_audit(baseline: Path, candidate: Path, *, as_json: bool) -> int:
    from throughline.presets import load_preset

    _ensure_presets_path()
    with tempfile.TemporaryDirectory(prefix="throughline-audit-") as tmp:
        work = Path(tmp)
        base = _to_neutral(baseline, work)
        cand = _to_neutral(candidate, work)
        flow = load_preset("agent-audit")
        started = time.perf_counter()
        result = flow.run({
            "baseline": str(base),
            "candidate": str(cand),
        })
        duration = time.perf_counter() - started

    out = result.output
    if as_json:
        payload = {
            "duration_s": round(duration, 3),
            "verdict": out["verdict"],
            "readiness_gate": out["readiness_gate"],
            "drift": out["drift"],
            "trace_health": out["trace_health"],
            "trace_divergence": out["trace_divergence"],
            "divergence": out["divergence"],
            "outcomes": out["outcomes"],
            "decisions": out["decisions"],
            "report": out["report"],
            "metrics": result.metrics.get("counters", {}),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(out["report"])
        print(file=sys.stderr)
        print(f"verdict={out['verdict']} readiness_gate={out['readiness_gate']} "
              f"duration_s={duration:.3f}", file=sys.stderr)
        first = next((item for item in out["trace_divergence"]
                      if item.get("kind") == "first_divergence"), None)
        if first:
            print(f"first_divergence@event {first.get('event')} "
                  f"({first.get('reason')})", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert two agent transcripts and run agent-audit")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASE,
                        help="baseline transcript (harness or neutral JSONL)")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CAND,
                        help="candidate transcript (harness or neutral JSONL)")
    parser.add_argument("--json", action="store_true",
                        help="print structured audit JSON instead of the report")
    args = parser.parse_args(argv)
    if not args.baseline.is_file():
        print(f"baseline not found: {args.baseline}", file=sys.stderr)
        return 2
    if not args.candidate.is_file():
        print(f"candidate not found: {args.candidate}", file=sys.stderr)
        return 2
    # Ensure local examples.* imports resolve when run as a script.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    return run_audit(args.baseline, args.candidate, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
