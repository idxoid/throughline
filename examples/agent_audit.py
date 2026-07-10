"""Example components for the `agent-audit` preset.

"Works on my machine" for AI agents: the same task succeeded yesterday and
failed today, and nothing in the diff of *your* code explains it. This preset
audits two recorded agent sessions (baseline vs candidate) and answers with
data instead of vibes:

    normalize -> load sessions -> build manifests -> extract decisions
              -> diff runs -> render report

- **manifest** ("lockfile" of a run): everything that shaped the agent's
  behavior, captured per session and grouped by category — model + sampling
  params, harness build + feature flags, runtime, repository state,
  dependency locks, whitelisted env value hashes, resolved prompt +
  instruction hashes, skills, MCP servers, tool schema hashes, network
  posture, workspace Merkle root, execution seed/locale/timezone;
- **drift**: a recursive diff of the two manifests as dotted paths
  (``model.temperature``, ``prompt.instructions.CLAUDE.md``), severity
  ranked by longest-prefix category match;
- **decisions**: commitment-like assistant lines ("I'll ...", "Plan: ...")
  with file:line provenance into the session transcript;
- **outcome**: status and token usage per session, with the failure excerpt.

The session fixtures are a neutral JSONL shape (one event per line) modeled
on what coding-agent harnesses record on disk; a real deployment would point
the same flow at exported Claude Code / Cursor / Codex transcripts. Secrets
never belong in a manifest — env vars are captured as whitelisted names with
value *hashes* — and policy egress redacts anything leaked into the report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from throughline.context import RunContext
from throughline.modules.policy import Transform

DATA_DIR = Path(__file__).resolve().parent / "data" / "agent_sessions"

DECISION_RE = re.compile(
    r"^(?:I'll|I will|Plan:|Decision:|Decided|Switching)", re.IGNORECASE
)
SECRET_RE = re.compile(r"\b(?:sk|key|token)-[A-Za-z0-9-]{8,}\b")

# Longest dotted-prefix match wins; anything unlisted defaults to "medium".
SEVERITY = {
    "model": "medium",              # release bumps are routine...
    "model.id": "high",             # ...a different model is not
    "model.temperature": "high",
    "model.top_p": "high",
    "model.reasoning_effort": "high",
    "prompt": "high",
    "skills": "high",
    "mcp": "high",
    "tools": "high",
    "dependencies": "high",
    "network": "high",
    "environment": "medium",
    "harness": "medium",
    "runtime": "medium",
    "repository": "medium",
    "execution": "medium",
    "workspace": "low",             # intended code changes live here
}


def normalize(payload, ctx: RunContext) -> dict:
    """Accept CLI strings and the real {"baseline", "candidate"} path shape."""
    baseline = DATA_DIR / "baseline.jsonl"
    candidate = DATA_DIR / "candidate.jsonl"
    if isinstance(payload, dict):
        baseline = Path(payload.get("baseline") or baseline)
        candidate = Path(payload.get("candidate") or candidate)
    return {"baseline_path": str(baseline), "candidate_path": str(candidate)}


def load_sessions(payload, ctx: RunContext) -> dict:
    """Read both session transcripts: one JSON event per line, order kept."""
    sessions = {}
    for label in ("baseline", "candidate"):
        path = Path(payload[f"{label}_path"])
        events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sessions[label] = events
        ctx.metric("audit.events", len(events))
    ctx.metric("audit.sessions", len(sessions))
    return {**payload, "sessions": sessions}


def build_manifests(payload, ctx: RunContext) -> dict:
    """Distill each session into manifest (effective config) + outcome."""
    manifests = {}
    for label, events in payload["sessions"].items():
        start = next(e for e in events if e["type"] == "session_start")
        end = next((e for e in events if e["type"] == "session_end"), {})
        usage = end.get("usage", {})
        manifests[label] = {
            "session_id": start["session_id"],
            "config": start["config"],
            "status": end.get("status", "unknown"),
            "usage": usage,
        }
        ctx.metric("llm.input_tokens", usage.get("input_tokens", 0))
        ctx.metric("llm.output_tokens", usage.get("output_tokens", 0))
    return {**payload, "manifests": manifests}


def extract_decisions(payload, ctx: RunContext) -> dict:
    """Commitment-like assistant lines, each with transcript-line provenance."""
    decisions = {}
    for label, events in payload["sessions"].items():
        found = []
        for line_no, event in enumerate(events, start=1):
            if event["type"] != "assistant":
                continue
            match = DECISION_RE.match(event["text"])
            if match:
                found.append({
                    "line": line_no,
                    "marker": match.group(0),
                    "quote": event["text"],
                })
        decisions[label] = found
        ctx.metric("audit.decisions", len(found))
    return {**payload, "decisions": decisions}


def diff_runs(payload, ctx: RunContext) -> dict:
    """Recursive manifest diff (dotted paths) plus the outcome comparison."""
    base = payload["manifests"]["baseline"]
    cand = payload["manifests"]["candidate"]
    drift: list[dict[str, Any]] = []
    _diff_tree(base["config"], cand["config"], "", drift)
    outcome_changed = base["status"] != cand["status"]
    if drift and outcome_changed:
        verdict = "drift_with_outcome_change"
    elif drift:
        verdict = "drift_only"
    elif outcome_changed:
        verdict = "outcome_change_no_drift"
    else:
        verdict = "clean"
    ctx.metric("audit.drift", len(drift))
    return {**payload, "drift": drift, "verdict": verdict}


def _diff_tree(base: dict, cand: dict, path: str, out: list) -> None:
    """Emit one drift entry per changed leaf; a subtree present on only one
    side (an added MCP server, a removed skill) stays one entry, not four."""
    for key in sorted(set(base) | set(cand)):
        sub_path = f"{path}.{key}" if path else str(key)
        before, after = base.get(key), cand.get(key)
        if isinstance(before, dict) and isinstance(after, dict):
            _diff_tree(before, after, sub_path, out)
        elif before != after:
            out.append({"field": sub_path, "baseline": before,
                        "candidate": after, "severity": _severity(sub_path)})


def _severity(path: str) -> str:
    parts = path.split(".")
    for size in range(len(parts), 0, -1):
        prefix = ".".join(parts[:size])
        if prefix in SEVERITY:
            return SEVERITY[prefix]
    return "medium"


def render_report(payload, ctx: RunContext) -> dict:
    """Return the strict public shape; raw transcripts stay internal."""
    manifests = payload["manifests"]
    lines = [
        f"# Agent run audit: {manifests['baseline']['session_id']}"
        f" vs {manifests['candidate']['session_id']}",
        "",
        f"Verdict: {payload['verdict']}",
        "",
        "## Manifest drift",
    ]
    for item in payload["drift"] or []:
        lines.append(f"- [{item['severity']}] {item['field']}:"
                     f" {_compact(item['baseline'])} -> {_compact(item['candidate'])}")
    if not payload["drift"]:
        lines.append("- none")
    lines.extend(["", "## Decisions"])
    for label in ("baseline", "candidate"):
        for decision in payload["decisions"][label]:
            lines.append(f"- {label} L{decision['line']}: \"{decision['quote']}\"")
    lines.extend(["", "## Outcome"])
    outcome = {}
    for label in ("baseline", "candidate"):
        manifest = manifests[label]
        usage = manifest["usage"]
        outcome[label] = {"status": manifest["status"], "usage": usage}
        lines.append(f"- {label}: {manifest['status']}"
                     f" (in={usage.get('input_tokens', 0)}"
                     f" out={usage.get('output_tokens', 0)} tokens)")
    excerpt = _failure_excerpt(payload["sessions"]["candidate"])
    if manifests["candidate"]["status"] != "ok" and excerpt:
        lines.extend(["", "## Failure excerpt", "", f"> {excerpt}"])
    return {
        "verdict": payload["verdict"],
        "drift": payload["drift"],
        "decisions": payload["decisions"],
        "outcome": outcome,
        "report": "\n".join(lines),
    }


def _compact(value: Any, limit: int = 48) -> str:
    if value is None:
        return "(absent)"
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) \
        else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _failure_excerpt(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event["type"] == "tool_result":
            return str(event.get("text", ""))
    return ""


def redact_secrets(checkpoint: str, value: Any, ctx: RunContext):
    """kind="policy": egress redaction of leaked keys in the public report."""
    if isinstance(value, dict) and isinstance(value.get("report"), str):
        report, count = SECRET_RE.subn("[secret redacted]", value["report"])
        if count:
            return Transform({**value, "report": report},
                             f"redacted {count} secret(s)")
    return None
