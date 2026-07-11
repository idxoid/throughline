"""Example components for the `agent-audit` preset.

"Works on my machine" for AI agents: the same task succeeded yesterday and
"succeeded" today — both runs green — yet today's is green for the wrong
reasons. This preset audits two recorded agent sessions and answers with
data instead of vibes:

    normalize -> load sessions -> build manifests -> assess outcomes
              -> extract traces -> assess readiness -> extract decisions
              -> diff runs -> render report

Three layers, deliberately separate:

- **manifest** ("lockfile" of a run): everything that shaped the agent's
  behavior — model + sampling params, harness build + feature flags,
  runtime, repository state, dependency locks, whitelisted env value hashes,
  resolved prompt + instruction hashes, skills, MCP servers, tool schema
  hashes, network posture, workspace Merkle root, execution seed. The
  **cause** side.
- **trace** ("flight recorder" of a run): the normalized tool-call sequence,
  one entry per call — tool, arguments hash, status (ok / error / denied /
  no result), result hash, duration. Results join their calls by
  ``call_id`` (the tool-use id real harnesses record); a transcript without
  ids falls back to tool-name inference — pairing quality is separate from
  trace completeness (unresolved calls, orphan results, duplicate call_ids).
  An empty trace diff means identical behavior even when manifests drifted.
- **outcome fingerprint**: what the run actually *did*, across dimensions —
  status is only one. Also: which files it touched (including test files),
  risky tool calls, parsed test
  results, and token spend. A run can be ``status=ok`` and still diverge on
  every other dimension. The diff over fingerprints is symmetric — a
  vanished risky call, a shrunken test suite, or a big token drop is still
  divergence — with a separate ``assessment`` (regression / improvement /
  neutral) judging direction. The **effect** side.
- **readiness** (environment readiness *assessment*): from a *completed*
  session's manifest and trace, was the recorded environment fit for a clean
  rerun — pinned snapshot, denials, risky workarounds after denial. This is
  not live preflight enforcement; it consumes recorded facts. Live enforcement
  sits upstream in ``ManifestGate`` / ``preflight_session_start``:
  verify → block/warn/pass → execute → trace → audit. The **assessment** side.

Decisions ride along as a fifth, softer channel: sentences classed as
**plan / decision / assumption / action**. Extraction is gated by event
type — tool-result bodies (source, tests, JSON, stack traces, whole files)
are **not** treated as agent speech by default:

* assistant / reasoning / plan / commentary — full text
* tool_call — command / intent-like args only
* tool_result — skip (opt-in per tool name via ``decision_tools``)
* user — optional (``include_user``, for constraints/requirements)
* session / system metadata — never

Pipeline after the event filter:

1. cheap marker / cue extraction on every kept sentence
2. semantic pass **only** on recall-filter candidates (decision-like
   stems: decided / will / should / because / instead / approach /
   assume / plan / switch / avoid / choose) — never on every sentence

Hard budgets (``max_chars`` / ``max_sentences`` / ``max_semantic``) raise
``DecisionBudgetExceeded`` when exhausted — never silent truncation.

Stage 1 markers are auditable and replayable; stage 2 is the optional
``@semantic`` preset slot — filled here with a cheap cue heuristic so the
example runs offline, pointed at an LLM extractor in a real deployment —
which may add classes the markers missed, never override or remove them.
Every item keeps its evidence: the cue that fired, the sentence quote, the
quote's char span, and the ``channel`` it came from.

`diff_runs` compares completed runs (why two finished sessions differ).
``assess_readiness`` assesses whether the recorded candidate environment
looked fit for a clean rerun (post-hoc). Denials surface here and in matched
trace diffs only when status differs between runs. The session
fixtures are a neutral JSONL shape (one event per line) modeled on what
coding-agent harnesses record on disk; a real deployment would point the
same flow at exported Claude Code / Cursor / Codex transcripts. Secrets
never belong in a manifest (env vars are name -> value-hash), and policy
egress recursively redacts anything an agent leaks — via an example-grade
``SECRET_RE`` (``sk-``/``key-``/``token-`` prefixes only; swap for a
``@secret_detector`` slot in production). Scrubbing applies to the report
string and every structured field of the public output alike. The lineage
blame trail applies the same scrub at capture time: lineage's run-end
sweep re-attributes the egress redaction in the final blame, but earlier
ledger records keep whatever text they captured, so the example scrubs
before recording anything at all.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from throughline.context import RunContext
from throughline.manifest.session import (
    capture_drift as compute_capture_drift,
    declared_config,
    effective_environment,
)
from throughline.modules.policy import Transform

DATA_DIR = Path(__file__).resolve().parent / "data" / "agent_sessions"

# Stage-1 decision markers, applied per sentence (anywhere in it, not only
# at line starts). Tuple order is class precedence — a sentence gets the
# first class whose pattern hits: plan outranks decision outranks assumption
# outranks action, because the generic cues ("I'll") appear inside nearly
# every planning or deciding sentence too.
MARKER_RULES = (
    ("plan", re.compile(
        r"^plan\b|\bplan:|\bmy plan\b|\bfirst\b.*\bthen\b", re.IGNORECASE)),
    ("decision", re.compile(
        r"\bdecided\b|\bdecision:|\bswitching to\b|\binstead of\b"
        r"|\bgoing with\b|\bopting for\b|\brather than\b", re.IGNORECASE)),
    ("assumption", re.compile(
        r"\bassum\w+\b|\bpresumably\b|\blikely\b|\bprobably\b"
        r"|\bshould (?:still |already )?(?:be|work|pass)\b"
        r"|\bexpect(?:s|ing)? that\b", re.IGNORECASE)),
    ("action", re.compile(
        r"\bi(?:'|’)ll\b|\bi will\b|\blet me\b|\bgoing to\b|\brunning\b",
        re.IGNORECASE)),
)

# Cheap offline stand-in for the @semantic slot: commitment phrasings that
# carry no stage-1 marker. A real deployment fills the slot with an
# LLM-backed classifier honoring the same contract.
SEMANTIC_CUES = (
    ("decision", re.compile(
        r"\bthe (?:right|simplest|safest|correct) (?:fix|approach|way|call)\b"
        r"|\bthe fix is\b", re.IGNORECASE)),
    ("assumption", re.compile(
        r"\bshould (?:be enough|suffice)\b|\bin theory\b", re.IGNORECASE)),
    ("action", re.compile(
        r"\bnext step is\b|\btime to\b", re.IGNORECASE)),
)
# Example-grade secret detector — catches sk-/key-/token- prefixes only.
# Production deployments should fill a @secret_detector slot (ghp_, AKIA…,
# Bearer tokens, JWT, etc.).
SECRET_RE = re.compile(r"\b(?:sk|key|token)-[A-Za-z0-9-]{8,}\b")
TEST_PATH_RE = re.compile(r"(^|/)tests?/|(^|/)test_|_test\.")
TESTS_RE = re.compile(
    r"(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+(?:skipped|xfailed)")
# Strip timing/noise from tool output before result hashing — otherwise
# "4 passed in 1.31s" and "4 passed in 1.52s" look like divergent behavior.
PYTEST_ELAPSED_RE = re.compile(r"\bin\s+\d[\d.]*s\b")
HTTP_TIMING_RE = re.compile(r"time=\d[\d.]*|<\s*\d[\d.]*")
EDIT_TOOLS = {"Edit", "Write", "Create"}
FILESYSTEM_TOOLS = frozenset({"Read", "Write", "Edit", "Create", "Grep", "Glob",
                              "ListDir"})
TOKEN_BLOWUP = 1.5  # token-ratio flag threshold, either direction (x or 1/x)

# Event types whose full text is agent speech / planning (not tool I/O).
_DECISION_FULL_TEXT_TYPES = frozenset({
    "assistant", "reasoning", "reasoning_summary", "plan", "commentary",
    "thinking",
})
# tool_call args keys that carry intent (commands, goals) — not file bodies.
_TOOL_ARG_INTENT_KEYS = frozenset({
    "command", "cmd", "query", "prompt", "message", "description",
    "goal", "intent", "reason", "title", "plan", "thought", "commentary",
    "rationale", "summary", "task",
})

# High-recall gate before @semantic — decision-like stems only. Markers still
# run on every kept sentence; semantic never scans the full transcript.
# ``fix`` is included so the example @semantic cues ("the right fix" / "the
# fix is") remain reachable without a full-sentence semantic scan.
_SEMANTIC_RECALL_RE = re.compile(
    r"(?i)(?:\b(?:decided|will|should|because|instead|approach|fix|"
    r"assum\w*|plans?|switch\w*|avoid\w*|choos\w*|chose|chosen)\b"
    r"|\bplan:|i['’]ll)"
)

# Hard caps for decision extraction. Exhaustion raises — never truncates.
DEFAULT_MAX_CHARS = 200_000
DEFAULT_MAX_SENTENCES = 4_000
DEFAULT_MAX_SEMANTIC = 400


# Bash substrings that make a tool call risky, with the risk label.
RISKY_PATTERNS = (
    ("rm -rf", "destructive-delete"),
    ("| sh", "pipe-to-shell"),
    ("| bash", "pipe-to-shell"),
    ("curl", "external-network"),
    ("wget", "external-network"),
    ("git push --force", "history-rewrite"),
    ("chmod 777", "permission-change"),
    ("sudo", "privilege-escalation"),
)

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
    "repository.dirty": "high",       # same commit, dirty tree -> WOM drift
    "execution": "medium",
    "workspace": "low",               # output artifacts; not input snapshot
    "workspace.merkle_root": "high",  # source snapshot must match for replay
}

# Severity per trace-divergence kind. Reorders rank low: parallel tool
# batches legitimately land in different orders between runs.
TRACE_SEVERITY = {
    "first_divergence": "high",
    "result_changed": "high",
    "denied": "high",
    "args_changed": "medium",
    "calls_missing": "medium",
    "calls_added": "medium",
    "reordered": "low",
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
    """Distill each session into its run manifest (effective config).

    When ``session_start.config`` carries harness metadata from Phase 3
    capture (``observed``, ``verify``), declared config is split out so
    audit diffs stay comparable while capture drift remains inspectable.
    """
    manifests = {}
    for label, events in payload["sessions"].items():
        start = next(e for e in events if e["type"] == "session_start")
        raw = start["config"]
        manifests[label] = {
            "session_id": start["session_id"],
            "config": declared_config(raw),
            "observed": raw.get("observed"),
            "verify": raw.get("verify"),
        }
    return {**payload, "manifests": manifests}


def assess_outcomes(payload, ctx: RunContext) -> dict:
    """Multidimensional outcome fingerprint per session — status is one axis.
    Besides the counts, tracks HOW a failing test run turned green: if only
    test files were edited between the red run and the green one, the fix
    lived entirely in the tests (``test_only_fix``), the strongest
    transcript-observable bend signal."""
    outcomes = {}
    for label, events in payload["sessions"].items():
        end = next((e for e in events if e["type"] == "session_end"), {})
        usage = end.get("usage", {})
        files, tests_touched, risky = [], [], []
        tests = {"passed": 0, "failed": 0, "skipped": 0}
        red = False  # a failing test run not yet followed by a green one
        red_test_edits = red_source_edits = 0
        test_only_fix = False
        tool_event = 0
        for event in events:
            if event["type"] == "tool_call":
                tool_event += 1
                if event["name"] in EDIT_TOOLS:
                    path = event.get("args", {}).get("file_path", "")
                    files.append(path)
                    if TEST_PATH_RE.search(path):
                        tests_touched.append(path)
                        red_test_edits += 1 if red else 0
                    else:
                        red_source_edits += 1 if red else 0
                elif event["name"] == "Bash":
                    command = event.get("args", {}).get("command", "")
                    for needle, risk in RISKY_PATTERNS:
                        if needle in command:
                            risky.append({"event": tool_event, "command": command,
                                            "risk": risk})
                            break
            elif event["type"] == "tool_result":
                matches = TESTS_RE.findall(event.get("text", ""))
                if matches:  # last test-bearing result wins (re-runs overwrite)
                    tests = {"passed": 0, "failed": 0, "skipped": 0}
                    for passed, failed, skipped in matches:
                        if passed:
                            tests["passed"] = int(passed)
                        if failed:
                            tests["failed"] = int(failed)
                        if skipped:
                            tests["skipped"] = int(skipped)
                    if tests["failed"]:
                        red = True
                        red_test_edits = red_source_edits = 0
                    elif red:
                        if red_test_edits and not red_source_edits:
                            test_only_fix = True
                        red = False
        total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        outcomes[label] = {
            "status": end.get("status", "unknown"),
            "files_touched": sorted(set(files)),
            "test_files_touched": sorted(set(tests_touched)),
            "risky_calls": risky,
            "tests": tests,
            "test_only_fix": test_only_fix,
            "usage": {**usage, "total_tokens": total},
        }
        ctx.metric("llm.input_tokens", usage.get("input_tokens", 0))
        ctx.metric("llm.output_tokens", usage.get("output_tokens", 0))
        ctx.metric("audit.risky_calls", len(risky))
    return {**payload, "outcomes": outcomes}


def extract_traces(payload, ctx: RunContext) -> dict:
    """Normalize tool activity into one comparable trace entry per call."""
    traces, health = {}, {}
    for label, events in payload["sessions"].items():
        traces[label], health[label] = _build_trace(events)
        ctx.metric("audit.tool_calls", len(traces[label]))
        if health[label]["pairing_quality"] == "inferred":
            ctx.metric("audit.trace_inferred", 1)
        if health[label]["trace_completeness"] == "partial":
            ctx.metric("audit.trace_partial", 1)
    return {**payload, "traces": traces, "trace_health": health}


def _build_trace(events: list[dict]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair each tool_result with its call. The join key is ``call_id``;
    tool-name inference is only a fallback. Returns trace entries plus a
    health record separating pairing quality from transcript completeness."""
    trace: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    inferred = False
    duplicate_call_ids = 0
    orphan_results = 0
    for line_no, event in enumerate(events, start=1):
        if event["type"] == "tool_call":
            args = event.get("args", {})
            entry = {
                "event": len(trace) + 1,
                "line": line_no,
                "call_id": event.get("call_id"),
                "tool": event["name"],
                "args": args,
                "args_hash": _hash(args),
                "status": "no_result",
                "result_hash": None,
                "result_head": "",
                "duration_ms": None,
            }
            trace.append(entry)
            call_id = entry["call_id"]
            if call_id is not None:
                if call_id in by_id:
                    duplicate_call_ids += 1
                by_id[call_id] = entry
        elif event["type"] == "tool_result":
            call_id = event.get("call_id")
            call = None
            if call_id is not None:
                call = by_id.get(call_id)
                if call is None:
                    orphan_results += 1
            else:
                call = next((t for t in trace
                             if t["tool"] == event.get("name")
                             and t["status"] == "no_result"), None)
                if call is not None:
                    inferred = True
                else:
                    orphan_results += 1
            if call is not None and call["status"] == "no_result":
                call["status"] = event.get("status", "ok")
                call["result_hash"] = _result_hash(
                    call["tool"], call["args"], event.get("text", ""))
                call["result_head"] = event.get("text", "")[:80]
                call["duration_ms"] = event.get("duration_ms")
    unresolved_calls = sum(1 for entry in trace if entry["status"] == "no_result")
    if duplicate_call_ids:
        pairing_quality = "invalid"
    elif inferred:
        pairing_quality = "inferred"
    else:
        pairing_quality = "exact"
    incomplete = unresolved_calls or orphan_results or duplicate_call_ids
    health = {
        "pairing_quality": pairing_quality,
        "trace_completeness": "partial" if incomplete else "complete",
        "unresolved_calls": unresolved_calls,
        "orphan_results": orphan_results,
        "duplicate_call_ids": duplicate_call_ids,
    }
    return trace, health


def assess_readiness(payload, ctx: RunContext) -> dict:
    """Environment readiness assessment per session (post-hoc, not live gate).

    Uses the recorded manifest plus trace evidence. High-severity config drift
    (prompt, model, MCP, …) is reported separately in ``drift`` — not folded
    into ``readiness_gate`` until a live ``verify_manifest`` layer exists."""
    reference = payload["manifests"]["baseline"]
    readiness = {}
    for label in ("baseline", "candidate"):
        readiness[label] = _readiness_for(
            payload["manifests"][label],
            payload["traces"][label],
            payload["outcomes"][label],
            reference if label == "candidate" else None,
        )
        ctx.metric(f"audit.readiness.blockers.{label}",
                   len(readiness[label]["blockers"]))
        ctx.metric(f"audit.readiness.warnings.{label}",
                   len(readiness[label]["warnings"]))
    gate = _readiness_gate(readiness["candidate"])
    ctx.metric("audit.readiness.gate", 1 if gate == "pass" else 0)
    return {**payload, "readiness": readiness, "readiness_gate": gate}


def _readiness_for(manifest: dict, trace: list, outcome: dict,
                   reference: dict | None) -> dict[str, Any]:
    config = effective_environment(manifest)
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if config.get("repository", {}).get("dirty"):
        blockers.append({
            "id": "repository_dirty",
            "detail": "Working tree is dirty; input source snapshot is not pinned.",
        })

    ref_manifest = reference if reference else None
    if ref_manifest:
        ref_config = effective_environment(ref_manifest)
        repo = config.get("repository", {})
        ref_repo = ref_config.get("repository", {})
        if repo.get("commit") == ref_repo.get("commit"):
            ref_root = ref_config.get("workspace", {}).get("merkle_root")
            root = config.get("workspace", {}).get("merkle_root")
            if ref_root and root and ref_root != root:
                blockers.append({
                    "id": "workspace_snapshot_mismatch",
                    "detail": (f"Same commit ({repo['commit']}) but workspace"
                               f" merkle {root} != reference {ref_root}."),
                })

    denied = [entry for entry in trace if entry["status"] == "denied"]
    for entry in denied:
        blockers.append({
            "id": "tool_denied",
            "event": entry["event"],
            "detail": (f"{entry['tool']} denied at event {entry['event']}:"
                       f" {_compact(entry['result_head'], 60)}"),
        })

    if denied:
        first_denial = min(entry["event"] for entry in denied)
        later_risky = [call for call in outcome.get("risky_calls", [])
                       if call.get("event", 0) > first_denial]
        if later_risky:
            blockers.append({
                "id": "sandbox_workaround_after_denial",
                "detail": ("Sandbox denial co-occurred with a later risky"
                           " workaround."),
            })

    if config.get("network", {}).get("mode") == "restricted":
        for entry in trace:
            if entry["tool"] != "Bash":
                continue
            command = entry.get("args", {}).get("command", "")
            if not re.search(r"\b(?:pip install|curl|wget)\b", command):
                continue
            if entry["status"] in ("denied", "error"):
                warnings.append({
                    "id": "network_restricted",
                    "event": entry["event"],
                    "detail": (f"Restricted network blocked {command!r}"
                               f" at event {entry['event']}."),
                })

    return {"can_start": not blockers, "blockers": blockers, "warnings": warnings}


def _readiness_gate(candidate: dict) -> str:
    if not candidate["can_start"]:
        return "block"
    if candidate["warnings"]:
        return "warn"
    return "pass"


def extract_decisions(semantic=None, include_user: bool = False,
                      decision_tools=None,
                      max_chars: int = DEFAULT_MAX_CHARS,
                      max_sentences: int = DEFAULT_MAX_SENTENCES,
                      max_semantic: int = DEFAULT_MAX_SEMANTIC):
    """Factory for the decisions step (the ``@semantic`` slot arrives here).

    Pipeline: event-type filter → cheap markers on every kept sentence →
    ``@semantic`` only for recall-filter candidates. Budgets are hard:
    ``DecisionBudgetExceeded`` on exhaustion (no silent truncation).

    Event policy (see module docstring): full text for assistant/reasoning/
    plan/commentary; intent-like tool_call args only; tool_result skipped
    unless the tool name is listed in ``decision_tools`` (opt-in for tools
    that return structured decisions); user text only when
    ``include_user=True``; session/system metadata never.
    """
    allowed_result_tools = {
        str(name) for name in (decision_tools or ()) if name
    }

    def step(payload, ctx: RunContext) -> dict:
        budget = DecisionBudget(
            max_chars=max_chars,
            max_sentences=max_sentences,
            max_semantic=max_semantic,
        )
        decisions = {}
        for label, events in payload["sessions"].items():
            found = []
            for line_no, event in enumerate(events, start=1):
                for text, channel in _decision_texts(
                        event, include_user=include_user,
                        decision_tools=allowed_result_tools):
                    found.extend(_classify_message(
                        text, line_no, semantic, channel=channel,
                        budget=budget))
            decisions[label] = found
            ctx.metric("audit.decisions", len(found))
            ctx.metric("audit.decisions.semantic",
                       sum(1 for f in found if f["source"] == "semantic"))
        ctx.metric("audit.decisions.chars", budget.chars)
        ctx.metric("audit.decisions.sentences", budget.sentences)
        ctx.metric("audit.decisions.recall_hits", budget.recall_hits)
        ctx.metric("audit.decisions.semantic_calls", budget.semantic_calls)
        return {**payload, "decisions": decisions}
    return step


class DecisionBudgetExceeded(ValueError):
    """Decision extraction hit a hard cap; refuse silent truncation."""


class DecisionBudget:
    """Mutable counters with hard limits for decision extraction."""

    __slots__ = ("max_chars", "max_sentences", "max_semantic",
                 "chars", "sentences", "recall_hits", "semantic_calls")

    def __init__(self, *, max_chars: int, max_sentences: int,
                 max_semantic: int) -> None:
        self.max_chars = max_chars
        self.max_sentences = max_sentences
        self.max_semantic = max_semantic
        self.chars = 0
        self.sentences = 0
        self.recall_hits = 0
        self.semantic_calls = 0

    def add_chars(self, n: int) -> None:
        self.chars += n
        if self.chars > self.max_chars:
            raise DecisionBudgetExceeded(
                f"decision extraction exceeded max_chars={self.max_chars}"
                f" (saw {self.chars})")

    def add_sentence(self) -> None:
        self.sentences += 1
        if self.sentences > self.max_sentences:
            raise DecisionBudgetExceeded(
                f"decision extraction exceeded max_sentences="
                f"{self.max_sentences} (saw {self.sentences})")

    def add_semantic(self) -> None:
        self.semantic_calls += 1
        if self.semantic_calls > self.max_semantic:
            raise DecisionBudgetExceeded(
                f"decision extraction exceeded max_semantic="
                f"{self.max_semantic} (saw {self.semantic_calls})")


def heuristic_semantics(sentence: str) -> tuple[str, str] | None:
    """Default fill for ``@semantic``: (class, cue) or None per sentence."""
    for cls, pattern in SEMANTIC_CUES:
        match = pattern.search(sentence)
        if match:
            return cls, match.group(0)
    return None


def _decision_texts(event: dict[str, Any], *, include_user: bool,
                    decision_tools: set[str]) -> list[tuple[str, str]]:
    """Select classifiable text slices from one event under the event policy."""
    kind = event.get("type")
    if kind in _DECISION_FULL_TEXT_TYPES:
        text = (event.get("text") or "").strip()
        return [(text, kind)] if text else []
    if kind == "user":
        if not include_user:
            return []
        text = (event.get("text") or "").strip()
        return [(text, "user")] if text else []
    if kind == "tool_call":
        text = _tool_call_intent_text(event.get("args") or {})
        return [(text, "tool_call")] if text else []
    if kind == "tool_result":
        name = event.get("name")
        if not name or str(name) not in decision_tools:
            return []
        text = (event.get("text") or "").strip()
        return [(text, "tool_result")] if text else []
    # session_start / session_end / system / unknown — never
    return []


def _tool_call_intent_text(args: Any) -> str:
    """Keep command/intent fields; drop paths, patches, file bodies, stdin."""
    if not isinstance(args, dict):
        return ""
    parts: list[str] = []
    for key in sorted(args):
        if key not in _TOOL_ARG_INTENT_KEYS:
            continue
        value = args[key]
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        text = text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _classify_message(text: str, line_no: int, semantic,
                      channel: str = "assistant",
                      budget: DecisionBudget | None = None) -> list[dict[str, Any]]:
    """Marker pass on every sentence; semantic only on recall candidates."""
    if budget is not None:
        budget.add_chars(len(text))
    found = []
    for start, end, sentence in _sentences(text):
        if budget is not None:
            budget.add_sentence()
        marker_cls = None
        for cls, pattern in MARKER_RULES:
            match = pattern.search(sentence)
            if match:
                marker_cls = cls
                found.append(_item(line_no, cls, "marker", "high",
                                   match.group(0), (start, end), sentence,
                                   channel=channel))
                break
        if semantic is None:
            continue
        # Stage 2 only for recall-filter candidates — not every sentence.
        if not _SEMANTIC_RECALL_RE.search(sentence):
            continue
        if budget is not None:
            budget.recall_hits += 1
            budget.add_semantic()
        verdict = semantic(sentence)
        if verdict:
            cls, cue = verdict
            if cls != marker_cls:
                found.append(_item(line_no, cls, "semantic", "medium",
                                   cue, (start, end), sentence,
                                   channel=channel))
    return found


def _sentences(text: str):
    """Yield (start, end, stripped sentence); example-grade splitter — no
    abbreviation or inline-path handling."""
    for match in re.finditer(r"[^.!?]+[.!?]*", text):
        raw = match.group(0)
        head = len(raw) - len(raw.lstrip())
        tail = len(raw) - len(raw.rstrip())
        if raw.strip():
            yield match.start() + head, match.end() - tail, raw.strip()


def _item(line: int, cls: str, source: str, confidence: str,
          cue: str, span: tuple[int, int], quote: str,
          channel: str = "assistant") -> dict[str, Any]:
    return {"line": line, "class": cls, "source": source,
            "confidence": confidence, "channel": channel,
            "evidence": {"cue": cue, "span": list(span), "quote": quote}}


def diff_runs(payload, ctx: RunContext) -> dict:
    """Config drift (cause) + trace (path) + outcome (effect) -> verdict."""
    base_m, cand_m = payload["manifests"]["baseline"], payload["manifests"]["candidate"]
    drift: list[dict[str, Any]] = []
    _diff_tree(base_m["config"], cand_m["config"], "", drift)

    manifest_capture_drift: dict[str, list[dict[str, Any]]] = {}
    for label, manifest in (("baseline", base_m), ("candidate", cand_m)):
        if manifest.get("observed"):
            manifest_capture_drift[label] = compute_capture_drift(
                manifest["config"], manifest["observed"])

    trace_div = _compare_traces(payload["traces"]["baseline"],
                                payload["traces"]["candidate"])

    base_o, cand_o = payload["outcomes"]["baseline"], payload["outcomes"]["candidate"]
    divergence = _compare_outcomes(base_o, cand_o)

    has_drift = bool(drift) or any(manifest_capture_drift.values())
    has_divergence = bool(trace_div) or bool(divergence)
    if has_drift and has_divergence:
        verdict = "drift_and_divergence"
    elif has_drift:
        verdict = "config_drift"
    elif has_divergence:
        verdict = "execution_divergence"
    else:
        verdict = "clean"

    ctx.metric("audit.drift", len(drift))
    ctx.metric("audit.trace_divergence", len(trace_div))
    ctx.metric("audit.divergence", len(divergence))
    ctx.metric("audit.capture_drift", sum(len(v) for v in manifest_capture_drift.values()))
    return {**payload, "drift": drift, "manifest_capture_drift": manifest_capture_drift,
            "trace_divergence": trace_div,
            "divergence": divergence, "verdict": verdict}


def _compare_outcomes(base: dict, cand: dict) -> list[dict[str, Any]]:
    """One entry per outcome axis that moved — in EITHER direction. The fact
    of the change is recorded symmetrically; ``assessment`` judges direction
    separately (regression / improvement / neutral). A diff that fires only
    on candidate degradations is a regression detector, not a divergence
    detector: a vanished risky call or a shrunken test suite is still a
    behavior change the audit must surface."""
    out: list[dict[str, Any]] = []

    def entry(dimension: str, assessment: str, severity: str, **extra) -> None:
        out.append({"dimension": dimension, "assessment": assessment,
                    "severity": severity, **extra})

    if base["status"] != cand["status"]:
        entry("status",
              "improvement" if cand["status"] == "ok" else "regression",
              "high", baseline=base["status"], candidate=cand["status"])

    base_files, cand_files = set(base["files_touched"]), set(cand["files_touched"])
    added_files = sorted(cand_files - base_files)
    removed_files = sorted(base_files - cand_files)
    if added_files:  # scope moved; which way is not knowable from paths alone
        entry("files_added", "neutral", "medium", added=added_files,
              baseline=base["files_touched"], candidate=cand["files_touched"])
    if removed_files:
        entry("files_removed", "neutral", "medium", removed=removed_files,
              baseline=base["files_touched"], candidate=cand["files_touched"])

    if base["tests"] != cand["tests"]:
        entry("tests_changed", _assess_tests(base["tests"], cand["tests"]),
              "high" if cand["tests"]["failed"] > base["tests"]["failed"]
              else "medium",
              baseline=base["tests"], candidate=cand["tests"])

    # Test-file edits are a neutral surface change; integrity violations need
    # corroborating signals (tests-only session, skipped/xfailed, suite shrink).
    if set(base["test_files_touched"]) != set(cand["test_files_touched"]):
        entry("test_surface_changed", "neutral", "low",
              baseline=base["test_files_touched"],
              candidate=cand["test_files_touched"])

    integrity_signals = _integrity_violation_signals(base, cand)
    if integrity_signals:
        cand_hits = [s for s in integrity_signals if s.startswith("candidate:")]
        base_hits = [s for s in integrity_signals if s.startswith("baseline:")]
        if cand_hits and not base_hits:
            assessment = "regression"
        elif base_hits and not cand_hits:
            assessment = "improvement"
        else:
            assessment = "neutral"
        entry("test_integrity_violation", assessment, "high",
              signals=integrity_signals,
              baseline=base["test_files_touched"],
              candidate=cand["test_files_touched"])

    base_risky = {(c["risk"], c["command"]) for c in base["risky_calls"]}
    cand_risky = {(c["risk"], c["command"]) for c in cand["risky_calls"]}
    added_risky = [c for c in cand["risky_calls"]
                   if (c["risk"], c["command"]) not in base_risky]
    removed_risky = [c for c in base["risky_calls"]
                     if (c["risk"], c["command"]) not in cand_risky]
    if added_risky:
        entry("risky_calls_added", "regression", "high", added=added_risky,
              baseline=base["risky_calls"], candidate=cand["risky_calls"])
    if removed_risky:
        entry("risky_calls_removed", "improvement", "medium",
              removed=removed_risky,
              baseline=base["risky_calls"], candidate=cand["risky_calls"])

    base_tokens = base["usage"]["total_tokens"]
    cand_tokens = cand["usage"]["total_tokens"]
    if base_tokens != cand_tokens:
        if base_tokens == 0:
            entry("tokens_changed", "regression", "medium",
                  baseline=0, candidate=cand_tokens, ratio=None)
        elif cand_tokens == 0:
            entry("tokens_changed", "improvement", "medium",
                  baseline=base_tokens, candidate=0, ratio=0)
        else:
            ratio = round(cand_tokens / base_tokens, 2)
            if ratio >= TOKEN_BLOWUP or ratio * TOKEN_BLOWUP <= 1:
                entry("tokens_ratio",
                      "regression" if ratio > 1 else "improvement", "medium",
                      baseline=base_tokens, candidate=cand_tokens, ratio=ratio)
    return out


def _integrity_violation_signals(base: dict, cand: dict) -> list[str]:
    """Signals that justify ``test_integrity_violation`` beyond a surface edit.

    Touching tests while green is not enough — new tests for a changed API,
    fixed assertions, and updated fixtures are all legitimate. Flag only when
    transcript-observable evidence points at bending or weakening the suite.
    """
    signals: list[str] = []

    def scan(outcome: dict, prefix: str) -> None:
        test_files = set(outcome["test_files_touched"])
        source_files = set(outcome["files_touched"]) - test_files
        tests = outcome["tests"]
        if test_files and not source_files and tests["failed"] == 0:
            signals.append(f"{prefix}:tests_only_no_source")
        if outcome.get("test_only_fix"):
            signals.append(f"{prefix}:test_only_fix_after_failure")
        skipped = tests.get("skipped", 0)
        if skipped:
            signals.append(f"{prefix}:tests_skipped_or_xfailed")

    scan(base, "baseline")
    scan(cand, "candidate")

    base_suite = base["tests"]["passed"] + base["tests"]["failed"]
    cand_suite = cand["tests"]["passed"] + cand["tests"]["failed"]
    if cand["tests"]["failed"] == 0 and cand_suite < base_suite:
        signals.append("candidate:suite_shrunk_while_green")
    if cand["tests"].get("skipped", 0) > base["tests"].get("skipped", 0):
        signals.append("candidate:skipped_increased")

    # A red-to-green path that touched only tests is suspicious only when the
    # whole session also avoided source — otherwise it may be API/test sync.
    signals = [s for s in signals
               if s != "candidate:test_only_fix_after_failure"
               or "candidate:tests_only_no_source" in signals]
    return sorted(set(signals))


def _assess_tests(base: dict, cand: dict) -> str:
    """Direction of a test-count change: more failures or a shrunken suite
    (tests vanished while the run stayed green) is a regression; fewer
    failures with the suite intact, or a grown suite, is an improvement."""
    if cand["failed"] > base["failed"]:
        return "regression"
    if cand["passed"] + cand["failed"] < base["passed"] + base["failed"]:
        return "regression"
    return "improvement"


def _compare_traces(base: list, cand: list) -> list[dict[str, Any]]:
    """Diff the tool-call sequences. LCS alignment on (tool, args-hash)
    splits the pair into matched calls and per-side leftovers; a leftover
    signature present on *both* sides is a reorder, not an add + remove.
    The earliest aligned position that is not a clean match becomes the
    headline ``first_divergence`` entry (it repeats one fact from the
    inventory below it on purpose — navigation vs. completeness)."""
    ops = _pair_arg_changes(_align(base, cand))
    moved = (Counter(_sig(b) for op, b, _ in ops if op == "base_only")
             & Counter(_sig(c) for op, _, c in ops if op == "cand_only"))

    out: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    missing, added, reordered = [], [], []
    budget_b, budget_c = Counter(moved), Counter(moved)
    for op, b, c in ops:
        kind = None
        if op == "match":
            if b["status"] != c["status"] or b["result_hash"] != c["result_hash"]:
                kind = "result_changed"
        elif op == "args_changed":
            kind = "args_changed"
        elif op == "base_only":
            if budget_b[_sig(b)]:
                budget_b[_sig(b)] -= 1
                reordered.append(b)
                kind = "reordered"
            else:
                missing.append(b)
                kind = "calls_missing"
        else:  # cand_only
            if budget_c[_sig(c)]:
                budget_c[_sig(c)] -= 1
                kind = "reordered"
            else:
                added.append(c)
                kind = "calls_added"
        if kind in ("result_changed", "args_changed"):
            out.append({"kind": kind, "severity": TRACE_SEVERITY[kind],
                        "baseline": _call_view(b), "candidate": _call_view(c)})
        if kind and first is None:
            first = {"kind": "first_divergence",
                     "severity": TRACE_SEVERITY["first_divergence"],
                     "reason": kind, "event": (b or c)["event"],
                     "baseline": _call_view(b) if b else None,
                     "candidate": _call_view(c) if c else None}

    if missing:
        out.append({"kind": "calls_missing", "severity": TRACE_SEVERITY["calls_missing"],
                    "side": "baseline", "calls": [_call_view(e) for e in missing]})
    if added:
        out.append({"kind": "calls_added", "severity": TRACE_SEVERITY["calls_added"],
                    "side": "candidate", "calls": [_call_view(e) for e in added]})
    if reordered:
        out.append({"kind": "reordered", "severity": TRACE_SEVERITY["reordered"],
                    "calls": [_call_view(e) for e in reordered]})
    if first:
        out.insert(0, first)
    return out


def _align(base: list, cand: list) -> list[tuple]:
    """LCS over call signatures -> ordered ops: ("match", b, c),
    ("base_only", b, None), ("cand_only", None, c). The forward walk over a
    suffix table matches the *earliest* possible occurrence on each side, so
    a re-run of an already-matched call surfaces as added, not matched."""
    m, n = len(base), len(cand)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if _sig(base[i]) == _sig(cand[j]):
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    ops: list[tuple] = []
    i = j = 0
    while i < m and j < n:
        if _sig(base[i]) == _sig(cand[j]) and dp[i][j] == dp[i + 1][j + 1] + 1:
            ops.append(("match", base[i], cand[j]))
            i, j = i + 1, j + 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            ops.append(("base_only", base[i], None))
            i += 1
        else:
            ops.append(("cand_only", None, cand[j]))
            j += 1
    ops.extend(("base_only", e, None) for e in base[i:])
    ops.extend(("cand_only", None, e) for e in cand[j:])
    return ops


def _pair_arg_changes(ops: list) -> list:
    """Within one contiguous non-match run, the k-th baseline-only and k-th
    candidate-only call with the same tool read as a single call whose
    arguments changed (the unified-diff "changed line" heuristic); pairing
    stops at the first tool mismatch."""
    out: list[tuple] = []
    run: list[tuple] = []

    def flush() -> None:
        dels = [b for op, b, _ in run if op == "base_only"]
        inss = [c for op, _, c in run if op == "cand_only"]
        paired = 0
        for b, c in zip(dels, inss):
            if b["tool"] != c["tool"]:
                break
            out.append(("args_changed", b, c))
            paired += 1
        out.extend(("base_only", b, None) for b in dels[paired:])
        out.extend(("cand_only", None, c) for c in inss[paired:])
        run.clear()

    for op in ops:
        if op[0] == "match":
            flush()
            out.append(op)
        else:
            run.append(op)
    flush()
    return out


def _sig(entry: dict) -> tuple[str, str]:
    return entry["tool"], entry["args_hash"]


def pytest_result_normalizer(text: str) -> str:
    return " ".join(PYTEST_ELAPSED_RE.sub("", text).split())


def filesystem_result_normalizer(text: str) -> str:
    return text.strip()


def http_result_normalizer(text: str) -> str:
    return " ".join(HTTP_TIMING_RE.sub("", text).split())


def git_result_normalizer(text: str) -> str:
    return text.strip()


def _result_normalizer(tool: str, args: dict):
    if tool == "Bash":
        command = args.get("command", "")
        if re.search(r"\bpytest\b|\bpy\.test\b", command):
            return pytest_result_normalizer
        if re.search(r"(^|\s|/)git\s", command):
            return git_result_normalizer
        if re.search(r"\bcurl\b|\bwget\b", command):
            return http_result_normalizer
    elif tool in FILESYSTEM_TOOLS:
        return filesystem_result_normalizer
    return None


def _result_hash(tool: str, args: dict, text: str) -> str:
    normalize = _result_normalizer(tool, args)
    body = normalize(text) if normalize else text.strip()
    return _hash(body)


def _hash(value: Any) -> str:
    canon = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def _call_view(entry: dict) -> dict[str, Any]:
    return {"event": entry["event"], "line": entry["line"],
            "call": _call_repr(entry), "status": entry["status"],
            "result": entry["result_head"], "duration_ms": entry["duration_ms"]}


def _call_repr(entry: dict) -> str:
    args = ", ".join(str(v) for v in entry["args"].values())
    return f"{entry['tool']}({_compact(args, 60)})"


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
    manifests, outcomes = payload["manifests"], payload["outcomes"]
    lines = [
        f"# Agent run audit: {manifests['baseline']['session_id']}"
        f" vs {manifests['candidate']['session_id']}",
        "",
        f"Readiness gate: {payload['readiness_gate']}"
        f" (post-hoc assessment of the candidate environment; not live preflight)",
        f"Verdict: {payload['verdict']}",
        "",
        "## Environment readiness (assessment)",
    ]
    for label in ("baseline", "candidate"):
        ready = payload["readiness"][label]
        status = "yes" if ready["can_start"] else "no"
        lines.append(f"- {label}: can_start={status}")
        for item in ready["blockers"]:
            lines.append(f"    [blocker] {item['id']}: {item['detail']}")
        for item in ready["warnings"]:
            lines.append(f"    [warning] {item['id']}: {item['detail']}")
        if not ready["blockers"] and not ready["warnings"]:
            lines.append("    (no blockers or warnings)")
    lines.extend(["", "## Config drift (cause)"])
    for item in payload["drift"] or []:
        lines.append(f"- [{item['severity']}] {item['field']}:"
                     f" {_compact(item['baseline'])} -> {_compact(item['candidate'])}")
    if not payload["drift"]:
        lines.append("- none")

    lines.extend(["", "## Declared vs captured (session start)"])
    capture = payload.get("manifest_capture_drift") or {}
    if not any(capture.values()):
        lines.append("- none (no observed snapshots recorded)")
    else:
        for label in ("baseline", "candidate"):
            items = capture.get(label) or []
            if not items:
                continue
            lines.append(f"- {label}:")
            for item in items:
                lines.append(
                    f"    [{_severity(item['field'])}] {item['field']}:"
                    f" declared {_compact(item['expected'])}"
                    f" -> captured {_compact(item['observed'])}")

    lines.extend(["", "## Trace divergence (behavior)"])
    for label in ("baseline", "candidate"):
        h = payload["trace_health"][label]
        lines.append(
            f"Trace health ({label}): pairing={h['pairing_quality']},"
            f" completeness={h['trace_completeness']}"
            f" (unresolved={h['unresolved_calls']},"
            f" orphan_results={h['orphan_results']},"
            f" duplicate_call_ids={h['duplicate_call_ids']})")
    caveat = ("" if all(h["pairing_quality"] == "exact"
                        for h in payload["trace_health"].values())
              else " — inferred pairing is best-effort; parallel same-tool"
                   " calls can mis-pair results")
    if caveat:
        lines.append(caveat.strip())
    for item in payload["trace_divergence"] or []:
        lines.extend(_render_trace_item(item))
    if not payload["trace_divergence"]:
        lines.append("- none — same tool calls, same order, same results")

    lines.extend(["", "## Outcome divergence (effect)"])
    for item in payload["divergence"] or []:
        lines.append(f"- [{item['severity']}] {item['dimension']}"
                     f" ({item['assessment']}): {_render_divergence(item)}")
    if not payload["divergence"]:
        lines.append("- none")

    lines.extend(["", "## Decisions"])
    for label in ("baseline", "candidate"):
        for decision in payload["decisions"][label]:
            lines.append(
                f"- {label} L{decision['line']}"
                f" [{decision['class']}/{decision['source']}"
                f" {decision['confidence']}"
                f" via {decision.get('channel', 'assistant')}]:"
                f" \"{decision['evidence']['quote']}\"")

    lines.extend(["", "## Outcome summary"])
    for label in ("baseline", "candidate"):
        outcome = outcomes[label]
        lines.append(f"- {label}: {outcome['status']}"
                     f" | tests {outcome['tests']['passed']}p/{outcome['tests']['failed']}f"
                     f" | {outcome['usage']['total_tokens']} tokens"
                     f" | {len(outcome['files_touched'])} file(s)")
    return {
        "verdict": payload["verdict"],
        "readiness_gate": payload["readiness_gate"],
        "readiness": payload["readiness"],
        "drift": payload["drift"],
        "manifest_capture_drift": payload.get("manifest_capture_drift", {}),
        "trace_health": payload["trace_health"],
        "trace_divergence": payload["trace_divergence"],
        "divergence": payload["divergence"],
        "outcomes": outcomes,
        "decisions": payload["decisions"],
        "report": "\n".join(lines),
    }


def _render_trace_item(item: dict[str, Any]) -> list[str]:
    kind, sev = item["kind"], item["severity"]
    if kind == "first_divergence":
        return [f"- [{sev}] first behavioral divergence at event {item['event']}"
                f" ({item['reason']}):",
                f"    baseline:  {_side_line(item['baseline'])}",
                f"    candidate: {_side_line(item['candidate'])}"]
    if kind == "result_changed":
        b, c = item["baseline"], item["candidate"]
        return [f"- [{sev}] result_changed at event {_event_ref(b, c)}: {b['call']}:"
                f" {b['status']} \"{_compact(b['result'], 40)}\" ->"
                f" {c['status']} \"{_compact(c['result'], 40)}\""]
    if kind == "args_changed":
        b, c = item["baseline"], item["candidate"]
        return [f"- [{sev}] args_changed at event {_event_ref(b, c)}:"
                f" {b['call']} -> {c['call']}"]
    if kind == "denied":
        call = item["call"]
        return [f"- [{sev}] denied on {item['side']} at event {call['event']}:"
                f" {call['call']}"]
    if kind in ("calls_missing", "calls_added"):
        where = "baseline only" if kind == "calls_missing" else "candidate only"
        calls = "; ".join(v["call"] for v in item["calls"])
        return [f"- [{sev}] {kind} ({where}): {calls}"]
    if kind == "reordered":
        calls = "; ".join(v["call"] for v in item["calls"])
        return [f"- [{sev}] reordered (same calls, different order): {calls}"]
    return [f"- [{sev}] {kind}"]


def _side_line(view: dict[str, Any] | None) -> str:
    if view is None:
        return "(no call at this point)"
    took = f" in {view['duration_ms']}ms" if view["duration_ms"] is not None else ""
    result = f": \"{_compact(view['result'], 60)}\"" if view["result"] else ""
    return f"L{view['line']} {view['call']} -> {view['status']}{took}{result}"


def _event_ref(base: dict[str, Any], cand: dict[str, Any]) -> str:
    if base["event"] == cand["event"]:
        return str(base["event"])
    return f"{base['event']}/{cand['event']}"


def _render_divergence(item: dict[str, Any]) -> str:
    dim = item["dimension"]
    if dim == "status":
        return f"{item['baseline']} -> {item['candidate']}"
    if dim == "files_added":
        return f"candidate also touched {item['added']}"
    if dim == "files_removed":
        return f"candidate never touched {item['removed']}"
    if dim == "tests_changed":
        base, cand = item["baseline"], item["candidate"]
        return (f"{base['passed']}p/{base['failed']}f ->"
                f" {cand['passed']}p/{cand['failed']}f")
    if dim == "test_surface_changed":
        return (f"{item['baseline'] or '(none)'} ->"
                f" {item['candidate'] or '(none)'}")
    if dim == "test_integrity_violation":
        return (f"{', '.join(item['signals'])}:"
                f" {item['candidate'] or item['baseline']}")
    if dim == "risky_calls_added":
        return "; ".join(f"{c['risk']}: {c['command']}" for c in item["added"])
    if dim == "risky_calls_removed":
        return "; ".join(f"{c['risk']}: {c['command']}" for c in item["removed"])
    if dim == "tokens_ratio":
        return f"{item['baseline']} -> {item['candidate']} ({item['ratio']}x)"
    if dim == "tokens_changed":
        ratio = item.get("ratio")
        suffix = f" ({ratio}x)" if ratio is not None else ""
        return f"{item['baseline']} -> {item['candidate']}{suffix}"
    return _compact(item.get("candidate"))


def _compact(value: Any, limit: int = 48) -> str:
    if value is None:
        return "(absent)"
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) \
        else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def redact_secrets(checkpoint: str, value: Any, ctx: RunContext):
    """kind="policy": recursive egress redaction over the whole public
    output — example-grade ``SECRET_RE`` only; swap for ``@secret_detector``
    in production (GitHub tokens, AWS keys, Bearer/JWT, etc.)."""
    redacted, count = _redact_value(value)
    if count:
        return Transform(redacted, f"redacted {count} secret(s)")
    return None


def lineage_report(payload: Any) -> Any:
    """Lineage extractor with scrub-at-capture (fills ``@lineage_extract``):
    the ledger snapshots each step's output at step end — before run-end
    policy egress can transform anything — so the audit trail must redact
    what it records; middleware ordering alone cannot keep it clean."""
    artifact = payload.get("report", payload) if isinstance(payload, dict) \
        else payload
    return _redact_value(artifact)[0]


def _redact_value(value: Any) -> tuple[Any, int]:
    """Return (redacted copy, substitution count) over str/dict/list."""
    if isinstance(value, str):
        return SECRET_RE.subn("[secret redacted]", value)
    if isinstance(value, dict):
        pairs = {key: _redact_value(item) for key, item in value.items()}
        return ({key: item for key, (item, _) in pairs.items()},
                sum(count for _, count in pairs.values()))
    if isinstance(value, list):
        items = [_redact_value(item) for item in value]
        return [item for item, _ in items], sum(count for _, count in items)
    return value, 0
