"""Example components for the `agent-audit` preset.

"Works on my machine" for AI agents: the same task succeeded yesterday and
"succeeded" today — both runs green — yet today's is green for the wrong
reasons. This preset audits two recorded agent sessions and answers with
data instead of vibes:

    normalize -> load sessions -> build manifests -> assess outcomes
              -> extract traces -> extract decisions -> diff runs -> render report

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
  ids falls back to tool-name inference and the whole trace is marked
  ``trace_quality="inferred"`` — with parallel same-tool calls in flight
  (the manifests literally declare ``parallel-tools``), name inference can
  attach both results to the wrong calls, so only an id join earns
  ``"exact"``. Two runs can share a manifest and both end green yet get
  there along different paths; the trace diff aligns both sequences on
  (tool, arguments), names the **first behavioral divergence**, then
  classifies the rest of the gap: changed arguments, changed result,
  permission denials, missing / added / reordered calls. An empty trace
  diff is itself a finding: identical behavior clears the run even when the
  manifests drifted. The **path** side.
- **outcome fingerprint**: what the run actually *did*, across dimensions —
  status is only one. Also: which files it touched (and whether any were
  test files, i.e. tests bent to pass), risky tool calls, parsed test
  results, and token spend. A run can be ``status=ok`` and still diverge on
  every other dimension. The **effect** side.

Decisions ride along as a fourth, softer channel: assistant sentences
classed as **plan / decision / assumption / action**. Stage 1 is a
deterministic marker table (auditable, replayable); stage 2 is the optional
``@semantic`` preset slot — filled here with a cheap cue heuristic so the
example runs offline, pointed at an LLM extractor in a real deployment —
which may only add what the markers missed, never override them. Every item
keeps its evidence: the cue that fired, the sentence quote, and the quote's
char span at its transcript line.

`diff_runs` compares all three and renders a verdict over the pair. The session
fixtures are a neutral JSONL shape (one event per line) modeled on what
coding-agent harnesses record on disk; a real deployment would point the
same flow at exported Claude Code / Cursor / Codex transcripts. Secrets
never belong in a manifest (env vars are name -> value-hash), and policy
egress recursively redacts anything an agent leaks — in the report string
and in every structured field of the public output alike. The lineage
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
SECRET_RE = re.compile(r"\b(?:sk|key|token)-[A-Za-z0-9-]{8,}\b")
TEST_PATH_RE = re.compile(r"(^|/)tests?/|(^|/)test_|_test\.")
TESTS_RE = re.compile(r"(\d+)\s+passed|(\d+)\s+failed")
EDIT_TOOLS = {"Edit", "Write", "Create"}
TOKEN_BLOWUP = 1.5  # candidate/baseline total-token ratio that trips a flag

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
    "execution": "medium",
    "workspace": "low",             # intended code changes live here
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
    """Distill each session into its run manifest (effective config)."""
    manifests = {}
    for label, events in payload["sessions"].items():
        start = next(e for e in events if e["type"] == "session_start")
        manifests[label] = {
            "session_id": start["session_id"],
            "config": start["config"],
        }
    return {**payload, "manifests": manifests}


def assess_outcomes(payload, ctx: RunContext) -> dict:
    """Multidimensional outcome fingerprint per session — status is one axis."""
    outcomes = {}
    for label, events in payload["sessions"].items():
        end = next((e for e in events if e["type"] == "session_end"), {})
        usage = end.get("usage", {})
        files, tests_touched, risky = [], [], []
        tests = {"passed": 0, "failed": 0}
        for event in events:
            if event["type"] == "tool_call" and event["name"] in EDIT_TOOLS:
                path = event.get("args", {}).get("file_path", "")
                files.append(path)
                if TEST_PATH_RE.search(path):
                    tests_touched.append(path)
            elif event["type"] == "tool_call" and event["name"] == "Bash":
                command = event.get("args", {}).get("command", "")
                for needle, risk in RISKY_PATTERNS:
                    if needle in command:
                        risky.append({"command": command, "risk": risk})
                        break
            elif event["type"] == "tool_result":
                matches = TESTS_RE.findall(event.get("text", ""))
                if matches:  # last test-bearing result wins (re-runs overwrite)
                    tests = {"passed": 0, "failed": 0}
                    for passed, failed in matches:
                        if passed:
                            tests["passed"] = int(passed)
                        if failed:
                            tests["failed"] = int(failed)
        total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        outcomes[label] = {
            "status": end.get("status", "unknown"),
            "files_touched": sorted(set(files)),
            "test_files_touched": sorted(set(tests_touched)),
            "risky_calls": risky,
            "tests": tests,
            "usage": {**usage, "total_tokens": total},
        }
        ctx.metric("llm.input_tokens", usage.get("input_tokens", 0))
        ctx.metric("llm.output_tokens", usage.get("output_tokens", 0))
        ctx.metric("audit.risky_calls", len(risky))
    return {**payload, "outcomes": outcomes}


def extract_traces(payload, ctx: RunContext) -> dict:
    """Normalize tool activity into one comparable trace entry per call."""
    traces, quality = {}, {}
    for label, events in payload["sessions"].items():
        traces[label], quality[label] = _build_trace(events)
        ctx.metric("audit.tool_calls", len(traces[label]))
        if quality[label] != "exact":
            ctx.metric("audit.trace_inferred", 1)
    return {**payload, "traces": traces, "trace_quality": quality}


def _build_trace(events: list[dict]) -> tuple[list[dict[str, Any]], str]:
    """Pair each tool_result with its call. The join key is ``call_id``;
    tool-name inference (first same-tool call still waiting — batched
    parallel calls conventionally stream results in call order) is only a
    fallback, and one inferred attachment downgrades the whole trace to
    trace_quality="inferred": with parallel same-tool calls in flight,
    name inference can attach both results to the wrong calls and nothing
    in the transcript would show it. A call accepts at most one result;
    a result whose call_id matches nothing is dropped, not guessed."""
    trace: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    inferred = False
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
            if entry["call_id"] is not None:
                by_id[entry["call_id"]] = entry
        elif event["type"] == "tool_result":
            call_id = event.get("call_id")
            if call_id is not None:
                call = by_id.get(call_id)
            else:
                call = next((t for t in trace
                             if t["tool"] == event.get("name")
                             and t["status"] == "no_result"), None)
                inferred = inferred or call is not None
            if call is not None and call["status"] == "no_result":
                call["status"] = event.get("status", "ok")
                call["result_hash"] = _hash(event.get("text", ""))
                # longer than any render width, so truncation shows "…"
                call["result_head"] = event.get("text", "")[:80]
                call["duration_ms"] = event.get("duration_ms")
    return trace, ("inferred" if inferred else "exact")


def extract_decisions(semantic=None):
    """Factory for the decisions step (the ``@semantic`` slot arrives here).

    Stage 1 tags sentences with deterministic markers; stage 2 hands only
    the sentences stage 1 left untagged to the optional ``semantic``
    extractor — it can add, never override, so the auditable core stays
    replayable. Every item keeps its evidence: the cue that fired, the
    sentence quote, and the quote's char span at its transcript line.
    """
    def step(payload, ctx: RunContext) -> dict:
        decisions = {}
        for label, events in payload["sessions"].items():
            found = []
            for line_no, event in enumerate(events, start=1):
                if event["type"] == "assistant":
                    found.extend(
                        _classify_message(event["text"], line_no, semantic))
            decisions[label] = found
            ctx.metric("audit.decisions", len(found))
            ctx.metric("audit.decisions.semantic",
                       sum(1 for f in found if f["source"] == "semantic"))
        return {**payload, "decisions": decisions}
    return step


def heuristic_semantics(sentence: str) -> tuple[str, str] | None:
    """Default fill for ``@semantic``: (class, cue) or None per sentence."""
    for cls, pattern in SEMANTIC_CUES:
        match = pattern.search(sentence)
        if match:
            return cls, match.group(0)
    return None


def _classify_message(text: str, line_no: int, semantic) -> list[dict[str, Any]]:
    found = []
    for start, end, sentence in _sentences(text):
        item = None
        for cls, pattern in MARKER_RULES:
            match = pattern.search(sentence)
            if match:
                item = _item(line_no, cls, "marker", "high",
                             match.group(0), (start, end), sentence)
                break
        if item is None and semantic is not None:
            verdict = semantic(sentence)
            if verdict:
                cls, cue = verdict
                item = _item(line_no, cls, "semantic", "medium",
                             cue, (start, end), sentence)
        if item:
            found.append(item)
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
          cue: str, span: tuple[int, int], quote: str) -> dict[str, Any]:
    return {"line": line, "class": cls, "source": source,
            "confidence": confidence,
            "evidence": {"cue": cue, "span": list(span), "quote": quote}}


def diff_runs(payload, ctx: RunContext) -> dict:
    """Config drift (cause) + trace (path) + outcome (effect) -> verdict."""
    base_m, cand_m = payload["manifests"]["baseline"], payload["manifests"]["candidate"]
    drift: list[dict[str, Any]] = []
    _diff_tree(base_m["config"], cand_m["config"], "", drift)

    trace_div = _compare_traces(payload["traces"]["baseline"],
                                payload["traces"]["candidate"])

    base_o, cand_o = payload["outcomes"]["baseline"], payload["outcomes"]["candidate"]
    divergence = _compare_outcomes(base_o, cand_o)

    has_drift = bool(drift)
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
    return {**payload, "drift": drift, "trace_divergence": trace_div,
            "divergence": divergence, "verdict": verdict}


def _compare_outcomes(base: dict, cand: dict) -> list[dict[str, Any]]:
    """One entry per outcome axis that moved; each carries its own severity."""
    out: list[dict[str, Any]] = []
    if base["status"] != cand["status"]:
        out.append({"dimension": "status", "baseline": base["status"],
                    "candidate": cand["status"], "severity": "high"})

    new_files = sorted(set(cand["files_touched"]) - set(base["files_touched"]))
    if new_files:
        out.append({"dimension": "files_touched", "baseline": base["files_touched"],
                    "candidate": cand["files_touched"], "added": new_files,
                    "severity": "medium"})

    # Tests green *and* a test file was edited: the classic pass-by-shortcut.
    if cand["test_files_touched"] and cand["tests"]["failed"] == 0:
        out.append({"dimension": "test_integrity", "baseline": base["test_files_touched"],
                    "candidate": cand["test_files_touched"],
                    "note": "tests pass after editing test files", "severity": "high"})

    if cand["risky_calls"]:
        out.append({"dimension": "risky_calls", "baseline": base["risky_calls"],
                    "candidate": cand["risky_calls"], "severity": "high"})

    base_total = base["usage"]["total_tokens"] or 1
    ratio = round(cand["usage"]["total_tokens"] / base_total, 2)
    if ratio >= TOKEN_BLOWUP:
        out.append({"dimension": "tokens", "baseline": base["usage"]["total_tokens"],
                    "candidate": cand["usage"]["total_tokens"], "ratio": ratio,
                    "severity": "medium"})
    return out


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

    for side, trace in (("baseline", base), ("candidate", cand)):
        for entry in trace:
            if entry["status"] == "denied":
                out.append({"kind": "denied", "severity": TRACE_SEVERITY["denied"],
                            "side": side, "call": _call_view(entry)})
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
        f"Verdict: {payload['verdict']}",
        "",
        "## Config drift (cause)",
    ]
    for item in payload["drift"] or []:
        lines.append(f"- [{item['severity']}] {item['field']}:"
                     f" {_compact(item['baseline'])} -> {_compact(item['candidate'])}")
    if not payload["drift"]:
        lines.append("- none")

    lines.extend(["", "## Trace divergence (behavior)"])
    quality = payload["trace_quality"]
    caveat = ("" if all(q == "exact" for q in quality.values())
              else " — inferred pairing is best-effort; parallel same-tool"
                   " calls can mis-pair results")
    lines.append(f"Trace quality: baseline={quality['baseline']},"
                 f" candidate={quality['candidate']}{caveat}")
    for item in payload["trace_divergence"] or []:
        lines.extend(_render_trace_item(item))
    if not payload["trace_divergence"]:
        lines.append("- none — same tool calls, same order, same results")

    lines.extend(["", "## Outcome divergence (effect)"])
    for item in payload["divergence"] or []:
        lines.append(f"- [{item['severity']}] {item['dimension']}:"
                     f" {_render_divergence(item)}")
    if not payload["divergence"]:
        lines.append("- none")

    lines.extend(["", "## Decisions"])
    for label in ("baseline", "candidate"):
        for decision in payload["decisions"][label]:
            lines.append(
                f"- {label} L{decision['line']}"
                f" [{decision['class']}/{decision['source']}"
                f" {decision['confidence']}]:"
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
        "drift": payload["drift"],
        "trace_quality": payload["trace_quality"],
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
    if dim == "files_touched":
        return f"added {item['added']}"
    if dim == "test_integrity":
        return f"{item['note']}: {item['candidate']}"
    if dim == "risky_calls":
        return "; ".join(f"{c['risk']}: {c['command']}" for c in item["candidate"])
    if dim == "tokens":
        return f"{item['baseline']} -> {item['candidate']} ({item['ratio']}x)"
    return _compact(item.get("candidate"))


def _compact(value: Any, limit: int = 48) -> str:
    if value is None:
        return "(absent)"
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) \
        else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def redact_secrets(checkpoint: str, value: Any, ctx: RunContext):
    """kind="policy": recursive egress redaction over the whole public
    output — a secret rides identically in the report string and in the
    structured fields (risky commands, trace call views), so scrubbing
    only the report leaves the JSON output leaking."""
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
