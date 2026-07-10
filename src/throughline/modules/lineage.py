"""Line-level lineage: `git blame` for pipeline output.

After every step the ledger diffs the textual form of the payload against the
previous version (difflib at line granularity):

  - identical lines  -> carry: the line keeps its record (and its origin);
  - similar lines    -> modify: new record, parent = the closest old line;
  - new lines        -> generate: new record attributed to the current step;
  - vanished lines   -> drop: recorded per step, removed from heads.

So for any line of the final output you can answer: which step wrote it,
which line(s) it descends from, and what the full ancestry chain is.

Payload -> lines: str splits on newlines; list -> one line per item;
dict -> one line per key. For structured payloads pass ``extract`` to point
the ledger at the textual artifact, e.g. LineageMiddleware(extract=lambda p:
p["answer"]).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

from ..context import RunContext
from ..middleware import Middleware
from ..step import Step

SIMILARITY_THRESHOLD = 0.5  # git's rename-detection ballpark; tune per ledger


def lines_of(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        return payload.splitlines()
    if isinstance(payload, dict):
        return [f"{key}: {_scalar(value)}" for key, value in payload.items()]
    if isinstance(payload, (list, tuple)):
        return [_scalar(item) for item in payload]
    return [_scalar(payload)]


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\n", "\\n")  # container items stay one line each
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


@dataclass
class LineRecord:
    id: str
    step: str                 # step that wrote this exact text ("input" = source)
    text: str
    parents: list[str] = field(default_factory=list)
    op: str = "generate"      # source | carry-origin recorded at creation | modify | generate
    line_no: int = 0          # position at creation time

    def to_dict(self) -> dict:
        return {"id": self.id, "step": self.step, "op": self.op,
                "line_no": self.line_no, "parents": self.parents, "text": self.text}


class LineageLedger:
    def __init__(self, run_id: str = "run", similarity: float = SIMILARITY_THRESHOLD):
        self.run_id = run_id
        self.similarity = similarity
        self.records: dict[str, LineRecord] = {}
        self.heads: list[str] = []       # record ids of the current artifact, in order
        self.steps: list[str] = []       # evolution history
        self.drops: dict[str, list[str]] = {}  # step -> dropped record ids
        self._counter = 0

    # -- building -------------------------------------------------------------
    def _new_record(self, step: str, text: str, parents: list[str],
                    op: str, line_no: int) -> LineRecord:
        self._counter += 1
        record = LineRecord(id=f"L{self._counter}", step=step, text=text,
                            parents=parents, op=op, line_no=line_no)
        self.records[record.id] = record
        return record

    def snapshot_source(self, payload: Any, label: str = "input") -> None:
        self.steps.append(label)
        self.heads = [
            self._new_record(label, text, [], "source", index).id
            for index, text in enumerate(lines_of(payload))
        ]

    def current_lines(self) -> list[str]:
        """Texts of the current artifact, in order (the heads)."""
        return [self.records[record_id].text for record_id in self.heads]

    def evolve(self, step_name: str, payload: Any) -> dict:
        """Diff the new payload against current heads; re-attribute lines."""
        self.steps.append(step_name)
        old_ids = self.heads
        old_lines = self.current_lines()
        new_lines = lines_of(payload)
        new_heads: list[str | None] = [None] * len(new_lines)
        stats = {"carry": 0, "modify": 0, "generate": 0, "drop": 0}
        dropped: list[str] = []

        matcher = SequenceMatcher(None, old_lines, new_lines, autojunk=False)
        for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
            if tag == "equal":
                for offset in range(old_end - old_start):
                    new_heads[new_start + offset] = old_ids[old_start + offset]
                    stats["carry"] += 1
            elif tag == "delete":
                dropped.extend(old_ids[old_start:old_end])
            elif tag == "insert":
                for index in range(new_start, new_end):
                    new_heads[index] = self._new_record(
                        step_name, new_lines[index], [], "generate", index).id
                    stats["generate"] += 1
            elif tag == "replace":
                block_old = list(range(old_start, old_end))
                for index in range(new_start, new_end):
                    parent_pos = self._best_match(new_lines[index], old_lines, block_old,
                                                  self.similarity)
                    if parent_pos is None:
                        record = self._new_record(step_name, new_lines[index], [], "generate", index)
                        stats["generate"] += 1
                    else:
                        block_old.remove(parent_pos)
                        record = self._new_record(step_name, new_lines[index],
                                                  [old_ids[parent_pos]], "modify", index)
                        stats["modify"] += 1
                    new_heads[index] = record.id
                dropped.extend(old_ids[pos] for pos in block_old)

        stats["drop"] = len(dropped)
        if dropped:
            self.drops[step_name] = self.drops.get(step_name, []) + dropped
        self.heads = [head for head in new_heads if head is not None]
        return stats

    @staticmethod
    def _best_match(text: str, old_lines: list[str], candidates: list[int],
                    threshold: float = SIMILARITY_THRESHOLD) -> int | None:
        best_pos, best_ratio = None, threshold
        for position in candidates:
            ratio = SequenceMatcher(None, old_lines[position], text, autojunk=False).ratio()
            if ratio > best_ratio:
                best_pos, best_ratio = position, ratio
        return best_pos

    # -- querying ---------------------------------------------------------------
    def trace(self, record_id: str) -> list[LineRecord]:
        """Ancestry chain from a record back to its root (first parent path)."""
        chain: list[LineRecord] = []
        current: str | None = record_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            record = self.records[current]
            chain.append(record)
            current = record.parents[0] if record.parents else None
        return chain

    def origin_of(self, record_id: str) -> LineRecord:
        return self.trace(record_id)[-1]

    def blame(self) -> list[dict]:
        """One entry per line of the current artifact — like `git blame`."""
        entries = []
        for line_no, record_id in enumerate(self.heads):
            record = self.records[record_id]
            origin = self.origin_of(record_id)
            entries.append({
                "line_no": line_no,
                "text": record.text,
                "step": record.step,          # last writer
                "op": record.op,
                "origin": origin.step,        # who introduced the ancestral line
                "depth": len(self.trace(record_id)),
                "id": record.id,
            })
        return entries

    def render_blame(self, max_width: int = 100) -> str:
        entries = self.blame()
        if not entries:
            return "(empty artifact)"
        step_width = max(len(e["step"]) for e in entries)
        lines = []
        for entry in entries:
            marker = {"source": "=", "modify": "~", "generate": "+"}.get(entry["op"], " ")
            text = entry["text"]
            if len(text) > max_width:
                text = text[: max_width - 1] + "…"
            lines.append(f"{entry['step']:>{step_width}} {marker}{entry['line_no'] + 1:>4}│ {text}")
        return "\n".join(lines)

    def stats(self) -> dict:
        by_step: dict[str, int] = {}
        for record_id in self.heads:
            step = self.records[record_id].step
            by_step[step] = by_step.get(step, 0) + 1
        return {
            "lines": len(self.heads),
            "records": len(self.records),
            "steps": list(self.steps),
            "lines_by_last_writer": by_step,
            "dropped": {step: len(ids) for step, ids in self.drops.items()},
        }

    def to_jsonl(self, path: str | None = None) -> str:
        payload = [json.dumps({"run_id": self.run_id, **record.to_dict()},
                              ensure_ascii=False) for record in self.records.values()]
        text = "\n".join(payload)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return text


class LineageMiddleware(Middleware):
    """Track line-level provenance of the payload as it moves through the flow.

    Args:
        extract: what to track. A callable ``payload -> artifact``, or a string
                 key: dict payloads are reduced to that key (payload itself
                 until the key appears). Default: the whole payload.
        source_label: attribution for the initial payload (default "input").
        similarity: 0..1 line-similarity threshold for modify-vs-generate.

    Besides per-step evolution the ledger can end with two run-end
    pseudo-steps: ``early_return`` when EarlyReturn substituted the output,
    and ``egress`` when an inner middleware transformed the final output on
    the way out — policy egress redaction is the canonical case. on_run_end
    hooks run innermost-first, so the sweep sees such transforms only from
    OUTSIDE the transforming middleware: list Lineage before Policy. The
    sweep re-attributes the *final* state; earlier records keep the text
    they captured at step end, so a ledger that must never hold a secret
    still needs a scrubbing ``extract`` (scrub at capture).
    """

    name = "lineage"

    def __init__(self, extract: Callable[[Any], Any] | str | None = None,
                 source_label: str = "input", similarity: float = SIMILARITY_THRESHOLD):
        if isinstance(extract, str):
            key = extract
            extract = lambda p: p.get(key, p) if isinstance(p, dict) else p  # noqa: E731
        self.extract = extract or (lambda payload: payload)
        self.source_label = source_label
        self.similarity = similarity

    def _artifact(self, payload: Any) -> Any:
        try:
            return self.extract(payload)
        except (KeyError, TypeError, AttributeError, IndexError):
            return payload

    def on_run_start(self, ctx: RunContext, payload):
        ledger = LineageLedger(run_id=ctx.run_id, similarity=self.similarity)
        ledger.snapshot_source(self._artifact(payload), self.source_label)
        ctx.artifacts["lineage"] = ledger
        return payload

    def on_step_end(self, ctx: RunContext, step: Step, payload, output):
        ledger: LineageLedger = ctx.artifacts["lineage"]
        stats = ledger.evolve(step.name, self._artifact(output))
        ctx.emit("lineage_evolved", step=step.name, **stats)
        return output

    def on_run_end(self, ctx: RunContext, output):
        # Finalizer sweep: the ledger must end on the state the caller
        # actually saw. Two ways the final output can differ from the last
        # on_step_end snapshot: EarlyReturn substituted it (attribute to
        # "early_return"), or an inner middleware transformed it during the
        # run-end unwind — policy egress redaction is the canonical case
        # (attribute to "egress").
        ledger = ctx.artifacts.get("lineage")  # may be absent: see Middleware docs
        if ledger is None:
            return output
        artifact = self._artifact(output)
        if ctx.short_circuited:
            step = "early_return"
        elif lines_of(artifact) != ledger.current_lines():
            step = "egress"
        else:
            return output
        stats = ledger.evolve(step, artifact)
        ctx.emit("lineage_evolved", step=step, **stats)
        return output
