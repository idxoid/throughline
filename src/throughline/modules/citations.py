"""Evidence & claim lineage — the two ledgers that complete line-level lineage.

throughline tracks three kinds of provenance, each with its own mechanism and
its own determinism budget:

  edit lineage      which step wrote each output line
                    -> LineageMiddleware (difflib, deterministic, free)
  evidence lineage  which source chunks the context came from
                    -> EvidenceLedger, filled by retriever steps
                       (metadata propagation, deterministic, free)
  claim lineage     which answer line is supported by which evidence
                    -> ClaimLedger via the *citation contract*: the LLM is
                       prompted to emit [e1]-style markers; ``citations_step``
                       parses and validates them deterministically. Optional
                       ``verify_claims_step`` adds a stochastic confidence
                       score (NLI / LLM judge) — opt-in, it costs money.

Generation is stochastic; verification of the links is deterministic.

Line statuses form a taxonomy that keeps facts and verdicts apart:

  uncited_line            structural FACT: the line cites nothing. Not a
                          judgment — headers, transitions, summaries and
                          style-mandated phrasing legitimately go uncited
                          (``exempt=`` whitelists them, ``require=`` sets
                          the policy for the rest).
  cited                   fact: the line cites valid evidence (default claim
                          status before verification).
  supported               VERDICTS, produced only by verify_claims_step:
  low_confidence_support  the cited evidence does / weakly does / does not /
  unsupported             actively does not back the claim.
  contradicted

"Hallucination" is a conclusion drawn from unsupported/contradicted verdicts
— never from the mere absence of a citation.

The ledgers join on line text/number: ``ClaimLedger.join_blame`` merges the
final artifact's blame (edit lineage) with claims and evidence into one
per-line view: who wrote the line, what it cites, where that evidence came
from, its status and the verifier's confidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..context import RunContext
from ..errors import ValidationError
from ..step import Step

CITATION_PATTERN = re.compile(r"\s*\[(e\d+)\]")


# ---------------------------------------------------------------------------
# Evidence lineage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceChunk:
    """The evidence *contract*: one retrieved chunk with explicit provenance.

    Retrievers (and any custom step) may hand these to the pipeline directly
    instead of framework doc objects — an explicit contract instead of
    duck-typed guessing. ``from_doc`` adapts foreign docs (LangChain,
    LlamaIndex, dicts) when you don't control the retriever.

    ``str(chunk)`` is the text, so chunks drop into any code that expects
    strings (prompt templates, joins) unchanged. ``source`` may be anything
    JSON-able — a path, doc metadata, an ``ArtifactRef.to_dict()`` pointing
    into the artifact store; ``span`` locates the chunk inside that source
    (line or char offsets — your convention, carried verbatim).
    """

    text: str
    source: Any = None
    span: tuple[int, int] | None = None
    score: float | None = None

    def __str__(self) -> str:
        return self.text

    @classmethod
    def from_doc(cls, doc: Any) -> "EvidenceChunk":
        """Duck-type a foreign doc object into the contract."""
        if isinstance(doc, EvidenceChunk):
            return doc
        return cls(text=_doc_text(doc), source=_doc_source(doc),
                   score=_doc_score(doc))

    def to_dict(self) -> dict:
        data: dict = {"text": self.text}
        if self.source is not None:
            data["source"] = self.source
        if self.span is not None:
            data["span"] = list(self.span)
        if self.score is not None:
            data["score"] = self.score
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceChunk":
        span = data.get("span")
        return cls(text=data["text"], source=data.get("source"),
                   span=tuple(span) if span else None, score=data.get("score"))


def _doc_text(doc: Any) -> str:
    for attr in ("page_content", "text", "content"):  # LangChain / LlamaIndex / generic
        value = getattr(doc, attr, None)
        if isinstance(value, str):
            return value
    if isinstance(doc, dict):
        for key in ("text", "content", "page_content"):
            if isinstance(doc.get(key), str):
                return doc[key]
    return str(doc)


def _doc_source(doc: Any) -> Any:
    for attr in ("metadata", "extra_info"):  # LangChain / LlamaIndex
        value = getattr(doc, attr, None)
        if isinstance(value, dict) and value:
            return value
    if isinstance(doc, dict):
        for key in ("metadata", "source", "id"):
            if doc.get(key):
                return doc[key]
    return None


def _doc_score(doc: Any) -> float | None:
    for attr in ("score", "similarity"):
        value = getattr(doc, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(doc, dict) and isinstance(doc.get("score"), (int, float)):
        return float(doc["score"])
    return None


@dataclass
class EvidenceRecord:
    """A ledger entry: an EvidenceChunk plus id and attribution."""

    id: str
    text: str
    source: Any = None          # doc metadata, path, ArtifactRef dict, ...
    span: tuple[int, int] | None = None
    score: float | None = None
    retriever: str = ""
    step: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "source": self.source,
                "span": list(self.span) if self.span else None,
                "score": self.score, "retriever": self.retriever, "step": self.step}


class EvidenceLedger:
    """Registry of retrieved chunks. Same text = same evidence id (dedup)."""

    def __init__(self) -> None:
        self.records: dict[str, EvidenceRecord] = {}
        self._by_text: dict[str, str] = {}

    def add(self, chunk: EvidenceChunk | str, source: Any = None,
            span: tuple[int, int] | None = None, score: float | None = None,
            retriever: str = "", step: str = "") -> EvidenceRecord:
        """Register a chunk (or bare text). Explicit kwargs override the
        chunk's own fields; identical text keeps its original record."""
        if isinstance(chunk, EvidenceChunk):
            text = chunk.text
            source = chunk.source if source is None else source
            span = chunk.span if span is None else span
            score = chunk.score if score is None else score
        else:
            text = str(chunk)
        existing = self._by_text.get(text)
        if existing is not None:
            return self.records[existing]
        record = EvidenceRecord(id=f"e{len(self.records) + 1}", text=text,
                                source=source, span=span, score=score,
                                retriever=retriever, step=step)
        self.records[record.id] = record
        self._by_text[text] = record.id
        return record

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return self.records.get(evidence_id)

    def id_for(self, text: str) -> str | None:
        return self._by_text.get(text)

    def to_jsonl(self, path: str | None = None) -> str:
        text = "\n".join(json.dumps(r.to_dict(), ensure_ascii=False, default=str)
                         for r in self.records.values())
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return text

    def __len__(self) -> int:
        return len(self.records)


def evidence_ledger(ctx: RunContext) -> EvidenceLedger:
    """The run's evidence ledger, created on first use."""
    ledger = ctx.artifacts.get("evidence")
    if ledger is None:
        ledger = ctx.artifacts["evidence"] = EvidenceLedger()
    return ledger


# ---------------------------------------------------------------------------
# Claim lineage
# ---------------------------------------------------------------------------

#: Facts (set by citations_step) vs verdicts (set only by verify_claims_step).
CLAIM_STATUSES = ("cited", "supported", "low_confidence_support",
                  "unsupported", "contradicted")
VERDICTS = ("supported", "low_confidence_support", "unsupported", "contradicted")


@dataclass
class ClaimRecord:
    line_no: int                 # position in the cleaned artifact at parse time
    text: str                    # cleaned line text
    evidence: list[str] = field(default_factory=list)
    step: str = ""
    method: str = "citation"     # "citation" | verifier name
    status: str = "cited"        # fact until a verifier upgrades it to a verdict
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {"line_no": self.line_no, "text": self.text, "evidence": self.evidence,
                "step": self.step, "method": self.method, "status": self.status,
                "confidence": self.confidence}


class ClaimLedger:
    def __init__(self) -> None:
        self.claims: list[ClaimRecord] = []
        self.uncited: list[dict] = []    # facts: {"line_no", "text"} — no judgment

    def add(self, claim: ClaimRecord) -> None:
        self.claims.append(claim)

    def add_uncited(self, line_no: int, text: str) -> None:
        self.uncited.append({"line_no": line_no, "text": text})

    def for_text(self, text: str) -> ClaimRecord | None:
        for claim in self.claims:
            if claim.text == text:
                return claim
        return None

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for claim in self.claims:
            counts[claim.status] = counts.get(claim.status, 0) + 1
        if self.uncited:
            counts["uncited_line"] = len(self.uncited)
        return counts

    def join_blame(self, lineage: Any = None, evidence: EvidenceLedger | None = None) -> list[dict]:
        """Per-line view joining edit lineage, claims and evidence.

        Without a lineage ledger, renders one entry per claim. With one,
        every line of the final artifact appears — cited or not. Each entry
        carries a ``status``: a claim's fact/verdict, "uncited_line" for
        lines recorded as uncited, None for everything else (empty or
        exempt lines).
        """
        entries: list[dict] = []
        if lineage is None:
            base = [{"line_no": c.line_no, "text": c.text} for c in self.claims]
        else:
            base = lineage.blame()
        uncited_texts = {item["text"] for item in self.uncited}
        for item in base:
            claim = self.for_text(item["text"])
            entry = dict(item)
            entry["evidence"] = list(claim.evidence) if claim else []
            entry["confidence"] = claim.confidence if claim else None
            if claim:
                entry["status"] = claim.status
            elif item["text"] in uncited_texts:
                entry["status"] = "uncited_line"
            else:
                entry["status"] = None
            if evidence is not None and claim:
                entry["sources"] = [
                    record.to_dict() for eid in claim.evidence
                    if (record := evidence.get(eid)) is not None
                ]
            entries.append(entry)
        return entries

    def to_jsonl(self, path: str | None = None) -> str:
        text = "\n".join(json.dumps(c.to_dict(), ensure_ascii=False, default=str)
                         for c in self.claims)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return text

    def __len__(self) -> int:
        return len(self.claims)


def citations_step(answer_key: str = "answer", require: str | None = None,
                   exempt: str | Callable[[str], bool] | None = None,
                   name: str = "citations") -> Step:
    """Parse & validate [eN] citation markers — the deterministic half of
    claim lineage. Place right after the generating step.

    - extracts markers per line, validates ids against the evidence ledger
      (a citation of unknown evidence is a violation);
    - strips markers from the answer, so downstream steps and edit lineage
      see clean text;
    - records line -> evidence links (claims, status "cited") and uncited
      lines (status "uncited_line") in ``ctx.artifacts["claims"]``.

    An uncited line is a structural FACT, not a judgment: headers,
    transitions, summaries and style-mandated phrasing legitimately cite
    nothing. Verdicts about claims (supported / unsupported / contradicted)
    belong to ``verify_claims_step``.

    Args:
        answer_key: payload key holding the generated text.
        require:    policy for non-empty, non-exempt lines with no citation:
                    None (default) — recorded as facts only; "warn" —
                    also a violation; "raise" — ValidationError.
        exempt:     lines that legitimately go uncited: a regex (matched
                    against the cleaned line) or a callable ``line -> bool``.
                    Exempt lines are neither recorded as uncited nor counted
                    by the ``require`` policy.
    """
    if require not in (None, "warn", "raise"):
        raise ValueError(f"require must be None, 'warn' or 'raise', got {require!r}")
    if isinstance(exempt, str):
        pattern = re.compile(exempt)
        exempt = lambda line: bool(pattern.search(line))  # noqa: E731

    def fn(payload, ctx: RunContext):
        text = payload[answer_key] if isinstance(payload, dict) else str(payload)
        evidence: EvidenceLedger | None = ctx.artifacts.get("evidence")
        claims = ctx.artifacts.setdefault("claims", ClaimLedger())
        violations = ctx.artifacts.setdefault("violations", [])

        cleaned_lines: list[str] = []
        for line_no, line in enumerate(text.splitlines()):
            ids = CITATION_PATTERN.findall(line)
            cleaned = CITATION_PATTERN.sub("", line).rstrip()
            cleaned_lines.append(cleaned)
            valid_ids = []
            for eid in ids:
                if evidence is None or evidence.get(eid) is None:
                    violations.append(
                        f"line {line_no + 1} cites unknown evidence {eid!r}")
                    ctx.emit("citation_invalid", line_no=line_no, evidence=eid)
                else:
                    valid_ids.append(eid)
            if valid_ids:
                claims.add(ClaimRecord(line_no=line_no, text=cleaned,
                                       evidence=valid_ids, step=name))
                ctx.metric("claims.cited")
            elif cleaned.strip() and not (exempt and exempt(cleaned)):
                claims.add_uncited(line_no, cleaned)
                ctx.metric("claims.uncited_lines")

        if claims.uncited:
            lines = [item["line_no"] + 1 for item in claims.uncited]
            ctx.emit("uncited_lines", lines=lines)
            if require is not None:
                message = (f"{len(lines)} uncited line(s): "
                           f"{', '.join(map(str, lines[:10]))}")
                violations.append(message)
                if require == "raise":
                    raise ValidationError(message, step=name)

        cleaned_text = "\n".join(cleaned_lines)
        if isinstance(payload, dict):
            return {**payload, answer_key: cleaned_text}
        return cleaned_text

    return Step(fn=fn, name=name, meta={"adapter": "citations"})


def _normalize_verdict(result: Any, threshold: float | None) -> tuple[str, float | None]:
    """Verifier output -> (verdict, confidence).

    Accepted shapes: a number (score; ``threshold`` splits supported vs
    low_confidence_support — a scalar cannot distinguish "no support" from
    "weak support"), a verdict string, a (verdict, confidence) tuple, or a
    {"verdict": ..., "confidence": ...} dict for verifiers that can tell
    unsupported from contradicted (NLI-style).
    """
    if isinstance(result, dict):
        verdict, confidence = result.get("verdict"), result.get("confidence")
    elif isinstance(result, tuple):
        verdict, confidence = result
    elif isinstance(result, str):
        verdict, confidence = result, None
    else:
        confidence = float(result)
        verdict = ("supported" if threshold is None or confidence >= threshold
                   else "low_confidence_support")
    if verdict not in VERDICTS:
        raise ValidationError(
            f"verifier returned unknown verdict {verdict!r}; "
            f"expected one of {VERDICTS}")
    return verdict, float(confidence) if confidence is not None else None


def verify_claims_step(verifier: Callable[[str, list[str]], Any] | str,
                       threshold: float | None = None, on_fail: str = "warn",
                       fail_on: tuple[str, ...] = ("unsupported", "contradicted",
                                                   "low_confidence_support"),
                       name: str = "verify-claims") -> Step:
    """Stochastic half of claim lineage: pass a VERDICT on each cited claim.

    This is where "hallucination" becomes a conclusion instead of a guess:
    the verifier judges claim-vs-evidence and upgrades the claim's status to
    supported / low_confidence_support / unsupported / contradicted. Uncited
    lines are out of scope here — they are facts recorded by citations_step,
    not claims to verify.

    ``verifier(claim_text, evidence_texts)`` may return a 0..1 score (split
    by ``threshold``), a verdict string, a ``(verdict, confidence)`` tuple or
    a ``{"verdict", "confidence"}`` dict (NLI-style verifiers that can tell
    contradiction from mere lack of support). This step costs tokens: it is
    a separate opt-in knob, goes through Quota, and is cacheable.

    Args:
        verifier:  callable or "pkg.mod:fn" import path (kind="verifier").
        threshold: for score-only verifiers: below it = low_confidence_support.
        fail_on:   statuses that count as violations (default: everything
                   except "supported").
        on_fail:   "warn" (default) or "raise" when flagged claims exist.
    """
    if isinstance(verifier, str):
        from ..registry import resolve
        verifier = resolve(verifier, kind="verifier")
    if on_fail not in ("warn", "raise"):
        raise ValueError(f"on_fail must be 'warn' or 'raise', got {on_fail!r}")
    unknown = set(fail_on) - set(VERDICTS)
    if unknown:
        raise ValueError(f"fail_on contains unknown verdicts: {sorted(unknown)}")

    def fn(payload, ctx: RunContext):
        claims: ClaimLedger | None = ctx.artifacts.get("claims")
        evidence: EvidenceLedger | None = ctx.artifacts.get("evidence")
        if not claims or evidence is None:
            return payload
        flagged: list[ClaimRecord] = []
        for claim in claims.claims:
            texts = [record.text for eid in claim.evidence
                     if (record := evidence.get(eid)) is not None]
            if not texts:
                continue
            verdict, confidence = _normalize_verdict(
                verifier(claim.text, texts), threshold)
            claim.status = verdict
            claim.confidence = confidence
            claim.method = getattr(verifier, "__name__", "verifier")
            ctx.metric("claims.verified")
            ctx.metric(f"claims.{verdict}")
            if verdict in fail_on:
                flagged.append(claim)
        if flagged:
            details = "; ".join(
                f"line {c.line_no + 1} ({c.status}"
                + (f", {c.confidence:.2f}" if c.confidence is not None else "")
                + ")"
                for c in flagged[:5])
            message = f"{len(flagged)} claim(s) flagged by verifier: {details}"
            ctx.artifacts.setdefault("violations", []).append(message)
            ctx.emit("claims_flagged", counts=claims.status_counts())
            if on_fail == "raise":
                raise ValidationError(message, step=name)
        return payload

    return Step(fn=fn, name=name, meta={"adapter": "verify-claims"})
