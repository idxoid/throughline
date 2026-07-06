"""Hero walkthrough: debug a bad RAG answer with throughline.

Run:  PYTHONPATH=src python3 examples/debug_bad_rag.py

The scenario is the one every RAG team hits in production: the model returns
a fluent, fully-cited answer — and two of its lines are wrong. One is a
*fabrication* (a specific the docs never mention); the other is a *contradiction*
(the docs say the opposite). A naive "every line has an [eN], ship it" check
passes. throughline does not: in a single run it surfaces

  * evidence   — which source chunk was retrieved, with path + span
  * citations  — which answer line claims to rest on which chunk
  * a verdict  — an opt-in verifier that judges claim-vs-evidence, so a
                 citation that does not actually support its line is caught
  * metrics    — token cost, cited/verified/flagged counters, USD budget
  * lineage    — git-blame for the answer: which step wrote each line

Everything here is offline, deterministic and stdlib-only. Swap the fake LLM
for a real one (`throughline.adapters.llm`) and the toy verifier for an NLI
model or an LLM judge — the plumbing is identical.
"""

from __future__ import annotations

import re

import throughline as tl
from throughline.adapters.rag import retriever_step
from throughline.modules import (
    EvidenceChunk,
    LineageMiddleware,
    MetricsMiddleware,
    Observe,
    Quota,
    Validate,
    citations_step,
    verify_claims_step,
)
from throughline.modules.citations import evidence_ledger

# --- the corpus: three chunks, each with real provenance -------------------
# A retriever that returns EvidenceChunk objects states its own provenance
# (path, char span) instead of being duck-typed — see README, "Evidence
# & claim lineage".
CORPUS = [
    EvidenceChunk(
        text="Internal chat logs are retained for 90 days, then purged automatically.",
        source={"path": "policies/retention.md", "title": "Data Retention"},
        span=(0, 71),
    ),
    EvidenceChunk(
        text="Chat logs may be used for internal analytics dashboards.",
        source={"path": "policies/data-use.md", "title": "Acceptable Use"},
        span=(0, 56),
    ),
    EvidenceChunk(
        text="Chat logs must not be used to train machine learning models, "
             "even after anonymization.",
        source={"path": "policies/data-use.md", "title": "Acceptable Use"},
        span=(140, 226),
    ),
]

QUESTION = ("How long do we keep internal chat logs, and can we use them for "
            "analytics or model training?")


class PolicyRetriever:
    """Trivial retriever: hands back the whole (tiny) policy corpus."""

    def retrieve(self, query: str) -> list[EvidenceChunk]:
        return list(CORPUS)


def bad_llm(payload, ctx) -> dict:
    """The answer we are debugging — as if a real model produced it.

    Two lines are grounded, two are not. Every line carries a citation, so
    the failure is invisible to a citation-*count* check. We look each chunk's
    id up in the evidence ledger the retriever just populated, exactly as a
    real cite-parsing model would be scored against it.
    """
    ledger = evidence_ledger(ctx)
    retention, analytics, training = (ledger.id_for(c.text) for c in CORPUS)

    lines = [
        # grounded — verbatim from the retention policy
        f"Internal chat logs are retained for 90 days, then purged automatically. [{retention}]",
        # grounded — verbatim from the acceptable-use policy
        f"They may be used for internal analytics dashboards. [{analytics}]",
        # FABRICATION — cites the retention chunk, which says nothing about backups
        f"Deleted logs are also archived to cold storage for one year. [{retention}]",
        # CONTRADICTION — cites the chunk that explicitly forbids this
        f"Logs can be used to train machine learning models after anonymization. [{training}]",
    ]
    answer = "\n".join(lines)

    ctx.metric("llm.calls")
    ctx.metric("llm.input_tokens", 180)
    ctx.metric("llm.output_tokens", len(answer.split()))
    return {**payload, "answer": answer}


# --- a toy NLI-style verifier (swap for a real model / LLM judge) ----------
NEGATION = ("not", "never", "cannot", "prohibited", "forbidden", "disallowed")
STOP = {"the", "for", "are", "and", "may", "can", "they", "them", "used",
        "also", "then", "after", "even"}


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 2 and w not in STOP}


def entailment_stub(claim: str, evidence_texts: list[str]) -> dict:
    """Judge one claim against the evidence it cites.

    Returns a {"verdict", "confidence"} dict — the NLI shape that can tell a
    contradiction from a mere lack of support. This stub reasons by token
    overlap + negation; a production verifier would be a real NLI model or an
    LLM judge. The point of the demo is the *plumbing*, not this heuristic.
    """
    claim_terms = _content(claim)
    if not claim_terms:
        return {"verdict": "unsupported", "confidence": 0.0}
    claim_negated = any(cue in claim.lower() for cue in NEGATION)

    best = 0.0
    for ev in evidence_texts:
        overlap = len(claim_terms & _content(ev)) / len(claim_terms)
        best = max(best, overlap)
        ev_negated = any(cue in ev.lower() for cue in NEGATION)
        # evidence forbids what the claim asserts affirmatively -> contradiction
        if ev_negated and not claim_negated and overlap >= 0.5:
            return {"verdict": "contradicted", "confidence": 0.9}

    if best >= 0.6:
        return {"verdict": "supported", "confidence": round(best, 2)}
    if best >= 0.35:
        return {"verdict": "low_confidence_support", "confidence": round(best, 2)}
    return {"verdict": "unsupported", "confidence": round(best, 2)}


flow = tl.Flow(
    [
        retriever_step(PolicyRetriever(), name="retrieve"),   # -> evidence ledger
        tl.as_step(bad_llm, "llm"),                           # the answer under test
        citations_step(require="warn", name="citations"),     # parse & validate [eN]
        verify_claims_step(entailment_stub, name="verify"),   # opt-in verdicts
    ],
    middleware=[
        Observe("console"),                                   # events -> stderr
        MetricsMiddleware(),                                  # timings + counters
        Validate(schema={"type": "object", "required": ["answer"]}),
        Quota(max_cost=0.05,                                  # USD budget -> metric
              cost={"llm.input_tokens": 1e-6, "llm.output_tokens": 3e-6}),
        LineageMiddleware(extract="answer"),                  # blame over the answer
    ],
    name="debug-rag",
)


STATUS_MARK = {
    "supported": "ok ",
    "contradicted": "!! ",
    "unsupported": "?? ",
    "low_confidence_support": " ~ ",
    "uncited_line": " · ",
    None: "   ",
}


def main() -> None:
    result = flow.run({"question": QUESTION})
    ctx = result.ctx
    evidence = ctx.artifacts["evidence"]
    claims = ctx.artifacts["claims"]

    print("\n=== question ===")
    print(QUESTION)

    print("\n=== answer (as returned to the user) ===")
    print(result.output["answer"])

    print("\n=== evidence ledger (what retrieval actually surfaced) ===")
    for record in evidence.records.values():
        src = record.source or {}
        loc = src.get("path", "?") if isinstance(src, dict) else str(src)
        print(f"  [{record.id}] {loc} span={record.span}")
        print(f"       {record.text}")

    print("\n=== per-line verdict (evidence + claim lineage, joined) ===")
    print("      ok supported   !! contradicted   ?? unsupported   · uncited\n")
    for entry in claims.join_blame(lineage=result.lineage, evidence=evidence):
        mark = STATUS_MARK.get(entry["status"], "   ")
        cites = ",".join(entry["evidence"]) or "-"
        conf = entry.get("confidence")
        conf_s = f" conf={conf:.2f}" if conf is not None else ""
        print(f"  {mark}L{entry['line_no'] + 1} [{cites}]{conf_s}  {entry['text']}")
        for source in entry.get("sources", []):
            path = (source.get("source") or {}).get("path", "?") \
                if isinstance(source.get("source"), dict) else "?"
            print(f"        └─ {source['id']} {path}: {source['text']}")

    print("\n=== violations (why this answer should NOT have shipped) ===")
    for violation in result.violations:
        print(f"  - {violation}")

    print("\n=== metrics ===")
    counters = result.metrics.get("counters", {})
    for key in sorted(counters):
        print(f"  {key:24} {counters[key]:g}")

    print("\n=== lineage: git-blame for the answer text ===")
    print(result.lineage.render_blame())

    print("\n=== claim status counts ===")
    print(f"  {claims.status_counts()}")


if __name__ == "__main__":
    main()
