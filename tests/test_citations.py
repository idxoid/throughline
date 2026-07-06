"""Three lineages: edit (existing), evidence, claim — and their join."""

import unittest

import followers as fl
from followers.adapters.rag import KeywordRetriever, prompt_step, retriever_step
from followers.errors import ValidationError
from followers.modules import LineageMiddleware
from followers.modules.citations import (ClaimLedger, EvidenceChunk, EvidenceLedger,
                                         citations_step, verify_claims_step)

CORPUS = [
    "Lineage answers which step wrote every line.",
    "Evidence lineage tracks which chunks the context came from.",
    "Claim lineage links answer lines to supporting evidence.",
]


class EvidenceLedgerTests(unittest.TestCase):
    def test_dedup_same_text_same_id(self):
        ledger = EvidenceLedger()
        first = ledger.add("chunk", source={"doc": "a.pdf"})
        second = ledger.add("chunk")
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(ledger), 1)

    def test_retriever_step_records_evidence(self):
        step = retriever_step(KeywordRetriever(CORPUS, top_k=2))
        result = fl.Flow([step]).run({"question": "which step wrote every line"})
        ledger = result.ctx.artifacts["evidence"]
        self.assertGreater(len(ledger), 0)
        record = next(iter(ledger.records.values()))
        self.assertEqual(record.retriever, "KeywordRetriever")
        self.assertEqual(record.step, "retrieve")

    def test_retriever_step_evidence_opt_out(self):
        step = retriever_step(KeywordRetriever(CORPUS), evidence=False)
        result = fl.Flow([step]).run({"question": "lineage"})
        self.assertNotIn("evidence", result.ctx.artifacts)

    def test_source_metadata_extraction(self):
        class Doc:
            def __init__(self, text, metadata, score):
                self.page_content = text
                self.metadata = metadata
                self.score = score

        class MetaRetriever:
            def retrieve(self, query):
                return [Doc("chunk one", {"source": "report.pdf", "page": 12}, 0.87)]

        result = fl.Flow([retriever_step(MetaRetriever())]).run({"question": "q"})
        record = next(iter(result.ctx.artifacts["evidence"].records.values()))
        self.assertEqual(record.source["source"], "report.pdf")
        self.assertAlmostEqual(record.score, 0.87)


class EvidenceChunkContractTests(unittest.TestCase):
    def test_str_is_the_text(self):
        chunk = EvidenceChunk(text="the chunk", source="a.pdf")
        self.assertEqual(str(chunk), "the chunk")
        self.assertEqual("\n".join(map(str, [chunk, chunk])), "the chunk\nthe chunk")

    def test_retriever_returning_chunks_skips_guessing(self):
        """The contract path: provenance is taken verbatim, not duck-typed."""
        class ContractRetriever:
            def retrieve(self, query):
                return [EvidenceChunk(text="exact chunk",
                                      source={"$artifact": "corpus/abc"},
                                      span=(120, 134), score=0.91)]

        result = fl.Flow([retriever_step(ContractRetriever())]).run({"question": "q"})
        self.assertEqual(result.output["context"], ["exact chunk"])
        record = next(iter(result.ctx.artifacts["evidence"].records.values()))
        self.assertEqual(record.source, {"$artifact": "corpus/abc"})
        self.assertEqual(record.span, (120, 134))
        self.assertAlmostEqual(record.score, 0.91)
        self.assertEqual(record.retriever, "ContractRetriever")

    def test_from_doc_adapts_foreign_objects(self):
        class Doc:
            page_content = "foreign text"
            metadata = {"source": "b.pdf"}
            score = 0.5

        chunk = EvidenceChunk.from_doc(Doc())
        self.assertEqual(chunk.text, "foreign text")
        self.assertEqual(chunk.source, {"source": "b.pdf"})
        self.assertAlmostEqual(chunk.score, 0.5)
        # idempotent on an already-conforming chunk
        self.assertIs(EvidenceChunk.from_doc(chunk), chunk)

    def test_dict_roundtrip(self):
        chunk = EvidenceChunk(text="t", source="s.pdf", span=(1, 5), score=0.7)
        again = EvidenceChunk.from_dict(chunk.to_dict())
        self.assertEqual(again, chunk)
        minimal = EvidenceChunk.from_dict({"text": "only text"})
        self.assertIsNone(minimal.source)
        self.assertIsNone(minimal.span)

    def test_ledger_accepts_chunk_and_dedups_by_text(self):
        ledger = EvidenceLedger()
        first = ledger.add(EvidenceChunk(text="same", source="a.pdf", span=(1, 2)))
        second = ledger.add("same")   # bare text, same content
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.source, "a.pdf")   # original provenance kept
        self.assertEqual(first.span, (1, 2))

    def test_prompt_cite_keeps_chunk_provenance(self):
        """Chunks placed straight into the payload keep their provenance when
        prompt_step registers them for citation."""
        chunk = EvidenceChunk(text="direct chunk", source="direct.pdf", span=(3, 4))
        result = fl.Flow([prompt_step("{context}", cite="context")]).run(
            {"question": "q", "context": [chunk]})
        self.assertRegex(result.output["prompt"], r"\[e1\] direct chunk")
        record = result.ctx.artifacts["evidence"].get("e1")
        self.assertEqual(record.source, "direct.pdf")
        self.assertEqual(record.span, (3, 4))

    def test_span_flows_into_join_blame_sources(self):
        class ContractRetriever:
            def retrieve(self, query):
                return [EvidenceChunk(text="spanned evidence",
                                      source="doc.md", span=(10, 20))]

        def fake_llm(payload, ctx):
            return {**payload, "answer": "A claim. [e1]"}

        result = fl.Flow([
            retriever_step(ContractRetriever()),
            prompt_step("{context}", cite="context"),
            fake_llm,
            citations_step(),
        ]).run({"question": "q"})
        joined = result.ctx.artifacts["claims"].join_blame(
            evidence=result.ctx.artifacts["evidence"])
        self.assertEqual(joined[0]["sources"][0]["span"], [10, 20])
        self.assertEqual(joined[0]["sources"][0]["source"], "doc.md")

    def test_strict_outputs_treats_chunks_as_plain(self):
        from followers.modules import StrictOutputs
        flow = fl.Flow([lambda p: {"context": [EvidenceChunk(text="ok")]}],
                       middleware=[StrictOutputs()])
        self.assertEqual(flow.run("q").violations, [])


class CitationContractTests(unittest.TestCase):
    def _flow(self, llm_text, require=None):
        def fake_llm(payload, ctx):
            return {**payload, "answer": llm_text}

        return fl.Flow([
            retriever_step(KeywordRetriever(CORPUS, top_k=3)),
            prompt_step("Context:\n{context}\n\nQ: {question}", cite="context"),
            fake_llm,
            citations_step(require=require),
        ])

    def test_prompt_cite_renders_markers(self):
        result = fl.Flow([
            retriever_step(KeywordRetriever(CORPUS, top_k=2)),
            prompt_step("{context}", cite="context"),
        ]).run({"question": "which step wrote every line"})
        self.assertRegex(result.output["prompt"], r"\[e\d+\] ")

    def test_valid_citations_become_claims_and_markers_are_stripped(self):
        result = self._flow(
            "Every line has an author. [e1]\nUnrelated remark."
        ).run({"question": "which step wrote every line"})
        self.assertNotIn("[e1]", result.output["answer"])
        claims = result.ctx.artifacts["claims"]
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims.claims[0].evidence, ["e1"])
        self.assertEqual(claims.claims[0].line_no, 0)

    def test_unknown_evidence_id_is_a_violation(self):
        result = self._flow("Bold claim. [e99]").run({"question": "lineage"})
        self.assertTrue(any("unknown evidence 'e99'" in v for v in result.violations))
        self.assertEqual(len(result.ctx.artifacts["claims"]), 0)

    def test_require_warn_flags_uncited_lines(self):
        result = self._flow("First. [e1]\nNo citation here.",
                            require="warn").run({"question": "lineage"})
        self.assertTrue(any("uncited line(s)" in v for v in result.violations))

    def test_require_raise(self):
        with self.assertRaises(fl.FlowError):
            self._flow("Naked claim.", require="raise").run({"question": "lineage"})

    def test_verifier_scores_claims(self):
        def overlap_verifier(claim, evidence_texts):
            claim_words = set(claim.lower().split())
            best = 0.0
            for text in evidence_texts:
                words = set(text.lower().split())
                if claim_words:
                    best = max(best, len(claim_words & words) / len(claim_words))
            return best

        flow = self._flow("Lineage answers which step wrote every line. [e1]")
        flow = flow.then(verify_claims_step(overlap_verifier, threshold=0.2))
        result = flow.run({"question": "which step wrote every line"})
        claim = result.ctx.artifacts["claims"].claims[0]
        self.assertIsNotNone(claim.confidence)
        self.assertGreater(claim.confidence, 0.2)
        self.assertEqual(claim.method, "overlap_verifier")
        self.assertEqual(claim.status, "supported")

    def test_verifier_threshold_violation(self):
        flow = self._flow("Completely unsupported nonsense. [e1]")
        flow = flow.then(verify_claims_step(lambda c, e: 0.0, threshold=0.5))
        result = flow.run({"question": "lineage"})
        claim = result.ctx.artifacts["claims"].claims[0]
        self.assertEqual(claim.status, "low_confidence_support")
        self.assertTrue(any("flagged by verifier" in v for v in result.violations))

    def test_verifier_raise_mode(self):
        flow = self._flow("Nonsense. [e1]")
        flow = flow.then(verify_claims_step(lambda c, e: 0.0, threshold=0.5,
                                            on_fail="raise"))
        with self.assertRaises(fl.FlowError):
            flow.run({"question": "lineage"})


class ClaimStatusTaxonomyTests(unittest.TestCase):
    """Facts (citations_step) vs verdicts (verify_claims_step)."""

    def _flow(self, llm_text, **citation_kwargs):
        def fake_llm(payload, ctx):
            return {**payload, "answer": llm_text}

        return fl.Flow([
            retriever_step(KeywordRetriever(CORPUS, top_k=3)),
            prompt_step("{context}\n\nQ: {question}", cite="context"),
            fake_llm,
            citations_step(**citation_kwargs),
        ])

    def test_uncited_is_a_recorded_fact_not_a_violation(self):
        result = self._flow(
            "Cited claim. [e1]\nA transitional phrase."
        ).run({"question": "lineage"})
        claims = result.ctx.artifacts["claims"]
        self.assertEqual(len(claims.uncited), 1)
        self.assertEqual(claims.uncited[0]["text"], "A transitional phrase.")
        self.assertEqual(result.violations, [])          # no require= -> no judgment
        self.assertEqual(claims.status_counts(),
                         {"cited": 1, "uncited_line": 1})

    def test_exempt_lines_are_not_even_facts(self):
        result = self._flow(
            "## Summary\nCited claim. [e1]\nIn conclusion, see above.",
            require="warn", exempt=r"^#|^In conclusion",
        ).run({"question": "lineage"})
        claims = result.ctx.artifacts["claims"]
        self.assertEqual(claims.uncited, [])             # both lines exempt
        self.assertEqual(result.violations, [])

    def test_verdict_string_and_tuple_and_dict(self):
        for returned, expected_status, expected_conf in [
            ("contradicted", "contradicted", None),
            (("unsupported", 0.9), "unsupported", 0.9),
            ({"verdict": "supported", "confidence": 0.97}, "supported", 0.97),
        ]:
            flow = self._flow("A claim. [e1]").then(
                verify_claims_step(lambda c, e, r=returned: r))
            claim = flow.run({"question": "lineage"}).ctx.artifacts["claims"].claims[0]
            self.assertEqual(claim.status, expected_status)
            self.assertEqual(claim.confidence, expected_conf)

    def test_unknown_verdict_is_rejected(self):
        flow = self._flow("A claim. [e1]").then(
            verify_claims_step(lambda c, e: "hallucination"))
        with self.assertRaises(fl.FlowError):
            flow.run({"question": "lineage"})

    def test_fail_on_narrows_the_policy(self):
        # low confidence tolerated, only hard verdicts are violations
        flow = self._flow("A claim. [e1]").then(
            verify_claims_step(lambda c, e: 0.1, threshold=0.5,
                               fail_on=("unsupported", "contradicted")))
        result = flow.run({"question": "lineage"})
        self.assertEqual(result.ctx.artifacts["claims"].claims[0].status,
                         "low_confidence_support")
        self.assertEqual(result.violations, [])

    def test_contradicted_can_raise(self):
        flow = self._flow("A claim. [e1]").then(
            verify_claims_step(lambda c, e: "contradicted", on_fail="raise"))
        with self.assertRaises(fl.FlowError):
            flow.run({"question": "lineage"})

    def test_join_blame_carries_statuses(self):
        flow = fl.Flow(
            [
                retriever_step(KeywordRetriever(CORPUS, top_k=3)),
                prompt_step("{context}\n\nQ: {question}", cite="context"),
                lambda p: {**p, "answer": "Cited claim. [e1]\nFree remark."},
                citations_step(),
                verify_claims_step(lambda c, e: ("supported", 0.9)),
            ],
            middleware=[LineageMiddleware(extract="answer")],
        )
        result = flow.run({"question": "lineage"})
        joined = result.ctx.artifacts["claims"].join_blame(lineage=result.lineage)
        by_text = {entry["text"]: entry["status"] for entry in joined}
        self.assertEqual(by_text["Cited claim."], "supported")
        self.assertEqual(by_text["Free remark."], "uncited_line")


class JoinBlameTests(unittest.TestCase):
    def test_three_ledgers_join_per_line(self):
        def fake_llm(payload, ctx):
            return {**payload,
                    "answer": "Lineage answers which step wrote every line. [e1]\n"
                              "Free remark without citation."}

        flow = fl.Flow(
            [
                retriever_step(KeywordRetriever(CORPUS, top_k=3)),
                prompt_step("{context}\n\nQ: {question}", cite="context"),
                fl.as_step(fake_llm, "draft"),
                citations_step(),
            ],
            middleware=[LineageMiddleware(extract="answer")],
        )
        result = flow.run({"question": "which step wrote every line"})

        joined = result.ctx.artifacts["claims"].join_blame(
            lineage=result.lineage,
            evidence=result.ctx.artifacts["evidence"])
        cited = [e for e in joined if e["evidence"]]
        self.assertEqual(len(cited), 1)
        entry = cited[0]
        # edit lineage: who wrote the line; claim lineage: what it cites;
        # evidence lineage: where that evidence came from.
        self.assertIn("step", entry)
        self.assertEqual(entry["evidence"], ["e1"])
        self.assertEqual(entry["sources"][0]["retriever"], "KeywordRetriever")
        uncited = [e for e in joined if not e["evidence"] and e["text"].strip()]
        self.assertGreaterEqual(len(uncited), 1)

    def test_join_without_lineage(self):
        claims = ClaimLedger()
        from followers.modules.citations import ClaimRecord
        claims.add(ClaimRecord(line_no=0, text="a claim", evidence=["e1"]))
        joined = claims.join_blame()
        self.assertEqual(joined[0]["evidence"], ["e1"])


if __name__ == "__main__":
    unittest.main()
