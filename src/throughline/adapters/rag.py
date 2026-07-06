"""RAG adapters: plug any retriever into a dict-shaped pipeline payload.

Convention: RAG payloads are dicts flowing through steps that each add a key —
{"question"} -> +{"context"} -> +{"prompt"} -> +{"answer"}. Every helper here
follows it, so third-party retrievers slot in with one line.

Evidence lineage comes for free with this adapter: ``retriever_step`` records
every returned chunk in the run's EvidenceLedger (id, source metadata, score,
which retriever, which step), and ``prompt_step(cite=...)`` renders chunks
with their [eN] ids so a downstream ``citations_step`` can link answer lines
back to sources. See ``throughline.modules.citations``.

The evidence *contract* is ``EvidenceChunk``: a retriever that returns chunks
states its provenance (source, span, score) explicitly and skips the
duck-typed doc extraction entirely. Foreign doc objects are adapted through
``EvidenceChunk.from_doc`` — guessing is the fallback, not the interface.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from ..context import RunContext
from ..errors import ThroughlineError
from ..modules.citations import EvidenceChunk, evidence_ledger
from ..step import Step

RETRIEVE_METHODS = ("retrieve", "get_relevant_documents", "search", "invoke", "query", "__call__")


def retriever_step(retriever: Any, top_k: int | None = None,
                   query_key: str = "question", out_key: str = "context",
                   name: str = "retrieve", evidence: bool = True) -> Step:
    """Wrap any duck-typed retriever: payload[query_key] -> payload[out_key]=list[str].

    A retriever may return ``EvidenceChunk`` objects — the explicit evidence
    contract (text + source + span + score), taken verbatim. Anything else
    (framework docs, dicts, strings) goes through ``EvidenceChunk.from_doc``
    duck-typing.

    With ``evidence=True`` (default) every chunk is also recorded in the
    run's evidence ledger — the deterministic half of evidence lineage.
    Costs nothing; set False to skip.
    """
    method = None
    for candidate in RETRIEVE_METHODS:
        if candidate == "__call__" and callable(retriever):
            method = retriever
            break
        if callable(getattr(retriever, candidate, None)):
            method = getattr(retriever, candidate)
            break
    if method is None:
        raise ThroughlineError(
            f"cannot adapt retriever {type(retriever).__name__}: "
            f"none of {RETRIEVE_METHODS} found")

    def fn(payload, ctx: RunContext):
        query = payload[query_key] if isinstance(payload, dict) else str(payload)
        documents = method(query)
        if top_k is not None:
            documents = list(documents)[:top_k]
        ledger = evidence_ledger(ctx) if evidence else None
        texts = []
        for doc in documents:
            chunk = EvidenceChunk.from_doc(doc)
            texts.append(chunk.text)
            if ledger is not None:
                ledger.add(chunk, retriever=type(retriever).__name__, step=name)
        ctx.metric("retrieval.docs", len(texts))
        if isinstance(payload, dict):
            return {**payload, out_key: texts}
        return {query_key: query, out_key: texts}

    return Step(fn=fn, name=name, meta={"adapter": "retriever"})


def prompt_step(template: str, out_key: str = "prompt", name: str = "prompt",
                cite: str | tuple[str, ...] = ()) -> Step:
    """Render a str.format template from the payload dict.

    List values are joined with newlines so {context} reads naturally.

    ``cite="context"`` renders that list's items prefixed with their evidence
    ids — ``[e1] chunk text`` — so the model can cite sources and a downstream
    ``citations_step`` can validate the links. Items may be strings or
    ``EvidenceChunk`` objects; chunks unknown to the evidence ledger are
    registered on the fly (with their own provenance, attributed to this step).
    """
    cite_keys = (cite,) if isinstance(cite, str) else tuple(cite)

    def fn(payload, ctx: RunContext):
        if not isinstance(payload, dict):
            payload = {"input": payload}
        view = {}
        for key, value in payload.items():
            if isinstance(value, list):
                if key in cite_keys:
                    ledger = evidence_ledger(ctx)
                    # original items go to the ledger: an EvidenceChunk keeps
                    # its provenance, a plain string registers as text-only
                    items = [f"[{ledger.add(item, step=name).id}] {item}"
                             for item in value]
                else:
                    items = list(map(str, value))
                view[key] = "\n".join(items)
            else:
                view[key] = value
        try:
            rendered = template.format(**view)
        except KeyError as exc:
            raise ThroughlineError(f"prompt template needs key {exc} "
                                 f"but payload has {sorted(payload)}") from exc
        return {**payload, out_key: rendered}

    return Step(fn=fn, name=name, meta={"adapter": "prompt"})


class KeywordRetriever:
    """Tiny dependency-free retriever: keyword-overlap scoring over a corpus.

    Good enough for demos, tests and smoke-checking pipelines before swapping
    in a real vector store (the step interface stays identical).
    """

    def __init__(self, corpus: Iterable[str], top_k: int = 3):
        self.corpus = list(corpus)
        self.top_k = top_k

    def retrieve(self, query: str) -> list[str]:
        query_words = {w.lower().strip(".,!?") for w in query.split() if len(w) > 2}
        scored = []
        for doc in self.corpus:
            doc_words = {w.lower().strip(".,!?") for w in doc.split()}
            score = len(query_words & doc_words)
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda pair: -pair[0])
        return [doc for _, doc in scored[: self.top_k]]


def make_keyword_retriever(corpus: list[str], top_k: int = 3,
                           **step_kwargs: Any) -> Step:
    """Factory (preset-friendly): corpus -> ready retriever step."""
    return retriever_step(KeywordRetriever(corpus, top_k=top_k), **step_kwargs)
