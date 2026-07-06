"""Example components for the `rag-docs` preset.

This stays in examples on purpose: production users normally swap the
`retriever` factory for their own vector store or search service. The local
factory is a dependency-free stand-in that reads an Obsidian-style docs
directory, chunks Markdown files, and exposes those chunks through the public
RAG adapter contract.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from throughline.adapters.rag import prompt_step, retriever_step
from throughline.context import RunContext
from throughline.modules.citations import EvidenceChunk, evidence_ledger
from throughline.step import Step

DEFAULT_CORPUS_DIR = "examples/corpus/obsidian"
DEFAULT_PATTERNS = ("**/*.md", "**/*.txt", "**/*.rst")
WORD_RE = re.compile(r"[A-Za-z0-9_/-]+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "the", "to", "what", "when",
    "where", "which", "with",
}


def normalize(payload, ctx: RunContext) -> dict:
    """Accept a CLI string or {"question": "..."} JSON payload."""
    if isinstance(payload, dict):
        question = str(payload.get("question", "")).strip()
        return {**payload, "question": question}
    return {"question": str(payload or "").strip()}


def retriever(corpus_dir: str = DEFAULT_CORPUS_DIR, top_k: int = 4,
              chunk_chars: int = 900, overlap: int = 120,
              patterns: Iterable[str] = DEFAULT_PATTERNS) -> Step:
    """Preset-friendly factory: docs directory -> chunks -> retriever step."""
    chunks = load_chunks(corpus_dir, chunk_chars=chunk_chars,
                         overlap=overlap, patterns=patterns)
    return retriever_step(FileKeywordRetriever(chunks), top_k=top_k, name="retrieve")


prompt: Step = prompt_step(
    "Use only the cited context to answer the internal docs question.\n"
    "Every factual answer line must end with an [eN] citation.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}",
    cite="context",
)


def llm(payload, ctx: RunContext) -> dict:
    """Deterministic fake LLM that emits citation markers for the demo."""
    context = payload.get("context", []) if isinstance(payload, dict) else []
    question = str(payload.get("question", "")) if isinstance(payload, dict) else ""
    ledger = evidence_ledger(ctx)
    lines: list[str] = []
    for item in context[:3]:
        text = str(item).strip()
        if not text:
            continue
        evidence_id = ledger.id_for(text) or ledger.add(text, step="llm").id
        lines.append(f"{best_sentence(text, question)} [{evidence_id}]")

    if not lines:
        lines = ["No matching documentation found."]

    answer = "\n".join(lines)
    prompt_text = str(payload.get("prompt", "")) if isinstance(payload, dict) else ""
    ctx.metric("llm.calls")
    ctx.metric("llm.input_tokens", len(prompt_text.split()))
    ctx.metric("llm.output_tokens", len(answer.split()))
    return {**payload, "answer": answer}


def lexical_embedder(text: str, dimensions: int = 64) -> list[float]:
    """Tiny deterministic embedder for SemanticCache in the example preset."""
    vector = [0.0] * dimensions
    for term in terms(text):
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=2).digest()
        vector[int.from_bytes(digest, "big") % dimensions] += 1.0
    return vector


class FileKeywordRetriever:
    """Keyword-overlap retriever over pre-built EvidenceChunk objects."""

    def __init__(self, chunks: Iterable[EvidenceChunk]):
        self.chunks = list(chunks)

    def retrieve(self, query: str) -> list[EvidenceChunk]:
        query_terms = set(terms(query))
        scored: list[tuple[float, int, EvidenceChunk]] = []
        for index, chunk in enumerate(self.chunks):
            chunk_terms = set(terms(chunk.text))
            if not chunk_terms:
                continue
            overlap = query_terms & chunk_terms
            if not overlap:
                continue
            score = len(overlap) / max(len(query_terms), 1)
            scored.append((score, index, EvidenceChunk(
                text=chunk.text,
                source=chunk.source,
                span=chunk.span,
                score=round(score, 4),
            )))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [chunk for _, _, chunk in scored]


def load_chunks(corpus_dir: str | Path, chunk_chars: int = 900,
                overlap: int = 120,
                patterns: Iterable[str] = DEFAULT_PATTERNS) -> list[EvidenceChunk]:
    root = resolve_corpus_dir(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"docs corpus directory not found: {root}")

    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in root.glob(pattern)
                     if path.is_file() and not is_hidden_path(path.relative_to(root)))
    paths = sorted(set(paths))

    chunks: list[EvidenceChunk] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for start, end, chunk_text in chunk_markdown(text, chunk_chars, overlap):
            chunks.append(EvidenceChunk(
                text=chunk_text,
                source={"path": str(path.relative_to(root)),
                        "title": path.stem,
                        "vault": root.name},
                span=(start, end),
            ))
    return chunks


def resolve_corpus_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    examples_dir = Path(__file__).resolve().parent
    repo_root = examples_dir.parent
    for base in (Path.cwd(), repo_root, examples_dir):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path.cwd() / candidate).resolve()


def is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def chunk_markdown(text: str, chunk_chars: int, overlap: int) -> list[tuple[int, int, str]]:
    paragraphs = [match for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)",
                                                 text, flags=re.S)]
    chunks: list[tuple[int, int, str]] = []
    current: list[str] = []
    start: int | None = None
    end = 0

    def flush() -> None:
        nonlocal current, start, end
        if current and start is not None:
            chunk = "\n\n".join(current).strip()
            if chunk:
                chunks.append((start, end, chunk))
        current = []
        start = None

    for match in paragraphs:
        paragraph = match.group(0).strip()
        if not paragraph:
            continue
        next_len = sum(len(part) + 2 for part in current) + len(paragraph)
        if current and next_len > chunk_chars:
            flush()
        if start is None:
            start = match.start()
        current.append(paragraph)
        end = match.end()
    flush()

    if overlap <= 0 or len(chunks) < 2:
        return chunks

    with_overlap: list[tuple[int, int, str]] = []
    previous_tail = ""
    for start, end, chunk in chunks:
        text_with_tail = f"{previous_tail}\n\n{chunk}".strip() if previous_tail else chunk
        with_overlap.append((start, end, text_with_tail))
        previous_tail = chunk[-overlap:]
    return with_overlap


def terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in WORD_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOP_WORDS
    ]


def best_sentence(text: str, query: str, limit: int = 220) -> str:
    cleaned = " ".join(line.strip("#- ") for line in text.splitlines() if line.strip())
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned)
                 if part.strip()]
    if not sentences:
        sentence = cleaned
    else:
        query_terms = set(terms(query))
        sentence = max(
            sentences,
            key=lambda candidate: len(query_terms & set(terms(candidate))),
        )
    if len(sentence) <= limit:
        return sentence
    return sentence[: limit - 1].rstrip() + "."
