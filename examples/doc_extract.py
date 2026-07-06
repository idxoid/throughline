"""Example components for the `doc-extract` preset.

The parser is deliberately a slot in the preset: real PDF parsing belongs in a
user plugin. The default here reads a text artifact/path and splits it into
page payloads so the rest of the extraction flow is runnable with stdlib only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from throughline.context import RunContext
from throughline.modules import structured_step
from throughline.step import Step
from throughline.store import ArtifactRef, MemoryArtifactStore

DEFAULT_DOCUMENT = "examples/documents/invoice.txt"
STORE = MemoryArtifactStore(default_ttl=None)

PAGE_SCHEMA = {
    "type": "object",
    "required": ["page", "fields", "entities"],
    "properties": {
        "page": {"type": "integer"},
        "fields": {
            "type": "object",
            "required": ["invoice_number", "date", "total", "currency"],
            "properties": {
                "invoice_number": {"type": ["string", "null"]},
                "date": {"type": ["string", "null"]},
                "total": {"type": ["number", "null"]},
                "currency": {"type": ["string", "null"]},
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "value"],
                "properties": {
                    "type": {"type": "string"},
                    "value": {"type": "string"},
                },
            },
        },
    },
}


def seed_document(path: str = DEFAULT_DOCUMENT, *, session: str = "examples") -> ArtifactRef:
    """Put the demo document into the example artifact store and return a ref."""
    document_path = resolve_path(path)
    text = document_path.read_text(encoding="utf-8")
    return STORE.put(text, session=session, key=document_path.name,
                     meta={"path": str(document_path)})


def parse_text_document(payload, ctx: RunContext) -> list[dict]:
    """Default parser slot: ArtifactRef/path/text -> page payloads."""
    if not isinstance(payload, dict):
        payload = {"document": payload}
    document = payload.get("document", DEFAULT_DOCUMENT)
    text, source = materialize_document(document)
    pages = split_pages(text)
    ctx.metric("doc.pages", len(pages))
    ctx.metric("doc.chars", len(text))
    return [
        {"document": source, "page": index, "text": page}
        for index, page in enumerate(pages, start=1)
    ]


def extract_page(fail_first: bool = True) -> Step:
    """Factory for the map step: page -> typed extraction JSON."""
    generator = DemoExtractor(fail_first=fail_first)
    structured = structured_step(
        generator,
        key="answer",
        out_key="extraction",
        schema=PAGE_SCHEMA,
        name="extract-page",
    )

    def fn(payload, ctx: RunContext) -> dict:
        output = structured.fn(payload, ctx)
        extraction = output["extraction"]
        return {**extraction, "document": payload["document"]}

    return Step(fn=fn, name="extract-page",
                meta={"adapter": "structured", "schema": "doc-page"})


class DemoExtractor:
    """Deterministic fake LLM; first call can be invalid to exercise Retry.

    This demo object is stateful for the lifetime of one Flow instance and is
    intentionally used with workers=1. Raising workers would share `calls`
    across threads and turn the teaching failure into a race.
    """

    def __init__(self, fail_first: bool = True):
        self.fail_first = fail_first
        self.calls = 0

    def __call__(self, payload, ctx: RunContext) -> dict:
        self.calls += 1
        ctx.metric("llm.calls")
        ctx.metric("llm.input_tokens", len(payload["text"].split()))
        if self.fail_first and self.calls == 1:
            ctx.metric("llm.output_tokens", 4)
            return {**payload, "answer": "not json yet"}
        data = extract_from_text(payload["text"], payload["page"])
        answer = json.dumps(data, ensure_ascii=False, sort_keys=True)
        ctx.metric("llm.output_tokens", len(answer.split()))
        return {**payload, "answer": answer}


def merge_pages(pages: list[dict], ctx: RunContext) -> dict:
    """Merge per-page extractions into the strict final JSON report."""
    fields: dict[str, Any] = {
        "invoice_number": None,
        "date": None,
        "total": None,
        "currency": None,
    }
    entities = []
    document = pages[0]["document"] if pages else ""
    for page in pages:
        for key, value in page["fields"].items():
            if value is not None:
                fields[key] = value
        for entity in page["entities"]:
            entities.append({**entity, "page": page["page"]})

    report = {
        "document": document,
        "page_count": len(pages),
        "fields": fields,
        "entities": entities,
        "pages": [
            {"page": page["page"], "fields": page["fields"],
             "entities": page["entities"]}
            for page in pages
        ],
    }
    ctx.metric("doc.entities", len(entities))
    return report


def extract_from_text(text: str, page: int) -> dict:
    invoice = match_one(r"Invoice\s+#:\s*([A-Z0-9-]+)", text)
    date = match_one(r"Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
    total_match = re.search(r"Total:\s*([A-Z]{3})\s*([0-9]+(?:\.[0-9]+)?)", text)
    currency = total_match.group(1) if total_match else None
    total = float(total_match.group(2)) if total_match else None
    entities = []
    for label, pattern in (
        ("vendor", r"Vendor:\s*(.+)"),
        ("customer", r"Bill To:\s*(.+)"),
        ("line_item", r"Item:\s*(.+)"),
    ):
        for value in re.findall(pattern, text):
            entities.append({"type": label, "value": value.strip()})
    return {
        "page": page,
        "fields": {
            "invoice_number": invoice,
            "date": date,
            "total": total,
            "currency": currency,
        },
        "entities": entities,
    }


def match_one(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def materialize_document(document: Any) -> tuple[str, str]:
    if isinstance(document, ArtifactRef):
        return as_text(STORE.get(document)), document.id
    if isinstance(document, dict) and "$artifact" in document:
        ref = ArtifactRef.from_dict(document)
        return as_text(STORE.get(ref)), ref.id
    if isinstance(document, str):
        path = resolve_path(document)
        if path.is_file():
            return path.read_text(encoding="utf-8"), str(path)
        return document, "inline-document"
    return as_text(document), type(document).__name__


def split_pages(text: str) -> list[str]:
    pages = [part.strip() for part in re.split(r"\n-{3,}\s*page\s*-{3,}\n|\f",
                                               text, flags=re.I)
             if part.strip()]
    return pages or [text]


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value if isinstance(value, str) else str(value)


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    repo_root = Path(__file__).resolve().parent.parent
    for base in (Path.cwd(), repo_root):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path.cwd() / candidate).resolve()
