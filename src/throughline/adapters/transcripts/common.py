"""Shared helpers for harness transcript → neutral throughline JSONL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


NEUTRAL_EVENT_TYPES = frozenset({
    "session_start", "session_end", "user", "assistant",
    "tool_call", "tool_result",
})


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL, skipping blank lines; raises on malformed JSON."""
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        events.append(item)
    return events


def write_jsonl(path: str | Path, events: Iterable[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(dict(event), ensure_ascii=False) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from read_jsonl(path)


def text_blocks(content: Any) -> str:
    """Flatten Anthropic-style content blocks (or a plain string) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, Mapping):
            continue
        if block.get("type") in ("text", "input_text", "output_text") and block.get("text"):
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def schema_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def args_sha256(args: Mapping[str, Any] | Sequence[Any] | None) -> str:
    return schema_sha256(args if args is not None else [])
