"""Convert Claude Code / Cursor / Codex transcripts to neutral JSONL.

Neutral event types (one JSON object per line)::

    session_start | user | assistant | tool_call | tool_result | session_end

These feed ``agent-audit`` and ``SessionRecorder`` without each harness
needing a custom audit parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .claude_code import convert_claude_code
from .codex import convert_codex
from .common import read_jsonl, write_jsonl
from .cursor import convert_cursor

FormatName = Literal["auto", "claude-code", "cursor", "codex", "neutral"]


def detect_format(raw: list[dict[str, Any]]) -> FormatName:
    """Best-effort format detection from the first few events."""
    if not raw:
        return "neutral"
    sample = raw[:12]
    types = {row.get("type") for row in sample}
    roles = {row.get("role") for row in sample}

    if types & {"thread.started", "turn.started", "item.completed", "item.started"}:
        return "codex"
    if types & {"session_meta", "response_item", "turn_context"}:
        return "codex"
    if types & {"queue-operation", "file-history-snapshot", "ai-title"}:
        return "claude-code"
    if any(row.get("sessionId") and row.get("type") in {"user", "assistant", "system"}
           for row in sample):
        return "claude-code"
    if roles & {"user", "assistant"} and any(
        isinstance((row.get("message") or {}).get("content"), list) for row in sample
        if isinstance(row.get("message"), dict)
    ):
        return "cursor"
    if types & {"session_start", "tool_call", "session_end"}:
        return "neutral"
    return "neutral"


def convert_events(raw: list[dict[str, Any]],
                   format: FormatName = "auto",
                   *,
                   session_id: str | None = None) -> list[dict[str, Any]]:
    kind = detect_format(raw) if format == "auto" else format
    if kind == "claude-code":
        return convert_claude_code(raw)
    if kind == "cursor":
        return convert_cursor(raw, session_id=session_id)
    if kind == "codex":
        return convert_codex(raw)
    if kind == "neutral":
        return list(raw)
    raise ValueError(f"unknown transcript format: {format!r}")


def convert_file(
    source: str | Path,
    dest: str | Path | None = None,
    *,
    format: FormatName = "auto",
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read a harness transcript, convert to neutral events, optionally write."""
    path = Path(source)
    raw = read_jsonl(path)
    if session_id is None and path.stem:
        session_id = path.stem
    events = convert_events(raw, format=format, session_id=session_id)
    if dest is not None:
        write_jsonl(dest, events)
    return events


__all__ = [
    "FormatName",
    "convert_claude_code",
    "convert_codex",
    "convert_cursor",
    "convert_events",
    "convert_file",
    "detect_format",
    "read_jsonl",
    "write_jsonl",
]
