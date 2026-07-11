"""Claude Code session JSONL → neutral throughline transcript events."""

from __future__ import annotations

import re
from typing import Any

from .common import text_blocks

# Claude Code writes this literal marker as a user turn when a run is stopped
# mid-flight (Esc / Ctrl-C). It is the one reliable session-level failure
# signal on disk — there is no explicit "completed ok" record.
_INTERRUPT_RE = re.compile(r"\[Request interrupted", re.IGNORECASE)


def convert_claude_code(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map Claude Code project JSONL into neutral session events.

    Recognizes ``user`` / ``assistant`` rows with Anthropic content blocks
    (``text``, ``tool_use``, ``tool_result``). Emits a synthetic
    ``session_start`` from the first row that carries ``sessionId``.

    Session-end status is *derived*, not assumed ``ok``: an interruption
    marker yields ``interrupted`` and an ``isApiErrorMessage`` row yields
    ``error``, so a crashed or stopped run is not silently recorded as a
    clean one. Absent any negative signal the status is ``ok`` — Claude Code
    stores no positive completion record.
    """
    events: list[dict[str, Any]] = []
    session_id = None
    harness_meta: dict[str, Any] = {}
    model_id = None
    status = "ok"

    for row in raw:
        kind = row.get("type")
        if kind in ("queue-operation", "attachment", "file-history-snapshot",
                    "ai-title", "last-prompt", "progress"):
            continue

        if row.get("isApiErrorMessage"):
            status = "error"

        if session_id is None and row.get("sessionId"):
            session_id = row["sessionId"]
            harness_meta = {
                "name": "claude-code",
                "version": row.get("version"),
            }
            events.append({
                "type": "session_start",
                "session_id": session_id,
                "ts": row.get("timestamp"),
                "config": {
                    "harness": {k: v for k, v in harness_meta.items() if v},
                    "model": {},
                },
            })

        message = row.get("message") if isinstance(row.get("message"), dict) else {}
        content = message.get("content")
        role = message.get("role") or kind

        if kind == "assistant" and message.get("model") and not model_id:
            model_id = message["model"]
            if events and events[0].get("type") == "session_start":
                events[0]["config"]["model"] = {"id": model_id}

        if kind == "user" or role == "user":
            if status == "ok" and _INTERRUPT_RE.search(text_blocks(content)):
                status = "interrupted"
            _emit_user_side(events, content, row.get("timestamp"))
            continue

        if kind == "assistant" or role == "assistant":
            _emit_assistant_side(events, content, row.get("timestamp"))
            continue

    if events and events[-1].get("type") != "session_end":
        events.append({"type": "session_end", "status": status, "usage": {}})
    return events


def _emit_user_side(events: list[dict[str, Any]], content: Any, ts: Any) -> None:
    text = text_blocks(content)
    tool_results = [
        block for block in (content or [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    for block in tool_results:
        result_text = text_blocks(block.get("content"))
        events.append({
            "type": "tool_result",
            "call_id": block.get("tool_use_id") or block.get("id"),
            "status": "error" if block.get("is_error") else "ok",
            "text": result_text[:4000],
            **({"ts": ts} if ts else {}),
        })
    if text and not tool_results:
        events.append({"type": "user", "text": text, **({"ts": ts} if ts else {})})
    elif text and tool_results:
        # Rare: user text alongside tool results — keep the text too.
        events.append({"type": "user", "text": text, **({"ts": ts} if ts else {})})


def _emit_assistant_side(events: list[dict[str, Any]], content: Any,
                         ts: Any) -> None:
    text = text_blocks(content)
    if text:
        events.append({
            "type": "assistant", "text": text,
            **({"ts": ts} if ts else {}),
        })
    for block in content or []:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        events.append({
            "type": "tool_call",
            "call_id": block.get("id"),
            "name": block.get("name"),
            "args": block.get("input") or {},
            **({"ts": ts} if ts else {}),
        })
