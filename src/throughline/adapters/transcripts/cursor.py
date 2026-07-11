"""Cursor agent transcript JSONL → neutral throughline transcript events."""

from __future__ import annotations

import re
from typing import Any

from .common import text_blocks

_USER_QUERY_RE = re.compile(
    r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL | re.IGNORECASE)


def convert_cursor(raw: list[dict[str, Any]], *,
                   session_id: str | None = None) -> list[dict[str, Any]]:
    """Map Cursor agent-transcript JSONL (role + message.content blocks).

    Cursor exports often omit ``tool_result`` and tool ids; synthetic
    ``call_id`` values are assigned so audit pairing still has a key.
    """
    events: list[dict[str, Any]] = [{
        "type": "session_start",
        "session_id": session_id or "cursor-session",
        "config": {
            "harness": {"name": "cursor"},
            "model": {},
        },
    }]
    call_seq = 0
    tools_seen: dict[str, dict[str, Any]] = {}

    for row in raw:
        role = row.get("role")
        message = row.get("message") if isinstance(row.get("message"), dict) else {}
        content = message.get("content")
        if role == "user":
            text = _cursor_user_text(text_blocks(content))
            if text:
                events.append({"type": "user", "text": text})
            continue
        if role != "assistant":
            continue
        text = text_blocks(content)
        if text:
            events.append({"type": "assistant", "text": text})
        for block in content or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            call_seq += 1
            name = block.get("name") or "unknown"
            call_id = block.get("id") or f"cursor-call-{call_seq}"
            args = block.get("input") or block.get("arguments") or {}
            events.append({
                "type": "tool_call",
                "call_id": call_id,
                "name": name,
                "args": args,
            })
            tools_seen[name] = {"name": name}

    if tools_seen and events[0].get("type") == "session_start":
        events[0]["config"]["tools"] = {
            name: {"schema_sha256": "from-transcript"}
            for name in sorted(tools_seen)
        }
    events.append({"type": "session_end", "status": "ok", "usage": {}})
    return events


def _cursor_user_text(text: str) -> str:
    match = _USER_QUERY_RE.search(text)
    if match:
        return match.group(1).strip()
    # Strip Cursor timestamp wrappers when present.
    cleaned = re.sub(r"<timestamp>.*?</timestamp>\s*", "", text,
                     flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()
