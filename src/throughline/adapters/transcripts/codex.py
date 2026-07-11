"""Codex / Codex-bridge event JSONL → neutral throughline transcript events."""

from __future__ import annotations

from typing import Any

from .common import text_blocks


def convert_codex(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map Codex thread event streams (``thread.*`` / ``item.*`` / ``turn.*``)."""
    events: list[dict[str, Any]] = []
    session_id = None
    usage: dict[str, Any] = {}

    for row in raw:
        kind = row.get("type")
        if kind == "thread.started":
            session_id = row.get("thread_id") or session_id
            events.append({
                "type": "session_start",
                "session_id": session_id or "codex-session",
                "config": {
                    "harness": {"name": "codex"},
                    "model": {},
                },
            })
            continue

        if kind == "turn.completed":
            usage = row.get("usage") or usage
            continue

        if kind not in ("item.completed", "item.started"):
            continue
        item = row.get("item") if isinstance(row.get("item"), dict) else {}
        item_type = item.get("type")
        if item_type == "agent_message":
            text = item.get("text") or text_blocks(item.get("content"))
            if text:
                events.append({"type": "assistant", "text": text})
            continue
        if item_type == "user_message":
            text = item.get("text") or text_blocks(item.get("content"))
            if text:
                events.append({"type": "user", "text": text})
            continue
        if item_type in ("mcp_tool_call", "command_execution", "tool_call",
                         "function_call"):
            if kind == "item.started" and item.get("status") == "in_progress":
                events.append({
                    "type": "tool_call",
                    "call_id": item.get("id"),
                    "name": item.get("tool") or item.get("name") or item_type,
                    "args": item.get("arguments") or item.get("args") or {},
                })
            elif kind == "item.completed":
                # Ensure a tool_call exists even if started was missing.
                if not any(
                    e.get("type") == "tool_call" and e.get("call_id") == item.get("id")
                    for e in events
                ):
                    events.append({
                        "type": "tool_call",
                        "call_id": item.get("id"),
                        "name": item.get("tool") or item.get("name") or item_type,
                        "args": item.get("arguments") or item.get("args") or {},
                    })
                result = item.get("result")
                text = ""
                if isinstance(result, dict):
                    text = text_blocks(result.get("content")) or str(
                        result.get("structured_content") or "")
                elif result is not None:
                    text = str(result)
                events.append({
                    "type": "tool_result",
                    "call_id": item.get("id"),
                    "name": item.get("tool") or item.get("name") or item_type,
                    "status": "error" if item.get("error") else "ok",
                    "text": (text or "")[:4000],
                })

    if not events or events[0].get("type") != "session_start":
        events.insert(0, {
            "type": "session_start",
            "session_id": session_id or "codex-session",
            "config": {"harness": {"name": "codex"}, "model": {}},
        })
    events.append({"type": "session_end", "status": "ok", "usage": usage or {}})
    return events
