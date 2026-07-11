"""Codex / Codex-bridge event JSONL → neutral throughline transcript events.

Supports two on-disk dialects:

* **Bridge / app-server** — ``thread.*`` / ``item.*`` / ``turn.*`` (fixtures,
  some SDK streams).
* **CLI / IDE rollout** — ``session_meta`` / ``response_item`` /
  ``event_msg`` / ``turn_context`` as written under ``~/.codex/sessions``.
"""

from __future__ import annotations

import json
from typing import Any

from .common import text_blocks

_ROLLOUT_TYPES = frozenset({
    "session_meta", "response_item", "event_msg", "turn_context",
})


def convert_codex(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map Codex event streams (bridge or rollout) to neutral session events."""
    types = {row.get("type") for row in raw[:24]}
    if types & _ROLLOUT_TYPES:
        return _convert_rollout(raw)
    return _convert_bridge(raw)


def _convert_bridge(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    return _finalize(events, session_id, usage)


def _convert_rollout(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map ``~/.codex/sessions`` rollout JSONL into neutral events.

    User/assistant text comes from ``event_msg`` (cleaner than duplicated
    ``response_item`` messages). Tool calls/results come from
    ``response_item`` ``function_call`` / ``custom_tool_call`` pairs.
    """
    events: list[dict[str, Any]] = []
    session_id = None
    usage: dict[str, Any] = {}
    model: dict[str, Any] = {}
    harness: dict[str, Any] = {"name": "codex"}

    for row in raw:
        kind = row.get("type")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        ts = row.get("timestamp") or payload.get("timestamp")

        if kind == "session_meta":
            session_id = payload.get("id") or session_id
            if payload.get("cli_version"):
                harness["version"] = payload["cli_version"]
            if any(e.get("type") == "session_start" for e in events):
                # Later meta rows (resume / fork) refresh harness only.
                if events and events[0].get("type") == "session_start":
                    events[0]["config"]["harness"] = dict(harness)
                    if session_id:
                        events[0]["session_id"] = session_id
                continue
            events.append({
                "type": "session_start",
                "session_id": session_id or "codex-session",
                "ts": ts,
                "config": {
                    "harness": dict(harness),
                    "model": dict(model),
                },
            })
            continue

        if kind == "turn_context":
            if payload.get("model"):
                model["id"] = payload["model"]
            collab = payload.get("collaboration_mode") or {}
            settings = collab.get("settings") if isinstance(collab, dict) else {}
            if isinstance(settings, dict):
                if settings.get("model") and "id" not in model:
                    model["id"] = settings["model"]
                if settings.get("reasoning_effort"):
                    model["reasoning_effort"] = settings["reasoning_effort"]
            if events and events[0].get("type") == "session_start":
                events[0]["config"]["model"] = dict(model)
                events[0]["config"]["harness"] = dict(harness)
            continue

        if kind == "event_msg":
            msg_type = payload.get("type")
            if msg_type == "user_message":
                text = payload.get("message") or ""
                if text:
                    events.append({
                        "type": "user", "text": text,
                        **({"ts": ts} if ts else {}),
                    })
            elif msg_type == "agent_message":
                text = payload.get("message") or ""
                if text:
                    events.append({
                        "type": "assistant", "text": text,
                        **({"ts": ts} if ts else {}),
                    })
            elif msg_type == "token_count":
                info = payload.get("info") or {}
                total = info.get("total_token_usage") or {}
                if isinstance(total, dict) and total:
                    usage = {
                        key: total[key]
                        for key in (
                            "input_tokens", "output_tokens", "total_tokens",
                            "cached_input_tokens", "reasoning_output_tokens",
                        )
                        if key in total
                    }
            continue

        if kind != "response_item":
            continue

        item_type = payload.get("type")
        if item_type == "function_call":
            events.append({
                "type": "tool_call",
                "call_id": payload.get("call_id") or payload.get("id"),
                "name": payload.get("name") or "function_call",
                "args": _parse_args(payload.get("arguments")),
                **({"ts": ts} if ts else {}),
            })
        elif item_type == "function_call_output":
            events.append({
                "type": "tool_result",
                "call_id": payload.get("call_id") or payload.get("id"),
                "status": "ok",
                "text": str(payload.get("output") or "")[:4000],
                **({"ts": ts} if ts else {}),
            })
        elif item_type == "custom_tool_call":
            events.append({
                "type": "tool_call",
                "call_id": payload.get("call_id") or payload.get("id"),
                "name": payload.get("name") or "custom_tool_call",
                "args": _custom_tool_args(payload),
                **({"ts": ts} if ts else {}),
            })
        elif item_type == "custom_tool_call_output":
            events.append({
                "type": "tool_result",
                "call_id": payload.get("call_id") or payload.get("id"),
                "status": "ok",
                "text": str(payload.get("output") or "")[:4000],
                **({"ts": ts} if ts else {}),
            })

    return _finalize(events, session_id, usage)


def _finalize(
    events: list[dict[str, Any]],
    session_id: str | None,
    usage: dict[str, Any],
) -> list[dict[str, Any]]:
    if not events or events[0].get("type") != "session_start":
        events.insert(0, {
            "type": "session_start",
            "session_id": session_id or "codex-session",
            "config": {"harness": {"name": "codex"}, "model": {}},
        })
    if not events or events[-1].get("type") != "session_end":
        events.append({"type": "session_end", "status": "ok", "usage": usage or {}})
    elif usage and not events[-1].get("usage"):
        events[-1]["usage"] = usage
    return events


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {}


def _custom_tool_args(payload: dict[str, Any]) -> dict[str, Any]:
    if "arguments" in payload:
        return _parse_args(payload.get("arguments"))
    if "input" in payload:
        return {"input": payload.get("input")}
    return {}
