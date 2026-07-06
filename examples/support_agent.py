"""Example components for the `support-agent` preset.

This is not a tool-loop agent in the core. If you already have an LLM <-> tools
agent, wrap it with ``tl.wrap(...)`` and put Policy/Quota/Validate around it.
This preset shows the lighter pattern throughline can express directly:

    normalize -> classify intent -> branch FAQ/RAG/escalate -> format

The policy layer guards the run perimeter, while Quota provides a graceful
"handing you to an operator" fallback when the budget is spent.
"""

from __future__ import annotations

import re
from typing import Any

from throughline.context import RunContext
from throughline.modules.policy import Transform, screen_with

FAQ = {
    "password": "Open Settings -> Security -> Reset password. The reset link is valid for 1 hour.",
    "refund": "Refunds are processed within 5 business days after the return is confirmed.",
    "shipping": "Standard shipping takes 3-7 days. Tracking arrives by email.",
    "cancel": "You can cancel an order until it ships from Orders -> Details -> Cancel.",
}

KNOWLEDGE_BASE = [
    {
        "topic": "sla",
        "text": "Enterprise support includes 99.9% uptime targets and priority incident review.",
    },
    {
        "topic": "billing",
        "text": "Billing disputes require account verification and are reviewed by a human specialist.",
    },
    {
        "topic": "api",
        "text": "API rate limits reset every minute; enterprise plans can request higher limits.",
    },
    {
        "topic": "security",
        "text": "Security issues should be escalated with timestamps, affected account ids, and impact.",
    },
]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_INJECTION_MARKERS = ("ignore previous", "ignore all", "system prompt",
                      "disregard", "you are now", "developer mode",
                      "reveal your instructions")


def normalize(payload, ctx: RunContext) -> dict:
    """Accept CLI strings and the real {"message", "history", "user_id"} shape."""
    if isinstance(payload, dict):
        message = str(payload.get("message") or payload.get("question") or "").strip()
        history = payload.get("history") or []
        if not isinstance(history, list):
            history = [str(history)]
        return {"message": message, "history": history,
                "user_id": str(payload.get("user_id", "anonymous"))}
    return {"message": str(payload or "").strip(), "history": [], "user_id": "anonymous"}


def classify_intent(payload, ctx: RunContext) -> dict:
    """Offline classifier stand-in. It reports LLM-like metrics for Quota."""
    message = payload["message"].lower()
    intent = "escalate"
    if any(word in message for word in ("human", "operator", "agent", "lawsuit")):
        intent = "escalate"
    elif any(topic in message for topic in FAQ):
        intent = "faq"
    elif any(row["topic"] in message for row in KNOWLEDGE_BASE):
        intent = "rag"
    elif any(word in message for word in ("account", "invoice", "api", "security", "sla")):
        intent = "rag"

    ctx.metric("llm.calls")
    ctx.metric("llm.input_tokens", len(payload["message"].split()) + history_tokens(payload))
    ctx.metric("llm.output_tokens", 1)
    ctx.metric(f"support.intent.{intent}")
    return {**payload, "intent": intent}


def faq_answer(payload, ctx: RunContext) -> dict:
    message = payload["message"].lower()
    topic, answer = next(
        ((topic, text) for topic, text in FAQ.items() if topic in message),
        ("general", "I do not have a matching FAQ entry."),
    )
    return {**payload, "draft_reply": answer, "action": "reply",
            "route": "faq", "topic": topic}


def rag_answer(payload, ctx: RunContext) -> dict:
    message = payload["message"]
    hits = retrieve_knowledge(message, top_k=2)
    ctx.metric("retrieval.docs", len(hits))
    ctx.metric("llm.calls")
    ctx.metric("llm.input_tokens", len(message.split()) + sum(len(hit.split()) for hit in hits))
    if hits:
        reply = " ".join(hits)
    else:
        reply = "I could not find enough knowledge-base context for this request."
    reply += " I can connect you with a specialist if you need account-specific help."
    ctx.metric("llm.output_tokens", len(reply.split()))
    return {**payload, "draft_reply": reply, "action": "reply", "route": "rag",
            "context": hits}


def escalate(payload, ctx: RunContext) -> dict:
    ctx.metric("support.escalations")
    return {**payload,
            "draft_reply": "I'm connecting you with a human operator for this request.",
            "action": "escalate",
            "route": "escalate"}


def format_response(payload, ctx: RunContext) -> dict:
    """Return the strict public shape. Internal routing data stays internal."""
    action = payload.get("action", "reply")
    reply = str(payload.get("draft_reply") or "I'm connecting you with a human operator.")
    if action == "reply":
        reply = f"{reply} Reply with 'agent' if you want a human handoff."
    return {"reply": reply, "action": action}


def retrieve_knowledge(message: str, top_k: int = 2) -> list[str]:
    query = {token.strip(".,!?").lower() for token in message.split() if len(token) > 2}
    scored: list[tuple[int, str]] = []
    for row in KNOWLEDGE_BASE:
        words = set(row["text"].lower().replace(".", "").split()) | {row["topic"]}
        score = len(query & words)
        if score:
            scored.append((score, row["text"]))
    scored.sort(key=lambda item: -item[0])
    return [text for _, text in scored[:top_k]]


def history_tokens(payload: dict) -> int:
    return sum(len(str(item).split()) for item in payload.get("history", []))


def injection_judge(claim: str, evidence: list[str]) -> float:
    """kind="verifier": deterministic prompt-injection screening stand-in."""
    text = " ".join(evidence).lower()
    hits = sum(marker in text for marker in _INJECTION_MARKERS)
    return min(1.0, hits / 2 + 0.5) if hits else 0.0


screen_injection = screen_with(injection_judge, threshold=0.5, key="message")


def redact_email_addresses(checkpoint: str, value: Any, ctx: RunContext):
    """kind="policy": egress redaction for the public reply field."""
    if isinstance(value, dict) and isinstance(value.get("reply"), str):
        reply, count = EMAIL_RE.subn("[email redacted]", value["reply"])
        if count:
            return Transform({**value, "reply": reply},
                             f"redacted {count} email address(es)")
    return None
