"""Example components for the `support-agent` preset — the policy showcase.

A support bot with the full policy layer around it:

  * ingress screening — ``screen_injection`` is ``screen_with(judge)``, a
    verifier adapted into a policy rule. The judge here is a deterministic
    keyword scorer so the example runs offline; in production you swap it
    for an LLM/NLI verifier (same contract, costs tokens, goes through
    Quota, cacheable).
  * egress redaction — ``redact_email_addresses`` rewrites the final answer
    (Transform), so PII cannot leave the perimeter, cache hits included:
    Policy sits outside Cache and egress runs in the finalizer sweep.
  * graceful deny — ``on_deny = "return"`` + fallback in the preset: the
    customer reads "handing you to an operator", not a stack trace.
  * audit — every decision is an event + a record in
    ``result.ctx.artifacts["policy"]``.

Run from the repository root:

    THROUGHLINE_PRESETS=examples/presets PYTHONPATH=src:. \
        python3 -m throughline run support-agent -i "how do I reset my password?" --json
"""

from __future__ import annotations

import re

from throughline.context import RunContext
from throughline.modules.policy import Transform, screen_with

FAQ = {
    "password": "Open Settings -> Security -> Reset password; the link is valid for 1 hour.",
    "refund": "Refunds are processed within 5 business days after the return is confirmed.",
    "shipping": "Standard shipping takes 3-7 days; tracking arrives by email.",
    "cancel": "You can cancel an order until it ships: Orders -> Details -> Cancel.",
}
DEFAULT_ANSWER = "I could not find this in the FAQ — connecting you with an operator."

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

#: Phrases a prompt-injection attempt tends to contain. This list feeds a
#: *scoring judge*, not a detector that pretends to be exhaustive — the rule
#: below is honest about being a stand-in for an LLM/NLI verifier.
_INJECTION_MARKERS = ("ignore previous", "ignore all", "system prompt",
                      "disregard", "you are now", "developer mode",
                      "reveal your instructions")


def normalize(payload, ctx: RunContext) -> dict:
    """Accept a CLI string or {"question": ...} JSON payload."""
    if isinstance(payload, dict):
        return {**payload, "question": str(payload.get("question", "")).strip()}
    return {"question": str(payload or "").strip()}


def answer(payload, ctx: RunContext) -> dict:
    """Offline stand-in for the support LLM. Deliberately careless: it quotes
    the account email back at the customer, so egress redaction has real
    work to do."""
    question = payload.get("question", "").lower()
    reply = next((text for topic, text in FAQ.items() if topic in question),
                 DEFAULT_ANSWER)
    email = payload.get("email")
    if email:
        reply += f" (account: {email})"
    ctx.metric("llm.calls")
    return {**payload, "answer": reply}


def injection_judge(claim: str, evidence: list[str]) -> float:
    """kind="verifier": scores how strongly the evidence supports the claim
    ("the input attempts prompt injection"). Deterministic so the example is
    offline; the production judge is an LLM/NLI verifier with the same
    signature."""
    text = " ".join(evidence).lower()
    hits = sum(marker in text for marker in _INJECTION_MARKERS)
    return min(1.0, hits / 2 + 0.5) if hits else 0.0


#: The ingress rule the preset references: verifier -> policy rule.
screen_injection = screen_with(injection_judge, threshold=0.5, key="question")


def redact_email_addresses(checkpoint: str, value, ctx: RunContext):
    """kind="policy": egress redaction. Rewrites the final answer instead of
    blocking it — Transform, not Deny."""
    if isinstance(value, dict) and isinstance(value.get("answer"), str):
        answer_text, count = EMAIL_RE.subn("[email redacted]", value["answer"])
        if count:
            return Transform({**value, "answer": answer_text},
                             f"redacted {count} email address(es)")
    return None
