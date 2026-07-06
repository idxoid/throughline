"""LLM adapters.

Core stays dependency-free: `anthropic_chat` imports the `anthropic` SDK
lazily at call time (pip install throughline[anthropic]); `from_callable` turns
any completion function into a step; `FakeLLM` powers offline demos/tests.

Payload convention for chat steps: a plain string is treated as the user
prompt; a dict may carry {"prompt": ...} or {"messages": [...]} plus anything
else — the step returns the dict with an added "answer" key so downstream
steps keep the full context (and lineage can extract any field).
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from ..context import RunContext
from ..step import Step

DEFAULT_MODEL = "claude-opus-4-8"


def _extract_prompt(payload: Any) -> tuple[str | None, list | None, dict | None]:
    """-> (prompt, messages, container_dict)"""
    if isinstance(payload, str):
        return payload, None, None
    if isinstance(payload, dict):
        if "messages" in payload:
            return None, payload["messages"], payload
        prompt = payload.get("prompt") or payload.get("question") or payload.get("input")
        return prompt, None, payload
    return str(payload), None, None


def _package(container: dict | None, answer: str) -> Any:
    if container is None:
        return answer
    return {**container, "answer": answer}


def from_callable(complete: Callable[[str], str], name: str = "llm") -> Step:
    """Any ``prompt -> text`` function becomes an LLM step."""
    def fn(payload, ctx: RunContext):
        prompt, messages, container = _extract_prompt(payload)
        text = complete(prompt if prompt is not None else str(messages))
        return _package(container, text)
    return Step(fn=fn, name=name, meta={"adapter": "from_callable"})


def anthropic_chat(model: str = DEFAULT_MODEL, system: str | None = None,
                   max_tokens: int = 16000, name: str = "llm",
                   **params: Any) -> Step:
    """Claude chat step (lazy import; requires `pip install throughline[anthropic]`).

    Uses the Messages API; records token usage into run metrics
    (llm.input_tokens / llm.output_tokens / llm.calls) when MetricsMiddleware
    is installed. Extra ``params`` (e.g. thinking={"type": "adaptive"},
    output_config={"effort": "high"}) are passed through to messages.create.
    """
    client_holder: dict[str, Any] = {}

    def fn(payload, ctx: RunContext):
        if "client" not in client_holder:
            import anthropic  # deferred so the core stays zero-dependency
            client_holder["client"] = anthropic.Anthropic()
        client = client_holder["client"]

        prompt, messages, container = _extract_prompt(payload)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        request: dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                                   "messages": messages, **params}
        if system:
            request["system"] = system
        response = client.messages.create(**request)

        ctx.metric("llm.calls")
        usage = getattr(response, "usage", None)
        if usage is not None:
            ctx.metric("llm.input_tokens", getattr(usage, "input_tokens", 0) or 0)
            ctx.metric("llm.output_tokens", getattr(usage, "output_tokens", 0) or 0)
        text = "".join(block.text for block in response.content
                       if getattr(block, "type", "") == "text")
        return _package(container, text)

    return Step(fn=fn, name=name, meta={"adapter": "anthropic", "model": model})


class FakeLLM:
    """Deterministic offline LLM for demos and tests.

    Produces a multi-line answer grounded in the payload's context so that
    lineage has interesting material: context lines are partially carried
    through, framing lines are generated.
    """

    def __init__(self, style: str = "qa"):
        self.style = style

    def complete(self, prompt: str) -> str:
        digest = hashlib.sha1(prompt.encode()).hexdigest()[:8]
        return f"[fake-llm {digest}] {prompt[:60]}"

    def answer_step(self, name: str = "llm") -> Step:
        def fn(payload, ctx: RunContext):
            prompt, _, container = _extract_prompt(payload)
            context_lines = []
            if isinstance(container, dict):
                context = container.get("context", [])
                context_lines = context if isinstance(context, list) else lines(context)
            question = (container or {}).get("question", prompt) if container else prompt
            body = [f"Answer to: {question}"]
            for index, line in enumerate(context_lines[:3], 1):
                body.append(f"  [{index}] {line}")
            body.append("Confidence: high (fake)")
            answer = "\n".join(body)
            ctx.metric("llm.calls")
            ctx.metric("llm.output_tokens", sum(len(l.split()) for l in body))
            return _package(container, answer)
        return Step(fn=fn, name=name, meta={"adapter": "fake"})


def lines(text: Any) -> list[str]:
    return str(text).splitlines()
