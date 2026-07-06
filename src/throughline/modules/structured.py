"""Structured output: parse (and validate) LLM text into data — retryably.

Three pieces, smallest first:

  parse_json        tolerant text -> data: direct parse, then the first
                    markdown code fence, then a braces-salvage pass
  json_step         standalone parsing step for an existing payload key
  structured_step   generate + parse + validate fused into ONE step

Why the fusion exists (the validate→retry recipe): per-step ``Validate``
runs in ``on_step_end``, which executes AFTER the ``wrap_step`` onion — a
``Retry`` around the LLM step never sees the validation raise, so
"regenerate on invalid JSON" cannot be assembled from Validate + Retry
(pinned in tests/test_structured.py). Regeneration requires the parse or
schema failure to originate INSIDE the retried step:

    flow = tl.Flow(
        [structured_step(llm, schema={...}, name="extract")],
        middleware=[Retry(attempts=3, step="extract")],
    )

Every failed attempt re-runs the *generator*, not just the parser.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..context import RunContext
from ..errors import ValidationError
from ..step import Step, _run_fn, as_step
from .validate import check_schema

_FENCE = re.compile(r"```[\w+-]*\s*\n(.*?)```", re.DOTALL)


def parse_json(text: str) -> Any:
    """Tolerant JSON extraction from model output.

    Tries, in order: the text as-is; the first markdown code fence; the
    outermost ``{...}`` / ``[...]`` slice (prose-wrapped JSON). Raises
    ``ValidationError`` naming what was tried and showing the head of the
    text — this is the exception Retry regenerates on in structured_step.
    """
    candidates = [text.strip()]
    fence = _FENCE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start, stop = text.find(opener), text.rfind(closer)
        if 0 <= start < stop:
            candidates.append(text[start:stop + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    head = text.strip().replace("\n", "\\n")[:120]
    raise ValidationError(
        f"output is not valid JSON (tried direct parse, code fence, "
        f"braces salvage); text starts: {head!r}")


def _check(data: Any, schema: dict | None, where: str) -> None:
    if not schema:
        return
    errors = check_schema(data, schema)
    if errors:
        raise ValidationError(f"{where}: parsed JSON fails schema: "
                              + "; ".join(errors[:5]))


def json_step(key: str = "answer", out_key: str | None = None,
              schema: dict | None = None, on_fail: str = "raise",
              parser: Callable[[str], Any] = parse_json,
              name: str = "json") -> Step:
    """Parse ``payload[key]`` (model text) into data at ``out_key``.

    ``out_key`` defaults to ``key`` (parse in place). ``schema`` (the same
    dialect Validate speaks) checks the parsed value. ``on_fail``:
    "raise" (default) -> ValidationError; "warn" -> record a violation +
    event and pass the payload through unchanged.

    Deterministic: retrying THIS step re-parses the same text. When failure
    should regenerate, use ``structured_step`` instead.
    """
    if on_fail not in ("raise", "warn"):
        raise ValueError(f"on_fail must be 'raise' or 'warn', got {on_fail!r}")
    target = out_key or key

    def fn(payload, ctx: RunContext):
        text = payload[key] if isinstance(payload, dict) else str(payload)
        try:
            data = parser(text)
            _check(data, schema, name)
        except ValidationError as exc:
            ctx.metric("json.invalid")
            ctx.emit("json_invalid", step=name, error=str(exc))
            if on_fail == "raise":
                raise
            ctx.artifacts.setdefault("violations", []).append(f"{name}: {exc}")
            return payload
        ctx.metric("json.parsed")
        if isinstance(payload, dict):
            return {**payload, target: data}
        return data

    return Step(fn=fn, name=name, meta={"adapter": "json"})


def structured_step(generator: Any, key: str = "answer",
                    out_key: str | None = None, schema: dict | None = None,
                    parser: Callable[[str], Any] = parse_json,
                    name: str | None = None) -> Step:
    """Fuse generator + parse + schema check into one retryable step.

    ``generator`` is anything step-like (an LLM step, a wrapped client, a
    plain function). Its output's ``key`` is parsed with ``parser`` and
    checked against ``schema``; any failure raises ``ValidationError`` from
    INSIDE this step, so ``Retry(step=<this name>)`` re-runs the generation
    — the whole point (see module docstring).
    """
    gen = as_step(generator)
    step_name = name or f"structured({gen.name})"
    target = out_key or key

    def fn(payload, ctx: RunContext):
        output = _run_fn(gen.fn, payload, ctx)
        text = output[key] if isinstance(output, dict) else str(output)
        data = parser(text)
        _check(data, schema, step_name)
        ctx.metric("json.parsed")
        if isinstance(output, dict):
            return {**output, target: data}
        return data

    return Step(fn=fn, name=step_name,
                meta={"adapter": "structured", "generator": gen.name})
