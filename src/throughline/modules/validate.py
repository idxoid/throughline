"""Validation middleware: schema / predicate / pydantic checks with a policy.

    Validate(scope="final", schema={"type": "object", "required": ["answer"]})
    Validate(check=lambda out: "answer" in out)                # scope="final" default
    Validate(step="retrieve", at="output", on_fail="warn")     # one step only
    Validate(scope="step")                                     # every step's output
    Validate(model=MyPydanticModel)                            # if pydantic installed

Scope is a first-class parameter, because it changes what a schema may
require:

  * ``scope="final"`` (default): the FINAL run output only, checked once in
    ``on_run_end``. Intermediate payloads are never inspected, so a schema
    may require fields that only the last step produces (e.g. "answer" in a
    normalize -> answer flow) without failing mid-run.
  * ``scope="step"``: per-step checks — every step matching ``step=``
    (fnmatch pattern; default "*" = all steps), at ``at="output"`` or
    ``at="input"``. The final output is then NOT checked by this instance.
    Passing ``step=`` alone implies ``scope="step"``; different stages with
    different contracts = several Validate instances in the stack.

Conflicting combinations (``scope="final"`` with ``step=`` or ``at="input"``)
are rejected at construction — never silently reinterpreted.

on_fail: "raise" (default) aborts the flow with ValidationError;
         "warn" records the violation (result.violations), emits an event,
         bumps the `validation.failures` metric and lets the run continue.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any, Callable

from ..context import RunContext
from ..errors import ValidationError
from ..middleware import Middleware
from ..step import Step

_TYPE_MAP = {
    "object": dict, "array": list, "string": str,
    "integer": int, "number": (int, float), "boolean": bool,
    "null": type(None),
}


def check_schema(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Minimal JSON-Schema-subset validator (stdlib only).

    Supports: type (str or list), enum, const, properties, required,
    additionalProperties=false, items. If the `jsonschema` package is
    installed, Validate uses it instead of this function.
    """
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        python_types = tuple(_TYPE_MAP[t] for t in types if t in _TYPE_MAP)
        ok = isinstance(value, python_types)
        if ok and isinstance(value, bool) and "boolean" not in types:
            ok = False  # bool is an int subclass; don't accept it for "integer"
        if not ok:
            errors.append(f"{path}: expected {expected}, got {type(value).__name__}")
            return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']!r}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: {value!r} != const {schema['const']!r}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, subschema in properties.items():
            if key in value:
                errors.extend(check_schema(value[key], subschema, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unexpected property {key!r}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(check_schema(item, schema["items"], f"{path}[{index}]"))
    return errors


class Validate(Middleware):
    name = "validate"

    def __init__(self, check: Callable | None = None, schema: dict | None = None,
                 model: Any = None, scope: str | None = None,
                 step: str | None = None, at: str = "output",
                 on_fail: str = "raise", label: str | None = None):
        if at not in ("output", "input"):
            raise ValueError("at must be 'output' or 'input'")
        if on_fail not in ("raise", "warn"):
            raise ValueError("on_fail must be 'raise' or 'warn'")
        if scope is None:
            scope = "step" if step is not None else "final"
        if scope not in ("final", "step"):
            raise ValueError(f"scope must be 'final' or 'step', got {scope!r}")
        if scope == "final" and step is not None:
            raise ValueError(
                "scope='final' checks the run output only — it cannot be "
                "combined with step=; use scope='step' (or drop scope=)")
        if scope == "final" and at == "input":
            raise ValueError(
                "at='input' is a per-step notion; scope='final' checks the "
                "final run output only")
        self.check = check
        self.schema = schema
        self.model = model
        self.scope = scope
        self.step = step if step is not None else ("*" if scope == "step" else None)
        self.at = at
        self.on_fail = on_fail
        self.label = label or (f"step:{self.step}" if scope == "step" else "run-output")

    # -- wiring --------------------------------------------------------------
    def on_step_start(self, ctx: RunContext, step: Step, payload):
        if (self.scope == "step" and self.at == "input"
                and fnmatch(step.name, self.step)):
            self._validate(ctx, payload, where=f"{step.name}.input")
        return payload

    def on_step_end(self, ctx: RunContext, step: Step, payload, output):
        if (self.scope == "step" and self.at == "output"
                and fnmatch(step.name, self.step)):
            self._validate(ctx, output, where=f"{step.name}.output")
        return output

    def on_run_end(self, ctx: RunContext, output):
        if self.scope == "final":
            self._validate(ctx, output, where="run.output")
        return output

    # -- checks ---------------------------------------------------------------
    def _validate(self, ctx: RunContext, value: Any, where: str) -> None:
        errors: list[str] = []
        if self.schema is not None:
            errors.extend(self._check_schema(value))
        if self.model is not None:
            errors.extend(self._check_model(value))
        if self.check is not None:
            errors.extend(self._check_callable(value, ctx))
        if not errors:
            return
        ctx.metric("validation.failures", len(errors))
        ctx.emit("validation_failed", where=where, violations=errors, policy=self.on_fail)
        if self.on_fail == "raise":
            raise ValidationError(
                f"validation failed at {where}: " + "; ".join(errors),
                violations=errors, step=where)
        ctx.artifacts.setdefault("violations", []).extend(f"{where}: {e}" for e in errors)

    def _check_schema(self, value: Any) -> list[str]:
        try:
            import jsonschema  # type: ignore
        except ImportError:
            return check_schema(value, self.schema)
        validator = jsonschema.Draft202012Validator(self.schema)
        return [f"$.{'/'.join(map(str, e.path))}: {e.message}"
                for e in validator.iter_errors(value)]

    def _check_model(self, value: Any) -> list[str]:
        validate = getattr(self.model, "model_validate", None)
        if validate is None:
            return [f"model {self.model!r} has no model_validate (pydantic v2 expected)"]
        try:
            validate(value)
            return []
        except Exception as exc:
            return [str(exc)]

    def _check_callable(self, value: Any, ctx: RunContext) -> list[str]:
        # arity is decided BEFORE the call: an AttributeError raised inside
        # the check itself must not trigger a second (side-effecting) call
        try:
            wants_ctx = self.check.__code__.co_argcount >= 2
        except AttributeError:  # builtins / callables without __code__
            wants_ctx = False
        try:
            result = self.check(value, ctx) if wants_ctx else self.check(value)
        except ValidationError as exc:
            return list(exc.violations)
        except Exception as exc:
            return [f"check raised {exc!r}"]
        if result is None or result is True:
            return []
        if result is False:
            return [f"check {getattr(self.check, '__name__', 'predicate')} returned False"]
        if isinstance(result, tuple):
            ok, message = result
            return [] if ok else [str(message)]
        return [str(result)]
