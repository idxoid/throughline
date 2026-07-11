"""Policy-based manifest verification: expected lockfile vs observed capture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .diff import diff_expected

Action = Literal["block", "warn", "ignore"]
Gate = Literal["pass", "warn", "block"]

DEFAULT_VERIFY_POLICY: dict[str, Action] = {
    "model": "block",
    "model.id": "block",
    "model.temperature": "block",
    "model.top_p": "block",
    "model.reasoning_effort": "block",
    "prompt": "block",
    "skills": "block",
    "mcp": "block",
    "tools": "block",
    "dependencies": "block",
    "network": "block",
    "repository.dirty": "block",
    "workspace.merkle_root": "block",
    "repository.commit": "warn",
    "repository": "warn",
    "environment": "warn",
    "harness": "ignore",
    "runtime": "ignore",
    "execution": "ignore",
    "workspace": "block",
}


@dataclass(frozen=True, slots=True)
class Violation:
    field: str
    expected: Any
    observed: Any
    action: Action


@dataclass(frozen=True, slots=True)
class VerifyResult:
    gate: Gate
    violations: list[Violation]


def policy_action(field: str, policy: dict[str, Action]) -> Action:
    """Longest dotted-prefix match wins; unknown fields default to warn."""
    parts = field.split(".")
    for size in range(len(parts), 0, -1):
        prefix = ".".join(parts[:size])
        if prefix in policy:
            return policy[prefix]
    return "warn"


def verify_manifest(
    expected: dict[str, Any],
    observed: dict[str, Any],
    policy: dict[str, Action] | None = None,
) -> VerifyResult:
    """Diff expected lockfile against observed capture; apply per-field policy."""
    rules = policy if policy is not None else DEFAULT_VERIFY_POLICY
    violations: list[Violation] = []
    for drift in diff_expected(expected, observed):
        action = policy_action(drift["field"], rules)
        if action == "ignore":
            continue
        violations.append(Violation(
            field=drift["field"],
            expected=drift["expected"],
            observed=drift["observed"],
            action=action,
        ))
    if any(v.action == "block" for v in violations):
        gate: Gate = "block"
    elif violations:
        gate = "warn"
    else:
        gate = "pass"
    return VerifyResult(gate=gate, violations=violations)


def load_lockfile(path: str | Path) -> dict[str, Any]:
    """Load an expected manifest from ``.json`` or ``.toml``."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".toml":
        import tomllib
        return tomllib.loads(text)
    return json.loads(text)
