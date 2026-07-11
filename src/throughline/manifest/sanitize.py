"""Sanitize manifests before they hit durable audit surfaces (JSONL, reports)."""

from __future__ import annotations

import re
from typing import Any

from .capture import HARNESS_KEYS, LIVE_KEYS

# Top-level keys safe to persist for post-hoc audit / drift.
AUDIT_KEYS = frozenset(HARNESS_KEYS | LIVE_KEYS)

_SECRET_KEY_RE = re.compile(
    r"(secret|api[_-]?key|password|credential|authorization|"
    r"access[_-]?token|refresh[_-]?token|session[_-]?token|bearer)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"\b(?:sk|key|token)-[A-Za-z0-9._-]{6,}\b|Bearer\s+[A-Za-z0-9._-]{8,}",
    re.IGNORECASE,
)

REDACTED = "[redacted]"


def redact_secrets(value: Any, key: str = "") -> Any:
    """Recursively scrub secret-like keys and token-shaped string values."""
    if key and _SECRET_KEY_RE.search(key):
        return REDACTED
    if isinstance(value, dict):
        return {item_key: redact_secrets(item_value, str(item_key))
                for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub(REDACTED, value)
    return value


def sanitize_for_audit(declared: dict[str, Any]) -> dict[str, Any]:
    """Allowlist manifest fields, then redact nested secrets.

    Unknown top-level keys (``api_key``, ``authorization``, …) are dropped
    so they never reach ``session_start.config`` JSONL. Nested secret-shaped
    fields inside allowlisted sections are replaced with ``[redacted]``.
    """
    allowlisted = {
        key: value for key, value in declared.items()
        if key in AUDIT_KEYS
    }
    return redact_secrets(allowlisted)
