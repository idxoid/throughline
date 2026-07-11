"""Steps for the agent-preflight preset (live manifest verify, no agent run)."""

from __future__ import annotations

import re
from typing import Any

from throughline.context import RunContext

_SECRET_KEY_RE = re.compile(
    r"(secret|api[_-]?key|password|credential|authorization|"
    r"access[_-]?token|refresh[_-]?token|session[_-]?token|bearer)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"\b(?:sk|key|token)-[A-Za-z0-9._-]{6,}\b|Bearer\s+[A-Za-z0-9._-]{8,}",
    re.IGNORECASE,
)


def render_report(payload, ctx: RunContext) -> dict[str, Any]:
    """Format the manifest artifact written by ManifestGate at ingress."""
    manifest = ctx.artifacts.get("manifest")
    if manifest is None:
        raise RuntimeError("manifest artifact missing — is ManifestGate enabled?")
    public = _scrub(manifest)

    lines = [
        "# Agent preflight",
        "",
        f"Gate: {public['gate']}",
        "",
    ]
    if public.get("lockfile"):
        lines.append(f"Lockfile: {public['lockfile']}")
    for item in public.get("violations") or []:
        lines.append(
            f"- [{item['action']}] {item['field']}:"
            f" expected {item['expected']!r} -> observed {item['observed']!r}")
    if not public.get("violations"):
        lines.append("- no violations")
    lines.extend(["", "## Observed snapshot"])
    observed = public.get("observed") or {}
    if "live" in observed and set(observed) <= {"live", "harness"}:
        lines.append("### Live probe (measured by Throughline)")
        for key in sorted(observed.get("live") or {}):
            lines.append(f"- {key}: {_compact(observed['live'][key])}")
        lines.append("### Harness-attested (adapter-reported)")
        harness = observed.get("harness") or {}
        if harness:
            for key in sorted(harness):
                lines.append(f"- {key}: {_compact(harness[key])}")
        else:
            lines.append("- (none)")
    else:
        for key in sorted(observed):
            lines.append(f"- {key}: {_compact(observed[key])}")

    return {
        "gate": public["gate"],
        "violations": public.get("violations") or [],
        "observed": public.get("observed"),
        "expected": public.get("expected"),
        "report": "\n".join(lines),
    }


def _scrub(value: Any, key: str = "") -> Any:
    if _SECRET_KEY_RE.search(key):
        return "[redacted]"
    if isinstance(value, dict):
        return {item_key: _scrub(item_value, str(item_key))
                for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub("[redacted]", value)
    return value


def _compact(value: Any, limit: int = 64) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
