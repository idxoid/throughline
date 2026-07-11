"""Steps for the agent-preflight preset (live manifest verify, no agent run)."""

from __future__ import annotations

from typing import Any

from throughline.context import RunContext
from throughline.manifest.sanitize import redact_secrets


def render_report(payload, ctx: RunContext) -> dict[str, Any]:
    """Format the manifest artifact written by ManifestGate at ingress."""
    manifest = ctx.artifacts.get("manifest")
    if manifest is None:
        raise RuntimeError("manifest artifact missing — is ManifestGate enabled?")
    public = redact_secrets(manifest)

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


def _compact(value: Any, limit: int = 64) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
