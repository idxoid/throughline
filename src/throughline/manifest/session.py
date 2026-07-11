"""Harness integration: capture + verify at session start, JSONL recording."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..errors import ManifestVerifyError
from .capture import HARNESS_KEYS, LIVE_KEYS, capture_environment, flatten_observed
from .diff import diff_tree
from .verify import (DEFAULT_VERIFY_POLICY, VerifyResult, load_lockfile,
                     verify_manifest)

_METADATA_KEYS = frozenset({"observed", "verify"})


def declared_config(config: dict[str, Any]) -> dict[str, Any]:
    """Strip capture metadata from a ``session_start.config`` record."""
    return {key: value for key, value in config.items() if key not in _METADATA_KEYS}


def session_start_event(session_id: str, config: dict[str, Any],
                        ts: str | None = None) -> dict[str, Any]:
    return {
        "type": "session_start",
        "session_id": session_id,
        "ts": ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": config,
    }


def verify_live(
    declared: dict[str, Any],
    *,
    root: str | Path = ".",
    env_allowlist: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    lockfile: str | Path | None = None,
    expected: dict[str, Any] | None = None,
    policy: dict[str, Literal["block", "warn", "ignore"]] | None = None,
) -> tuple[dict[str, Any], VerifyResult | None]:
    """Capture the live workspace and optionally verify against a lockfile.

    Declared live-observed fields (repository, workspace, …) are kept for
    audit/drift but never passed as harness input to ``capture_environment``.
    """
    declared = declared_config(declared)
    harness = {
        key: value for key, value in declared.items()
        if key in HARNESS_KEYS
    }
    observed = capture_environment(
        root, harness=harness, env_allowlist=env_allowlist, environ=environ)
    if expected is None and lockfile is None:
        return observed, None
    if expected is None:
        expected = load_lockfile(lockfile)
    rules = dict(policy) if policy is not None else dict(DEFAULT_VERIFY_POLICY)
    result = verify_manifest(expected, flatten_observed(observed), rules)
    return observed, result


def preflight_session_start(
    declared: dict[str, Any],
    *,
    root: str | Path = ".",
    env_allowlist: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    lockfile: str | Path | None = None,
    expected: dict[str, Any] | None = None,
    policy: dict[str, Literal["block", "warn", "ignore"]] | None = None,
    on_block: Literal["raise", "return"] = "raise",
) -> tuple[dict[str, Any], VerifyResult | None]:
    """Build a ``session_start.config`` with ``observed`` (+ ``verify``).

    Harnesses call this before the first tool call. When ``on_block='raise'``
    and verify returns ``block``, raises ``ManifestVerifyError`` and the
    session must not start.
    """
    declared = declared_config(declared)
    observed, result = verify_live(
        declared, root=root, env_allowlist=env_allowlist, environ=environ,
        lockfile=lockfile, expected=expected, policy=policy)
    config = dict(declared)
    config["observed"] = observed
    if result is not None:
        config["verify"] = {
            "gate": result.gate,
            "violations": [asdict(v) for v in result.violations],
        }
        if result.gate == "block" and on_block == "raise":
            summary = "; ".join(
                f"{v.field} ({v.action})" for v in result.violations[:5])
            raise ManifestVerifyError(
                f"manifest verify blocked session start: {summary}",
                gate=result.gate,
                violations=config["verify"]["violations"],
            )
    return config, result


def capture_drift(declared: dict[str, Any],
                  observed: dict[str, Any]) -> list[dict[str, Any]]:
    """Declared manifest vs live capture recorded at session start."""
    return diff_tree(declared_config(declared), flatten_observed(observed))


def effective_environment(manifest: dict[str, Any]) -> dict[str, Any]:
    """Workspace facts for readiness: prefer live probe over declared lies."""
    config = declared_config(manifest.get("config", manifest))
    observed = manifest.get("observed")
    if not observed:
        return config
    if (
        isinstance(observed.get("live"), dict)
        and set(observed) <= {"live", "harness"}
    ):
        probe = observed["live"]
    else:
        probe = observed  # flat legacy captures
    effective = dict(config)
    for key in LIVE_KEYS:
        if key in probe:
            effective[key] = probe[key]
    return effective


class SessionRecorder:
    """Append-only JSONL session transcript writer for agent harnesses."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._active = False

    def start(
        self,
        session_id: str,
        declared: dict[str, Any],
        *,
        ts: str | None = None,
        **preflight_kw: Any,
    ) -> VerifyResult | None:
        config, result = preflight_session_start(declared, **preflight_kw)
        self.append(session_start_event(session_id, config, ts=ts))
        self._active = True
        return result

    def append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def end(self, status: str = "ok", usage: dict[str, Any] | None = None) -> None:
        self.append({
            "type": "session_end",
            "status": status,
            "usage": usage or {},
        })
        self._active = False

    @property
    def active(self) -> bool:
        return self._active
