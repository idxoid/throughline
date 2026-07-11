"""Lockfile IO helpers: capture / update harness-attested agent manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .capture import HARNESS_KEYS, LIVE_KEYS, capture_environment
from .harness import HarnessKind, extract_harness_config
from .sanitize import sanitize_for_audit
from .verify import VerifyResult, load_lockfile
from .session import verify_live


def write_lockfile(path: str | Path, data: dict[str, Any],
                   *, format: Literal["json", "toml"] | None = None) -> Path:
    """Write a lockfile as JSON (default) or TOML based on suffix / format."""
    file_path = Path(path)
    kind = format or ("toml" if file_path.suffix.lower() == ".toml" else "json")
    # Lockfiles store harness-attested expectations, not live probes.
    payload = _lockfile_payload(data)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "toml":
        file_path.write_text(_to_toml(payload), encoding="utf-8")
    else:
        file_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
            encoding="utf-8")
    return file_path


def capture_lockfile(
    path: str | Path,
    *,
    root: str | Path = ".",
    harness: HarnessKind = "auto",
    declared: dict[str, Any] | None = None,
    home: str | Path | None = None,
    include_live_snapshot: bool = False,
) -> dict[str, Any]:
    """Extract harness config (and optionally a live snapshot) and write it.

    By default only ``HARNESS_KEYS`` are written — live fields are measured
    at verify time, not frozen into the lockfile.
    """
    config = dict(declared) if declared else extract_harness_config(
        root, kind=harness, home=home)
    config = _lockfile_payload(config)
    if include_live_snapshot:
        observed = capture_environment(root, harness=config)
        # Side-channel for humans; strip before verify expectations.
        config = dict(config)
        config["_live_snapshot"] = observed.get("live", {})
    write_lockfile(path, {k: v for k, v in config.items() if not k.startswith("_")})
    return config


def update_lockfile(
    path: str | Path,
    *,
    root: str | Path = ".",
    harness: HarnessKind = "auto",
    home: str | Path | None = None,
) -> dict[str, Any]:
    """Refresh harness-attested fields from the current harness."""
    file_path = Path(path)
    existing = load_lockfile(file_path) if file_path.is_file() else {}
    fresh = extract_harness_config(root, kind=harness, home=home)
    merged = {
        key: value for key, value in existing.items()
        if key not in HARNESS_KEYS and key not in LIVE_KEYS
    }
    merged.update(fresh)
    payload = _lockfile_payload(merged)
    write_lockfile(file_path, payload)
    return payload


def verify_lockfile(
    path: str | Path,
    *,
    root: str | Path = ".",
    harness: HarnessKind | None = "auto",
    declared: dict[str, Any] | None = None,
    home: str | Path | None = None,
    env_allowlist: list[str] | None = None,
) -> tuple[dict[str, Any], VerifyResult]:
    """Capture current harness+live state and verify against the lockfile."""
    expected = _lockfile_payload(load_lockfile(path))
    if declared is None:
        if harness is None:
            declared = expected
        else:
            declared = extract_harness_config(root, kind=harness, home=home)
    declared = _lockfile_payload(declared)
    observed, result = verify_live(
        declared,
        root=root,
        expected=expected,
        env_allowlist=env_allowlist or (),
    )
    assert result is not None
    return observed, result


def _lockfile_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Harness-only, audit-safe lockfile payload."""
    sanitized = sanitize_for_audit(data)
    return {key: value for key, value in sanitized.items() if key in HARNESS_KEYS}


def _to_toml(data: dict[str, Any]) -> str:
    """Minimal TOML writer for nested dict lockfiles (stdlib only)."""
    lines: list[str] = []

    def emit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            if prefix:
                lines.append(f"[{prefix}]")
            scalars = {k: v for k, v in value.items() if not isinstance(v, dict)}
            nested = {k: v for k, v in value.items() if isinstance(v, dict)}
            for key, item in scalars.items():
                lines.append(f"{key} = {_toml_scalar(item)}")
            if scalars and nested:
                lines.append("")
            for key, item in nested.items():
                path = f"{prefix}.{key}" if prefix else key
                emit(path, item)
                lines.append("")
        else:
            lines.append(f"{prefix} = {_toml_scalar(value)}")

    emit("", data)
    text = "\n".join(lines).rstrip() + "\n"
    return text


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)
