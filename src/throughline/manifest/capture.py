"""Probe the live workspace into the same manifest shape agent sessions record."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv",
})

# Fields the harness may declare (agent config the OS cannot observe).
HARNESS_KEYS = frozenset({
    "model",
    "harness",
    "prompt",
    "skills",
    "mcp",
    "tools",
    "network",
    "dependencies",
    "execution",
})

# Fields always measured from the live workspace / process — never from harness.
LIVE_KEYS = frozenset({
    "repository",
    "runtime",
    "workspace",
    "environment",
})

# Top-level sections of a provenance-structured capture.
PROVENANCE_SECTIONS = frozenset({"live", "harness"})

SOURCE_LIVE_PROBE = "live_probe"
SOURCE_HARNESS = "harness"


def env_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def git_snapshot(root: Path) -> dict[str, Any]:
    """Best-effort repository facts; marks dirty when not a git checkout."""
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

    if run("rev-parse").returncode != 0:
        return {"commit": None, "branch": None, "dirty": True}

    commit = run("rev-parse", "HEAD").stdout.strip()
    branch = run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    dirty = bool(run("status", "--porcelain").stdout.strip())
    return {
        "commit": commit[:8] if commit else None,
        "branch": branch or None,
        "dirty": dirty,
    }


def runtime_snapshot() -> dict[str, str]:
    return {
        "os": platform.system().lower(),
        "arch": platform.machine(),
        "python": ".".join(map(str, sys.version_info[:3])),
    }


def env_hashes(allowlist: Sequence[str],
               environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Whitelisted environment variables as name -> value-hash (never raw)."""
    source = environ if environ is not None else os.environ
    return {name: env_hash(source[name])
            for name in allowlist
            if name in source}


def workspace_merkle_root(root: Path) -> str:
    """Content fingerprint of the workspace tree (input snapshot, not output)."""
    git_files = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if git_files.returncode == 0:
        paths = [root / item.decode("utf-8", "surrogateescape")
                 for item in git_files.stdout.split(b"\0")
                 if item]
        return _hash_files(root, paths)

    return _hash_files(root, sorted(root.rglob("*")))


def _hash_files(root: Path, paths: Iterable[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        parts.append(f"{rel.as_posix()}:{digest}")
    combined = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"m-{combined}"


def flatten_observed(observed: Mapping[str, Any]) -> dict[str, Any]:
    """Merge provenance sections into a flat manifest for lockfile verify.

    Structured captures use ``{"live": ..., "harness": ...}``. Flat (legacy
    or hand-built) dicts are returned unchanged so ``verify_manifest`` keeps
    working on either shape.
    """
    if (
        "live" in observed
        and set(observed) <= PROVENANCE_SECTIONS
        and isinstance(observed.get("live"), Mapping)
    ):
        flat = dict(observed["live"])
        harness = observed.get("harness") or {}
        if isinstance(harness, Mapping):
            _reject_harness_live_keys(harness)
            flat.update({key: value for key, value in harness.items()
                         if key in HARNESS_KEYS})
        return flat
    return dict(observed)


def observed_sources(observed: Mapping[str, Any]) -> dict[str, str]:
    """Map each top-level field to ``live_probe`` or ``harness``."""
    if "live" in observed and set(observed) <= PROVENANCE_SECTIONS:
        sources: dict[str, str] = {}
        live = observed.get("live") or {}
        harness = observed.get("harness") or {}
        if isinstance(live, Mapping):
            sources.update({key: SOURCE_LIVE_PROBE for key in live})
        if isinstance(harness, Mapping):
            _reject_harness_live_keys(harness)
            sources.update({key: SOURCE_HARNESS for key in harness
                            if key in HARNESS_KEYS})
        return sources
    # Flat capture: classify by known key sets.
    return {
        key: (SOURCE_LIVE_PROBE if key in LIVE_KEYS else SOURCE_HARNESS)
        for key in observed
        if key in LIVE_KEYS or key in HARNESS_KEYS
    }


def capture_environment(
    root: str | Path = ".",
    *,
    harness: dict[str, Any] | None = None,
    env_allowlist: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a provenance-structured observed manifest.

    Returns::

        {
            "live": {repository, runtime, workspace, environment},
            "harness": {model, prompt, mcp, tools, ...},  # attested only
        }

    Throughline measures ``live`` directly. Harness fields are attested by
    the adapter — the OS cannot observe effective model/tools/prompts.
    Live-observed keys in ``harness`` are rejected so declared config
    cannot spoof repository / workspace / runtime / environment.

    Use ``flatten_observed`` when comparing against a flat lockfile.
    """
    root = Path(root).resolve()
    live: dict[str, Any] = {
        "repository": git_snapshot(root),
        "runtime": runtime_snapshot(),
        "workspace": {"merkle_root": workspace_merkle_root(root)},
        "environment": env_hashes(env_allowlist or (), environ),
    }
    attested: dict[str, Any] = {}
    if harness:
        _reject_harness_live_keys(harness)
        attested = {
            key: value
            for key, value in harness.items()
            if key in HARNESS_KEYS
        }
    return {"live": live, "harness": attested}


def _reject_harness_live_keys(harness: Mapping[str, Any]) -> None:
    collisions = LIVE_KEYS.intersection(harness)
    if collisions:
        raise ValueError(
            f"harness cannot supply live-observed fields: {sorted(collisions)}"
        )
