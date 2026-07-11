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


def capture_environment(
    root: str | Path = ".",
    *,
    harness: dict[str, Any] | None = None,
    env_allowlist: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the observed manifest for live verify.

    Observable facts come from the workspace and process environment.
    Harness-supplied fields (model, prompt, MCP, tools, …) are merged
    from ``harness`` because the OS cannot infer effective agent config.
    """
    root = Path(root).resolve()
    observed: dict[str, Any] = {
        "repository": git_snapshot(root),
        "runtime": runtime_snapshot(),
        "workspace": {"merkle_root": workspace_merkle_root(root)},
        "environment": env_hashes(env_allowlist or (), environ),
    }
    if harness:
        observed.update(harness)
    return observed
