"""Extract harness-attested agent config from Claude Code / Cursor / Codex.

The OS cannot observe effective model, tools, or MCP — these helpers read
what each harness stores on disk and shape it as ``HARNESS_KEYS`` fields for
lockfiles and ``verify_live``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .capture import HARNESS_KEYS, env_hash
from ..adapters.transcripts.common import args_sha256, file_sha256, schema_sha256

HarnessKind = Literal["auto", "claude-code", "cursor", "codex"]


def extract_harness_config(
    root: str | Path = ".",
    *,
    kind: HarnessKind = "auto",
    home: str | Path | None = None,
) -> dict[str, Any]:
    """Build a harness-attested manifest fragment from on-disk harness config.

    Only keys in ``HARNESS_KEYS`` are returned. Missing sources yield partial
    manifests — callers should treat empty sections as "not attested".
    """
    root = Path(root).resolve()
    home_path = Path(home) if home is not None else Path.home()
    detected = detect_harness(root, home=home_path) if kind == "auto" else kind
    if detected == "claude-code":
        return _claude_code_config(root, home_path)
    if detected == "cursor":
        return _cursor_config(root, home_path)
    if detected == "codex":
        return _codex_config(root, home_path)
    return {"harness": {"name": detected or "unknown"}}


def detect_harness(root: str | Path = ".",
                   *,
                   home: str | Path | None = None) -> HarnessKind | None:
    """Prefer project-local markers, then user-level harness homes."""
    root = Path(root).resolve()
    home_path = Path(home) if home is not None else Path.home()
    if (root / "CLAUDE.md").exists() or (root / ".claude").is_dir():
        return "claude-code"
    if (root / ".cursor").is_dir() or (root / ".cursorrules").exists():
        return "cursor"
    if (root / "AGENTS.md").exists() and (home_path / ".codex" / "config.toml").exists():
        return "codex"
    if (home_path / ".claude" / "settings.json").exists():
        return "claude-code"
    if (home_path / ".codex" / "config.toml").exists():
        return "codex"
    if (home_path / ".cursor").is_dir():
        return "cursor"
    return None


def _claude_code_config(root: Path, home: Path) -> dict[str, Any]:
    settings = _load_json(home / ".claude" / "settings.json")
    local = _load_json(root / ".claude" / "settings.local.json")
    project = _load_json(root / ".claude" / "settings.json")
    merged = {**settings, **project, **local}

    # Claude Code chooses the model at runtime and usually omits it from
    # settings.json, so model.id is frequently absent here (unlike Codex's
    # config.toml). When it matters, the session transcript's assistant rows
    # carry the effective model — see adapters/transcripts/claude_code.py.
    # Sampling params (temperature/top_p/max_tokens) are not persisted by any
    # harness and are never captured; pin them by hand in the lockfile.
    model_raw = merged.get("model") or ""
    model_id = str(model_raw).split("[", 1)[0] if model_raw else None
    config: dict[str, Any] = {
        "harness": {"name": "claude-code"},
        "model": {},
        "prompt": {"instructions": {}},
        "mcp": {},
        "tools": {},
        "network": {},
        "execution": {},
    }
    if model_id:
        config["model"]["id"] = model_id
    if merged.get("effortLevel"):
        config["model"]["reasoning_effort"] = merged["effortLevel"]

    for name in ("CLAUDE.md", ".claude/settings.json", ".claude/settings.local.json"):
        path = root / name
        if path.is_file():
            config["prompt"]["instructions"][name] = file_sha256(path)

    # MCP: Claude Code may store servers in settings under mcpServers.
    servers = merged.get("mcpServers") or merged.get("mcp") or {}
    if isinstance(servers, dict):
        config["mcp"] = _mcp_from_servers(servers)

    perms = (local.get("permissions") or project.get("permissions")
             or settings.get("permissions") or {})
    if isinstance(perms, dict) and perms.get("allow"):
        config["tools"] = {
            "permissions": {
                "allow_sha256": schema_sha256(perms.get("allow")),
                "count": len(perms.get("allow") or []),
            }
        }
    return _prune_empty(config)


def _cursor_config(root: Path, home: Path) -> dict[str, Any]:
    config: dict[str, Any] = {
        "harness": {"name": "cursor"},
        "model": {},
        "prompt": {"instructions": {}},
        "mcp": {},
        "tools": {},
    }
    for name in ("AGENTS.md", ".cursorrules", ".cursor/rules"):
        path = root / name
        if path.is_file():
            config["prompt"]["instructions"][name] = file_sha256(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    rel = child.relative_to(root).as_posix()
                    config["prompt"]["instructions"][rel] = file_sha256(child)

    for candidate in (
        root / ".cursor" / "mcp.json",
        home / ".cursor" / "mcp.json",
    ):
        data = _load_json(candidate)
        servers = data.get("mcpServers") or data.get("mcp") or {}
        if isinstance(servers, dict) and servers:
            config["mcp"] = _mcp_from_servers(servers)
            break
    return _prune_empty(config)


def _codex_config(root: Path, home: Path) -> dict[str, Any]:
    path = home / ".codex" / "config.toml"
    data = _load_toml(path)
    config: dict[str, Any] = {
        "harness": {"name": "codex"},
        "model": {},
        "mcp": {},
        "execution": {},
    }
    if data.get("model"):
        config["model"]["id"] = data["model"]
    if data.get("model_reasoning_effort"):
        config["model"]["reasoning_effort"] = data["model_reasoning_effort"]

    servers = data.get("mcp_servers") or {}
    if isinstance(servers, dict):
        normalized = {}
        for name, spec in servers.items():
            if not isinstance(spec, dict):
                continue
            normalized[name] = {
                "command": spec.get("command"),
                "args": spec.get("args") or [],
                "env": {
                    key: env_hash(str(value))
                    for key, value in (spec.get("env") or {}).items()
                },
            }
        config["mcp"] = _mcp_from_servers(normalized)
    return _prune_empty(config)


def _mcp_from_servers(servers: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            continue
        entry: dict[str, Any] = {}
        if spec.get("command"):
            entry["command"] = spec["command"]
        args = spec.get("args")
        if args is not None:
            entry["args_sha256"] = args_sha256(args)
        if spec.get("url"):
            entry["url_sha256"] = schema_sha256(spec["url"])
        # Never store raw env values — hash if present.
        env = spec.get("env")
        if isinstance(env, dict) and env:
            entry["env"] = {
                key: (value if _looks_hashed(value) else env_hash(str(value)))
                for key, value in env.items()
            }
        out[name] = entry
    return out


def _looks_hashed(value: Any) -> bool:
    """True only for a full SHA-256 digest already stored (e.g. codex env,
    pre-hashed in ``_codex_config``), so it is not hashed a second time.

    Deliberately does *not* treat short hex strings as hashed: an 8- or
    12-char hex secret (``deadbeef``) is a real value on the capture path and
    must still be hashed. Only a 64-char lowercase-hex digest is skipped.
    """
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import tomllib
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _prune_empty(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if key not in HARNESS_KEYS:
            continue
        if value in (None, {}, [], ""):
            continue
        if isinstance(value, dict):
            nested = {k: v for k, v in value.items() if v not in (None, {}, [], "")}
            if nested:
                out[key] = nested
        else:
            out[key] = value
    return out
