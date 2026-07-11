"""Recursive dotted-path diff for manifest trees."""

from __future__ import annotations

from typing import Any


def diff_tree(base: dict, cand: dict, path: str = "") -> list[dict[str, Any]]:
    """Symmetric diff — union of keys on both sides (audit/post-hoc)."""
    out: list[dict[str, Any]] = []
    for key in sorted(set(base) | set(cand)):
        sub_path = f"{path}.{key}" if path else str(key)
        before, after = base.get(key), cand.get(key)
        if isinstance(before, dict) and isinstance(after, dict):
            out.extend(diff_tree(before, after, sub_path))
        elif before != after:
            out.append({"field": sub_path, "expected": before, "observed": after})
    return out


def diff_expected(expected: dict, observed: dict, path: str = "") -> list[dict[str, Any]]:
    """Asymmetric diff — only keys declared in ``expected`` are checked."""
    out: list[dict[str, Any]] = []
    for key in sorted(expected):
        sub_path = f"{path}.{key}" if path else str(key)
        exp_val = expected[key]
        obs_val = observed.get(key)
        if isinstance(exp_val, dict) and isinstance(obs_val, dict):
            out.extend(diff_expected(exp_val, obs_val, sub_path))
        elif exp_val != obs_val:
            out.append({"field": sub_path, "expected": exp_val, "observed": obs_val})
    return out
