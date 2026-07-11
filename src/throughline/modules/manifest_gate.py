"""Ingress middleware: live manifest capture + policy verify before steps run.

Recommended stack position: Observe/Metrics → ManifestGate → Policy → Cache → …
so cache hits cannot bypass environment verification.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from ..context import RunContext
from ..errors import ManifestVerifyError
from ..manifest.verify import DEFAULT_VERIFY_POLICY, VerifyResult, load_lockfile
from ..manifest.session import verify_live
from ..middleware import Middleware

_HARNESS_KEYS = frozenset({
    "model", "harness", "prompt", "skills", "mcp", "tools", "network",
    "dependencies", "execution",
})


class ManifestGate(Middleware):
    name = "manifest_gate"

    def __init__(
        self,
        lockfile: str | None = None,
        expected: dict[str, Any] | None = None,
        policy: dict[str, Literal["block", "warn", "ignore"]] | None = None,
        root: str = ".",
        env_allowlist: list[str] | None = None,
        on_fail: str = "raise",
    ):
        if on_fail not in ("raise", "warn"):
            raise ValueError("on_fail must be 'raise' or 'warn'")
        if expected is None and not lockfile:
            raise ValueError("ManifestGate requires lockfile= or expected=")
        self.lockfile = lockfile
        self.expected = expected
        self.policy = dict(policy) if policy is not None else dict(DEFAULT_VERIFY_POLICY)
        self.root = root
        self.env_allowlist = list(env_allowlist or ())
        self.on_fail = on_fail

    def on_run_start(self, ctx: RunContext, payload: Any) -> Any:
        cfg = payload if isinstance(payload, dict) else {}
        expected = self.expected
        lockfile = self.lockfile
        if expected is None:
            expected = load_lockfile(lockfile)
        harness = _harness_from(cfg)
        observed, result = verify_live(
            harness or {},
            root=self.root,
            env_allowlist=self.env_allowlist,
            expected=expected,
            policy=self.policy,
        )
        ctx.artifacts["manifest"] = {
            "lockfile": lockfile,
            "expected": expected,
            "observed": observed,
            "gate": result.gate if result else "pass",
            "violations": ([asdict(v) for v in result.violations]
                           if result else []),
        }
        ctx.metric("manifest.verify.violations",
                   len(result.violations) if result else 0)
        if result:
            self._record_gate(ctx, result)
            if result.gate == "block":
                self._handle_fail(ctx, result)
            elif result.gate == "warn":
                ctx.metric("manifest.verify.warned", 1)
                ctx.emit("manifest_verify_warn", gate=result.gate,
                           violations=len(result.violations))
        else:
            ctx.metric("manifest.verify.passed", 1)
        return payload

    def _record_gate(self, ctx: RunContext, result: VerifyResult) -> None:
        if result.gate == "pass":
            ctx.metric("manifest.verify.passed", 1)
        elif result.gate == "block":
            ctx.metric("manifest.verify.blocked", 1)

    def _handle_fail(self, ctx: RunContext, result: VerifyResult) -> None:
        ctx.emit("manifest_verify_blocked", violations=len(result.violations))
        summary = "; ".join(
            f"{v.field} ({v.action})" for v in result.violations[:5])
        if self.on_fail == "raise":
            raise ManifestVerifyError(
                f"manifest verify blocked: {summary or 'policy violation'}",
                gate=result.gate,
                violations=[asdict(v) for v in result.violations],
            )
        ctx.artifacts.setdefault("violations", []).append(
            f"manifest: {summary}")


def _harness_from(cfg: dict) -> dict[str, Any]:
    harness = dict(cfg.get("harness_config") or {})
    for key in _HARNESS_KEYS:
        if key in cfg and key not in harness:
            harness[key] = cfg[key]
    return harness
