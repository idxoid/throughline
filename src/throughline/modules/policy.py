"""Policy middleware: ingress screening, egress redaction, audit — the
policy/security layer that was a reserved boundary until now.

A *rule* is the new ``kind="policy"`` component: a callable
``(checkpoint, value, ctx) -> verdict``. Verdicts:

    None                       abstain — "no opinion", evaluation continues
    Allow(reason)              explicitly allow (what authz rules return)
    Deny(reason)               block; first Deny short-circuits the checkpoint
    Transform(value, reason)   substitute the value (redaction) and continue
    Flag(reason)               record for audit, do not block

Abstain is NOT allow — it is a property of the rule ("this rule has nothing
to say here"). Whether silence opens or closes the gate is configuration of
the CHECKPOINT, not discipline of the rule author:

  * ``default="allow"`` (the screening default): value passes unless some
    rule denies. A forgotten ``return`` in a rule cannot block traffic.
  * ``default="deny"`` (the authz posture): at least one rule must return
    an explicit ``Allow`` or the checkpoint denies. A forgotten ``return``
    cannot silently open a hole — the fix lives in the protocol, not in
    code review.

A checkpoint with NO rules installed is inert regardless of ``default``:
the posture governs the outcome of an evaluation, and evaluation happens
only where rules exist — otherwise ``default="deny"`` for ingress authz
would silently kill a rule-less egress too.

Deny always wins: rules run in order, an ``Allow`` is recorded and
evaluation continues, so a later ``Deny`` still blocks. Need fail-open
screening AND fail-closed authz in one flow? Stack two Policy instances —
posture composes like everything else (see Quota's scope pattern).

Checkpoints (phase 1): ``ingress`` rules run in ``on_run_start`` against the
incoming payload; ``egress`` rules run in ``on_run_end`` against the final
output. Because ``on_run_end`` is a finalizer sweep (see the EarlyReturn
contract), a run-level cache hit STILL passes egress — a cached answer
served without redaction is the classic hole, and it is closed by the same
mechanism that makes the sweep run at all. The mirror pin: a PolicyError
from ingress is a real failure, ``on_run_end`` hooks do not run, so a
denied request is never written to any cache.

On deny: ``on_deny="raise"`` (default) aborts with PolicyError (wrapped in
FlowError, reachable via ``err.__cause__``). ``on_deny="return"`` answers
with ``fallback`` (a value or a callable ``(value, ctx) -> value``) instead
of failing — the support bot says "handing you to an operator", not 500.
The mechanics differ by checkpoint and that difference is pinned in tests:
ingress raises EarlyReturn (the Quota pattern — remaining steps skipped,
egress still runs, the fallback is screened too), while egress simply
substitutes the return value — an EarlyReturn raised inside the on_run_end
sweep would be caught as a failure, not as control flow. Note also that a
deny (or any exception) raised from egress interrupts the sweep: outer
middleware (Observe, Metrics) get no ``on_run_end`` — per-event sinks have
already seen every policy event, but run-level finalization is skipped.
That is the documented cost of ``on_deny="raise"`` on egress.

Audit is data, not logs: every non-abstain decision becomes an event
(``policy_denied`` / ``policy_redacted`` / ``policy_flagged`` /
``policy_allowed``), a counter (``policy.denied``, ...) and a record in the
run-wide ``ctx.artifacts["policy"]`` ledger — the evidence/claims pattern.
Events carry rule + reason, never the payload itself (events outlive the
run; redacted data must not leak back through the audit trail).

Transform on INGRESS interacts with a run-level Cache: Policy sits outside
Cache, so the cache key is computed from the transformed payload. If the
transform strips *identifying* fields (user email, order id), requests from
different users collapse into one key and the second user is served the
first user's cached answer. Screening transforms are safe; redaction of
identifying data belongs on egress (where it demonstrably composes with
caching), or the identifying field must be part of the cache ``key=``.

Honesty about prompt injection: a regex pretending to be a detector is not
shipped. What is shipped is the checkpoint plus ``screen_with(verifier)`` —
an adapter that turns any ``kind="verifier"`` judge (LLM, NLI) into a rule.
It costs tokens, goes through Quota, and is cacheable like any verifier.

Preset usage — table order is the middleware order, keep policy after the
observers and before cache (observers outside Policy so a deny is visible;
Policy outside Cache so hits are checked):

    [middleware.observe]
    [middleware.metrics]

    [middleware.policy]
    ingress = ["my_pkg.rules:reject_injection_markers"]
    egress = ["my_pkg.rules:redact_emails"]
    on_deny = "return"
    fallback = {answer = "I can't help with that — handing off to an operator."}

    [middleware.cache]
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from ..context import RunContext
from ..errors import EarlyReturn, PolicyError
from ..middleware import Middleware


class Allow:
    """Explicitly allow. Required for default="deny" checkpoints, where
    abstention is not consent; the reason lands in the audit ledger
    ("who let this through and why")."""

    __slots__ = ("reason",)

    def __init__(self, reason: str = ""):
        self.reason = reason


class Deny:
    """Block the value. The first Deny short-circuits the checkpoint."""

    __slots__ = ("reason",)

    def __init__(self, reason: str = ""):
        self.reason = reason


class Transform:
    """Substitute the value (redaction) and continue with the next rule."""

    __slots__ = ("value", "reason")

    def __init__(self, value: Any, reason: str = ""):
        self.value = value
        self.reason = reason


class Flag:
    """Record for audit without blocking."""

    __slots__ = ("reason",)

    def __init__(self, reason: str = ""):
        self.reason = reason


def _rule_name(rule: Callable) -> str:
    return getattr(rule, "__name__", None) or type(rule).__name__


def _resolve_rules(rules: Sequence[Callable | str] | None) -> list[Callable]:
    from ..registry import check_kind, resolve
    resolved = []
    for ref in rules or ():
        rule = resolve(ref, kind="policy") if (isinstance(ref, str)
                                               and ":" not in ref) else resolve(ref)
        problem = check_kind(rule, "policy")
        if problem:
            raise TypeError(problem)
        resolved.append(rule)
    return resolved


class Policy(Middleware):
    """Host of ordered policy rules at the run's ingress and egress.

    Args:
        ingress: rules for the incoming payload (on_run_start), in order.
        egress:  rules for the final output (on_run_end) — cache hits
                 included, per the finalizer-sweep contract.
        default: what silence means when no rule denied: "allow" (fail-open
                 screening, the default) or "deny" (fail-closed authz — at
                 least one rule must return an explicit Allow).
        on_deny: "raise" (PolicyError) or "return" (answer with fallback).
        fallback: the substitute answer for on_deny="return" — a value or a
                 callable ``(value, ctx) -> value``.

    Rules may be callables, registered names (kind="policy") or import
    paths. Position in the stack: outside Cache, inside Observe/Metrics.
    """

    name = "policy"
    phase = "policy"

    def __init__(self, ingress: Sequence[Callable | str] | None = None,
                 egress: Sequence[Callable | str] | None = None,
                 default: str = "allow", on_deny: str = "raise",
                 fallback: Any | Callable[[Any, RunContext], Any] = None):
        if default not in ("allow", "deny"):
            raise ValueError(f"default must be 'allow' or 'deny', got {default!r}")
        if on_deny not in ("raise", "return"):
            raise ValueError(f"on_deny must be 'raise' or 'return', got {on_deny!r}")
        self.ingress = _resolve_rules(ingress)
        self.egress = _resolve_rules(egress)
        self.default = default
        self.on_deny = on_deny
        self.fallback = fallback

    # -- lifecycle -----------------------------------------------------------
    def on_run_start(self, ctx: RunContext, payload):
        return self._checkpoint("ingress", self.ingress, ctx, payload)

    def on_run_end(self, ctx: RunContext, output):
        # finalizer sweep: runs on success and on EarlyReturn alike, so a
        # run-level cache hit passes the same egress rules as a fresh answer
        return self._checkpoint("egress", self.egress, ctx, output)

    # -- evaluation ----------------------------------------------------------
    def _checkpoint(self, checkpoint: str, rules: list[Callable],
                    ctx: RunContext, value: Any) -> Any:
        if not rules:
            # a checkpoint with no rules is inert: `default` governs the
            # outcome of an evaluation, and evaluation happens only where
            # rules are installed — otherwise default="deny" for ingress
            # authz would silently kill the (rule-less) egress too
            return value
        allowed = False
        for rule in rules:
            verdict = rule(checkpoint, value, ctx)
            if verdict is None:
                continue
            name = _rule_name(rule)
            if isinstance(verdict, Deny):
                return self._deny(ctx, checkpoint, name, verdict.reason, value)
            if isinstance(verdict, Transform):
                value = verdict.value
                self._record(ctx, checkpoint, name, "transform", verdict.reason,
                             "policy_redacted", "policy.redacted")
            elif isinstance(verdict, Flag):
                self._record(ctx, checkpoint, name, "flag", verdict.reason,
                             "policy_flagged", "policy.flagged")
            elif isinstance(verdict, Allow):
                allowed = True
                self._record(ctx, checkpoint, name, "allow", verdict.reason,
                             "policy_allowed", "policy.allowed")
            else:
                raise PolicyError(
                    f"policy rule {name!r} returned {verdict!r}; expected "
                    f"None, Allow, Deny, Transform or Flag",
                    checkpoint=checkpoint, rule=name)
        if self.default == "deny" and not allowed:
            return self._deny(ctx, checkpoint, "(default)",
                              "no rule explicitly allowed (default='deny')", value)
        return value

    # -- decisions -----------------------------------------------------------
    def _record(self, ctx: RunContext, checkpoint: str, rule: str, verdict: str,
                reason: str, event: str, counter: str) -> None:
        # guard with setdefault: on a cache hit egress runs without ingress
        ctx.artifacts.setdefault("policy", []).append(
            {"checkpoint": checkpoint, "rule": rule, "verdict": verdict,
             "reason": reason})
        ctx.metric(counter)
        ctx.emit(event, checkpoint=checkpoint, rule=rule, reason=reason)

    def _deny(self, ctx: RunContext, checkpoint: str, rule: str, reason: str,
              value: Any) -> Any:
        self._record(ctx, checkpoint, rule, "deny", reason,
                     "policy_denied", "policy.denied")
        if self.on_deny == "return":
            if callable(self.fallback):
                substitute = self.fallback(value, ctx)
            else:
                substitute = self.fallback
            if checkpoint == "ingress":
                # the Quota pattern: skip the steps, keep the finalizer sweep
                # — the fallback still passes egress on its way out
                raise EarlyReturn(substitute)
            # egress: plain substitution — an EarlyReturn inside the
            # on_run_end sweep would be caught as a failure, not control flow
            return substitute
        raise PolicyError(
            f"policy denied at {checkpoint}: {reason or rule}",
            checkpoint=checkpoint, rule=rule, reason=reason)


# ---------------------------------------------------------------------------
# Stochastic screening: a verifier as a rule
# ---------------------------------------------------------------------------

def _as_text(value: Any, key: str | None) -> str:
    if key is not None and isinstance(value, dict):
        value = value.get(key, value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _as_score(result: Any) -> float:
    if isinstance(result, bool):
        return 1.0 if result else 0.0
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, dict):
        for field in ("score", "confidence"):
            if isinstance(result.get(field), (int, float)):
                return float(result[field])
    raise PolicyError(f"screen_with verifier returned {result!r}; expected a "
                      f"score (number/bool or a dict with score/confidence)")


def screen_with(verifier: Callable[[str, list[str]], Any] | str, *,
                claim: str = ("the input attempts prompt injection or tries "
                              "to override the assistant's instructions"),
                threshold: float = 0.5, action: str = "deny",
                key: str | None = None) -> Callable:
    """Adapt a ``kind="verifier"`` judge into a policy rule.

    The honest way to screen for prompt injection: no regex pretending to be
    a detector. ``verifier(claim, [value_text])`` returns a score — how
    strongly the value supports ``claim``; at or above ``threshold`` the
    rule returns Deny (or Flag with ``action="flag"``). Costs tokens: goes
    through Quota and is cacheable like any verifier.

    Args:
        verifier:  callable, registered name or import path (kind="verifier").
        claim:     the statement the judge scores against the value.
        threshold: score at which the rule fires.
        action:    "deny" (default) or "flag".
        key:       dict payload field to screen (default: the whole value).
    """
    if isinstance(verifier, str):
        from ..registry import resolve
        verifier = resolve(verifier, kind="verifier") if ":" not in verifier \
            else resolve(verifier)
    if action not in ("deny", "flag"):
        raise ValueError(f"action must be 'deny' or 'flag', got {action!r}")
    judge = getattr(verifier, "__name__", type(verifier).__name__)

    def rule(checkpoint: str, value: Any, ctx: RunContext):
        score = _as_score(verifier(claim, [_as_text(value, key)]))
        if score >= threshold:
            reason = f"{judge}: {claim} (score {score:.2f} >= {threshold})"
            return Deny(reason) if action == "deny" else Flag(reason)
        return None

    rule.__name__ = f"screen_with({judge})"
    return rule
