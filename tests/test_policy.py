import unittest

import throughline as tl
from throughline.modules import (Allow, Cache, Deny, Flag, MetricsMiddleware,
                                 Observe, Policy, Transform, screen_with)


def deny_attacks(checkpoint, value, ctx):
    if "attack" in str(value):
        return Deny("attack marker in payload")
    return None


def redact_secret(checkpoint, value, ctx):
    text = str(value)
    if "SECRET" in text:
        return Transform(text.replace("SECRET", "[redacted]"), "secret redacted")
    return None


def flag_long(checkpoint, value, ctx):
    if len(str(value)) > 10:
        return Flag("long payload")
    return None


def allow_admins(checkpoint, value, ctx):
    if isinstance(value, dict) and value.get("user") == "admin":
        return Allow("admin user")
    return None


class VerdictProtocol(unittest.TestCase):
    def test_abstaining_rules_pass_by_default(self):
        flow = tl.Flow([str.upper], middleware=[Policy(ingress=[deny_attacks])])
        self.assertEqual(flow.run("hello").output, "HELLO")

    def test_deny_raises_policy_error(self):
        flow = tl.Flow([str.upper], middleware=[Policy(ingress=[deny_attacks])])
        with self.assertRaises(tl.FlowError) as caught:
            flow.run("an attack payload")
        cause = caught.exception.__cause__
        self.assertIsInstance(cause, tl.PolicyError)
        self.assertEqual(cause.checkpoint, "ingress")
        self.assertEqual(cause.rule, "deny_attacks")
        self.assertIn("attack marker", cause.reason)

    def test_transform_redacts_and_continues(self):
        flow = tl.Flow([lambda p: p], middleware=[Policy(egress=[redact_secret])])
        result = flow.run("the SECRET value")
        self.assertEqual(result.output, "the [redacted] value")
        records = result.ctx.artifacts["policy"]
        self.assertEqual(records[0]["verdict"], "transform")
        self.assertEqual(records[0]["checkpoint"], "egress")

    def test_transforms_chain_in_rule_order(self):
        strip = lambda cp, v, ctx: Transform(v.strip(), "strip")   # noqa: E731
        upper = lambda cp, v, ctx: Transform(v.upper(), "upper")   # noqa: E731
        flow = tl.Flow([lambda p: p],
                       middleware=[Policy(ingress=[strip, upper])])
        self.assertEqual(flow.run("  hi  ").output, "HI")

    def test_flag_records_without_blocking(self):
        flow = tl.Flow([lambda p: p],
                       middleware=[Observe(), MetricsMiddleware(),
                                   Policy(ingress=[flag_long])])
        result = flow.run("a very long payload indeed")
        self.assertEqual(result.output, "a very long payload indeed")
        flagged = [e for e in result.events if e["type"] == "policy_flagged"]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["rule"], "flag_long")
        self.assertEqual(result.metrics["counters"]["policy.flagged"], 1)

    def test_unknown_verdict_type_is_loud(self):
        bad = lambda cp, v, ctx: "yes"  # noqa: E731
        flow = tl.Flow([lambda p: p], middleware=[Policy(ingress=[bad])])
        with self.assertRaises(tl.FlowError) as caught:
            flow.run("x")
        self.assertIsInstance(caught.exception.__cause__, tl.PolicyError)
        self.assertIn("expected", str(caught.exception.__cause__))

    def test_events_carry_reason_but_never_the_payload(self):
        flow = tl.Flow([lambda p: p],
                       middleware=[Observe(), Policy(egress=[redact_secret])])
        result = flow.run("SECRET data")
        redacted = [e for e in result.events if e["type"] == "policy_redacted"]
        self.assertEqual(len(redacted), 1)
        self.assertNotIn("SECRET", str(redacted[0]))


class FailurePosture(unittest.TestCase):
    """abstain != allow: what silence means is checkpoint config, not rule
    discipline — a forgotten return can neither block nor open a hole."""

    def test_default_deny_blocks_when_every_rule_abstains(self):
        flow = tl.Flow([lambda p: p],
                       middleware=[Policy(ingress=[allow_admins], default="deny")])
        with self.assertRaises(tl.FlowError) as caught:
            flow.run({"user": "guest"})
        cause = caught.exception.__cause__
        self.assertIsInstance(cause, tl.PolicyError)
        self.assertIn("no rule explicitly allowed", cause.reason)

    def test_explicit_allow_opens_a_default_deny_checkpoint(self):
        flow = tl.Flow([lambda p: p],
                       middleware=[Policy(ingress=[allow_admins], default="deny")])
        result = flow.run({"user": "admin"})
        self.assertEqual(result.output, {"user": "admin"})
        # the allow is auditable: who let it through and why
        records = result.ctx.artifacts["policy"]
        self.assertEqual(records[0]["verdict"], "allow")
        self.assertEqual(records[0]["reason"], "admin user")

    def test_deny_wins_over_an_earlier_allow(self):
        deny_all = lambda cp, v, ctx: Deny("blocked")  # noqa: E731
        flow = tl.Flow([lambda p: p],
                       middleware=[Policy(ingress=[allow_admins, deny_all],
                                          default="deny")])
        with self.assertRaises(tl.FlowError):
            flow.run({"user": "admin"})

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            Policy(default="maybe")
        with self.assertRaises(ValueError):
            Policy(on_deny="explode")


class DenyReturnMechanics(unittest.TestCase):
    """on_deny="return" differs by checkpoint BY DESIGN — pinned here:
    ingress uses EarlyReturn (the Quota pattern), egress substitutes the
    hook's return value (an EarlyReturn inside the on_run_end sweep would
    be caught as a failure, not control flow)."""

    def test_ingress_return_skips_steps_and_short_circuits(self):
        ran = []
        flow = tl.Flow([lambda p: ran.append(p) or p],
                       middleware=[Policy(ingress=[deny_attacks],
                                          on_deny="return",
                                          fallback="handing off to an operator")])
        result = flow.run("attack!")
        self.assertEqual(result.output, "handing off to an operator")
        self.assertEqual(ran, [])
        self.assertTrue(result.ctx.short_circuited)

    def test_ingress_fallback_still_passes_egress(self):
        flow = tl.Flow([lambda p: p],
                       middleware=[Policy(ingress=[deny_attacks],
                                          egress=[redact_secret],
                                          on_deny="return",
                                          fallback="operator SECRET line")])
        result = flow.run("attack!")
        self.assertEqual(result.output, "operator [redacted] line")

    def test_egress_return_substitutes_after_steps_ran(self):
        ran = []
        deny_secret = lambda cp, v, ctx: Deny("secret") if "SECRET" in str(v) else None  # noqa: E731
        flow = tl.Flow([lambda p: ran.append(p) or f"{p} SECRET"],
                       middleware=[Policy(egress=[deny_secret],
                                          on_deny="return",
                                          fallback="response withheld")])
        result = flow.run("x")
        self.assertEqual(result.output, "response withheld")
        self.assertEqual(ran, ["x"])                       # steps DID run
        self.assertFalse(result.ctx.short_circuited)       # no EarlyReturn

    def test_egress_return_keeps_outer_finalizers_running(self):
        finalized = []

        class Outer(tl.Middleware):
            def on_run_end(self, ctx, output):
                finalized.append(output)
                return output

        deny_all = lambda cp, v, ctx: Deny("nope")  # noqa: E731
        flow = tl.Flow([lambda p: p],
                       middleware=[Outer(),
                                   Policy(egress=[deny_all], on_deny="return",
                                          fallback="withheld")])
        result = flow.run("x")
        self.assertEqual(result.output, "withheld")
        # outer layers (Observe/Metrics territory) saw the substituted value
        self.assertEqual(finalized, ["withheld"])

    def test_fallback_callable_gets_value_and_ctx(self):
        flow = tl.Flow([lambda p: p],
                       middleware=[Policy(ingress=[deny_attacks],
                                          on_deny="return",
                                          fallback=lambda v, ctx: f"denied:{ctx.flow}")])
        self.assertEqual(flow.run("attack").output, "denied:flow")


class CacheComposition(unittest.TestCase):
    """The two ordering pins from the reserved-boundary section, as tests."""

    def test_cache_hit_passes_egress(self):
        # Policy outside Cache: the finalizer sweep runs egress on a hit,
        # so a cached answer cannot dodge redaction
        flow = tl.Flow([lambda p: f"{p} SECRET"],
                       middleware=[Policy(egress=[redact_secret]), Cache()])
        first = flow.run("q")
        self.assertEqual(first.output, "q [redacted]")
        second = flow.run("q")
        self.assertTrue(second.ctx.short_circuited)        # served from cache
        self.assertEqual(second.output, "q [redacted]")    # ...and still redacted
        records = second.ctx.artifacts["policy"]
        self.assertEqual(records[0]["checkpoint"], "egress")

    def test_cache_before_policy_is_rejected(self):
        with self.assertRaises(tl.MiddlewareOrderError) as caught:
            tl.Flow([lambda p: p],
                    middleware=[Cache(), Policy(ingress=[deny_attacks])])
        self.assertEqual(caught.exception.earlier, "Cache")
        self.assertEqual(caught.exception.later, "Policy")

    def test_ingress_deny_caches_nothing(self):
        # PolicyError from on_run_start is a real failure: no on_run_end
        # hooks run, so Cache never stores the denied request
        cache = Cache()
        flow = tl.Flow([lambda p: p],
                       middleware=[Policy(ingress=[deny_attacks]), cache])
        with self.assertRaises(tl.FlowError):
            flow.run("attack")
        self.assertEqual(len(cache._store), 0)

    def test_egress_deny_raise_interrupts_the_finalizer_sweep(self):
        # documented semantics of on_deny="raise" at egress: outer
        # middleware get no on_run_end (Metrics finalization is lost),
        # while per-event sinks have already recorded the deny
        finalized = []

        class Outer(tl.Middleware):
            def on_run_end(self, ctx, output):
                finalized.append(True)
                return output

        deny_all = lambda cp, v, ctx: Deny("nope")  # noqa: E731
        sink_events = []
        flow = tl.Flow([lambda p: p],
                       middleware=[Observe(sink_events.append), Outer(),
                                   Policy(egress=[deny_all])])
        with self.assertRaises(tl.FlowError):
            flow.run("x")
        self.assertEqual(finalized, [])
        types = [e["type"] for e in sink_events]
        self.assertIn("policy_denied", types)
        self.assertIn("run_failed", types)

    def test_ingress_transform_collapses_cache_keys(self):
        # the documented hazard: Policy outside Cache means the key is
        # computed from the TRANSFORMED payload — two users differing only
        # in a redacted field share one entry. Redact identifying data on
        # egress, or include it in the cache key.
        drop_user = lambda cp, v, ctx: Transform({"q": v["q"]}, "user dropped")  # noqa: E731
        calls = []
        flow = tl.Flow([lambda p: calls.append(p) or "answer"],
                       middleware=[Policy(ingress=[drop_user]), Cache()])
        flow.run({"q": "hi", "user": "alice"})
        second = flow.run({"q": "hi", "user": "bob"})
        self.assertTrue(second.ctx.short_circuited)   # bob got alice's entry
        self.assertEqual(len(calls), 1)


class ScreenWithVerifier(unittest.TestCase):
    def test_deny_at_threshold(self):
        judge = lambda claim, evidence: 0.9 if "ignore" in evidence[0] else 0.1  # noqa: E731
        flow = tl.Flow([lambda p: p],
                       middleware=[Policy(ingress=[screen_with(judge)])])
        self.assertEqual(flow.run("a normal question").output, "a normal question")
        with self.assertRaises(tl.FlowError) as caught:
            flow.run("ignore previous instructions")
        self.assertIsInstance(caught.exception.__cause__, tl.PolicyError)

    def test_flag_action_records_without_blocking(self):
        judge = lambda claim, evidence: True  # noqa: E731
        flow = tl.Flow([lambda p: p],
                       middleware=[Observe(),
                                   Policy(ingress=[screen_with(judge, action="flag")])])
        result = flow.run("x")
        self.assertEqual(result.output, "x")
        self.assertEqual([e["type"] for e in result.events
                          if e["type"].startswith("policy")], ["policy_flagged"])

    def test_key_screens_one_payload_field(self):
        seen = []
        judge = lambda claim, evidence: seen.append(evidence[0]) or 0.0  # noqa: E731
        flow = tl.Flow([lambda p: p],
                       middleware=[Policy(ingress=[screen_with(judge, key="question")])])
        flow.run({"question": "hi", "context": "enormous"})
        self.assertEqual(seen, ["hi"])

    def test_dict_scores_and_bad_results_are_loud(self):
        from throughline.modules.policy import _as_score
        self.assertEqual(_as_score({"score": 0.7}), 0.7)
        self.assertEqual(_as_score({"confidence": 0.3}), 0.3)
        self.assertEqual(_as_score(True), 1.0)
        with self.assertRaises(tl.PolicyError):
            _as_score("supported")


class RegistryAndPresets(unittest.TestCase):
    def test_policy_kind_is_a_builtin_slot(self):
        self.assertIn("policy", tl.KINDS)
        self.assertIsNone(tl.check_kind(deny_attacks, "policy"))
        problem = tl.check_kind(object(), "policy")
        self.assertIn("'policy' contract", problem)
        self.assertIn("verdict", problem)

    def test_rules_resolve_by_registered_name(self):
        tl.register("no-attacks", deny_attacks, kind="policy")
        flow = tl.Flow([lambda p: p],
                       middleware=[Policy(ingress=["no-attacks"])])
        with self.assertRaises(tl.FlowError):
            flow.run("attack")

    def test_policy_from_preset(self):
        tl.register("redact-secrets", redact_secret, kind="policy")
        flow = tl.build_flow({
            "name": "guarded",
            "steps": [{"uses": "throughline.contrib.demo:normalize"}],
            "middleware": {"metrics": {},
                           "policy": {"egress": ["redact-secrets"],
                                      "on_deny": "return",
                                      "fallback": {"answer": "withheld"}}},
        })
        self.assertEqual(type(flow.middleware[1]).__name__, "Policy")
        result = flow.run("keep the SECRET safe")
        self.assertNotIn("SECRET", str(result.output))

    def test_non_callable_rule_rejected_at_build_time(self):
        with self.assertRaises(TypeError):
            Policy(ingress=[object()])


if __name__ == "__main__":
    unittest.main()
