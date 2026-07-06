import time
import unittest

import followers as fl
from followers.modules import MetricsMiddleware
from followers.modules.cache import _MISS, Cache, LRUCache, SemanticCache, SemanticStore


class LRUCacheUnit(unittest.TestCase):
    def test_get_set_and_miss(self):
        store = LRUCache(max_size=4)
        self.assertIs(store.get("ns", "k", _MISS), _MISS)
        store.set("ns", "k", 42)
        self.assertEqual(store.get("ns", "k"), 42)
        self.assertIs(store.get("other-ns", "k", _MISS), _MISS)  # namespaced

    def test_lru_eviction(self):
        store = LRUCache(max_size=2)
        store.set("ns", "a", 1)
        store.set("ns", "b", 2)
        store.get("ns", "a")          # refresh a
        store.set("ns", "c", 3)       # evicts b (least recently used)
        self.assertEqual(store.get("ns", "a"), 1)
        self.assertIs(store.get("ns", "b", _MISS), _MISS)
        self.assertEqual(store.get("ns", "c"), 3)

    def test_ttl_expiry(self):
        store = LRUCache(max_size=4, ttl=0.01)
        store.set("ns", "k", 1)
        self.assertEqual(store.get("ns", "k"), 1)
        time.sleep(0.02)
        self.assertIs(store.get("ns", "k", _MISS), _MISS)


class SemanticStoreUnit(unittest.TestCase):
    VECTORS = {"q1": (1.0, 0.0), "q1-near": (0.98, 0.199), "q2": (0.0, 1.0)}

    def embed(self, text):
        return self.VECTORS[text]

    def test_exact_similar_and_far(self):
        store = SemanticStore(self.embed, threshold=0.92)
        store.set("ns", "q1", "answer-1")
        self.assertEqual(store.get("ns", "q1"), "answer-1")        # exact
        self.assertEqual(store.get("ns", "q1-near"), "answer-1")   # cosine ~0.98
        self.assertIs(store.get("ns", "q2", _MISS), _MISS)         # orthogonal

    def test_threshold_respected(self):
        store = SemanticStore(self.embed, threshold=0.999)
        store.set("ns", "q1", "answer-1")
        self.assertIs(store.get("ns", "q1-near", _MISS), _MISS)


class StepLevelCache(unittest.TestCase):
    def make_flow(self, cache):
        calls = {"n": 0}

        def llm(payload, ctx):
            calls["n"] += 1
            return f"answer:{payload}"
        flow = fl.Flow([fl.as_step(llm, "llm")], middleware=[MetricsMiddleware(), cache])
        return flow, calls

    def test_hit_skips_invocation(self):
        flow, calls = self.make_flow(Cache(step="llm*"))
        first = flow.run("q").output
        second = flow.run("q").output
        self.assertEqual(first, second)
        self.assertEqual(calls["n"], 1)
        metrics = flow.run("q").metrics  # third run: another hit
        self.assertEqual(calls["n"], 1)
        self.assertEqual(metrics["counters"]["cache.hits"], 1)

    def test_different_payloads_miss(self):
        flow, calls = self.make_flow(Cache(step="llm*"))
        flow.run("a")
        flow.run("b")
        self.assertEqual(calls["n"], 2)

    def test_pattern_mismatch_never_caches(self):
        flow, calls = self.make_flow(Cache(step="other*"))
        flow.run("q")
        flow.run("q")
        self.assertEqual(calls["n"], 2)

    def test_copy_isolation(self):
        cache = Cache(step="build")

        def build(payload, ctx):
            return {"items": [1, 2]}
        flow = fl.Flow([fl.as_step(build, "build")], middleware=[cache])
        first = flow.run("x").output
        first["items"].append(999)  # mutate the returned object
        second = flow.run("x").output
        self.assertEqual(second["items"], [1, 2])

    def test_key_field(self):
        calls = {"n": 0}

        def llm(payload, ctx):
            calls["n"] += 1
            return payload["question"]
        cache = Cache(step="llm", key="question")
        flow = fl.Flow([fl.as_step(llm, "llm")], middleware=[cache])
        flow.run({"question": "q", "session": "s1"})
        flow.run({"question": "q", "session": "s2"})  # different payload, same key
        self.assertEqual(calls["n"], 1)


class RunLevelCache(unittest.TestCase):
    def test_short_circuits_whole_flow(self):
        calls = {"n": 0}

        def heavy(payload, ctx):
            calls["n"] += 1
            return f"result:{payload}"
        flow = fl.Flow([fl.as_step(heavy, "heavy")], middleware=[Cache()])
        first = flow.run("q")
        second = flow.run("q")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(second.output, first.output)

    def test_run_end_hooks_still_apply_on_hit(self):
        from followers.modules import Validate
        flow = fl.Flow([lambda p: {"answer": p}],
                       middleware=[Cache(),
                                   Validate(schema={"required": ["answer"]})])
        flow.run("q")
        self.assertEqual(flow.run("q").output, {"answer": "q"})  # validated hit

    def test_hit_is_visible_with_recommended_order(self):
        """Observers outside Cache: a run-level hit must not be an invisible run."""
        from followers.modules import Observe
        flow = fl.Flow(
            [lambda p: {"answer": p}],
            middleware=[Observe(),            # observers first (outermost)...
                        MetricsMiddleware(),
                        Cache()],             # ...then the short-circuiter
        )
        flow.run("q")
        hit = flow.run("q")
        self.assertEqual(hit.metrics["counters"]["cache.hits"], 1)
        types = [e["type"] for e in hit.events]
        self.assertIn("cache_hit", types)
        self.assertIn("run_short_circuited", types)
        self.assertIn("run_finished", types)
        self.assertNotIn("run_started", types)  # short-circuited, never "started"
        self.assertNotIn("step_started", types)

    def test_outermost_cache_hit_is_invisible_documented_pitfall(self):
        """The anti-pattern the docs warn about: Cache outermost swallows
        observability on a hit. Pinned so the semantics never change silently."""
        flow = fl.Flow(
            [lambda p: {"answer": p}],
            middleware=[Cache(), MetricsMiddleware()],
        )
        flow.run("q")
        hit = flow.run("q")
        self.assertEqual(hit.metrics, {})  # metrics never attached: invisible run

    def test_run_started_reaches_observe_sinks(self):
        from followers.modules import Observe
        flow = fl.Flow([lambda p: p], middleware=[Observe()])
        types = [e["type"] for e in flow.run("q").events]
        self.assertIn("run_started", types)

    def test_semantic_run_cache(self):
        vectors = {"q1": (1.0, 0.0), "q1-near": (0.98, 0.199), "q2": (0.0, 1.0)}
        calls = {"n": 0}

        def heavy(payload, ctx):
            calls["n"] += 1
            return f"result:{payload}"
        flow = fl.Flow([fl.as_step(heavy, "heavy")],
                       middleware=[SemanticCache(embedder=lambda t: vectors[t])])
        first = flow.run("q1")
        near = flow.run("q1-near")
        self.assertEqual(near.output, first.output)   # semantic hit
        self.assertEqual(calls["n"], 1)
        flow.run("q2")                                # orthogonal -> miss
        self.assertEqual(calls["n"], 2)


class PurityGuard(unittest.TestCase):
    """A cached hit skips the step; skipped side effects are silent data
    loss. Purity is declarative (effects=...), Cache enforces it."""

    def make_effectful_flow(self, cache, effects="db.write"):
        sent = []

        def notify(payload, ctx):
            sent.append(payload)
            return payload
        flow = fl.Flow([fl.as_step(notify, "notify", effects=effects)],
                       middleware=[MetricsMiddleware(), cache])
        return flow, sent

    def test_effects_declaration_lands_in_meta(self):
        self.assertEqual(fl.as_step(lambda p: p, "s", effects="db.write").effects,
                         ("db.write",))
        self.assertEqual(fl.as_step(lambda p: p, "s", effects="pure").effects, ())
        self.assertEqual(fl.as_step(lambda p: p, "s", effects=True).effects,
                         ("unlabeled",))
        self.assertIsNone(fl.as_step(lambda p: p, "s").effects)  # unknown

    def test_step_decorator_declares_effects(self):
        @fl.step("save", effects=("db.write", "email.send"))
        def save(record):
            return record
        self.assertEqual(save.effects, ("db.write", "email.send"))

    def test_step_cache_skips_effectful_step(self):
        from followers.modules import Observe
        sent = []

        def notify(payload, ctx):
            sent.append(payload)
            return payload
        flow = fl.Flow([fl.as_step(notify, "notify", effects="db.write")],
                       middleware=[Observe(), MetricsMiddleware(),
                                   Cache(step="notify")])
        flow.run("a")
        result = flow.run("a")                    # would be a hit without the guard
        self.assertEqual(len(sent), 2)            # side effect happened both times
        events = [e for e in result.events if e["type"] == "cache_effects_bypass"]
        self.assertEqual(events[0]["step"], "notify")
        self.assertEqual(events[0]["effects"], ["db.write"])
        self.assertEqual(result.metrics["counters"]["cache.effects_bypass"], 1)

    def test_run_cache_never_stores_effectful_run(self):
        flow, sent = self.make_effectful_flow(Cache())  # run-level
        flow.run("a")
        flow.run("a")                              # never stored -> never a hit
        self.assertEqual(len(sent), 2)
        self.assertNotIn("cache.hits", flow.run("a").metrics["counters"])

    def test_declared_pure_step_is_cached(self):
        flow, sent = self.make_effectful_flow(Cache(step="notify"), effects="pure")
        flow.run("a")
        flow.run("a")
        self.assertEqual(len(sent), 1)             # cached: purity declared

    def test_undeclared_step_is_cached_as_before(self):
        flow, sent = self.make_effectful_flow(Cache(step="notify"), effects=None)
        flow.run("a")
        flow.run("a")
        self.assertEqual(len(sent), 1)             # guard trusts declarations only

    def test_on_effects_raise_is_a_config_error(self):
        flow, _ = self.make_effectful_flow(Cache(step="notify", on_effects="raise"))
        with self.assertRaises(fl.FlowError) as caught:
            flow.run("a")
        self.assertIn("side effects", str(caught.exception))
        self.assertIn("db.write", str(caught.exception))

    def test_on_effects_allow_caches_anyway(self):
        flow, sent = self.make_effectful_flow(Cache(step="notify", on_effects="allow"))
        flow.run("a")
        flow.run("a")
        self.assertEqual(len(sent), 1)             # explicit opt-in: hit skips effect

    def test_invalid_on_effects_rejected(self):
        with self.assertRaises(ValueError):
            Cache(on_effects="warn-maybe")

    def test_effects_from_preset(self):
        flow = fl.build_flow({
            "name": "declared",
            "steps": [{"uses": "followers.contrib.demo:normalize",
                       "effects": "db.write"}],
        })
        self.assertEqual(flow.steps[0].effects, ("db.write",))


class PresetIntegration(unittest.TestCase):
    def test_cache_from_preset(self):
        flow = fl.build_flow({
            "name": "cached",
            "steps": [{"uses": "followers.contrib.demo:normalize"}],
            "middleware": {"cache": {"step": "*", "max_size": 16, "ttl": 60}},
        })
        self.assertEqual(type(flow.middleware[0]).__name__, "Cache")
        self.assertEqual(flow.run("x").output, {"question": "x"})


if __name__ == "__main__":
    unittest.main()
