import unittest

import followers as fl
from followers import registry
from followers.errors import RegistryError
from followers.registry import RegistryEntry, check_kind


class DummyStore:
    def get(self, namespace, text, default=None): ...
    def set(self, namespace, text, value): ...


class KindChecksTests(unittest.TestCase):
    def test_step_accepts_callable_and_wrappable(self):
        self.assertIsNone(check_kind(lambda p: p, "step"))

        class Engine:
            def query(self, q): ...
        self.assertIsNone(check_kind(Engine(), "step"))

    def test_step_rejects_inert_object(self):
        problem = check_kind(object(), "step")
        self.assertIn("does not satisfy the 'step' contract", problem)

    def test_middleware_check(self):
        self.assertIsNone(check_kind(fl.modules.Retry(), "middleware"))
        self.assertIsNotNone(check_kind(lambda p: p, "middleware"))

    def test_store_check_accepts_both_shapes(self):
        self.assertIsNone(check_kind(DummyStore(), "store"))              # get/set
        self.assertIsNone(check_kind(fl.MemoryArtifactStore(), "store"))  # put/get
        self.assertIsNotNone(check_kind(object(), "store"))

    def test_store_subkinds_pin_the_protocol(self):
        """'store' is an umbrella; the subkinds tell the two protocols apart."""
        cache, artifact = DummyStore(), fl.MemoryArtifactStore()
        self.assertIsNone(check_kind(cache, "store.cache"))
        self.assertIsNone(check_kind(artifact, "store.artifact"))
        # a cache store is NOT an artifact store, and vice versa
        problem = check_kind(cache, "store.artifact")
        self.assertIn("'store.artifact' contract", problem)
        self.assertIn("put", problem)
        problem = check_kind(artifact, "store.cache")
        self.assertIn("'store.cache' contract", problem)
        # ...but both satisfy the umbrella
        self.assertIsNone(check_kind(cache, "store"))
        self.assertIsNone(check_kind(artifact, "store"))

    def test_builtin_namespace_is_reserved(self):
        # register_kind cannot squat store.* / step.* / ...
        with self.assertRaises(RegistryError) as caught:
            fl.register_kind("store.redis", check=callable)
        self.assertIn("built-in kind taxonomy", str(caught.exception))
        # neither can a plain registration
        with self.assertRaises(RegistryError):
            fl.register("x", DummyStore(), kind="store.weird")
        # and check_kind explains instead of silently passing
        message = check_kind(object(), "store.weird")
        self.assertIn("reserved for built-in subkinds", message)

    def test_unknown_bare_kind(self):
        message = check_kind(lambda: 0, "wizard")
        self.assertIn("unknown kind", message)
        self.assertIn("yourpkg.wizard", message)  # points at the namespacing rule


class TypedRegistryTests(unittest.TestCase):
    def setUp(self):
        registry._reset_for_tests()

    def tearDown(self):
        registry._reset_for_tests()

    def test_register_default_kind_is_step(self):
        @fl.register("clean")
        def clean(text):
            return text
        self.assertIs(fl.resolve("clean"), clean)
        self.assertIs(fl.resolve("clean", kind="step"), clean)

    def test_register_typed(self):
        store = DummyStore()
        fl.register("mystore", store, kind="store")
        self.assertIs(fl.resolve("mystore", kind="store"), store)

    def test_kind_mismatch_is_loud(self):
        fl.register("mystore", DummyStore(), kind="store")
        with self.assertRaises(RegistryError) as caught:
            fl.resolve("mystore", kind="step")
        self.assertIn("registered as store, not 'step'", str(caught.exception))

    def test_same_name_multiple_kinds_prefers_slot_kind(self):
        fl.register("thing", lambda p: p, kind="step")
        store = DummyStore()
        fl.register("thing", store, kind="store")
        self.assertIs(fl.resolve("thing", kind="store"), store)

    def test_precedence_local_beats_plugin(self):
        registry._store("x", "from-plugin", "step", "plugin:pkg")
        registry._store("x", "from-local", "step", "local")
        self.assertEqual(fl.resolve("x"), "from-local")
        # and a plugin cannot shadow an existing local registration
        registry._store("x", "from-plugin-2", "step", "plugin:pkg2")
        self.assertEqual(fl.resolve("x"), "from-local")

    def test_manifest_registration(self):
        manifest = {
            "step:clean": lambda p: p,
            "store:redis": DummyStore(),
            "plainname": lambda p: p,        # legacy un-prefixed -> step
        }
        loaded = registry._register_manifest(manifest, "plugin:test")
        self.assertEqual(sorted(loaded), ["clean", "plainname", "redis"])
        self.assertIsNotNone(fl.resolve("redis", kind="store"))
        self.assertIsNotNone(fl.resolve("plainname", kind="step"))

    def test_requires_gate(self):
        self.assertIsNone(registry._compatible("followers>=0.1"))
        self.assertIn("requires followers>=99", registry._compatible("followers>=99.0"))
        self.assertIn("unsupported requires spec",
                      registry._compatible("followers==1.0"))

    def test_entries_catalog(self):
        fl.register("a", lambda p: p, kind="step")
        fl.register("b", DummyStore(), kind="store")
        catalog = registry.entries()
        self.assertEqual([(e.kind, e.name) for e in catalog],
                         [("step", "a"), ("store", "b")])
        self.assertTrue(all(isinstance(e, RegistryEntry) for e in catalog))


class CustomKindsTests(unittest.TestCase):
    """Closed built-in slots, open taxonomy: namespaced kinds for plugins."""

    def setUp(self):
        registry._reset_for_tests()

    def tearDown(self):
        registry._reset_for_tests()

    def test_namespaced_kind_registers_and_resolves(self):
        reranker = lambda docs: docs  # noqa: E731
        fl.register("fast", reranker, kind="acme.reranker")
        self.assertIs(fl.resolve("fast", kind="acme.reranker"), reranker)
        self.assertEqual(registry.entries()[0].kind, "acme.reranker")

    def test_bare_unknown_kind_is_rejected_at_registration(self):
        with self.assertRaises(RegistryError) as caught:
            fl.register("x", lambda p: p, kind="reranker")
        self.assertIn("namespaced", str(caught.exception))
        self.assertIn("yourpkg.reranker", str(caught.exception))

    def test_undeclared_custom_kind_is_catalog_only(self):
        """Core enforces nothing for kinds it does not own."""
        self.assertIsNone(check_kind(object(), "acme.reranker"))

    def test_declared_protocol_is_enforced(self):
        fl.register_kind("acme.reranker",
                         check=lambda obj: callable(getattr(obj, "rerank", None)),
                         shape="an object with rerank(docs)")

        class Good:
            def rerank(self, docs): ...

        self.assertIsNone(check_kind(Good(), "acme.reranker"))
        message = check_kind(object(), "acme.reranker")
        self.assertIn("'acme.reranker' contract", message)
        self.assertIn("rerank(docs)", message)

    def test_register_kind_rejects_builtin_and_bare_names(self):
        with self.assertRaises(RegistryError):
            fl.register_kind("step", check=callable)
        with self.assertRaises(RegistryError):
            fl.register_kind("reranker", check=callable)

    def test_manifest_with_namespaced_kind(self):
        loaded = registry._register_manifest(
            {"acme.reranker:fast": lambda d: d}, "plugin:acme")
        self.assertEqual(loaded, ["fast"])
        self.assertIsNotNone(fl.resolve("fast", kind="acme.reranker"))

    def test_kind_mismatch_message_covers_custom_kinds(self):
        fl.register("fast", lambda d: d, kind="acme.reranker")
        with self.assertRaises(RegistryError) as caught:
            fl.resolve("fast", kind="step")
        self.assertIn("registered as acme.reranker, not 'step'", str(caught.exception))

    def test_builtin_slot_wins_name_collisions_over_custom(self):
        fl.register("thing", "custom-kind-obj", kind="acme.widget")
        fl.register("thing", "step-obj", kind="step")
        self.assertEqual(fl.resolve("thing"), "step-obj")


if __name__ == "__main__":
    unittest.main()
