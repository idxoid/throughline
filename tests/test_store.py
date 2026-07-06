import time
import unittest

import followers as fl
from followers.errors import ArtifactExpired, StoreError
from followers.store import ArtifactRef, MemoryArtifactStore


class ArtifactRefTests(unittest.TestCase):
    def test_roundtrip(self):
        ref = ArtifactRef(id="sess/abc", meta={"size": 3})
        again = ArtifactRef.from_dict(ref.to_dict())
        self.assertEqual(again.id, "sess/abc")
        self.assertEqual(again.session, "sess")
        self.assertEqual(again.key, "abc")
        self.assertEqual(again.meta["size"], 3)

    def test_from_dict_rejects_garbage(self):
        with self.assertRaises(StoreError):
            ArtifactRef.from_dict({"nope": 1})


class MemoryArtifactStoreTests(unittest.TestCase):
    def test_put_get(self):
        store = MemoryArtifactStore()
        ref = store.put([1, 2, 3], session="s1")
        self.assertEqual(store.get(ref), [1, 2, 3])
        self.assertEqual(store.get(ref.id), [1, 2, 3])  # string id works too
        self.assertEqual(ref.meta["kind"], "list")
        self.assertEqual(ref.meta["items"], 3)

    def test_slice(self):
        store = MemoryArtifactStore()
        ref = store.put(list(range(100)))
        self.assertEqual(store.slice(ref, 10, 13), [10, 11, 12])
        text_ref = store.put("hello world")
        self.assertEqual(store.slice(text_ref, 0, 5), "hello")

    def test_slice_unsliceable(self):
        store = MemoryArtifactStore()
        ref = store.put({"a": 1})
        with self.assertRaises(StoreError):
            store.slice(ref, 0, 1)

    def test_expiry_is_a_lease(self):
        store = MemoryArtifactStore()
        ref = store.put("data", ttl=0.01)
        time.sleep(0.02)
        with self.assertRaises(ArtifactExpired) as caught:
            store.get(ref)
        self.assertEqual(caught.exception.artifact_id, ref.id)
        self.assertIn("re-run", str(caught.exception))

    def test_missing_artifact(self):
        store = MemoryArtifactStore()
        with self.assertRaises(ArtifactExpired):
            store.get("nowhere/nothing")

    def test_session_count_cap_evicts_oldest(self):
        store = MemoryArtifactStore(max_per_session=2)
        first = store.put("a", session="s")
        store.put("b", session="s")
        store.put("c", session="s")
        with self.assertRaises(ArtifactExpired):
            store.get(first)
        self.assertEqual(len(store), 2)

    def test_session_byte_cap(self):
        store = MemoryArtifactStore(max_bytes_per_session=100)
        first = store.put("x" * 80, session="s")
        store.put("y" * 80, session="s")
        with self.assertRaises(ArtifactExpired):
            store.get(first)

    def test_sessions_are_isolated(self):
        store = MemoryArtifactStore(max_per_session=1)
        ref_a = store.put("a", session="one")
        ref_b = store.put("b", session="two")
        self.assertEqual(store.get(ref_a), "a")
        self.assertEqual(store.get(ref_b), "b")

    def test_drop_session(self):
        store = MemoryArtifactStore()
        ref = store.put("a", session="gone")
        keep = store.put("b", session="kept")
        removed = store.drop_session("gone")
        self.assertEqual(removed, 1)
        with self.assertRaises(ArtifactExpired):
            store.get(ref)
        self.assertEqual(store.get(keep), "b")

    def test_ref_usable_in_payload(self):
        """The intended pattern: heavy data by handle, payload stays small."""
        store = MemoryArtifactStore()
        corpus_ref = store.put(["doc"] * 1000, session="run")

        def uses_corpus(payload, ctx):
            corpus = store.get(payload["corpus"])
            return {**payload, "n_docs": len(corpus)}

        result = fl.Flow([uses_corpus]).run({"corpus": corpus_ref})
        self.assertEqual(result.output["n_docs"], 1000)


if __name__ == "__main__":
    unittest.main()
