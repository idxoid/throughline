import unittest

import throughline as tl
from throughline.adapters.llm import FakeLLM, from_callable
from throughline.adapters.rag import KeywordRetriever, prompt_step, retriever_step


class FakeLangChainRunnable:
    """Mimics a LangChain Runnable (invoke)."""

    def invoke(self, value):
        return f"invoked:{value}"


class FakeLlamaIndexEngine:
    """Mimics a LlamaIndex query engine (query -> response object)."""

    class Response:
        def __init__(self, text):
            self.response = text

    def query(self, value):
        return self.Response(f"queried:{value}")


class FakeLegacyRetriever:
    def get_relevant_documents(self, query):
        class Doc:
            def __init__(self, text):
                self.page_content = text
        return [Doc(f"doc about {query}"), Doc("other doc")]


class FakeAgent:
    def run(self, task):
        return f"agent did: {task}"


class WrapTests(unittest.TestCase):
    def test_wrap_invoke_priority(self):
        step = tl.wrap(FakeLangChainRunnable())
        self.assertEqual(step.meta["adapter"], "invoke")
        self.assertEqual(tl.Flow([step]).run("x").output, "invoked:x")

    def test_wrap_query_with_unwrap(self):
        step = tl.wrap(FakeLlamaIndexEngine(), unwrap=lambda r: r.response)
        self.assertEqual(tl.Flow([step]).run("y").output, "queried:y")

    def test_wrap_agent_run(self):
        step = tl.wrap(FakeAgent(), name="agent")
        self.assertEqual(step.name, "agent")
        self.assertEqual(tl.Flow([step]).run("task").output, "agent did: task")

    def test_wrap_explicit_method(self):
        class Multi:
            def invoke(self, x):
                return "wrong"

            def run(self, x):
                return "right"
        step = tl.wrap(Multi(), method="run")
        self.assertEqual(tl.Flow([step]).run("_").output, "right")

    def test_wrap_plain_callable_object(self):
        class CallMe:
            def __call__(self, x):
                return x + 1
        self.assertEqual(tl.Flow([tl.wrap(CallMe())]).run(1).output, 2)

    def test_wrap_unadaptable_raises(self):
        class Opaque:
            pass
        with self.assertRaises(tl.ThroughlineError):
            tl.wrap(Opaque())

    def test_as_step_uses_wrap_for_objects(self):
        step = tl.as_step(FakeLangChainRunnable())
        self.assertEqual(step.meta.get("adapter"), "invoke")


class RagAdapterTests(unittest.TestCase):
    def test_retriever_step_duck_typing(self):
        step = retriever_step(FakeLegacyRetriever(), top_k=1)
        result = tl.Flow([step]).run({"question": "lineage"})
        self.assertEqual(result.output["context"], ["doc about lineage"])
        self.assertEqual(result.output["question"], "lineage")

    def test_retriever_step_accepts_bare_string(self):
        step = retriever_step(KeywordRetriever(["about cats here", "about dogs"]))
        result = tl.Flow([step]).run("cats")
        self.assertEqual(result.output["context"], ["about cats here"])

    def test_prompt_step_renders_lists(self):
        step = prompt_step("Q: {question}\nCTX:\n{context}")
        out = tl.Flow([step]).run({"question": "q", "context": ["a", "b"]}).output
        self.assertEqual(out["prompt"], "Q: q\nCTX:\na\nb")

    def test_prompt_step_missing_key(self):
        with self.assertRaises(tl.FlowError):
            tl.Flow([prompt_step("{missing}")]).run({"question": "q"})


class LlmAdapterTests(unittest.TestCase):
    def test_from_callable_str_and_dict(self):
        step = from_callable(lambda prompt: f"echo:{prompt}")
        self.assertEqual(tl.Flow([step]).run("hi").output, "echo:hi")
        out = tl.Flow([step]).run({"prompt": "hi", "extra": 1}).output
        self.assertEqual(out["answer"], "echo:hi")
        self.assertEqual(out["extra"], 1)

    def test_fake_llm_grounds_in_context(self):
        step = FakeLLM().answer_step()
        payload = {"question": "what?", "context": ["ctx line one", "ctx line two"]}
        out = tl.Flow([step]).run(payload).output
        self.assertIn("Answer to: what?", out["answer"])
        self.assertIn("ctx line one", out["answer"])

    def test_registry_register_resolve(self):
        @tl.register("my-step")
        def my(payload):
            return payload
        self.assertIs(tl.resolve("my-step"), my)
        self.assertIn("my-step", tl.available())

    def test_resolve_import_path(self):
        target = tl.resolve("throughline.adapters.rag:KeywordRetriever")
        self.assertIs(target, KeywordRetriever)

    def test_resolve_unknown_raises(self):
        with self.assertRaises(tl.RegistryError):
            tl.resolve("definitely-not-registered")
        with self.assertRaises(tl.RegistryError):
            tl.resolve("throughline.adapters.rag:Nope")


if __name__ == "__main__":
    unittest.main()
