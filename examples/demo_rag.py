"""End-to-end offline demo: RAG flow with the full middleware stack.

Run:  PYTHONPATH=src python3 examples/demo_rag.py

Shows the two flagship features together:
  * pluggable middleware (metrics, observability, validation, retry, lineage)
  * line-level lineage over the *answer text* as it is drafted, grounded and
    edited by consecutive steps — `git blame` for your pipeline output.
"""

import followers as fl
from followers.adapters.rag import make_keyword_retriever, prompt_step
from followers.modules import LineageMiddleware, MetricsMiddleware, Observe, Retry, Validate

CORPUS = [
    "followers is a lightweight orchestrator for agents and LLM pipelines.",
    "Line-level lineage answers which step wrote every line of the output.",
    "Middleware plugs in validation, metrics, observability and lineage.",
    "Presets are TOML files that describe steps, middleware and config.",
]


def normalize(payload) -> dict:
    return {"question": str(payload).strip()}


def draft(payload, ctx) -> dict:
    """First LLM pass (faked): draft an answer grounded in retrieved context."""
    lines = [f"Draft answer to: {payload['question']}"]
    lines += [f"- {doc}" for doc in payload["context"][:2]]
    lines += ["TODO: add citations", "Sincerely, the pipeline"]
    ctx.metric("llm.calls")
    return {**payload, "answer": "\n".join(lines)}


def refine(payload, ctx) -> dict:
    """Second LLM pass (faked): edit some lines, drop the TODO, add a source."""
    edited = []
    for line in payload["answer"].splitlines():
        if line.startswith("Draft answer"):
            edited.append(line.replace("Draft answer", "Answer"))   # modify
        elif line.startswith("TODO"):
            continue                                                # drop
        else:
            edited.append(line)                                     # carry
    edited.append("Sources: internal corpus")                       # generate
    ctx.metric("llm.calls")
    return {**payload, "answer": "\n".join(edited)}


flow = fl.Flow(
    [
        fl.as_step(normalize, "normalize"),
        make_keyword_retriever(CORPUS, top_k=2, name="retrieve"),
        prompt_step("Q: {question}\nCTX:\n{context}"),
        fl.as_step(draft, "draft"),
        fl.as_step(refine, "refine"),
    ],
    middleware=[
        Observe("console"),                       # events -> stderr
        MetricsMiddleware(),                      # timings + counters
        Retry(attempts=2, backoff=0.01),          # flaky-step protection
        Validate(schema={"type": "object", "required": ["question", "answer"]}),
        LineageMiddleware(extract="answer"),      # blame over the answer text
    ],
    name="rag-demo",
)

if __name__ == "__main__":
    result = flow.run("Which step wrote every line of this answer?")

    print("\n=== answer ===")
    print(result.output["answer"])

    print("\n=== metrics ===")
    for key, value in sorted(result.metrics["counters"].items()):
        print(f"  {key:28} {value:g}")

    print("\n=== line-level lineage (blame) ===")
    print(result.lineage.render_blame())

    print("\n=== lineage stats ===")
    print(result.lineage.stats())
