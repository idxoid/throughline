"""Self-contained demo components for the builtin `demo` preset.

Everything is offline and deterministic: a keyword retriever over a tiny
corpus about the followers project itself, a prompt template, and a fake LLM.
Referenced from presets by import path, e.g.:

    [[steps]]
    uses = "followers.contrib.demo:retriever"

It can also be invoked directly from a source checkout for a quick smoke test:

    python src/followers/contrib/demo.py --input "how does lineage work?"
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from followers.adapters.llm import FakeLLM
from followers.adapters.rag import make_keyword_retriever, prompt_step
from followers.step import Step

CORPUS = [
    "followers is a lightweight orchestrator for agents and LLM pipelines.",
    "A Flow is an ordered chain of steps executed under a middleware stack.",
    "Middleware plugs in validation, metrics, observability and lineage.",
    "Line-level lineage answers which step wrote every line of the output.",
    "Presets are TOML files that describe steps, middleware and config.",
    "Third-party RAG components onboard through duck-typed adapters.",
    "The core has zero dependencies and needs only Python 3.11+.",
]


def normalize(payload, ctx) -> dict:
    """Accept str or dict; normalize to the RAG payload convention."""
    if isinstance(payload, dict):
        question = str(payload.get("question", "")).strip()
        return {**payload, "question": question}
    return {"question": str(payload).strip()}


def retriever(top_k: int = 3) -> Step:
    """Factory form: referenced with a [steps.with] table in presets."""
    return make_keyword_retriever(CORPUS, top_k=top_k)


# Ready-made Step form: referenced directly (no [steps.with]).
prompt: Step = prompt_step(
    "Answer the question using only the context.\n"
    "Context:\n{context}\n"
    "Question: {question}"
)

llm: Step = FakeLLM().answer_step()


def main(argv: list[str] | None = None) -> int:
    """Run the builtin demo preset when this module is executed as a script."""
    import argparse

    from followers.cli import main as cli_main

    parser = argparse.ArgumentParser(
        description="Run the builtin offline followers demo preset."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="how does lineage work?",
        help="input question for the demo flow",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    parser.add_argument("--metrics", action="store_true", help="print metrics to stderr")
    parser.add_argument("--blame", action="store_true", help="print line-level lineage")
    args = parser.parse_args(argv)

    cli_args = ["run", "demo", "--input", args.input]
    for flag in ("json", "metrics", "blame"):
        if getattr(args, flag):
            cli_args.append(f"--{flag}")
    return cli_main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
