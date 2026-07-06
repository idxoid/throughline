"""Deterministic blast radius -> cited change-risk note.

Run (needs a live surgical_context index — see README.md in this directory):

  SC=~/surgical_context
  PYTHONPATH=src:$SC:$SC/mcp_server $SC/.venv/bin/python \
      examples/surgical_context/impact_note.py [symbol]

Flow: impact-evidence (pure Neo4j walk + exact call-site source, NO
      embeddings, NO vector search) -> prompt(cite) -> LLM -> citations_step.
The LLM only summarizes; every dependency fact is graph-derived and every
cited line joins back to a real file:line span.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import throughline as tl
from throughline.adapters.rag import prompt_step, retriever_step
from throughline.modules import LineageMiddleware, MetricsMiddleware, citations_step

from adapter import WORKSPACE, ImpactRetriever, default_llm, print_claim_report

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "run_axis_retrieval"

PROMPT = """\
The function `{symbol}` is about to be changed. Below are its downstream
dependents (exact source around each call site).

Write a change-risk note as a plain list, one line per dependent that really
calls or wraps `{symbol}`, in exactly this format:
- <function name>: <one short clause why it breaks>. [eN]
Then one final line: Re-test: <the most critical call sites>. [eN] [eM]

Rules: no preamble, no analysis — start directly with the first "- " line.
Every line ends with its [eN] marker(s). At most 7 lines total.

Dependents:
{context}

Risk note:"""

if __name__ == "__main__":
    flow = tl.Flow(
        [
            retriever_step(ImpactRetriever(), query_key="symbol",
                           name="impact-evidence"),
            prompt_step(PROMPT, cite="context"),
            default_llm(num_predict=220),
            citations_step(require="warn", exempt=r"^#"),
        ],
        middleware=[MetricsMiddleware(), LineageMiddleware(extract="answer")],
        name="code-intel-impact",
    )

    result = flow.run({"symbol": SYMBOL})

    print("\n=== risk note (markers stripped) ===")
    print(result.output["answer"])
    print_claim_report(result)
    print("workspace:", WORKSPACE, "| symbol:", SYMBOL)
