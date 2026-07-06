"""Live code-QA with claim lineage: surgical_context ask path + citations.

Run (needs a live surgical_context index — see README.md in this directory):

  SC=~/surgical_context
  PYTHONPATH=src:$SC:$SC/mcp_server $SC/.venv/bin/python \
      examples/surgical_context/ask_lineage.py

Flow: retrieve (axis graph+vector, forced intent) -> prompt(cite) -> LLM
      -> citations_step. The join_blame report links every answer line to
      the file:line span of the code it cites.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import throughline as tl
from throughline.adapters.rag import prompt_step, retriever_step
from throughline.modules import LineageMiddleware, MetricsMiddleware, citations_step

from adapter import WORKSPACE, SurgicalRetriever, default_llm, print_claim_report
from surgical_context_mcp.engine import AxisEngine

QUESTION = ("How does axis retrieval rank and prune candidate symbols "
            "before they reach the prompt context?")

PROMPT = """\
Answer the question about this codebase using ONLY the evidence below.
Write a plain list, one line per relevant fact, each line ending with the
[eN] marker(s) of the evidence it uses, exactly like this example:
- Results are deduped per source object, keeping the best score. [e5]
No preamble, no other citation format, at most 7 lines.

Evidence:
{context}

Q: {question}
A:"""

if __name__ == "__main__":
    # Preview the intent classifier, then force its top roles: on this
    # question the top similarities sit just under the 0.20 threshold, so
    # without the override retrieval gets no intent at all.
    probe = AxisEngine()
    preview = probe.classify_intent(QUESTION, top_roles=5, threshold=0.0)
    probe.close()
    print("intent preview:", [(r, round(s, 3)) for r, s, _ in preview[:3]])
    roles = [role for role, _sim, _desc in preview[:3]]

    flow = tl.Flow(
        [
            retriever_step(SurgicalRetriever(roles=roles), top_k=8),
            prompt_step(PROMPT, cite="context"),
            default_llm(num_predict=400),
            citations_step(require="warn", exempt=r"^#"),
        ],
        middleware=[MetricsMiddleware(), LineageMiddleware(extract="answer")],
        name="code-intel-ask",
    )

    result = flow.run({"question": QUESTION})

    print("\n=== answer (markers stripped) ===")
    print(result.output["answer"])
    print_claim_report(result)
    print("workspace:", WORKSPACE)
