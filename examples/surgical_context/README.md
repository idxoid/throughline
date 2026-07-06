# surgical_context integration: code-intel with claim lineage

A live integration example — unlike `examples/demo_rag.py` it is **not**
offline. It plugs [surgical_context](https://github.com/idxoid/surgical_context)
(a local-first code context engine: Neo4j call graph + LanceDB vectors) into
throughline's evidence contract, giving code-QA answers where **every line
cites a real `file:line` span**, validated deterministically.

| File | Flow | Shows |
|---|---|---|
| `adapter.py` | — | `SurgicalRetriever` (ask path), `ImpactRetriever` (impact path), local Ollama LLM step |
| `ask_lineage.py` | retrieve → prompt(cite) → LLM → citations | evidence + claim lineage over graph-expanded retrieval; forcing intent roles |
| `impact_note.py` | impact → prompt(cite) → LLM → citations | a flow whose facts are 100% deterministic (pure Neo4j walk); the LLM only summarizes |

## Prerequisites

- a surgical_context checkout with its services up (Neo4j, LanceDB) and the
  target repo indexed under `axis_python_v1`;
- its virtualenv (`$SC/.venv`) — throughline itself stays zero-dependency,
  the example runs inside surgical_context's environment;
- an LLM: [Ollama](https://ollama.com) with any local model
  (`TL_OLLAMA_MODEL`, default `qwen3:4b` — the prefix avoids the
  `OLLAMA_MODEL` that surgical_context's own `.env` exports), or `REAL_LLM=1`
  for Anthropic (needs `ANTHROPIC_API_KEY` with credits).

## Run

```console
$ SC=~/surgical_context
$ PYTHONPATH=src:$SC:$SC/mcp_server $SC/.venv/bin/python \
      examples/surgical_context/ask_lineage.py
$ PYTHONPATH=src:$SC:$SC/mcp_server $SC/.venv/bin/python \
      examples/surgical_context/impact_note.py run_axis_retrieval
```

Environment knobs: `SURGICAL_WORKSPACE` (default
`qa_repo/surgical_context@main`), `TL_OLLAMA_MODEL`, `TL_OLLAMA_URL`,
`REAL_LLM=1`.

Expected tail of `impact_note.py` — the join of edit lineage, claims and
evidence:

```
[cited        ] e3        - assemble_and_ask: calls run_axis_retrieval directly.
               -> QA/run_demo.py:113-132 (d1)
[cited        ] e5        - ask_axis: builds the /ask/axis response from it.
               -> context_engine/ask/service.py:664-683 (d1)
[uncited_line ] -         Wait, let me re-read the problem.   <- model drift, caught by policy
```

## Lessons baked into the code

- **Evidence must show the call site.** `ImpactRetriever` centers each chunk's
  window on the line that mentions the target symbol; with a "first N lines"
  trim the model rightly denies the dependency it cannot see.
- **Exact spans come from `read_symbol`,** not from the ask path's packer:
  packer-rendered chunks carry the full symbol span but trimmed text.
- **Small thinking-tuned models need a prefill.** See `ollama_llm` in
  `adapter.py`: an assistant prefill skips the thinking phase; a tight
  `num_predict` cuts the rambling tail; whatever slips through is flagged
  by `citations_step` as `uncited_line` — the guardrail earns its keep on
  weak models, not strong ones.
- **Forcing intent is legitimate.** The embedding classifier's top roles can
  sit just under its threshold; `SurgicalRetriever(roles=[...])` passes them
  anyway (preview with `AxisEngine.classify_intent`).
