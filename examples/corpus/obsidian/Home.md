# Home

Throughline is a lightweight orchestration kernel for internal agent and LLM
pipelines. A flow is an ordered chain of steps executed under middleware for
observability, validation, retry, lineage, cache, and quota controls.

Internal documentation RAG should treat the docs vault as the source of truth.
The retriever returns source chunks, the prompt renders those chunks with
evidence ids, and the answer cites the evidence ids so claims can be traced.

See [[RAG Operations]] for the retrieval contract and [[Lineage]] for the
claim-to-source reporting path.
