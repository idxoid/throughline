# RAG Operations

The RAG payload convention is a dictionary that starts as `{"question": ...}`.
The retriever adds `context`, the prompt step adds `prompt`, and the LLM step
adds `answer`. Downstream middleware can validate only the final answer while
still preserving the intermediate keys for debugging.

Production docs RAG usually replaces the example keyword retriever with a
team-owned retriever via a preset `uses` path. The replacement should return
chunks with explicit provenance whenever possible, including a source path,
span, and score.

Repeated documentation questions are good candidates for SemanticCache. Cache
by the normalized question so near-duplicate questions can reuse the completed
run instead of paying for retrieval and generation again.
