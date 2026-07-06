# followers
Follow every step. Trace every line.

**A framework-neutral control plane for LLM/RAG/agent pipelines.**
Zero-dependency core (Python 3.11+, stdlib only). Pluggable presets and
modules for pre/post-processing, validation, metrics, observability and
**line-level lineage**. Third-party RAG components, chains and agents onboard
with one line — no framework imports required.

```
pip install followers            # core: zero dependencies
pip install followers[anthropic] # + Claude adapter
```

## Five concepts

| Concept        | What it is |
|----------------|------------|
| **Step**       | anything callable (or wrappable) that transforms a payload |
| **Flow**       | an ordered chain of steps |
| **Middleware** | pluggable layers around every step: validation, metrics, observability, lineage, retry, cache, quota, your own |
| **Preset**     | a TOML file describing steps + middleware + config; supports `extends` |
| **Context**    | carried through the run; collects events, metrics and artifacts |

## 60-second tour

```python
import followers as fl
from followers.modules import MetricsMiddleware, LineageMiddleware, Validate, Retry, Observe

def normalize(payload):                      # plain function = step
    return {"question": str(payload).strip()}

def answer(payload, ctx):                    # (payload, ctx) also works
    ctx.metric("llm.calls")                  # domain metrics from inside steps
    return {**payload, "answer": f"42 (re: {payload['question']})"}

flow = fl.Flow(
    [normalize, answer],
    middleware=[
        Observe("console"),                                  # events -> stderr
        MetricsMiddleware(),                                 # timings + counters
        Retry(attempts=3, step="answer"),                    # backoff on flaky steps
        Validate(scope="final",                              # final output only (default);
                 schema={"type": "object",                   # intermediate payloads are
                         "required": ["answer"]}),           # not checked. Per-step:
        LineageMiddleware(extract="answer"),                 # Validate(step="normalize")
    ],
    name="qa",
)

result = flow.run("  what is the answer?  ")
result.output          # {'question': ..., 'answer': ...}
result.metrics         # {'counters': {...}, 'observations': {...}}
result.events          # structured event log
result.lineage.blame() # which step wrote every line of the answer
```

Or the same thing declaratively — try the builtin demo (fully offline):

```console
$ followers run demo --input "how does lineage work?" --blame --metrics
$ followers run demo --input "..." --json          # machine-readable report
$ followers presets                                # list discoverable presets
$ followers doctor demo                            # dry-check: resolve every slot, show wrap decisions
$ followers components                             # typed catalog: kind, source, broken plugins
$ followers mcp --preset demo                      # serve flows as MCP tools (stdio)
```

## Presets

```toml
# presets/rag-qa.toml
name = "rag-qa"
extends = "base"                       # deep-merge another preset

[config]
top_k = 3

[[steps]]
uses = "my_project.rag:make_retriever" # import path — no registration needed
name = "retrieve"
[steps.with]                           # kwargs => uses(**kwargs) builds the step
top_k = 3

[[steps]]
uses = "answer"                        # or a name from the registry

[middleware.observe]
[middleware.metrics]
[middleware.retry]
attempts = 3
[middleware.validate]
scope = "final"                        # final output only; "step" = per-step
on_fail = "warn"                       # or "raise"
[middleware.validate.schema]
type = "object"
required = ["answer"]
[middleware.lineage]
extract = "answer"
```

`followers run rag-qa -i "..."` — search order: explicit path → `./presets/` →
`$FOLLOWERS_PRESETS` → builtin. `extends` deep-merges config and middleware;
child steps replace parent steps wholesale. Custom middleware plugs in with
`uses = "pkg.mod:Class"` inside its table.

## Onboarding third-party RAG / chains / agents

`followers.wrap(obj)` duck-types foreign objects — **no framework imports**,
so anything with a recognizable method works, today and for frameworks that
don't exist yet:

| Your object | Detected method | One-liner |
|---|---|---|
| LangChain runnable / LCEL chain / LangGraph app | `invoke` | `fl.wrap(chain)` |
| LlamaIndex query engine | `query` | `fl.wrap(engine, unwrap=lambda r: r.response)` |
| LlamaIndex / LangChain retriever | `retrieve` / `get_relevant_documents` | `fl.wrap(retriever)` |
| Vector store / search client | `search` | `fl.wrap(store)` |
| Agent (most frameworks) | `run` | `fl.wrap(agent)` |
| LLM client | `complete` / `generate` | `fl.wrap(client)` |
| Anything callable | `__call__` | `fl.wrap(fn)` |

Force a method with `method=`, post-process results with `unwrap=`. A whole
external flow (LangGraph graph, LlamaIndex pipeline) is just **one step** of a
followers flow — orchestrate around orchestrators.

For dict-shaped RAG payloads (`{"question"} → +context → +prompt → +answer`),
`followers.adapters.rag` adds ready helpers:

```python
from followers.adapters.rag import retriever_step, prompt_step

flow = fl.Flow([
    retriever_step(any_retriever, top_k=5),   # duck-typed, normalizes doc objects
    prompt_step("Context:\n{context}\n\nQ: {question}"),
    fl.wrap(my_llm_client, unwrap=lambda r: r.content),
])
```

### Registry & pip plugins

```python
@fl.register("clean")                       # kind defaults to "step"
def clean(text): ...

fl.register("redis", RedisCache(), kind="store.cache")   # subkind pins the protocol
```

The registry is a **typed catalog**: the core knows a closed set of built-in
runtime slots — `step`, `middleware`, `store` (an umbrella kind with subkinds
`store.cache` and `store.artifact` pinning the two protocols), `embedder`,
`llm`, `retriever`, `sink`, `verifier` — and never a concrete implementation.
Kind is checked structurally **at the point of use** (preset slots, doctor),
so plugging a store where a step belongs fails at build time with a message
that names both kinds.

The slots are closed; the **taxonomy is not**: plugins introduce namespaced
kinds (`kind="acme.reranker"`), catalog-only by default, enforced if the
author declares a protocol via `fl.register_kind(check=..., shape=...)`.
Bare unknown kinds and built-in namespaces (`store.*`) are rejected loudly.

Pip-installed packages expose many components at once via a *manifest* in the
`followers.plugins` entry-point group:

```python
COMPONENTS = {
    "requires": "followers>=0.1",     # incompatible plugins are skipped, not fatal
    "step:clean": clean,
    "middleware:audit": Audit,
    "store.cache:redis": RedisCache,  # subkind pins the protocol
}
```

Discovery is automatic; name collisions resolve deterministically (builtin <
plugin < local). `followers components` prints everything found, grouped by
kind with sources — including broken plugins and why they failed.

### When duck typing breaks: fast answers

Implicit while it works, explicit when it does not:

- `fl.wrap(obj)` fails **at wrap time** with the full detection trace — what
  was tried, what the object actually has, `Hint: fl.wrap(obj, method='fetch')`.
- `fl.explain(obj)` shows the decision before anything runs: detected method,
  skipped lower-priority candidates.
- `followers doctor my-preset` dry-checks a whole preset — resolves every
  slot, runs detection and kind checks, prints the plan without executing it.
- `fl.modules.StrictOutputs()` catches the distant-failure classic: a
  forgotten `unwrap=` leaks a framework object into the payload and blows up
  three steps later — this middleware names the cause at the step that
  produced it, with the offender's exact path (`$.results[0].docs[3].meta:
  Response`). The "plain data" contract it enforces is formal — see
  ARCHITECTURE for the full definition (recursion, budgets, `allow=`).

### Real LLMs

```python
from followers.adapters.llm import anthropic_chat, from_callable

llm = anthropic_chat(model="claude-opus-4-8", system="Answer briefly.")
# lazy import: pip install followers[anthropic]; token usage lands in metrics

any_llm = from_callable(lambda prompt: my_client.complete(prompt))  # provider-agnostic
```

## Three lineages

followers tracks provenance at three levels, each with its own mechanism and
cost:

| Lineage | Answers | Mechanism | Cost |
|---|---|---|---|
| **edit** | which step wrote each output line | `LineageMiddleware` (difflib) | free, deterministic |
| **evidence** | which source chunks the context came from | `retriever_step` metadata propagation | free, deterministic |
| **claim** | which answer line is backed by which evidence | citation contract + optional verifier | markers are cheap; verification costs tokens (opt-in) |

### Edit lineage

`LineageMiddleware` diffs the textual artifact after every step and keeps a
provenance record per line — `git blame` for pipeline output:

```
$ PYTHONPATH=src python3 examples/demo_rag.py
=== line-level lineage (blame) ===
refine ~   1│ Answer to: Which step wrote every line of this answer?
 draft ~   2│ - Line-level lineage answers which step wrote every line of the output.
 draft +   3│ Sincerely, the pipeline
refine +   4│ Sources: internal corpus
```

Unchanged lines carry their origin, similar lines become modifies (parent
link preserved), new lines are generates. `ledger.blame()` / `trace(id)` /
`to_jsonl()` query and export; `extract=` targets the field to track.

### Evidence & claim lineage

The evidence **contract** is `EvidenceChunk` — text plus explicit provenance:

```python
from followers.modules import EvidenceChunk

class MyRetriever:
    def retrieve(self, query):
        return [EvidenceChunk(
            text="the chunk",
            source={"$artifact": "corpus/abc"},   # path, metadata, ArtifactRef...
            span=(120, 134),                      # where inside the source
            score=0.91,
        )]
```

A retriever that returns chunks states its provenance and skips duck-typing
entirely; foreign doc objects (LangChain, LlamaIndex, dicts) are adapted via
`EvidenceChunk.from_doc` — guessing is the fallback, not the interface.
`str(chunk)` is the text, so chunks drop into templates and joins unchanged;
`to_dict`/`from_dict` carry them across the MCP boundary.

`retriever_step` records every returned chunk in the run's evidence ledger
(id, source, span, score, retriever, step) — automatically, it costs
nothing. `prompt_step(cite="context")` renders chunks with their ids
(`[e1] chunk text`) so the model can cite; `citations_step` then parses and
**deterministically validates** the markers (a citation of unknown evidence
is a violation), strips them from the answer and records line→evidence links.
Generation is stochastic; verification of the links is not.

```python
from followers.modules import citations_step, verify_claims_step

flow = fl.Flow(
    [
        retriever_step(any_retriever, top_k=5),
        prompt_step("Cite sources as [eN].\n{context}\n\nQ: {question}", cite="context"),
        llm,
        citations_step(require="warn",           # policy for uncited lines...
                       exempt=r"^#|^In summary"), # ...headers etc. are fine uncited
        verify_claims_step(my_nli, threshold=0.7),  # opt-in: costs tokens
    ],
    middleware=[LineageMiddleware(extract="answer")],
)
result = flow.run({"question": "..."})
result.ctx.artifacts["claims"].join_blame(     # the three ledgers, joined:
    lineage=result.lineage,                    # who wrote the line, its status,
    evidence=result.ctx.artifacts["evidence"]) # what it cites, where that came from
```

Line statuses keep **facts** and **verdicts** apart: `uncited_line` is a
structural fact (headers and summaries legitimately cite nothing), while
`supported` / `unsupported` / `contradicted` / `low_confidence_support` come
only from the verifier. "Hallucination" is a verifier conclusion, never the
mere absence of a citation. The full taxonomy is in ARCHITECTURE.

## High-load: caching & quotas

Two more boxed middleware for production traffic. **Cache** short-circuits
repeated requests before they reach the heavy LLM/RAG steps; **Quota** stops a
run once its budget is spent — checks fire *before* each step, so the next
expensive call never happens.

```python
from followers.modules import Cache, SemanticCache, Quota

flow = fl.Flow(
    [retrieve, prompt, llm],
    middleware=[
        MetricsMiddleware(),                  # observers first: hits must be visible
        Cache(ttl=600, max_size=2048),        # run-level: a hit skips the whole flow
        # Cache(step="llm*", key="prompt")    # ...or memoize just the LLM step
        Quota(
            limits={"llm.calls": 20, "llm.output_tokens": 100_000},
            cost={"llm.input_tokens": 5e-6, "llm.output_tokens": 25e-6},
            max_cost=0.50,                    # USD budget -> "quota.cost" metric
            max_seconds=30, max_steps=100,
            scope="run",                      # explicit: this budget covers ONE run
            on_exceed="return",               # or "raise" -> QuotaExceeded
            fallback=lambda p, ctx: {**p, "answer": "(budget exhausted)"},
            warn_at=0.8,                      # one-shot "quota_warning" event at 80%
        ),
        # global kill switch: lifetime of this instance, across all runs
        Quota(max_cost=50.0, cost={"llm.output_tokens": 25e-6}, scope="global"),
    ],
)
```

- **Semantic mode**: hand `Cache` any `text -> vector` embedder (an API call,
  sentence-transformers — still zero deps in core) and lookups match on cosine
  similarity: `SemanticCache(embedder=my_embed, threshold=0.92)`. Exact hits
  are checked first. In presets: `embedder = "my_pkg.embeddings:embed"`.
- The cache store lives on the middleware instance (shared across runs);
  `ttl`, LRU `max_size`, `version=` salt for invalidation, `store=` to plug
  Redis-alikes (`get/set(namespace, text, ...)` duck-type). Hit values are
  deep-copied so callers can't corrupt the cache.
- **Purity guard**: a cache hit skips the step — side effects inside it
  silently do not happen. Purity is declarative, not guessed:
  `@fl.step("save", effects="db.write")` (or `effects=` in a preset), and
  `Cache(on_effects="skip"|"raise"|"allow")` enforces the declaration.
  Semantics in ARCHITECTURE.
- Quota reads the same counters steps already report via `ctx.metric()`.
  Budget scope is **explicit**: `scope="run"` (default, robust even against
  a shared metrics collector) or `scope="global"` (lifetime of the
  instance); need both, stack two instances. Details in ARCHITECTURE.
- Both build on the core `fl.EarlyReturn(output)` primitive — raise it from
  any step or hook to finish the run early with `output`. It is never
  retried and never counted as an error.

```toml
[middleware.cache]
ttl = 600
# step = "llm*"                     # omit for run-level short-circuit
# embedder = "my_pkg.emb:embed"     # semantic mode

[middleware.quota]
scope = "run"                       # or "global": lifetime of the instance
max_cost = 0.5
max_seconds = 30
on_exceed = "return"
[middleware.quota.limits]
"llm.calls" = 20
[middleware.quota.cost]
"llm.input_tokens" = 5e-6
"llm.output_tokens" = 25e-6
```

Recommended stack order: **observers first** (`Observe`,
`MetricsMiddleware`), then `Cache`, then everything else. The first
middleware is the outermost layer, and a run-level cache hit short-circuits
everything inside it — with `Cache` outermost the hit would be an invisible
run (no metric, no event). The full reasoning is in ARCHITECTURE.

## Artifact store: control plane vs data plane

Payloads stay small (the control plane); bulk data — corpora, embeddings,
thousand-document reports — lives in an **artifact store** and travels as an
`ArtifactRef` handle:

```python
store = fl.MemoryArtifactStore(default_ttl=1800)
ref = store.put(chunks, session="run-42")     # -> ArtifactRef with a summary
payload = {"question": "...", "corpus": ref}  # payload stays tiny
store.slice(ref, 100, 200)                    # fetch only what you need
store.drop_session("run-42")                  # the GC for cross-boundary data
```

A handle is a **lease**, not a reference: artifacts expire (TTL) and get
evicted (per-session caps); `ArtifactExpired` is a normal condition. A re-run
re-creates the artifact only if the flow is **replayable** — a property of
your flow, not a guarantee of the store (details in ARCHITECTURE). External
backends plug in through the same duck-typed contract, distributed as
`store.artifact:` components.

## MCP: flows as agent tools (optional)

MCP is deliberately **not** part of the core or the adapters: adapters bring
third-party components *into* flows, MCP serves flows *outward* to one
particular protocol. It lives in `followers.contrib.mcp` — still zero
dependencies, but fully detachable: nothing else imports it, and it is one of
possibly many serving layers (HTTP, queue consumers) built on the same public
surface.

```console
$ followers mcp --preset rag-qa        # stdio MCP server, zero dependencies
```

Every preset becomes a tool (`run_rag_qa`); the boundary is a serialized
snapshot with an explicit contract, in both directions:

- **in**: tool arguments are the payload — plain JSON, validated by the
  flow's own middleware;
- **out**: the Result is *projected* — output + metrics + lineage stats under
  a hard byte budget. Oversized outputs land in the artifact store and return
  as a handle + preview; the agent pulls slices via the `get_artifact` tool.
  A gigabyte table cannot reach the model context by construction;
- **trace**: the agent's `trace_id` stamps every event of the run — one trace
  from the agent's reasoning through every step of the graph (OTel bridge
  included via `Observe`).

The reverse direction needs no adapter at all: an agent inside a flow is just
`fl.wrap(agent)` — its budget counted by `Quota`, its output tracked by
lineage, like any other step. In code:
`followers.contrib.mcp.MCPServer(flows={...}, presets=[...])`;
`project_result()` is reusable on its own.

## Reserved boundary: policy / security (future)

Not implemented in v1, but the boundary is pinned now (see ARCHITECTURE):
authz for MCP tool calls, PII redaction, egress rules and prompt-injection
screening belong to a **policy layer** — a middleware sitting *outside*
`Cache`, so a cached hit passes the same checks as a fresh answer. The bare
kind name `policy` is reserved in the registry (custom kinds must be
namespaced, so no plugin can squat it); ecosystem experiments live as
`yourpkg.policy` and can migrate later without collisions.

## Composites & custom middleware

```python
fl.map_step(step, workers=8)                  # fan out over items (threads)
fl.parallel({"a": step_a, "b": step_b})       # same payload, gathered dict
fl.branch(lambda p: p["lang"], {"ru": ru_flow_step, "en": en_step}, default=en_step)

class Audit(fl.Middleware):                   # your own module
    def on_step_end(self, ctx, step, payload, output):
        ctx.emit("audit", step=step.name)
        return output
```

Debugging aids (opt-in): `Snapshots()` records every payload version — the
single sanctioned way to break the core invariant that stock middleware never
retain payload versions between steps. You pay with memory at the plug-in
site, not via defaults.

Middleware hooks: `on_run_start/end`, `on_step_start/end`, `on_step_error`
(return `fl.Handled(value)` to recover), `wrap_step` (full control — retries,
tracing). Raise `fl.EarlyReturn(output)` anywhere to finish the run early;
its exact semantics (what is skipped, what still runs, `ctx.short_circuited`)
are a formal contract — see ARCHITECTURE and `tests/test_early_return.py`.
First middleware in the list is the outermost layer.

## Development

```console
$ PYTHONPATH=src python3 -m unittest discover -s tests   # 253 tests, stdlib only
$ PYTHONPATH=src python3 examples/demo_rag.py            # offline end-to-end demo
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions.
