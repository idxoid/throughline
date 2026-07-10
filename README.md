# throughline
Follow every step. Trace every line.

**A framework-neutral control plane for LLM/RAG/agent pipelines.**
Zero-dependency core (Python 3.11+, stdlib only). Pluggable presets and
modules for pre/post-processing, validation, metrics, observability and
**line-level lineage**. Third-party RAG components, chains and agents onboard
with one line — no framework imports required.

```
pip install throughline            # core: zero dependencies
pip install throughline[anthropic] # + Claude adapter
```

## Five concepts

| Concept        | What it is |
|----------------|------------|
| **Step**       | anything callable (or wrappable) that transforms a payload |
| **Flow**       | an ordered chain of steps |
| **Middleware** | pluggable layers around every step: validation, metrics, observability, lineage, retry, cache, quota, policy, your own |
| **Preset**     | a TOML file describing steps + middleware + config; supports `extends` |
| **Context**    | carried through the run; collects events, metrics and artifacts |

## When to use it

Use throughline when the value of your LLM system is not just the model call,
but the controlled pipeline around it: retrieval, deterministic checks,
structured extraction, report assembly, budgets, cache hits, and line-level
provenance you can inspect after the fact.

It is a good fit when you need to answer:

- which step wrote this output line?
- which source chunk or artifact backed this claim?
- did validation, quota, cache, or retry change the run?
- can I swap my retriever/parser/fetcher without rewriting the flow?

## 60-second tour

```python
import throughline as tl
from throughline.modules import MetricsMiddleware, LineageMiddleware, Validate, Retry, Observe

def normalize(payload):                      # plain function = step
    return {"question": str(payload).strip()}

def answer(payload, ctx):                    # (payload, ctx) also works
    ctx.metric("llm.calls")                  # domain metrics from inside steps
    return {**payload, "answer": f"42 (re: {payload['question']})"}

flow = tl.Flow(
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
$ throughline run demo --input "how does lineage work?" --blame --metrics
$ throughline run demo --input "..." --json          # machine-readable report
$ throughline presets                                # list discoverable presets
$ throughline doctor demo                            # dry-check: resolve every slot, show wrap decisions
$ throughline components                             # typed catalog: kind, source, broken plugins
$ throughline mcp --preset demo                      # optional contrib: serve flows as MCP tools
```

## Hero walkthrough: debug a bad RAG answer

The fastest way to see the point: one offline run over a fully-cited answer
that catches the two lines a human would have missed — a *fabrication* and a
*contradiction*, both wearing real citations.

```console
$ PYTHONPATH=src python3 examples/debug_bad_rag.py
```

```
  ok L1 [e1] conf=1.00  Internal chat logs are retained for 90 days, then purged automatically.
  ok L2 [e2] conf=1.00  They may be used for internal analytics dashboards.
  ?? L3 [e1] conf=0.14  Deleted logs are also archived to cold storage for one year.
  !! L4 [e3] conf=0.90  Logs can be used to train machine learning models after anonymization.
```

L3 cites the retention chunk, which never mentions backups (**unsupported**);
L4 cites the chunk that *forbids* training on logs (**contradicted**). A
citation-count check ships this answer; throughline flags it — with evidence,
citations, verdicts, metrics and line-level lineage in a single run. The full
annotated walkthrough is in **[examples/README.md](examples/README.md)**.

## Example presets

These examples are the fastest way to see what the control plane is for:

| Preset | Use case | Shows |
|---|---|---|
| `rag-docs` | Internal documentation RAG | evidence lineage, citations, semantic cache, quota |
| `report-gen` | Artifact-backed report generation | slots, map steps, report lineage, artifact refs |
| `data-qa` | Data quality assistant | deterministic checks, step validation, strict report schema |
| `doc-extract` | Document extraction pipeline | parser slot, page map, retryable structured extraction |
| `support-agent` | Guarded support bot | intent routing, policy screening/redaction, quota fallback, audit |
| `agent-audit` | AI-agent workflow reproducibility | run manifests ("lockfile"), config drift diff, tool-call trace alignment (first behavioral divergence), multidimensional outcome fingerprint, decision provenance, secret redaction |
| `surgical_context` | Code intelligence / change impact | file:line citations, code QA, real integration |

```bash
THROUGHLINE_PRESETS=examples/presets PYTHONPATH=src:. python3 -m throughline presets

THROUGHLINE_PRESETS=examples/presets PYTHONPATH=src:. \
python3 -m throughline run rag-docs -i "how should answers cite docs?" --json --blame
```

All presets except the last are runnable offline from this repository. The
`surgical_context` example is a live integration under
[`examples/surgical_context/`](examples/surgical_context/) and needs its own
external services.

For deeper mechanics, jump to [presets](#presets), [slots](#slots-presets-with-holes),
[structured output](#structured-output-parse-validate-regenerate),
[lineage](#three-lineages), [cache and quotas](#high-load-caching--quotas),
[artifact store](#artifact-store-control-plane-vs-data-plane), or
[MCP](#mcp-flows-as-agent-tools-optional).

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

`throughline run rag-qa -i "..."` — search order: explicit path → `./presets/` →
`$THROUGHLINE_PRESETS` → builtin. `extends` deep-merges config and middleware;
child steps replace parent steps wholesale. Custom middleware plugs in with
`uses = "pkg.mod:Class"` inside its table.

### Slots: presets with holes

A reusable preset declares the components it *cannot* ship — your retriever,
your LLM — as **slots**, and stays honestly abstract until they are filled:

```toml
[slots.retriever]
kind = "step"                        # contract of whatever fills the hole
description = "team-owned retriever"
# default = "examples.rag_docs:retriever"   # optional: makes the slot optional

[[steps]]
uses = "@retriever"
[steps.with]
top_k = 4
```

`@name` works anywhere a component reference does — step/middleware `uses`,
composite inner refs, and whole-string values inside `[steps.with]` or
middleware options (the **resolved object** is substituted, so factories get
live components; `@@x` escapes a literal `@`). Fills, in ascending
precedence: the slot's `default` → the `[fill]` table (deep-merged through
`extends`) → `load_preset("x", fill={...})` / `--fill` on the CLI:

```toml
extends = "rag-docs-abstract"          # team preset: plug the real component
[fill]
retriever = "acme.search:make_retriever"
```

```console
$ throughline run rag-abstract -i "..." --fill retriever=acme.search:make_retriever
$ throughline doctor rag-abstract      # slots section: filled by what — or missing
```

Building with an unfilled slot fails with the full shopping list (every
missing slot, kind, description); `doctor` reports instead of failing. The
slot's `kind` is checked against what actually ends up in the slot — after
the factory call when the fill is used as a factory.

### Composites in TOML

The Python composites (`tl.map_step` / `parallel` / `branch`), declaratively —
a `[[steps]]` entry takes exactly one of `uses`/`map`/`parallel`/`branch`:

```toml
[[steps]]
map = "my_pkg.report:section_step"   # fan out over an iterable payload
workers = 4
[steps.with]                         # optional: the INNER factory's kwargs
style = "brief"

[[steps]]
[steps.parallel]                     # same payload to every entry -> dict
summary = "my_pkg:summarize"
stats = "my_pkg:stats"

[[steps]]
[steps.branch]
selector = "lang"                    # payload key; "pkg.mod:fn" (with ':') imports
default = "my_pkg:en_step"
[steps.branch.routes]
ru = "my_pkg:ru_step"
en = "my_pkg:en_step"
```

Inner refs (including `"@slot"`) are used directly — no per-route factory
call; wrap factories in your own module.

## Onboarding third-party RAG / chains / agents

`throughline.wrap(obj)` duck-types foreign objects — **no framework imports**,
so anything with a recognizable method works, today and for frameworks that
don't exist yet:

| Your object | Detected method | One-liner |
|---|---|---|
| LangChain runnable / LCEL chain / LangGraph app | `invoke` | `tl.wrap(chain)` |
| LlamaIndex query engine | `query` | `tl.wrap(engine, unwrap=lambda r: r.response)` |
| LlamaIndex / LangChain retriever | `retrieve` / `get_relevant_documents` | `tl.wrap(retriever)` |
| Vector store / search client | `search` | `tl.wrap(store)` |
| Agent (most frameworks) | `run` | `tl.wrap(agent)` |
| LLM client | `complete` / `generate` | `tl.wrap(client)` |
| Anything callable | `__call__` | `tl.wrap(fn)` |

Force a method with `method=`, post-process results with `unwrap=`. A whole
external flow (LangGraph graph, LlamaIndex pipeline) is just **one step** of a
throughline flow — orchestrate around orchestrators.

For dict-shaped RAG payloads (`{"question"} → +context → +prompt → +answer`),
`throughline.adapters.rag` adds ready helpers:

```python
from throughline.adapters.rag import retriever_step, prompt_step

flow = tl.Flow([
    retriever_step(any_retriever, top_k=5),   # duck-typed, normalizes doc objects
    prompt_step("Context:\n{context}\n\nQ: {question}"),
    tl.wrap(my_llm_client, unwrap=lambda r: r.content),
])
```

### Registry & pip plugins

```python
@tl.register("clean")                       # kind defaults to "step"
def clean(text): ...

tl.register("redis", RedisCache(), kind="store.cache")   # subkind pins the protocol
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
author declares a protocol via `tl.register_kind(check=..., shape=...)`.
Bare unknown kinds and built-in namespaces (`store.*`) are rejected loudly.

Pip-installed packages expose many components at once via a *manifest* in the
`throughline.plugins` entry-point group:

```python
COMPONENTS = {
    "requires": "throughline>=0.1",     # incompatible plugins are skipped, not fatal
    "step:clean": clean,
    "middleware:audit": Audit,
    "store.cache:redis": RedisCache,  # subkind pins the protocol
}
```

Discovery is automatic; name collisions resolve deterministically (builtin <
plugin < local). `throughline components` prints everything found, grouped by
kind with sources — including broken plugins and why they failed.

### When duck typing breaks: fast answers

Implicit while it works, explicit when it does not:

- `tl.wrap(obj)` fails **at wrap time** with the full detection trace — what
  was tried, what the object actually has, `Hint: tl.wrap(obj, method='fetch')`.
- `tl.explain(obj)` shows the decision before anything runs: detected method,
  skipped lower-priority candidates.
- `throughline doctor my-preset` dry-checks a whole preset — resolves every
  slot, runs detection and kind checks, prints the plan without executing it.
- `tl.modules.StrictOutputs()` catches the distant-failure classic: a
  forgotten `unwrap=` leaks a framework object into the payload and blows up
  three steps later — this middleware names the cause at the step that
  produced it, with the offender's exact path (`$.results[0].docs[3].meta:
  Response`). The "plain data" contract it enforces is formal — see
  ARCHITECTURE for the full definition (recursion, budgets, `allow=`).

### Real LLMs

```python
from throughline.adapters.llm import anthropic_chat, from_callable

llm = anthropic_chat(model="claude-opus-4-8", system="Answer briefly.")
# lazy import: pip install throughline[anthropic]; token usage lands in metrics

any_llm = from_callable(lambda prompt: my_client.complete(prompt))  # provider-agnostic
```

### Structured output: parse, validate, regenerate

```python
from throughline.modules import json_step, structured_step, Retry

flow = tl.Flow([llm, json_step(schema={"type": "object", "required": ["name"]})])
```

`json_step` parses `payload["answer"]` tolerantly (code fences, prose-wrapped
JSON) and checks it against the same schema dialect `Validate` speaks;
`on_fail="warn"` records a violation instead of raising.

When an invalid answer should **regenerate**, parsing must live inside the
retried step — per-step `Validate` raises in `on_step_end`, which runs
*after* the `wrap_step` onion, so `Retry` never sees it (this non-composition
is pinned in `tests/test_structured.py`). `structured_step` fuses
generator + parse + schema into one step:

```python
flow = tl.Flow(
    [structured_step(llm, schema={"type": "object", "required": ["name"]},
                     name="extract")],
    middleware=[Retry(attempts=3, step="extract")],   # re-runs the LLM
)
```

## Three lineages

throughline tracks provenance at three levels, each with its own mechanism and
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
from throughline.modules import EvidenceChunk

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
from throughline.modules import citations_step, verify_claims_step

flow = tl.Flow(
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
from throughline.modules import Cache, SemanticCache, Quota

flow = tl.Flow(
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
  `@tl.step("save", effects="db.write")` (or `effects=` in a preset), and
  `Cache(on_effects="skip"|"raise"|"allow")` enforces the declaration.
  Semantics in ARCHITECTURE.
- Quota reads the same counters steps already report via `ctx.metric()`.
  Budget scope is **explicit**: `scope="run"` (default, robust even against
  a shared metrics collector) or `scope="global"` (lifetime of the
  instance); need both, stack two instances. Details in ARCHITECTURE.
- Both build on the core `tl.EarlyReturn(output)` primitive — raise it from
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
`MetricsMiddleware`), then `Policy`, then `Cache`, then everything else. The
first middleware is the outermost layer, and a run-level cache hit
short-circuits everything inside it — with `Cache` outermost the hit would be an invisible
run (no metric, no event). Final `Validate` may sit outside `Policy` when the
public post-redaction/fallback shape must be checked. The full reasoning is in
ARCHITECTURE.

## Artifact store: control plane vs data plane

Payloads stay small (the control plane); bulk data — corpora, embeddings,
thousand-document reports — lives in an **artifact store** and travels as an
`ArtifactRef` handle:

```python
store = tl.MemoryArtifactStore(default_ttl=1800)
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
particular protocol. It lives in `throughline.contrib.mcp` — still zero
dependencies, but fully detachable: nothing else imports it, and it is one of
possibly many serving layers (HTTP, queue consumers) built on the same public
surface.

```console
$ throughline mcp --preset rag-qa        # stdio MCP server, zero dependencies
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
`tl.wrap(agent)` — its budget counted by `Quota`, its output tracked by
lineage, like any other step. In code:
`throughline.contrib.mcp.MCPServer(flows={...}, presets=[...])`;
`project_result()` is reusable on its own.

### MCP servers as steps (client direction)

`throughline.adapters.mcp` is the mirror image — any external MCP server
(stdio) becomes a step, so the existing ecosystem of MCP tools plugs into
flows, budgeted by `Quota` and cached like any component. Zero dependencies:
subprocess + newline-delimited JSON-RPC.

```python
from throughline.adapters.mcp import MCPClient, tool_step

client = MCPClient(["python", "-m", "some_mcp_server"])
flow = tl.Flow([
    tool_step(client, "search_docs", params=["question"], out_key="context"),
    prompt_step("Context:\n{context}\n\nQ: {question}"),
    llm,
])
```

A dict payload becomes the tool arguments (`params=` selects keys, `args=`
adds static ones; `build=` — a `payload -> arguments` callable or its import
path — replaces the mapping when the server's schema differs from the
payload shape). The result — `structuredContent` when the server provides
it, else text — lands at `out_key`, post-processed by `unwrap=`. In presets,
`mcp_tool` owns its client: `uses = "throughline.adapters.mcp:mcp_tool"` with
`command = [...]` and `tool = "..."` in `[steps.with]`.

## Policy: screening, redaction, audit

The formerly reserved policy/security boundary is now a protocol. A rule is
the `kind="policy"` component — `(checkpoint, value, ctx) -> verdict`, where
the verdict is `None` (abstain), `Allow`, `Deny`, `Transform` (redaction) or
`Flag` (record, don't block):

```python
from throughline.modules import Policy, Deny, Transform, screen_with

def redact_emails(checkpoint, value, ctx):
    text, n = EMAIL_RE.subn("[redacted]", value["answer"])
    return Transform({**value, "answer": text}, f"{n} emails") if n else None

flow = tl.Flow(
    [retrieve, prompt, llm],
    middleware=[
        Observe(), MetricsMiddleware(),       # observers: a deny must be visible
        Policy(
            ingress=[screen_with(judge)],     # any kind="verifier" as a rule
            egress=[redact_emails],
            on_deny="return",                 # or "raise" -> PolicyError
            fallback={"answer": "handing you to an operator"},
        ),
        Cache(ttl=600),                       # inside Policy — see below
    ],
)
```

- **Abstain ≠ allow.** What silence means is checkpoint config, not rule
  discipline: `default="allow"` (fail-open screening) or `default="deny"`
  (fail-closed authz — some rule must return an explicit `Allow`). A
  forgotten `return` in a rule can neither block traffic nor open a hole.
  Deny always wins over an earlier Allow; need both postures, stack two
  instances (the Quota pattern).
- **Position is pinned by tests**: Observe/Metrics → Policy → Cache. Egress
  runs in the `on_run_end` finalizer sweep, so **a cache hit passes the same
  redaction as a fresh answer**; a denied request fails before `on_run_end`,
  so **nothing denied is ever cached**. One hazard is documented: `Transform`
  on *ingress* feeds the cache key — strip identifying fields on egress, not
  ingress, or keep them in `key=`. A final public-schema `Validate` can wrap
  `Policy` when it must check the post-redaction/fallback output.
- **Audit is data**: every decision is an event (`policy_denied` /
  `policy_redacted` / `policy_flagged` / `policy_allowed`), a counter and a
  record in `ctx.artifacts["policy"]` — reasons only, never payloads.
- **Honest injection screening**: no regex pretending to detect prompt
  injection. `screen_with(verifier)` adapts an LLM/NLI judge
  (`kind="verifier"`) into a rule — it costs tokens, goes through Quota and
  is cacheable.

```toml
[middleware.policy]
ingress = ["my_pkg.rules:screen_injection"]
egress = ["my_pkg.rules:redact_emails"]
on_deny = "return"
fallback = { answer = "I can't help with that — connecting you with an operator." }
```

The `support-agent` example preset is the live showcase: intent routing,
FAQ/RAG/escalation branches, policy screening/redaction and quota fallback.
Phase 2 (per-step checkpoints, MCP `tools/call` authz, event redaction in
Observe sinks) is pinned in ARCHITECTURE's reserved boundaries.

## Composites & custom middleware

```python
tl.map_step(step, workers=8)                  # fan out over items (threads)
tl.parallel({"a": step_a, "b": step_b})       # same payload, gathered dict
tl.branch(lambda p: p["lang"], {"ru": ru_flow_step, "en": en_step}, default=en_step)

class Audit(tl.Middleware):                   # your own module
    def on_step_end(self, ctx, step, payload, output):
        ctx.emit("audit", step=step.name)
        return output
```

Debugging aids (opt-in): `Snapshots()` records every payload version — the
single sanctioned way to break the core invariant that stock middleware never
retain payload versions between steps. You pay with memory at the plug-in
site, not via defaults.

Middleware hooks: `on_run_start/end`, `on_step_start/end`, `on_step_error`
(return `tl.Handled(value)` to recover), `wrap_step` (full control — retries,
tracing). Raise `tl.EarlyReturn(output)` anywhere to finish the run early;
its exact semantics (what is skipped, what still runs, `ctx.short_circuited`)
are a formal contract — see ARCHITECTURE and `tests/test_early_return.py`.
First middleware in the list is the outermost layer.

## Development

```console
$ PYTHONPATH=src python3 -m unittest discover -s tests   # stdlib-only test suite
$ PYTHONPATH=src pytest -q tests                         # ...or via pytest ([dev] extra)
$ PYTHONPATH=src python3 examples/demo_rag.py            # offline end-to-end demo
```

[examples/surgical_context/](examples/surgical_context/) is a *live* integration
example: code-QA and change-impact flows over a real code graph
([surgical_context](https://github.com/idxoid/surgical_context)), where every
answer line cites a validated `file:line` span — evidence & claim lineage
end-to-end. Needs external services; see its README.

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions.
