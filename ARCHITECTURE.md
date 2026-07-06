# throughline — architecture

Goal: a lightweight orchestrator for agents and LLM pipelines with
pluggable presets and modules (pre/post-processing, validation, metrics,
observability, line-level lineage) and low-friction onboarding of third-party
RAG components and flows.

## Principles

1. **Lightness = zero dependencies + five concepts.** The core is pure stdlib
   (Python 3.11+: `tomllib`, `difflib`, `importlib.metadata`). Everything
   optional (anthropic, jsonschema, pydantic, otel) is imported lazily and
   degrades gracefully. The API surface: Step, Flow, Middleware, Preset,
   Context — and nothing else.
2. **Universality through duck typing, not integrations.** We do not import
   LangChain/LlamaIndex and do not chase their versions: the adapter looks at
   the object itself (`invoke`/`query`/`retrieve`/`run`/...). Any future
   framework with a similar signature onboards without changing throughline.
3. **Everything cross-cutting is middleware.** Validation, metrics,
   observability, lineage, retry are not baked into the core — they attach as
   layers. Your own module = a class with 1–2 methods.
4. **Declarativity is optional.** The same Flow is built in code or from a
   TOML preset; presets inherit (`extends`), components are addressed by a
   registry name or an import path (`pkg.mod:attr`) without registration.

## Layers

```
        ┌───── CLI (throughline run/presets/steps/components/doctor/mcp) ──┐
        │              presets.py (TOML -> Flow, inspect_preset)           │
        ├───────────────────────────────────────────────────────────────────
core    │  flow.py     Flow: a chain of steps + an onion of middleware     │
        │  step.py     Step, as_step, map_step / parallel / branch         │
        │  middleware  hooks: on_run_*, on_step_*, wrap_step, Handled      │
        │  context.py  RunContext(events, state, artifacts), Result        │
        │  registry.py typed catalog: kinds, manifests, precedence,        │
        │              check_kind, entry points                            │
        │  store.py    ArtifactRef (lease) + MemoryArtifactStore           │
        ├───────────────────────────────────────────────────────────────────
modules │  metrics     counters + observations, per-step timings           │
        │  observe     sinks (console/jsonl/memory) + OTel bridge          │
        │  validate    schema-subset | jsonschema | pydantic | predicate   │
        │  lineage     edit lineage: difflib diffs, blame/trace/jsonl      │
        │  citations   evidence + claim lineage, join_blame, verifier      │
        │  retry       backoff, fnmatch scoping by step name               │
        │  cache       LRU+TTL / semantic, run|step, purity guard          │
        │  quota       budgets: counters/cost/seconds/steps, scope=run|global │
        │  debug       Snapshots (opt-in history), StrictOutputs           │
        ├───────────────────────────────────────────────────────────────────
adapters│  wrap()      duck typing + explain()/WrapError with a trace      │
        │  llm         anthropic (lazy), from_callable, FakeLLM            │
        │  rag         retriever_step (+evidence), prompt_step (+cite)     │
        ├───────────────────────────────────────────────────────────────────
contrib │  demo        offline steps for demos and smoke tests             │
(opt)   │  mcp         serving layer: flows-as-tools, projection, handles  │
        └───────────────────────────────────────────────────────────────────
```

MCP lives in contrib, not in adapters, deliberately: adapters bring foreign
components *into* a flow, MCP serves flows *outward* to one particular
protocol. The core and the adapters never import it (the `mcp` CLI command
loads it lazily) — the module can be deleted and the framework won't notice.

## Key decisions

### Data flow
One payload travels down the chain; the context (`RunContext`) is a separate
channel: `events` (the bus), `state` (user scratch space), `artifacts` (what
middleware publish: metrics, lineage, events). `Result` is a facade over the
context. A step failure is wrapped into `FlowError` with `step` and `ctx` —
the metrics/events collected so far stay inspectable even on failure.

### Onion middleware
The first middleware in the list is the outermost layer. Call order:
`on_step_start` in list order → `wrap_step` (outer wraps inner) →
`on_step_end` in reverse order. `on_step_error` may return `Handled(value)`
and extinguish the error. Observers (metrics/observe) return nothing —
`None` is read as "unchanged", so a forgotten `return` cannot break them.

Consequence for ordering: **observers on the outside, short-circuiters
inside them**. An EarlyReturn from `on_run_start` (run-level Cache) skips the
hooks of every middleware to its right in the list; if Cache were outermost,
a hit would fire before Observe subscribed its sinks and Metrics attached
its collector — and become invisible. Hence the recommended stack:
Observe/Metrics → Cache → Quota/Retry/Validate/Lineage (a hit spends none of
Quota's budget, yet shows up in metrics and events). `run_started` is
emitted by the core after the `on_run_start` hooks — so that Observe's
subscribers actually see it; a short-circuited run announces itself as
`run_short_circuited` instead.

### The EarlyReturn / on_run_end contract (formal)
Pinned down in the docstrings of Flow.run / Middleware / EarlyReturn and in
tests/test_early_return.py — changing those tests means changing the
semantics for every cache/quota/debug module.

- EarlyReturn **skips**: the remaining `on_run_start` hooks, the remaining
  steps, and — if raised mid-step — that step's `on_step_end` hooks.
- EarlyReturn **bypasses**: `on_step_error`, Retry, error counters — it is
  control flow, not an error; every `wrap_step` must re-raise it untouched.
- `on_run_end` is a **finalizer sweep, not stack unwinding**: it executes
  for every middleware exactly once, in reverse order, on success and on
  EarlyReturn alike — even if that middleware's `on_run_start` never got to
  run (everything inside a run-level Cache, on a hit). Consequences for a
  module author: `on_run_end` has no right to assume its state exists
  (`ctx.artifacts.get`/`setdefault` — see Cache, Quota);
  `ctx.short_circuited` says the output came from an EarlyReturn — if the
  module must account for skipped work, it does so here (Lineage attributes
  the substituted output to the `early_return` step, Snapshots appends it to
  the trail).
- A real error (any other exception): `on_run_end` does **not** run;
  FlowError carries the ctx with everything collected.

### Quota: budget scope is explicit, not a side effect
Previously a "global budget" arose implicitly: share one Metrics collector
across runs — and per-run limits silently became lifetime limits. It was the
configuration of a *different* middleware changing Quota's semantics. Now
scope is a property of the Quota itself:

- `scope="run"` (default): the budget covers one run. Robust by
  construction — a baseline snapshot of the counters is taken at
  `on_run_start`, and consumption is measured as a delta against it. A
  shared collector can no longer silently change what the limits mean.
- `scope="global"`: the lifetime of the middleware instance. Consumption of
  finished runs is folded into the instance under a lock in `on_run_end`;
  concurrent in-flight runs are counted approximately (only their own
  delta). "seconds" means cumulative time spent inside runs, "steps" means
  total steps, "quota.cost" is the lifetime figure.

Mixed requirements (a per-request ceiling + a global kill switch) are two
Quota instances in the stack: scope composes like everything else instead of
turning into a parameter matrix inside one middleware. `QuotaExceeded` and
the `quota_exceeded`/`quota_warning` events carry `scope` — alerts can tell
"an expensive request" from "the daily budget ran out".

### Cache: purity guard — a cache hit skips side effects
A hit skips the step (or the whole flow) entirely — a database write, an
email, a webhook inside it silently do not happen. That is not a cache bug,
it is the cache's definition, so the guard cannot be a heuristic: purity is
statically undecidable, and guessing would give false confidence. The
contract is declarative — the step declares its own effects
(`@tl.step("save", effects="db.write")`, `effects="pure"`, `effects = "..."`
in a preset's `[[steps]]`; None = undeclared), the declaration lives in
`Step.meta["effects"]` and survives rename/presets.

Cache applies the declaration via `on_effects`:
- `"skip"` (default): a declared-effectful step is never served from the
  cache and never written to it. Step-level — the step simply executes past
  the cache; run-level — a run containing such a step is not stored (and a
  run that was never stored can never become a hit: the guard works even
  though on_run_start cannot see the steps yet). A `cache_effects_bypass`
  event + metric say why.
- `"raise"`: the combination "cache over an effectful step" is a
  configuration error; the run fails with an explanation. For pipelines
  where a skipped write is unacceptable.
- `"allow"`: an explicit opt-in (idempotent effects).

Undeclared steps are cached as before: the guard trusts declarations rather
than guessing — otherwise every callable would have to be presumed
effectful and the cache would be dead on arrival.

### Line-level lineage
After every step, difflib compares the textual form of the artifact against
the previous version: equal → carry (the record and its origin are kept),
replace → pairwise matching by similarity ≥ 0.5 → modify (with a parent
link) or generate, insert → generate, delete → drop (recorded against the
step). The result is exactly the git-blame model: for every line of the
final output you know the last writer, the root origin and the full ancestry
chain (`blame()/trace()/to_jsonl()`). `extract="answer"` points the ledger
at the right field of a dict payload. Worst-case complexity is
O(steps × lines²) — acceptable for textual artifacts; the threshold is
tunable.

### Three lineages, three mechanisms
Edit lineage (who wrote the line) — difflib, deterministic, free, on by
default. Evidence lineage (where the context came from) — the EvidenceChunk
contract (text + source + span + score): your own retriever states its
provenance explicitly, foreign doc objects are adapted via from_doc —
guessing is the fallback, not the interface. str(chunk) = text, so chunks
don't break code that expects strings; to_dict/from_dict carry them across
the MCP boundary. retriever_step writes chunks into the EvidenceLedger,
prompt_step(cite=...) renders them with [eN] identifiers; all deterministic
and free. Claim lineage (which answer line is backed by what) is
fundamentally stochastic: the link is created by the LLM through the
citation contract, but *validating* the links (citations_step) is
deterministic — a citation of nonexistent evidence is a violation.
verify_claims_step (NLI / LLM judge) is a separate opt-in knob because it
burns tokens; it goes through Quota and is cached like any other step. Three
ledgers with shared keys rather than one mega-module: the cost and the
on/off state of each are visible separately. join_blame assembles the full
per-line picture with a join.

Line statuses keep **facts** and **verdicts** apart. `uncited_line` is a
structural fact, not an accusation: headers, transitions, summaries and
style-mandated phrasing legitimately cite nothing (`exempt=` is the
allowlist, `require=` is the policy for the rest). Verdicts — `supported` /
`low_confidence_support` / `unsupported` / `contradicted` — are issued only
by the verifier (a score, a string, or an NLI-style dict
`{"verdict", "confidence"}`; a scalar cannot distinguish "no support" from
"weak support", so a score below the threshold means low_confidence_support,
while unsupported/contradicted require a verdict-capable verifier).
"Hallucination" is a conclusion drawn from unsupported/contradicted — never
from the mere absence of a citation. `fail_on=` selects which verdicts count
as violations.

### Control plane / data plane
The payload is the control plane and stays a small dict. Heavy data
(corpora, embeddings, reports) lives in the ArtifactStore and travels by
reference (ArtifactRef). A handle is a lease, not a reference: TTL +
eviction by session caps; ArtifactExpired is a normal condition, not a bug.
A re-run re-creates the artifact only if the flow is **replayable**: same
inputs/config/sources, stochastic steps cached or pinned. Replayability is a
property of the specific flow, not a guarantee of the store (a run-level
Cache is the cheapest way to make an LLM flow replayable within the cache
TTL); otherwise the caller must handle expiry explicitly — a longer TTL,
persisting the output, or accepting the loss. A session namespace dropped
wholesale is the garbage collector for data whose consumers (agents) live
outside the process and are invisible to refcounting. Core invariant: stock
middleware never retain payload versions between steps (1–2 versions are
alive at any moment regardless of pipeline length); pinned by a weakref
test. Snapshots is the single sanctioned way to break it, consciously. An
Arrow/Parquet backend is a kind=store plugin, not core.

### Typed registry
The core knows a closed set of **builtin slots** (step/middleware/store/
embedder/llm/retriever/sink/verifier) — structural protocols, not base
classes. Kind is checked at the point of use (a preset slot, doctor) at
build time: the error names both kinds ("registered as store, not step").
Plugins export a manifest `{"kind:name": obj}` through a single entry point;
`requires = "throughline>=X.Y"` filters out incompatible ones without
crashing the host; a broken plugin shows up in `throughline components`
with its reason instead of killing discovery.

`store` is deliberately an **umbrella kind**: two different protocols hide
behind one word — the cache store (`get/set(namespace, text)`) and the
artifact store (`put -> ref, get(ref)`). The umbrella accepts either; the
subkinds `store.cache` / `store.artifact` pin the exact protocol — a Redis
cache is not an artifact store, and registering under the subkind makes
that checkable. Namespaces of builtin kinds (`store.*`, `step.*`, ...) are
reserved for the core: `register_kind("store.anything")` is rejected —
custom kinds live in their package's namespace.
Collisions are deterministic: builtin < plugin < local.

The slots are closed, but the taxonomy is not: a plugin may introduce its
own namespaced kind (`acme.reranker`) — the core catalogs it (resolve,
components) but enforces nothing until the kind's author declares a protocol
via `register_kind(kind, check=..., shape=...)`. A bare unknown kind is
rejected at registration with a namespacing hint — typos fail loudly,
extension stays open. The namespace guarantees a custom kind can never
collide with a current or future builtin slot. The principle: strict at the
point of use, open at the point of definition.

### Duck typing that explains itself
Implicit while it works, explicit when it does not: wrap() fails at wrap
time (not on the first call) with the full detection trace — what was
tried, what the object actually has, how to force a method. explain() shows
the decision before anything runs; `throughline doctor` runs
resolution+detection+kind checks across a whole preset without executing
it. StrictOutputs catches "it crashed far from the cause": a foreign object
(a forgotten `unwrap=`) is named at the step that produced it, with the
exact path to the offender (`$.results[0].meta`). The "plain data" contract
is formal: scalars + dict/list/tuple recursively (cycle-safe, a max_nodes
budget with a truncation event) + ArtifactRef/EvidenceChunk + your own
types via `allow=`; `step=` scoping by fnmatch; early-returned output,
which bypasses on_step_end per the EarlyReturn contract, is checked in
on_run_end.

### The agent ↔ flow boundary (MCP — optional, contrib)
The *boundary* itself is a core architectural decision; MCP is merely one
transport for it, which is why it lives in `throughline.contrib.mcp` and is
never imported by the core. Boundary principles: no shared mutable state
between the two sides — always a serialized snapshot. Inbound: tool
arguments are the payload (JSON), validated by the flow's own middleware.
Outbound: a projection of the Result (output + metrics + lineage stats)
under a hard byte budget; oversized data goes to the ArtifactStore and
returns as a handle with a preview — a gigabyte cannot reach the model
context by construction (get_artifact serves slices). The agent's trace_id
stamps every event of the run (TracedEventBus) — one trace from the agent's
reasoning down to every step of the graph. The transport is stdio JSON-RPC,
written by hand (~a hundred lines), zero-dep; handle() is a pure, testable
function. Another serving layer (HTTP, a queue) builds on the same public
surface — presets, Flow.run, the Result projection, the store — touching
neither the core nor MCP.

### Presets
`tomllib` + lookup across `./presets`, `$THROUGHLINE_PRESETS`, builtin.
`extends` deep-merges config/middleware; steps are replaced wholesale (step
order is the essence of the pipeline — merging it is dangerous).
`[steps.with]` (even an empty table) means "call the factory"; its absence
means "this is already a ready step". Middleware are enabled by the presence
of their table, disabled with `enabled = false`; third-party ones plug in
with `uses = "pkg.mod:Class"`.

### Third-party onboarding — three paths
1. **wrap()/as_step** — the object already exists in code: one line.
2. **An import path in a preset** — the code is not touched at all:
   `uses = "their.module:thing"`.
3. **Entry points** (`throughline.plugins`) — a pip package publishes
   components, the registry picks them up automatically.
Plus the dict-payload convention for RAG (`question/context/prompt/answer`)
— the `rag.py` adapters follow it, so retrievers are interchangeable.

### Async
The core is synchronous; coroutine steps are bridged via `asyncio.run` (with
a clear error if a loop is already running outside). A full async runner is
deliberately out of v1: it doubles the surface, and fan-out is covered by
`map_step(workers=N)`.

## Reserved boundaries (future)

### Policy / security
Not implemented in v1, but the boundary is pinned now so that the ecosystem
does not have to be broken later. What belongs here: authz on tool
invocation (who may call which flow through MCP `tools/call`), PII redaction
in payloads and events, egress rules (what data may leave the perimeter),
prompt-injection screening on input.

What is fixed already today:

- **The slot name.** The bare name `policy` is reserved in the registry:
  custom kinds must be namespaced, so no plugin can squat `policy` before
  the core defines its protocol. Ecosystem experiments live as `acme.policy`
  and can migrate later without collisions.
- **The position in the stack.** Policy sits *outside* Cache: a cache hit
  must pass the same checks as a fresh answer (a cached response served
  without authz/redaction is the classic hole). Relative to observers the
  order depends on whether the logs themselves get redacted: policy outside
  Observe = redaction is visible in events too; inside = events stay raw.
  That is a decision for the future protocol; both positions are already
  legal for middleware today.
- **The mechanism.** Policy is middleware (on_run_start for input
  screening, on_run_end for output redaction, wrap_step for per-step rules)
  plus a hook in MCP `tools/call` for authz before the flow runs. No new
  core primitives are required — the boundary already exists; only the name
  and the position are reserved.

## Non-goals for v1
A general-purpose DAG engine (composites cover 90%), state persistence
(RunContext is ephemeral; only artifacts in the store persist), distributed
execution, a UI. All of it can be layered on top without changing the core —
events and artifacts are already structured. Policy/security — see
"Reserved boundaries" above: a non-goal as an implementation, but the
boundary is pinned.

## Verification
`tests/` — unit and integration tests on stdlib `unittest`
(core, modules, the three lineages, the store with lease semantics, the
typed registry, wrap/doctor diagnostics, the weakref payload-retention
invariant, the MCP protocol, presets with extends, CLI end-to-end).
`examples/demo_rag.py` — an offline demo of every module at once.
