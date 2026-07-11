# Debug a bad RAG answer with throughline

The failure every RAG team ships at least once: the model returns a fluent,
**fully-cited** answer — and some of it is wrong. One line is a *fabrication*
(a specific the docs never mention); another is a *contradiction* (the docs say
the opposite). Every line ends in an `[eN]`, so a "did each line cite
something?" check waves it through.

This walkthrough runs one flow that catches it. In a single offline,
deterministic run you see all five signals:

| Signal | Answers |
|---|---|
| **evidence** | which source chunk retrieval actually surfaced — with path + span |
| **citations** | which answer line claims to rest on which chunk |
| **verdict** | an opt-in verifier judging claim-vs-evidence, so a citation that does *not* support its line is flagged |
| **metrics** | token cost, cited / verified / flagged counters, USD budget |
| **lineage** | git-blame for the answer: which step last wrote each line |

```console
$ PYTHONPATH=src python3 examples/debug_bad_rag.py
```

Source: [`debug_bad_rag.py`](debug_bad_rag.py) — ~200 lines, stdlib only. The
fake LLM and the toy verifier are the only stand-ins; swap them for
[`throughline.adapters.llm`](../src/throughline/adapters/llm.py) and a real
NLI model / LLM judge and the plumbing is unchanged.

---

## The answer under test

The user asks one question; the model returns four lines, each cited:

```
How long do we keep internal chat logs, and can we use them for analytics or model training?

Internal chat logs are retained for 90 days, then purged automatically.
They may be used for internal analytics dashboards.
Deleted logs are also archived to cold storage for one year.
Logs can be used to train machine learning models after anonymization.
```

Reads clean. Two of those lines are not in the docs.

## Step 1 — evidence: what retrieval actually surfaced

The retriever returns `EvidenceChunk` objects, so each chunk carries its own
provenance (path + char span) instead of being guessed at. `retriever_step`
records every one in the run's **evidence ledger**, for free:

```
=== evidence ledger (what retrieval actually surfaced) ===
  [e1] policies/retention.md span=(0, 71)
       Internal chat logs are retained for 90 days, then purged automatically.
  [e2] policies/data-use.md span=(0, 56)
       Chat logs may be used for internal analytics dashboards.
  [e3] policies/data-use.md span=(140, 226)
       Chat logs must not be used to train machine learning models, even after anonymization.
```

Note `e3`: the docs *forbid* training on chat logs.

## Step 2 — citations + verdict: the joined per-line view

`citations_step` parses the `[eN]` markers, validates each id against the
evidence ledger, and strips them into clean text. `verify_claims_step` then
runs an opt-in verifier over each cited claim and upgrades its status from the
bare fact *"cited"* to a **verdict**. `claims.join_blame(...)` joins the three
ledgers — edit lineage, claims, evidence — into one line-by-line view:

```
=== per-line verdict (evidence + claim lineage, joined) ===
      ok supported   !! contradicted   ?? unsupported   · uncited

  ok L1 [e1] conf=1.00  Internal chat logs are retained for 90 days, then purged automatically.
        └─ e1 policies/retention.md: Internal chat logs are retained for 90 days, then purged automatically.
  ok L2 [e2] conf=1.00  They may be used for internal analytics dashboards.
        └─ e2 policies/data-use.md: Chat logs may be used for internal analytics dashboards.
  ?? L3 [e1] conf=0.14  Deleted logs are also archived to cold storage for one year.
        └─ e1 policies/retention.md: Internal chat logs are retained for 90 days, then purged automatically.
  !! L4 [e3] conf=0.90  Logs can be used to train machine learning models after anonymization.
        └─ e3 policies/data-use.md: Chat logs must not be used to train machine learning models, even after anonymization.
```

There is the bug, twice over:

- **L3 is `unsupported` (0.14).** It cites `e1`, the retention chunk — which
  says nothing about cold-storage backups. The model invented a plausible
  specific and pinned a real-but-irrelevant citation to it. A citation-count
  check passes; the verifier does not.
- **L4 is `contradicted` (0.90).** It cites `e3` — the chunk that *explicitly
  forbids* training on chat logs. The citation is real; the claim is its
  opposite.

This is the point throughline insists on: **"hallucination" is a verifier
conclusion, never the mere absence of a citation.** Facts (`cited`,
`uncited_line`) and verdicts (`supported` / `unsupported` / `contradicted` /
`low_confidence_support`) are kept apart on purpose — see
[ARCHITECTURE.md](../ARCHITECTURE.md).

## Step 3 — violations: the ship-blocker

Both flagged lines land in the run's violation list — one machine-readable
place a gate (or a `raise` policy) can act on:

```
=== violations (why this answer should NOT have shipped) ===
  - 2 claim(s) flagged by verifier: line 3 (unsupported, 0.14); line 4 (contradicted, 0.90)
```

## Step 4 — metrics: cost and counts of the same run

The verifier's verdicts are counters, so the health of an answer is a metric
you can alert on (`claims.unsupported`, `claims.contradicted`), alongside the
token cost and USD budget the run already tracks:

```
=== metrics ===
  claims.cited             4
  claims.contradicted      1
  claims.supported         2
  claims.unsupported       1
  claims.verified          4
  llm.calls                1
  llm.input_tokens         180
  llm.output_tokens        45
  quota.cost               0.000315
  retrieval.docs           3
  ...
```

`4 cited, 2 supported` — the two-line gap *is* the story, and it is a number.

## Step 5 — lineage: git-blame for the answer text

Finally, edit lineage attributes every line of the final text to the step that
last wrote it. Here that is `citations` for all four — it rewrote each line
when it stripped the `[eN]` markers into the clean text the user sees:

```
=== lineage: git-blame for the answer text ===
citations ~   1│ Internal chat logs are retained for 90 days, then purged automatically.
citations ~   2│ They may be used for internal analytics dashboards.
citations ~   3│ Deleted logs are also archived to cold storage for one year.
citations ~   4│ Logs can be used to train machine learning models after anonymization.
```

In a multi-pass flow (draft → refine → …) this is where you see *which pass*
introduced a bad line — `~` is a modify, `+` a generate, and unchanged lines
keep their original author. See [`demo_rag.py`](demo_rag.py) for a run with
several authoring steps.

---

## What you just saw

A cited answer is not a correct answer. The run gave you, deterministically and
for the price of one opt-in verifier call per claim:

1. the exact chunk behind every line (**evidence**),
2. the line→chunk links, validated (**citations**),
3. a verdict flagging the two lines a human would have missed (**unsupported /
   contradicted**),
4. those failures as counters and a cost you can gate on (**metrics**),
5. the step that wrote each line (**lineage**).

Swap the fake LLM for `anthropic_chat(...)` and the toy verifier for your NLI
model, and this is the same debugging surface over your real corpus.

## The other example presets

The rest of `examples/` shows the control plane from other angles — run
`throughline presets` to list them:

| Preset | Use case | Shows |
|---|---|---|
| [`rag-docs`](presets/rag-docs.toml) | Internal documentation RAG | evidence lineage, citations, semantic cache, quota |
| [`report-gen`](presets/report-gen.toml) | Artifact-backed report generation | slots, map steps, report lineage, artifact refs |
| [`data-qa`](presets/data-qa.toml) | Data quality assistant | deterministic checks, step validation, strict report schema |
| [`doc-extract`](presets/doc-extract.toml) | Document extraction pipeline | parser slot, page map, retryable structured extraction |
| [`support-agent`](presets/support-agent.toml) | Guarded support bot | intent routing, policy screening/redaction, quota fallback, audit |
| [`agent-preflight`](presets/agent-preflight.toml) | Live agent environment gate | ManifestGate, lockfile verify, live vs harness-attested provenance |
| [`agent-audit`](presets/agent-audit.toml) | AI-agent workflow reproducibility | run manifests ("lockfile"), config drift diff, tool-call trace alignment (call_id-paired, exact/inferred quality, first behavioral divergence), symmetric multidimensional outcome diff (change + regression/improvement assessment), two-stage classed decision extraction (markers, then optional @semantic slot) with evidence spans, recursive secret redaction (public output + blame trail) |
| [`surgical_context/`](surgical_context/) | Code intelligence / change impact | file:line citations, code QA, real integration |

Harness integrations (library + CLI):

- `throughline lockfile capture|update|verify` — Claude Code / Cursor / Codex config → lockfile
- `throughline transcript convert` — normalize harness JSONL for `agent-audit`
- [`ci-agent-manifest.yml`](ci-agent-manifest.yml) — sample CI job that blocks on lockfile drift
- [`artifacts/harness-workflow-verification.canvas.tsx`](artifacts/harness-workflow-verification.canvas.tsx) — live-host verification snapshot (extractors operational vs attestable settings; Cursor source-format limits)

```console
$ THROUGHLINE_PRESETS=examples/presets PYTHONPATH=src:. \
    python3 -m throughline run rag-docs -i "how should answers cite docs?" --json --blame
```
