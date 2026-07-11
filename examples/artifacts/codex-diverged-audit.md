# Codex diverged runs — agent-audit demo

Real pair from `~/.codex/sessions` (2026-06-24), audited 2026-07-10 on
throughline `@9dc99b3`.

## Task

Same prompt, two runs: diagnose how to split `context_engine/main.py` in
`surgical_context` (FastAPI “god-module” startup).

| | baseline | candidate |
|---|---|---|
| session | `019efb59…` | `019efb5f…` |
| events | 71 | 44 |
| tool calls | 32 | 19 |
| status | ok | ok |
| tokens | 1 182 694 | 812 985 |

## Workflow

```console
$ PYTHONPATH=src:. THROUGHLINE_PRESETS=examples/presets \
    python3 examples/audit_diverged_runs.py
```

Data: [`examples/data/agent_sessions/codex-diverged/`](../data/agent_sessions/codex-diverged/).
Script: [`examples/audit_diverged_runs.py`](../audit_diverged_runs.py).

## Result

| Signal | Value |
|---|---|
| Verdict | `execution_divergence` |
| Readiness gate | `pass` |
| Config drift | none |
| Trace pairing | exact / exact |
| Trace completeness | complete / complete |
| First divergence | event **1** (`args_changed`) |
| Trace divergence | 1 first + 19 args_changed + 1 calls_missing |
| Outcome divergence | none (both green; no files/tests axis) |
| Audit duration | ~0.05 s |

**First tool call:**

- baseline → `graphify --help | tee /tmp/token_compare_op01.txt …`
- candidate → `rg --files context_engine/api/routes context_engine | sort …`

Same harness/model attestation, different execution path from the first
command — the audit surfaces behavior without inventing config drift.
