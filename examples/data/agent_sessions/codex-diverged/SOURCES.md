# Codex diverged pair (real sessions)

Neutral JSONL converted from local `~/.codex/sessions` rollouts (2026-07-10).

| Role | Session | Events | Tools |
|---|---|---|---|
| baseline | `019efb59-6e4b-71b2-b53a-b434930d82ce` | 71 | 32 |
| candidate | `019efb5f-815f-7212-aabc-1b6bc0e22cf5` | 44 | 19 |

**Task (same prompt):** diagnose how to split `context_engine/main.py` in
`surgical_context` (FastAPI “god-module” startup).

**Sources:**

- `~/.codex/sessions/2026/06/24/rollout-2026-06-24T13-36-42-019efb59-6e4b-71b2-b53a-b434930d82ce.jsonl`
- `~/.codex/sessions/2026/06/24/rollout-2026-06-24T13-43-20-019efb5f-815f-7212-aabc-1b6bc0e22cf5.jsonl`

Used by [`examples/audit_diverged_runs.py`](../../../audit_diverged_runs.py).
