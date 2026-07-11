# Example artifacts

Checked-in verification snapshots and other durable reports from live
harness probes. These are not fixtures for unit tests — they record what
was observed on a real machine at a point in time.

| Artifact | What it is |
|---|---|
| [`harness-workflow-verification.canvas.tsx`](harness-workflow-verification.canvas.tsx) | Cursor canvas: lockfile → preflight → session hook → transcript convert → agent-audit on live Claude Code / Cursor / Codex configs |
| [`claude-decision-budget-bench.md`](claude-decision-budget-bench.md) | Live Claude Code sessions vs decision-extraction defaults (200k/4k/400) — duration, chars, sentences, recall_hits, semantic_calls |
| [`claude-decision-budget-bench.json`](claude-decision-budget-bench.json) | Raw numbers for the budget bench above |

Open the `.canvas.tsx` in Cursor (Canvases) or read the source as a structured
report. Re-run the probes before treating numbers as current.
