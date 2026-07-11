# Claude Code decision-budget bench

Live-host check of `extract_decisions` defaults (`max_chars=200000`,
`max_sentences=4000`, `max_semantic=400`) against full Claude Code sessions
under `~/.claude/projects/-home-idxoid-throughline/`.

Date: 2026-07-10 · throughline decision pipeline after event-type filter +
recall gate. Raw numbers: [`claude-decision-budget-bench.json`](claude-decision-budget-bench.json).

## Verdict on defaults

**Defaults are fine for these sessions.** Worst case (two largest sessions
paired) finishes in **42 ms**, uses **63%** of the char budget, **39%** of
the sentence budget, and **4.5%** of the semantic budget. No
`DecisionBudgetExceeded`.

Previous hang (>2.5 min) was from scanning tool_result bodies as speech;
with the event-type filter those bodies are skipped, so decision extraction
is no longer the bottleneck.

## Worst pair — `ce013a3b` (2.3 MiB, 431 events) vs `cfef387a` (1.6 MiB, 283 events)

| Metric | Value | vs default |
|---|---:|---|
| duration | 0.042 s | — |
| chars | 125 223 | 200 000 (62.6%) |
| sentences | 1 547 | 4 000 (38.7%) |
| recall_hits | 18 | — |
| semantic_calls | 18 | 400 (4.5%) |
| decisions found | 10 | — |
| status | ok | — |

## Per-session (session + tiny fixture partner)

| Session | size | events | dur_s | chars | sents | recall | sem |
|---|---:|---:|---:|---:|---:|---:|---:|
| ce013a3b | 2.3 MiB | 431 | 0.029 | 86 465 | 1 111 | 6 | 6 |
| cfef387a | 1.6 MiB | 283 | 0.013 | 39 182 | 444 | 14 | 14 |
| c5b7fb95 | 1.3 MiB | 189 | 0.007 | 21 094 | 330 | 1 | 1 |
| 3ee2041a | 1.0 MiB | 173 | 0.011 | 29 520 | 370 | 3 | 3 |
| c44d64a3 | 375 KiB | 68 | 0.002 | 7 259 | 103 | 4 | 4 |
| c262713d | 161 KiB | 29 | 0.002 | 6 600 | 79 | 1 | 1 |

## Note on full `agent-audit`

`extract_decisions` alone is fast on these transcripts. A full
`throughline run agent-audit` over the same pair still stalled (>3 min) in
this check — remaining cost is outside decision extraction (likely
lineage/report over large tool_result payloads). Track that separately from
these budget defaults.
