# Agent Audit: Token Efficiency Metrics

## What this shows

When an agent run succeeds but takes a different path, the `agent-audit` preset now profiles four token efficiency dimensions — modeled on database `EXPLAIN` output. These metrics expose hidden costs: retries, errors, delays to first productive output, and context pollution.

## The fixture runs

Both runs **succeeded** (status=ok, tests pass), yet efficiency metrics show dramatic divergence.

### Baseline: Clean path (COR=0.0, CPI=0.0, STC=2)

```
User: add retry with backoff to fetch_users() in api/client.py

Agent:
  1. Edit api/client.py (call 1)
  2. Test with pytest (call 2, succeeds)
  → Done

Metrics:
  COR = 0.0  (no retries; 0 errors / 1 successful call)
  CPI = 0.0  (no pollution; 0 failed / 2 total calls)
  STC = 2    (edit succeeded on 2nd call in sequence)
  Problems = 0
```

**Profile**: 2 tool calls, first code edit succeeds immediately. No errors or denials.

---

### Candidate: Error path (COR=1.0, CPI=0.333, STC=6)

```
User: add retry with backoff to fetch_users() in api/client.py

Agent:
  Plan: project instructions prefer async IO; rewrite as async first
  
  1. Edit api/client.py (async rewrite, call 1)
  2. Test (call 2, fails: TypeError coroutine)  ← ERROR
  3. Try pip install aiohttp (call 3, denied)  ← DENIED
  4. Workaround with curl vendor installer (call 4, succeeds)
  5. Edit tests/test_client.py for async (call 5)
  6. Test (call 6, succeeds)
  → Done

Metrics:
  COR = 1.0  (1 error + 1 denied = 2 problems / 2 successful calls)
  CPI = 0.333  (2 problems / 6 total calls)
  STC = 6    (edit at call 5, but first productive output at call 6)
  Problems = 2 (1 error + 1 denied)
```

**Profile**: 6 tool calls, 2 failures (1 error, 1 denied). First code edit at call 5. Took 4x longer to reach stability.

---

## The divergence report

```
## Token efficiency (efficiency)
- baseline: COR=0.0 (overhead) | CPI=0.0 (pollution) | STC=2 (steps to code) | 0e/0d errors
- candidate: COR=1.0 (overhead) | CPI=0.333 (pollution) | STC=6 (steps to code) | 1e/1d errors
  [medium] overhead_ratio (regression): 0.0 → 1.0 (delta=1.0)
  [medium] pollution_index (regression): 0.0 → 0.333 (delta=0.333)
  [high] steps_to_code (regression): 2 → 6 (delta=4)
  [medium] problem_calls (regression): 0 → 2 (delta=2)
```

**Verdict:** `drift_and_divergence` — candidate has config drift + execution divergence + **efficiency regression**. The run succeeded, but at 2x token cost.

---

## Understanding the metrics

| Metric | Formula | Good | Watch | Problem |
|--------|---------|------|-------|---------|
| **COR** | errors+denied / successful | 0.0 | 0.3–1.0 | 1.0+ |
| **CPI** | errors+denied / total | 0.0 | 0.0–0.2 | 0.4+ (hallucination) |
| **STC** | index of first Edit/Write | 1–2 | 3–6 | 6+ |

### COR (Context Overhead Ratio)
- Shows retry cost as a ratio
- **0.0**: first-pass success, no errors
- **1.0**: one error per success; expensive recovery
- **2.0+**: more errors than successes; run is thrashing

### CPI (Context Pollution Index)
- Fraction of work wasted on failures
- **0.0–0.2**: good, minor friction
- **0.2–0.4**: increasing noise, context crowding
- **0.4+**: hallucination risk — model forgetting constraints

### STC (Steps-to-Code)
- Call-sequence index of first productive output
- **1–2**: injected context worked, agent knew the task
- **3–4**: quick exploration, brief read before action
- **4–6**: typical agent behavior
- **6+**: slow path, too much reading

### Problem Calls
- Absolute count: error_calls + denied_calls
- Tracks environmental resistance (permission denials, network blocks, test failures)

---

## When to worry

Token metrics diverge when:

1. **Environment changes** — new MCP server, network restrictions, sandbox policy
   - → More denials, higher CPI
   
2. **Model or sampling changes** — higher temperature or different model family
   - → More exploration, higher STC (natural)
   
3. **Prompt degrades** — weaker injection or ambiguous requirements
   - → More reads before action, higher STC; more retries, higher COR
   
4. **Concurrency / contention** — race conditions on shared files or DB
   - → Retryable errors, higher COR
   
5. **Reasoning strategy changes** — agent switching from "plan then act" to "explore then refactor"
   - → Natural STC increase; watch if CPI also rises (pollution)

---

## Implementation

**Step in preset:** `assess_token_metrics` runs after `extract_traces`, before `drop_raw_sessions` (needs raw events).

**Per-session calculation:**
1. Scan events: tool_call → tool_result (paired by call sequence)
2. Count outcomes: ok / error / denied
3. Find first Edit/Write that succeeded
4. Calculate COR, CPI, STC

**Comparison:** `_compare_token_metrics` flags severity:
- CPI > 0.4 → high severity (hallucination threshold)
- COR > 1.0 → medium severity (expensive retries)
- STC jumps > 3 → high severity (late to productivity)

**Output:**
- JSON: `token_metrics` (per-session values) + `token_divergence` (severity-tagged differences)
- Report: "## Token efficiency" section with summary + divergence items
- Metrics: `audit.token_cor`, `audit.token_cpi`, `audit.steps_to_code`, `audit.token_divergence`

---

## Running the example

```bash
THROUGHLINE_PRESETS=examples/presets PYTHONPATH=src:. \
  python3 -m throughline run agent-audit -i "audit" --json --blame
```

Output includes full `token_metrics` and `token_divergence` in the JSON, plus a human-readable "## Token efficiency" section in the report.

The metrics layer is designed to answer: **Did this agent's task succeed?** Yes. **Did it do so efficiently?** Not always — and now you have data.
