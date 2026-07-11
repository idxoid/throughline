import {
  Callout,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

/**
 * Checked-in mirror: examples/artifacts/codex-diverged-audit.canvas.tsx
 * Data from real ~/.codex/sessions pair audited 2026-07-10.
 */
export default function CodexDivergedAuditDemo() {
  const { tokens } = useHostTheme();

  return (
    <Stack gap={24} style={{ padding: 24, maxWidth: 920 }}>
      <Stack gap={8}>
        <H1>Codex diverged runs</H1>
        <Text tone="secondary">
          Same surgical_context task, two real Codex sessions · agent-audit demo ·
          2026-07-10 · throughline @ 9dc99b3
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value="execution_divergence" label="Verdict" tone="warning" />
        <Stat value="pass" label="Readiness gate" tone="success" />
        <Stat value="event 1" label="First divergence" tone="warning" />
        <Stat value="~0.05s" label="Audit duration" tone="success" />
      </Grid>

      <Callout tone="warning" title="What diverged">
        No config drift. Both runs finished ok with exact tool pairing. Paths
        split at the first tool call: baseline opened with{" "}
        <Code>graphify --help</Code>, candidate with <Code>rg --files …</Code>.
        Outcome axes (files/tests/risky) stayed empty — this is pure execution
        divergence.
      </Callout>

      <H2>Pair</H2>
      <Table
        headers={["", "Baseline", "Candidate"]}
        rows={[
          ["Session", "019efb59…", "019efb5f…"],
          ["Events", "71", "44"],
          ["Tool calls", "32", "19"],
          ["Status", "ok", "ok"],
          ["Tokens", "1,182,694", "812,985"],
          [
            "First cmd",
            "graphify --help | tee /tmp/token_compare_op01.txt …",
            "rg --files context_engine/api/routes context_engine | sort …",
          ],
        ]}
      />

      <H2>Trace health</H2>
      <Table
        headers={["Side", "Pairing", "Completeness", "Unresolved", "Orphans"]}
        rows={[
          ["baseline", "exact", "complete", "0", "0"],
          ["candidate", "exact", "complete", "0", "0"],
        ]}
        rowTone={["success", "success"]}
      />

      <H2>Trace divergence</H2>
      <Table
        headers={["Kind", "Count", "Severity"]}
        rows={[
          ["first_divergence", "1", <Pill tone="warning" size="sm" active>high</Pill>],
          ["args_changed", "19", <Pill tone="neutral" size="sm" active>medium</Pill>],
          ["calls_missing", "1", <Pill tone="neutral" size="sm" active>medium</Pill>],
        ]}
      />

      <H2>User workflow</H2>
      <Text tone="secondary" style={{ fontFamily: tokens.font.mono, fontSize: 12 }}>
        PYTHONPATH=src:. THROUGHLINE_PRESETS=examples/presets python3
        examples/audit_diverged_runs.py
      </Text>
      <Text tone="secondary">
        Data: examples/data/agent_sessions/codex-diverged/ · convert (if needed) →
        agent-audit → report. Pass --baseline / --candidate for any harness pair.
      </Text>

      <Divider />
      <Text tone="secondary" size="small">
        Sources: ~/.codex/sessions/2026/06/24/rollout-…019efb59…jsonl and
        rollout-…019efb5f…jsonl · task: split context_engine/main.py
      </Text>
    </Stack>
  );
}
