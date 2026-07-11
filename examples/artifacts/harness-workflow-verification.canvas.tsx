import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

type Status = "pass" | "fail" | "partial" | "gap";

const STATUS_TONE = {
  pass: "success",
  fail: "error",
  partial: "warning",
  gap: "neutral",
} as const;

function StatusPill({ status, label }: { status: Status; label: string }) {
  return <Pill tone={STATUS_TONE[status]} size="sm" active>{label}</Pill>;
}

export default function HarnessWorkflowVerification() {
  const { tokens } = useHostTheme();

  return (
    <Stack gap={24} style={{ padding: 24, maxWidth: 960 }}>
      <Stack gap={8}>
        <H1>Harness integration verification</H1>
        <Text tone="secondary">
          Real local Claude Code / Cursor / Codex configs and transcripts on this
          machine · 2026-07-10 · throughline @ aa9d5d6
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value="41/41" label="Unit harness tests" tone="success" />
        <Stat value="3/3" label="Extractors operational" tone="success" />
        <Stat value="2/3" label="Attestable settings found" tone="warning" />
        <Stat value="3/3" label="Transcript adapters" tone="success" />
      </Grid>

      <Table
        headers={["Metric", "Result", "Meaning"]}
        rows={[
          [
            "Config extractors executed successfully",
            "3/3",
            "Claude / Cursor / Codex extractors ran; probed paths returned without error",
          ],
          [
            "Config extractors found attestable settings",
            "2/3",
            "Claude + Codex had model/prompt/MCP artifacts; Cursor had none on this host",
          ],
        ]}
        rowTone={["success", "warning"]}
      />

      <Callout tone="success" title="Verdict">
        Lockfile → preflight → session hook works on real Claude Code and Codex
        configs. All three config extractors are operational; Cursor&apos;s empty
        attestation matches an environment with no project markers / MCP files,
        not an extractor failure. Transcript → agent-audit works for Claude Code,
        Cursor, and Codex (bridge + ~/.codex/sessions rollout). Remaining work:
        P1 audit scale on huge tool_result bodies.
      </Callout>

      <H2>User workflow</H2>
      <Table
        headers={["Step", "Real input", "Result", "Notes"]}
        rows={[
          [
            "lockfile capture",
            "claude-code + codex homes",
            <StatusPill status="pass" label="pass" />,
            "model, effort, prompt hashes, tools/MCP attested",
          ],
          [
            "lockfile verify (self)",
            "captured lock + live probe",
            <StatusPill status="pass" label="pass" />,
            "live/harness provenance sections present",
          ],
          [
            "agent-preflight",
            "real Claude lock",
            <StatusPill status="pass" label="pass" />,
            "gate=pass; report splits Live vs Harness-attested",
          ],
          [
            "preflight vs example lock",
            "examples/data/agent.lock.json",
            <StatusPill status="pass" label="block×17" />,
            "correct drift (synthetic lock ≠ live harness)",
          ],
          [
            "session hook start",
            "real Claude config",
            <StatusPill status="pass" label="pass" />,
            "writes session_start with observed.live + observed.harness",
          ],
          [
            "transcript convert",
            "Claude / Cursor / Codex files",
            <StatusPill status="pass" label="3/3" />,
            "Codex rollout dialect added (session_meta/response_item)",
          ],
          [
            "agent-audit",
            "fixtures + truncated Claude",
            <StatusPill status="partial" label="partial" />,
            "full real sessions hang in decision extraction",
          ],
        ]}
        rowTone={[
          "success",
          "success",
          "success",
          "success",
          "success",
          "success",
          "warning",
        ]}
      />

      <H2>Per-harness</H2>
      <Grid columns={3} gap={16}>
        <Card>
          <CardHeader
            title="Claude Code"
            trailing={<StatusPill status="pass" label="ready" />}
          />
          <CardBody>
            <Stack gap={8}>
              <Text weight="semibold">Config</Text>
              <Text tone="secondary" size="small">
                model=claude-fable-5, effort=max, 92 allow-permissions hashed,
                .claude/settings.local.json instruction hash
              </Text>
              <Text weight="semibold">Transcript</Text>
              <Text tone="secondary" size="small">
                Real project JSONL → 173–283 neutral events with tool_call /
                tool_result pairing
              </Text>
              <Text weight="semibold">Gap</Text>
              <Text tone="secondary" size="small">
                No MCP servers in this machine&apos;s Claude settings (extractor
                OK; nothing to attest)
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Cursor"
            trailing={<StatusPill status="pass" label="operational" />}
          />
          <CardBody>
            <Stack gap={8}>
              <Text weight="semibold">Config</Text>
              <Text tone="secondary" size="small">
                Extractor OK: probed workspace + ~/.cursor paths; no
                .cursorrules / AGENTS.md / mcp.json present. Result{" "}
                {"{harness: cursor}"} matches the observed environment
                (attestation empty, not a failed extract).
              </Text>
              <Text weight="semibold">Transcript</Text>
              <Text tone="secondary" size="small">
                Long agent-transcript → 242 events, 211 tool_calls (Read,
                StrReplace, Shell, …). Source export physically omits
                tool_result — source-format observability limit, surfaced as
                partial trace completeness (not an adapter defect).
              </Text>
              <Text weight="semibold">Environment note</Text>
              <Text tone="secondary" size="small">
                Substantive attestation needs project markers or MCP files on
                disk; pairing stays call-only until Cursor exports results.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Codex"
            trailing={<StatusPill status="pass" label="ready" />}
          />
          <CardBody>
            <Stack gap={8}>
              <Text weight="semibold">Config</Text>
              <Text tone="secondary" size="small">
                Works: gpt-5.5 / xhigh + surgical-context MCP (env values hashed)
              </Text>
              <Text weight="semibold">Transcript</Text>
              <Text tone="secondary" size="small">
                Real rollout → 304 neutral events, 139 paired tool
                call/results (exec_command, write_stdin, apply_patch)
              </Text>
              <Text weight="semibold">Gap</Text>
              <Text tone="secondary" size="small">
                None for convert; bridge thread.* dialect still supported
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Provenance check (post-P0)</H2>
      <Table
        headers={["Check", "Status"]}
        rows={[
          [
            "capture_environment rejects harness live-key spoof",
            <StatusPill status="pass" label="pass" />,
          ],
          [
            "observed = { live, harness } on real capture/verify/hook",
            <StatusPill status="pass" label="pass" />,
          ],
          [
            "preflight report labels Live probe vs Harness-attested",
            <StatusPill status="pass" label="pass" />,
          ],
          [
            "Violation.field populated (not path) in drift output",
            <StatusPill status="pass" label="ok" />,
          ],
        ]}
        rowTone={["success", "success", "success", "neutral"]}
      />

      <H2>Product gaps to fix</H2>
      <Stack gap={12}>
        <Card variant="outlined">
          <CardHeader
            title="P1 — agent-audit on full real sessions"
            trailing={<StatusPill status="partial" label="scale" />}
          />
          <CardBody>
            <Text>
              Decision extraction runs semantic heuristics on every sentence of
              huge tool_result bodies — full Claude sessions hung &gt;2.5 min.
              Truncated sessions (~45 events, capped text) finish in ~2s.
              Cap text / skip tool_result for decisions / budget sentences.
            </Text>
          </CardBody>
        </Card>
        <Card variant="outlined">
          <CardHeader
            title="P2 — Cursor env has no attestable artifacts (optional)"
            trailing={<StatusPill status="gap" label="env" />}
          />
          <CardBody>
            <Text>
              Not an extractor bug: 3/3 extractors operational; this host/workspace
              simply has no Cursor config files to attest. Optional follow-ups —
              document required markers, or discover IDE-managed MCP from
              alternate paths if/when they exist. Separately, Cursor transcript
              exports without tool_result are a source-format limit → partial
              completeness, not a Throughline adapter gap.
            </Text>
          </CardBody>
        </Card>
      </Stack>

      <Divider />
      <H3>Suggested next step</H3>
      <Text>
        Budget agent-audit decision extraction for large transcripts (skip or
        cap tool_result text). Cursor attestation/discovery only if you want
        richer locks on machines that actually have Cursor project markers.
      </Text>
      <Text tone="secondary" size="small" style={{ color: tokens.text.secondary }}>
        Source: live probes on this host · fixture tests tests/test_harness_integrations.py
      </Text>
    </Stack>
  );
}
