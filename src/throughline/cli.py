"""CLI: run presets, inspect components, dry-check pipelines, serve MCP.

    throughline run demo --input "how does lineage work?" --blame
    throughline run pipeline.toml --input-file question.txt --json
    throughline presets
    throughline steps
    throughline components            # typed catalog (kind, source, plugins)
    throughline doctor rag-qa         # resolve every slot without running
    throughline mcp --preset rag-qa   # [contrib] expose flows as MCP tools (stdio)
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .errors import ThroughlineError
from .modules.observe import ConsoleSink
from .presets import inspect_preset, list_presets, load_preset, load_preset_config
from .registry import available, entries, unavailable


def _read_input(args: argparse.Namespace):
    if args.input is not None:
        text = args.input
    elif args.input_file:
        if args.input_file == "-":
            text = sys.stdin.read()
        else:
            with open(args.input_file, encoding="utf-8") as fh:
                text = fh.read()
    else:
        return None
    text = text.strip()
    if args.json_input:
        return json.loads(text)
    return text


def _cmd_run(args: argparse.Namespace) -> int:
    flow = load_preset(args.preset)
    ctx = None
    if args.events:  # subscribe the console before the run starts
        from .context import RunContext
        ctx = RunContext(flow=flow.name, config=dict(flow.config))
        ctx.events.subscribe(ConsoleSink(verbose=args.verbose))

    result = flow.run(_read_input(args), ctx=ctx)

    if args.json:
        report = {
            "run_id": result.run_id,
            "output": result.output,
            "metrics": result.metrics,
            "violations": result.violations,
        }
        if result.lineage is not None:
            report["lineage"] = {"stats": result.lineage.stats(),
                                 "blame": result.lineage.blame()}
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0

    output = result.output
    if isinstance(output, dict) and "answer" in output:
        output = output["answer"]
    print(output)
    if args.metrics:
        print("\n-- metrics --", file=sys.stderr)
        print(json.dumps(result.metrics, indent=2, default=str), file=sys.stderr)
    if args.blame and result.lineage is not None:
        print("\n-- lineage (blame) --", file=sys.stderr)
        print(result.lineage.render_blame(), file=sys.stderr)
    if result.violations:
        print("\n-- violations --", file=sys.stderr)
        for violation in result.violations:
            print(f"  ! {violation}", file=sys.stderr)
    if args.lineage_out and result.lineage is not None:
        result.lineage.to_jsonl(args.lineage_out)
        print(f"lineage written to {args.lineage_out}", file=sys.stderr)
    return 0


def _cmd_presets(_: argparse.Namespace) -> int:
    for name, path in sorted(list_presets().items()):
        try:
            description = load_preset_config(name).get("description", "")
        except ThroughlineError:
            description = "(unreadable)"
        print(f"{name:20} {path}  {description}")
    return 0


def _cmd_steps(_: argparse.Namespace) -> int:
    components = available()
    if not components:
        print("(registry is empty — components register via @throughline.register, "
              "entry points, or are referenced by import path in presets)")
    for name, obj in sorted(components.items()):
        kind = type(obj).__name__
        print(f"{name:24} {kind:12} {getattr(obj, '__doc__', '') or ''}".rstrip()[:100])
    return 0


def _cmd_components(_: argparse.Namespace) -> int:
    catalog = entries()
    if not catalog:
        print("(no components discovered — register via @throughline.register, "
              "manifests in the 'throughline.plugins' entry-point group, or "
              "reference by import path in presets)")
    current_kind = None
    for entry in catalog:
        if entry.kind != current_kind:
            current_kind = entry.kind
            print(f"[{current_kind}]")
        doc = (getattr(entry.obj, "__doc__", "") or "").strip().splitlines()
        print(f"  {entry.name:24} {entry.source:20} {doc[0] if doc else ''}"[:100])
    broken = unavailable()
    if broken:
        print("[unavailable]")
        for name, reason in sorted(broken.items()):
            print(f"  {name:24} {reason}"[:120])
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    report = inspect_preset(args.preset)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report["ok"] else 1
    print(f"preset {report['name']!r}")
    for row in report["steps"] + report["middleware"]:
        mark = {"ok": "+", "disabled": "-"}.get(row["status"], "!")
        uses = f" uses={row['uses']}" if row.get("uses") else ""
        factory = f" ({row['factory']})" if row.get("factory") else ""
        print(f"  {mark} {row['slot']:20}{uses}{factory}")
        print(f"      {row.get('detail', '')}")
    print("ok" if report["ok"] else "PROBLEMS FOUND", file=sys.stderr)
    return 0 if report["ok"] else 1


def _cmd_mcp(args: argparse.Namespace) -> int:
    # optional serving layer: imported only when the command is actually used
    from .contrib.mcp import MCPServer
    server = MCPServer(presets=args.preset or None,
                       max_result_bytes=args.max_result_bytes)
    print(f"throughline MCP server: {len(server._flows)} flow tool(s) + get_artifact "
          f"(stdio, newline-delimited JSON-RPC)", file=sys.stderr)
    server.serve_stdio()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="throughline",
        description="Framework-neutral control plane for agents and LLM pipelines.")
    parser.add_argument("--version", action="version", version=f"throughline {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a preset")
    run.add_argument("preset", help="preset name or path to a .toml file")
    run.add_argument("--input", "-i", help="input payload (string)")
    run.add_argument("--input-file", "-f", help="read input from file ('-' = stdin)")
    run.add_argument("--json-input", action="store_true", help="parse input as JSON")
    run.add_argument("--json", action="store_true", help="print full JSON report")
    run.add_argument("--metrics", action="store_true", help="print metrics to stderr")
    run.add_argument("--blame", action="store_true", help="print line-level lineage table")
    run.add_argument("--events", action="store_true", help="stream events to stderr")
    run.add_argument("--verbose", action="store_true", help="include all event types")
    run.add_argument("--lineage-out", help="write lineage records to a JSONL file")
    run.set_defaults(func=_cmd_run)

    presets = commands.add_parser("presets", help="list discoverable presets")
    presets.set_defaults(func=_cmd_presets)

    steps = commands.add_parser("steps", help="list registered components")
    steps.set_defaults(func=_cmd_steps)

    components = commands.add_parser(
        "components", help="typed component catalog (kind, source, broken plugins)")
    components.set_defaults(func=_cmd_components)

    doctor = commands.add_parser(
        "doctor", help="dry-check a preset: resolve every slot, show wrap decisions")
    doctor.add_argument("preset", help="preset name or path to a .toml file")
    doctor.add_argument("--json", action="store_true", help="machine-readable report")
    doctor.set_defaults(func=_cmd_doctor)

    mcp = commands.add_parser(
        "mcp", help="[optional, contrib] serve presets as MCP tools over stdio")
    mcp.add_argument("--preset", "-p", action="append",
                     help="preset to expose (repeatable; default: all discoverable)")
    mcp.add_argument("--max-result-bytes", type=int, default=32 * 1024,
                     help="inline tool result budget; larger outputs become artifact handles")
    mcp.set_defaults(func=_cmd_mcp)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ThroughlineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
