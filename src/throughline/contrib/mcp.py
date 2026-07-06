"""MCP serving layer: expose flows as tools for agents. Zero dependencies.

Lives in ``throughline.contrib`` deliberately: adapters bring third-party
components INTO flows, MCP serves flows OUTWARD to one particular protocol.
Nothing in the core (or in the adapters) imports this module — delete it and
the framework doesn't notice. It is one of possibly many serving layers
(HTTP, a queue consumer, ...) built on the same public surface: presets,
Flow.run, project_result-style Result projection, the artifact store.

The agent<->flow boundary, as designed:

  * tool arguments are the payload — a JSON dict, validated by the flow's own
    middleware; nothing outside the contract can come in;
  * the Result is never returned whole: ``project_result`` renders output +
    selected artifacts (metrics, lineage stats, violations) under a hard byte
    budget. Oversized outputs land in the artifact store and come back as a
    handle + summary — the agent pulls slices via the ``get_artifact`` tool.
    A gigabyte table cannot end up in the model context *by construction*;
  * handles are leases (see throughline.store): expiry is a normal condition.
    Whether a re-run re-creates the artifact depends on the flow being
    replayable (same inputs/config/sources, stochastic steps cached or
    seeded) — the error tells the agent to re-run *if* that holds;
  * the agent's trace id travels in tool arguments (``trace_id``) and stamps
    every event of the run — one trace from the agent's reasoning through
    every step of the graph, ready for the OTel bridge.

The transport is MCP over stdio (JSON-RPC 2.0, newline-delimited), small
enough to implement by hand — no SDK. ``MCPServer.handle`` is the pure,
testable unit; ``serve_stdio`` is the loop:

    from throughline.contrib.mcp import MCPServer
    MCPServer(presets=["rag-qa"]).serve_stdio()

or from the CLI:  throughline mcp --preset rag-qa
"""

from __future__ import annotations

import json
import sys
import uuid
from typing import Any

from .. import __version__
from ..context import EventBus, RunContext
from ..errors import ArtifactExpired, ThroughlineError
from ..flow import Flow
from ..store import MemoryArtifactStore

PROTOCOL_VERSION = "2025-06-18"
DEFAULT_MAX_RESULT_BYTES = 32 * 1024

_RUN_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "input": {"description": "payload for the flow (string or JSON object)"},
        "trace_id": {"type": "string",
                     "description": "caller's trace id; stamps every event of the run"},
    },
    "required": ["input"],
}

_GET_ARTIFACT_SCHEMA = {
    "type": "object",
    "properties": {
        "artifact": {"type": "string", "description": "artifact id ('session/key')"},
        "start": {"type": "integer", "description": "slice start (items or chars)"},
        "stop": {"type": "integer", "description": "slice stop (exclusive)"},
    },
    "required": ["artifact"],
}


class TracedEventBus(EventBus):
    """EventBus that stamps every event with the caller's trace id."""

    def __init__(self, trace_id: str):
        super().__init__()
        self.trace_id = trace_id

    def emit(self, type: str, **fields: Any) -> dict:
        return super().emit(type, trace_id=self.trace_id, **fields)


def project_result(result: Any, *, store: Any = None, session: str = "default",
                   max_bytes: int = DEFAULT_MAX_RESULT_BYTES) -> dict:
    """Result -> agent-safe report: output under a byte budget + artifact digest.

    Oversized output is swapped for a stored artifact handle with a preview.
    """
    report: dict[str, Any] = {"run_id": result.run_id}
    trace_id = result.ctx.state.get("trace_id")
    if trace_id:
        report["trace_id"] = trace_id

    output = result.output
    try:
        rendered = json.dumps(output, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = str(output)
    if len(rendered) > max_bytes and store is not None:
        ref = store.put(output, session=session)
        report["output"] = {**ref.to_dict(),
                            "preview": rendered[:512],
                            "note": "output exceeds the context budget; "
                                    "fetch slices via the get_artifact tool"}
    else:
        report["output"] = output

    metrics = result.metrics
    if metrics.get("counters"):
        report["metrics"] = metrics["counters"]
    if result.violations:
        report["violations"] = result.violations
    if result.lineage is not None:
        report["lineage"] = result.lineage.stats()
    claims = result.ctx.artifacts.get("claims")
    if claims is not None and len(claims):
        report["claims"] = len(claims)
    return report


class MCPServer:
    """Serve flows as MCP tools over stdio.

    Args:
        presets: preset names to expose (default: every discoverable preset).
        flows:   {tool_name: Flow} for flows built in code (merged with presets).
        store:   artifact store for oversized outputs (default: in-memory,
                 30 min TTL).
        max_result_bytes: byte budget for inline tool results.
    """

    def __init__(self, presets: list[str] | None = None,
                 flows: dict[str, Flow] | None = None,
                 store: Any = None,
                 max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES):
        from ..presets import list_presets, load_preset
        self.store = store if store is not None else MemoryArtifactStore(default_ttl=1800)
        self.max_result_bytes = max_result_bytes
        self.session = f"mcp-{uuid.uuid4().hex[:8]}"
        self._flows: dict[str, Flow] = {}
        names = presets if presets is not None else sorted(list_presets())
        for name in names:
            self._flows[f"run_{name.replace('-', '_')}"] = load_preset(name)
        for name, flow in (flows or {}).items():
            self._flows[name] = flow

    # -- tool surface -----------------------------------------------------------
    def _tools(self) -> list[dict]:
        tools = []
        for tool_name, flow in self._flows.items():
            chain = " -> ".join(s.name for s in flow.steps)
            tools.append({
                "name": tool_name,
                "description": f"Run the flow {flow.name!r} ({chain}). "
                               f"Returns output, metrics and provenance "
                               f"stats; oversized outputs come back as "
                               f"artifact handles.",
                "inputSchema": _RUN_TOOL_SCHEMA,
            })
        tools.append({
            "name": "get_artifact",
            "description": "Fetch (a slice of) a stored artifact by handle. "
                           "Handles are leases: an expired artifact can be "
                           "re-created by re-running its flow only if that "
                           "flow is replayable with the same arguments.",
            "inputSchema": _GET_ARTIFACT_SCHEMA,
        })
        return tools

    def _call_flow(self, flow: Flow, arguments: dict) -> dict:
        trace_id = arguments.get("trace_id") or uuid.uuid4().hex[:12]
        ctx = RunContext(flow=flow.name, config=dict(flow.config),
                         events=TracedEventBus(trace_id))
        ctx.state["trace_id"] = trace_id
        ctx.state["session"] = self.session
        result = flow.run(arguments.get("input"), ctx=ctx)
        return project_result(result, store=self.store, session=self.session,
                              max_bytes=self.max_result_bytes)

    def _call_get_artifact(self, arguments: dict) -> Any:
        ref = arguments["artifact"]
        if "start" in arguments or "stop" in arguments:
            return self.store.slice(ref, arguments.get("start", 0),
                                    arguments.get("stop"))
        value = self.store.get(ref)
        rendered = json.dumps(value, ensure_ascii=False, default=str)
        if len(rendered) > self.max_result_bytes:
            if isinstance(value, (list, tuple)):
                shape = f"list of {len(value)} items"
            elif isinstance(value, str):
                shape = f"text, {value.count(chr(10)) + 1} lines"
            else:
                shape = type(value).__name__
            raise ThroughlineError(
                f"artifact {ref!r} exceeds the context budget "
                f"({len(rendered)} bytes, {shape}); request a slice with start/stop")
        return value

    # -- JSON-RPC ---------------------------------------------------------------
    def handle(self, message: dict) -> dict | None:
        """Process one JSON-RPC message; None for notifications."""
        method = message.get("method", "")
        message_id = message.get("id")
        if message_id is None:  # notification (e.g. notifications/initialized)
            return None
        try:
            result = self._dispatch(method, message.get("params") or {})
        except ThroughlineError as exc:
            return self._tool_error(message_id, exc)
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": message_id,
                    "error": {"code": -32603, "message": repr(exc)}}
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    def _dispatch(self, method: str, params: dict) -> dict:
        if method == "initialize":
            return {"protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "throughline", "version": __version__}}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self._tools()}
        if method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments") or {}
            if name == "get_artifact":
                payload = self._call_get_artifact(arguments)
            elif name in self._flows:
                payload = self._call_flow(self._flows[name], arguments)
            else:
                known = ", ".join([*self._flows, "get_artifact"])
                raise ThroughlineError(f"unknown tool {name!r}; available: {known}")
            text = json.dumps(payload, ensure_ascii=False, default=str)
            return {"content": [{"type": "text", "text": text}], "isError": False}
        raise ThroughlineError(f"unsupported method {method!r}")

    @staticmethod
    def _tool_error(message_id: Any, exc: Exception) -> dict:
        # Tool-level failures go back as tool results (isError), so the agent
        # can read the message and correct course — e.g. re-run after expiry.
        note = {"error": str(exc)}
        if isinstance(exc, ArtifactExpired):
            note["expired"] = exc.artifact_id
        return {"jsonrpc": "2.0", "id": message_id,
                "result": {"content": [{"type": "text",
                                        "text": json.dumps(note, ensure_ascii=False)}],
                           "isError": True}}

    # -- transport ----------------------------------------------------------------
    def serve_stdio(self, stdin=None, stdout=None) -> None:
        """Newline-delimited JSON-RPC loop; blocks until stdin closes."""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        try:
            for line in stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response = self.handle(message)
                if response is not None:
                    stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                    stdout.flush()
        finally:
            self.store.drop_session(self.session)
