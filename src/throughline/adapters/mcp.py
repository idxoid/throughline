"""MCP client: any external MCP server becomes a step of a flow.

Direction matters: ``contrib.mcp`` serves flows OUT to agents; this adapter
brings third-party MCP servers IN as steps — the thousands of existing MCP
tools (code search, browsers, databases) plug into a flow like any other
component, budgeted by Quota, tracked by lineage, cacheable.

Zero dependencies: a subprocess + newline-delimited JSON-RPC over stdio
(the same wire format ``throughline mcp`` itself speaks, which doubles as
the integration test target).

    from throughline.adapters.mcp import MCPClient, tool_step

    client = MCPClient(["python", "-m", "surgical_context_mcp"])
    flow = tl.Flow([
        tool_step(client, "ask_code", out_key="context"),
        prompt_step("{context}\n\nQ: {question}"),
        llm,
    ])

Payload contract mirrors the server side ("tool arguments are the payload"):
a dict payload becomes the tool arguments (optionally filtered with
``params=``), anything else is sent as ``{"input": payload}``. The result
lands at ``out_key`` ("result" by default): ``structuredContent`` when the
server provides it, else the concatenated text content (``unwrap=`` post-
processes, e.g. ``json.loads``).

In presets, ``mcp_tool`` owns its client:

    [[steps]]
    uses = "throughline.adapters.mcp:mcp_tool"
    [steps.with]
    command = ["python", "-m", "throughline", "mcp", "--preset", "demo"]
    tool = "run_demo"
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
from typing import Any, Callable, Sequence

from ..errors import ThroughlineError
from ..step import Step

PROTOCOL_VERSION = "2025-06-18"


class MCPError(ThroughlineError):
    """A JSON-RPC error or ``isError`` tool result from the server."""


class MCPClient:
    """Minimal MCP client over stdio (newline-delimited JSON-RPC).

    The server process is spawned lazily on first use and the ``initialize``
    handshake is performed once. One lock serializes requests — MCP stdio is
    a single ordered channel. ``close()`` (or the context manager) terminates
    the server; a client left to the GC closes on ``__del__`` as a fallback.
    """

    def __init__(self, command: Sequence[str], *, env: dict | None = None,
                 cwd: str | None = None, timeout: float = 60.0):
        self.command = list(command)
        self.env = env
        self.cwd = cwd
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._initialized = False

    # -- lifecycle ----------------------------------------------------------
    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            env = {**os.environ, **self.env} if self.env else None
            try:
                self._proc = subprocess.Popen(
                    self.command, cwd=self.cwd, env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=sys.stderr, text=True, bufsize=1)
            except OSError as exc:
                raise MCPError(f"cannot start MCP server {self.command}: {exc}") from exc
            self._initialized = False
        if not self._initialized:
            self._initialized = True  # before the call: _request needs the flag
            try:
                self._request("initialize", {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "throughline", "version": "adapter"},
                })
                self._notify("notifications/initialized")
            except Exception:
                self._initialized = False
                raise
        return self._proc

    def close(self) -> None:
        proc, self._proc = self._proc, None
        self._initialized = False
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self):  # fallback only; prefer close()
        try:
            self.close()
        except Exception:
            pass

    # -- transport ------------------------------------------------------------
    def _send(self, message: dict) -> None:
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    def _read_line(self) -> str:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        if hasattr(select, "select") and os.name == "posix":
            ready, _, _ = select.select([proc.stdout], [], [], self.timeout)
            if not ready:
                raise MCPError(f"MCP server {self.command[0]!r}: no response "
                               f"within {self.timeout}s")
        line = proc.stdout.readline()
        if not line:
            raise MCPError(f"MCP server {self.command[0]!r} closed the pipe "
                           f"(exit code {proc.poll()})")
        return line

    def _notify(self, method: str, params: dict | None = None) -> None:
        message = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = params
        self._send(message)

    def _request(self, method: str, params: dict | None = None) -> Any:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method,
                    "params": params or {}})
        while True:  # skip server notifications / stray messages
            try:
                message = json.loads(self._read_line())
            except json.JSONDecodeError:
                continue  # a log line that leaked into stdout
            if message.get("id") != request_id:
                continue
            if "error" in message:
                err = message["error"]
                raise MCPError(f"{method}: server error {err.get('code')}: "
                               f"{err.get('message')}")
            return message.get("result")

    # -- API -------------------------------------------------------------------
    def tools(self) -> list[dict]:
        """The server's tool catalog (name, description, inputSchema)."""
        with self._lock:
            self._ensure()
            result = self._request("tools/list") or {}
        return result.get("tools", [])

    def call(self, tool: str, arguments: dict | None = None) -> Any:
        """Call one tool. Returns ``structuredContent`` when present, else
        the concatenated text content. ``isError`` results raise MCPError —
        tool-level failures are failures of the step."""
        with self._lock:
            self._ensure()
            result = self._request("tools/call",
                                   {"name": tool, "arguments": arguments or {}})
        if not isinstance(result, dict):
            return result
        text = "\n".join(block.get("text", "")
                         for block in result.get("content") or []
                         if block.get("type") == "text")
        if result.get("isError"):
            raise MCPError(f"tool {tool!r} failed: {text or result!r}")
        if "structuredContent" in result:
            return result["structuredContent"]
        return text


def tool_step(client: MCPClient, tool: str, *, params: Sequence[str] | None = None,
              args: dict | None = None, build: Callable[[Any], dict] | str | None = None,
              out_key: str = "result", unwrap: Callable[[Any], Any] | None = None,
              name: str | None = None) -> Step:
    """One MCP tool as a flow step.

    Arguments sent to the tool: a dict payload as-is (or just its ``params``
    keys when given), merged over static ``args``; a non-dict payload as
    ``{"input": payload}``. When the server's schema differs from the payload
    shape, ``build`` (a ``payload -> arguments`` callable, or its import path
    for presets) replaces the default mapping entirely — e.g. throughline's
    own server nests the payload: ``build=lambda p: {"input": p}``. The
    (optionally ``unwrap``-ed) result lands at ``payload[out_key]`` for dict
    payloads, or is returned bare.
    """
    step_name = name or f"mcp:{tool}"
    if isinstance(build, str):
        from ..registry import resolve
        build = resolve(build)

    def fn(payload, ctx):
        if build is not None:
            arguments = build(payload)
        elif isinstance(payload, dict):
            selected = ({k: payload[k] for k in params if k in payload}
                        if params is not None else payload)
            arguments = {**(args or {}), **selected}
        else:
            arguments = {**(args or {}), "input": payload}
        ctx.metric("mcp.calls")
        result = client.call(tool, arguments)
        if unwrap is not None:
            result = unwrap(result)
        if isinstance(payload, dict):
            return {**payload, out_key: result}
        return result

    return Step(fn=fn, name=step_name, meta={"adapter": "mcp", "tool": tool})


def mcp_tool(command: Sequence[str], tool: str, *, env: dict | None = None,
             cwd: str | None = None, timeout: float = 60.0,
             **step_kwargs: Any) -> Step:
    """Preset-friendly factory: spawn-and-own an MCP server for one tool.

    The step owns its client (one server per step instance); flows that call
    several tools of the same server should share one ``MCPClient`` and use
    ``tool_step`` instead.
    """
    client = MCPClient(command, env=env, cwd=cwd, timeout=timeout)
    return tool_step(client, tool, **step_kwargs)
