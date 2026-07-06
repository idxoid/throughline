"""MCP client adapter, integration-tested against throughline's own MCP
server (`python -m throughline mcp --preset demo`) — the two speak the same
newline-delimited JSON-RPC, so the server doubles as the test fixture."""

import json
import os
import sys
import unittest
from pathlib import Path

import throughline as tl
from throughline.adapters.mcp import MCPClient, MCPError, mcp_tool, tool_step

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_CMD = [sys.executable, "-m", "throughline", "mcp"]
SERVER_ENV = {"PYTHONPATH": str(REPO_ROOT / "src")}


def nest_input(payload):
    """throughline's own server nests the flow payload under "input"."""
    return {"input": payload}


def demo_client(**kwargs) -> MCPClient:
    return MCPClient(SERVER_CMD, env=SERVER_ENV, cwd=str(REPO_ROOT),
                     timeout=30, **kwargs)


class MCPClientTests(unittest.TestCase):
    def test_handshake_and_tool_catalog(self):
        with demo_client() as client:
            tools = client.tools()
        names = {tool["name"] for tool in tools}
        self.assertIn("run_demo", names)
        self.assertIn("get_artifact", names)

    def test_call_returns_text_result(self):
        with demo_client() as client:
            raw = client.call("run_demo",
                              {"input": {"question": "how does lineage work?"}})
        payload = json.loads(raw)
        self.assertIn("how does lineage work", payload["output"]["answer"])
        self.assertIn("lineage", payload)

    def test_tool_error_raises(self):
        with demo_client() as client:
            with self.assertRaises(MCPError) as caught:
                client.call("no_such_tool", {})
        self.assertIn("no_such_tool", str(caught.exception))

    def test_tool_step_in_a_flow(self):
        with demo_client() as client:
            flow = tl.Flow([
                tool_step(client, "run_demo", build=nest_input,
                          out_key="demo", unwrap=json.loads),
            ], middleware=[tl.modules.MetricsMiddleware()])
            result = flow.run({"question": "what is a preset?", "extra": "kept"})
        self.assertIn("what is a preset?", result.output["demo"]["output"]["answer"])
        self.assertEqual(result.output["extra"], "kept")
        self.assertEqual(result.metrics["counters"].get("mcp.calls"), 1)

    def test_mcp_tool_factory_owns_its_client(self):
        step = mcp_tool(SERVER_CMD, "run_demo", env=SERVER_ENV,
                        cwd=str(REPO_ROOT), unwrap=json.loads,
                        build="tests.test_mcp_client:nest_input")  # import-path form
        result = tl.Flow([step]).run({"question": "hi"})
        self.assertIn("hi", result.output["result"]["output"]["answer"])

    def test_dead_server_reports_exit(self):
        client = MCPClient([sys.executable, "-c", "import sys; sys.exit(3)"],
                           timeout=5)
        with self.assertRaises(MCPError):
            client.tools()
        client.close()


if __name__ == "__main__":
    unittest.main()
