"""MCP adapter: agent<->flow boundary as serialized snapshots + handles."""

import io
import json
import unittest

import throughline as tl
from throughline.contrib.mcp import MCPServer, project_result
from throughline.store import MemoryArtifactStore


def _make_server(**kwargs) -> MCPServer:
    flow = tl.Flow(
        [lambda p: {"question": str(p)}, lambda p: {**p, "answer": f"42 (re: {p['question']})"}],
        middleware=[tl.modules.MetricsMiddleware()],
        name="qa",
    )
    big_flow = tl.Flow([lambda p: ["item"] * 5000], name="bulk")
    return MCPServer(presets=[], flows={"run_qa": flow, "run_bulk": big_flow}, **kwargs)


def _call(server: MCPServer, method: str, params: dict | None = None, id: int = 1) -> dict:
    return server.handle({"jsonrpc": "2.0", "id": id, "method": method,
                          "params": params or {}})


def _tool_payload(response: dict) -> dict:
    return json.loads(response["result"]["content"][0]["text"])


class MCPProtocolTests(unittest.TestCase):
    def test_initialize(self):
        response = _call(_make_server(), "initialize")
        self.assertEqual(response["result"]["serverInfo"]["name"], "throughline")
        self.assertIn("tools", response["result"]["capabilities"])

    def test_tools_list_exposes_flows_and_get_artifact(self):
        response = _call(_make_server(), "tools/list")
        names = [t["name"] for t in response["result"]["tools"]]
        self.assertIn("run_qa", names)
        self.assertIn("get_artifact", names)
        qa = next(t for t in response["result"]["tools"] if t["name"] == "run_qa")
        self.assertIn("input", qa["inputSchema"]["properties"])
        self.assertIn("trace_id", qa["inputSchema"]["properties"])

    def test_notifications_get_no_response(self):
        server = _make_server()
        self.assertIsNone(server.handle({"jsonrpc": "2.0",
                                         "method": "notifications/initialized"}))

    def test_unknown_tool_is_a_tool_error(self):
        response = _call(_make_server(), "tools/call", {"name": "nope"})
        self.assertTrue(response["result"]["isError"])
        self.assertIn("unknown tool", response["result"]["content"][0]["text"])


class AgentCallsFlowTests(unittest.TestCase):
    def test_tool_call_runs_flow_and_projects_result(self):
        response = _call(_make_server(), "tools/call",
                         {"name": "run_qa", "arguments": {"input": "meaning of life"}})
        payload = _tool_payload(response)
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(payload["output"]["answer"], "42 (re: meaning of life)")
        self.assertIn("run_id", payload)

    def test_trace_id_stamps_every_event(self):
        server = _make_server()
        flow = server._flows["run_qa"]
        seen = []
        original_run = flow.run

        def spying_run(payload=None, **kwargs):
            kwargs["ctx"].events.subscribe(seen.append)
            return original_run(payload, **kwargs)

        flow.run = spying_run
        response = _call(server, "tools/call",
                         {"name": "run_qa",
                          "arguments": {"input": "q", "trace_id": "agent-trace-7"}})
        payload = _tool_payload(response)
        self.assertEqual(payload["trace_id"], "agent-trace-7")
        self.assertTrue(seen)
        self.assertTrue(all(e["trace_id"] == "agent-trace-7" for e in seen))

    def test_oversized_output_becomes_a_handle(self):
        server = _make_server(max_result_bytes=1024)
        response = _call(server, "tools/call",
                         {"name": "run_bulk", "arguments": {"input": ""}})
        payload = _tool_payload(response)
        self.assertIn("$artifact", payload["output"])
        self.assertIn("preview", payload["output"])
        # the agent pulls a slice through get_artifact
        slice_response = _call(server, "tools/call",
                               {"name": "get_artifact",
                                "arguments": {"artifact": payload["output"]["$artifact"],
                                              "start": 0, "stop": 3}})
        self.assertEqual(json.loads(slice_response["result"]["content"][0]["text"]),
                         ["item", "item", "item"])

    def test_full_fetch_of_oversized_artifact_is_refused_with_guidance(self):
        server = _make_server(max_result_bytes=1024)
        response = _call(server, "tools/call",
                         {"name": "run_bulk", "arguments": {"input": ""}})
        ref = _tool_payload(response)["output"]["$artifact"]
        fetch = _call(server, "tools/call",
                      {"name": "get_artifact", "arguments": {"artifact": ref}})
        self.assertTrue(fetch["result"]["isError"])
        self.assertIn("slice", fetch["result"]["content"][0]["text"])

    def test_expired_handle_is_a_readable_tool_error(self):
        server = _make_server()
        gone = f"{server.session}/away"  # this server's session, but expired
        response = _call(server, "tools/call",
                         {"name": "get_artifact", "arguments": {"artifact": gone}})
        self.assertTrue(response["result"]["isError"])
        note = json.loads(response["result"]["content"][0]["text"])
        self.assertIn("re-run", note["error"])
        self.assertEqual(note["expired"], gone)

    def test_foreign_session_handles_are_not_served(self):
        # a shared store must not be readable by session-guessing: only
        # handles minted by this server's own session come back
        store = MemoryArtifactStore()
        secret = store.put("other tenant's data", session="tenant-b")
        server = _make_server(store=store)
        response = _call(server, "tools/call",
                         {"name": "get_artifact",
                          "arguments": {"artifact": secret.id}})
        self.assertTrue(response["result"]["isError"])
        note = json.loads(response["result"]["content"][0]["text"])
        self.assertIn("not part of this server session", note["error"])
        self.assertNotIn("other tenant", json.dumps(response))


class ProjectResultTests(unittest.TestCase):
    def test_projection_includes_metrics_and_lineage_stats(self):
        flow = tl.Flow(
            [lambda p: {"answer": "hi"}],
            middleware=[tl.modules.MetricsMiddleware(),
                        tl.modules.LineageMiddleware(extract="answer")],
        )
        report = project_result(flow.run("q"))
        self.assertEqual(report["output"]["answer"], "hi")
        self.assertIn("metrics", report)
        self.assertIn("lineage", report)
        self.assertEqual(report["lineage"]["lines"], 1)

    def test_projection_respects_byte_budget(self):
        flow = tl.Flow([lambda p: "x" * 10_000])
        store = MemoryArtifactStore()
        report = project_result(flow.run("q"), store=store, max_bytes=100)
        self.assertIn("$artifact", report["output"])
        self.assertEqual(store.get(report["output"]["$artifact"]), "x" * 10_000)


class StdioTransportTests(unittest.TestCase):
    def test_serve_stdio_roundtrip_and_session_cleanup(self):
        server = _make_server(max_result_bytes=1024)
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "run_bulk", "arguments": {"input": ""}}},
        ]
        stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
        stdout = io.StringIO()
        server.serve_stdio(stdin=stdin, stdout=stdout)

        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(len(responses), 2)  # notification got no reply
        self.assertEqual(responses[0]["id"], 1)
        # session namespace dropped on shutdown: the handle is now a dead lease
        self.assertEqual(len(server.store), 0)

    def test_invalid_json_gets_a_parse_error_not_silence(self):
        server = _make_server()
        stdin = io.StringIO("this is not json\n")
        stdout = io.StringIO()
        server.serve_stdio(stdin=stdin, stdout=stdout)
        response = json.loads(stdout.getvalue())
        self.assertEqual(response["error"]["code"], -32700)


if __name__ == "__main__":
    unittest.main()
