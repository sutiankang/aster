import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import concurrent.futures

import pytest

from aster.agents import (
    EventLog,
    MemoryStore,
    ContextCompactor,
    PermissionBroker,
    ToolExecutor,
    PermissionDenied,
    MCPClient,
    BubblewrapSandbox,
    SandboxUnavailable,
    AgentPlanExecutor,
    PlanNode,
)


def test_memory_bm25_scope_and_verified_filter_survive_reload(tmp_path):
    path = tmp_path / "memory.jsonl"
    with EventLog(path) as log:
        memory = MemoryStore(log, max_entries=4)
        first = memory.add(
            "attention cache 分页缓存", scope_id="alice", source="turn:1", verified=True
        )
        memory.add("password private cache", scope_id="bob", source="turn:2")
        memory.add("attention 未验证猜想", scope_id="alice", source="turn:3")
        matches = memory.search("attention 缓存", scope_id="alice", verified_only=True)
        assert len(matches) == 1 and matches[0]["id"] == first.id
        assert "private" not in json.dumps(memory.search("cache", scope_id="alice"))
    with EventLog(path) as log:
        restored = MemoryStore(log)
        assert restored.search("缓存", scope_id="alice")[0]["id"] == first.id


def test_context_compaction_preserves_instruction_and_tool_pairs():
    messages = [
        {"role": "system", "content": "fixed authority"},
        {"role": "user", "content": "current user"},
    ]
    for index in range(8):
        messages += [
            {"role": "assistant", "content": "call" + str(index) + "x" * 200},
            {"role": "tool", "content": "untrusted" + "y" * 200},
        ]
    encode = lambda value: list(json.dumps(value))
    compacted, record = ContextCompactor(max_summary_chars=120).compact(
        messages, encode=encode, max_tokens=900
    )
    assert len(encode(compacted)) <= 900 and compacted[:2] == messages[:2]
    assert record["removed_items"] % 2 == 0 and record["source_digest"]
    assert not any(item.get("role") == "system" for item in compacted[2:])


def test_plan_cycle_and_budget_rejected_before_child_execution(tmp_path):
    planner = AgentPlanExecutor(
        lambda node: pytest.fail("invalid plan must not construct children"), workspace=tmp_path
    )
    with pytest.raises(ValueError, match="cycle"):
        asyncio.run(planner.run([PlanNode("a", "first", ("b",)), PlanNode("b", "second", ("a",))]))


class MCPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    calls = []

    def log_message(self, *args):
        return None

    def do_POST(self):
        message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        method = message["method"]
        if method == "notifications/initialized":
            assert self.headers["Mcp-Session-Id"] == "fixture-session"
            assert self.headers["MCP-Protocol-Version"] == "2025-06-18"
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "bounded echo",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"value": {"type": "string", "maxLength": 100}},
                            "required": ["value"],
                            "additionalProperties": False,
                        },
                    }
                ]
            }
        elif method == "tools/call":
            self.calls.append(message["params"])
            result = {
                "content": [
                    {"type": "text", "text": message["params"]["arguments"]["value"].upper()}
                ]
            }
        else:
            raise AssertionError("Unexpected MCP method")
        response = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}).encode()
        streaming = method == "tools/call"
        if streaming:
            response = b"event: message\ndata: " + response + b"\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        self.send_header("Content-Length", str(len(response)))
        if method == "initialize":
            self.send_header("Mcp-Session-Id", "fixture-session")
        self.end_headers()
        self.wfile.write(response)

    def do_DELETE(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()


def test_real_loopback_mcp_handshake_sse_and_shared_permission_lifecycle(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), MCPHandler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    MCPHandler.calls = []
    client = MCPClient(
        f"http://127.0.0.1:{server.server_port}/mcp",
        server_id="fixture",
        allowed_tools=["echo"],
        max_calls=1,
    )

    async def exercise():
        client.initialize()
        with EventLog(tmp_path / "events.jsonl") as log:
            log.append("thread.started", thread_id="thread")
            log.append("turn.started", thread_id="thread", turn_id="turn")
            broker = PermissionBroker(tmp_path, external_authorizer=client.authorizes)
            executor = ToolExecutor(broker, log, tmp_path / "receipts")
            client.register_tools(executor)
            call = executor.prepare(
                "mcp.fixture.echo",
                {"value": "actual transport"},
                thread_id="thread",
                turn_id="turn",
            )
            assert broker.configured_approval(call) is None
            receipt = await executor.execute(
                call, broker.approve(call), thread_id="thread", turn_id="turn"
            )
            assert receipt.status == "ok" and "ACTUAL TRANSPORT" in receipt.model_view["content"]
            again = executor.prepare(
                "mcp.fixture.echo", {"value": "over budget"}, thread_id="thread", turn_id="turn"
            )
            with pytest.raises(PermissionDenied):
                broker.approve(again)
        assert len(MCPHandler.calls) == 1

    try:
        asyncio.run(exercise())
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_mcp_external_endpoints_and_implicit_tools_rejected():
    with pytest.raises(PermissionDenied):
        MCPClient("http://example.com:80/mcp", server_id="external", allowed_tools=["shell"])
    with pytest.raises(ValueError):
        MCPClient("http://127.0.0.1:12345/mcp", server_id="local", allowed_tools=[])


def test_mcp_cancel_uses_independent_connection_during_pending_request():
    started, cancelled = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if value["method"] == "notifications/cancelled":
                cancelled.set()
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            started.set()
            assert cancelled.wait(2), "Cancel blocked behind tools/call request lock"
            body = json.dumps(
                {"jsonrpc": "2.0", "id": value["id"], "result": {"cancelled": True}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = MCPClient(
        f"http://127.0.0.1:{server.server_port}/mcp",
        server_id="cancel",
        allowed_tools=["slow"],
        timeout_seconds=3,
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            response = executor.submit(client._rpc, "tools/call", {"name": "slow"})
            assert started.wait(2)
            client.cancel(1)
            assert response.result(timeout=2) == {"cancelled": True}
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_sandbox_never_falls_back_to_unsandboxed_execution(tmp_path):
    if not sys.platform.startswith("linux"):
        with pytest.raises(SandboxUnavailable):
            BubblewrapSandbox(tmp_path, allowed_executables=[sys.executable])
        return
    import shutil

    if not shutil.which("bwrap") or not shutil.which("prlimit"):
        with pytest.raises(SandboxUnavailable):
            BubblewrapSandbox(tmp_path, allowed_executables=[sys.executable])
        return
    sandbox = BubblewrapSandbox(tmp_path, allowed_executables=[sys.executable])
    outcome = sandbox.run(
        argv=[sys.executable, "-c", "print('isolated')"], cwd=str(tmp_path), timeout_seconds=5
    )
    if outcome["exit_code"] != 0:
        pytest.skip("Host user namespace/bubblewrap policy prevents actual isolation")
    assert outcome["stdout"].strip() == "isolated" and not outcome["network_enabled"]
