import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

from aster.agents import EventLog, PermissionBroker, ToolExecutor, PermissionDenied
from aster.agents.mcp_stdio import LocalMCPProcessGrant, MCPStdioClient


def grant():
    script = Path(__file__).parents[1] / "fixtures/mcp_stdio_server.py"
    executable = Path(sys.executable).resolve()
    return LocalMCPProcessGrant(
        (str(executable), "-u", str(script)),
        str(script.parent),
        hashlib.sha256(executable.read_bytes()).hexdigest(),
        source_files=((str(script), hashlib.sha256(script.read_bytes()).hexdigest()),),
        environment=(("ASTER_EXPLICIT_VALUE", "granted"),),
        trusted_local_process=True,
    )


def test_real_stdio_tools_reuse_permission_receipt_replay_and_do_not_inherit_secrets(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ASTER_HOST_ONLY_SECRET", "must_not_reach_server")
    with MCPStdioClient(
        grant(), server_id="stdio", allowed_tools=["echo", "inspect"], max_calls=2
    ) as client:
        client.initialize()
        result = client.call_tool("inspect", {})
        assert json.loads(result["content"][0]["text"]) == {
            "inherited_secret": False,
            "explicit": "granted",
        }

        async def run():
            with EventLog(tmp_path / "events.jsonl") as log:
                log.append("thread.started", thread_id="t")
                log.append("turn.started", thread_id="t", turn_id="u")
                broker = PermissionBroker(tmp_path, external_authorizer=client.authorizes)
                executor = ToolExecutor(broker, log, tmp_path / "receipts")
                client.register_tools(executor)
                call = executor.prepare(
                    "mcp.stdio.echo", {"value": "actual stdio"}, thread_id="t", turn_id="u"
                )
                assert broker.configured_approval(call) is None
                receipt = await executor.execute(
                    call, broker.approve(call), thread_id="t", turn_id="u"
                )
                assert receipt.status == "ok" and "ACTUAL STDIO" in receipt.model_view["content"]
                assert "stdio:" in client.endpoint and client._calls == 2
                with pytest.raises(PermissionDenied):
                    broker.approve(call)

        asyncio.run(run())
        assert all(not entry["trusted"] for entry in client.drain_notifications())
        with pytest.raises(PermissionDenied):
            client.call_tool("echo", {})
    assert client._process.poll() is not None and not client._read_thread.is_alive()


def test_stdio_cancel_progresses_while_tool_call_is_waiting():
    with MCPStdioClient(
        grant(), server_id="cancel", allowed_tools=["wait"], timeout_seconds=3
    ) as client:
        client.initialize()
        with ThreadPoolExecutor(1) as worker:
            pending = worker.submit(client.call_tool, "wait", {})
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with client._pending_lock:
                    identifiers = list(client._pending)
                if identifiers:
                    break
                time.sleep(0.005)
            assert identifiers
            client.cancel(identifiers[0])
            assert pending.result(timeout=2)["content"][0]["text"] == "CANCELLED"


@pytest.mark.parametrize(
    "name,error", [("bad", ValueError), ("large", ValueError), ("exit", EOFError)]
)
def test_malformed_oversize_or_exited_server_fails_without_replay(name, error):
    with MCPStdioClient(
        grant(), server_id="bad", allowed_tools=[name], max_response_bytes=2048
    ) as client:
        client.initialize()
        with pytest.raises(error):
            client.call_tool(name, {})
        assert client._calls == 1 and not client._initialized
        with pytest.raises(PermissionDenied):
            client.call_tool(name, {})


def test_total_deadline_terminates_server_and_consumes_attempt():
    with MCPStdioClient(
        grant(), server_id="timeout", allowed_tools=["wait"], timeout_seconds=0.5
    ) as client:
        client.initialize()
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            client.call_tool("wait", {})
        assert time.monotonic() - start < 2 and client._calls == 1
    assert client._process.poll() is not None


def test_server_cannot_obtain_undeclared_sampling_callback():
    with MCPStdioClient(grant(), server_id="callback", allowed_tools=["callback"]) as client:
        client.initialize()
        assert client.call_tool("callback", {})["content"][0]["text"] == "-32601"


def test_process_permission_and_source_identity_checked_before_launch():
    valid = grant()
    for invalid in (
        replace(valid, trusted_local_process=False),
        replace(valid, executable_sha256="0" * 64),
        replace(valid, source_files=((valid.source_files[0][0], "0" * 64),)),
    ):
        with pytest.raises(PermissionDenied):
            MCPStdioClient(invalid, server_id="x", allowed_tools=["echo"])
