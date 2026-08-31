import asyncio
import pytest
from aster.agents import (
    MCPStdioClient,
    MCPContextProvider,
    PermissionDenied,
    EventLog,
    PermissionBroker,
    ToolExecutor,
)
from test_mcp_stdio import grant


def test_real_stdio_resources_prompts_pagination_and_permission_receipts(tmp_path):
    with MCPStdioClient(grant(), server_id="context", allowed_tools=["echo"]) as client:
        client.initialize()
        context = MCPContextProvider(
            client, allowed_resources=["test://text", "test://blob"], allowed_prompts=["review"]
        )
        assert client._calls == 3
        assert context.read_resource("test://blob")["contents"][0]["blob"] == "YWJj"
        prompt = context.get_prompt("review", {"code": "print(1)"})
        assert prompt["trust"] == "untrusted_mcp_prompt"
        with pytest.raises(PermissionDenied):
            context.read_resource("file:///private")
        with pytest.raises(ValueError):
            context.get_prompt("review", {"unlisted": "value"})

        async def run():
            with EventLog(tmp_path / "events.jsonl") as log:
                log.append("thread.started", thread_id="t")
                log.append("turn.started", thread_id="t", turn_id="u")
                broker = PermissionBroker(tmp_path, external_authorizer=context.authorizes)
                executor = ToolExecutor(broker, log, tmp_path / "receipts")
                context.register_tools(executor)
                spec = context._specs["resource", "test://text"]
                call = executor.prepare(spec.name, {}, thread_id="t", turn_id="u")
                assert broker.configured_approval(call) is None
                receipt = await executor.execute(
                    call, broker.approve(call), thread_id="t", turn_id="u"
                )
                assert (
                    receipt.status == "ok"
                    and "untrusted_mcp_resource" in receipt.model_view["content"]
                )
                assert "Ignore prior" in receipt.model_view["content"]

        asyncio.run(run())


def test_expiry_budget_schema_uri_escape_and_new_session_revoke_old_context(monkeypatch):
    with MCPStdioClient(grant(), server_id="context", allowed_tools=["echo"]) as client:
        client.initialize()
        context = MCPContextProvider(client, allowed_resources=["test://text"])
        original = client._rpc

        def escaped(method, params=None, **kwargs):
            if method == "resources/read":
                return {"contents": [{"uri": "file:///secret", "text": "secret"}]}
            return original(method, params, **kwargs)

        monkeypatch.setattr(client, "_rpc", escaped)
        with pytest.raises(ValueError, match="escaped"):
            context.read_resource("test://text")
        monkeypatch.setattr(client, "_rpc", original)
        client._initialized = False
        client.initialize()
        with pytest.raises(PermissionDenied):
            context.read_resource("test://text")
        context = MCPContextProvider(client, allowed_resources=["test://text"])
        client.max_calls = client._calls
        with pytest.raises(PermissionDenied):
            context.read_resource("test://text")


def test_failed_initialize_does_not_leave_half_authorized_tools(monkeypatch):
    with MCPStdioClient(grant(), server_id="context", allowed_tools=["unavailable"]) as client:
        with pytest.raises(ValueError):
            client.initialize()
        assert not client._initialized and client._tools == client._specs == {}
        assert client.server_info is None
        with pytest.raises(PermissionDenied):
            client.call_tool("unavailable", {})


def test_missing_context_capability_and_ungranted_prompt_are_rejected():
    with MCPStdioClient(grant(), server_id="context", allowed_tools=["echo"]) as client:
        client.initialize()
        client.server_capabilities.pop("resources")
        with pytest.raises(ValueError):
            MCPContextProvider(client, allowed_resources=["test://text"])
        context = MCPContextProvider(client, allowed_prompts=["review"])
        with pytest.raises(PermissionDenied):
            context.get_prompt("another", {})
        with pytest.raises(ValueError):
            context.get_prompt("review", {})
