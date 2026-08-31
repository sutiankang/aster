import asyncio
from dataclasses import replace
import hashlib
import json
import time

import pytest

from aster.agents import (
    EventLog,
    read_events,
    replay,
    PermissionBroker,
    PermissionDenied,
    ToolSpec,
    Workspace,
    ToolExecutor,
    sanitize,
)


def begin(log):
    log.append("thread.started", thread_id="thread", payload={})
    log.append("turn.started", thread_id="thread", turn_id="turn", payload={})


def test_single_writer_hash_chain_and_readonly_replay(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLog(path) as log:
        with pytest.raises(FileExistsError):
            EventLog(path)
        begin(log)
        for kind in ("prepared", "approved", "started"):
            log.append("tool." + kind, thread_id="thread", turn_id="turn", item_id="call")
        before = path.read_bytes()
        state = replay(path)
        assert state.items["call"]["status"] == "ambiguous"
        assert not state.approvals_active
        assert before == path.read_bytes()
    event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    event["payload"] = {"tampered": True}
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        read_events(path)


def test_approval_exact_binding_version_expiry_revoke_and_one_use(tmp_path):
    broker = PermissionBroker(tmp_path)
    spec = ToolSpec("read", "1", "digest", "read", "read")
    call = broker.prepare(spec, {"path": "a.txt"}, thread_id="t", turn_id="u")
    approval = broker.approve(call)
    with pytest.raises(PermissionDenied):
        broker.consume(
            replace(call, arguments_json='{"path":"secret.txt"}'),
            approval,
            spec,
            thread_id="t",
            turn_id="u",
        )
    with pytest.raises(PermissionDenied):
        broker.consume(call, approval, replace(spec, version="2"), thread_id="t", turn_id="u")
    with pytest.raises(PermissionDenied):
        broker.consume(call, approval, spec, thread_id="t", turn_id="other")
    broker.consume(call, approval, spec, thread_id="t", turn_id="u")
    with pytest.raises(PermissionDenied):
        broker.consume(call, approval, spec, thread_id="t", turn_id="u")
    revoked = broker.approve(call)
    broker.revoke(revoked)
    with pytest.raises(PermissionDenied):
        broker.consume(call, revoked, spec, thread_id="t", turn_id="u")
    expired = broker.approve(call, ttl_seconds=0.00001)
    time.sleep(0.001)
    with pytest.raises(PermissionDenied):
        broker.consume(call, expired, spec, thread_id="t", turn_id="u")


def test_workspace_escape_and_dangerous_command_fail_closed(tmp_path):
    workspace = Workspace(tmp_path)
    with pytest.raises(PermissionDenied):
        workspace.resolve("../outside", must_exist=False)
    with pytest.raises(PermissionDenied):
        workspace.resolve("\\\\server\\share")
    broker = PermissionBroker(workspace)
    dangerous = ToolSpec("shell", "1", "hash", "isolated_process", "run arbitrary code")
    call = broker.prepare(
        dangerous, {"argv": ["python", "-c", "print(1)"]}, thread_id="t", turn_id="u"
    )
    with pytest.raises(PermissionDenied, match="actual isolation"):
        broker.approve(call)


def test_symlink_cannot_expand_workspace(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "_outside.txt")
    outside.write_text("not authorized", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Host has no symlink privilege")
    with pytest.raises(PermissionDenied):
        Workspace(tmp_path).resolve("link.txt")


def test_read_patch_receipts_and_replay_never_reexecutes(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    target = workspace_dir / "note.txt"
    target.write_text("hello\napi_key=abcdef\n", encoding="utf-8")
    path = tmp_path / "events.jsonl"

    async def exercise():
        with EventLog(path) as log:
            begin(log)
            broker = PermissionBroker(workspace_dir)
            executor = ToolExecutor(broker, log, tmp_path / "receipts")
            call = executor.prepare(
                "workspace.read", {"path": "note.txt"}, thread_id="thread", turn_id="turn"
            )
            read = await executor.execute(
                call, broker.configured_approval(call), thread_id="thread", turn_id="turn"
            )
            assert read.status == "ok"
            assert "abcdef" not in json.dumps(read.model_view)
            original = target.read_bytes()
            args = {
                "path": "note.txt",
                "expected_sha256": hashlib.sha256(original).hexdigest(),
                "old_text": "hello",
                "new_text": "world",
            }
            patch = executor.prepare(
                "workspace.apply_patch", args, thread_id="thread", turn_id="turn"
            )
            assert broker.configured_approval(patch) is None
            result = await executor.execute(
                patch, broker.approve(patch), thread_id="thread", turn_id="turn"
            )
            assert result.status == "ok"
            assert target.read_text(encoding="utf-8").startswith("world")
        changed = target.read_bytes()
        restored = replay(path)
        assert target.read_bytes() == changed
        assert restored.items[patch.call_id]["status"] == "result_committed"
        assert all(not item.get("approval_active", False) for item in restored.items.values())

    asyncio.run(exercise())


def test_stale_patch_not_written_and_cannot_auto_retry(tmp_path):
    (tmp_path / "file.txt").write_text("current", encoding="utf-8")

    async def exercise():
        with EventLog(tmp_path / "events.jsonl") as log:
            begin(log)
            broker = PermissionBroker(tmp_path)
            executor = ToolExecutor(broker, log, tmp_path / "receipts")
            call = executor.prepare(
                "workspace.apply_patch",
                {
                    "path": "file.txt",
                    "expected_sha256": "bad",
                    "old_text": "current",
                    "new_text": "lost",
                },
                thread_id="thread",
                turn_id="turn",
            )
            result = await executor.execute(
                call, broker.approve(call), thread_id="thread", turn_id="turn"
            )
            assert result.status == "ambiguous"
            assert (tmp_path / "file.txt").read_text() == "current"
            with pytest.raises(PermissionDenied):
                await executor.execute(
                    call, broker.approve(call), thread_id="different", turn_id="turn"
                )

    asyncio.run(exercise())


def test_agent_tokenizer_metadata_missing_explicit_and_mutation_rejection():
    from aster.agents import NativeAgentPolicy
    from aster.core import digest_json

    class BareTokenizer:
        def encode(self, text):
            return [1]

    bare = NativeAgentPolicy(None, BareTokenizer())
    assert bare.tokenizer_fingerprint is None and bare.encode([]) == [1]
    explicit = NativeAgentPolicy(
        None, BareTokenizer(), tokenizer_fingerprint="explicit-fixture-tokenizer-v1"
    )
    assert explicit.tokenizer_fingerprint == "explicit-fixture-tokenizer-v1"

    class MutableTokenizer(BareTokenizer):
        version = 1

        def to_dict(self):
            return {"version": self.version}

    tokenizer = MutableTokenizer()
    policy = NativeAgentPolicy(None, tokenizer)
    assert policy.tokenizer_fingerprint == digest_json(tokenizer.to_dict())
    with pytest.raises(ValueError):
        NativeAgentPolicy(None, tokenizer, tokenizer_fingerprint="incorrect")
    tokenizer.version = 2
    with pytest.raises(ValueError, match="Tokenizer changed"):
        policy.encode([])


@pytest.mark.parametrize("payload", ['{"x":1e999}', '{"x":-1e999}', '{"x":NaN}', '{"x":1,"x":2}'])
def test_agent_json_rejects_numeric_overflow_and_duplicate_keys(payload):
    from aster.agents.events import strict_loads

    with pytest.raises(ValueError):
        strict_loads(payload)
