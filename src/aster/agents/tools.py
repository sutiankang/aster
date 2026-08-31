"""Controlled tool execution with durable receipts before result events are committed."""

from __future__ import annotations
import asyncio
from aster.core.async_work import settle_thread
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import tempfile
import time

from .events import canonical_json, digest
from .permissions import PermissionDenied, ToolSpec


def sanitize(value, *, max_chars=12000):
    """Return a labeled untrusted view with common secrets removed; raw receipts stay
    in controlled storage. This filter is not a general data-loss-prevention system."""
    text = canonical_json(value)
    text = re.sub(
        r"(?i)(api[_-]?key|access[_-]?token|password|secret)([\s\"':=]+)([^\s,;\"}]+)",
        r"\1\2[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+", "Bearer [REDACTED]", text)
    truncated = len(text) > max_chars
    return {"trust": "untrusted_tool_data", "content": text[:max_chars], "truncated": truncated}


@dataclass(frozen=True)
class ToolReceipt:
    call_id: str
    status: str
    raw_receipt_path: str | None
    raw_sha256: str | None
    model_view: dict
    elapsed_seconds: float


class ToolExecutor:
    def __init__(
        self,
        broker,
        event_log,
        receipt_dir,
        *,
        max_file_bytes=1024 * 1024,
        max_search_files=1000,
        max_search_hits=100,
    ):
        self.broker, self.log = broker, event_log
        self.receipt_dir = Path(receipt_dir).absolute()
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_bytes, self.max_search_files, self.max_search_hits = (
            max_file_bytes,
            max_search_files,
            max_search_hits,
        )
        if min(max_file_bytes, max_search_files, max_search_hits) < 1:
            raise ValueError("Tool resource limits must be positive")
        source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        declarations = [
            (
                "workspace.read",
                "read",
                "读取工作区UTF-8文本；path，start_line/max_lines可选。",
                self._read,
            ),
            (
                "workspace.search",
                "read",
                "有界字面检索；pattern，path可选；不执行正则或命令。",
                self._search,
            ),
            (
                "workspace.apply_patch",
                "workspace_write",
                "单文件精确替换；path/expected_sha256/old_text/new_text。",
                self._patch,
            ),
            (
                "command.run",
                "isolated_process",
                "仅真实隔离backend执行显式argv；无隔离时拒绝。",
                self._command,
            ),
        ]
        self._tools = {
            name: (ToolSpec(name, "1", source_digest, effect, description), function)
            for name, effect, description, function in declarations
        }

    @property
    def tool_specs(self):
        return tuple(spec for spec, _ in self._tools.values())

    def register(self, spec, function):

        if not isinstance(spec, ToolSpec) or spec.name in self._tools or not callable(function):
            raise ValueError("Tool name must be unique and implementation callable")
        self._tools[spec.name] = (spec, function)

    def prepare(self, name, arguments, *, thread_id, turn_id):
        if name not in self._tools:
            raise PermissionDenied("Tool is not registered")
        spec, _ = self._tools[name]
        call = self.broker.prepare(spec, arguments, thread_id=thread_id, turn_id=turn_id)
        self.log.append(
            "tool.prepared",
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=call.call_id,
            payload={
                "tool": spec.__dict__,
                "binding": call.binding,
                "arguments_digest": digest(arguments),
                "arguments_view": sanitize(arguments),
                "workspace": call.workspace,
            },
        )
        return call

    def deny(self, call):
        self.log.append(
            "tool.denied",
            thread_id=call.thread_id,
            turn_id=call.turn_id,
            item_id=call.call_id,
            payload={"reason": "permission_not_granted"},
        )

    async def execute(self, call, approval, *, thread_id, turn_id):
        spec, function = self._tools.get(call.tool.name, (None, None))
        self.broker.consume(call, approval, spec, thread_id=thread_id, turn_id=turn_id)
        self.log.append(
            "tool.approved",
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=call.call_id,
            payload={"binding": call.binding, "scope": approval.scope},
        )
        self.log.append(
            "tool.started",
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=call.call_id,
            payload={"binding": call.binding},
        )
        started, cancelled = time.monotonic(), False
        try:
            work = await settle_thread(function, call.arguments)
            cancelled = work.cancelled
            raw = work.unwrap()
            status = "ok"
        except Exception as error:
            if call.tool.effect != "read":
                self.log.append(
                    "tool.ambiguous",
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=call.call_id,
                    payload={"reason": "side_effect_outcome_not_proven"},
                )
                receipt = ToolReceipt(
                    call.call_id,
                    "ambiguous",
                    None,
                    None,
                    {"trust": "untrusted_tool_data", "error": "manual_resolution_required"},
                    time.monotonic() - started,
                )
                if cancelled:
                    raise asyncio.CancelledError from error
                return receipt
            status, raw = "error", {"error": "tool_execution_failed"}
        data = {"call_id": call.call_id, "binding": call.binding, "status": status, "result": raw}
        encoded = canonical_json(data).encode("utf-8")
        path = self.receipt_dir / (call.call_id + ".json")

        with path.open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        sha = hashlib.sha256(encoded).hexdigest()
        receipt = ToolReceipt(
            call.call_id, status, str(path), sha, sanitize(raw), time.monotonic() - started
        )
        self.log.append(
            "tool.result_committed",
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=call.call_id,
            payload=receipt.__dict__,
        )
        if cancelled:
            raise asyncio.CancelledError
        return receipt

    @staticmethod
    def _keys(arguments, required, optional=()):
        if not set(required) <= arguments.keys() or set(arguments) - set(required) - set(optional):
            raise ValueError("Tool arguments do not match its exact schema")

    def _bytes(self, path):
        self.broker.workspace._reject_links(path)
        if not path.is_file() or path.stat().st_size > self.max_file_bytes:
            raise ValueError("Not a bounded regular text file")
        with path.open("rb") as source:
            data = source.read(self.max_file_bytes + 1)
        if len(data) > self.max_file_bytes:
            raise ValueError("File grew beyond read limit")
        self.broker.workspace._reject_links(path)
        return data

    def _read(self, arguments):
        self._keys(arguments, ("path",), ("start_line", "max_lines"))
        path = self.broker.workspace.resolve(arguments["path"])
        data = self._bytes(path)
        text = data.decode("utf-8")
        start, maximum = arguments.get("start_line", 1), arguments.get("max_lines", 200)
        if (
            type(start) is not int
            or type(maximum) is not int
            or min(start, maximum) < 1
            or maximum > 1000
        ):
            raise ValueError("Invalid line bounds")
        lines = text.splitlines(keepends=True)
        return {
            "path": str(path.relative_to(self.broker.workspace.root)),
            "content": "".join(lines[start - 1 : start - 1 + maximum]),
            "start_line": start,
            "sha256": hashlib.sha256(data).hexdigest(),
            "truncated": start - 1 + maximum < len(lines),
        }

    def _search(self, arguments):
        self._keys(arguments, ("pattern",), ("path",))
        pattern = arguments["pattern"]
        if not isinstance(pattern, str) or not pattern or len(pattern) > 1000:
            raise ValueError("Invalid literal pattern")
        root = self.broker.workspace.resolve(arguments.get("path", "."))
        hits, examined = [], 0
        paths = [root] if root.is_file() else root.rglob("*")
        for candidate in paths:
            examined += 1
            if examined > self.max_search_files or len(hits) >= self.max_search_hits:
                break
            try:
                path = self.broker.workspace.resolve(str(candidate))
                if not path.is_file():
                    continue
                text = self._bytes(path).decode("utf-8")
            except (ValueError, UnicodeError, PermissionDenied, OSError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if pattern in line:
                    hits.append(
                        {
                            "path": str(path.relative_to(self.broker.workspace.root)),
                            "line": number,
                            "text": line[:1000],
                        }
                    )
                    if len(hits) >= self.max_search_hits:
                        break
        return {
            "hits": hits,
            "examined_entries": min(examined, self.max_search_files),
            "truncated": examined > self.max_search_files or len(hits) >= self.max_search_hits,
        }

    def _patch(self, arguments):
        self._keys(arguments, ("path", "expected_sha256", "old_text", "new_text"))
        path = self.broker.workspace.resolve(arguments["path"])
        original = self._bytes(path)
        old, new = arguments["old_text"], arguments["new_text"]
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("Patch requires nonempty exact old_text and text new_text")
        if hashlib.sha256(original).hexdigest() != arguments["expected_sha256"]:
            raise ValueError("Patch base changed since approval")
        text = original.decode("utf-8")
        if text.count(old) != 1:
            raise ValueError("Patch anchor must match exactly once")
        result = text.replace(old, new, 1).encode("utf-8")
        if len(result) > self.max_file_bytes:
            raise ValueError("Patched file exceeds resource policy")
        descriptor, temporary = tempfile.mkstemp(prefix=".aster-patch-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(result)
                output.flush()
                os.fsync(output.fileno())

            if self._bytes(path) != original:
                raise ValueError("Patch target changed before commit")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {
            "path": str(path.relative_to(self.broker.workspace.root)),
            "before_sha256": hashlib.sha256(original).hexdigest(),
            "after_sha256": hashlib.sha256(result).hexdigest(),
            "replacements": 1,
        }

    def _command(self, arguments):
        self._keys(arguments, ("argv",), ("timeout_seconds",))
        backend = self.broker.isolation_backend
        if backend is None:
            raise PermissionDenied("No actual OS isolation backend")
        argv = arguments["argv"]
        timeout = arguments.get("timeout_seconds", 30.0)
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) or "\x00" in value for value in argv)
        ):
            raise ValueError("Command requires an explicit argv vector")
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
            raise ValueError("Command timeout exceeds policy")

        allowed = {"SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "LANG", "LC_ALL"}
        environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}

        return backend.run(
            argv=argv,
            cwd=str(self.broker.workspace.root),
            timeout_seconds=timeout,
            environment=environment,
        )
