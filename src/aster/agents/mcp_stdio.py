"""Bounded MCP stdio transport for explicitly trusted local subprocesses."""

from collections import deque
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .events import canonical_json, digest, strict_loads
from .mcp import MCPClient
from .permissions import PermissionDenied


@dataclass(frozen=True)
class LocalMCPProcessGrant:
    argv: tuple[str, ...]
    cwd: str
    executable_sha256: str

    source_files: tuple[tuple[str, str], ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    trusted_local_process: bool = False

    def validate(self):
        if not self.trusted_local_process or type(self.argv) is not tuple or not self.argv:
            raise PermissionDenied("Host must explicitly authorize a trusted local MCP process")
        if any(not isinstance(x, str) or "\x00" in x for x in self.argv):
            raise ValueError("MCP process needs literal argv without NUL")
        executable, cwd = Path(self.argv[0]), Path(self.cwd)
        if (
            not executable.is_absolute()
            or not executable.is_file()
            or not cwd.is_absolute()
            or not cwd.is_dir()
        ):
            raise PermissionDenied("MCP executable and cwd must be existing absolute paths")
        for name, expected in ((str(executable), self.executable_sha256), *self.source_files):
            source = Path(name)
            if (
                not source.is_absolute()
                or not source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest() != expected
            ):
                raise PermissionDenied("MCP executable/source changed after host grant")
        environment = dict(self.environment)
        if len(environment) != len(self.environment) or any(
            not isinstance(k, str)
            or not k
            or "=" in k
            or "\x00" in k
            or not isinstance(v, str)
            or "\x00" in v
            for k, v in environment.items()
        ):
            raise ValueError("Invalid explicit MCP environment")
        return environment

    @property
    def fingerprint(self):

        return digest(
            {
                "argv": self.argv,
                "cwd": self.cwd,
                "executable": self.executable_sha256,
                "source_files": self.source_files,
                "environment_digest": digest(dict(self.environment)),
            }
        )


@dataclass
class _PendingRPC:
    ready: threading.Event = field(default_factory=threading.Event)
    sent: threading.Event = field(default_factory=threading.Event)
    message: dict | None = None
    error: Exception | None = None


class MCPStdioClient(MCPClient):
    """Accept bounded newline JSON-RPC on stdout; drain stderr separately so logs
    cannot be mistaken for tool responses."""

    def __init__(
        self,
        grant,
        *,
        server_id,
        allowed_tools,
        grant_ttl_seconds=300.0,
        timeout_seconds=10.0,
        max_response_bytes=1024 * 1024,
        max_calls=100,
    ):
        if not isinstance(grant, LocalMCPProcessGrant):
            raise PermissionDenied("A host-owned process grant is required")
        environment = grant.validate()
        self._configure(
            "stdio:" + grant.fingerprint,
            server_id=server_id,
            allowed_tools=allowed_tools,
            grant_ttl_seconds=grant_ttl_seconds,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
            max_calls=max_calls,
        )

        if os.name == "nt" and "SystemRoot" not in environment:
            environment["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
        self._send_lock = threading.Lock()
        self._pending_lock = threading.RLock()
        self._pending = {}
        self._transport_error = None
        self._closed = False
        self._notifications = deque(maxlen=64)
        self._stderr = bytearray()
        self._stderr_lock = threading.Lock()
        self._process = subprocess.Popen(
            grant.argv,
            cwd=grant.cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            bufsize=0,
            close_fds=True,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._read_thread = threading.Thread(
            target=self._read_stdout, name="aster-mcp-stdout", daemon=True
        )
        self._err_thread = threading.Thread(
            target=self._read_stderr, name="aster-mcp-stderr", daemon=True
        )
        self._read_thread.start()
        self._err_thread.start()

    def _fail(self, error):
        with self._pending_lock:
            self._transport_error = self._transport_error or error
            self._initialized = False
            for request in self._pending.values():
                request.error = self._transport_error
                request.ready.set()

    def _send(self, message):
        data = canonical_json(message).encode("utf-8") + b"\n"
        if len(data) > self.max_response_bytes:
            raise ValueError("MCP request exceeds message bound")
        with self._send_lock:
            if (
                self._closed
                or self._transport_error is not None
                or self._process.poll() is not None
            ):
                raise RuntimeError("MCP stdio transport is closed/failed; no side-effect retry")

            view = memoryview(data)
            while view:
                written = self._process.stdin.write(view)
                if not written:
                    raise BrokenPipeError("MCP stdin closed during message")
                view = view[written:]

    def _read_stdout(self):
        try:
            while True:
                line = self._process.stdout.readline(self.max_response_bytes + 1)
                if not line:
                    raise EOFError("MCP server exited before transport close")
                if len(line) > self.max_response_bytes or not line.endswith(b"\n"):
                    raise ValueError("MCP stdio message exceeds limit or lacks newline")
                message = strict_loads(line.decode("utf-8"))
                if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                    raise ValueError("MCP stdout contains non-protocol data")
                if "method" in message:
                    if "result" in message or "error" in message:
                        raise ValueError("Mixed RPC request/response")
                    if (
                        not isinstance(message["method"], str)
                        or not message["method"]
                        or not isinstance(message.get("params", {}), dict)
                    ):
                        raise ValueError("Malformed MCP server request/notification")
                    if "id" in message:
                        if type(message["id"]) not in {int, str}:
                            raise ValueError("Invalid server request id")

                        reply = {"jsonrpc": "2.0", "id": message["id"]}
                        if message["method"] == "ping":
                            reply["result"] = {}
                        else:
                            reply["error"] = {
                                "code": -32601,
                                "message": "Client capability not granted",
                            }
                        self._send(reply)
                    else:
                        with self._pending_lock:
                            self._notifications.append({"trusted": False, "message": message})
                            if message["method"] == "notifications/tools/list_changed":
                                self._initialized = False
                    continue
                if type(message.get("id")) is not int or ("result" in message) == (
                    "error" in message
                ):
                    raise ValueError("Invalid MCP response envelope")
                with self._pending_lock:
                    request = self._pending.get(message["id"])
                    if request is None or request.ready.is_set():
                        raise ValueError("Unknown or duplicate MCP response id")
                    request.message = message
                    request.ready.set()
        except Exception as error:
            self._fail(error)

    def _read_stderr(self):
        try:
            while True:
                chunk = self._process.stderr.read(4096)
                if not chunk:
                    return
                with self._stderr_lock:
                    self._stderr.extend(chunk)
                    if len(self._stderr) > self.max_response_bytes:
                        del self._stderr[: -self.max_response_bytes]
        except (ValueError, OSError):
            return

    def _rpc(self, method, params=None, *, notification=False):
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if notification:
            self._send(message)
            return None
        with self._pending_lock:
            if self._closed or self._transport_error is not None:
                raise RuntimeError("MCP transport unavailable")
            self._next_id += 1
            identifier = self._next_id
            pending = _PendingRPC()
            self._pending[identifier] = pending
        message["id"] = identifier

        def expired():
            with self._pending_lock:
                if pending.ready.is_set():
                    return
                self._fail(TimeoutError("MCP total request deadline exceeded"))
                self._terminate()

        timer = threading.Timer(self.timeout_seconds, expired)
        timer.daemon = True
        timer.start()
        try:
            self._send(message)
            pending.sent.set()
            pending.ready.wait(self.timeout_seconds + 1.0)
            if pending.error is not None:
                raise pending.error
            if pending.message is None:
                raise TimeoutError("MCP response did not complete")
            if "error" in pending.message:
                raise RuntimeError("Remote MCP error")
            return pending.message["result"]
        except Exception as error:
            if not pending.ready.is_set():
                self._fail(error)
            raise
        finally:
            timer.cancel()
            with self._pending_lock:
                self._pending.pop(identifier, None)

    def _terminate(self):
        if self._process.poll() is None:
            try:
                if os.name == "nt":
                    self._process.kill()
                else:
                    os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def drain_notifications(self):
        with self._pending_lock:
            result = tuple(self._notifications)
            self._notifications.clear()
            return result

    def cancel(self, request_id):

        with self._pending_lock:
            pending = self._pending.get(request_id)
        if pending is None:
            return False
        if not pending.sent.wait(self.timeout_seconds):
            raise TimeoutError("MCP request was not sent before cancellation deadline")
        return super().cancel(request_id)

    def authorizes(self, call):
        return (
            not self._closed
            and self._transport_error is None
            and self._process.poll() is None
            and super().authorizes(call)
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._fail(RuntimeError("MCP client closed"))
        self._terminate()
        self._process.wait(timeout=self.timeout_seconds)
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            stream.close()
        self._read_thread.join(timeout=self.timeout_seconds)
        self._err_thread.join(timeout=self.timeout_seconds)
        if self._read_thread.is_alive() or self._err_thread.is_alive():
            raise RuntimeError("MCP stream reader did not stop; process descendants may hold pipes")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
