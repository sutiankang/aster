"""Explicitly authorized MCP 2025-06-18 HTTP transport and tool calls."""

from __future__ import annotations
import http.client
from contextlib import nullcontext
import ipaddress
import math
import re
import socket
import threading
import time
from urllib.parse import urlsplit

from .events import canonical_json, digest, strict_loads
from .permissions import PermissionDenied, ToolSpec


def validate_schema(value, schema, depth=0):

    if depth > 16 or not isinstance(schema, dict):
        raise ValueError("Invalid/deep tool schema")
    supported = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "description",
        "title",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    }
    if set(schema) - supported:
        raise ValueError("Unsupported MCP input schema keyword")
    kind = schema.get("type")
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: type(value) is int,
        "number": lambda: type(value) in {int, float} and math.isfinite(value),
        "boolean": lambda: type(value) is bool,
        "null": lambda: value is None,
    }
    if kind not in checks or not checks[kind]():
        raise ValueError("MCP argument type mismatch")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("MCP enum mismatch")
    if kind == "object":
        properties = schema.get("properties", {})
        if not set(schema.get("required", ())) <= value.keys():
            raise ValueError("Missing MCP arguments")
        if schema.get("additionalProperties", True) is False and set(value) - properties.keys():
            raise ValueError("Unexpected MCP arguments")
        if type(schema.get("additionalProperties", True)) is not bool:
            raise ValueError("Schema-valued additionalProperties is not supported")
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key], depth + 1)
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10000):
            raise ValueError("Array argument exceeds schema bounds")
        for item in value:
            validate_schema(item, schema["items"], depth + 1)
    if kind == "string" and not schema.get("minLength", 0) <= len(value) <= schema.get(
        "maxLength", 100000
    ):
        raise ValueError("String argument exceeds schema bounds")
    if kind in {"integer", "number"} and not schema.get(
        "minimum", -math.inf
    ) <= value <= schema.get("maximum", math.inf):
        raise ValueError("Number argument exceeds schema bounds")


class MCPClient:
    protocol_version = "2025-06-18"

    def __init__(
        self,
        endpoint,
        *,
        server_id,
        allowed_tools,
        grant_ttl_seconds=300.0,
        timeout_seconds=10.0,
        max_response_bytes=1024 * 1024,
        max_calls=100,
    ):
        location = urlsplit(endpoint)
        try:
            local = ipaddress.ip_address(location.hostname or "").is_loopback
        except ValueError:
            local = location.hostname == "localhost"
        if (
            location.scheme != "http"
            or not local
            or not location.port
            or location.username
            or location.password
            or location.query
            or location.fragment
        ):
            raise PermissionDenied(
                "MCP requires an explicitly granted loopback HTTP endpoint and port"
            )
        self.host, self.port, self.path = location.hostname, location.port, location.path or "/"
        self._configure(
            endpoint,
            server_id=server_id,
            allowed_tools=allowed_tools,
            grant_ttl_seconds=grant_ttl_seconds,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
            max_calls=max_calls,
        )

    def _configure(
        self,
        endpoint,
        *,
        server_id,
        allowed_tools,
        grant_ttl_seconds=300.0,
        timeout_seconds=10.0,
        max_response_bytes=1024 * 1024,
        max_calls=100,
    ):
        if (
            not isinstance(server_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", server_id)
            or isinstance(allowed_tools, str)
            or not allowed_tools
            or any(
                not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name)
                for name in allowed_tools
            )
        ):
            raise ValueError("MCP server identity and explicit tool allowlist are required")
        if not all(
            math.isfinite(value) and value > 0 for value in (grant_ttl_seconds, timeout_seconds)
        ) or any(type(value) is not int or value < 1 for value in (max_response_bytes, max_calls)):
            raise ValueError("MCP limits must be positive")
        self.endpoint, self.server_id = endpoint, server_id
        self.allowed_tools = frozenset(allowed_tools)
        self.expires_at = time.monotonic() + grant_ttl_seconds
        self.timeout_seconds, self.max_response_bytes, self.max_calls = (
            timeout_seconds,
            max_response_bytes,
            max_calls,
        )
        self._session = None
        self._next_id = 0
        self._lock = threading.RLock()
        self._calls = 0
        self._tools, self._specs = {}, {}
        self.server_info = None
        self.server_capabilities = {}
        self.generation = 0
        self._initialized = False

    def _rpc(self, method, params=None, *, notification=False):

        with nullcontext() if notification else self._lock:
            if not notification:
                self._next_id += 1
            identifier = self._next_id
            request = {"jsonrpc": "2.0", "method": method, "params": params or {}}
            if not notification:
                request["id"] = identifier
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            if self._initialized:
                headers["MCP-Protocol-Version"] = self.protocol_version
            if self._session:
                headers["Mcp-Session-Id"] = self._session
            connection = http.client.HTTPConnection(
                self.host, self.port, timeout=self.timeout_seconds
            )
            timer = None
            try:
                deadline = time.monotonic() + self.timeout_seconds
                connection.connect()
                transport_socket = connection.sock

                def abort_deadline():
                    try:
                        transport_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

                timer = threading.Timer(max(0.0, deadline - time.monotonic()), abort_deadline)
                timer.daemon = True
                timer.start()
                connection.request(
                    "POST", self.path, body=canonical_json(request).encode("utf-8"), headers=headers
                )
                response = connection.getresponse()
                if notification:
                    if response.status != 202:
                        raise RuntimeError("MCP notification was not acknowledged")
                    return None
                if response.status != 200:
                    raise RuntimeError("MCP transport/session failed; no automatic retry")
                session = response.getheader("Mcp-Session-Id")
                if session is not None:
                    if (
                        any(not 0x21 <= ord(char) <= 0x7E for char in session)
                        or len(session) > 1024
                    ):
                        raise ValueError("Invalid MCP session identifier")
                    if method != "initialize" and session != self._session:
                        raise ValueError("MCP session changed unexpectedly")
                    self._session = session
                content_type = response.getheader("Content-Type", "").split(";", 1)[0]
                if content_type == "application/json":
                    data = response.read(self.max_response_bytes + 1)
                    if len(data) > self.max_response_bytes:
                        raise ValueError("MCP response exceeds bound")
                    message = strict_loads(data.decode("utf-8"))
                elif content_type == "text/event-stream":
                    size, data_lines, message = 0, [], None
                    while True:
                        line = response.readline(self.max_response_bytes + 1)
                        if not line:
                            break
                        size += len(line)
                        if size > self.max_response_bytes:
                            raise ValueError("MCP SSE response exceeds bound")
                        if line.startswith(b"data:"):
                            data_lines.append(line[5:].lstrip().rstrip(b"\r\n"))
                        elif line in {b"\n", b"\r\n"} and data_lines:
                            candidate = strict_loads(b"\n".join(data_lines).decode("utf-8"))
                            data_lines = []
                            if candidate.get("id") == identifier:
                                message = candidate
                                break
                            if "id" in candidate:
                                raise ValueError("Server-initiated MCP requests are not enabled")
                    if message is None:
                        raise ValueError("SSE ended before the requested MCP result")
                else:
                    raise ValueError("Unsupported MCP response content type")
                if (
                    not isinstance(message, dict)
                    or message.get("jsonrpc") != "2.0"
                    or message.get("id") != identifier
                    or ("result" in message) == ("error" in message)
                ):
                    raise ValueError("Invalid MCP JSON-RPC envelope")
                if "error" in message:
                    raise RuntimeError("Remote MCP error")
                return message["result"]
            finally:
                if timer is not None:
                    timer.cancel()
                connection.close()

    def initialize(self):
        if self._initialized:
            raise ValueError("MCP client is already initialized")
        self._tools, self._specs = {}, {}
        self.generation += 1
        try:
            return self._initialize_handshake()
        except BaseException:
            self._initialized = False
            self._tools, self._specs = {}, {}
            self.server_info, self.server_capabilities = None, {}
            raise

    def _initialize_handshake(self):
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "aster-native", "version": "1"},
            },
        )
        if result.get("protocolVersion") != self.protocol_version or "tools" not in result.get(
            "capabilities", {}
        ):
            raise ValueError("MCP server lacks the explicitly supported protocol/tools capability")
        info = result.get("serverInfo")
        if not isinstance(info, dict) or not info.get("name") or not info.get("version"):
            raise ValueError("MCP server version identity is missing")
        self.server_info = info
        self.server_capabilities = result["capabilities"]
        self._initialized = True
        self._rpc("notifications/initialized", notification=True)
        cursor, seen = None, set()
        for _ in range(16):
            result = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            for tool in result.get("tools", []):
                if tool["name"] in self.allowed_tools:
                    if tool["name"] in self._tools:
                        raise ValueError("Duplicate MCP tool declaration")
                    self._tools[tool["name"]] = tool
            cursor = result.get("nextCursor")
            if cursor is None:
                break
            if cursor in seen:
                raise ValueError("MCP tools pagination cycle")
            seen.add(cursor)
        else:
            raise ValueError("MCP tools pagination exceeds bound")
        if set(self._tools) != set(self.allowed_tools):
            raise ValueError("An explicitly granted MCP tool is unavailable")
        for name, tool in self._tools.items():
            contract = {
                "endpoint": self.endpoint,
                "server": self.server_info,
                "protocol": self.protocol_version,
                "tool": tool,
            }
            full_name = "mcp." + self.server_id + "." + name
            self._specs[name] = ToolSpec(
                full_name,
                str(self.server_info["version"]),
                digest(contract),
                "external",
                "远端工具（元数据不授予权限）: " + str(tool.get("description", ""))[:1000],
            )
        return tuple(self._specs.values())

    def authorizes(self, call):
        return (
            self._initialized
            and time.monotonic() < self.expires_at
            and self._calls < self.max_calls
            and any(call.tool == spec for spec in self._specs.values())
        )

    def call_tool(self, name, arguments):
        with self._lock:
            if (
                not self._initialized
                or name not in self.allowed_tools
                or time.monotonic() >= self.expires_at
                or self._calls >= self.max_calls
            ):
                raise PermissionDenied("MCP endpoint/tool grant unavailable")
            validate_schema(arguments, self._tools[name]["inputSchema"])
            self._calls += 1
            result = self._rpc("tools/call", {"name": name, "arguments": arguments})
            if not isinstance(result, dict) or not isinstance(result.get("content"), list):
                raise ValueError("MCP tool result lacks typed content")
            return result

    def register_tools(self, executor):
        if not self._initialized:
            raise ValueError("Initialize and pin remote tool contracts before registration")
        for name, spec in self._specs.items():
            executor.register(
                spec, lambda arguments, tool_name=name: self.call_tool(tool_name, arguments)
            )

    def cancel(self, request_id):
        return self._rpc(
            "notifications/cancelled",
            {"requestId": request_id, "reason": "host cancellation"},
            notification=True,
        )

    def close(self):
        if self._session:
            connection = http.client.HTTPConnection(
                self.host, self.port, timeout=self.timeout_seconds
            )
            try:
                connection.request(
                    "DELETE",
                    self.path,
                    headers={
                        "Mcp-Session-Id": self._session,
                        "MCP-Protocol-Version": self.protocol_version,
                    },
                )
                response = connection.getresponse()
                if response.status not in {200, 202, 204, 404, 405}:
                    raise RuntimeError("MCP session close failed")
            finally:
                connection.close()
        self._initialized = False
        self._session = None
