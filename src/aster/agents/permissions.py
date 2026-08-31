"""Host-issued capabilities and scoped approval; model text cannot authorize actions."""

from __future__ import annotations
from dataclasses import dataclass
import math
import os
from pathlib import Path
import secrets
import stat
import time

from .events import canonical_json, digest, strict_loads


class PermissionDenied(RuntimeError):
    pass


class Workspace:
    def __init__(self, root):
        raw = Path(root).absolute()
        self._reject_links(raw)
        self.root = raw.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace must be an existing directory")

    @staticmethod
    def _reject_links(path):
        for entry in (path, *path.parents):
            if not entry.exists() and not entry.is_symlink():
                continue
            info = entry.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
                stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
            ):
                raise PermissionDenied("Symlink/junction/reparse paths are forbidden")

    def resolve(self, value, *, must_exist=True):
        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
            or value.startswith(("\\\\", "//"))
        ):
            raise PermissionDenied("Invalid workspace path")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        self._reject_links(candidate)
        resolved = candidate.resolve(strict=must_exist)
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise PermissionDenied("Path escapes the workspace") from None
        return resolved


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    implementation_digest: str
    effect: str
    description: str

    def __post_init__(self):
        if self.effect not in {"read", "workspace_write", "isolated_process", "external"}:
            raise ValueError("Unknown tool effect")
        if not self.name or not self.version or not self.implementation_digest:
            raise ValueError("Tool implementation identity is required")


@dataclass(frozen=True)
class PreparedCall:
    call_id: str
    tool: ToolSpec
    arguments_json: str
    workspace: str
    thread_id: str
    turn_id: str
    binding: str

    @property
    def arguments(self):
        return strict_loads(self.arguments_json)


@dataclass(frozen=True)
class Approval:
    token: str
    binding: str
    scope: str
    thread_id: str
    turn_id: str | None
    expires_at: float


class PermissionBroker:
    def __init__(
        self, workspace, *, allow_read=True, isolation_backend=None, external_authorizer=None
    ):
        self.workspace = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        self.allow_read = allow_read
        self.isolation_backend = isolation_backend
        self.external_authorizer = external_authorizer
        self._approvals = {}
        self._used = set()
        self._consumed_calls = set()

    def prepare(self, tool, arguments, *, thread_id, turn_id):
        if not isinstance(arguments, dict) or not thread_id or not turn_id:
            raise ValueError("Tool arguments and thread/turn identity are mandatory")
        encoded = canonical_json(arguments)
        if len(encoded.encode("utf-8")) > 256 * 1024:
            raise ValueError("Tool arguments exceed policy limit")
        call_id = secrets.token_hex(16)
        binding = digest(
            {
                "call_id": call_id,
                "tool": tool.__dict__,
                "arguments": arguments,
                "cwd": str(self.workspace.root),
                "thread_id": thread_id,
                "turn_id": turn_id,
            }
        )
        return PreparedCall(
            call_id, tool, encoded, str(self.workspace.root), thread_id, turn_id, binding
        )

    def approve(self, call, *, scope="turn", ttl_seconds=60.0):
        """Trusted-host/UI operation; models cannot grant themselves approval through JSON."""
        if scope not in {"turn", "session"} or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("Invalid approval scope or lifetime")
        if call.tool.effect == "isolated_process" and self.isolation_backend is None:
            raise PermissionDenied("Dangerous tool requires an actual isolation backend")
        if call.tool.effect == "external" and (
            self.external_authorizer is None or not self.external_authorizer(call)
        ):
            raise PermissionDenied(
                "External tool needs a current endpoint/tool-specific host grant"
            )
        approval = Approval(
            secrets.token_hex(32),
            call.binding,
            scope,
            call.thread_id,
            call.turn_id if scope == "turn" else None,
            time.monotonic() + ttl_seconds,
        )
        self._approvals[approval.token] = approval
        return approval

    def configured_approval(self, call):
        if call.tool.effect == "read" and self.allow_read:
            return self.approve(call)
        return None

    def revoke(self, approval):
        self._approvals.pop(approval.token, None)

    def consume(self, call, approval, registered_spec, *, thread_id, turn_id):
        recomputed = digest(
            {
                "call_id": call.call_id,
                "tool": call.tool.__dict__,
                "arguments": call.arguments,
                "cwd": call.workspace,
                "thread_id": call.thread_id,
                "turn_id": call.turn_id,
            }
        )
        if recomputed != call.binding or registered_spec != call.tool:
            raise PermissionDenied("Prepared content or tool implementation changed")
        if (
            str(self.workspace.root) != call.workspace
            or call.thread_id != thread_id
            or call.turn_id != turn_id
        ):
            raise PermissionDenied("Execution context differs from approved context")
        known = self._approvals.get(getattr(approval, "token", None))
        if (
            known != approval
            or known is None
            or known.binding != call.binding
            or known.thread_id != thread_id
        ):
            raise PermissionDenied("Approval is missing or does not bind this exact call")
        if known.scope == "turn" and known.turn_id != turn_id:
            raise PermissionDenied("Approval belongs to a different turn")
        if (
            known.expires_at <= time.monotonic()
            or known.token in self._used
            or call.call_id in self._consumed_calls
        ):
            raise PermissionDenied("Approval expired or has already been consumed")
        if call.tool.effect == "isolated_process" and self.isolation_backend is None:
            raise PermissionDenied("Isolation unavailable")
        if call.tool.effect == "external" and (
            self.external_authorizer is None or not self.external_authorizer(call)
        ):
            raise PermissionDenied("External grant expired or does not match this tool")
        self._used.add(known.token)
        self._consumed_calls.add(call.call_id)
        self._approvals.pop(known.token, None)
