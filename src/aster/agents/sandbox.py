"""Optional Linux isolation using bubblewrap, resource limits, and process-group cleanup."""

from __future__ import annotations
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time

from .permissions import Workspace, PermissionDenied


class SandboxUnavailable(RuntimeError):
    pass


class BubblewrapSandbox:
    isolation_kind = "linux_user_mount_pid_network_namespaces"

    def __init__(
        self,
        workspace,
        *,
        allowed_executables,
        allow_workspace_write=False,
        allow_network=False,
        max_output_bytes=1024 * 1024,
        memory_bytes=2 * 1024**3,
        cpu_seconds=60,
        max_processes=64,
    ):
        if not sys.platform.startswith("linux"):
            raise SandboxUnavailable("Linux bubblewrap isolation is unavailable on this host")
        self.bwrap, self.prlimit = shutil.which("bwrap"), shutil.which("prlimit")
        if not self.bwrap or not self.prlimit:
            raise SandboxUnavailable(
                "Install/authorize bubblewrap and prlimit externally; no unsandboxed fallback"
            )
        if (
            min(max_output_bytes, memory_bytes, cpu_seconds, max_processes) < 1
            or not allowed_executables
        ):
            raise ValueError("Sandbox requires explicit executable and resource limits")
        self.workspace = Workspace(workspace)
        self.allowed_executables = frozenset(
            str(Path(path).resolve(strict=True)) for path in allowed_executables
        )
        if any(
            not Path(path).is_file() or not os.access(path, os.X_OK)
            for path in self.allowed_executables
        ):
            raise ValueError("Allowed executable must be an existing executable file")
        self.allow_workspace_write, self.allow_network = allow_workspace_write, allow_network
        self.max_output_bytes, self.memory_bytes = max_output_bytes, memory_bytes
        self.cpu_seconds, self.max_processes = cpu_seconds, max_processes

    def _command(self, argv, cwd):
        if not argv or not all(isinstance(value, str) and "\x00" not in value for value in argv):
            raise ValueError("Sandbox command needs an explicit argv")
        if not Path(argv[0]).is_absolute():
            raise PermissionDenied("Sandbox executable must be an explicit absolute path")
        executable = str(Path(argv[0]).resolve(strict=True))
        if executable not in self.allowed_executables:
            raise PermissionDenied("Executable is outside the host allowlist")
        actual_cwd = self.workspace.resolve(cwd)
        command = [
            self.prlimit,
            f"--as={self.memory_bytes}",
            f"--cpu={self.cpu_seconds}",
            f"--nproc={self.max_processes}",
            "--nofile=256",
            "--",
            self.bwrap,
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
        ]
        if not self.allow_network:
            command += ["--unshare-net"]
        for directory in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(directory).exists():
                command += ["--ro-bind", directory, directory]

        if Path("/etc/ld.so.cache").is_file():
            command += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
        command += [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind" if self.allow_workspace_write else "--ro-bind",
            str(self.workspace.root),
            "/workspace",
            "--chdir",
            "/workspace/" + actual_cwd.relative_to(self.workspace.root).as_posix(),
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "HOME",
            "/nonexistent",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--",
            executable,
            *argv[1:],
        ]
        return command

    def run(self, *, argv, cwd, timeout_seconds, environment=None):
        if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 300:
            raise ValueError("Invalid sandbox timeout")
        command = self._command(argv, cwd)

        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            start_new_session=True,
        )
        selector = selectors.DefaultSelector()
        outputs = {"stdout": bytearray(), "stderr": bytearray()}
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        deadline, stop_reason, total = time.monotonic() + timeout_seconds, "exited", 0
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stop_reason = "timeout"
                    break
                for key, _ in selector.select(min(0.1, remaining)):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    room = max(0, self.max_output_bytes - total)
                    outputs[key.data].extend(data[:room])
                    total += len(data)
                    if total > self.max_output_bytes:
                        stop_reason = "output_limit"
                        break
                if stop_reason != "exited":
                    break
            if stop_reason != "exited" or process.poll() is None:
                if stop_reason == "exited":
                    try:
                        process.wait(timeout=max(0.01, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        stop_reason = "timeout"
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            selector.close()
            process.stdout.close()
            process.stderr.close()
        return {
            "exit_code": process.returncode,
            "stop_reason": stop_reason,
            "stdout": outputs["stdout"].decode("utf-8", errors="replace"),
            "stderr": outputs["stderr"].decode("utf-8", errors="replace"),
            "isolation": self.isolation_kind,
            "network_enabled": self.allow_network,
            "workspace_writable": self.allow_workspace_write,
        }
