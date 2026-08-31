"""Explicit native/torchrun launch arguments and process-group lifecycle."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
from .parallel import ParallelConfig, ParallelContext


@dataclass(frozen=True)
class DistributedEnvironment:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    master_addr: str
    master_port: int

    @classmethod
    def from_mapping(cls, env: Mapping[str, str]):
        names = (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        )
        if not any(name in env for name in names):
            return cls(0, 1, 0, 1, "127.0.0.1", 29500)
        missing = [name for name in names if name not in env]
        if missing:
            raise ValueError(f"分布式环境不完整，缺少 {missing}")
        try:
            result = cls(
                *(int(env[name]) for name in names[:4]), env["MASTER_ADDR"], int(env["MASTER_PORT"])
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("rank/world/port 必须是整数") from exc
        if not (
            0 <= result.rank < result.world_size
            and 0 <= result.local_rank < result.local_world_size <= result.world_size
        ):
            raise ValueError("rank/world/local_rank 超出合法范围")
        if result.world_size % result.local_world_size:
            raise ValueError("当前同构 launcher 要求 WORLD_SIZE 整除 LOCAL_WORLD_SIZE")
        if result.rank % result.local_world_size != result.local_rank:
            raise ValueError("rank 与 LOCAL_RANK 不符合连续同构节点布局")
        if (
            not result.master_addr
            or any(char.isspace() for char in result.master_addr)
            or not 0 < result.master_port < 65536
        ):
            raise ValueError("非法 master 地址/端口")
        return result


@dataclass(frozen=True)
class LaunchConfig:
    nproc_per_node: int = 1
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    backend: str = "gloo"
    timeout_seconds: int = 180
    launcher: str = "torchrun"
    store_backend: str = "default"

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1
            for value in (self.nproc_per_node, self.nnodes, self.timeout_seconds)
        ):
            raise ValueError("进程/节点/timeout 必须为正整数")
        if type(self.node_rank) is not int or not 0 <= self.node_rank < self.nnodes:
            raise ValueError("node_rank 越界")
        if self.backend not in {"gloo", "nccl"}:
            raise ValueError("backend 只接受明确 gloo/nccl")
        if self.launcher not in {"torchrun", "native"}:
            raise ValueError("launcher 仅 torchrun/native，不静默切换")
        if self.store_backend not in {"default", "legacy_tcp"} or (
            self.launcher != "torchrun" and self.store_backend != "default"
        ):
            raise ValueError("Explicit legacy_tcp store is supported by the torchrun launcher only")
        if (
            not self.master_addr
            or any(char.isspace() for char in self.master_addr)
            or type(self.master_port) is not int
            or not 0 < self.master_port < 65536
        ):
            raise ValueError("非法 master 地址/端口")
        if self.nnodes > 1 and self.master_addr in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("多机 master 不能用 loopback 地址")

    def command(self, script: str | Path, arguments: Sequence[str] = ()) -> list[str]:
        path = Path(script).absolute()
        if not path.is_file() or path.suffix != ".py":
            raise ValueError("训练入口必须为已存在的本地 Python 文件")
        if any(not isinstance(arg, str) or "\x00" in arg for arg in arguments):
            raise ValueError("训练参数必须为无 NUL 的字符串数组")
        if self.launcher == "native":
            return [sys.executable, str(path), *arguments]
        entry = (
            "aster.training.torchrun_compat"
            if self.store_backend == "legacy_tcp"
            else "torch.distributed.run"
        )
        rendezvous = [
            f"--master-addr={self.master_addr}",
            f"--master-port={self.master_port}",
            f"--rdzv-conf=timeout={self.timeout_seconds}",
        ]
        if self.store_backend == "legacy_tcp":
            address = f"[{self.master_addr}]" if ":" in self.master_addr else self.master_addr
            rendezvous = [
                "--rdzv-backend=aster_static_tcp",
                f"--rdzv-endpoint={address}:{self.master_port}",
                f"--rdzv-conf=rank={self.node_rank},timeout={self.timeout_seconds}",
                "--rdzv-id=aster",
            ]
        return [
            sys.executable,
            "-m",
            entry,
            f"--nnodes={self.nnodes}",
            f"--nproc-per-node={self.nproc_per_node}",
            f"--node-rank={self.node_rank}",
            *rendezvous,
            "--max-restarts=0",
            str(path),
            *arguments,
        ]


def launch(
    config: LaunchConfig,
    script: str | Path,
    arguments: Sequence[str] = (),
    *,
    execute=False,
    capture_output=False,
):
    command = config.command(script, arguments)
    if not execute:
        return command
    if config.backend == "nccl" and (
        not torch.cuda.is_available() or torch.cuda.device_count() < config.nproc_per_node
    ):
        raise RuntimeError("当前 torch/CUDA 或本地GPU数量不满足 NCCL 启动；不自动降为 Gloo")
    environment = dict(os.environ)
    environment["ASTER_DISTRIBUTED_BACKEND"] = config.backend
    environment["ASTER_DISTRIBUTED_TIMEOUT"] = str(config.timeout_seconds)
    environment["ASTER_DISTRIBUTED_STORE_BACKEND"] = config.store_backend

    if os.name == "nt":
        environment.setdefault("USE_LIBUV", "0")
    if config.store_backend == "legacy_tcp":
        environment["USE_LIBUV"] = "0"
    if config.launcher == "native":
        return _native_launch(config, command, environment, capture_output)
    return subprocess.run(
        command, env=environment, shell=False, check=True, text=True, capture_output=capture_output
    )


def _native_launch(config, command, environment, capture_output):

    processes = []
    output, errors = [], []
    try:
        for local_rank in range(config.nproc_per_node):
            env = {
                **environment,
                "RANK": str(config.node_rank * config.nproc_per_node + local_rank),
                "WORLD_SIZE": str(config.nnodes * config.nproc_per_node),
                "LOCAL_RANK": str(local_rank),
                "LOCAL_WORLD_SIZE": str(config.nproc_per_node),
                "MASTER_ADDR": config.master_addr,
                "MASTER_PORT": str(config.master_port),
            }
            processes.append(
                subprocess.Popen(
                    command,
                    env=env,
                    shell=False,
                    text=True,
                    stdout=subprocess.PIPE if capture_output else None,
                    stderr=subprocess.PIPE if capture_output else None,
                )
            )
        with ThreadPoolExecutor(max_workers=len(processes)) as pool:
            futures = {pool.submit(process.communicate): process for process in processes}
            for future in as_completed(futures):
                process = futures[future]
                stdout, stderr = future.result()
                output.append(stdout or "")
                errors.append(stderr or "")
                if process.returncode:
                    for peer in processes:
                        if peer.poll() is None:
                            peer.terminate()
                    raise subprocess.CalledProcessError(
                        process.returncode, command, "".join(output), "".join(errors)
                    )
        return subprocess.CompletedProcess(
            command,
            0,
            "".join(output) if capture_output else None,
            "".join(errors) if capture_output else None,
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@contextmanager
def distributed_session(
    parallel: ParallelConfig | None = None,
    *,
    backend: str | None = None,
    timeout_seconds: int | None = None,
):
    env = DistributedEnvironment.from_mapping(os.environ)
    backend = backend or os.environ.get("ASTER_DISTRIBUTED_BACKEND", "gloo")
    timeout_seconds = (
        timeout_seconds
        if timeout_seconds is not None
        else int(os.environ.get("ASTER_DISTRIBUTED_TIMEOUT", "180"))
    )
    if backend not in {"gloo", "nccl"} or type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("非法 backend/timeout")
    if parallel is not None and parallel.world_size != env.world_size:
        raise ValueError("模型并行网格与 torchrun WORLD_SIZE 不一致")
    if backend == "nccl":
        if not torch.cuda.is_available() or env.local_world_size > torch.cuda.device_count():
            raise RuntimeError("NCCL 需要可用CUDA及足够GPU，不接受CPU伪验收")
        torch.cuda.set_device(env.local_rank)
    owned = False
    try:
        if dist.is_initialized():
            if (dist.get_rank(), dist.get_world_size(), dist.get_backend()) != (
                env.rank,
                env.world_size,
                backend,
            ):
                raise ValueError("已有进程组与 launcher 环境/backend 不一致")
        elif env.world_size > 1:
            try:
                timeout = timedelta(seconds=timeout_seconds)
                store_backend = os.environ.get("ASTER_DISTRIBUTED_STORE_BACKEND", "default")
                if store_backend == "legacy_tcp":
                    if os.environ.get("TORCHELASTIC_USE_AGENT_STORE") != "True":
                        raise ValueError(
                            "Legacy agent store requires the explicit torchrun compatibility entrypoint"
                        )
                    raw_store = dist.TCPStore(
                        env.master_addr,
                        env.master_port,
                        env.world_size,
                        False,
                        timeout,
                        use_libuv=False,
                    )
                    run_id = os.environ["TORCHELASTIC_RUN_ID"]
                    attempt = os.environ["TORCHELASTIC_RESTART_COUNT"]
                    store = dist.PrefixStore(f"aster-worker/{run_id}/{attempt}", raw_store)
                    dist.init_process_group(
                        backend,
                        store=store,
                        rank=env.rank,
                        world_size=env.world_size,
                        timeout=timeout,
                    )
                elif store_backend == "default":
                    dist.init_process_group(
                        backend,
                        init_method="env://",
                        rank=env.rank,
                        world_size=env.world_size,
                        timeout=timeout,
                    )
                else:
                    raise ValueError("Unknown distributed store backend")
                owned = True
            except Exception as exc:
                raise RuntimeError(
                    f"rank {env.rank}/{env.world_size} 初始化 {backend} 失败（timeout={timeout_seconds}s，master={env.master_addr}:{env.master_port}）；检查每节点启动数、端口与网卡"
                ) from exc
        yield ParallelContext(parallel)
    finally:
        if owned and dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description="显式启动 Aster native 分布式训练")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--launcher", choices=("torchrun", "native"), default="torchrun")
    parser.add_argument("--store-backend", choices=("default", "legacy_tcp"), default="default")
    parser.add_argument("script")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    config = LaunchConfig(
        args.nproc_per_node,
        args.nnodes,
        args.node_rank,
        args.master_addr,
        args.master_port,
        args.backend,
        args.timeout_seconds,
        args.launcher,
        args.store_backend,
    )
    result = launch(config, args.script, args.arguments, execute=args.execute)
    if not args.execute:
        print(result)


if __name__ == "__main__":
    main()
