from copy import deepcopy
from datetime import timedelta
import os
from pathlib import Path
import shutil
import tempfile
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from aster.core.contracts import LossTerm
from aster.training import Trainer, ParallelContext, ParallelConfig


class Tied(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.embedding = nn.Embedding(7, 4, device=device)
        self.head = nn.Linear(4, 7, bias=False, device=device)
        self.head.weight = self.embedding.weight

    def forward(self, tokens):
        return self.head(self.embedding(tokens))


def objective(model, batch):
    x, y = batch
    return LossTerm(
        nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum"),
        torch.tensor(y.numel()),
        "tokens",
    )


def worker(rank, rendezvous, path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        context = ParallelContext(ParallelConfig(data_parallel=2))
        calls = []

        def initialize(name, shape, dtype, offset, count, device):
            calls.append((offset, count))
            return torch.arange(offset, offset + count, dtype=dtype, device=device) * 0.01

        trainer = Trainer(
            Tied("meta"),
            objective,
            zero_stage=3,
            parallel=context,
            lr=0.02,
            offload_parameters="cpu",
            sharded_initializer=initialize,
        )
        assert calls == [(rank * 14, 14)] and len(trainer.roles["model"].parameters) == 1
        shard = trainer.roles["model"].parameters[0]
        assert shard.numel() == 14 and shard.device.type == "cpu"
        reference = Tied()
        with torch.no_grad():
            reference.embedding.weight.copy_(torch.arange(28).reshape(7, 4) * 0.01)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=0.02)
        x = torch.tensor([[0, 1, 2], [6, 5, 3], [4, 1, 6]])
        y = (x + 1) % 7
        local = (x[:2], y[:2]) if rank == 0 else (x[2:], y[2:])
        for _ in range(2):
            optimizer.zero_grad()
            objective(reference, (x, y)).mean.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            optimizer.step()
            result = trainer.step([local])
            assert result.updated
            torch.testing.assert_close(trainer.model(x), reference(x), atol=2e-6, rtol=2e-5)
        saved = trainer.save_checkpoint(Path(path) / "meta.json")
        trainer.step([local])
        expected = trainer.model(x).detach().clone()
        trainer.load_checkpoint(saved)
        trainer.step([local])
        torch.testing.assert_close(trainer.model(x), expected, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_real_dp_meta_init_shared_shard_cpu_storage_and_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    directory = Path(tempfile.mkdtemp(prefix="aster-meta-gloo-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-meta-gloo-")
    try:
        mp.spawn(worker, args=(str(directory / "rdzv"), str(tmp_path)), nprocs=2, join=True)
    finally:
        shutil.rmtree(directory)
