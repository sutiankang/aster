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
from aster.training import (
    Trainer,
    ParallelContext,
    ParallelConfig,
    ColumnParallelLinear,
    RowParallelLinear,
)


def objective(model, batch):
    x, y = batch
    values = (model(x) - y).square()
    return LossTerm(values.sum(), torch.tensor(values.numel()), "elements")


def full_batch():
    generator = torch.Generator().manual_seed(412)
    return torch.randn(8, 3, generator=generator), torch.randn(8, 2, generator=generator)


def save_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    try:
        context = ParallelContext(ParallelConfig(tensor_parallel=2, data_parallel=2))
        torch.manual_seed(7)
        full = nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 2))
        model = nn.Sequential(
            ColumnParallelLinear(3, 6, context.tp), nn.Tanh(), RowParallelLinear(6, 2, context.tp)
        )
        with torch.no_grad():
            model[0].weight.copy_(full[0].weight.chunk(2, 0)[context.tp.rank])
            model[0].bias.copy_(full[0].bias.chunk(2)[context.tp.rank])
            model[2].weight.copy_(full[2].weight.chunk(2, 1)[context.tp.rank])
            model[2].bias.copy_(full[2].bias)
        trainer = Trainer(
            model,
            objective,
            lr=0.01,
            zero_stage=3,
            parallel=context,
            max_grad_norm=0.4,
            ema_decay=0.9,
        )
        x, y = full_batch()
        batch = x.chunk(2)[context.dp.rank], y.chunk(2)[context.dp.rank]
        trainer.step([batch])
        trainer.step([batch])
        trainer.save_portable_checkpoint(Path(directory) / "portable.json")
        trainer.step([batch])
        expected = trainer.export_state_dict()
        expected_ema = trainer.export_state_dict(ema=True)
        if rank == 0:
            torch.save({"model": expected, "ema": expected_ema}, Path(directory) / "expected.pt")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_tp2_dp2_zero3_to_single_process_optimizer_reshard(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    directory = Path(tempfile.mkdtemp(prefix="aster-reshard-gloo-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-reshard-gloo-")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        mp.spawn(save_worker, args=(str(directory / "rdzv"), str(tmp_path)), nprocs=4, join=True)
        target = Trainer(
            nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 2)),
            objective,
            lr=0.99,
            max_grad_norm=0.4,
            ema_decay=0.9,
        )
        target.load_portable_checkpoint(tmp_path / "portable.json", seed=88)
        target.step([full_batch()])
        expected = torch.load(tmp_path / "expected.pt", weights_only=True)
        for name, value in target.export_state_dict().items():
            torch.testing.assert_close(value, expected["model"][name], atol=2e-7, rtol=3e-6)
        for name, value in target.export_state_dict(ema=True).items():
            torch.testing.assert_close(value, expected["ema"][name], atol=2e-7, rtol=3e-6)
        assert target.steps == 3
    finally:
        torch.set_num_threads(previous)
        shutil.rmtree(directory)
