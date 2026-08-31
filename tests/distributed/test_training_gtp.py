from copy import deepcopy
from datetime import timedelta
import os
from pathlib import Path
import shutil
import tempfile
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from aster.core.contracts import LossTerm
from aster.training import (
    Trainer,
    ParallelConfig,
    ParallelContext,
    ColumnParallelLinear,
    RowParallelLinear,
    rematerialize_weights,
)
from aster.training.sharding import zero3_units


def objective(model, batch):
    x, y = batch
    error = (model(x) - y).square()
    return LossTerm(error.sum(), torch.tensor(error.numel()), "elements")


def batches(replicas):
    result = []
    for i in range(replicas):
        generator = torch.Generator().manual_seed(811 + i)
        result.append(
            (torch.randn(2 + i, 3, generator=generator), torch.randn(2 + i, 2, generator=generator))
        )
    return result


def worker(rank, rendezvous, path, tp):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    try:
        context = ParallelContext(
            ParallelConfig(tensor_parallel=tp, data_parallel=2 // tp, gtp_remat=2)
        )
        torch.manual_seed(177)
        reference = nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 2))
        model = nn.Sequential(
            ColumnParallelLinear(3, 6, context.tp), nn.Tanh(), RowParallelLinear(6, 2, context.tp)
        )
        with torch.no_grad():
            model[0].weight.copy_(reference[0].weight.chunk(tp, 0)[context.tp.rank])
            model[0].bias.copy_(reference[0].bias.chunk(tp)[context.tp.rank])
            model[2].weight.copy_(reference[2].weight.chunk(tp, 1)[context.tp.rank])
            model[2].bias.copy_(reference[2].bias)
        model = rematerialize_weights(model, context)
        units = zero3_units(model)
        assert sum(sum(p.numel() for p in unit.shards) for unit in units) < sum(
            sum(unit.sizes) for unit in units
        )
        assert all(p.numel() == 0 for unit in units for p in unit.module.parameters())
        trainer = Trainer(
            model, objective, parallel=context, lr=0.01, max_grad_norm=0.4, ema_decay=0.9
        )
        optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
        all_batches = batches(context.dp_gtp.size)
        local = all_batches[context.dp_gtp.rank]
        for _ in range(3):
            optimizer.zero_grad()
            losses = [objective(reference, batch) for batch in all_batches]
            loss = sum(item.numerator for item in losses) / sum(item.denominator for item in losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.4)
            optimizer.step()
            trainer.step([local])
            exported = trainer.export_state_dict(only_rank_zero=False)
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(exported[key], value, atol=2e-6, rtol=2e-5)
        checkpoint = trainer.save_checkpoint(Path(path) / f"gtp-{tp}.json")
        trainer.step([local])
        expected = trainer.export_state_dict(only_rank_zero=False)
        trainer.load_checkpoint(checkpoint)
        trainer.step([local])
        actual = trainer.export_state_dict(only_rank_zero=False)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp", [1, 2])
def test_gtp_remat_independent_of_dp_and_tp_real_storage_update_resume(tp, tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    directory = Path(tempfile.mkdtemp(prefix="aster-gtp-gloo-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-gtp-gloo-")
    try:
        mp.spawn(worker, args=(str(directory / "rdzv"), str(tmp_path), tp), nprocs=4, join=True)
    finally:
        shutil.rmtree(directory)
