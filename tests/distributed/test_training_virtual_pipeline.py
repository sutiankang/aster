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

from aster.core.contracts import LossTerm, LossBundle
from aster.training import (
    Trainer,
    ParallelConfig,
    ParallelContext,
    VirtualPipelineStage,
    PipelineObjective,
    PipelineLossSpec,
)


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.Tanh()) for _ in range(4)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def criterion(output, target):
    return LossBundle(
        (
            LossTerm(
                (output - target).square().sum(),
                torch.tensor(output.numel()),
                "elements",
                "mse",
                0.7,
            ),
            LossTerm(output.abs().sum(), torch.tensor(output.shape[0]), "samples", "l1", 0.3),
        )
    )


def worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
    )
    try:
        context = ParallelContext(ParallelConfig(pipeline_parallel=2, data_parallel=2))
        for zero in (0, 1, 2, 3):
            torch.manual_seed(845)
            reference = Dense()
            chunks = [deepcopy(reference.blocks[context.pp.rank + 2 * i]) for i in range(2)]
            names = {
                f"module.{i}.{key}": f"blocks.{context.pp.rank + 2 * i}.{key}"
                for i, block in enumerate(chunks)
                for key in block.state_dict()
            }
            model = VirtualPipelineStage(chunks, context.pp, parameter_names=names)
            objective = PipelineObjective(
                criterion,
                specs=(
                    PipelineLossSpec("mse", "elements", 0.7),
                    PipelineLossSpec("l1", "samples", 0.3),
                ),
            )
            trainer = Trainer(
                model,
                objective,
                parallel=context,
                zero_stage=zero,
                lr=0.01,
                accumulation_steps=4,
                max_grad_norm=0.3,
            )
            optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
            generator = torch.Generator().manual_seed(991)
            full = [
                (torch.randn(5, 4, generator=generator), torch.randn(5, 4, generator=generator))
                for _ in range(4)
            ]
            local = [
                (x.tensor_split(2)[context.dp.rank], y.tensor_split(2)[context.dp.rank])
                for x, y in full
            ]
            for _ in range(2):
                optimizer.zero_grad()
                terms = [criterion(reference(x), y).terms for x, y in full]
                loss = (
                    0.7 * sum(record[0].numerator for record in terms) / 80
                    + 0.3 * sum(record[1].numerator for record in terms) / 20
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
                optimizer.step()
                result = trainer.step(local)
                assert (
                    result.updated
                    and result.terms["mse"]["denominator"] == 80
                    and result.terms["l1"]["denominator"] == 20
                )
                exported = trainer.export_state_dict(only_rank_zero=False)
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(exported[name], value, atol=3e-6, rtol=3e-5)
            records = trainer.evaluate(local)
            expected = (
                sum(float(criterion(reference(x), y).terms[0].numerator.detach()) for x, y in full)
                / 80
            )
            assert abs(records["mse"]["mean"] - expected) < 1e-6
    finally:
        dist.destroy_process_group()


def test_interleaved_virtual_pipeline_multiple_losses_uneven_dp_counts():
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    directory = Path(tempfile.mkdtemp(prefix="aster-vpp-gloo-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-vpp-gloo-")
    try:
        mp.spawn(worker, args=(4, str(directory / "rdzv")), nprocs=4, join=True)
    finally:
        shutil.rmtree(directory)
