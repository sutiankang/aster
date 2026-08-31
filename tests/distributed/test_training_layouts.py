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
from torch.nn import functional as F

from aster.core.contracts import LossTerm
from aster.training import (
    Trainer,
    ParallelConfig,
    ParallelContext,
    Group,
    PipelineStage,
    PipelineObjective,
    SequenceParallelMLP,
    ExpertParallelMLP,
    context_parallel_attention,
)
from aster.training import ring_context_parallel_attention
from aster.training import PipelineLossSpec


def criterion(output, target):
    error = (output - target).square()
    return LossTerm(error.sum(), torch.tensor(error.numel()), "elements")


def objective(model, batch):
    return criterion(model(batch[0]), batch[1])


class CPBlock(nn.Module):
    def __init__(self, group, ring=False):
        super().__init__()
        self.group, self.ring = group, ring
        self.q = nn.Linear(4, 4)
        self.k = nn.Linear(4, 4)
        self.v = nn.Linear(4, 4)
        self.out = nn.Linear(4, 4)

    def forward(self, x):
        q, k, v = [
            layer(x).view(x.shape[0], x.shape[1], 2, 2).transpose(1, 2)
            for layer in (self.q, self.k, self.v)
        ]
        result = (ring_context_parallel_attention if self.ring else context_parallel_attention)(
            q, k, v, self.group
        )
        return self.out(result.transpose(1, 2).reshape_as(x))


class DenseExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(3, 4, bias=False)
        self.experts = nn.ModuleDict(
            {str(i): nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 3)) for i in range(4)}
        )

    def forward(self, x):
        probs = self.router(x).float().softmax(-1)
        weights, indices = probs.topk(2, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        output = torch.zeros_like(x)
        for key, expert in self.experts.items():
            token, slot = (indices == int(key)).nonzero(as_tuple=True)
            output = output.index_add(0, token, expert(x[token]) * weights[token, slot, None])
        return output


def worker(rank, world, rendezvous, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
    )
    config = {
        "pp": ParallelConfig(pipeline_parallel=2, data_parallel=2),
        "pp_1f1b": ParallelConfig(pipeline_parallel=2, data_parallel=2),
        "sp": ParallelConfig(tensor_parallel=2),
        "cp": ParallelConfig(context_parallel=2),
        "ep": ParallelConfig(data_parallel=2),
    }[mode]
    try:
        context = ParallelContext(config)
        for stage in (0, 1, 2, 3):
            torch.manual_seed(74)
            if mode.startswith("pp"):
                reference = nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 2))
                local = deepcopy(
                    nn.Sequential(reference[0], reference[1])
                    if context.pp.rank == 0
                    else reference[2]
                )
                model = PipelineStage(
                    local, context.pp, schedule="1f1b" if mode == "pp_1f1b" else "gpipe"
                )
                generation = torch.Generator().manual_seed(150)
                full_x, full_y = (
                    torch.randn(6, 3, generator=generation),
                    torch.randn(6, 2, generator=generation),
                )
                local_x, local_y = (
                    full_x.chunk(2)[context.dp.rank],
                    full_y.chunk(2)[context.dp.rank],
                )
                obj, accumulation = (
                    PipelineObjective(
                        criterion,
                        specs=(PipelineLossSpec("loss", "elements"),)
                        if mode == "pp_1f1b"
                        else None,
                    ),
                    4 if mode == "pp_1f1b" else 2,
                )
            elif mode == "sp":
                reference = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Linear(6, 3))
                model = SequenceParallelMLP(3, 6, context.tp)
                with torch.no_grad():
                    model.up.weight.copy_(reference[0].weight.chunk(2, 0)[rank])
                    model.up.bias.copy_(reference[0].bias.chunk(2)[rank])
                    model.down.weight.copy_(reference[2].weight.chunk(2, 1)[rank])
                    model.bias.copy_(reference[2].bias)
                generation = torch.Generator().manual_seed(150)
                full_x, full_y = (
                    torch.randn(6, 3, generator=generation),
                    torch.randn(6, 3, generator=generation),
                )
                local_x, local_y = full_x.chunk(2)[rank], full_y.chunk(2)[rank]
                obj, accumulation = objective, 1
            elif mode == "cp":
                reference = CPBlock(Group())
                model = CPBlock(context.cp, ring=True)
                model.load_state_dict(reference.state_dict())
                generation = torch.Generator().manual_seed(150)
                full_x, full_y = (
                    torch.randn(2, 6, 4, generator=generation),
                    torch.randn(2, 6, 4, generator=generation),
                )
                local_x, local_y = full_x.chunk(2, 1)[rank], full_y.chunk(2, 1)[rank]
                obj, accumulation = objective, 1
            else:
                reference = DenseExperts()
                model = ExpertParallelMLP(3, 5, 4, context.dp, top_k=2)
                model.router.load_state_dict(reference.router.state_dict())
                for key, expert in model.experts.items():
                    expert.load_state_dict(reference.experts[key].state_dict())
                generation = torch.Generator().manual_seed(150)
                full_x, full_y = (
                    torch.randn(6, 3, generator=generation),
                    torch.randn(6, 3, generator=generation),
                )
                local_x, local_y = full_x.chunk(2)[rank], full_y.chunk(2)[rank]
                obj, accumulation = objective, 1
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
            trainer = Trainer(
                model,
                obj,
                lr=0.01,
                parallel=context,
                zero_stage=stage,
                max_grad_norm=0.3,
                accumulation_steps=accumulation,
            )
            if mode == "sp":
                trainer.register_loss_group("loss", context.tp)
            for _ in range(2):
                reference_optimizer.zero_grad(set_to_none=True)
                expected_loss = criterion(reference(full_x), full_y).mean
                expected_loss.backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
                reference_optimizer.step()
                result = trainer.step([(local_x, local_y)] * accumulation)
                assert result.updated
                with torch.no_grad():
                    actual = trainer.model(local_x)
                    expected = reference(full_x)
                    if mode.startswith("pp"):
                        if context.pp.rank == 1:
                            torch.testing.assert_close(
                                actual, expected.chunk(2)[context.dp.rank], atol=4e-6, rtol=4e-5
                            )
                        trainer.model.flush_pending()
                        if mode == "pp_1f1b":
                            assert trainer.model.peak_live_graphs <= 2 - context.pp.rank
                    elif mode == "cp":
                        torch.testing.assert_close(
                            actual, expected.chunk(2, 1)[rank], atol=4e-6, rtol=4e-5
                        )
                    else:
                        torch.testing.assert_close(
                            actual, expected.chunk(2)[rank], atol=4e-6, rtol=4e-5
                        )
                dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode,world", [("sp", 2), ("cp", 2), ("ep", 2), ("pp", 4), ("pp_1f1b", 4)])
def test_native_layouts_update_matches_dense(mode, world):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    directory = Path(tempfile.mkdtemp(prefix="aster-layout-gloo-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-layout-gloo-")
    try:
        mp.spawn(worker, args=(world, str(directory / "rdzv"), mode), nprocs=world, join=True)
    finally:
        shutil.rmtree(directory)
