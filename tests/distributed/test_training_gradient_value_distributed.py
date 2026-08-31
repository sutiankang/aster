from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.core import LossTerm
from aster.training import ParallelContext, ParallelConfig, Trainer


def objective(model, batch):
    return LossTerm(model(batch).sum(), torch.tensor(len(batch), dtype=torch.int64), "sample")


def worker(rank, rendezvous, directory):
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
        for zero in (0, 1, 2, 3):
            torch.manual_seed(735)
            reference = nn.Linear(2, 2)
            model = deepcopy(reference)
            engine = Trainer(
                model,
                objective,
                parallel=context,
                zero_stage=zero,
                optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.1),
                max_grad_norm=None,
                max_grad_value=0.5,
                offload_optimizer="cpu",
            )
            left, right = (
                torch.tensor([[10.0, 1.0]]),
                torch.tensor([[-3.0, 2.0], [-3.0, 2.0], [-3.0, 2.0]]),
            )
            optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                objective(reference, torch.cat((left, right))).mean.backward()
                torch.nn.utils.clip_grad_value_(reference.parameters(), 0.5)
                optimizer.step()
                result = engine.step([left if rank == 0 else right])
                assert result.updated
                for key, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(
                        value, reference.state_dict()[key], atol=2e-7, rtol=2e-6
                    )
            engine.max_grad_value = 0.5 + rank
            with pytest.raises(ValueError, match="Phase declaration"):
                engine.step([left])
        with pytest.raises(ValueError, match="clipping"):
            Trainer(
                nn.Linear(2, 2),
                objective,
                parallel=context,
                zero_stage=3,
                max_grad_value=1.0 + rank,
            )
        with pytest.raises(ValueError, match="max_grad_value"):
            Trainer(
                nn.Linear(2, 2),
                objective,
                parallel=context,
                zero_stage=3,
                max_grad_value=-1.0 if rank else 1.0,
            )
    finally:
        dist.destroy_process_group()


def test_value_clip_reduces_global_unequal_counts_before_clamping(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-clip-value-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-clip-value-")
    try:
        mp.spawn(worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        shutil.rmtree(directory)
