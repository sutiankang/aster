from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models.interval_dit import IntervalDiT, IntervalDiTConfig
from aster.methods.meanflow import MeanFlowObjective
from aster.training import Trainer, ParallelContext


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        context = ParallelContext()
        torch.manual_seed(334)
        config = IntervalDiTConfig(
            input_size=4, in_channels=1, hidden_size=16, num_layers=1, num_heads=2, num_classes=2
        )
        initial = IntervalDiT(config).state_dict()
        complete = dict(
            sample=torch.randn(5, 1, 4, 4),
            noise=torch.randn(5, 1, 4, 4),
            time=torch.tensor([0.7, 0.8, 0.5, 0.3, 0.9]),
            reference_time=torch.tensor([0.7, 0.2, 0.5, 0.1, 0.0]),
            labels=torch.tensor([0, 1, 0, 1, 1]),
            drop_count=0,
        )
        selection = slice(0, 2) if rank == 0 else slice(2, 5)
        local = {
            k: value[selection] if isinstance(value, torch.Tensor) else value
            for k, value in complete.items()
        }
        for stage in range(4):
            dense = IntervalDiT(config)
            dense.load_state_dict(initial)
            model = IntervalDiT(config)
            model.load_state_dict(initial)
            objective = MeanFlowObjective(omega=1.2, kappa=0.3)
            dense_optimizer = torch.optim.SGD(dense.parameters(), lr=0.001, momentum=0.9)
            engine = Trainer(
                model,
                objective,
                parallel=context,
                zero_stage=stage,
                max_grad_norm=None,
                optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.001, momentum=0.9),
            )
            invalid = {**local, "drop_count": 100 if rank == 1 else 0}
            with pytest.raises(ValueError, match="drop_count"):
                engine.step([invalid])
            for iteration in range(2):
                dense_optimizer.zero_grad()
                reference = objective(dense, complete)
                reference.mean.backward()
                norm = torch.linalg.vector_norm(
                    torch.stack([p.grad.norm() for p in dense.parameters()])
                ).item()
                dense_optimizer.step()
                result = engine.step([local])
                assert result.updated and abs(result.loss - reference.mean.item()) < 1e-6
                assert abs(norm - result.grad_norm) < 2e-5
                for key, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(value, dense.state_dict()[key], atol=5e-7, rtol=3e-5)
            path = engine.save_checkpoint(Path(output) / f"zero{stage}")
            expected = engine.step([local])
            weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(path, trusted=True)
            actual = engine.step([local])
            assert expected.loss == actual.loss
            for key, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_meanflow_real_dp2_zero_0_to_3_dense_update_and_restore(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_meanflow_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_meanflow_"):
            shutil.rmtree(directory)
