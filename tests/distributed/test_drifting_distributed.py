from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models.drifting import DriftingConfig, DriftingGenerator
from aster.methods.drifting import DriftingMethod, SpatialFeatureStatistics
from aster.methods.generative_distillation import drifting_loss
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

        torch.manual_seed(92)
        complete = torch.randn(5, 3, 7, requires_grad=True)
        positive, negative = torch.randn(5, 4, 7) + 2, torch.randn(5, 2, 7) - 1
        weights = torch.tensor([0.1, 1.0, 2.0, 0.3, 4.0])[:, None].expand(5, 2)
        reference, info = drifting_loss(complete, positive, negative, negative_weights=weights)
        (reference.numerator / 5).backward()
        selection = slice(0, 2) if rank == 0 else slice(2, 5)
        local = complete.detach()[selection].clone().requires_grad_()
        loss, actual_info = drifting_loss(
            local,
            positive[selection],
            negative[selection],
            negative_weights=weights[selection],
            statistics_group=context.dp,
        )
        (loss.numerator / 5).backward()
        torch.testing.assert_close(local.grad, complete.grad[selection], atol=4e-7, rtol=2e-5)
        for key in info:
            torch.testing.assert_close(actual_info[key], info[key], atol=2e-6, rtol=2e-5)
        expected = None
        for stage in range(4):
            torch.manual_seed(85)
            config = DriftingConfig(
                input_size=4,
                in_channels=1,
                out_channels=1,
                hidden_size=16,
                cond_dim=12,
                num_layers=1,
                num_heads=2,
                num_classes=2,
                noise_classes=3,
                noise_coords=2,
            )
            engine = Trainer(
                DriftingGenerator(config),
                parallel=context,
                zero_stage=stage,
                max_grad_norm=None,
                optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.001, momentum=0.9),
                ema_decay=0.9,
            )
            method = DriftingMethod(
                engine,
                SpatialFeatureStatistics(patch_sizes=(2,), use_std=False),
                feature_identity="pixels-global-v1",
                positive_capacity=4,
                negative_capacity=8,
                positive_samples=3,
                negative_samples=2,
                generated_samples=3,
                seed=3,
            )
            torch.manual_seed(918 + rank)
            batch = dict(samples=torch.randn(rank + 2, 1, 4, 4), labels=torch.arange(rank + 2) % 2)
            invalid = deepcopy(batch)
            if rank:
                invalid["samples"][0, 0, 0, 0] = float("nan")
            with pytest.raises(ValueError, match="preflight"):
                method.update([invalid])
            assert method.positive.count.sum() == 0 and engine.steps == 0
            assert method.update([batch]).updated
            checkpoint = engine.save_checkpoint(Path(output) / f"zero{stage}")
            result = method.update([batch])
            weights = engine.export_state_dict(only_rank_zero=False)
            if expected is None:
                expected = deepcopy(weights)
            else:
                for key in expected:
                    torch.testing.assert_close(weights[key], expected[key], atol=5e-7, rtol=3e-5)
            queues = method.state_dict()
            engine.load_checkpoint(checkpoint, trusted=True)
            replayed = method.update([batch])
            assert result.loss == replayed.loss
            for key, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
            torch.testing.assert_close(
                method.positive.values, queues["positive"]["values"], atol=0, rtol=0
            )

        features = SpatialFeatureStatistics(patch_sizes=(2,), use_std=bool(rank))
        with pytest.raises(ValueError, match="configurations"):
            DriftingMethod(engine, features, feature_identity="inconsistent-feature-contract")
    finally:
        dist.destroy_process_group()


def test_drifting_real_dp2_global_statistics_all_zero_and_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_drifting_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_drifting_"):
            shutil.rmtree(directory)
