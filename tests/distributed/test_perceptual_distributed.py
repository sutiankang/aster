from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models.generative import AutoencoderKL, AutoencoderConfig
from aster.models.perceptual import LPIPS, LPIPSConfig
from aster.methods.perceptual_autoencoder import (
    PerceptualAutoencoderMethod,
    PerceptualAutoencoderObjective,
)
from aster.training import Trainer, ParallelContext


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        parallel = ParallelContext()
        config = AutoencoderConfig(
            base_channels=4, latent_channels=2, channel_mult=(1, 2), num_res_blocks=1
        )
        metric_config = LPIPSConfig(channels=(2, 3, 4, 4, 4), allow_untrained=True)
        for stage in range(4):
            torch.manual_seed(713)
            model = AutoencoderKL(config)
            initial = deepcopy(model.state_dict())
            metric = LPIPS(metric_config)
            metric_weights = deepcopy(metric.state_dict())
            engine = Trainer(
                model,
                parallel=parallel,
                zero_stage=stage,
                accumulation_steps=2,
                max_grad_norm=None,
                optimizer_factory=lambda parameters: torch.optim.SGD(
                    parameters, lr=0.001, momentum=0.9
                ),
            )
            method = PerceptualAutoencoderMethod(engine, metric, pixel_reduction="mean")
            generator = torch.Generator().manual_seed(18 + rank)
            batches = [
                dict(
                    sample=torch.rand(n, 3, 16, 16, generator=generator) * 2 - 1,
                    posterior_noise=torch.randn(n, 2, 8, 8, generator=generator),
                )
                for n in (1 + rank, 2 - rank)
            ]
            all_batches = parallel.world.gather_objects(batches)
            full = {
                key: torch.cat([batch[key] for rows in all_batches for batch in rows])
                for key in batches[0]
            }
            dense = AutoencoderKL(config)
            dense.load_state_dict(initial)
            dense_metric = LPIPS(metric_config)
            dense_metric.load_state_dict(metric_weights)
            objective = PerceptualAutoencoderObjective(dense_metric, pixel_reduction="mean")
            optimizer = torch.optim.SGD(dense.parameters(), lr=0.001, momentum=0.9)
            loss = sum(term.mean * term.weight for term in objective(dense, full).terms)
            loss.backward()
            expected_norm = torch.sqrt(
                sum(p.grad.float().square().sum() for p in dense.parameters() if p.grad is not None)
            )
            optimizer.step()
            actual = method.update(batches)
            assert actual.updated and abs(actual.loss - loss.item()) < 2e-5
            assert abs(actual.grad_norm - float(expected_norm)) < 3e-5
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, dense.state_dict()[name], atol=2e-6, rtol=3e-5)
            checkpoint = engine.save_checkpoint(Path(output) / f"perceptual_zero{stage}")
            expected = method.update(batches)
            expected_weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(checkpoint, trusted=True)
            restored = method.update(batches)
            assert restored.loss == expected.loss
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, expected_weights[name], atol=0, rtol=0)

            invalid = deepcopy(batches)
            if rank == 0:
                invalid[1]["posterior_noise"] = torch.zeros(1)
            rejected = False
            try:
                method.update(invalid)
            except ValueError as error:
                rejected = "posterior noise" in str(error)
            assert rejected and not engine._failed and method.updates == 2
    finally:
        dist.destroy_process_group()


def test_native_perceptual_true_dp2_all_zero_unequal_batches_and_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_perceptual_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_perceptual_"):
            shutil.rmtree(directory)
