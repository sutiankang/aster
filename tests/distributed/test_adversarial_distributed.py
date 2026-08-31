from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from aster.core import LossTerm
from aster.models.adversarial import ActNorm2d, PatchDiscriminator, PatchDiscriminatorConfig
from aster.training import ParallelContext, Trainer


class DiscriminatorObjective:
    def config_dict(self):
        return dict(type="patchgan_hinge_test")

    def __call__(self, model, batch):
        real, fake = model(batch["real"]).float(), model(batch["fake"]).float()

        value = 0.5 * (F.relu(1 - real).flatten(1).mean(1) + F.relu(1 + fake).flatten(1).mean(1))
        return LossTerm(value.sum(), value.new_tensor(len(value)), "sample", "discriminator")


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

        torch.manual_seed(60 + rank)
        calibration = torch.randn(1 + rank, 3, 4 + rank, 6, dtype=torch.float64) + rank * 3
        act = ActNorm2d(3).double()
        act.initialize(calibration, group=parallel.dp)
        values = parallel.dp.gather_objects(calibration)
        flat = torch.cat([v.permute(1, 0, 2, 3).reshape(3, -1) for v in values], 1)
        torch.testing.assert_close(act.affine.loc.flatten(), -flat.mean(1), atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(
            act.affine.scale.flatten(), (flat.std(1) + 1e-6).reciprocal(), atol=1e-13, rtol=1e-13
        )
        config = PatchDiscriminatorConfig(base_channels=4, num_layers=1)
        for stage in range(4):
            torch.manual_seed(731)
            model = PatchDiscriminator(config)
            initial = deepcopy(model.state_dict())
            generator = torch.Generator().manual_seed(18 + rank)
            batches = [
                dict(
                    real=torch.randn(n, 3, 16, 16, generator=generator),
                    fake=torch.randn(n, 3, 16, 16, generator=generator) * 0.8,
                )
                for n in (1 + rank, 2 - rank)
            ]
            all_batches = parallel.dp.gather_objects(batches)
            full = {
                key: torch.cat([batch[key] for rows in all_batches for batch in rows])
                for key in batches[0]
            }
            dense = PatchDiscriminator(config)
            dense.load_state_dict(initial)
            dense.initialize(full["real"])
            model.initialize(torch.cat([b["real"] for b in batches]), group=parallel.dp)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, dense.state_dict()[name], atol=3e-6, rtol=2e-6)

            dense.load_state_dict(model.state_dict())
            objective = DiscriminatorObjective()
            engine = Trainer(
                model,
                objective,
                parallel=parallel,
                zero_stage=stage,
                accumulation_steps=2,
                max_grad_norm=None,
                optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.001, momentum=0.9),
            )
            optimizer = torch.optim.SGD(dense.parameters(), lr=0.001, momentum=0.9)
            loss = objective(dense, full).mean
            loss.backward()
            norm = torch.sqrt(sum(p.grad.float().square().sum() for p in dense.parameters()))
            optimizer.step()
            result = engine.step(batches)
            assert result.updated and abs(result.loss - loss.item()) < 2e-6
            assert abs(result.grad_norm - float(norm)) < 2e-5
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, dense.state_dict()[name], atol=2e-6, rtol=3e-5)
            checkpoint = engine.save_checkpoint(Path(output) / f"patchgan_zero{stage}")
            expected = engine.step(batches)
            expected_weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(checkpoint, trusted=True)
            actual = engine.step(batches)
            assert actual.loss == expected.loss and actual.grad_norm == expected.grad_norm
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, expected_weights[name], atol=0, rtol=0)

        torch.manual_seed(9)
        broken = PatchDiscriminator(config)
        if rank == 0:
            with torch.no_grad():
                next(broken.parameters()).add_(1)
        before = deepcopy(broken.state_dict())
        rejected = False
        try:
            broken.initialize(batches[0]["real"], group=parallel.dp)
        except ValueError as error:
            rejected = "differ" in str(error)
        assert rejected and all(torch.equal(v, broken.state_dict()[k]) for k, v in before.items())
    finally:
        dist.destroy_process_group()


def test_patchgan_true_dp2_global_calibration_all_zero_updates_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_patchgan_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_patchgan_"):
            shutil.rmtree(directory)
