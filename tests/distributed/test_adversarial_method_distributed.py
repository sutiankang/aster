from copy import deepcopy
from datetime import timedelta
import importlib.util
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.methods.adversarial_autoencoder import AdversarialAutoencoderMethod
from aster.models.adversarial import PatchDiscriminator, PatchDiscriminatorConfig
from aster.models.generative import AutoencoderKL, AutoencoderConfig
from aster.models.perceptual import LPIPS, LPIPSConfig
from aster.training import ParallelContext, Trainer


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
        spec = importlib.util.spec_from_file_location(
            "gan_formula", Path(__file__).parents[1] / "unit/test_adversarial_method.py"
        )
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        context = ParallelContext()
        generator_config = AutoencoderConfig(
            base_channels=4, latent_channels=2, channel_mult=(1, 2), num_res_blocks=1
        )
        discriminator_config = PatchDiscriminatorConfig(base_channels=4, num_layers=1)
        metric_config = LPIPSConfig(channels=(2, 3, 4, 4, 4), allow_untrained=True)
        for stage in range(4):
            torch.manual_seed(61)
            generator, discriminator, metric = (
                AutoencoderKL(generator_config),
                PatchDiscriminator(discriminator_config),
                LPIPS(metric_config),
            )
            random = torch.Generator().manual_seed(51 + rank)
            batches = [
                dict(
                    sample=torch.rand(n, 3, 16, 16, generator=random) * 2 - 1,
                    posterior_noise=torch.randn(n, 2, 8, 8, generator=random),
                )
                for n in (1 + rank, 2 - rank)
            ]
            all_batches = context.dp.gather_objects(batches)
            full = {
                key: torch.cat([batch[key] for rows in all_batches for batch in rows])
                for key in batches[0]
            }
            discriminator.initialize(
                torch.cat([batch["sample"] for batch in batches]), group=context.dp
            )
            dense_g, dense_d, dense_p = (
                deepcopy(generator),
                deepcopy(discriminator),
                deepcopy(metric),
            )
            factory = lambda parameters: torch.optim.SGD(parameters, lr=0.001, momentum=0.9)
            go, do = factory(dense_g.parameters()), factory(dense_d.parameters())
            engine = Trainer(
                generator,
                parallel=context,
                zero_stage=stage,
                accumulation_steps=2,
                optimizer_factory=factory,
                max_grad_norm=None,
            )
            method = AdversarialAutoencoderMethod(
                engine,
                metric,
                discriminator,
                discriminator_optimizer_factory=factory,
                pixel_reduction="mean",
                disc_factor=0.7,
                disc_weight=0.4,
                disc_start=1,
            )
            for step in range(2):
                expected_g, expected_d, coefficient = oracle.independent_step(
                    dense_g, dense_d, dense_p, full, go, do, active=step >= 1
                )
                result = method.update(batches)
                assert result.updated and abs(result.generator.loss - float(expected_g)) < 5e-6
                assert abs(result.discriminator.loss - float(expected_d)) < 5e-6
                assert (
                    abs(
                        engine.last_gradient_ratio(method.policy_name)["effective_weight"]
                        - float(coefficient)
                    )
                    < 2e-6
                )
                for role, dense in (("model", dense_g), (method.discriminator_role, dense_d)):
                    for name, value in engine.export_state_dict(
                        role=role, only_rank_zero=False
                    ).items():
                        torch.testing.assert_close(
                            value, dense.state_dict()[name], atol=3e-6, rtol=3e-5
                        )
            checkpoint = engine.save_checkpoint(Path(output) / f"gan_zero{stage}")
            expected = method.update(batches)
            weights = {
                role: deepcopy(engine.export_state_dict(role=role, only_rank_zero=False))
                for role in ("model", method.discriminator_role)
            }
            ratio = engine.last_gradient_ratio(method.policy_name)
            engine.load_checkpoint(checkpoint, trusted=True)
            actual = method.update(batches)
            assert (
                expected.generator.loss == actual.generator.loss
                and expected.discriminator.loss == actual.discriminator.loss
            )
            assert engine.last_gradient_ratio(method.policy_name) == ratio and method.updates == 3
            for role, state in weights.items():
                for name, value in engine.export_state_dict(
                    role=role, only_rank_zero=False
                ).items():
                    torch.testing.assert_close(value, state[name], atol=0, rtol=0)
            bad = deepcopy(batches)
            if rank == 1:
                bad[1]["posterior_noise"] = torch.zeros(1)
            calls = []
            hook = engine.model.encoder.register_forward_pre_hook(lambda *_: calls.append(True))
            rejected = False
            try:
                method.update(bad)
            except ValueError as error:
                rejected = "posterior noise" in str(error)
            finally:
                hook.remove()
            assert rejected and not calls and not engine._failed and method.updates == 3
    finally:
        dist.destroy_process_group()


def test_adversarial_vae_true_dp2_all_zero_global_ratio_complete_updates_and_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_gan_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_gan_"):
            shutil.rmtree(directory)
