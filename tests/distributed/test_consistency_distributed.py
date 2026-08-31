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
from aster.models.generative import UNet2D, UNetConfig
from aster.methods.consistency import ConsistencyConfig, ConsistencyMethod, _ConsistencyObjective
from aster.training import Trainer, ParallelContext, ParallelConfig
from aster.training.sharding import Zero3Unit
from aster.training.portable import logical_tensors, optimizer_mapping, gather_tensor


def _model(teacher=False):
    return UNet2D(
        UNetConfig(
            in_channels=1,
            model_channels=4,
            num_res_blocks=1,
            channel_mult=(1,),
            num_heads=1,
            attention_levels=(),
            prediction_type="edm_residual" if teacher else "consistency_residual",
        )
    )


def _batch(rank):
    generator = torch.Generator().manual_seed(32 + rank)
    count = 2 - rank
    return {
        "sample": torch.randn(count, 1, 4, 4, generator=generator),
        "noise": torch.randn(count, 1, 4, 4, generator=generator),
        "interval_indices": torch.full((count,), rank + 1, dtype=torch.long),
    }


def _mse(model, batch):
    x, y = batch
    loss = (model(x) - y).square()
    return LossTerm(loss.sum(), torch.tensor(loss.numel(), dtype=torch.int64), "element")


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=160),
    )
    try:
        context = ParallelContext(ParallelConfig(data_parallel=2))
        for mode in ("ct", "cd", "ict"):
            for zero in (0, 1, 2, 3):
                torch.manual_seed(743)
                model = _model()
                oracle = deepcopy(model)
                target = deepcopy(model).requires_grad_(False)
                teacher = _model(True) if mode == "cd" else None
                config = ConsistencyConfig(
                    mode=mode,
                    total_steps=20,
                    initial_scales=4,
                    final_scales=4,
                    curriculum="fixed",
                    target_ema_mode="fixed",
                    sampling_ema=0.8,
                    time_scale=250.0,
                )

                factory = lambda parameters: torch.optim.RAdam(
                    parameters, lr=0.003, betas=(0.7, 0.91), eps=1e-4, weight_decay=0.03
                )
                optimizer = factory(oracle.parameters())
                objective = _ConsistencyObjective(config, target, deepcopy(teacher))
                engine = Trainer(
                    model,
                    parallel=context,
                    zero_stage=zero,
                    optimizer_factory=factory,
                    max_grad_norm=None,
                )
                method = ConsistencyMethod(
                    engine, config=config, target_factory=_model, teacher=teacher
                )
                for step in range(7):
                    levels = config.levels(step)
                    terms = []
                    for replica in range(2):
                        batch = _batch(replica)
                        indices = batch["interval_indices"]
                        prepared = {
                            **batch,
                            "sigma_high": levels[indices + 1].float(),
                            "sigma_low": levels[indices].float(),
                        }
                        terms.append(objective(oracle, prepared))
                    optimizer.zero_grad(set_to_none=True)
                    expected_loss = sum(term.numerator for term in terms) / sum(
                        term.denominator for term in terms
                    )
                    expected_loss.backward()
                    optimizer.step()
                    _, decay = config.scales_and_ema(step)
                    with torch.no_grad():
                        for a, b in zip(target.parameters(), oracle.parameters()):
                            a.lerp_(b, 1 - decay)
                    result = method.update([_batch(rank)])
                    assert result.loss == pytest.approx(
                        float(expected_loss.detach()), rel=5e-5, abs=3e-7
                    )
                    assert result.terms["consistency"]["denominator"] == 3
                    actual = engine.export_state_dict(only_rank_zero=False)
                    _, owners, sharded = optimizer_mapping(engine.roles["model"])
                    gradients = {
                        entry.name: gather_tensor(
                            owners[id(entry.tensor)].grad, entry, context, optimizer_sharded=sharded
                        )
                        for entry in logical_tensors(model, context)
                        if entry.parameter
                    }
                    for key, parameter in oracle.named_parameters():
                        torch.testing.assert_close(
                            gradients[key],
                            parameter.grad,
                            rtol=2e-4,
                            atol=5e-7,
                            msg=lambda message: (
                                f"{mode}/zero{zero}/step{step}/{key} gradient: " + message
                            ),
                        )
                    for key, value in oracle.state_dict().items():
                        torch.testing.assert_close(
                            actual[key],
                            value,
                            rtol=2e-4,
                            atol=6e-7,
                            msg=lambda message: (
                                f"{mode}/zero{zero}/step{step}/{key}: "
                                + message
                                + f"\ndense grad={dict(oracle.named_parameters())[key].grad}\nactual grad={gradients[key]}"
                            ),
                        )
                        torch.testing.assert_close(
                            method.target.state_dict()[key],
                            target.state_dict()[key],
                            rtol=2e-4,
                            atol=6e-7,
                        )
                if zero == 3:
                    units = [module for module in model.modules() if isinstance(module, Zero3Unit)]
                    assert units and all(
                        parameter.numel() == 0
                        for unit in units
                        for parameter in unit.module.parameters()
                    )
                path = engine.save_checkpoint(Path(output) / f"{mode}-zero{zero}")
                random_batch = {"sample": _batch(rank)["sample"]}
                expected_result = method.update([random_batch])
                expected = engine.export_state_dict(only_rank_zero=False)
                expected_target = deepcopy(method.target.state_dict())
                expected_rng = method.generator.get_state()
                engine.load_checkpoint(path)
                resumed = method.update([random_batch])
                assert resumed == expected_result
                assert torch.equal(method.generator.get_state(), expected_rng)
                for key, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                for key, value in method.target.state_dict().items():
                    torch.testing.assert_close(value, expected_target[key], rtol=0, atol=0)
                before = engine.export_state_dict(only_rank_zero=False)
                bad = _batch(rank)
                if rank == 1:
                    bad["sample"][0, 0, 0, 0] = torch.nan
                with pytest.raises(ValueError, match="collective preflight"):
                    method.update([bad])
                assert not method._incomplete and not engine._failed
                for key, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)

        torch.manual_seed(337)
        model = nn.Linear(2, 1)
        engine = Trainer(
            model,
            _mse,
            parallel=context,
            zero_stage=3,
            optimizer_factory=lambda p: torch.optim.RAdam(p, lr=0.03, betas=(0.7, 0.91)),
            max_grad_norm=None,
        )
        batch = (torch.tensor([[1.0, rank + 0.1]]), torch.tensor([[rank * 0.4]]))
        for _ in range(7):
            engine.step([batch])
        engine.save_portable_checkpoint(Path(output) / "radam-portable")
        engine.step([batch])
        expected = engine.export_state_dict()
        if rank == 0:
            torch.save(expected, Path(output) / "radam-next.pt")
    finally:
        dist.destroy_process_group()


def test_unet_consistency_dp2_zero0_to3_native_restore_and_radam_reshard(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_consistency_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    finally:
        assert directory.parent == root.resolve() and directory.name.startswith(
            "aster_consistency_"
        )
        shutil.rmtree(directory)
    restored = Trainer(
        nn.Linear(2, 1),
        _mse,
        optimizer_factory=lambda p: torch.optim.RAdam(p, lr=0.03, betas=(0.7, 0.91)),
        max_grad_norm=None,
    )
    restored.load_portable_checkpoint(tmp_path / "radam-portable", seed=531)
    restored.step([(torch.tensor([[1.0, 0.1], [1.0, 1.1]]), torch.tensor([[0.0], [0.4]]))])
    expected = torch.load(tmp_path / "radam-next.pt", weights_only=True)
    for key, value in restored.export_state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=5e-6, atol=2e-7)
