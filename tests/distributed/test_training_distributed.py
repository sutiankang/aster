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

from aster.core.contracts import LossTerm, LossBundle
from aster.training import (
    Trainer,
    ParallelConfig,
    ParallelContext,
    ColumnParallelLinear,
    RowParallelLinear,
    vocab_parallel_cross_entropy,
)
from aster.training.sharding import Zero3Unit


def _objective(model, batch):
    x, y, mask = batch
    output = model(x)
    return LossBundle(
        (
            LossTerm(((output - y).square() * mask).sum(), mask.sum(), "tokens", "mse", 0.7),
            LossTerm(output.abs().sum(), torch.tensor(output.shape[0]), "samples", "l1", 0.3),
        )
    )


def _batch(replica):
    generator = torch.Generator().manual_seed(300 + replica)
    count = 2 + 3 * replica
    x, y = torch.randn(count, 3, generator=generator), torch.randn(count, 2, generator=generator)
    mask = torch.zeros_like(y) if replica == 0 else torch.ones_like(y)
    return x, y, mask


def _dense_step(model, optimizer):
    terms = [_objective(model, _batch(i)).terms for i in range(2)]
    loss = (
        sum(0.7 * term[0].numerator for term in terms) / 10
        + sum(0.3 * term[1].numerator for term in terms) / 7
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4)
    optimizer.step()


def _worker(rank, world_size, rendezvous, checkpoint_dir, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    try:
        context = ParallelContext(
            ParallelConfig(tensor_parallel=2, data_parallel=2)
            if mode == "tp"
            else ParallelConfig(data_parallel=2)
        )
        for stage in (0,) if mode == "dp_overlap" else (0, 1, 2, 3):
            torch.manual_seed(55)
            full = nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 2))
            reference = deepcopy(full)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
            if mode == "tp":
                model = nn.Sequential(
                    ColumnParallelLinear(3, 6, context.tp),
                    nn.Tanh(),
                    RowParallelLinear(6, 2, context.tp),
                )
                with torch.no_grad():
                    model[0].weight.copy_(full[0].weight.chunk(2, dim=0)[context.tp.rank])
                    model[0].bias.copy_(full[0].bias.chunk(2)[context.tp.rank])
                    model[2].weight.copy_(full[2].weight.chunk(2, dim=1)[context.tp.rank])
                    model[2].bias.copy_(full[2].bias)
            else:
                model = full
            trainer = Trainer(
                model,
                _objective,
                lr=0.01,
                max_grad_norm=0.4,
                zero_stage=stage,
                parallel=context,
                communication_overlap=mode == "dp_overlap",
                bucket_bytes=32,
                accumulation_steps=2 if mode == "dp_overlap" else 1,
            )
            for _ in range(2):
                _dense_step(reference, reference_optimizer)
                result = trainer.step([_batch(context.dp.rank)] * trainer.accumulation_steps)
                assert (
                    result.updated
                    and result.terms["mse"]["denominator"] == 10 * trainer.accumulation_steps
                )
                if mode == "dp_overlap":
                    assert trainer.last_communication_buckets > 1
                with torch.no_grad():
                    actual = trainer.model(_batch(0)[0])
                    expected = reference(_batch(0)[0])
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
            path = Path(checkpoint_dir) / f"{mode}-zero{stage}.json"
            trainer.save_checkpoint(path)
            trainer.step([_batch(context.dp.rank)] * trainer.accumulation_steps)
            with torch.no_grad():
                expected_next = trainer.model(_batch(0)[0]).clone()
            trainer.load_checkpoint(path)
            trainer.step([_batch(context.dp.rank)] * trainer.accumulation_steps)
            with torch.no_grad():
                actual_next = trainer.model(_batch(0)[0])
            torch.testing.assert_close(actual_next, expected_next, atol=0, rtol=0)
            if stage == 3:
                units = [unit for unit in trainer.model.modules() if isinstance(unit, Zero3Unit)]
                assert all(p.numel() == 0 for unit in units for p in unit.module.parameters())
                resident = sum(p.numel() for unit in units for p in unit.shards)
                local_full = sum(sum(unit.sizes) for unit in units)
                assert resident < local_full
            if mode == "dp":
                target = trainer.clone_target(
                    "model",
                    "target",
                    factory=lambda: nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 2)),
                )
                before = deepcopy(target.state_dict())
                trainer.step([_batch(context.dp.rank)])
                trainer.update_target("model", "target", 0.8)
                full_state = trainer.export_state_dict(only_rank_zero=False)
                for key, value in target.state_dict().items():
                    torch.testing.assert_close(value, before[key] * 0.8 + full_state[key] * 0.2)
            dist.barrier()
        if mode.startswith("dp"):
            model = nn.Linear(1, 1)
            before = deepcopy(model.state_dict())
            trainer = Trainer(
                model,
                lambda m, x: LossTerm(
                    m(x).sum() * (float("nan") if rank == 1 else 1.0), torch.tensor(1), "sample"
                ),
                parallel=context,
                ema_decay=0.9,
            )
            before = deepcopy(model.state_dict())
            result = trainer.step([torch.ones(1, 1)])
            assert result.overflow and not result.updated and trainer.steps == 0
            for key, value in before.items():
                torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
        else:
            generator = torch.Generator().manual_seed(99)
            logits = torch.randn(5, 8, generator=generator, requires_grad=True)
            targets = torch.tensor([0, 7, -100, 4, 1])
            local = logits.detach().chunk(2, -1)[context.tp.rank].clone().requires_grad_()
            output = vocab_parallel_cross_entropy(local, targets, context.tp)
            reference_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            torch.testing.assert_close(output, reference_loss)
            output.sum().backward()
            reference_loss.sum().backward()
            torch.testing.assert_close(local.grad, logits.grad.chunk(2, -1)[context.tp.rank])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode,world_size", [("dp", 2), ("tp", 4), ("dp_overlap", 2)])
def test_gloo_zero_updates_and_resume(tmp_path, mode, world_size):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    if not root.exists():
        raise RuntimeError("需要可写 ASCII 临时目录供 Windows FileStore 使用")
    directory = Path(tempfile.mkdtemp(prefix="aster-training-gloo-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-training-gloo-")
    try:
        mp.spawn(
            _worker,
            args=(world_size, str(directory / "rendezvous"), str(tmp_path), mode),
            nprocs=world_size,
            join=True,
        )
    finally:
        shutil.rmtree(directory)
