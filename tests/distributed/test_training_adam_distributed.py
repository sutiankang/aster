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
from aster.training import Trainer, ParallelContext
from aster.training.portable import optimizer_mapping


def _model():
    return nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))


def _factory(parameters):
    parameters = list(parameters)
    return torch.optim.Adam(
        [
            {"params": parameters[::2], "lr": 0.02, "eps": 1e-5, "weight_decay": 0.15},
            {"params": parameters[1::2], "lr": 0.007, "eps": 1e-3, "weight_decay": 0.3},
        ],
        betas=(0.7, 0.91),
        amsgrad=True,
    )


def _objective(model, batch):
    x, target = batch
    values = (model(x) - target).square()
    return LossTerm(values.sum(), torch.tensor(values.numel()), "elements")


def _batch():
    generator = torch.Generator().manual_seed(725)
    return torch.randn(6, 2, generator=generator), torch.randn(6, 1, generator=generator)


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
        full_batch = _batch()
        local = tuple(value[:2] if rank == 0 else value[2:] for value in full_batch)
        for stage in range(4):
            for offload in ("none", "cpu", "nvme"):
                torch.manual_seed(316)
                model = _model()
                dense = deepcopy(model)
                optimizer = _factory(dense.parameters())
                directory = Path(output) / f"zero{stage}_{offload}"
                engine = Trainer(
                    model,
                    _objective,
                    parallel=context,
                    optimizer_factory=_factory,
                    zero_stage=stage,
                    offload_optimizer=offload,
                    offload_directory=directory / f"disk_{rank}" if offload == "nvme" else None,
                    max_grad_norm=None,
                )
                actual, _, _ = optimizer_mapping(engine.roles["model"])
                assert type(actual) is torch.optim.Adam
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    _objective(dense, full_batch).mean.backward()
                    optimizer.step()
                    engine.step([local])
                    exported = engine.export_state_dict(only_rank_zero=False)
                    for name, value in exported.items():
                        torch.testing.assert_close(
                            value, dense.state_dict()[name], atol=3e-7, rtol=3e-6
                        )
                if stage:
                    assert sum(p.numel() for g in actual.param_groups for p in g["params"]) < sum(
                        p.numel() for p in dense.parameters()
                    )
                checkpoint = engine.save_checkpoint(directory / "native")
                engine.step([local])
                expected = engine.export_state_dict(only_rank_zero=False)
                engine.load_checkpoint(checkpoint)
                engine.step([local])
                for name, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(value, expected[name], atol=0, rtol=0)

        engine.save_portable_checkpoint(Path(output) / "portable")
        engine.step([local])
        expected = engine.export_state_dict()
        if rank == 0:
            torch.save(expected, Path(output) / "next.pt")
        context.world.barrier()
        for stage in (0, 3):

            def fail_on_one_rank(parameters):
                if rank == 1:
                    raise RuntimeError("injected factory error")
                return _factory(parameters)

            with pytest.raises(ValueError, match="injected factory error"):
                Trainer(
                    _model(), parallel=context, zero_stage=stage, optimizer_factory=fail_on_one_rank
                )
    finally:
        dist.destroy_process_group()


def test_dp2_adam_all_zero_offload_factory_and_portable_dense_next_update(tmp_path):
    temp_root = Path(tempfile.gettempdir())
    if not str(temp_root).isascii():
        temp_root = Path("C:/Temp")
    rendezvous_dir = Path(tempfile.mkdtemp(prefix="aster_adam_", dir=temp_root))
    try:
        mp.spawn(_worker, args=(str(rendezvous_dir / "store"), str(tmp_path)), nprocs=2, join=True)
        torch.set_num_threads(1)
        target = Trainer(_model(), _objective, optimizer_factory=_factory, max_grad_norm=None)
        target.load_portable_checkpoint(tmp_path / "portable", seed=31)
        target.step([_batch()])
        expected = torch.load(tmp_path / "next.pt", weights_only=True)
        for name, value in target.export_state_dict().items():
            torch.testing.assert_close(value, expected[name], atol=3e-7, rtol=3e-6)
    finally:
        if rendezvous_dir.parent == temp_root and rendezvous_dir.name.startswith("aster_adam_"):
            shutil.rmtree(rendezvous_dir)
