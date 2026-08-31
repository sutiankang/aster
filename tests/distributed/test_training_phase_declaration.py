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
from aster.training import ParallelConfig, ParallelContext, Trainer
from aster.training.sharding import zero3_units


class RepeatedObjective(nn.Module):
    def __init__(self, steps=2, invalid=None):
        super().__init__()
        self.steps, self.invalid = steps, invalid
        self.preflights = self.forwards = 0

    def config_dict(self):
        if self.invalid == "raise":
            raise ValueError("broken objective codec")
        return {"steps": self.steps, "coefficient": float("nan") if self.invalid == "nan" else 1.0}

    def preflight_microbatches(self, model, batches):
        self.preflights += 1
        return batches

    def forward(self, model, batch):
        self.forwards += 1
        value = batch
        for _ in range(self.steps):
            value = model(value)
        return LossTerm(
            value.square().sum(), torch.tensor(value.numel(), dtype=torch.int64), "element"
        )


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=80),
    )
    try:
        context = ParallelContext(ParallelConfig(data_parallel=2))
        torch.manual_seed(441)
        default = RepeatedObjective()
        engine = Trainer(nn.Linear(3, 3), default, parallel=context, zero_stage=3)
        engine.add_role("other", nn.Linear(3, 3))
        engine.add_role("target", nn.Linear(3, 3), trainable=False)
        states = {
            name: {key: value.clone() for key, value in role.model.state_dict().items()}
            for name, role in engine.roles.items()
        }
        rng = torch.get_rng_state().clone()
        units = [unit for role in engine.roles.values() for unit in zero3_units(role.model)]
        gathers = [unit.gathers for unit in units]
        for case in (
            "objective",
            "mutation",
            "codec_error",
            "codec_nan",
            "name",
            "role",
            "freeze",
            "invalid_freeze",
        ):
            default.steps = 2
            objective = (
                RepeatedObjective(steps=2 + rank) if case == "objective" else RepeatedObjective()
            )
            if case == "mutation":
                default.steps = 2 + rank
                objective = None
            if case in {"codec_error", "codec_nan"} and rank == 1:
                objective.invalid = "raise" if case == "codec_error" else "nan"
            name = f"train-{rank}" if case == "name" else "train"
            role = "other" if case == "role" and rank else "model"
            freeze = ("target",) if case == "freeze" and rank else ()
            if case == "invalid_freeze" and rank:
                freeze = ("absent",)
            with pytest.raises(ValueError, match="Phase declaration"):
                engine.phase(
                    name,
                    role=role,
                    objective=objective,
                    microbatches=[torch.ones(2, 3)],
                    freeze_roles=freeze,
                )
            assert not engine._busy and not engine._failed and engine.steps == 0
            assert [unit.gathers for unit in units] == gathers
            assert torch.equal(torch.get_rng_state(), rng)
            for key, item in engine.roles.items():
                assert item.updates == 0
                for parameter, value in item.model.state_dict().items():
                    assert torch.equal(value, states[key][parameter])
            actual = default if objective is None else objective
            assert actual.preflights == actual.forwards == 0

        default.steps = 2
        batch = torch.full((2, 3), 0.25 + rank)
        assert engine.step([batch]).updated
        path = Path(output) / "phase-complete.json"
        engine.save_checkpoint(path)
        result = engine.step([batch])
        expected = engine.export_state_dict(only_rank_zero=False)
        engine.load_checkpoint(path)
        repeated = engine.step([batch])
        assert result == repeated
        actual = engine.export_state_dict(only_rank_zero=False)
        for key in expected:
            assert torch.equal(actual[key], expected[key])
    finally:
        dist.destroy_process_group()


def test_phase_declaration_is_collectively_rejected_before_zero3_materialization(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-phase-contract-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-phase-contract-")
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        shutil.rmtree(directory)
