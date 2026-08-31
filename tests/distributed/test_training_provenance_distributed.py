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

from aster.core import LossTerm, atomic_json, read_json
from aster.training import Trainer, ParallelContext
from aster.training.state import read_payload, write_payload


class Objective:
    def __init__(self, scale=1.0):
        self.scale = scale

    def config_dict(self):
        return {"scale": self.scale}

    def __call__(self, model, batch):
        return LossTerm(
            (model(batch) - 1).square().sum() * self.scale,
            torch.tensor(len(batch), dtype=torch.int64),
            "example",
        )


def _worker(rank, rendezvous, directory):
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
        data = torch.ones(rank + 1, 2) * (rank + 1)
        for zero in range(4):
            torch.manual_seed(985)
            engine = Trainer(
                nn.Linear(2, 1),
                Objective(),
                zero_stage=zero,
                parallel=context,
                optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.02, momentum=0.7),
            )
            engine.phase("override", objective=Objective(3.0), microbatches=[data])
            expected = engine.last_successful_update()
            assert all(item == expected for item in context.world.gather_objects(expected))
            path = engine.save_checkpoint(Path(directory) / f"native-{zero}")
            portable = engine.save_portable_checkpoint(Path(directory) / f"portable-{zero}")
            engine.step([data])
            engine.load_checkpoint(path)
            assert engine.last_successful_update() == expected
            engine.load_portable_checkpoint(portable, seed=4)
            assert engine.last_successful_update() == expected

            optimizer = engine.roles["model"].optimizer
            old = optimizer.step
            if rank == 1:

                def partial():
                    old()
                    raise RuntimeError("one rank optimizer failed after write")

                optimizer.step = partial
            with pytest.raises(RuntimeError, match="failed"):
                engine.phase("must_not_commit", objective=Objective(7.0), microbatches=[data])
            assert engine.roles["model"].successful_update == expected and engine._failed
            optimizer.step = old
            engine.load_checkpoint(path)
            assert engine.last_successful_update() == expected

            manifest = read_json(path)
            if rank == 0:
                payload = read_payload(path.parent, manifest["entries"][1], trusted=False)
                payload["roles"]["model"]["successful_update"]["objective_configuration"][
                    "configuration"
                ]["scale"] = 29.0
                manifest["entries"][1] = write_payload(path.parent, f"changed-{zero}", payload)
                atomic_json(Path(directory) / f"changed-{zero}.json", manifest)
            context.world.barrier()
            before = engine.export_state_dict(only_rank_zero=False)
            with pytest.raises(ValueError, match="differs across WORLD"):
                engine.load_checkpoint(Path(directory) / f"changed-{zero}.json")
            assert not engine._failed and engine.last_successful_update() == expected
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                assert torch.equal(value, before[name])
            assert engine.step([data]).updated
    finally:
        dist.destroy_process_group()


def test_successful_objective_provenance_real_dp2_all_zero_and_failed_commit(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-provenance-", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster-provenance-"):
            shutil.rmtree(directory)
