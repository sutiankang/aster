from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.methods.conservative import CQLPolicyConfig, CQLPolicy, CQLTwinQ, CQLMethod
from aster.training import Trainer, ParallelContext


def _build(context, stage):
    torch.manual_seed(301)
    engine = Trainer(
        CQLPolicy(CQLPolicyConfig(3, 2, hidden=8)),
        parallel=context,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.001),
        zero_stage=stage,
        max_grad_norm=None,
    )
    method = CQLMethod(
        engine, CQLTwinQ(3, 2, 8), num_random=3, lagrange=True, deterministic_backup=False
    )
    return engine, method


def _export(engine):
    return {
        name: engine.export_state_dict(role=name, only_rank_zero=False) for name in engine.roles
    }


def _equal(left, right, *, exact=False):
    for role in left:
        for name in left[role]:
            torch.testing.assert_close(
                left[role][name],
                right[role][name],
                atol=0 if exact else 4e-7,
                rtol=0 if exact else 4e-5,
            )


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
        context, expected = ParallelContext(), None
        for stage in range(4):
            engine, method = _build(context, stage)
            torch.manual_seed(901 + rank)
            size = rank + 2
            batch = dict(
                observations=torch.randn(size, 3),
                next_observations=torch.randn(size, 3),
                actions=torch.randn(size, 2).tanh(),
                rewards=torch.randn(size),
                terminated=torch.arange(size) == 1,
                truncated=torch.arange(size) == 0,
            )
            before = _export(engine)
            invalid = {**batch, "actions": batch["actions"].clone()}
            if rank == 1:
                invalid["actions"][0, 0] = float("nan")
            with pytest.raises(ValueError, match="preflight"):
                method.update([invalid])
            _equal(_export(engine), before, exact=True)
            assert not method._incomplete and engine.steps == 0
            assert all(value.updated for value in method.update([batch]).values())
            checkpoint = engine.save_checkpoint(Path(output) / f"zero{stage}")
            result = method.update([batch])
            updated = _export(engine)
            if expected is None:
                expected = updated
            else:
                _equal(updated, expected)
            engine.load_checkpoint(checkpoint, trusted=True)
            replayed = method.update([batch])
            assert {k: v.loss for k, v in result.items()} == {
                k: v.loss for k, v in replayed.items()
            }
            _equal(_export(engine), updated, exact=True)
        # Diverging role declarations fail before alpha/dual broadcasts or any updates.
        torch.manual_seed(51)
        engine = Trainer(CQLPolicy(CQLPolicyConfig(3, 2, hidden=8)), parallel=context)
        with pytest.raises(ValueError, match="identical method settings"):
            CQLMethod(engine, CQLTwinQ(3, 2, 8), lagrange=bool(rank))
    finally:
        dist.destroy_process_group()


def test_cql_real_dp2_all_zero_and_symmetric_preflight(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_cql_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_cql_"):
            shutil.rmtree(directory)
