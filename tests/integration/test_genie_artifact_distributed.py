from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.core import ArtifactStore
from aster.methods.genie_artifact_training import (
    BoundGenieWorldObjective,
    publish_genie_world,
    load_trained_genie,
)
from aster.training import ParallelContext, Trainer


def _worker(rank, rendezvous, directory, trace_id, world_id):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        context = ParallelContext()
        root = Path(directory)
        store = ArtifactStore(root / "store")

        with pytest.raises(RuntimeError, match="Load bound Genie"):
            BoundGenieWorldObjective(store, trace_id if rank == 0 else "0" * 64, parallel=context)
        with pytest.raises(RuntimeError, match="Load bound Genie"):
            BoundGenieWorldObjective(
                store, trace_id, commitment_cost=0.25 if rank == 0 else -1.0, parallel=context
            )
        objective = BoundGenieWorldObjective(store, trace_id, parallel=context)
        model, _ = load_trained_genie(store, world_id, world=True)
        engine = Trainer(
            model,
            objective,
            parallel=context,
            zero_stage=3,
            accumulation_steps=2,
            ema_decay=0.9,
            lr=0.001,
        )
        batch = objective.batch([rank])
        batches = [deepcopy(batch), deepcopy(batch)]
        assert engine.step(batches).updated
        published = publish_genie_world(engine, store, root / "dp_world", ema=True)
        assert len(set(context.world.gather_objects(published.id))) == 1
        checkpoint = engine.save_checkpoint(root / "dp_checkpoint")
        first = engine.step(batches)
        expected = deepcopy(engine.export_state_dict(only_rank_zero=False))
        engine.load_checkpoint(checkpoint, trusted=True)
        second = engine.step(batches)
        assert first.updated and second.updated and first.loss == second.loss
        for name, value in engine.export_state_dict(only_rank_zero=False).items():
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
        calls = []
        hook = engine.roles["model"].model.register_forward_pre_hook(lambda *_: calls.append(1))
        broken = deepcopy(batches)
        if rank == 1:
            broken[1]["tokens"][0, 0, 0] = (broken[1]["tokens"][0, 0, 0] + 1) % 5
        try:
            with pytest.raises((ValueError, RuntimeError), match="bound video/token"):
                engine.step(broken)
            assert not calls
        finally:
            hook.remove()
        for name, value in engine.export_state_dict(only_rank_zero=False).items():
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)

        engine.load_checkpoint(checkpoint, trusted=True)
        assert (
            publish_genie_world(engine, store, root / "dp_world_restored", ema=True).id
            == published.id
        )
    finally:
        dist.destroy_process_group()


def test_genie_actual_dp2_zero3_trace_preflight_collective_publication_and_resume(tmp_path):
    from test_genie_generation_evaluation import prepare

    store, _, _, _, trace, engine, _ = prepare(tmp_path)
    source = publish_genie_world(engine, store, tmp_path / "initial_world")
    base = Path(tempfile.gettempdir())
    if not str(base).isascii():
        base = Path("C:/Temp")
    rendezvous = Path(tempfile.mkdtemp(prefix="aster_genie_artifact_", dir=base)).resolve()
    try:
        mp.spawn(
            _worker,
            args=(str(rendezvous / "store"), str(tmp_path), trace.id, source.id),
            nprocs=2,
            join=True,
        )
    finally:
        if rendezvous.parent == base.resolve() and rendezvous.name.startswith(
            "aster_genie_artifact_"
        ):
            shutil.rmtree(rendezvous)
