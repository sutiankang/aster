"""LeWM nonlinear global statistics: real DP2 and every ZeRO stage."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from aster.models.lewm import LeWorldModel
from aster.methods.lewm import LeWMMethod, LeWMObjective
from aster.training import Trainer, ParallelContext


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=150),
    )
    try:
        torch.manual_seed(736)
        data = dict(pixels=torch.randn(5, 4, 3, 16, 16), actions=torch.randn(5, 3, 2))
        initial = LeWorldModel().state_dict()
        section = slice(0, 2) if rank == 0 else slice(2, 5)
        local = {key: value[section] for key, value in data.items()}
        chunks = [
            {key: value[:1] for key, value in local.items()},
            {key: value[1:] for key, value in local.items()},
        ]
        for stage in range(4):
            dense, model = LeWorldModel(), LeWorldModel()
            dense.load_state_dict(initial)
            model.load_state_dict(initial)

            def create(model):
                engine = Trainer(
                    model,
                    parallel=ParallelContext(),
                    zero_stage=stage,
                    max_grad_norm=None,
                    optimizer_factory=lambda parameters: torch.optim.SGD(
                        parameters, lr=0.001, momentum=0.9
                    ),
                )
                return engine, LeWMMethod(engine, objective=LeWMObjective(num_proj=32), seed=737)

            engine, method = create(model)

            rng = torch.Generator().manual_seed(737)
            projection = torch.randn(32, 32, generator=rng)
            projection /= projection.norm(dim=0)
            drop_seed = int(torch.randint(2**31, (), generator=rng))
            optimizer = torch.optim.SGD(dense.parameters(), lr=0.001, momentum=0.9)
            with torch.random.fork_rng():
                torch.manual_seed(drop_seed)
                bundle = LeWMObjective(num_proj=32)(dense, dict(**data, projections=projection))
                loss = sum(term.weight * term.mean for term in bundle.terms)
                loss.backward()
                optimizer.step()
            result = method.update(chunks)
            assert result.updated and abs(result.loss - loss.item()) < 2e-6
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(
                    value, dense.state_dict()[name], atol=5e-7, rtol=6e-5, msg=name
                )
            if stage == 3:
                checkpoint = engine.save_checkpoint(Path(output) / "lewm_zero3")
                expected = method.update(chunks)
                weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
                fresh, fresh_method = create(LeWorldModel())
                fresh.load_checkpoint(checkpoint, trusted=True)
                actual = fresh_method.update(chunks)
                assert actual.loss == expected.loss
                for name, value in fresh.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(value, weights[name], atol=0, rtol=0, msg=name)
        forwards = []
        handle = engine.model.register_forward_pre_hook(lambda *args: forwards.append(True))
        try:
            with pytest.raises(ValueError, match="LeWMMethod"):
                engine.phase(
                    "invalid_local_statistics",
                    microbatches=[local],
                    objective=LeWMObjective(num_proj=32),
                )
            bad = {key: value.clone() for key, value in local.items()}
            if rank == 1:
                bad["actions"][0, 0, 0] = float("nan")
            steps = engine.steps
            with pytest.raises(ValueError, match="finite"):
                method.update([bad])
            assert engine.steps == steps and not forwards and not method.incomplete
        finally:
            handle.remove()
    finally:
        dist.destroy_process_group()


def test_models_lewm_real_dp2_all_zero_global_bn_sigreg_fresh_resume_preflight(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_lewm_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_lewm_"):
            shutil.rmtree(directory)
