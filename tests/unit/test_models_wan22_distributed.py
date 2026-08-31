from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from aster.models import Wan22VAEConfig, build_model
from aster.methods.cosmos3 import Wan22AutoencoderObjective
from aster.training import ParallelContext, Trainer


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
        torch.manual_seed(561)
        config = Wan22VAEConfig()
        initial = build_model(config).state_dict()
        batches = [dict(sample=torch.randn(i + 1, 3, 5, 16, 16 * (i + 1)) * 0.2) for i in range(2)]
        objective = Wan22AutoencoderObjective(
            sequence_length=5, sample_posterior=False, kl_weight=0.001
        )
        for stage in range(4):
            dense, native = build_model(config), build_model(config)
            dense.load_state_dict(initial)
            native.load_state_dict(initial)
            optimizer = torch.optim.SGD(dense.parameters(), lr=0.0001, momentum=0.9)
            engine = Trainer(
                native,
                objective,
                parallel=ParallelContext(),
                zero_stage=stage,
                max_grad_norm=None,
                optimizer_factory=lambda parameters: torch.optim.SGD(
                    parameters, lr=0.0001, momentum=0.9
                ),
            )
            left, right = objective(dense, batches[0]).terms, objective(dense, batches[1]).terms
            loss = sum(
                a.weight * (a.numerator + b.numerator) / (a.denominator + b.denominator)
                for a, b in zip(left, right)
            )
            loss.backward()
            optimizer.step()
            result = engine.step([batches[rank]])
            assert result.updated and abs(result.loss - loss.item()) < 2e-5
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(
                    value, dense.state_dict()[name], atol=3e-7, rtol=4e-5, msg=name
                )
            checkpoint = engine.save_checkpoint(Path(output) / f"wan22_zero{stage}")
            expected = engine.step([batches[rank]])
            weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(checkpoint, trusted=True)
            actual = engine.step([batches[rank]])
            assert actual.loss == expected.loss
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
        bad = dict(sample=torch.randn(1, 3, 9 if rank else 5, 16, 16))
        steps = engine.steps
        try:
            engine.step([bad])
            raise AssertionError("different time-call graphs must be rejected before gathers")
        except ValueError as error:
            assert "sequence_length" in str(error)
        assert engine.steps == steps
    finally:
        dist.destroy_process_group()


def test_models_wan22_dp2_all_zero_unequal_shapes_resume_and_symmetric_time_preflight(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_wan22_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_wan22_"):
            shutil.rmtree(directory)
