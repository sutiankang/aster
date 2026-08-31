from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from aster.models.cosmos3_audio import Cosmos3AudioConfig, Cosmos3AudioCodec
from aster.methods.cosmos3 import Cosmos3AudioAutoencoderObjective
from aster.training import Trainer, ParallelContext


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
        torch.manual_seed(686)
        config = Cosmos3AudioConfig(normalize_volume=False)
        initial = Cosmos3AudioCodec(config).state_dict()
        batches = [
            dict(sample=torch.randn(1, 2, 35) * 0.2),
            dict(sample=torch.randn(2, 2, 48) * 0.2),
        ]
        objective = Cosmos3AudioAutoencoderObjective(sample_posterior=False, kl_weight=0.001)
        for stage in range(4):
            dense, native = Cosmos3AudioCodec(config), Cosmos3AudioCodec(config)
            dense.load_state_dict(initial)
            native.load_state_dict(initial)
            factory = lambda p: torch.optim.SGD(p, lr=0.0002, momentum=0.9)
            optimizer = factory(dense.parameters())
            engine = Trainer(
                native,
                objective,
                zero_stage=stage,
                parallel=ParallelContext(),
                optimizer_factory=factory,
                max_grad_norm=None,
            )
            a, b = [objective(dense, batch).terms for batch in batches]
            loss = sum(
                x.weight * (x.numerator + y.numerator) / (x.denominator + y.denominator)
                for x, y in zip(a, b)
            )
            loss.backward()
            optimizer.step()
            actual = engine.step([batches[rank]])
            assert abs(actual.loss - loss.item()) < 1e-6
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(
                    value, dense.state_dict()[name], atol=2e-7, rtol=3e-5, msg=name
                )
            checkpoint = engine.save_checkpoint(Path(output) / f"audio_zero{stage}")
            expected = engine.step([batches[rank]])
            weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(checkpoint, trusted=True)
            actual = engine.step([batches[rank]])
            assert actual.loss == expected.loss
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, weights[name], atol=0, rtol=0)

        bad = {"sample": torch.randn(1, 1 if rank else 2, 48)}
        steps = engine.steps
        try:
            engine.step([bad])
            raise AssertionError("asymmetric invalid audio must fail before gathers")
        except ValueError as error:
            assert "waveform" in str(error)
        assert engine.steps == steps
    finally:
        dist.destroy_process_group()


def test_models_avae2_dp2_zero_all_variable_duration_resume_symmetric_preflight(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_audio_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_audio_"):
            shutil.rmtree(directory)
