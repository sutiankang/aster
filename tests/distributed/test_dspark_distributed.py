from copy import deepcopy
from datetime import timedelta
import importlib.util
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.methods.dspark import DSparkMethod
from aster.training import ParallelContext, Trainer


def _worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "dspark_formula", Path(__file__).parents[1] / "unit/test_dspark_training.py"
        )
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        context = ParallelContext()
        for stage in range(4):
            model, batches = oracle.objects()
            dense = deepcopy(model)

            if rank == 1:
                for batch in batches:
                    batch["loss_mask"].zero_()
                    batch["block_keep_mask"].zero_()
            all_batches = context.dp.gather_objects(batches)
            full = {
                key: torch.cat([batch[key] for rows in all_batches for batch in rows])
                for key in batches[0]
            }
            factory = lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)
            optimizer = factory(dense.parameters())
            expected = oracle.independent_loss(dense(**full))
            expected.backward()
            optimizer.step()
            batches = [
                dict(
                    batch,
                    teacher_identity=model.teacher_identity,
                    vocabulary_fingerprint="unit_vocab23",
                )
                for batch in batches
            ]
            engine = Trainer(
                model,
                parallel=context,
                zero_stage=stage,
                accumulation_steps=2,
                optimizer_factory=factory,
                max_grad_norm=None,
            )
            method = DSparkMethod(
                engine, vocabulary_fingerprint="unit_vocab23", normalization_profile="global_window"
            )
            result = method.update(batches)
            assert result.updated and abs(result.loss - float(expected.detach())) < 8e-7
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, dense.state_dict()[name], atol=1e-7, rtol=3e-5)
            checkpoint = engine.save_checkpoint(Path(directory) / f"zero{stage}")
            expected = method.update(batches)
            weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            fresh, _ = oracle.objects()
            other = Trainer(
                fresh,
                parallel=context,
                zero_stage=stage,
                accumulation_steps=2,
                optimizer_factory=factory,
                max_grad_norm=None,
            )
            restored = DSparkMethod(
                other, vocabulary_fingerprint="unit_vocab23", normalization_profile="global_window"
            )
            other.load_checkpoint(checkpoint, trusted=True)
            actual = restored.update(batches)
            assert actual.loss == expected.loss and restored.updates == 2
            for name, value in other.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
            bad = deepcopy(batches)
            if rank == 1:
                bad[-1]["teacher_identity"] = "0" * 64
            calls = []
            handle = fresh.fc.register_forward_pre_hook(lambda *_: calls.append(1))
            rejected = False
            try:
                restored.update(bad)
            except ValueError as error:
                rejected = "another teacher" in str(error)
            finally:
                handle.remove()
            assert rejected and not calls and not other._failed
    finally:
        dist.destroy_process_group()


def test_dspark_true_dp2_all_zero_empty_rank_global_weighting_checkpoint_and_symmetric_preflight(
    tmp_path,
):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_dspark_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_dspark_"):
            shutil.rmtree(directory)
