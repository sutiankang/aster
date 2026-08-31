from datetime import timedelta
from pathlib import Path
import shutil
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import build_model, LlamaConfig
from aster.methods import CrossEntropyObjective
from aster.optimization.fused_attention import set_attention_backend
from aster.training import Trainer, ParallelConfig, ParallelContext


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=100),
    )
    try:
        context = ParallelContext(ParallelConfig(data_parallel=2))
        config = LlamaConfig(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=24,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
        )

        def make(zero, fused):
            torch.manual_seed(635)
            model = build_model(config)
            if fused:
                set_attention_backend(model, query_block_size=3, key_block_size=2)
            return Trainer(
                model, CrossEntropyObjective(), lr=0.001, zero_stage=zero, parallel=context
            )

        local = (
            torch.tensor([[1, 2, 3, 4]])
            if rank == 0
            else torch.tensor([[2, 4, 6, 8, 10], [3, 5, 7, 9, 11]])
        )
        batch = {"input_ids": local}
        for zero in (0, 3):
            reference, actual = make(zero, False), make(zero, True)
            a, b = reference.step([batch]), actual.step([batch])
            assert a.updated and b.updated and abs(a.loss - b.loss) < 2e-6
            expected, got = (
                reference.export_state_dict(only_rank_zero=False),
                actual.export_state_dict(only_rank_zero=False),
            )
            for name in expected:
                torch.testing.assert_close(got[name], expected[name], atol=3e-5, rtol=3e-4)
            path = actual.save_checkpoint(Path(output) / f"fused-dp-{zero}.json")
            resumed = make(zero, True)
            resumed.load_checkpoint(path)
            actual.step([batch])
            resumed.step([batch])
            expected, got = (
                actual.export_state_dict(only_rank_zero=False),
                resumed.export_state_dict(only_rank_zero=False),
            )
            for name in expected:
                torch.testing.assert_close(got[name], expected[name], atol=0, rtol=0)
            for fused in (False, True):
                for corruption in ("mask", "labels", "positions", "ids"):
                    tested = make(zero, fused)
                    invalid = {
                        "input_ids": local.clone(),
                        "attention_mask": torch.ones_like(local),
                        "labels": local.clone(),
                        "position_ids": torch.arange(local.shape[1])[None]
                        .expand(local.shape[0], -1)
                        .clone(),
                    }
                    if rank == 1:
                        if corruption == "mask":
                            invalid["attention_mask"][0, 0] = 2
                        elif corruption == "labels":
                            invalid["labels"][0, 0] = 100
                        elif corruption == "positions":
                            invalid["position_ids"][0, 0] = -1
                        else:
                            invalid["input_ids"][0, 0] = 100

                    with pytest.raises(ValueError):
                        tested.step([invalid])

                    assert tested.steps == 0 and tested.step([batch]).updated
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_real_dp2_tiled_attention_update_resume_and_one_rank_invalid_mask(tmp_path):
    root = Path(tempfile.gettempdir()).resolve()
    if not str(root).isascii():
        root = Path("C:/Temp").resolve()
    directory = Path(tempfile.mkdtemp(prefix="aster-fused-attn-", dir=root)).resolve()
    assert directory.parent == root and directory.name.startswith("aster-fused-attn-")
    processes = None
    try:
        processes = mp.spawn(
            worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=False
        )
        deadline = time.monotonic() + 100
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("Attention DP regression exceeded bounded collective test time")
    finally:
        if processes is not None:
            for process in processes.processes:
                if process.is_alive():
                    process.terminate()
            for process in processes.processes:
                process.join(timeout=5)

        shutil.rmtree(directory)
