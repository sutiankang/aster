from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import LlamaConfig, build_model
from aster.methods import CrossEntropyObjective
from aster.training import (
    Trainer,
    ParallelContext,
    ParallelConfig,
    parallelize_causal_lm,
    TensorParallelCrossEntropyObjective,
    CausalPipelineCrossEntropyObjective,
    apply_runtime_state,
)
from aster.training.runtime_state import runtime_buffers


def worker(rank, rendezvous, directory, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=100),
    )
    try:
        grid = (
            ParallelConfig(data_parallel=2)
            if mode == "dp"
            else (
                ParallelConfig(tensor_parallel=2)
                if mode == "tp"
                else ParallelConfig(pipeline_parallel=2)
            )
        )
        context = ParallelContext(grid)
        config = LlamaConfig(
            vocab_size=17,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=True,
        )

        def make(zero, rounded):
            torch.manual_seed(973)
            model = build_model(config)
            if rounded:
                model.bfloat16().float()
            if mode != "dp":
                model = parallelize_causal_lm(model, context)
            objective = (
                CrossEntropyObjective()
                if mode == "dp"
                else (
                    TensorParallelCrossEntropyObjective(context)
                    if mode == "tp"
                    else CausalPipelineCrossEntropyObjective(context)
                )
            )
            return Trainer(
                model,
                objective,
                parallel=context,
                zero_stage=zero,
                ema_decay=0.8,
                accumulation_steps=2,
                optimizer_factory=lambda p: torch.optim.AdamW(p, lr=0.001),
                offload_optimizer="cpu",
            )

        ids = torch.tensor([[1, 5, 8, 9], [4, 2, 11, 6]])
        local = ids[:1] if mode == "dp" and rank == 0 else ids
        batches = [
            {"input_ids": local, "labels": local},
            {"input_ids": local[:, :3], "labels": local[:, :3]},
        ]
        for zero in (0, 1, 2, 3) if mode == "dp" else (3,):
            source = make(zero, True)
            source.step(batches)
            native = source.save_checkpoint(Path(directory) / f"{mode}-{zero}")
            portable = source.save_portable_checkpoint(Path(directory) / f"{mode}-{zero}-portable")
            state = source.export_runtime_state(only_rank_zero=False)
            fresh = make(zero, False)
            fresh.load_checkpoint(native)
            actual = fresh.export_runtime_state(only_rank_zero=False)
            assert actual["semantic_buffers"].keys() == state["semantic_buffers"].keys()
            for name, value in actual["semantic_buffers"].items():
                assert torch.equal(value, state["semantic_buffers"][name])
            source.step(batches)
            fresh.step(batches)
            expected, got = (
                source.export_state_dict(only_rank_zero=False),
                fresh.export_state_dict(only_rank_zero=False),
            )
            assert all(torch.equal(value, got[name]) for name, value in expected.items())
            fresh.load_portable_checkpoint(portable, seed=9)
            fresh.step(batches)
            assert all(
                torch.equal(value, fresh.export_state_dict(only_rank_zero=False)[name])
                for name, value in expected.items()
            )
            independent = build_model(config)
            independent.load_state_dict(expected)
            apply_runtime_state(independent, state)
            reference = build_model(config).bfloat16().float()
            reference.load_state_dict(expected)
            assert torch.equal(independent(ids).logits, reference(ids).logits)
            if mode != "pp":
                value = next(iter(runtime_buffers(fresh.model).values()))
                if rank:
                    value.add_(1.0)
                with pytest.raises(ValueError, match="replicas disagree"):
                    fresh.export_runtime_state()
                fresh.load_checkpoint(native)
                if rank:
                    next(iter(runtime_buffers(fresh.model).values())).fill_(float("nan"))
                with pytest.raises(ValueError, match="finite"):
                    fresh.export_runtime_state()
                fresh.load_checkpoint(native)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["dp", "tp", "pp"])
def test_native_semantic_runtime_fresh_instance_and_collective_export(mode, tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-runtime-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-runtime-")
    try:
        mp.spawn(worker, args=(str(directory / "store"), str(tmp_path), mode), nprocs=2, join=True)
    finally:
        shutil.rmtree(directory)
