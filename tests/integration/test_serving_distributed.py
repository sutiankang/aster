from datetime import timedelta
from pathlib import Path
import tempfile
import shutil
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import build_model, LlamaConfig, Qwen2Config, Qwen3Config
from aster.training import ParallelConfig, ParallelContext
from aster.inference import (
    ParallelCausalPredictor,
    CollectiveGenerator,
    SamplingConfig,
    sample_token,
)
from aster.inference import load_hf_safetensors


def worker(rank, world, rendezvous, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=50),
    )
    try:
        config = {
            "tp_dp": ParallelConfig(tensor_parallel=2, data_parallel=2),
            "tp_pp": ParallelConfig(tensor_parallel=2, pipeline_parallel=2),
            "pp": ParallelConfig(pipeline_parallel=2),
        }[mode]
        context = ParallelContext(config)
        for family in (LlamaConfig, Qwen2Config, Qwen3Config):
            torch.manual_seed(617)
            dense = build_model(
                family(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=24,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                )
            ).eval()
            parallel = ParallelCausalPredictor(dense, context)
            prompt = [1, 3 + context.dp.rank, 8]
            ids = torch.tensor([prompt])
            with torch.no_grad():
                expected = dense(ids, use_cache=True)
                actual = parallel(ids, use_cache=True)
                torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
                assert len(actual.state.layers) == 2 // context.pp.size
                assert actual.state.layers[0][0].shape[1] == 2 // context.tp.size
                for token in (7, 9):
                    ids = torch.tensor([[token]])
                    expected = dense(ids, state=expected.state, use_cache=True)
                    actual = parallel(ids, state=actual.state, use_cache=True)
                    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
            generator = CollectiveGenerator(
                parallel, context, policy_artifact_id="unit-native", block_size=2, max_blocks=16
            )
            sampling = SamplingConfig(max_new_tokens=4, temperature=0.8, top_k=8, seed=9)
            result = generator.generate(prompt if context.tp_pp.rank == 0 else None, sampling)
            seed, output, raw = torch.Generator().manual_seed(9), [], []
            with torch.no_grad():
                for i in range(4):
                    logits = dense(torch.tensor([prompt + output])).logits[0, -1]
                    selected = sample_token(
                        logits, sampling, seed, context_ids=prompt + output, generated_count=i
                    )
                    output.append(selected.token_id)
                    raw.append(selected.raw_model_logprob)
            assert result.token_ids == tuple(output)
            torch.testing.assert_close(
                torch.tensor(result.raw_model_logprobs), torch.tensor(raw), atol=2e-6, rtol=2e-5
            )
            assert generator.runner.input_tokens_computed == len(prompt) + 3
            assert generator.runner.pool.used_blocks == 0
            events = []
            cancelled = generator.generate(
                prompt, sampling, on_token=events.append, cancelled=lambda: bool(events)
            )
            assert len(cancelled.token_ids) == 1 and cancelled.stop_reason == "cancelled"
            assert generator.runner.pool.used_blocks == 0
            with pytest.raises(ValueError, match="Invalid collective"):
                generator.generate([-1], sampling)
            with pytest.raises(ValueError, match="sampling failed"):
                generator.generate(
                    prompt, SamplingConfig(max_new_tokens=1, logit_bias=((99, 1.0),))
                )
            dist.barrier()

        from safetensors.torch import save_file
        from dataclasses import asdict
        import json

        directory = Path(rendezvous).parent / "checkpoint"
        if rank == 0:
            directory.mkdir()
            values = asdict(dense.config)
            values.pop("rope")
            values.update(model_type="qwen3", rope_theta=dense.config.rope.theta, hidden_act="silu")
            (directory / "config.json").write_text(json.dumps(values), encoding="utf-8")
            save_file(
                {key: value.clone() for key, value in dense.state_dict().items()},
                directory / "model.safetensors",
            )
        dist.barrier()
        loaded = load_hf_safetensors(directory, parallel=context)
        with torch.no_grad():
            ids = torch.tensor([[1, 6, 8]])
            torch.testing.assert_close(loaded(ids).logits, dense(ids).logits, atol=3e-6, rtol=3e-5)
        assert (
            loaded.load_report["loaded_slice_bytes"] < loaded.load_report["source_parameter_bytes"]
        )
        assert loaded.load_report["construction"] == "meta_then_parameter_slices"
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode,world", [("tp_dp", 4), ("tp_pp", 4), ("pp", 2)])
def test_native_parallel_inference_matches_full_reference(mode, world):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    directory = Path(tempfile.mkdtemp(prefix="aster-infer-gloo-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-infer-gloo-")
    try:
        mp.spawn(worker, args=(world, str(directory / "rdzv"), mode), nprocs=world, join=True)
    finally:
        shutil.rmtree(directory)
