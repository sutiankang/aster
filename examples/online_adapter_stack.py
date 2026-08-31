import argparse
import asyncio
import copy
import json
import torch
from aster.models import LlamaConfig, build_model
from aster.training import Trainer
from aster.methods.supervised import CrossEntropyObjective
from aster.methods.distillation import inject_lora
from aster.methods.rollout_distillation import tensor_state_identity
from aster.inference import (
    PagedAttentionRunner,
    KVQuantization,
    MultiLoRARunner,
    PagedStateArchive,
    InferenceEngine,
    SamplingConfig,
)


async def run(format="int8"):
    torch.manual_seed(24)
    torch.set_num_threads(1)
    base = build_model(
        LlamaConfig(
            vocab_size=24,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
    )

    base_id = tensor_state_identity(base.state_dict(), base.config.to_dict())
    trained = inject_lora(copy.deepcopy(base), targets=["lm_head"], rank=4, alpha=8.0)
    trainer = Trainer(trained, CrossEntropyObjective(), lr=0.01)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[1, 2, 3, 4, 5]]),
    }
    for _ in range(3):
        trainer.step([batch])
    paged = PagedAttentionRunner(
        base,
        policy_artifact_id=base_id,
        backend="torch_online_paged",
        block_size=3,
        max_blocks=4,
        kv_quantization=KVQuantization(format),
    )
    runner = MultiLoRARunner(paged)
    adapter = runner.register_trained_adapter(trainer.model, base_artifact_id=base_id)
    archive = PagedStateArchive(runner.pool, max_bytes=1024**2)
    engine = InferenceEngine(
        runner, max_active=2, max_batch_tokens=6, prefill_chunk_size=3, offload_archive=archive
    )
    try:
        handles = [
            await engine.submit(
                [1, 2, 3, 4, 5],
                SamplingConfig(max_new_tokens=5, temperature=0),
                identity=runner.resolve_model_identity(model_id),
            )
            for model_id in (base_id, adapter)
        ]
        results = await asyncio.gather(*(handle.collect() for handle in handles))
        if any(result.stop_reason != "length" for result in results):
            raise RuntimeError("Workflow did not finish")
        observation = engine.observation()
    finally:
        await engine.close()
    runner.remove_adapter(adapter)
    return {
        "evidence": "tiny_native_workflow_not_quality_or_gpu_speed",
        "kv_format": format,
        "requests": [
            {
                "adapter": result.adapter_id,
                "tokens": result.token_ids,
                "preemptions": result.preemption_count,
            }
            for result in results
        ],
        "offload": observation["offload"],
        "remaining_pages": runner.pool.used_blocks,
        "remaining_host_bytes": archive.stored_bytes,
        "remaining_adapter_bytes": runner.resident_bytes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv", default="int8", choices=["int8", "fp8_e4m3fn", "fp8_e5m2"])
    print(json.dumps(asyncio.run(run(parser.parse_args().kv)), ensure_ascii=False, indent=2))
