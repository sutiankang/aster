import asyncio
from copy import deepcopy
import torch
import pytest

from aster.models import build_model, LlamaConfig
from aster.methods.distillation import DistillationObjective
from aster.training import Trainer
from aster.optimization import prune_mlp, prepare_qat, configure_qat, convert_qat
from aster.inference import (
    save_optimized_model,
    load_optimized_model,
    ModelRunner,
    InferenceEngine,
    SamplingConfig,
)


def test_prune_kd_qat_checkpoint_packed_reload_and_cached_inference(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(38)
        teacher = build_model(
            LlamaConfig(
                vocab_size=24,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        ).eval()
        with torch.no_grad():
            for p in teacher.model.layers[0].mlp.parameters():
                p.mul_(5)
        pruned = prune_mlp(teacher, intermediate_size=16, parent_artifact_id="teacher-fixture-v1")
        student = pruned.model.train()
        ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
        batch = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
        objective = DistillationObjective(
            teacher, kd_weight=1.0, tokenizer_fingerprints=("token-fixture", "token-fixture")
        )

        def loss(model):
            with torch.no_grad():
                term = objective(model, batch)
                return float(term.numerator / term.denominator)

        baseline = loss(student)
        trainer = Trainer(student, objective, lr=0.002, max_grad_norm=1.0)
        for _ in range(25):
            assert trainer.step([batch]).updated
        assert loss(student) < baseline

        targets = ["model.layers.0.mlp." + name for name in ("gate_proj", "up_proj", "down_proj")]
        qat = prepare_qat(student, targets=targets, group_size=8)
        configure_qat(qat, observe=False)
        qat_trainer = Trainer(qat, objective, lr=0.0005, max_grad_norm=1.0)
        before = qat.get_submodule(targets[0]).weight.detach().clone()
        for _ in range(4):
            assert qat_trainer.step([batch]).updated
        assert not torch.equal(before, qat.get_submodule(targets[0]).weight)
        checkpoint = qat_trainer.save_checkpoint(tmp_path / "qat-checkpoint.json")
        incompatible = prepare_qat(student, targets=targets, bits=8, group_size=8)
        with pytest.raises(ValueError):
            Trainer(incompatible, objective, lr=0.0005, max_grad_norm=1.0).load_checkpoint(
                checkpoint
            )
        expected_result = qat_trainer.step([batch])
        expected_state = deepcopy(qat.state_dict())
        restored = prepare_qat(student, targets=targets, group_size=8)
        restored_trainer = Trainer(restored, objective, lr=0.0005, max_grad_norm=1.0)
        restored_trainer.load_checkpoint(checkpoint)
        assert restored_trainer.step([batch]) == expected_result
        for name, tensor in restored.state_dict().items():
            torch.testing.assert_close(tensor, expected_state[name], rtol=0, atol=0)
        packed = convert_qat(restored)
        save_optimized_model(
            packed,
            tmp_path / "deployment",
            base_artifact_id="pruned-qat-fixture-v2",
            transformation_metadata={"pruning": pruned.manifest, "qat": "native_weight_only"},
        )
        loaded = load_optimized_model(tmp_path / "deployment")
        with torch.no_grad():
            torch.testing.assert_close(
                restored(ids).logits, loaded(ids).logits, atol=1e-6, rtol=1e-5
            )

        async def run():
            engine = InferenceEngine(
                ModelRunner(
                    loaded, policy_artifact_id="packed-fixture-v3", block_size=2, max_blocks=16
                )
            )
            try:
                handle = await engine.submit(
                    [1, 3], SamplingConfig(max_new_tokens=3, temperature=0)
                )
                result = await handle.collect()
                assert len(result.token_ids) == 3 and len(result.raw_model_logprobs) == 3
            finally:
                await engine.close()

        asyncio.run(run())
    finally:
        torch.set_num_threads(previous)
