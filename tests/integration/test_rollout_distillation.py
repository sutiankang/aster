import asyncio
import copy
import torch
from aster.core import digest_json, read_json
from aster.data import ByteTokenizer
from aster.models import LlamaConfig, build_model
from aster.training import Trainer
from aster.inference import SamplingConfig
from aster.methods.rollout_distillation import (
    OnPolicyDistillationMethod,
    sequence_distillation_examples,
    save_distillation_rollouts,
)


def test_native_rollout_teacher_score_student_update_checkpoint(tmp_path):
    torch.manual_seed(41)
    torch.set_num_threads(1)
    tokenizer = ByteTokenizer()
    fingerprint = digest_json(tokenizer.to_dict())
    config = LlamaConfig(
        vocab_size=259,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    student, teacher = build_model(config), build_model(config)
    original = copy.deepcopy(student.state_dict())
    teacher_before = copy.deepcopy(teacher.state_dict())
    engine = Trainer(student, lr=0.001, accumulation_steps=2)
    method = OnPolicyDistillationMethod(
        engine, teacher, tokenizer, teacher_tokenizer_fingerprint=fingerprint, max_prompt_tokens=16
    )

    settings = SamplingConfig(
        max_new_tokens=3, temperature=0.0, logit_bias=((ord("a") + 3, 100.0),)
    )
    result = asyncio.run(
        method.update([tokenizer.encode("Q "), tokenizer.encode("R ")], sampling=settings)
    )
    assert result.updated and method.updates == 1 and len(method.last_rollouts) == 2
    assert any(
        not torch.equal(value, student.state_dict()[name]) for name, value in original.items()
    )
    for name, value in teacher_before.items():
        torch.testing.assert_close(teacher.state_dict()[name], value)
    for row in method.last_rollouts:
        assert row.behavior_logprobs == (0.0, 0.0, 0.0)
        assert all(value < 0 for value in row.raw_model_logprobs)
    examples, receipts = sequence_distillation_examples(
        method.last_rollouts, ["Q ", "R "], tokenizer, accept_length=True
    )
    assert len(examples) == 2 and all(row["accepted"] for row in receipts)
    assert examples[0]["labels"][:2] == [-100, -100]
    save_distillation_rollouts(
        tmp_path / "rollouts.json",
        method.last_rollouts,
        tokenizer_fingerprint=fingerprint,
        dataset_fingerprint="test-prompts",
    )
    assert len(read_json(tmp_path / "rollouts.json")["payload"]["rollouts"]) == 2
    engine.save_checkpoint(tmp_path / "checkpoint")
    method.updates = 9
    engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    assert method.updates == 1
