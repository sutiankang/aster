from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch

from aster.models import (
    build_model,
    LlamaConfig,
    Qwen2Config,
    Qwen3Config,
    MistralConfig,
    MixtralConfig,
    DeepSeekV3Config,
)
from aster.methods import CrossEntropyObjective
from aster.training import Trainer
from aster.optimization.fused_attention import set_attention_backend, UnsupportedAttentionBackend


@pytest.mark.parametrize(
    "config",
    [
        LlamaConfig(),
        Qwen2Config(),
        Qwen3Config(),
        MistralConfig(),
        MixtralConfig(),
        DeepSeekV3Config(),
    ],
)
def test_pure_token_static_vocabulary_and_masks_are_checked_without_forward(config):
    model = build_model(config)
    objective = CrossEntropyObjective()
    batch = {"input_ids": torch.tensor([[1, 2, 3]])}
    assert objective.preflight_microbatches(model, [batch])[0] is batch
    cases = [
        {"input_ids": torch.tensor([[1, -1, 2]])},
        {"labels": torch.tensor([[1, 999999, 2]])},
        {"attention_mask": torch.tensor([[1, 2, 1]])},
        {"position_ids": torch.tensor([[0, 1, -1]])},
        {"position_ids": torch.zeros(1, 3)},
        {"loss_mask": torch.tensor([[1, 2, 1]])},
        {"state": None},
        {"use_cache": False},
    ]
    for change in cases:
        invalid = {**batch, **change}
        with pytest.raises(ValueError):
            objective.preflight_microbatches(model, [batch, invalid])


def test_embedding_input_nested_supervision_and_full_window_before_zero3_gather():
    model = build_model(LlamaConfig())
    objective = CrossEntropyObjective()
    ids = torch.tensor([[1, 2, 3]])
    embeds = torch.randn(1, 3, model.config.hidden_size)
    valid = {"model_inputs": {"inputs_embeds": embeds}, "labels": ids}
    assert objective.preflight_microbatches(model, [valid])[0] is valid
    with pytest.raises(ValueError, match="Labels"):
        objective.preflight_microbatches(model, [{"model_inputs": {"input_ids": ids}}])
    with pytest.raises(ValueError, match="mix"):
        objective.preflight_microbatches(
            model, [{"model_inputs": {"input_ids": ids}, "input_ids": ids}]
        )
    with pytest.raises(ValueError, match="media or cache"):
        objective.preflight_microbatches(
            model, [{"model_inputs": {"input_ids": ids, "past_key_values": None}, "labels": ids}]
        )
    called = []
    model.register_forward_pre_hook(lambda *_: called.append(True))
    trainer = Trainer(model, objective, zero_stage=3, accumulation_steps=2)
    invalid = {"input_ids": ids, "attention_mask": torch.tensor([[1, 2, 1]])}
    with pytest.raises(ValueError):
        trainer.step([{"input_ids": ids}, invalid])
    assert not called and trainer.steps == 0
    with pytest.raises(UnsupportedAttentionBackend, match="training-owned"):
        set_attention_backend(model)


def test_unknown_or_rich_model_is_not_guessed_as_dense_causal_layout():
    class Unknown(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=3)

        def forward(self, **kwargs):
            raise AssertionError("Preflight must never execute the model")

    batch = {
        "model_inputs": {
            "pixel_values": torch.randn(1, 3, 8, 8),
            "position_ids": torch.zeros(3, 1, 2, dtype=torch.long),
        }
    }
    assert CrossEntropyObjective().preflight_microbatches(Unknown(), [batch])[0] is batch

    native = build_model(LlamaConfig())
    assert CrossEntropyObjective(causal=False).preflight_microbatches(native, [batch])[0] is batch
