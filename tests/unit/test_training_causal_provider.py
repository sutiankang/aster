from copy import deepcopy

import pytest
import torch

from aster.models import LlamaConfig, Qwen2Config, Qwen3Config, MistralConfig, build_model
from aster.methods import CrossEntropyObjective
from aster.training import (
    Trainer,
    ParallelContext,
    parallelize_causal_lm,
    TensorParallelCrossEntropyObjective,
)


def config(kind=LlamaConfig, **options):
    return kind(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        **options,
    )


def batch():
    ids = torch.tensor([[1, 3, 5, 7, 9, 2], [1, 4, 6, 0, 0, 0]])
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": ids.ne(0),
        "position_ids": torch.tensor([[2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5]]),
    }


@pytest.mark.parametrize("kind", [LlamaConfig, Qwen2Config, Qwen3Config])
@pytest.mark.parametrize("stage", [0, 3])
def test_full_provider_single_rank_dense_equivalence_tied_and_resume(kind, stage, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(52)
    dense = build_model(config(kind, tie_word_embeddings=True))
    original = deepcopy(dense.state_dict())
    context = ParallelContext()
    model = parallelize_causal_lm(dense, context)
    for key, value in dense.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    torch.testing.assert_close(
        model(batch()["input_ids"]).logits, dense(batch()["input_ids"]).logits
    )
    objective = TensorParallelCrossEntropyObjective(context)
    assert objective(model, batch()).denominator.dtype == torch.int64
    reference = Trainer(dense, CrossEntropyObjective(), lr=0.002)
    engine = Trainer(model, objective, lr=0.002, zero_stage=stage)
    reference.step([batch()])
    engine.step([batch()])
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, reference.export_state_dict()[key], rtol=2e-5, atol=1e-7)
    engine.save_checkpoint(tmp_path / "checkpoint")
    expected = engine.step([batch()])
    weights = engine.export_state_dict()
    engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    actual = engine.step([batch()])
    assert expected.loss == actual.loss
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    exported = build_model(dense.config)
    exported.load_state_dict(weights, strict=True)
    assert exported.lm_head.weight is exported.model.embed_tokens.weight
    with pytest.raises(RuntimeError, match="local TP shards"):
        model.save_pretrained(tmp_path / "bad")
    with pytest.raises(ValueError, match="inference cache"):
        model(batch()["input_ids"], use_cache=True)


@pytest.mark.parametrize(
    "change", ["input_range", "label_range", "position", "empty", "nested_ambiguous", "unknown"]
)
def test_full_provider_batch_errors_precede_zero3_mutation(change):
    context = ParallelContext()
    model = parallelize_causal_lm(build_model(config()), context)
    objective = TensorParallelCrossEntropyObjective(context)
    engine = Trainer(model, objective, zero_stage=3)
    data = batch()
    if change == "input_range":
        data["input_ids"][0, 1] = 17
    elif change == "label_range":
        data["labels"][0, 1] = 17
    elif change == "position":
        data["position_ids"][0, 1] = -1
    elif change == "empty":
        data = {key: value[:0] for key, value in data.items()}
    elif change == "nested_ambiguous":
        data["model_inputs"] = {"input_ids": data["input_ids"]}
    else:
        data["images"] = torch.ones(1)
    before = {name: value.clone() for name, value in engine.model.state_dict().items()}
    with pytest.raises(ValueError, match="preflight"):
        objective(model, data)
    for name, value in engine.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_provider_rejects_unsupported_family_before_modifying_source():
    model = build_model(config(MistralConfig))
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="exact dense"):
        parallelize_causal_lm(model, ParallelContext())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_nested_model_inputs_and_fully_masked_sample():
    context = ParallelContext()
    model = parallelize_causal_lm(build_model(config()), context)
    objective = TensorParallelCrossEntropyObjective(context)
    data = batch()
    labels = data.pop("labels")
    result = objective(
        model,
        {
            "model_inputs": data,
            "labels": labels,
            "loss_mask": torch.zeros_like(labels, dtype=torch.bool),
        },
    )
    assert result.numerator.item() == 0 and result.denominator.item() == 0


@pytest.mark.parametrize("constructor", ["parallelize", "pipeline"])
def test_attention_backend_installed_before_parallelization_is_rejected(constructor):
    from aster.training.causal_pipeline import CausalPipelineStage

    model = build_model(config())
    before = deepcopy(model.state_dict())
    model.model.layers[0].self_attn.attention_backend = object()
    with pytest.raises(ValueError, match="attention_backend"):
        if constructor == "parallelize":
            parallelize_causal_lm(model, ParallelContext())
        else:
            CausalPipelineStage(model, ParallelContext(), schedule="1f1b")
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
