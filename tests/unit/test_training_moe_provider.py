from copy import deepcopy

import pytest
import torch

from aster.core import LossBundle
from aster.models import MixtralConfig, LlamaConfig, build_model
from aster.training import (
    Trainer,
    ParallelContext,
    parallelize_mixtral,
    ExpertParallelCrossEntropyObjective,
)


def configuration(**kwargs):
    values = dict(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        sliding_window=3,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    return MixtralConfig(**{**values, **kwargs})


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "zero,precision", [(0, "fp32"), (1, "fp32"), (2, "fp32"), (3, "fp32"), (0, "bf16"), (3, "bf16")]
)
def test_complete_mixtral_provider_zero_resume_standard_export(zero, precision, tmp_path):
    torch.manual_seed(341)
    context = ParallelContext()
    original = build_model(configuration(router_jitter_noise=0.05, tie_word_embeddings=True))
    model = parallelize_mixtral(original, context)
    objective = ExpertParallelCrossEntropyObjective(context, router_aux_coefficient=0.02)
    engine = Trainer(model, objective, parallel=context, zero_stage=zero, precision=precision)
    batch = {
        "input_ids": torch.tensor([[1, 3, 4, 6, 2], [1, 7, 9, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
    }
    first = engine.step([batch])
    assert first.updated
    assert first.terms["router_seq_aux"]["denominator"] == 2
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint.json")
    result = engine.step([batch])
    expected = engine.export_state_dict()
    engine.load_checkpoint(checkpoint)
    assert engine.step([batch]) == result
    actual = engine.export_state_dict()
    for name in actual:
        assert torch.equal(actual[name], expected[name])
    exported = build_model(original.config)
    exported.load_state_dict(actual, strict=True)
    assert exported.lm_head.weight is exported.model.embed_tokens.weight
    with pytest.raises(RuntimeError, match="collective"):
        model.save_pretrained(tmp_path / "bad")
    assert set(actual) == set(original.state_dict())


def test_seq_router_aux_is_explicit_and_accumulation_additive():
    torch.manual_seed(981)
    context = ParallelContext()
    model = build_model(configuration())
    replica = deepcopy(model)
    objective = ExpertParallelCrossEntropyObjective(context, router_aux_coefficient=0.2)
    ids = torch.tensor([[1, 3, 8, 4, 2], [1, 7, 4, 2, 0], [1, 8, 2, 0, 0]])
    mask = ids.ne(0)
    complete = objective(model, {"input_ids": ids, "attention_mask": mask})
    parts = [
        objective(replica, {"input_ids": ids[:1], "attention_mask": mask[:1]}),
        objective(replica, {"input_ids": ids[1:], "attention_mask": mask[1:]}),
    ]
    assert isinstance(complete, LossBundle)
    loss = sum(term.mean * term.weight for term in complete.terms)
    divided = sum(
        sum(part.terms[i].numerator for part in parts)
        / sum(part.terms[i].denominator for part in parts)
        * term.weight
        for i, term in enumerate(complete.terms)
    )
    loss.backward()
    divided.backward()
    for a, b in zip(model.parameters(), replica.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=3e-5)

    outputs = model(ids)
    expected = 0.0
    for layer in outputs.auxiliary["router"]:
        for row in range(len(ids)):
            logits = layer["logits"].reshape(3, 5, 4)[row, mask[row]]
            probabilities = logits.softmax(-1)
            selected = probabilities.topk(2, -1).indices
            frequency = torch.bincount(selected.flatten(), minlength=4).float() / (2 * len(logits))
            expected += 4 * (frequency * probabilities.mean(0)).sum()
    torch.testing.assert_close(complete.terms[1].numerator, expected)


def test_moe_provider_rejects_unsupported_family():
    with pytest.raises(ValueError, match="Mixtral"):
        parallelize_mixtral(build_model(LlamaConfig()), ParallelContext())


def test_empty_batch_is_rejected_before_forward():
    context = ParallelContext()
    model = parallelize_mixtral(build_model(configuration()), context)
    engine = Trainer(model, ExpertParallelCrossEntropyObjective(context), zero_stage=3)
    with pytest.raises(ValueError, match="B>=1"):
        engine.step([{"input_ids": torch.empty(0, 5, dtype=torch.long)}])
    from aster.training.sharding import zero3_units

    assert all(unit.gathers == 0 for unit in zero3_units(engine.model)) and not engine._failed
