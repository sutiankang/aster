import pytest
import torch
from torch import nn

from aster.models import MixtralConfig, LlamaConfig, build_model
from aster.training import ParallelConfig, ParallelContext, Trainer
from aster.training.parallel import vocab_parallel_cross_entropy
from aster.training.moe_tensor_parallel import (
    parallelize_mixtral_tensor,
    ExpertTensorParallelCrossEntropyObjective,
)
from aster.training.portable import logical_tensors, local_tensor, gather_tensor


def test_expert_axis_is_folded_not_world_multiplier():
    assert (
        ParallelConfig(
            tensor_parallel=2, data_parallel=4, expert_parallel=2, expert_tensor_parallel=2
        ).world_size
        == 8
    )
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            ParallelConfig(expert_tensor_parallel=bad)


@pytest.mark.parametrize("axis,stripes", [(0, 2), (1, 3)])
def test_stripe_codec_identity_and_rejection(axis, stripes):
    context = ParallelContext()
    model = nn.Linear(6, 4, bias=False)
    model.weight._aster_tp_dimension = axis
    model.weight._aster_tp_group = context.etp
    model.weight._aster_tp_stripes = stripes
    entry = logical_tensors(model, context)[0]
    assert torch.equal(
        local_tensor(gather_tensor(model.weight, entry, context), entry, context), model.weight
    )
    model.weight._aster_tp_stripes = 5
    with pytest.raises(ValueError, match="stripe"):
        logical_tensors(model, context)
    model.weight._aster_tp_stripes = 1
    del model.weight._aster_tp_dimension
    with pytest.raises(ValueError, match="explicit"):
        logical_tensors(model, context)


def test_provider_exact_model_and_actual_empty_mask_step(tmp_path):
    torch.set_num_threads(1)
    context = ParallelContext()
    with pytest.raises(ValueError, match="Mixtral"):
        parallelize_mixtral_tensor(
            build_model(
                LlamaConfig(
                    hidden_size=8,
                    intermediate_size=12,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                )
            ),
            context,
        )
    cfg = MixtralConfig(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    model = parallelize_mixtral_tensor(build_model(cfg), context)
    engine = Trainer(
        model,
        ExpertTensorParallelCrossEntropyObjective(context, router_aux_coefficient=0.02),
        zero_stage=3,
    )
    batch = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.zeros(1, 5, dtype=torch.long),
    }
    result = engine.step([batch])
    assert not result.updated and result.loss == 0 and not result.overflow
    assert engine.last_successful_update() is None
    dense = build_model(cfg)
    dense.load_state_dict(engine.export_state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="collective"):
        model.save_pretrained(tmp_path / "invalid")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_vocab_ce_uses_standard_low_precision_accumulation(dtype):

    logits = torch.tensor(
        [[12.0, 11.7, -19.0, 0.2, -0.3], [1.0, -2.0, 3.0, 2.0, 7.0]],
        dtype=dtype,
        requires_grad=True,
    )
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([1, -100])
    actual = vocab_parallel_cross_entropy(logits, labels, ParallelContext().tp)
    expected = torch.nn.functional.cross_entropy(
        reference.float() if dtype in {torch.float16, torch.bfloat16} else reference,
        labels,
        reduction="none",
        ignore_index=-100,
    )
    torch.testing.assert_close(
        actual, expected, atol=1e-7 if dtype == torch.float64 else 2e-6, rtol=1e-6
    )
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        logits.grad, reference.grad, atol=1e-7 if dtype == torch.float64 else 2e-6, rtol=1e-6
    )
