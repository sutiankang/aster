from copy import deepcopy
import pytest
import torch
import torch.nn.functional as F

from aster.models import QwenMTPConfig, build_model
from aster.methods import MultiTokenPredictionObjective
from aster.training import Trainer


def _batch():
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 2], [1, 4, 6, 8, 10, 0, 0]])
    padding = ids.ne(0)
    labels = ids.clone()
    labels[:, :2] = -100
    mask = padding.clone()
    mask[0, 3] = False
    return dict(input_ids=ids, attention_mask=padding, labels=labels, loss_mask=mask)


def test_mtp_target_coordinates_mask_counts_and_independent_loss():
    torch.manual_seed(73)
    model = build_model(QwenMTPConfig(num_mtp_layers=2))
    objective = MultiTokenPredictionObjective(depth=2, base_weight=0.7, mtp_weight=0.3)
    batch = _batch()
    result = objective(model, batch)
    prediction = model(batch["input_ids"], attention_mask=batch["attention_mask"], mtp_depth=2)
    for offset, logits, term in zip(
        (0, 1, 2), (prediction.logits, *prediction.auxiliary["mtp_logits"]), result.terms
    ):
        mask = (batch["labels"].ne(-100) & batch["loss_mask"] & batch["attention_mask"])[
            :, offset + 1 :
        ]
        target = batch["labels"][:, offset + 1 :].masked_fill(~mask, 0)
        oracle = (
            F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                reduction="none",
            )
            .reshape_as(target)[mask]
            .sum()
        )
        torch.testing.assert_close(term.numerator, oracle)
        assert term.denominator.dtype == torch.int64 and int(term.denominator) == int(mask.sum())
        assert term.weight == (0.7 if offset == 0 else 0.15)
    sum(term.weight * term.mean for term in result.terms).backward()
    assert model.mtp.fc.weight.grad.abs().sum() > 0
    assert model.backbone.model.layers[0].linear_attn.in_proj_qkv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("stage", [0, 3])
def test_mtp_shared_parameter_training_accumulation_and_resume(stage, tmp_path):
    torch.manual_seed(75)
    model = build_model(QwenMTPConfig(num_mtp_layers=2))
    # SGD is a direct test of gradient/denominator equivalence. Adam's first-step
    # normalization can amplify GEMM rounding at almost-zero gradients; exact Adam
    # checkpoint continuation is tested separately below and in the model tests.
    factory = lambda p: torch.optim.SGD(p, lr=0.001, momentum=0.9)
    full = Trainer(
        deepcopy(model),
        MultiTokenPredictionObjective(depth=2),
        optimizer_factory=factory,
        max_grad_norm=None,
        zero_stage=stage,
    )
    split = Trainer(
        deepcopy(model),
        MultiTokenPredictionObjective(depth=2),
        optimizer_factory=factory,
        max_grad_norm=None,
        zero_stage=stage,
        accumulation_steps=2,
    )
    batch = _batch()
    shards = [
        {key: value[:1] for key, value in batch.items()},
        {key: value[1:] for key, value in batch.items()},
    ]
    full.step([batch])
    split.step(shards)
    for key, value in full.export_state_dict().items():
        torch.testing.assert_close(value, split.export_state_dict()[key], atol=4e-7, rtol=4e-5)
    split.save_checkpoint(tmp_path / "mtp")
    expected = split.step(shards)
    weights = split.export_state_dict()
    split.load_checkpoint(tmp_path / "mtp", trusted=True)
    actual = split.step(shards)
    assert expected.loss == actual.loss
    for key, value in split.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)


def test_mtp_formal_objective_adam_exact_checkpoint(tmp_path):
    torch.manual_seed(76)
    engine = Trainer(
        build_model(QwenMTPConfig(num_mtp_layers=2)),
        MultiTokenPredictionObjective(depth=2),
        lr=0.001,
    )
    batch = _batch()
    engine.step([batch])
    engine.save_checkpoint(tmp_path / "adam")
    expected = engine.step([batch])
    weights = engine.export_state_dict()
    engine.load_checkpoint(tmp_path / "adam", trusted=True)
    actual = engine.step([batch])
    assert expected.loss == actual.loss
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)


def test_mtp_rejects_ambiguous_padding_segments_and_configuration():
    model = build_model(QwenMTPConfig())
    objective = MultiTokenPredictionObjective()
    batch = _batch()
    batch["attention_mask"][0, 1] = False
    with pytest.raises(ValueError, match="right padding"):
        objective(model, batch)
    with pytest.raises(ValueError, match="unpacked"):
        objective(model, {**_batch(), "segment_ids": torch.zeros(2, 7)})
    with pytest.raises(ValueError, match="weights"):
        MultiTokenPredictionObjective(mtp_weight=float("nan"))
