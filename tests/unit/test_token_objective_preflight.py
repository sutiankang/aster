from copy import deepcopy

import pytest
import torch

from aster.models import build_model, LlamaConfig
from aster.methods import CrossEntropyObjective, DistillationObjective, PreferenceObjective
from aster.methods.reinforcement import GRPOObjective
from aster.methods.supervised import (
    preflight_causal_microbatches,
    native_causal_config,
    supervision_mask,
    sequence_logprobs,
)
from aster.training import Trainer


def model(**changes):
    torch.set_num_threads(1)
    torch.manual_seed(463)
    config = dict(
        vocab_size=23,
        hidden_size=16,
        intermediate_size=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=64,
    )
    return build_model(LlamaConfig(**{**config, **changes}))


def tokens(length=5, rows=1):
    ids = torch.arange(1, length + 1)[None].expand(rows, -1).clone()
    labels = ids.clone()
    labels[:, 0] = -100
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(length)[None].expand(rows, -1).clone(),
    }


def grpo_batch(policy, length=5, rows=1):
    batch = tokens(length, rows)
    with torch.no_grad():
        logp, _ = sequence_logprobs(policy, batch)
    return {
        **batch,
        "old_behavior_log_probs": logp.clone(),
        "reference_log_probs": logp.clone(),
        "advantages": torch.ones(rows),
    }


def pair(length=5, rows=1):
    chosen, rejected = tokens(length, rows), tokens(length + 1, rows)
    rejected["input_ids"][:, 1:3] = torch.tensor([3, 2])
    rejected["labels"][:, 1:3] = torch.tensor([3, 2])
    return {"chosen": chosen, "rejected": rejected}


@pytest.mark.parametrize("kind", ["forward_kl", "reverse_kl", "mixed_kl", "js"])
def test_distillation_checks_both_model_configs_and_feature_indices_without_forward(kind):
    student = model(max_position_embeddings=32)
    teacher = model(max_position_embeddings=5)
    calls = []
    student.register_forward_pre_hook(lambda *_: calls.append("student"))
    teacher.register_forward_pre_hook(lambda *_: calls.append("teacher"))
    objective = DistillationObjective(teacher, kind=kind)
    first, later = tokens(4), tokens(6)
    with pytest.raises(ValueError, match="length"):
        objective.preflight_microbatches(student, [first, later])
    assert not calls
    bounded_teacher = model(vocab_size=7)
    objective = DistillationObjective(bounded_teacher, kind=kind, kd_weight=0.0)
    with pytest.raises(ValueError, match="input_ids"):
        objective.preflight_microbatches(student, [tokens(8)])
    with pytest.raises(ValueError, match="vocabulary dimensions"):
        objective.preflight_microbatches(student, [first])
    objective = DistillationObjective(
        model(num_hidden_layers=1), kind=kind, feature_weight=0.1, layer_pairs=((1, 2),)
    )
    with pytest.raises(ValueError, match="layer index"):
        objective.preflight_microbatches(student, [first])
    objective = DistillationObjective(
        model(hidden_size=24),
        kind=kind,
        feature_weight=0.1,
        layer_pairs=((-1, -1),),
        feature_kind="relation",
    )
    assert objective.preflight_microbatches(student, [first])[0] is first
    objective.feature_kind = "mse"
    with pytest.raises(ValueError, match="equal hidden"):
        objective.preflight_microbatches(student, [first])


@pytest.mark.parametrize("method", ["dpo", "ipo", "simpo"])
def test_preference_entire_pair_graph_and_only_actual_reference_branch(method):
    policy, reference = model(), model(max_position_embeddings=5)
    objective = PreferenceObjective(reference, method=method)
    data = pair(5)
    if method == "simpo":
        assert objective.preflight_microbatches(policy, [data])[0] is data
        reference.forward = lambda **_: (_ for _ in ()).throw(
            AssertionError("SimPO never calls reference")
        )
        assert torch.isfinite(objective(policy, data).mean)
    else:
        with pytest.raises(ValueError, match="length"):
            objective.preflight_microbatches(policy, [data])
    objective = PreferenceObjective(model(), method=method)
    bad = pair(5)
    bad["rejected"]["labels"].fill_(-100)
    with pytest.raises(ValueError, match="supervised"):
        objective.preflight_microbatches(policy, [pair(4), bad])
    bad = {"chosen": tokens(rows=2), "rejected": tokens(rows=1)}
    with pytest.raises(ValueError, match="pairs must align"):
        objective.preflight_microbatches(policy, [bad])


@pytest.mark.parametrize("reduction", ["sequence", "token", "constant"])
def test_grpo_actual_next_token_trajectory_and_denominator_preflight(reduction):
    policy = model()
    objective = GRPOObjective(
        reduction=reduction, max_completion_length=6 if reduction == "constant" else None
    )
    batch = grpo_batch(policy)
    assert objective.preflight_microbatches(policy, [batch])[0] is batch
    cases = [
        ("old_behavior_log_probs", torch.ones(1, 5)),
        ("reference_log_probs", torch.full((1, 4), float("nan"))),
        ("advantages", torch.ones(1, 1)),
        ("advantages", torch.tensor([float("inf")])),
    ]
    for name, value in cases:
        with pytest.raises(ValueError, match=name):
            objective.preflight_microbatches(policy, [batch, {**batch, name: value}])
    empty = deepcopy(batch)
    empty["labels"].fill_(-100)
    with pytest.raises(ValueError, match="empty completions"):
        objective.preflight_microbatches(policy, [empty])
    if reduction == "constant":
        with pytest.raises(ValueError, match="fixed"):
            objective.preflight_microbatches(policy, [grpo_batch(policy, length=8)])


def test_helper_preserves_ce_supervision_and_nested_label_source():
    policy = model()
    batch = tokens(rows=2)
    batch["loss_mask"] = torch.ones_like(batch["input_ids"])
    batch["loss_mask"][0, 2] = 0
    batch["attention_mask"][1, -1] = 0
    prepared = preflight_causal_microbatches(policy, [batch])
    assert prepared[0] is batch and native_causal_config(policy) is policy.config
    valid = supervision_mask(batch, batch["labels"])[:, 1:]
    assert CrossEntropyObjective()(policy, batch).denominator == valid.sum()
    with pytest.raises(ValueError, match="Labels"):
        preflight_causal_microbatches(policy, [{"model_inputs": {"input_ids": batch["input_ids"]}}])


@pytest.mark.parametrize("kind", ["mse", "cosine", "relation"])
def test_native_feature_kd_nested_mask_matches_top_level_and_padding_gradient_zero(kind):
    student, teacher = model(), model()
    with torch.no_grad():
        teacher.model.layers[0].mlp.down_proj.weight.add_(0.02)
    objective = DistillationObjective(
        teacher, feature_weight=0.4, feature_kind=kind, layer_pairs=((1, 1), (-1, -1))
    )
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    labels = tokens(5, 2)["labels"].masked_fill(mask == 0, -100)
    embeddings = torch.randn(2, 5, student.config.hidden_size)

    def evaluate(nested, perturb_padding=False):
        values = embeddings.clone()
        if perturb_padding:
            values[mask == 0] = 11 * values[mask == 0] + 3
        values.requires_grad_(True)
        inputs = {"inputs_embeds": values, "attention_mask": mask}
        data = (
            {"model_inputs": inputs, "labels": labels} if nested else {**inputs, "labels": labels}
        )
        objective.preflight_microbatches(student, [data])
        terms = objective(student, data)

        feature_loss = sum(term.mean * term.weight for term in terms.terms[1:])
        gradient = torch.autograd.grad(feature_loss, values)[0]
        return feature_loss.detach(), gradient

    direct, direct_gradient = evaluate(False)
    nested, nested_gradient = evaluate(True)
    changed, _ = evaluate(True, True)
    torch.testing.assert_close(nested, direct, atol=0, rtol=0)
    torch.testing.assert_close(nested_gradient, direct_gradient, atol=0, rtol=0)
    torch.testing.assert_close(changed, direct, atol=0, rtol=0)
    assert nested_gradient[mask == 0].count_nonzero() == 0
    assert nested_gradient[mask == 1].abs().sum() > 0


def test_native_feature_kd_explicit_none_padding_equals_omitted_padding():
    student, teacher = model(), model()
    objective = DistillationObjective(teacher, feature_weight=0.2, layer_pairs=((1, 1),))
    batch = tokens(4)
    batch.pop("attention_mask")
    omitted = objective(student, batch)
    explicit = {
        "model_inputs": {"input_ids": batch["input_ids"], "attention_mask": None},
        "labels": batch["labels"],
    }
    objective.preflight_microbatches(student, [explicit])
    actual = objective(student, explicit)
    for left, right in zip(omitted.terms, actual.terms):
        torch.testing.assert_close(left.numerator, right.numerator, atol=0, rtol=0)
        torch.testing.assert_close(left.denominator, right.denominator, atol=0, rtol=0)


@pytest.mark.parametrize("method", ["dpo", "ipo", "simpo"])
def test_preference_reference_stop_gradient_does_not_detach_policy(method):
    policy, reference = model(), model()
    objective = PreferenceObjective(reference, method=method, beta=0.2)
    batch = pair(5)
    chosen, cmask = sequence_logprobs(policy, batch["chosen"])
    rejected, rmask = sequence_logprobs(policy, batch["rejected"])
    normalize = method in {"ipo", "simpo"}
    score = chosen.sum(-1) / (cmask.sum(-1) if normalize else 1) - rejected.sum(-1) / (
        rmask.sum(-1) if normalize else 1
    )
    with torch.no_grad():
        rc, rcmask = sequence_logprobs(reference, batch["chosen"])
        rr, rrmask = sequence_logprobs(reference, batch["rejected"])
        ref = rc.sum(-1) / (rcmask.sum(-1) if normalize else 1) - rr.sum(-1) / (
            rrmask.sum(-1) if normalize else 1
        )
    difference = score - (ref if method != "simpo" else 0)
    oracle = (
        (difference - 1 / (2 * 0.2)).square()
        if method == "ipo"
        else -torch.nn.functional.logsigmoid(0.2 * difference - (0.5 if method == "simpo" else 0))
    ).mean()
    term = objective(policy, batch)
    assert term.mean.requires_grad
    torch.testing.assert_close(term.mean, oracle)
    actual = torch.autograd.grad(term.mean, tuple(policy.parameters()))
    expected = torch.autograd.grad(oracle, tuple(policy.parameters()))
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
    assert any(got.abs().sum() > 0 for got in actual)
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in reference.parameters()
    )
    before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
    assert Trainer(policy, objective).step([batch]).updated
    assert any(
        not torch.equal(policy.state_dict()[name], tensor) for name, tensor in before.items()
    )


@pytest.mark.parametrize("algorithm", ["kd", "dpo", "grpo"])
def test_later_microbatch_rejected_before_any_forward_and_corrected_input_continues(algorithm):
    policy = model()
    reference = model()
    if algorithm == "kd":
        objective, good = DistillationObjective(reference), [tokens(4), tokens(5)]
    elif algorithm == "dpo":
        objective, good = PreferenceObjective(reference), [pair(4), pair(5)]
    else:
        objective, good = GRPOObjective(), [grpo_batch(policy, 4), grpo_batch(policy, 5)]
    trainer = Trainer(policy, objective, zero_stage=3, accumulation_steps=2)
    if algorithm != "grpo":
        trainer.add_role("reference", reference, trainable=False)
    calls = []
    policy.register_forward_pre_hook(lambda *_: calls.append("policy"))
    reference.register_forward_pre_hook(lambda *_: calls.append("reference"))
    bad = deepcopy(good)
    target = bad[1]["rejected"] if algorithm == "dpo" else bad[1]
    target["position_ids"][0, -1] = -1
    with pytest.raises(ValueError):
        trainer.step(bad)
    assert calls == [] and trainer.steps == 0
    assert trainer.step(good).updated and calls


def test_grpo_config_codec_checkpoint_does_not_guess_unknown_prior_settings(tmp_path):
    policy = model()
    batch = grpo_batch(policy)
    engine = Trainer(policy, GRPOObjective(kl_weight=0.03))
    assert engine.step([batch]).updated
    receipt = engine.last_successful_update()["objective_configuration"]
    assert receipt["codec"] == "config_dict" and receipt["configuration"]["kl_weight"] == 0.03
    checkpoint = engine.save_checkpoint(tmp_path / "grpo.json")
    changed = Trainer(model(), GRPOObjective(kl_weight=0.09))
    with pytest.raises(ValueError, match="配置"):
        changed.load_checkpoint(checkpoint)
