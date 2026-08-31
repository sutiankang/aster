from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from aster.models import CausalLM, Qwen3Config
from aster.models.dspark import DSparkConfig, DSparkDraft
from aster.methods.dspark import DSparkMethod, dspark_loss_terms
from aster.training import Trainer


def objects(*, stochastic=False):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    target = CausalLM(
        Qwen3Config(
            vocab_size=23,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()
    draft = DSparkDraft(
        DSparkConfig(
            target.config,
            target_layer_ids=(-1, 1),
            num_anchors=2,
            block_size=3,
            markov_rank=4,
            markov_head_type="rnn",
        )
    ).initialize_from_target(target)
    batches = []
    for n in (1, 2):
        ids = torch.randint(23, (n, 7))
        with torch.no_grad():
            output = target(ids, output_hidden_states=True)
        batch = dict(
            input_ids=ids,
            loss_mask=torch.ones_like(ids),
            target_hidden_states=torch.cat((output.hidden_states[0], output.hidden_states[2]), -1),
            target_last_hidden_states=output.hidden_states[-1],
        )
        batch["loss_mask"][-1, -2] = 0
        if not stochastic:
            batch.update(
                anchor_positions=torch.tensor([[0, 2]]).expand(n, -1),
                block_keep_mask=torch.ones(n, 2, dtype=torch.bool),
            )
        batches.append(batch)
    return draft, batches


def independent_loss(output):
    logits, teacher = output.draft_logits.float(), output.aligned_target_logits.detach().float()
    w = output.eval_mask.float() * torch.exp(-torch.arange(logits.shape[-2]) / 4.0)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), output.target_ids.reshape(-1), reduction="none"
    ).reshape_as(w)
    l1 = (logits.softmax(-1) - teacher.softmax(-1)).abs().sum(-1)
    confidence = F.binary_cross_entropy_with_logits(
        output.confidence_pred, (1 - 0.5 * l1.detach()).clamp(0, 1), reduction="none"
    )
    return ((0.1 * ce + 0.9 * l1 + confidence) * w).sum() / (w.sum().double() + 1e-6)


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_dspark_whole_window_matches_full_batch_loss_and_gradients(stage):
    model, batches = objects()
    dense = deepcopy(model)
    full = {key: torch.cat([b[key] for b in batches]) for key in batches[0]}
    factory = lambda params: torch.optim.SGD(params, lr=0.03, momentum=0.9)
    optimizer = factory(dense.parameters())
    expected = independent_loss(dense(**full))
    expected.backward()
    optimizer.step()
    engine = Trainer(
        model, accumulation_steps=2, zero_stage=stage, max_grad_norm=None, optimizer_factory=factory
    )
    method = DSparkMethod(
        engine, vocabulary_fingerprint="test_vocab23_v1", normalization_profile="global_window"
    )
    batches = [
        dict(b, teacher_identity=model.teacher_identity, vocabulary_fingerprint="test_vocab23_v1")
        for b in batches
    ]
    result = method.update(batches)
    assert result.updated and abs(result.loss - float(expected.detach())) < 5e-7
    for name, actual in engine.export_state_dict(only_rank_zero=False).items():
        torch.testing.assert_close(actual, dense.state_dict()[name], atol=1e-7, rtol=3e-5)
    assert method.updates == 1


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_dspark_stochastic_sampler_fresh_checkpoint_preserves_rng_and_frozen_roles(
    precision, tmp_path
):
    model, batches = objects(stochastic=True)
    before = model.lm_head.weight.detach().clone()
    engine = Trainer(model, accumulation_steps=2, zero_stage=3, precision=precision, ema_decay=0.9)
    method = DSparkMethod(
        engine, vocabulary_fingerprint="test_vocab23_v1", normalization_profile="global_window"
    )
    batches = [
        dict(b, teacher_identity=model.teacher_identity, vocabulary_fingerprint="test_vocab23_v1")
        for b in batches
    ]
    assert method.update(batches).updated
    checkpoint = engine.save_checkpoint(tmp_path / precision)
    expected = method.update(batches)
    state = deepcopy(engine.export_state_dict(only_rank_zero=False))
    fresh, _ = objects(stochastic=True)
    other = Trainer(fresh, accumulation_steps=2, zero_stage=3, precision=precision, ema_decay=0.9)
    restored = DSparkMethod(
        other, vocabulary_fingerprint="test_vocab23_v1", normalization_profile="global_window"
    )
    other.load_checkpoint(checkpoint, trusted=True)
    actual = restored.update(batches)
    assert actual.loss == expected.loss and restored.updates == 2
    for name, value in other.export_state_dict(only_rank_zero=False).items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    torch.testing.assert_close(before, state["lm_head.weight"], atol=0, rtol=0)


def test_dspark_bad_last_microbatch_rejected_before_backbone_and_optimizer():
    model, batches = objects()
    engine = Trainer(model, accumulation_steps=2, zero_stage=3)
    method = DSparkMethod(
        engine, vocabulary_fingerprint="test_vocab23_v1", normalization_profile="global_window"
    )
    batches = [
        dict(b, teacher_identity=model.teacher_identity, vocabulary_fingerprint="test_vocab23_v1")
        for b in batches
    ]
    bad = deepcopy(batches)
    bad[-1]["anchor_positions"][0, 1] = 6
    calls = []
    handle = model.fc.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(ValueError, match="anchor"):
            method.update(bad)
    finally:
        handle.remove()
    assert not calls and not engine._failed and method.updates == 0


def test_dspark_empty_rank_anchors_are_finite_and_do_not_leak_teacher_gradients():
    model, batches = objects(stochastic=True)
    batch = batches[0]
    batch["loss_mask"].zero_()
    output = model(**batch)
    assert not output.eval_mask.any() and not output.block_keep_mask.any()
    assert torch.isfinite(output.draft_logits).all()
    loss = sum(term.weight * term.mean for term in dspark_loss_terms(output).terms)
    loss.backward()
    assert loss == 0 and all(p.grad is None or not p.grad.any() for p in model.parameters())
    assert batch["target_hidden_states"].grad is None


def test_dspark_confidence_supervision_does_not_backpropagate_through_acceptance_target():
    model, batches = objects()
    output = model(**batches[0])
    gradients = torch.autograd.grad(
        dspark_loss_terms(output).terms[-1].numerator,
        (output.draft_logits, output.confidence_pred),
        allow_unused=True,
    )
    assert gradients[0] is None and gradients[1].isfinite().all()
