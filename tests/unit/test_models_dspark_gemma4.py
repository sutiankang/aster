from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from torch import nn
from aster.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
from aster.models.dspark_gemma4 import Gemma4DSparkConfig, Gemma4DSparkDraft
from aster.methods.dspark import dspark_loss_terms
from aster.training import Trainer


def objects(stochastic=False, freeze=True):
    torch.set_num_threads(1)
    torch.manual_seed(763)
    config = Gemma4TextConfig(
        vocab_size=31,
        hidden_size=24,
        intermediate_size=32,
        head_dim=4,
        global_head_dim=8,
        hidden_size_per_layer_input=0,
        final_logit_softcapping=0.7,
    )
    teacher = Gemma4ForCausalLM(config).eval()
    draft = Gemma4DSparkDraft(
        Gemma4DSparkConfig(
            config,
            num_draft_layers=2,
            target_layer_ids=(-1, 1),
            num_anchors=2,
            block_size=3,
            markov_rank=4,
            markov_head_type="rnn",
            freeze_embedding_head=freeze,
        )
    )
    draft.initialize_from_target(teacher)
    batches = []
    for size in (1, 2):
        ids = torch.randint(1, 31, (size, 7))
        with torch.no_grad():
            output = teacher(ids, output_hidden_states=True)
        batch = dict(
            input_ids=ids,
            loss_mask=torch.ones_like(ids),
            target_hidden_states=torch.cat((output.hidden_states[0], output.hidden_states[2]), -1),
            target_last_hidden_states=output.hidden_states[-1],
        )
        batch["loss_mask"][-1, -2] = 0
        if not stochastic:
            batch.update(
                anchor_positions=torch.tensor([[0, 2]]).expand(size, -1).clone(),
                block_keep_mask=torch.ones(size, 2, dtype=torch.bool),
            )
        batches.append(batch)
    return draft, batches, teacher


class GemmaLoss(nn.Module):
    def config_dict(self):
        return {"type": "gemma4_dspark_test", "window": 2, "epsilon": 1e-6}

    def preflight_microbatches(self, model, batches):
        if type(model) is not Gemma4DSparkDraft or len(batches) != 2:
            raise ValueError("Gemma4 draft/window mismatch")
        for batch in batches:
            model.validate_batch(
                batch["input_ids"],
                batch["target_hidden_states"],
                batch["loss_mask"],
                batch.get("target_last_hidden_states"),
            )
            if "anchor_positions" in batch:
                model.validate_anchors(
                    batch["input_ids"],
                    batch["loss_mask"],
                    batch["anchor_positions"],
                    batch["block_keep_mask"],
                )
        return batches

    def forward(self, model, batch):
        return dspark_loss_terms(model(**batch), denominator_offset=0.5e-6)


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_models_gemma4_dspark_real_teacher_whole_batch_vs_accumulated_update(stage):
    draft, batches, teacher = objects()
    dense = deepcopy(draft)
    full = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
    factory = lambda parameters: torch.optim.SGD(parameters, lr=0.03, momentum=0.9)
    optimizer = factory(dense.parameters())
    terms = dspark_loss_terms(dense(**full)).terms
    expected = sum(term.weight * term.mean for term in terms)
    expected.backward()
    optimizer.step()
    engine = Trainer(
        draft, accumulation_steps=2, zero_stage=stage, max_grad_norm=None, optimizer_factory=factory
    )
    result = engine.phase("draft", objective=GemmaLoss(), microbatches=batches)
    assert result.updated and abs(result.loss - expected.item()) < 5e-7
    for name, value in engine.export_state_dict(only_rank_zero=False).items():
        torch.testing.assert_close(value, dense.state_dict()[name], atol=1e-7, rtol=4e-5, msg=name)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    torch.testing.assert_close(teacher.lm_head.weight, dense.lm_head.weight, atol=0, rtol=0)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_models_gemma4_dspark_stochastic_fresh_resume_includes_semantic_scale(precision, tmp_path):
    draft, batches, _ = objects(stochastic=True)
    engine = Trainer(draft, accumulation_steps=2, zero_stage=3, precision=precision, ema_decay=0.9)
    loss = GemmaLoss()
    assert engine.phase("draft", objective=loss, microbatches=batches).updated
    checkpoint = engine.save_checkpoint(tmp_path / precision)
    expected = engine.phase("draft", objective=loss, microbatches=batches)
    weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
    fresh, _, _ = objects(stochastic=True)
    resumed = Trainer(fresh, accumulation_steps=2, zero_stage=3, precision=precision, ema_decay=0.9)
    resumed.load_checkpoint(checkpoint, trusted=True)
    actual = resumed.phase("draft", objective=GemmaLoss(), microbatches=batches)
    assert actual.loss == expected.loss and actual.updated
    for name, value in resumed.export_state_dict(only_rank_zero=False).items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0, msg=name)


def test_models_gemma4_dspark_config_boundaries_softcap_and_zero_mask():
    draft, batches, _ = objects()
    with pytest.raises(ValueError, match="MoE"):
        replace(draft.config, target=replace(draft.config.target, enable_moe_block=True))
    with pytest.raises(ValueError, match="per-layer"):
        replace(draft.config, target=replace(draft.config.target, hidden_size_per_layer_input=8))
    assert (
        Gemma4DSparkConfig(
            **{k: v for k, v in draft.config.to_dict().items() if k != "architecture"}
        )
        == draft.config
    )
    batch = batches[0]
    bad = dict(batch, target_hidden_states=batch["target_hidden_states"].requires_grad_())
    with pytest.raises(ValueError, match="detached"):
        draft(**bad)
    batch["target_hidden_states"] = batch["target_hidden_states"].detach()
    batch["loss_mask"].zero_()
    batch.pop("anchor_positions")
    batch.pop("block_keep_mask")
    output = draft(**batch)
    assert torch.isfinite(output.draft_logits).all() and not output.eval_mask.any()
    loss = sum(term.weight * term.mean for term in dspark_loss_terms(output).terms)
    loss.backward()
    assert loss == 0 and all(p.grad is None or not p.grad.any() for p in draft.parameters())

    hidden = torch.randn(2, 24)
    raw = draft.lm_head(hidden)
    cap = draft.config.target.final_logit_softcapping
    torch.testing.assert_close(
        draft.compute_logits(hidden), cap * torch.tanh(raw / cap), atol=0, rtol=0
    )


def test_models_gemma4_dspark_invalid_last_microbatch_preflights_before_forward():
    draft, batches, _ = objects()
    engine = Trainer(draft, accumulation_steps=2, zero_stage=3)
    batches[-1]["anchor_positions"][0, 1] = 6
    calls = []
    handle = draft.fc.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        with pytest.raises(ValueError, match="anchor"):
            engine.phase("draft", objective=GemmaLoss(), microbatches=batches)
    finally:
        handle.remove()
    assert not calls and not engine._failed


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_models_gemma4_dspark_safe_artifact_and_lazy_factory_roundtrip(dtype, tmp_path):
    from aster.models import build_model, load_model, Gemma4DSparkConfig as ExportedConfig
    from aster.models.config import config_from_dict

    draft, batches, _ = objects()
    draft.to(dtype=dtype)
    assert ExportedConfig is Gemma4DSparkConfig
    assert type(build_model(config_from_dict(draft.config.to_dict()))) is Gemma4DSparkDraft
    batch = {
        key: value.to(dtype) if value.is_floating_point() else value
        for key, value in batches[0].items()
    }
    expected = draft(**batch)
    draft.save_pretrained(tmp_path / "draft")
    restored = load_model(tmp_path / "draft")
    actual = restored(**batch)
    assert restored.teacher_identity == draft.teacher_identity
    for field in ("draft_logits", "confidence_pred", "aligned_target_logits"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=0, rtol=0)
    torch.testing.assert_close(restored.embed_scale, draft.embed_scale, atol=0, rtol=0)
    assert restored.embed_scale.dtype == dtype
