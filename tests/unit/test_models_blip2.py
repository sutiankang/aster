from dataclasses import replace
import pytest
import torch
from aster.models import Blip2Config, Blip2QFormerConfig, LlamaConfig, build_model, load_model
from aster.methods.supervised import CrossEntropyObjective
from aster.training import Trainer


def sample(c):
    ids = torch.tensor([[31, 31, 31, 31, 1, 3, 5], [31, 31, 31, 31, 2, 4, 6]])
    inputs = dict(input_ids=ids, pixel_values=torch.randn(2, 3, 8, 8))
    if c.text_config.architecture == "t5":
        labels = torch.tensor([[2, 8, 9, 3], [4, 7, 3, 2]])
        inputs["decoder_input_ids"] = torch.cat(
            (torch.zeros(2, 1, dtype=torch.long), labels[:, :-1]), 1
        )
    else:
        labels = ids.clone()
        labels[:, :4] = -100
    return dict(model_inputs=inputs, labels=labels)


@pytest.mark.parametrize("family", ["t5", "llama"])
def test_models_blip2_shared_training_export_and_frozen_gradient(tmp_path, family):
    torch.set_num_threads(1)
    torch.manual_seed(98)
    c = Blip2Config()
    if family == "llama":
        c = replace(c, text_config=LlamaConfig())
    model = build_model(c)
    batch = sample(c)
    objective = CrossEntropyObjective(causal=family != "t5")
    initial = float(objective(model, batch).mean.detach())
    trainer = Trainer(model, objective, lr=0.005)
    for _ in range(20):
        trainer.step([batch])
    assert float(objective(model, batch).mean.detach()) < initial * 0.5

    model.zero_grad(set_to_none=True)
    model.vision_model.requires_grad_(False)
    model.language_model.requires_grad_(False)
    objective(model, batch).mean.backward()
    assert model.query_tokens.grad.abs().sum() > 0
    assert model.qformer.encoder.layer[0].crossattention.attention.value.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.language_model.parameters())
    model.eval()
    expected = model(**batch["model_inputs"]).logits
    model.save_pretrained(tmp_path / "blip2")
    restored = load_model(tmp_path / "blip2").eval()
    torch.testing.assert_close(restored(**batch["model_inputs"]).logits, expected, atol=0, rtol=0)
    if family == "t5":
        assert restored.language_model.lm_head.weight is restored.language_model.shared.weight
    state = model(**batch["model_inputs"], use_cache=True).state
    assert state.kind == "blip2_language_state" and state.fork().seen_tokens == state.seen_tokens
    assert state.truncate(1).seen_tokens == 1
    with pytest.raises(ValueError, match="new prefill"):
        model(**batch["model_inputs"], state=state)


def test_models_blip2_qformer_visibility_and_rejections():
    torch.set_num_threads(1)
    torch.manual_seed(99)
    c = Blip2QFormerConfig(use_qformer_text_input=True)
    model = build_model(c).eval()
    queries, vision = torch.randn(2, 6, c.hidden_size), torch.randn(2, 8, c.encoder_hidden_size)
    visibility = torch.ones(2, 1, 6, 6, dtype=torch.bool)
    visibility[:, :, :3, 3:] = False
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[:, -2:] = False
    result = model(
        queries,
        query_length=3,
        attention_mask=visibility,
        encoder_hidden_states=vision,
        encoder_attention_mask=mask,
    ).last_hidden_state
    changed_queries, changed_vision = queries.clone(), vision.clone()
    changed_queries[:, 3:] += torch.randn_like(queries[:, 3:]) * 10
    changed_vision[:, -2:] += torch.randn_like(vision[:, -2:]) * 100
    changed = model(
        changed_queries,
        query_length=3,
        attention_mask=visibility,
        encoder_hidden_states=changed_vision,
        encoder_attention_mask=mask,
    ).last_hidden_state
    torch.testing.assert_close(result[:, :3], changed[:, :3], atol=0, rtol=0)
    with pytest.raises(ValueError, match="append-only"):
        model(queries, use_cache=True)
    with pytest.raises(ValueError, match="visual encoder"):
        model(queries)
    with pytest.raises(ValueError, match="independent"):
        build_model(replace(c, use_qformer_text_input=False))(
            queries, query_length=3, encoder_hidden_states=vision
        )


def test_models_blip2_stochastic_shared_trainer_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(100)
    base = Blip2Config()
    c = replace(
        base,
        vision_config=replace(base.vision_config, attention_dropout=0.15),
        qformer_config=replace(
            base.qformer_config, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.1
        ),
        text_config=replace(base.text_config, dropout_rate=0.2),
    )
    model = build_model(c)
    batch = sample(c)
    trainer = Trainer(model, CrossEntropyObjective(causal=False), lr=0.003)
    trainer.step([batch])
    trainer.save_checkpoint(tmp_path / "checkpoint.json")
    expected = trainer.step([batch])
    weights = {n: v.detach().clone() for n, v in model.state_dict().items()}
    other = build_model(c)
    resumed = Trainer(other, CrossEntropyObjective(causal=False), lr=0.003)
    resumed.load_checkpoint(tmp_path / "checkpoint.json")
    actual = resumed.step([batch])
    assert expected.loss == actual.loss
    for name, value in other.state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
