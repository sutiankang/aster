from dataclasses import replace
import pytest
import torch
from aster.models import GrootActionConfig, GrootConfig, GrootCondition, build_model, load_model
from aster.models.qwen_vl import pack_qwen_pixels
from aster.methods.groot import GrootFlowObjective
from aster.training import Trainer


def condition(c, batch=2):
    return GrootCondition(
        torch.randn(batch, 5, c.backbone_embedding_dim),
        torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool).expand(batch, -1),
        torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.bool).expand(batch, -1),
        torch.randn(batch, c.state_history_length, c.max_state_dim),
        torch.arange(batch) % c.max_num_embodiments,
    )


def observation(c):
    pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 8), c.backbone_config.vision_config)
    ids = torch.tensor([[1, 26, 28, 28, 28, 28, 27, 3]])
    return dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids, dtype=torch.bool),
        pixel_values=pixels,
        image_grid_thw=grid,
        proprio=torch.randn(1, c.action_config.state_history_length, c.action_config.max_state_dim),
        embodiment_id=torch.tensor([1]),
    )


def test_models_groot_train_cache_rtc_and_storage(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(101)
    c = GrootActionConfig()
    model = build_model(c)
    context = condition(c)
    actions, noise, time = torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.tensor([0.23, 0.71])
    batch = dict(actions=actions, noise=noise, time=time, condition=context)
    objective = GrootFlowObjective()
    initial = float(objective(model, batch).mean.detach())
    trainer = Trainer(model, objective, lr=0.003, max_grad_norm=10.0)
    for _ in range(60):
        trainer.step([batch])
    assert float(objective(model, batch).mean.detach()) < initial * 0.3
    model.eval()
    cached = model.sample_actions(context, noise=noise, cache_cross_attention=True)
    plain = model.sample_actions(context, noise=noise, cache_cross_attention=False)
    torch.testing.assert_close(cached, plain, atol=2e-6, rtol=2e-5)
    old = torch.randn(2, 6, 3)
    result = model.sample_actions(
        context, noise=noise, previous_actions=old, overlap_steps=3, frozen_steps=2
    )
    torch.testing.assert_close(result[:, :2], old[:, -3:-1], atol=0, rtol=0)
    model.save_pretrained(tmp_path / "head")
    restored = load_model(tmp_path / "head").eval()
    torch.testing.assert_close(
        restored.sample_actions(context, noise=noise), cached, atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="inference-only"):
        model.prepare_condition(context, cache_cross_attention=True)
    with pytest.raises(ValueError, match="frozen"):
        model.sample_actions(context, noise=noise, overlap_steps=1, frozen_steps=2)


def test_models_groot_full_vision_action_gradient_and_reload(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(102)
    c = GrootConfig()
    model = build_model(c)
    obs = observation(c)
    target, noise = torch.randn(1, 4, 3), torch.randn(1, 4, 3)
    time = torch.tensor([0.43])
    objective = GrootFlowObjective()
    batch = dict(actions=target, noise=noise, time=time, condition=obs)
    initial = float(objective(model, batch).mean.detach())
    trainer = Trainer(model, objective, lr=0.001)
    for _ in range(100):
        trainer.step([batch])
    assert float(objective(model, batch).mean.detach()) < initial * 0.01
    model.zero_grad()
    objective(model, batch).mean.backward()
    assert model.backbone.model.model.visual.patch_embed.proj.weight.grad.norm() > 0
    assert (
        model.backbone.model.model.language_model.layers[0].self_attn.q_proj.weight.grad.norm() > 0
    )
    model.eval()
    result = model.sample_actions(obs, noise=noise)
    model.save_pretrained(tmp_path / "vla")
    restored = load_model(tmp_path / "vla").eval()
    torch.testing.assert_close(result, restored.sample_actions(obs, noise=noise), atol=0, rtol=0)
    assert restored.predict_chunk(obs).pad_logits.isneginf().all()


def test_models_groot_masks_buckets_and_beta():
    torch.set_num_threads(1)
    torch.manual_seed(103)
    c = GrootActionConfig()
    model = build_model(c).eval()
    context = condition(c)
    sample = torch.randn(2, 4, 3)
    time = torch.tensor([0.1, 0.4])
    base = model(sample, time, context).prediction
    features = context.features.clone()
    features[:, -1] += 1000
    torch.testing.assert_close(
        base, model(sample, time, replace(context, features=features)).prediction
    )
    torch.testing.assert_close(
        base, model(sample, time + 0.0001, context).prediction, atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="embodiment"):
        model(sample, time, replace(context, embodiment_id=torch.tensor([1, 8])))
    objective = GrootFlowObjective()
    draws = objective.sample_time(10000, torch.device("cpu"), torch.float32)
    assert 0 <= float(draws.min()) < float(draws.max()) < 0.999
    assert abs(float(draws.mean()) - 0.999 / (1.5 + 1)) < 0.012
    targets = torch.randn_like(sample)
    mask = torch.ones_like(targets, dtype=torch.bool)
    mask[:, -1] = False
    term = objective(
        model, dict(actions=targets, noise=sample, time=time, condition=context, action_mask=mask)
    )
    assert int(term.denominator) == 18 and torch.isfinite(term.numerator)
    with pytest.raises(ValueError, match="padding"):
        plain = build_model(replace(c, use_alternate_vl_dit=False))
        plain(sample, time, context)


def test_models_groot_shared_trainer_exact_stochastic_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(104)
    c = GrootActionConfig(dropout=0.2, state_dropout_prob=0.4)
    model = build_model(c)
    context = condition(c)
    batch = dict(actions=torch.randn(2, 4, 3), condition=context)
    trainer = Trainer(model, GrootFlowObjective(), lr=0.001)
    trainer.step([batch])
    trainer.save_checkpoint(tmp_path / "checkpoint.json")
    expected = trainer.step([batch])
    tensors = {n: value.detach().clone() for n, value in model.state_dict().items()}
    new_model = build_model(c)
    resumed = Trainer(new_model, GrootFlowObjective(), lr=0.001)
    resumed.load_checkpoint(tmp_path / "checkpoint.json")
    actual = resumed.step([batch])
    assert actual.loss == expected.loss
    for name, value in new_model.state_dict().items():
        torch.testing.assert_close(value, tensors[name], atol=0, rtol=0, msg=name)
