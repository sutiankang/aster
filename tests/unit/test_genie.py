from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from aster.models import build_model
from aster.models.config import config_from_dict
from aster.models.genie import (
    GenieTokenizerConfig,
    GenieTokenizer,
    GenieActionConfig,
    GenieLatentAction,
    GenieDynamicsConfig,
    GenieWorldConfig,
    GenieWorld,
    STStack,
    patch_video,
    unpatch_video,
    VideoVectorQuantizer,
)
from aster.methods.genie import GenieVQObjective, GenieWorldObjective, encode_genie_video
from aster.methods.factory import build_objective
from aster.planning.genie import generate_genie_video, sample_maskgit, maskgit_remask
from aster.training import Trainer


def configs():
    common = dict(
        image_height=8,
        image_width=8,
        image_channels=1,
        hidden_size=8,
        num_heads=2,
        head_dim=4,
        encoder_layers=1,
        decoder_hidden_size=8,
        decoder_num_heads=2,
        decoder_head_dim=4,
        decoder_layers=1,
        latent_dim=3,
        max_frames=4,
        intermediate_ratio=2,
    )
    tokenizer = GenieTokenizerConfig(**common, patch_size=4, num_codes=5)
    action = GenieActionConfig(**common, patch_size=8, num_codes=3)
    dynamics = GenieDynamicsConfig(
        spatial_tokens=4,
        vocab_size=5,
        action_dim=3,
        hidden_size=8,
        num_heads=2,
        head_dim=4,
        num_layers=1,
        intermediate_ratio=2,
        max_frames=4,
    )
    return tokenizer, GenieWorldConfig(action, dynamics)


def video():
    return torch.rand(2, 4, 1, 8, 8)


def test_genie_patch_geometry_temporal_causality_and_lam_no_future_leak():
    torch.set_num_threads(1)
    torch.manual_seed(456)
    tc, wc = configs()
    x = video()
    assert torch.equal(unpatch_video(patch_video(x, tc), tc), x)
    stack = STStack(8, 2, 4, 2, 4, 4, 2, True, 1e-5)
    features = torch.randn(2, 4, 4, 8)
    expected = stack(features)
    changed = features.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 5
    torch.testing.assert_close(stack(changed)[:, :2], expected[:, :2], atol=0, rtol=0)
    tokenizer = GenieTokenizer(tc)
    first = tokenizer(x)
    changed = x.clone()
    changed[:, 2:] = torch.rand_like(changed[:, 2:])
    torch.testing.assert_close(
        tokenizer(changed).reconstruction[:, :2], first.reconstruction[:, :2], atol=0, rtol=0
    )
    action = GenieLatentAction(wc.action)
    ids = action.encode(x).indices
    assert torch.equal(action.encode(changed).indices[:, :1], ids[:, :1])
    actions = torch.randn(2, 3, 3, requires_grad=True)
    context = x[:, :-1].clone().requires_grad_()
    action.decode(context, actions)[:, 0].sum().backward()
    assert context.grad[:, 1:].abs().sum() == 0 and actions.grad[:, 1:].abs().sum() == 0


def test_genie_vq_nearest_codes_and_stop_gradient_formula():
    torch.manual_seed(591)
    quantizer = VideoVectorQuantizer(5, 3)
    x = torch.randn(2, 4, 3, requires_grad=True)
    out = quantizer(x)
    distance = (x[..., None, :] - quantizer.embedding.weight[None, None]).square().sum(-1)
    assert torch.equal(out.indices, distance.argmin(-1))
    out.quantized.sum().backward(retain_graph=True)
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    assert quantizer.embedding.weight.grad is None
    x.grad = None
    out.codebook_errors.sum().backward(retain_graph=True)
    assert x.grad is None and quantizer.embedding.weight.grad is not None
    out.commitment_errors.sum().backward()
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("stage,precision", [(0, "fp32"), (3, "fp32"), (0, "bf16"), (3, "bf16")])
def test_genie_native_tokenizer_joint_world_train_and_exact_restore(stage, precision, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(655)
    tc, wc = configs()
    tokenizer = build_model(config_from_dict(tc.to_dict()))
    x = video()
    codec_engine = Trainer(
        tokenizer,
        GenieVQObjective(sequence_length=4),
        zero_stage=stage,
        precision=precision,
        optimizer_factory=lambda p: torch.optim.AdamW(p, lr=0.002),
    )
    assert codec_engine.step([dict(video=x)]).updated

    frozen = GenieTokenizer(tc)
    frozen.load_state_dict(codec_engine.export_state_dict())
    data = encode_genie_video(frozen, x)
    world = build_model(config_from_dict(wc.to_dict()))
    objective = build_objective(dict(name="genie_world", sequence_length=4))
    engine = Trainer(
        world,
        objective,
        zero_stage=stage,
        precision=precision,
        accumulation_steps=2,
        ema_decay=0.9,
        optimizer_factory=lambda p: torch.optim.AdamW(p, lr=0.002),
    )
    assert engine.step([data, data]).updated
    checkpoint = engine.save_checkpoint(tmp_path / "state")
    expected = engine.step([data, data])
    weights = deepcopy(engine.export_state_dict())
    engine.load_checkpoint(checkpoint, trusted=True)
    actual = engine.step([data, data])
    assert expected.loss == actual.loss
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    restored = GenieWorld(wc)
    restored.load_state_dict(weights)
    restored.save_pretrained(tmp_path / "world")
    restored = GenieWorld.from_pretrained(tmp_path / "world")

    def forbidden(*args, **kwargs):
        raise AssertionError("LAM encoder/decoder must not run during interactive inference")

    restored.action_model.forward = forbidden
    restored.action_model.encode = forbidden
    restored.action_model.decode = forbidden
    rng = torch.get_rng_state().clone()
    generated, info = generate_genie_video(
        frozen,
        restored,
        x[:, :1],
        torch.tensor([[0, 1], [2, 0]]),
        steps=3,
        generator=torch.Generator().manual_seed(713),
    )
    assert torch.equal(torch.get_rng_state(), rng)
    assert generated.shape == (2, 3, 1, 8, 8) and info["model_calls"] == 6
    assert torch.equal(generated[:, :1], x[:, :1])


def test_genie_joint_dynamics_cannot_backpropagate_into_action_encoder():
    torch.set_num_threads(1)
    tc, wc = configs()
    world = GenieWorld(wc)
    _, logits = world(video(), torch.full((2, 4, 4), 5, dtype=torch.int64))
    logits.sum().backward()
    assert all(p.grad is None for p in world.action_model.parameters())
    assert all(p.grad is not None for p in world.dynamics.parameters())


def test_maskgit_fixed_tokens_schedule_formula_and_invalid_sampling():
    current = torch.tensor([[0, 3, 3, 1], [3, 3, 3, 3]])
    calls = []

    def predict(tokens):
        calls.append(tokens.clone())
        return torch.tensor([1.0, 2.0, 3.0]).expand(*tokens.shape, 3)

    result, info = sample_maskgit(
        predict, current, mask_token_id=3, steps=4, generator=torch.Generator().manual_seed(542)
    )
    assert len(calls) == info["model_calls"] == 4
    assert (result < 3).all() and result[0, 0] == 0 and result[0, 3] == 1
    assert all(torch.equal(step[0, [0, 3]], current[0, [0, 3]]) for step in calls)
    probs = torch.tensor([[0.1, 0.8, 0.3, 0.5]])
    unknown = torch.tensor([[True, False, True, True]])
    gumbel = torch.tensor([[0.3, -0.2, 0.7, -0.5]])
    confidence = probs.log() + 0.4 * gumbel
    confidence[~unknown] = torch.inf
    expected = confidence < confidence.sort(-1).values[:, 2:3]
    assert torch.equal(maskgit_remask(probs, unknown, torch.tensor([2]), gumbel, 0.4), expected)
    with pytest.raises(ValueError, match="temperature"):
        sample_maskgit(predict, current, mask_token_id=3, token_temperature=0.0)
    known, known_info = sample_maskgit(predict, result, mask_token_id=3)
    assert known_info["model_calls"] == 0 and torch.equal(known, result)


def test_genie_mask_valid_count_preflight_and_small_learning():
    torch.set_num_threads(1)
    torch.manual_seed(736)
    tc, wc = configs()
    tokenizer = GenieTokenizer(tc)

    x = torch.full((2, 4, 1, 8, 8), 0.2)
    x[:, :, :, 2:6, 2:6] = 0.8
    trainer = Trainer(tokenizer, GenieVQObjective(sequence_length=4), lr=0.003)
    initial = (tokenizer(x).reconstruction - x).square().mean().item()
    for _ in range(100):
        assert trainer.step([dict(video=x)]).updated
    assert (tokenizer(x).reconstruction - x).square().mean().item() < initial * 0.4
    batch = encode_genie_video(tokenizer, x)
    world = GenieWorld(wc)
    batch["mask"] = torch.ones_like(batch["tokens"], dtype=torch.bool)
    batch["mask"][:, 0] = False
    objective = GenieWorldObjective(sequence_length=4)
    engine = Trainer(world, objective, lr=0.005)
    before = objective(world, batch).terms[-1].mean.item()
    for _ in range(30):
        assert engine.step([batch]).updated
    assert objective(world, batch).terms[-1].mean.item() < before * 0.4
    bad = dict(batch, mask=torch.ones_like(batch["mask"]))
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="mask"):
        engine.step([bad])
    assert torch.equal(rng, torch.get_rng_state())


def test_genie_paired_controllability_metric_runs_both_real_rollouts():
    from aster.evaluation.genie_world import paired_delta_psnr, evaluate_genie_controllability

    torch.set_num_threads(1)
    torch.manual_seed(822)
    reference = torch.full((2, 1, 4, 4), 0.5, dtype=torch.float64)
    delta, direct, baseline = paired_delta_psnr(reference, reference + 0.1, reference + 0.2)
    torch.testing.assert_close(
        delta,
        torch.full_like(delta, 20 * torch.log10(torch.tensor(2.0, dtype=torch.float64)).item()),
    )
    tc, wc = configs()
    tokenizer, world = GenieTokenizer(tc), GenieWorld(wc)
    data = video()
    rng = torch.get_rng_state().clone()
    result = evaluate_genie_controllability(tokenizer, world, data, time_index=2, steps=2, seed=2)
    assert result["metric"] == "delta_psnr_t2" and result["per_sample"].shape == (2,)
    assert result["model_calls"] == 8 and result["public_quality_evaluated"] is False
    assert torch.equal(rng, torch.get_rng_state())
    repeated = evaluate_genie_controllability(tokenizer, world, data, time_index=2, steps=2, seed=2)
    assert torch.equal(result["per_sample"], repeated["per_sample"])
    with pytest.raises(ValueError, match="horizon"):
        evaluate_genie_controllability(tokenizer, world, data, time_index=4)
