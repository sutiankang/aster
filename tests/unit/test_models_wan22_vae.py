from dataclasses import replace
import pytest
import torch
from aster.models import Wan22VAEConfig, build_model, load_model
from aster.models.wan22_vae import wan22_patchify, wan22_unpatchify
from aster.methods.cosmos3 import Wan22AutoencoderObjective
from aster.training import Trainer


def test_models_wan22_patch_order_stats_and_config_validation():
    video = torch.arange(16.0).view(1, 1, 1, 4, 4)
    patches = wan22_patchify(video, 2)
    torch.testing.assert_close(
        patches[0, :, 0, 0, 0], torch.tensor([0.0, 4.0, 1.0, 5.0]), atol=0, rtol=0
    )
    torch.testing.assert_close(wan22_unpatchify(patches, 2), video, atol=0, rtol=0)
    config = Wan22VAEConfig(latents_mean=(0.2, -0.3), latents_std=(0.5, 2.0))
    upstream = config.to_dict()
    upstream.pop("architecture")
    upstream.update(
        is_residual=True,
        in_channels=12,
        out_channels=12,
        attn_scales=[],
        scale_factor_spatial=16,
        scale_factor_temporal=4,
        clip_output=False,
        _class_name="AutoencoderKLWan",
    )
    assert Wan22VAEConfig.from_diffusers_config(upstream) == config
    with pytest.raises(ValueError, match="fields"):
        Wan22VAEConfig.from_diffusers_config(dict(upstream, different_math=True))
    with pytest.raises(ValueError, match="strides"):
        Wan22VAEConfig.from_diffusers_config(dict(upstream, scale_factor_spatial=8))
    with pytest.raises(ValueError, match="residual"):
        Wan22VAEConfig.from_diffusers_config(dict(upstream, is_residual=False))
    model = build_model(config)
    latent = torch.randn(2, 2, 3, 2, 2)
    torch.testing.assert_close(model.transform(model.transform(latent), inverse=True), latent)
    with pytest.raises(ValueError, match="statistics"):
        replace(config, latents_std=(0.0, 2.0))
    with pytest.raises(ValueError, match="stride 4"):
        replace(config, temperal_downsample=(False, False, True))


def test_models_wan22_causal_chunks_history_gradients_and_no_cache_leak():
    torch.set_num_threads(1)
    torch.manual_seed(558)
    model = build_model(Wan22VAEConfig())
    video = torch.randn(1, 3, 9, 16, 16, requires_grad=True)
    posterior = model.encode(video)
    assert posterior.mean.shape == (1, 2, 3, 1, 1)
    torch.testing.assert_close(
        model.encode(video[:, :, :5]).mean, posterior.mean[:, :, :2], atol=0, rtol=0
    )
    posterior.mean[:, :, -1:].square().sum().backward()
    assert video.grad[:, :, :1].abs().sum() > 0, "不能detach跨chunk因果激活"
    latent = posterior.mean.detach().requires_grad_()
    pieces = tuple(model.decode_chunks(latent, clip_output=False))
    assert [part.shape[2] for part in pieces] == [1, 4, 4]
    full = torch.cat(pieces, 2)
    torch.testing.assert_close(
        model.decode(latent[:, :, :2], clip_output=False), full[:, :, :5], atol=0, rtol=0
    )

    pending = model.decode_chunks(latent, clip_output=False)
    first = next(pending)
    model.decode(torch.randn_like(latent), clip_output=False)
    torch.testing.assert_close(torch.cat((first, *pending), 2), full, atol=0, rtol=0)
    full[:, :, -4:].square().mean().backward()
    assert latent.grad[:, :, :1].abs().sum() > 0
    torch.testing.assert_close(model.decode(latent), full.clamp(-1, 1), atol=0, rtol=0)
    with pytest.raises(ValueError, match=r"1\+4k"):
        model.encode(video[:, :, :4])


@pytest.mark.parametrize("stage", [0, 3])
def test_models_wan22_shared_trainer_learning_posterior_rng_resume_and_export(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(559)
    c = Wan22VAEConfig(dropout=0.1)
    model = build_model(c)
    batch = dict(sample=torch.randn(1, 3, 5, 16, 16) * 0.2)
    engine = Trainer(
        model, Wan22AutoencoderObjective(sequence_length=5), zero_stage=stage, lr=0.001
    )
    initial = engine.step([batch]).loss
    for _ in range(14):
        final = engine.step([batch]).loss
    assert final < initial
    checkpoint = engine.save_checkpoint(tmp_path / "resume")
    expected = engine.step([batch])
    weights = engine.export_state_dict()
    engine.load_checkpoint(checkpoint, trusted=True)
    actual = engine.step([batch])
    assert actual.loss == expected.loss
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    restored = build_model(c)
    restored.load_state_dict(weights, strict=True)
    restored.eval()
    restored.save_pretrained(tmp_path / "artifact")
    duplicate = load_model(tmp_path / "artifact").eval()
    a, posterior = restored(batch["sample"], sample_posterior=False)
    b, _ = duplicate(batch["sample"], sample_posterior=False)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.isfinite(posterior.kl()).all()


def test_models_wan22_bfloat16_export_and_streaming(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(560)
    model = build_model(Wan22VAEConfig()).bfloat16()
    video = torch.randn(1, 3, 5, 16, 16).bfloat16()
    output, _ = model(video, sample_posterior=False)
    output.float().square().mean().backward()
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.save_pretrained(tmp_path / "bf16")
    restored = load_model(tmp_path / "bf16")
    torch.testing.assert_close(output, restored(video, sample_posterior=False)[0], atol=0, rtol=0)


@pytest.mark.parametrize("dtype,log_variance", [(torch.bfloat16, 0.01), (torch.float16, 12.0)])
def test_models_wan22_kl_statistics_avoid_low_precision_negative_or_overflow(dtype, log_variance):
    torch.set_num_threads(1)
    model = build_model(Wan22VAEConfig()).to(dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.quant_conv.bias[model.config.z_dim :].fill_(log_variance)
    video = torch.zeros(1, 3, 5, 16, 16, dtype=dtype)
    objective = Wan22AutoencoderObjective(sequence_length=5, sample_posterior=False)
    term = objective(model, {"sample": video}).terms[1]
    posterior = model.encode(video)

    mean, logvar = posterior.mean.double(), posterior.logvar.double()
    expected = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).sum()
    assert (
        term.numerator.dtype == torch.float32
        and torch.isfinite(term.numerator)
        and term.numerator > 0
    )
    torch.testing.assert_close(term.numerator.double(), expected, atol=3e-9, rtol=2e-5)
    (term.mean * term.weight).backward()

    assert torch.isfinite(model.quant_conv.bias.grad).all()
    assert model.quant_conv.bias.grad[model.config.z_dim :].gt(0).all()
