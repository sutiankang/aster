from copy import deepcopy
from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from aster.models.cosmos3_audio import Cosmos3AudioConfig, Cosmos3AudioCodec, AudioGaussian
from aster.methods.cosmos3 import Cosmos3AudioAutoencoderObjective
from aster.training import Trainer


def test_models_avae2_padding_stereo_peak_semantics_and_source_gaussian():
    torch.set_num_threads(1)
    torch.manual_seed(684)
    c = Cosmos3AudioConfig()
    model = Cosmos3AudioCodec(c).eval()
    x = torch.randn(2, 2, 47)
    posterior = model.encode(x)
    assert posterior.mean.shape == (2, 4, 4)
    raw = Cosmos3AudioCodec(replace(c, normalize_volume=False)).eval()
    raw.load_state_dict(model.state_dict())
    expected = raw.encode(x / (x.abs().max() + 1e-5) * 0.95)
    torch.testing.assert_close(posterior.parameters, expected.parameters, atol=0, rtol=0)
    torch.testing.assert_close(
        model.decode(posterior.mode()[0]),
        model.decode(posterior.mode()[:1])[0],
        atol=1e-7,
        rtol=1e-6,
    )

    changed = x.clone()
    changed[1] *= 10
    assert not torch.allclose(model.encode(changed).mean[0], posterior.mean[0])
    moments = torch.randn(2, 8, 3)
    gaussian = AudioGaussian(moments)
    mean, scale = moments.chunk(2, 1)
    std = F.softplus(scale) + 1e-4
    torch.testing.assert_close(
        gaussian.kl(), (mean.square() + std.square() - std.square().log() - 1).sum(1).mean()
    )
    one = gaussian.sample(torch.Generator().manual_seed(7))
    two = gaussian.sample(torch.Generator().manual_seed(7))
    torch.testing.assert_close(one, two, atol=0, rtol=0)
    with pytest.raises(ValueError, match="Decoder-only"):
        Cosmos3AudioCodec(replace(c, encoder_enabled=False)).encode(x)
    with pytest.raises(ValueError, match="compression"):
        replace(c, hop_size=24)
    with pytest.raises(ValueError, match="channels"):
        model.encode(x[:, :1])


@pytest.mark.parametrize("stage", [0, 3])
def test_models_avae2_true_training_unequal_accumulation_rng_exact_restore(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(685)
    c = Cosmos3AudioConfig(normalize_volume=False)
    model = Cosmos3AudioCodec(c)
    objective = Cosmos3AudioAutoencoderObjective(sample_posterior=False)
    trainer = Trainer(model, objective, zero_stage=stage, lr=0.003)
    batch = {"sample": torch.sin(torch.arange(48).float() * 0.3)[None, None].expand(1, 2, -1) * 0.2}
    first = trainer.step([batch]).loss
    for _ in range(14):
        last = trainer.step([batch]).loss
    assert last < first * 0.85
    checkpoint = trainer.save_checkpoint(tmp_path / "checkpoint")
    stochastic = Cosmos3AudioAutoencoderObjective(sample_posterior=True)
    expected = trainer.phase("sample_posterior", objective=stochastic, microbatches=[batch])
    weights = deepcopy(trainer.export_state_dict())
    trainer.load_checkpoint(checkpoint, trusted=True)
    actual = trainer.phase("sample_posterior", objective=stochastic, microbatches=[batch])
    assert actual.loss == expected.loss
    for name, value in trainer.export_state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)

    batches = [{"sample": torch.randn(1, 2, 35) * 0.2}, {"sample": torch.randn(2, 2, 48) * 0.2}]
    reference = Cosmos3AudioCodec(c)
    accumulated = Cosmos3AudioCodec(c)
    accumulated.load_state_dict(reference.state_dict())
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.0001)
    terms = [objective(reference, b).terms for b in batches]
    total = sum(
        a.weight * (a.numerator + b.numerator) / (a.denominator + b.denominator)
        for a, b in zip(*terms)
    )
    total.backward()
    optimizer.step()
    engine = Trainer(
        accumulated,
        objective,
        zero_stage=stage,
        accumulation_steps=2,
        max_grad_norm=None,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.0001),
    )
    result = engine.step(batches)
    assert abs(result.loss - total.item()) < 1e-6
    for name, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=2e-7, rtol=3e-5)


def test_models_avae2_objective_rejects_batch_coupled_normalization_and_finite_kl():
    torch.set_num_threads(1)
    objective = Cosmos3AudioAutoencoderObjective(sample_posterior=False)
    with pytest.raises(ValueError, match="normalize_volume=False"):
        objective(Cosmos3AudioCodec(Cosmos3AudioConfig()), {"sample": torch.randn(1, 2, 48)})
    model = Cosmos3AudioCodec(Cosmos3AudioConfig(normalize_volume=False)).bfloat16()
    terms = objective(model, {"sample": torch.randn(1, 2, 48).bfloat16()}).terms
    assert terms[1].numerator.dtype == torch.float32 and terms[1].numerator >= 0
    sum(t.weight * t.numerator / t.denominator for t in terms).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_models_avae2_public_config_strict_parser_and_bfloat16_artifact(tmp_path):
    from aster.models import build_model, load_model

    source = dict(
        model_type="autoencoder_v2",
        sampling_rate=48000,
        vocoder_input_dim=64,
        dec_dim=320,
        dec_c_mults=[1, 2, 4, 8, 16],
        dec_strides=[2, 4, 5, 6, 8],
        dec_out_channels=2,
        enc_dim=192,
        enc_num_blocks=2,
        enc_n_fft=64,
        enc_hop_length=16,
        enc_latent_dim=128,
        enc_c_mults=[1, 2, 4],
        enc_strides=[4, 5, 6],
        enc_intermediate_dim=768,
        enc_num_layers=12,
        hop_size=1920,
        bottleneck={"type": "vae"},
        causal=False,
        latent_mean=None,
        latent_std=None,
    )
    real = Cosmos3AudioConfig.from_diffusers_config(source)
    assert real.sampling_rate / real.hop_size == 25 and real.enc_latent_dim == 128
    with pytest.raises(ValueError, match="causal"):
        Cosmos3AudioConfig.from_diffusers_config(dict(source, causal=True))
    with pytest.raises(ValueError, match="Unknown"):
        Cosmos3AudioConfig.from_diffusers_config(dict(source, imaginary_encoder=True))
    torch.set_num_threads(1)
    torch.manual_seed(692)
    model = build_model(Cosmos3AudioConfig()).bfloat16().eval()
    x = torch.randn(1, 2, 47).bfloat16()
    output, _ = model(x)
    model.save_pretrained(tmp_path / "audio")
    restored = load_model(tmp_path / "audio").eval()
    torch.testing.assert_close(output, restored(x)[0], atol=0, rtol=0)
    assert all(p.dtype == torch.bfloat16 for p in restored.parameters())
