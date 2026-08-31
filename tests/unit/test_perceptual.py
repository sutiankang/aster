from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from aster.models.perceptual import LPIPS, LPIPSConfig
from aster.models.generative import AutoencoderKL, AutoencoderConfig
from aster.methods.perceptual_autoencoder import (
    PerceptualAutoencoderObjective,
    PerceptualAutoencoderMethod,
)
from aster.training import Trainer


def tiny(backbone="vgg", **kwargs):
    return LPIPS(
        LPIPSConfig(backbone=backbone, channels=(2, 3, 4, 4, 4), allow_untrained=True, **kwargs)
    )


def formula(model, left, right):
    c = model.config
    if c.version == "0.1":
        shift = torch.tensor([-0.030, -0.088, -0.188], dtype=torch.float32).to(left)[
            None, :, None, None
        ]
        scale = torch.tensor([0.458, 0.448, 0.450], dtype=torch.float32).to(left)[
            None, :, None, None
        ]
        left, right = (left - shift) / scale, (right - shift) / scale
    ends = (3, 8, 15, 22, 29) if c.backbone == "vgg" else (1, 4, 7, 9, 11)
    features = []
    for x in (left, right):
        outputs = []
        for index, layer in enumerate(model.features):
            if isinstance(layer, torch.nn.Conv2d):
                x = F.conv2d(x, layer.weight, layer.bias, layer.stride, layer.padding)
            elif isinstance(layer, torch.nn.MaxPool2d):
                x = F.max_pool2d(x, layer.kernel_size, layer.stride)
            else:
                x = torch.relu(x)
            if index in ends:
                outputs.append(x)
        features.append(outputs)
    distances = []
    for index, (a, b) in enumerate(zip(*features)):
        a = a / (a.square().sum(1, keepdim=True).sqrt() + 1e-10)
        b = b / (b.square().sum(1, keepdim=True).sqrt() + 1e-10)
        d = (a - b).square()
        d = (
            (d * model.linear[index].weight.view(1, -1, 1, 1)).sum(1, keepdim=True)
            if c.learned
            else d.sum(1, keepdim=True)
        )
        distances.append(
            F.interpolate(d, left.shape[-2:], mode="bilinear", align_corners=False)
            if c.spatial
            else d.mean((2, 3), keepdim=True)
        )
    return sum(distances), distances


@pytest.mark.parametrize("backbone", ["vgg", "alex"])
@pytest.mark.parametrize(
    "version,learned,spatial", [("0.1", True, False), ("0.0", True, True), ("0.1", False, False)]
)
def test_native_lpips_architecture_formula_and_input_gradients(backbone, version, learned, spatial):
    torch.set_num_threads(1)
    torch.manual_seed(807)
    model = tiny(backbone, version=version, learned=learned, spatial=spatial).double()

    with torch.no_grad():
        for layer in model.features:
            if isinstance(layer, torch.nn.Conv2d):
                layer.bias.fill_(0.3)
    size = 16 if backbone == "vgg" else 32
    left = torch.randn(2, 3, size, size, dtype=torch.float64, requires_grad=True)
    right = torch.randn_like(left, requires_grad=True)
    actual, levels = model(left, right, return_layers=True)
    expected, expected_levels = formula(model, left, right)
    assert model.endpoints == ((3, 8, 15, 22, 29) if backbone == "vgg" else (1, 4, 7, 9, 11))
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    for a, b in zip(levels, expected_levels):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
    actual_grads = torch.autograd.grad(actual.sum(), (left, right), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (left, right))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=2e-11, rtol=1e-9)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    torch.testing.assert_close(model((left + 1) / 2, (right + 1) / 2, normalize=True), actual)


def test_lpips_zero_feature_gradient_is_finite_and_amp_keeps_frozen_fp32():
    torch.set_num_threads(1)
    model = tiny()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    left = torch.rand(1, 3, 16, 16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        distance = model(left, torch.zeros_like(left))
    distance.sum().backward()
    assert distance.dtype == torch.float32 and distance.item() == 0
    assert torch.isfinite(left.grad).all() and torch.count_nonzero(left.grad) == 0
    model.train()
    assert not model.training


def test_lpips_complete_local_weight_import_and_atomic_rejection(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(55)
    c = LPIPSConfig(backbone="alex", channels=(2, 3, 4, 4, 4))
    model = LPIPS(c)
    x = torch.randn(1, 3, 32, 32)
    with pytest.raises(RuntimeError, match="weights not loaded"):
        model(x, x)
    backbone = {
        f"features.{name}": value.clone() for name, value in model.features.state_dict().items()
    }
    calibration = {
        f"lin{i}.model.1.weight": layer.weight.clone() for i, layer in enumerate(model.linear)
    }
    before = model.weight_identity()
    bad = deepcopy(backbone)
    bad["features.0.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="Invalid LPIPS tensor"):
        model.load_reference_weights(bad, calibration)
    assert model.weight_identity() == before
    with pytest.raises(ValueError, match="exactly one"):
        model.load_reference_weights(backbone, {})
    model.load_reference_weights(backbone, calibration)
    assert bool(model.weights_loaded) and not model.config.standard_architecture
    assert model(x, x).item() == 0
    model.save_pretrained(tmp_path / "metric")
    restored = LPIPS.from_pretrained(tmp_path / "metric")
    assert restored.weight_identity() == model.weight_identity()
    torch.testing.assert_close(model(x, -x), restored(x, -x), atol=0, rtol=0)


def vae_config():
    return AutoencoderConfig(
        base_channels=4, latent_channels=2, channel_mult=(1, 2), num_res_blocks=1
    )


@pytest.mark.parametrize("stage,precision", [(0, "fp32"), (3, "fp32"), (3, "bf16")])
def test_perceptual_training_frozen_role_and_exact_checkpoint(stage, precision, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(187)
    engine = Trainer(
        AutoencoderKL(vae_config()),
        zero_stage=stage,
        precision=precision,
        accumulation_steps=2,
        max_grad_norm=None,
        ema_decay=0.9,
    )
    method = PerceptualAutoencoderMethod(engine, tiny(), pixel_reduction="mean")
    samples = [{"sample": torch.rand(n, 3, 16, 16) * 2 - 1} for n in (1, 2)]
    frozen = method.perceptual.weight_identity()
    result = method.update(samples)
    assert result.updated and method.updates == 1 and method.perceptual.weight_identity() == frozen
    assert (
        engine.last_successful_update()["objective_configuration"]["configuration"][
            "perceptual_weights"
        ]
        == frozen
    )
    checkpoint = engine.save_checkpoint(tmp_path / f"zero{stage}_{precision}")
    expected = method.update(samples)
    weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
    engine.load_checkpoint(checkpoint, trusted=True)
    actual = method.update(samples)
    assert expected.loss == actual.loss and method.updates == 2
    for key, value in engine.export_state_dict(only_rank_zero=False).items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    assert all(p.grad is None for p in method.perceptual.parameters())


def test_perceptual_nll_broadcast_and_separate_global_denominators():
    torch.set_num_threads(1)
    torch.manual_seed(15)
    model, metric = AutoencoderKL(vae_config()), tiny()
    objective = PerceptualAutoencoderObjective(
        metric, logvar=0.3, kl_weight=0.2, perceptual_weight=0.7, sample_posterior=False
    )
    clean = torch.rand(2, 3, 16, 16) * 2 - 1
    posterior = model.encode(clean)
    reconstruction = model.decode(posterior.mode())
    expected_nll = (
        ((reconstruction - clean).abs() + 0.7 * metric(clean, reconstruction))
        * torch.exp(torch.tensor(-0.3))
        + 0.3
    ).sum() / 2
    terms = objective(model, {"sample": clean}).terms
    torch.testing.assert_close(terms[0].mean, expected_nll)
    torch.testing.assert_close(terms[1].mean, posterior.kl().mean())
    assert terms[0].unit == "sample" and terms[1].weight == 0.2
    sum(term.mean * term.weight for term in terms).backward()
    assert torch.isfinite(model.decoder[-1].weight.grad).all()


def test_perceptual_whole_window_preflight_rejects_before_encoder_or_weight_update():
    torch.set_num_threads(1)
    engine = Trainer(AutoencoderKL(vae_config()), accumulation_steps=2, zero_stage=3)
    method = PerceptualAutoencoderMethod(engine, tiny())
    good = {"sample": torch.zeros(1, 3, 16, 16)}
    bad = {"sample": torch.full((1, 3, 16, 16), 2.0)}
    calls = []
    hook = engine.model.encoder.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        with pytest.raises(ValueError, match="normalized"):
            method.update([good, bad])
    finally:
        hook.remove()
    assert not calls and not engine._failed and method.updates == 0
    with torch.no_grad():
        next(method.perceptual.parameters()).add_(1)
    with pytest.raises(ValueError, match="weights changed"):
        method.update([good, good])
    assert not engine._failed and method.updates == 0
