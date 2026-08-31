from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from aster.models.adversarial import ActNorm2d, PatchDiscriminator, PatchDiscriminatorConfig
from aster.core import LossTerm
from aster.training import Trainer


def test_actnorm_source_statistics_reverse_and_logdet_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(531)
    x = torch.randn(3, 4, 5, 6, dtype=torch.float64, requires_grad=True)
    module = ActNorm2d(4, logdet=True).double()
    with pytest.raises(ValueError, match="explicit calibration"):
        module(x)
    module.initialize(x.detach())
    flattened = x.detach().permute(1, 0, 2, 3).reshape(4, -1)
    loc = -flattened.mean(1).reshape(1, 4, 1, 1)
    scale = (flattened.std(1, correction=1) + 1e-6).reciprocal().reshape(1, 4, 1, 1)
    torch.testing.assert_close(module.affine.loc, loc, atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(module.affine.scale, scale, atol=1e-14, rtol=1e-13)
    actual, logdet = module(x)
    expected = scale * (x + loc)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        logdet, scale.abs().log().sum() * 30 * torch.ones(3, dtype=torch.float64)
    )
    torch.testing.assert_close(module(actual, reverse=True), x)
    actual.sum().backward()
    torch.testing.assert_close(x.grad, scale.expand_as(x))
    with pytest.raises(ValueError, match="already initialized"):
        module.initialize(x.detach())


@pytest.mark.parametrize("normalization", ["batchnorm", "actnorm"])
def test_patchgan_official_convolutions_normalization_and_state_mapping(normalization):
    torch.set_num_threads(1)
    torch.manual_seed(728)
    model = PatchDiscriminator(
        PatchDiscriminatorConfig(base_channels=4, num_layers=2, normalization=normalization)
    )
    images = torch.randn(2, 3, 32, 32, requires_grad=True)
    if normalization == "actnorm":
        model.initialize(images.detach())
    model.eval()
    expected = images
    for layer in model.main:
        if isinstance(layer, torch.nn.Conv2d):
            expected = F.conv2d(expected, layer.weight, layer.bias, layer.stride, layer.padding)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            expected = F.batch_norm(
                expected,
                layer.running_mean,
                layer.running_var,
                layer.weight,
                layer.bias,
                training=False,
                eps=1e-5,
            )
        elif isinstance(layer, ActNorm2d):
            expected = layer.affine.scale * (expected + layer.affine.loc)
        else:
            expected = F.leaky_relu(expected, 0.2)
    actual = model(images)
    assert actual.shape == (2, 1, 6, 6)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual.square().sum(), (images, *model.parameters()), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (images, *model.parameters()))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)
    state = {name.replace(".affine.", "."): value for name, value in model.state_dict().items()}
    clone = PatchDiscriminator(model.config).load_reference_state(state).eval()
    torch.testing.assert_close(clone(images), actual, rtol=0, atol=0)
    bad = deepcopy(state)
    bad[next(iter(bad))] = torch.zeros(1)
    previous = deepcopy(clone.state_dict())
    with pytest.raises(ValueError, match="Invalid reference"):
        clone.load_reference_state(bad)
    assert all(torch.equal(v, clone.state_dict()[k]) for k, v in previous.items())


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_initialized_actnorm_zero3_uses_real_parameter_leaves_and_resume(precision, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(127)
    model = PatchDiscriminator(PatchDiscriminatorConfig(base_channels=4, num_layers=1))
    images = torch.randn(2, 3, 16, 16)
    model.initialize(images)

    class Objective:
        def config_dict(self):
            return {"type": "patchgan_squared_test_not_adversarial_training"}

        def __call__(self, model, batch):
            output = model(batch["sample"]).float()
            return LossTerm(
                output.square().sum(), output.new_tensor(output.numel()), "patch", "squared"
            )

    engine = Trainer(model, Objective(), zero_stage=3, precision=precision, max_grad_norm=None)
    result = engine.step([{"sample": images}])
    assert result.updated
    checkpoint = engine.save_checkpoint(tmp_path / precision)
    expected = engine.step([{"sample": images}])
    weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
    engine.load_checkpoint(checkpoint, trusted=True)
    actual = engine.step([{"sample": images}])
    assert expected.loss == actual.loss
    for name, value in engine.export_state_dict(only_rank_zero=False).items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


def test_default_patchgan_receptive_field_and_small_input_rejection():
    torch.set_num_threads(1)
    model = PatchDiscriminator(
        PatchDiscriminatorConfig(base_channels=2, normalization="batchnorm")
    ).eval()
    assert model(torch.zeros(1, 3, 256, 256)).shape == (1, 1, 30, 30)
    with pytest.raises(ValueError, match="too small"):
        model(torch.zeros(1, 3, 23, 23))


def test_patchgan_local_registered_save_load_preserves_calibrated_state(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(515)
    model = PatchDiscriminator(PatchDiscriminatorConfig(base_channels=4, num_layers=1))
    images = torch.randn(2, 3, 16, 16)
    model.initialize(images)
    expected = model(images)
    model.save_pretrained(tmp_path / "discriminator")
    restored = PatchDiscriminator.from_pretrained(tmp_path / "discriminator")
    assert all(bool(m.initialized) for m in restored.modules() if isinstance(m, ActNorm2d))
    torch.testing.assert_close(restored(images), expected, atol=0, rtol=0)
