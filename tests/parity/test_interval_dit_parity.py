import math
import pytest
import torch
import torch.nn.functional as F

from aster.models.interval_dit import IntervalDiTConfig, IntervalDiT


def _formula(parameters, position, sample, time, duration, labels, config):
    def linear(value, name):
        return F.linear(value, parameters[name + ".weight"], parameters[name + ".bias"])

    def embed(value, name):
        frequency = (-math.log(10000) * torch.arange(128) / 128).exp()
        angles = value[:, None] * frequency[None]
        return linear(
            F.silu(linear(torch.cat([angles.cos(), angles.sin()], -1), name + ".0")), name + ".2"
        )

    def normalize(value):
        return F.layer_norm(value, (value.shape[-1],), eps=1e-6)

    x = (
        F.conv2d(
            sample, parameters["patch.weight"], parameters["patch.bias"], stride=config.patch_size
        )
        .flatten(2)
        .transpose(1, 2)
        + position
    )
    conditioning = (
        embed(time, "time")
        + embed(duration, "interval")
        + F.embedding(labels, parameters["classes.weight"])
    )
    for index in range(config.num_layers):
        key = f"blocks.{index}"
        s1, a1, g1, s2, a2, g2 = [
            v[:, None] for v in linear(F.silu(conditioning), key + ".modulation.1").chunk(6, -1)
        ]
        v = normalize(x) * (1 + a1) + s1
        q, k, value = [
            item.reshape(len(x), x.shape[1], config.num_heads, -1)
            for item in linear(v, key + ".qkv").chunk(3, -1)
        ]
        power = -0.5 if config.variant == "meanflow" else -1.0
        attention = torch.einsum("bqhd,bkhd->bhqk", q * q.shape[-1] ** power, k).softmax(-1)
        combined = torch.einsum("bhqk,bkhd->bqhd", attention, value).reshape_as(x)
        x = x + g1 * linear(combined, key + ".projection")
        hidden = F.gelu(linear(normalize(x) * (1 + a2) + s2, key + ".mlp.0"), approximate="tanh")
        x = x + g2 * linear(hidden, key + ".mlp.2")
    shift, scale = linear(F.silu(conditioning), "modulation.1").chunk(2, -1)
    x = linear(normalize(x) * (1 + scale[:, None]) + shift[:, None], "output")
    p, grid = config.patch_size, config.input_size // config.patch_size
    x = x.reshape(len(x), grid, grid, p, p, config.in_channels).permute(0, 5, 1, 3, 2, 4)
    return x.reshape_as(sample)


@pytest.mark.parametrize("variant", ["meanflow", "shortcut"])
def test_interval_dit_author_formula_and_all_gradients(variant):
    torch.set_num_threads(1)
    torch.manual_seed(515)
    config = IntervalDiTConfig(
        variant=variant,
        input_size=4,
        in_channels=2,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        num_classes=3,
    )
    model = IntervalDiT(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "modulation" in name or name.startswith("output."):
                parameter.normal_(std=0.1)
    parameters = {
        name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()
    }
    x = torch.randn(2, 2, 4, 4, requires_grad=True)
    independent = x.detach().clone().requires_grad_()
    t, h, labels = (
        torch.tensor([0.8, 0.4]),
        torch.tensor([0.3, 0.0]) if variant == "meanflow" else torch.tensor([1.0, 3.0]),
        torch.tensor([0, 2]),
    )
    actual = model(x, t, h, labels).prediction
    expected = _formula(parameters, model.position, independent, t, h, labels, config)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, independent.grad, atol=1e-5, rtol=2e-4)
    for name, value in model.named_parameters():
        torch.testing.assert_close(
            value.grad, parameters[name].grad, atol=1e-5, rtol=2e-4, msg=name
        )
