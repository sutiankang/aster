import copy
import math

import pytest
import torch
import torch.nn.functional as F

from aster.models.video_world import WanVideoConfig, WanVideoDiT


def explicit_reference(model, sample, time, condition):
    c = model.config
    linear = lambda x, m: F.linear(x, m.weight, m.bias)
    norm = lambda x, m: F.layer_norm(x, m.normalized_shape, m.weight, m.bias, m.eps)
    rms = lambda x, m: (
        x / (x.square().mean(-1, keepdim=True) + m.eps).sqrt() * m.weight
        if hasattr(m, "eps")
        else x
    )
    b = len(sample)
    image = None
    if c.image_conditioned:
        sample = torch.cat((sample, condition["video_condition"]), 1)
        image = condition["image_features"]
        if c.first_last_frames:
            image = image + model.image_position.weight
        image = norm(
            linear(
                F.gelu(linear(norm(image, model.image_projection[0]), model.image_projection[1])),
                model.image_projection[3],
            ),
            model.image_projection[4],
        )
    patch = model.patch_embedding
    x = F.conv3d(sample, patch.weight, patch.bias, stride=c.patch_size)
    grid = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)
    text = F.pad(condition["text"], (0, 0, 0, c.text_length - condition["text"].shape[1]))
    text = linear(
        F.gelu(linear(text, model.text_embedding[0]), approximate="tanh"), model.text_embedding[2]
    )
    half = c.frequency_dim // 2
    angles = time.double()[:, None] * c.time_scale * 10000 ** (-torch.arange(half).double() / half)
    e = linear(
        F.silu(
            linear(torch.cat((angles.cos(), angles.sin()), -1).float(), model.time_embedding[0])
        ),
        model.time_embedding[2],
    )
    modulation = linear(F.silu(e), model.time_projection[1]).reshape(b, 6, c.hidden_size)

    def rotate(values):
        dimensions = c.hidden_size // c.num_heads
        axis_dimensions = (
            dimensions - 4 * (dimensions // 6),
            2 * (dimensions // 6),
            2 * (dimensions // 6),
        )
        frequencies = []
        coords = torch.meshgrid(*(torch.arange(size) for size in grid), indexing="ij")
        for coordinates, d in zip(coords, axis_dimensions):
            angle = coordinates.flatten().double()[:, None] / 10000 ** (
                torch.arange(0, d, 2).double() / max(d, 1)
            )
            frequencies.append(torch.polar(torch.ones_like(angle), angle))
        frequency = torch.cat(frequencies, -1)[None, :, None]
        complex_value = torch.view_as_complex(
            values.double().reshape(*values.shape[:-1], dimensions // 2, 2)
        )
        return torch.view_as_real(complex_value * frequency).flatten(-2).float()

    def attention(m, values, source, *, self_attention=False):
        split = lambda tensor: tensor.reshape(b, -1, c.num_heads, c.hidden_size // c.num_heads)
        q, k, v = (
            split(rms(linear(values, m.q), m.norm_q)),
            split(rms(linear(source, m.k), m.norm_k)),
            split(linear(source, m.v)),
        )
        if self_attention:
            q, k = rotate(q), rotate(k)
        scores = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(q.shape[-1])
        if self_attention and c.window != (-1, -1):
            pos = torch.arange(values.shape[1])
            mask = torch.ones_like(scores, dtype=torch.bool)
            if c.window[0] >= 0:
                mask &= pos[None, :] >= pos[:, None] - c.window[0]
            if c.window[1] >= 0:
                mask &= pos[None, :] <= pos[:, None] + c.window[1]
            scores = scores.masked_fill(~mask, -torch.inf)
        out = torch.einsum("bhij,bjhd->bihd", scores.softmax(-1), v)
        if c.image_conditioned and not self_attention:
            ki, vi = split(rms(linear(image, m.k_img), m.norm_k_img)), split(linear(image, m.v_img))
            image_scores = torch.einsum("bihd,bjhd->bhij", q, ki) / math.sqrt(q.shape[-1])
            out = out + torch.einsum("bhij,bjhd->bihd", image_scores.softmax(-1), vi)
        return linear(out.flatten(2), m.o)

    for block in model.blocks:
        shift, scale, gate, shift_ffn, scale_ffn, gate_ffn = (
            block.modulation.weight + modulation
        ).unbind(1)
        h = norm(x, block.norm1) * (1 + scale[:, None]) + shift[:, None]
        x = x + attention(block.self_attn, h, h, self_attention=True) * gate[:, None]
        normalized = norm(x, block.norm3) if c.cross_attention_norm else x
        x = x + attention(block.cross_attn, normalized, text)
        h = norm(x, block.norm2) * (1 + scale_ffn[:, None]) + shift_ffn[:, None]
        x = (
            x
            + linear(F.gelu(linear(h, block.ffn[0]), approximate="tanh"), block.ffn[2])
            * gate_ffn[:, None]
        )
    shift, scale = (model.head.modulation.weight + e[:, None]).unbind(1)
    x = linear(norm(x, model.head.norm) * (1 + scale[:, None]) + shift[:, None], model.head.head)
    x = x.reshape(b, *grid, *c.patch_size, c.latent_channels)
    return torch.einsum("bfhwpqrc->bcfphqwr", x).reshape(
        b, c.latent_channels, *(g * p for g, p in zip(grid, c.patch_size))
    )


@pytest.mark.parametrize(
    "image,first_last,window",
    [(False, False, (-1, -1)), (True, False, (-1, -1)), (True, True, (2, 1))],
)
def test_wan_full_native_vs_functional_forward_all_gradients(image, first_last, window):
    torch.manual_seed(331)
    torch.set_num_threads(1)
    config = WanVideoConfig(
        latent_channels=2,
        condition_channels=3 if image else 0,
        hidden_size=24,
        intermediate_size=32,
        num_heads=2,
        num_layers=2,
        text_dim=8,
        text_length=5,
        frequency_dim=8,
        image_dim=6,
        image_conditioned=image,
        first_last_frames=first_last,
        image_tokens_per_frame=2,
        window=window,
    )
    native = WanVideoDiT(config)
    torch.nn.init.normal_(native.head.head.weight, std=0.05)
    reference = copy.deepcopy(native)
    sample = torch.randn(2, 2, 2, 4, 4, requires_grad=True)
    condition = {"text": torch.randn(2, 3, 8, requires_grad=True)}
    if image:
        condition.update(
            image_features=torch.randn(2, 4, 6, requires_grad=True),
            video_condition=torch.randn(2, 3, 2, 4, 4, requires_grad=True),
        )
    expected_sample = sample.detach().clone().requires_grad_()
    expected_condition = {k: v.detach().clone().requires_grad_() for k, v in condition.items()}
    time = torch.tensor([0.13, 0.74])
    actual = native(sample, time, condition).prediction
    expected = explicit_reference(reference, expected_sample, time, expected_condition)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    cotangent = torch.randn_like(actual)
    (actual * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    for (name, param), (_, ref_param) in zip(
        native.named_parameters(), reference.named_parameters()
    ):
        assert param.grad is not None, name
        torch.testing.assert_close(param.grad, ref_param.grad, atol=1e-5, rtol=3e-4, msg=name)
    torch.testing.assert_close(sample.grad, expected_sample.grad, atol=3e-6, rtol=3e-4)
    for key in condition:
        torch.testing.assert_close(
            condition[key].grad, expected_condition[key].grad, atol=1e-5, rtol=3e-4
        )

    with torch.no_grad():
        padded = native(sample, time, condition, sequence_length=11).prediction
    torch.testing.assert_close(padded, actual, atol=3e-6, rtol=3e-5)
