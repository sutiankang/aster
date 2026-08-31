from dataclasses import replace
import math
import pytest
import torch
import torch.nn.functional as F
from aster.models import (
    CosmosPredict1Config,
    CosmosPredict1Condition,
    CosmosPredict1ModelConfig,
    build_model,
)
from aster.methods.cosmos_predict1 import CosmosPredict1Objective


def source_dit(c, w, x, time, condition):
    def linear(x, name):
        return F.linear(x, w[name + ".weight"], w.get(name + ".bias"))

    def rms(x, name):
        if name + ".weight" not in w:
            return x
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            * w[name + ".weight"].float()
        ).to(x.dtype)

    def adaln(embedding, base):
        value = linear(F.silu(embedding), base + ".1")
        return linear(value, base + ".2") if c.use_adaln_lora else value

    b, channels, time_count, height, width = x.shape
    p, r = c.patch_spatial, c.patch_temporal
    t, h, v = time_count // r, height // p, width // p
    if c.concat_padding_mask:
        mask = F.interpolate(
            condition.padding_mask[:, None].to(x), size=(height, width), mode="nearest"
        )
        x = torch.cat((x, mask[:, :, None].repeat(1, 1, time_count, 1, 1)), 1)
        channels += 1
    patches = torch.einsum("bctrhmwn->bthwcrmn", x.reshape(b, channels, t, r, h, p, v, p)).reshape(
        b, t * h * v, -1
    )
    hidden = linear(patches, "x_embedder.proj.1")
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(c.model_channels // 2).float() / (c.model_channels // 2)
    )
    angles = time.float()[:, None] * frequencies
    embedded = torch.cat((angles.cos(), angles.sin()), -1).to(time.dtype)
    transformed = linear(F.silu(linear(embedded, "t_embedder.1.linear_1")), "t_embedder.1.linear_2")
    shared = transformed if c.use_adaln_lora else None
    embedded = rms(embedded if c.use_adaln_lora else transformed, "affline_norm")
    dim = c.model_channels // c.num_heads
    spatial = dim // 6 * 2
    temporal = dim - 2 * spatial
    half_phases = []
    for length, size, ratio, is_time in (
        (t, temporal, c.rope_t_extrapolation_ratio, True),
        (h, spatial, c.rope_h_extrapolation_ratio, False),
        (v, spatial, c.rope_w_extrapolation_ratio, False),
    ):
        theta = 10000 * ratio ** (size / (size - 2))
        freq = 1 / (theta ** (torch.arange(0, size, 2).float() / size))
        positions = w["pos_embedder.seq"][:length].float()
        if is_time and condition.fps is not None:
            positions = positions / condition.fps[0] * 24
        half_phases.append(torch.outer(positions, freq))
    phase = torch.cat(
        (
            half_phases[0][:, None, None].expand(t, h, v, -1),
            half_phases[1][None, :, None].expand(t, h, v, -1),
            half_phases[2][None, None].expand(t, h, v, -1),
        ),
        -1,
    )
    phase = torch.cat((phase, phase), -1).reshape(t * h * v, dim)
    extra = None
    if c.extra_per_block_abs_pos_emb:
        extra = (
            w["extra_pos_embedder.pos_emb_t"][:t, None, None]
            + w["extra_pos_embedder.pos_emb_h"][None, :h, None]
            + w["extra_pos_embedder.pos_emb_w"][None, None, :v]
        )
        norm = torch.linalg.vector_norm(extra, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(torch.tensor(1e-6), norm, alpha=1 / math.sqrt(c.model_channels))
        extra = (extra / norm.to(extra.dtype)).reshape(1, t * h * v, -1)
    for index in range(c.num_blocks):
        if extra is not None:
            hidden = hidden + extra
        for subindex, kind in enumerate(c.block_config.split("-")):
            base = f"blocks.block{index}.blocks.{subindex}"
            modulation = adaln(embedded, base + ".adaLN_modulation")
            if shared is not None:
                modulation = modulation + shared
            shift, scale, gate = modulation.chunk(3, -1)
            normalized = (
                F.layer_norm(hidden, (c.model_channels,), eps=1e-6) * (1 + scale[:, None])
                + shift[:, None]
            )
            if kind == "MLP":
                update = linear(
                    F.gelu(linear(normalized, base + ".block.layer1"), approximate="none"),
                    base + ".block.layer2",
                )
            else:
                attention = base + ".block.attn"
                context = condition.text_embeddings if kind == "CA" else normalized

                def proj(x, name):
                    return rms(
                        linear(x, attention + "." + name + ".0").reshape(
                            b, x.shape[1], c.num_heads, dim
                        ),
                        attention + "." + name + ".1",
                    ).transpose(1, 2)

                q, k, value = proj(normalized, "to_q"), proj(context, "to_k"), proj(context, "to_v")
                if kind == "FA":

                    def rotate(x):
                        left, right = x.chunk(2, -1)
                        return (
                            x * phase.cos()[None, None]
                            + torch.cat((-right, left), -1) * phase.sin()[None, None]
                        )

                    q, k = rotate(q), rotate(k)
                update = (
                    F.scaled_dot_product_attention(q, k, value)
                    .transpose(1, 2)
                    .reshape(b, t * h * v, c.model_channels)
                )
                update = linear(update, attention + ".to_out.0")
            hidden = hidden + gate[:, None] * update
    affine = adaln(embedded, "final_layer.adaLN_modulation")
    if shared is not None:
        affine = affine + shared[:, : 2 * c.model_channels]
    shift, scale = affine.chunk(2, -1)
    hidden = (
        F.layer_norm(hidden, (c.model_channels,), eps=1e-6) * (1 + scale[:, None]) + shift[:, None]
    )
    output = linear(hidden, "final_layer.linear").reshape(b, t, h, v, p, p, r, c.out_channels)
    return torch.einsum("bthwpmrc->bctrhpwm", output).reshape(
        b, c.out_channels, time_count, height, width
    )


@pytest.mark.parametrize("lora,extra", [(True, True), (False, False)])
def test_models_cosmos_predict1_official_layout_adaln_ntk_all_parameter_and_input_gradients(
    lora, extra
):
    torch.set_num_threads(1)
    torch.manual_seed(552)
    c = CosmosPredict1Config(
        use_adaln_lora=lora,
        extra_per_block_abs_pos_emb=extra,
        patch_temporal=2,
        rope_h_extrapolation_ratio=1.25,
        rope_w_extrapolation_ratio=1.5,
        rope_t_extrapolation_ratio=2.0,
    )
    native = build_model(c)

    with torch.no_grad():
        for parameter in native.parameters():
            parameter.uniform_(-0.2, 0.2)
    weights = {
        name: value.detach().clone().requires_grad_(name in dict(native.named_parameters()))
        for name, value in native.state_dict().items()
    }
    x = torch.randn(2, 2, 4, 4, 6, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    time = torch.tensor([-0.7, 0.25], requires_grad=True)
    time_ref = time.detach().clone().requires_grad_()
    text = torch.randn(2, 3, c.crossattn_emb_channels, requires_grad=True)
    text_ref = text.detach().clone().requires_grad_()
    condition = CosmosPredict1Condition(text, torch.tensor([30.0, 30.0]), torch.randn(2, 2, 3))
    expected = source_dit(c, weights, other, time_ref, replace(condition, text_embeddings=text_ref))
    actual = native(x, time, condition).prediction
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=7e-5)
    factor = torch.randn_like(actual) / actual.numel()
    (actual * factor).sum().backward()
    (expected * factor).sum().backward()
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter.grad, weights[name].grad, atol=5e-6, rtol=9e-4, msg=name
        )
    for left, right in ((x, other), (time, time_ref), (text, text_ref)):
        torch.testing.assert_close(left.grad, right.grad, atol=4e-6, rtol=6e-4)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_models_cosmos_predict1_official_kendall_fourier_and_weighted_loss(reduction):
    torch.set_num_threads(1)
    torch.manual_seed(553)
    c = CosmosPredict1ModelConfig()
    model = build_model(c)
    weights = {
        name: value.detach().clone().requires_grad_(name in dict(model.named_parameters()))
        for name, value in model.state_dict().items()
    }
    clean = torch.randn(2, 2, 2, 4, 4)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0.3, 0.7])
    condition = CosmosPredict1Condition(
        torch.randn(2, 3, c.net.crossattn_emb_channels),
        torch.tensor([24.0, 24.0]),
        torch.zeros(2, 4, 4),
    )
    sample_weights = torch.tensor([0.5, 2.0])
    mask = torch.rand_like(clean)
    batch = dict(
        sample=clean,
        sigma=sigma,
        noise=noise,
        condition=condition,
        sample_weight=sample_weights,
        loss_mask=mask,
    )
    actual = CosmosPredict1Objective(loss_reduce=reduction, loss_scale=0.3)(model, batch).mean
    noisy = clean + sigma[:, None, None, None, None] * noise
    denom = sigma.square() + 0.25
    time = sigma.log() / 4
    residual = source_dit(
        c.net,
        {key[4:]: value for key, value in weights.items() if key.startswith("net.")},
        noisy / denom.sqrt()[:, None, None, None, None],
        time,
        condition,
    )
    predicted = (
        0.25 / denom[:, None, None, None, None] * noisy
        + (sigma * 0.5 / denom.sqrt())[:, None, None, None, None] * residual
    )
    fourier = (
        time[:, None] * weights["logvar.0.freqs"] + weights["logvar.0.phases"]
    ).cos() * math.sqrt(2)
    logvar = F.linear(fourier, weights["logvar.1.weight"]).flatten()
    mse = ((predicted - clean).square() * mask).flatten(1)
    error = (
        mse * (denom / (sigma * 0.5).square() * sample_weights * torch.exp(-logvar))[:, None]
        + logvar[:, None]
    )
    expected = (error.sum(-1) if reduction == "sum" else error.mean(-1)).mean() * 0.3
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    actual.backward()
    expected.backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, weights[name].grad, atol=9e-5, rtol=2e-3, msg=name
        )
