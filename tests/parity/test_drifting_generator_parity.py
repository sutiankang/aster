from dataclasses import replace
import math
import pytest
import torch
import torch.nn.functional as F

from aster.models.drifting import DriftingConfig, DriftingGenerator


def _author_function(weights, noise, labels, noise_labels, cfg, config, dtype):
    def linear(x, key, precise=False):
        kind = torch.float32 if precise else dtype
        return F.linear(
            x.to(kind), weights[key + ".weight"].to(kind), weights[key + ".bias"].to(kind)
        )

    def rms(x, key):
        return (
            x.float()
            * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            * weights[key + ".weight"]
        ).to(x.dtype)

    def norm(x, key, affine=False):
        if config.use_rmsnorm:
            return rms(x, key)
        return F.layer_norm(
            x,
            (x.shape[-1],),
            weights[key + ".weight"] if affine else None,
            weights[key + ".bias"] if affine else None,
            eps=1e-6,
        )

    def rotation(x):
        d, n = x.shape[-1], x.shape[1]
        work_dtype = torch.float32 if config.attn_fp32 else dtype
        freq = (1.0 / 10000 ** (torch.arange(d // 2).float() / (d // 2))).to(work_dtype)
        angles = torch.arange(n, dtype=work_dtype)[:, None] * freq
        angles = torch.cat([angles, angles], -1)[None, :, None]
        a, b = x.chunk(2, -1)
        return x * angles.cos() + torch.cat([-b, a], -1) * angles.sin()

    b, c, h, w = noise.shape
    p, grid = config.patch_size, config.input_size // config.patch_size
    condition = F.embedding(labels, weights["class_embed.weight"]).to(dtype)
    for i in range(config.noise_coords if config.noise_classes else 0):
        condition = condition + F.embedding(
            noise_labels[:, i], weights[f"noise_embeds.{i}.weight"]
        ).to(dtype)
    freq = torch.exp(-math.log(10000) * torch.arange(128).float() / 128)
    angle = cfg[:, None] * freq[None]
    guide = linear(
        F.silu(linear(torch.cat([angle.cos(), angle.sin()], -1), "cfg_embedder.0")),
        "cfg_embedder.2",
    )
    condition = (condition + 0.02 * rms(guide, "cfg_norm")).to(dtype)

    x = noise.permute(0, 2, 3, 1).to(dtype).reshape(b, grid, p, grid, p, c)
    x = x.transpose(2, 3).reshape(b, grid * grid, p * p * c)
    x = (linear(x, "patch").float() + weights["position.weight"]).to(dtype)
    if config.n_cls_tokens:
        prefix = linear(condition, "class_projection")[:, None].expand(-1, config.n_cls_tokens, -1)
        prefix = (prefix.float() + weights["class_position.weight"]).to(dtype)
        x = torch.cat([prefix, x], 1)
    for i in range(config.num_layers):
        key = f"blocks.{i}"
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            linear(F.silu(condition.float()), key + ".modulation", True).to(dtype).chunk(6, -1)
        )
        normalized = norm(x, key + ".norm1") * (1 + scale1[:, None]) + shift1[:, None]
        qkv = linear(normalized, key + ".attention.qkv").reshape(
            b, x.shape[1], 3, config.num_heads, -1
        )
        q, k, v = qkv.unbind(2)
        if config.use_qknorm:
            q, k = (
                norm(q, key + ".attention.q_norm", True),
                norm(k, key + ".attention.k_norm", True),
            )
        if config.use_rope:
            q, k = rotation(q), rotation(k)
        kind = torch.float32 if config.attn_fp32 else dtype
        q, k, v = q.to(kind), k.to(kind), v.to(kind)
        score = torch.einsum("bnhd,bmhd->bhnm", q * (q.shape[-1] ** -0.5), k)
        attended = torch.einsum("bhnm,bmhd->bnhd", score.softmax(-1), v).reshape(b, x.shape[1], -1)
        x = x + gate1[:, None] * linear(attended, key + ".attention.proj")
        normalized = norm(x, key + ".norm2") * (1 + scale2[:, None]) + shift2[:, None]
        hidden = linear(normalized, key + ".mlp.up")
        hidden = (
            F.silu(hidden) * linear(normalized, key + ".mlp.gate")
            if config.use_swiglu
            else F.gelu(hidden, approximate="none")
        )
        x = x + gate2[:, None] * linear(hidden, key + ".mlp.down")
    shift, scale = linear(F.silu(condition.float()), "modulation", True).to(dtype).chunk(2, -1)
    x = linear(norm(x, "norm") * (1 + scale[:, None]) + shift[:, None], "output")[
        :, config.n_cls_tokens :
    ]
    x = x.reshape(b, grid, grid, p, p, config.out_channels).transpose(2, 3)
    return x.reshape(b, h, w, config.out_channels).permute(0, 3, 1, 2)


@pytest.mark.parametrize("alternative", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_drifting_generator_author_graph_all_gradients(alternative, dtype):
    torch.set_num_threads(1)
    torch.manual_seed(805)
    config = DriftingConfig(
        input_size=4,
        in_channels=2,
        out_channels=2,
        patch_size=2,
        hidden_size=16,
        cond_dim=12,
        num_layers=2,
        num_heads=2,
        num_classes=3,
        noise_classes=4,
        noise_coords=2,
        n_cls_tokens=2,
    )
    if alternative:
        config = replace(
            config,
            use_rmsnorm=False,
            use_swiglu=False,
            use_rope=False,
            noise_classes=0,
            n_cls_tokens=0,
        )
    model = DriftingGenerator(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "modulation" in name or name.startswith("output."):
                parameter.normal_(std=0.12)
    weights = {
        name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()
    }
    noise = torch.randn(2, 2, 4, 4, requires_grad=True)
    independent = noise.detach().clone().requires_grad_()
    labels, noise_labels, cfg = (
        torch.tensor([0, 2]),
        torch.tensor([[0, 2], [1, 3]]),
        torch.tensor([1.2, 3.1]),
    )
    condition = dict(labels=labels, noise_labels=noise_labels) if config.noise_classes else labels
    with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
        actual = model(noise, cfg, condition).prediction
    expected = _author_function(weights, independent, labels, noise_labels, cfg, config, dtype)
    torch.testing.assert_close(
        actual,
        expected,
        atol=3e-6 if dtype == torch.float32 else 0,
        rtol=3e-5 if dtype == torch.float32 else 0,
    )
    probe = torch.randn_like(actual)
    (actual.float() * probe).sum().backward()
    (expected.float() * probe).sum().backward()
    tolerance = dict(atol=5e-5, rtol=2e-4) if dtype == torch.float32 else dict(atol=0.06, rtol=0.02)
    torch.testing.assert_close(noise.grad, independent.grad, **tolerance)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and weights[name].grad is not None, name
        torch.testing.assert_close(parameter.grad, weights[name].grad, msg=name, **tolerance)
