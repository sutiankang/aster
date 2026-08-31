import pytest
import torch
import torch.nn.functional as F
from aster.models import KimiK3VisionConfig, KimiK3Config, build_model
from test_models_k3_parity import _full


def _vision(pixels, grids, weights, c):
    x = F.conv2d(pixels, weights["patch_embed.proj.weight"], stride=c.patch_size).flatten(1)
    positions, phases, lengths = [], [], []
    table = weights["patch_embed.pos_emb.weight"]
    d = c.qkv_hidden_size // c.num_attention_heads
    for t, h, w in grids.tolist():
        spatial = (
            table.flatten(0, 1)
            if (h, w) == table.shape[:2]
            else F.interpolate(table.permute(2, 0, 1)[None], (h, w), mode="bilinear")[0]
            .permute(1, 2, 0)
            .reshape(h * w, -1)
        )
        if t == 1:
            positions.append(spatial)
        else:
            angle = torch.arange(t).float()[:, None] * 10000 ** (
                -torch.arange(c.hidden_size // 2).float()[None] / (c.hidden_size / 2)
            )
            temporal = torch.cat((angle.sin(), angle.cos()), -1).to(spatial.dtype)
            positions.append((spatial[None] + temporal[:, None]).reshape(-1, c.hidden_size))
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        base = 10000 ** (-torch.arange(0, d, 4).float() / d)
        phase = (
            torch.stack((xx.flatten()[:, None] * base, yy.flatten()[:, None] * base), -1)
            .flatten(-2)
            .repeat(t, 1)
        )
        phases.append(torch.polar(torch.ones_like(phase), phase))
        lengths.append(t * h * w)
    x = x + torch.cat(positions)
    cis = torch.cat(phases)[:, None]
    for i in range(c.num_hidden_layers):
        stem = f"encoder.blocks.{i}."
        y = F.rms_norm(x, (c.hidden_size,), weights[stem + "norm0.weight"])
        q, k, v = (
            F.linear(y, weights[stem + "wqkv.weight"])
            .reshape(len(x), 3, c.num_attention_heads, d)
            .unbind(1)
        )

        def rotate(z):
            return (
                torch.view_as_real(
                    torch.view_as_complex(z.float().reshape(*z.shape[:-1], -1, 2)) * cis
                )
                .flatten(-2)
                .to(z.dtype)
            )

        q, k = rotate(q), rotate(k)
        parts = []
        offset = 0
        for count in lengths:
            chunks = [z[offset : offset + count].transpose(0, 1)[None] for z in (q, k, v)]
            parts.append(F.scaled_dot_product_attention(*chunks)[0].transpose(0, 1).flatten(1))
            offset += count
        x = x + F.linear(torch.cat(parts), weights[stem + "wo.weight"])
        y = F.rms_norm(x, (c.hidden_size,), weights[stem + "norm1.weight"])
        x = x + F.linear(
            F.gelu(F.linear(y, weights[stem + "mlp.fc0.weight"]), approximate="tanh"),
            weights[stem + "mlp.fc1.weight"],
        )
    x = F.rms_norm(x, (c.hidden_size,), weights["encoder.final_layernorm.weight"])
    merged = []
    offset = 0
    kh, kw = c.merge_kernel_size
    for t, h, w in grids.tolist():
        clip = (
            x[offset : offset + t * h * w]
            .reshape(t, h // kh, kh, w // kw, kw, c.hidden_size)
            .permute(0, 1, 3, 2, 4, 5)
        )
        merged.append(clip.mean(0).reshape(h // kh * w // kw, kh * kw, c.hidden_size))
        offset += t * h * w
    return x, torch.cat(merged)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_models_k3_vision_qkv_width_complex_rope_and_dtype_rms(dtype):
    torch.set_num_threads(1)
    torch.manual_seed(320)
    c = KimiK3VisionConfig()
    model = build_model(c).to(dtype)
    grid = torch.tensor([[1, 4, 4], [2, 2, 4]])
    pixels = torch.randn(32, 3, 2, 2, dtype=dtype, requires_grad=True)
    oracle_pixels = pixels.detach().clone().requires_grad_()
    parameters = {n: p.detach().clone().requires_grad_() for n, p in model.named_parameters()}
    actual = model(pixels, grid)
    expected, merged = _vision(oracle_pixels, grid, parameters, c)
    tolerance = 0 if dtype == torch.bfloat16 else 3e-6
    torch.testing.assert_close(actual.last_hidden_state, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual.pooler_output, merged, atol=tolerance, rtol=tolerance)
    target = torch.randn_like(merged)
    (actual.pooler_output.float() * target).sum().backward()
    (merged.float() * target).sum().backward()
    torch.testing.assert_close(
        pixels.grad,
        oracle_pixels.grad,
        atol=0 if dtype == torch.bfloat16 else 2e-5,
        rtol=0 if dtype == torch.bfloat16 else 2e-4,
    )
    for name, p in model.named_parameters():
        torch.testing.assert_close(
            p.grad,
            parameters[name].grad,
            atol=0 if dtype == torch.bfloat16 else 5e-5,
            rtol=0 if dtype == torch.bfloat16 else 5e-4,
            msg=name,
        )


def test_models_k3_full_multimodal_formula_all_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(321)
    c = KimiK3Config()
    model = build_model(c)
    tokens = torch.tensor([[1, 31, 31, 31, 31, 4, 5], [1, 31, 31, 2, 3, 4, 5]])
    grid = torch.tensor([[1, 4, 4], [2, 2, 4]])
    pixels = torch.randn(32, 3, 2, 2, requires_grad=True)
    other_pixels = pixels.detach().clone().requires_grad_()
    parameters = {n: p.detach().clone().requires_grad_() for n, p in model.named_parameters()}
    parameters.update({n: b.clone() for n, b in model.named_buffers()})
    visual = {
        n[len("vision_tower.") :]: p for n, p in parameters.items() if n.startswith("vision_tower.")
    }
    _, merged = _vision(other_pixels, grid, visual, c.vision_config)
    projected = F.linear(
        F.gelu(F.linear(merged.flatten(1), parameters["mm_projector.proj.0.weight"])),
        parameters["mm_projector.proj.2.weight"],
    )
    projected = F.rms_norm(
        projected,
        (c.text_config.hidden_size,),
        parameters["mm_projector.post_norm.weight"],
        c.projector_ln_eps,
    )
    text = {
        n[len("language_model.") :]: p
        for n, p in parameters.items()
        if n.startswith("language_model.")
    }
    embeds = F.embedding(tokens, text["model.embed_tokens.weight"])
    embeds = embeds.masked_scatter(tokens.eq(31)[..., None].expand_as(embeds), projected)
    expected = _full(tokens, text, c.text_config, inputs_embeds=embeds)
    actual = model(tokens, pixel_values=pixels, grid_thw=grid).logits
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=4e-5)
    target = torch.randn_like(actual)
    (actual * target).sum().backward()
    (expected * target).sum().backward()
    torch.testing.assert_close(pixels.grad, other_pixels.grad, atol=3e-5, rtol=5e-4)
    for name, p in model.named_parameters():
        if parameters[name].grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(
                p.grad, parameters[name].grad, atol=1e-4, rtol=1e-3, msg=name
            )
