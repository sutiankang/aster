from dataclasses import asdict, replace
import math
import pytest
import torch
import torch.nn.functional as F
from aster.models import GrootActionConfig, GrootConfig, GrootCondition, build_model
from aster.models.qwen_vl import pack_qwen_pixels


def reference_head(p, c, sample, time, context):
    def linear(x, prefix):
        return F.linear(x, p[prefix + ".weight"], p.get(prefix + ".bias"))

    def norm(x, prefix, eps):
        return F.layer_norm(
            x, (x.shape[-1],), p.get(prefix + ".weight"), p.get(prefix + ".bias"), eps
        )

    def category(x, prefix):
        return (
            torch.bmm(x, p[prefix + ".W"][context.embodiment_id])
            + p[prefix + ".b"][context.embodiment_id][:, None]
        )

    def category_mlp(x, prefix):
        return category(F.relu(category(x, prefix + ".layer1")), prefix + ".layer2")

    def sinusoidal(t, width, flip=False, shift=0):
        exponent = (
            -math.log(10000)
            * torch.arange(width // 2, dtype=torch.float32, device=t.device)
            / (width // 2 - shift)
        )
        angle = t[:, None].float() * exponent.exp()[None]
        return torch.cat((angle.cos(), angle.sin()) if flip else (angle.sin(), angle.cos()), -1)

    def split(x):
        return x.view(
            x.shape[0], x.shape[1], c.num_attention_heads, c.attention_head_dim
        ).transpose(1, 2)

    discrete = (time * c.num_timestep_buckets).long()
    features = norm(context.features, "vlln", 1e-5) if c.use_vlln else context.features
    states = category_mlp(context.proprio.reshape(len(sample), 1, -1), "state_encoder")
    action = category(sample, "action_encoder.W1")
    t = (
        sinusoidal(discrete, c.input_embedding_dim)
        .to(action)[:, None]
        .expand(-1, sample.shape[1], -1)
    )
    action = category(
        F.silu(category(torch.cat((action, t), -1), "action_encoder.W2")), "action_encoder.W3"
    )
    if c.add_pos_embed:
        action = (
            action
            + F.embedding(torch.arange(c.action_horizon), p["position_embedding.weight"])[None]
        )
    x = torch.cat((states, action), 1)
    timestep = sinusoidal(discrete, 256, flip=True, shift=1).to(x)
    timestep = linear(
        F.silu(linear(timestep, "model.timestep_encoder.timestep_embedder.linear_1")),
        "model.timestep_encoder.timestep_embedder.linear_2",
    )
    for index in range(c.num_layers):
        prefix = f"model.transformer_blocks.{index}"
        if c.norm_type == "ada_norm":
            scale, shift = linear(F.silu(timestep), prefix + ".norm1.linear").chunk(2, -1)
            value = norm(x, prefix + ".norm1.norm", 1e-5) * (1 + scale[:, None]) + shift[:, None]
        else:
            value = norm(x, prefix + ".norm1", c.norm_eps)
        if c.positional_embeddings:
            angles = torch.arange(x.shape[1])[:, None] * torch.exp(
                torch.arange(0, x.shape[-1], 2) * (-math.log(10000.0) / x.shape[-1])
            )
            positions = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)
            value = value + positions.to(value)[None]
        self_attention = c.interleave_self_attention and index % 2
        encoder = value if self_attention else features
        mask = (
            torch.ones(encoder.shape[:2], dtype=torch.bool, device=x.device)
            if self_attention
            else context.attention_mask
        )
        if not self_attention and c.use_alternate_vl_dit:
            selected = (
                ~context.image_mask
                if index % (2 * c.attend_text_every_n_blocks) == 0
                else context.image_mask
            )
            mask = mask & selected
        q, k, v = (
            split(linear(value, prefix + ".attn1.to_q")),
            split(linear(encoder, prefix + ".attn1.to_k")),
            split(linear(encoder, prefix + ".attn1.to_v")),
        )
        attended = F.scaled_dot_product_attention(q, k, v, mask[:, None, None], dropout_p=0.0)
        x = x + linear(attended.transpose(1, 2).reshape_as(x), prefix + ".attn1.to_out.0")
        projected = linear(norm(x, prefix + ".norm3", c.norm_eps), prefix + ".ff.net.0.proj")
        if c.activation_fn == "geglu":
            value, gate = projected.chunk(2, -1)
            projected = value * F.gelu(gate)
        else:
            projected = F.gelu(
                projected, approximate="tanh" if c.activation_fn == "gelu-approximate" else "none"
            )
        x = x + linear(projected, prefix + ".ff.net.2")
    shift, scale = linear(F.silu(timestep), "model.proj_out_1").chunk(2, -1)
    x = linear(
        norm(x, "model.norm_out", 1e-6) * (1 + scale[:, None]) + shift[:, None], "model.proj_out_2"
    )
    return category_mlp(x, "action_decoder")[:, -c.action_horizon :]


@pytest.mark.parametrize("variant", ["alternate", "geglu", "plain", "layer_norm"])
def test_models_groot_source_formula_all_gradients(variant):
    torch.set_num_threads(1)
    torch.manual_seed(111)
    c = GrootActionConfig()
    if variant == "geglu":
        c = replace(c, activation_fn="geglu", positional_embeddings="sinusoidal")
    if variant == "plain":
        c = replace(
            c, use_alternate_vl_dit=False, interleave_self_attention=False, attention_bias=False
        )
    if variant == "layer_norm":
        c = replace(c, norm_type="layer_norm", norm_elementwise_affine=True, activation_fn="gelu")
    model = build_model(c).eval()
    params = {
        name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()
    }
    features = torch.randn(2, 5, c.backbone_embedding_dim, requires_grad=True)
    proprio = torch.randn(2, 1, c.max_state_dim, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    if variant != "plain":
        mask[:, -1] = False
    context = GrootCondition(
        features,
        mask,
        torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.bool).expand(2, -1),
        proprio,
        torch.tensor([0, 2]),
    )
    ref_context = replace(
        context,
        features=features.detach().clone().requires_grad_(),
        proprio=proprio.detach().clone().requires_grad_(),
    )
    sample, time = torch.randn(2, 4, 3, requires_grad=True), torch.tensor([0.1, 0.83])
    ref_sample = sample.detach().clone().requires_grad_()
    actual = model(sample, time, context).prediction
    reference = reference_head(params, c, ref_sample, time, ref_context)
    torch.testing.assert_close(actual, reference, atol=2e-7, rtol=3e-5)
    coefficient = torch.randn_like(actual)
    (actual * coefficient).sum().backward()
    (reference * coefficient).sum().backward()
    for name, value in model.named_parameters():
        torch.testing.assert_close(value.grad, params[name].grad, atol=3e-6, rtol=3e-4, msg=name)
    for a, b in (
        (features, ref_context.features),
        (proprio, ref_context.proprio),
        (sample, ref_sample),
    ):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=3e-4)


@pytest.mark.oracle
def test_models_groot_qwen_official_compositional_pixel_and_all_parameter_gradients():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(112)
    c = GrootConfig()
    model = build_model(c).eval()
    values = asdict(c.backbone_config.text_config)
    for name in ("rope", "mrope_section", "layer_types", "sliding_window"):
        values.pop(name)
    values["num_hidden_layers"] = c.select_layer
    values["rope_parameters"] = dict(
        rope_type="default",
        rope_theta=c.backbone_config.text_config.rope.theta,
        mrope_section=list(c.backbone_config.text_config.mrope_section),
    )
    config = tf.Qwen3VLConfig(
        text_config=tf.Qwen3VLTextConfig(**values),
        vision_config=tf.Qwen3VLVisionConfig(**asdict(c.backbone_config.vision_config)),
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=26,
        vision_end_token_id=27,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    oracle = tf.Qwen3VLForConditionalGeneration(config).eval()
    oracle.load_state_dict(model.backbone.model.state_dict(), strict=True)
    params = {
        n: p.detach().clone().requires_grad_() for n, p in model.action_head.named_parameters()
    }
    pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 8), c.backbone_config.vision_config)
    pixels.requires_grad_()
    ref_pixels = pixels.detach().clone().requires_grad_()
    ids = torch.tensor([[1, 26, 28, 28, 28, 28, 27, 3]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    image = ids == 28
    state = torch.randn(1, 1, 6)
    embodiment = torch.tensor([1])
    sample = torch.randn(1, 4, 3)
    time = torch.tensor([0.42])
    obs = dict(
        input_ids=ids,
        attention_mask=mask,
        pixel_values=pixels,
        image_grid_thw=grid,
        proprio=state,
        embodiment_id=embodiment,
    )
    actual = model(sample, time, obs).prediction
    features = oracle(
        ids,
        attention_mask=mask,
        pixel_values=ref_pixels,
        image_grid_thw=grid,
        mm_token_type_ids=image.long(),
        output_hidden_states=True,
        use_cache=False,
    ).hidden_states[-1]
    reference = reference_head(
        params,
        c.action_config,
        sample,
        time,
        GrootCondition(features, mask, image, state, embodiment),
    )
    torch.testing.assert_close(actual, reference, atol=3e-7, rtol=4e-5)
    coeff = torch.randn_like(actual)
    (actual * coeff).sum().backward()
    (reference * coeff).sum().backward()
    for name, value in model.backbone.model.named_parameters():
        torch.testing.assert_close(
            value.grad, dict(oracle.named_parameters())[name].grad, atol=2e-6, rtol=5e-4, msg=name
        )
    for name, value in model.action_head.named_parameters():
        torch.testing.assert_close(value.grad, params[name].grad, atol=2e-6, rtol=5e-4, msg=name)
    torch.testing.assert_close(pixels.grad, ref_pixels.grad, atol=2e-6, rtol=5e-4)
