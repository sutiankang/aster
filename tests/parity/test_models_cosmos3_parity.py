from dataclasses import replace
import math
import pytest
import torch
import torch.nn.functional as F

from aster.models.cosmos3 import Cosmos3Config, Cosmos3MoT, Cosmos3Vision, Cosmos3Sequence
from aster.nn.parameter_codec import public_parameter_names


def source_forward(c, weights, data, inverse_frequency):
    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name):
        w = weights.get(name + ".weight")
        if w is None:
            return x
        value = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + c.rms_norm_eps)
        if c.hidden_act == "relu2":
            return (value * w.float()).to(x.dtype)
        if w.dtype in (torch.float16, torch.bfloat16):
            value = value.to(w.dtype)
        return value * w

    def mlp(x, name):
        up = linear(x, name + ".up_proj")
        return linear(
            up.relu().square()
            if c.hidden_act == "relu2"
            else F.silu(linear(x, name + ".gate_proj")) * up,
            name + ".down_proj",
        )

    def time_embed(t):
        freq = torch.exp(-math.log(10000) * torch.arange(128).float() / 128)
        values = t.float()[:, None] * c.timestep_scale * freq
        x = torch.cat((values.cos(), values.sin()), -1).to(
            weights["time_embedder.linear_1.weight"].dtype
        )
        return linear(F.silu(linear(x, "time_embedder.linear_1")), "time_embedder.linear_2")

    def domain(x, ids, name):
        rows = F.embedding(ids, weights[name + ".fc.weight"]).reshape(len(x), x.shape[-1], -1)
        return torch.bmm(x[:, None], rows).squeeze(1) + F.embedding(
            ids, weights[name + ".bias.weight"]
        )

    def rotary(positions, dtype):
        frequencies = (
            inverse_frequency[None, None, :, None].float().expand(3, 1, -1, 1)
            @ positions[:, None, None].float()
        ).transpose(2, 3)
        mixed = frequencies[0].clone()
        for axis in (1, 2):
            mixed[..., axis : c.rope_axes_dim[axis] * 3 : 3] = frequencies[
                axis, ..., axis : c.rope_axes_dim[axis] * 3 : 3
            ]
        phase = torch.cat((mixed, mixed), -1).squeeze(0)
        return phase.cos().to(dtype)[:, None], phase.sin().to(dtype)[:, None]

    def apply(x, pair):
        half = x.shape[-1] // 2
        return x * pair[0] + torch.cat((-x[..., half:], x[..., :half]), -1) * pair[1]

    def attention(q, k, v, visible):

        rep = c.num_attention_heads // c.num_key_value_heads
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            attn_mask=visible,
        )
        return out.squeeze(0).transpose(0, 1).flatten(-2)

    result = dict(text=[], vision=[], sound=[], action=[])
    for row in range(len(data["input_ids"])):
        und = F.embedding(data["input_ids"][row], weights["embed_tokens.weight"])
        parts, positions, masks, shapes = [], [], [], {}
        for name in ("vision", "sound", "action"):
            field = data[name]
            sample = field.sample[row]
            t = sample.shape[1] if name == "vision" else sample.shape[0]
            if name == "vision":
                channels, _, height, width = sample.shape
                p = c.latent_patch_size
                hp, wp = math.ceil(height / p), math.ceil(width / p)
                padded = sample.new_zeros(channels, t, hp * p, wp * p)
                padded[:, :, :height, :width] = sample
                patch = torch.einsum(
                    "cthpwq->thwpqc", padded.reshape(channels, t, hp, p, wp, p)
                ).reshape(-1, c.patch_latent_dim)
                x = linear(patch, "proj_in")
                area = hp * wp
                shapes[name] = t, height, width, hp, wp
            elif name == "sound":
                x = linear(sample, "audio_proj_in") + weights["audio_modality_embed"]
                area = 1
            else:
                ids = field.domain_ids[row].expand(t)
                x = domain(sample, ids, "action_proj_in") + weights["action_modality_embed"]
                area = 1
            selected = field.noisy_frames[row].nonzero().flatten()
            noisy_indexes = (selected[:, None] * area + torch.arange(area)[None]).flatten()
            updates = (
                time_embed(field.timesteps[row, selected]).to(und.dtype).repeat_interleave(area, 0)
            )
            x = x.index_add(0, noisy_indexes, updates)
            parts.append(x)
            positions.append(field.positions[:, row])
            valid = (
                torch.ones(t, dtype=torch.bool)
                if field.valid_frames is None
                else field.valid_frames[row]
            )
            masks.append(valid.repeat_interleave(area))
        gen = torch.cat(parts, 0)
        gen_positions = torch.cat(positions, -1)
        und_positions = data.get(
            "understanding_positions",
            torch.arange(len(und))[None, None].expand(3, len(data["input_ids"]), -1),
        )[:, row]
        rotation_u, rotation_g = rotary(und_positions, und.dtype), rotary(gen_positions, und.dtype)
        valid_u = data.get("attention_mask", torch.ones_like(data["input_ids"], dtype=torch.bool))[
            row
        ]
        visible_u = torch.ones(len(und), len(und), dtype=torch.bool).tril() & valid_u[None]
        visible_g = torch.cat((valid_u, *masks))[None, None, None]
        for index in range(c.num_hidden_layers):
            base = f"layers.{index}"
            att = base + ".self_attn"
            u, g = (
                norm(und, base + ".input_layernorm"),
                norm(gen, base + ".input_layernorm_moe_gen"),
            )

            def project(x, name, heads):
                return linear(x, att + "." + name).reshape(len(x), heads, c.head_dim)

            q = norm(project(u, "to_q", c.num_attention_heads), att + ".norm_q")
            k = norm(project(u, "to_k", c.num_key_value_heads), att + ".norm_k")
            v = project(u, "to_v", c.num_key_value_heads)
            kg = norm(k, att + ".k_norm_und_for_gen")
            q, k, kg = apply(q, rotation_u), apply(k, rotation_u), apply(kg, rotation_u)
            qg = apply(
                norm(project(g, "add_q_proj", c.num_attention_heads), att + ".norm_added_q"),
                rotation_g,
            )
            keyg = apply(
                norm(project(g, "add_k_proj", c.num_key_value_heads), att + ".norm_added_k"),
                rotation_g,
            )
            valueg = project(g, "add_v_proj", c.num_key_value_heads)
            und = und + linear(attention(q, k, v, visible_u), att + ".to_out")
            gen = gen + linear(
                attention(qg, torch.cat((kg, keyg)), torch.cat((v, valueg)), visible_g),
                att + ".to_add_out",
            )
            und = und + mlp(norm(und, base + ".post_attention_layernorm"), base + ".mlp")
            gen = gen + mlp(
                norm(gen, base + ".post_attention_layernorm_moe_gen"), base + ".mlp_moe_gen"
            )
        result["text"].append(linear(norm(und, "norm"), "lm_head"))
        gen = norm(gen, "norm_moe_gen")
        start = 0
        for name, part in zip(("vision", "sound", "action"), parts):
            field = data[name]
            t = field.noisy_frames.shape[1]
            hidden = gen[start : start + len(part)]
            start += len(part)
            selected = field.noisy_frames[row].nonzero().flatten()
            if name == "vision":
                t, height, width, hp, wp = shapes[name]
                p = c.latent_patch_size
                selected_patches = (
                    selected[:, None] * (hp * wp) + torch.arange(hp * wp)[None]
                ).flatten()
                decoded = linear(hidden[selected_patches], "proj_out").reshape(
                    len(selected), hp, wp, p, p, c.latent_channel
                )
                decoded = torch.einsum("thwpqc->cthpwq", decoded).reshape(
                    c.latent_channel, len(selected), hp * p, wp * p
                )
                values = decoded.new_zeros(c.latent_channel, t, height, width)
                values[:, selected] = decoded[:, :, :height, :width]
            else:
                decoded = (
                    linear(hidden[selected], "audio_proj_out")
                    if name == "sound"
                    else domain(
                        hidden[selected],
                        field.domain_ids[row].expand(len(selected)),
                        "action_proj_out",
                    )
                )
                values = decoded.new_zeros(t, decoded.shape[-1])
                values[selected] = decoded
            result[name].append(values)
    return {name: torch.stack(values) for name, values in result.items()}


@pytest.mark.parametrize(
    "activation,dtype",
    [
        ("silu", torch.float32),
        ("relu2", torch.float32),
        ("silu", torch.bfloat16),
        ("relu2", torch.bfloat16),
    ],
)
def test_models_cosmos3_locked_packed_formula_forward_and_all_gradients(activation, dtype):
    torch.set_num_threads(1)
    torch.manual_seed(545)
    c = Cosmos3Config(
        hidden_act=activation,
        qk_norm_for_text=activation == "silu",
        use_und_k_norm_for_gen=activation == "relu2",
        attention_bias=True,
    )
    model = Cosmos3MoT(c).to(dtype)
    ids = torch.tensor([[1, 3, 5, 2], [1, 7, 2, 0]])

    def positions(n):
        return torch.stack(
            (torch.arange(n) + 15004.0, torch.arange(n) * 0.25, torch.arange(n) * 0.5)
        )[:, None].expand(-1, 2, -1)

    data = dict(
        input_ids=ids,
        attention_mask=ids.ne(0),
        vision=Cosmos3Vision(
            torch.randn(2, 2, 2, 3, 4, dtype=dtype, requires_grad=True),
            positions(8),
            torch.tensor([[0.0, 630.0], [610.0, 0.0]]),
            torch.tensor([[False, True], [True, False]]),
        ),
        sound=Cosmos3Sequence(
            torch.randn(2, 3, 4, dtype=dtype, requires_grad=True),
            positions(3),
            torch.full((2, 3), 540.0),
            torch.tensor([[True, True, False], [True, False, False]]),
            torch.tensor([[True, True, True], [True, False, False]]),
        ),
        action=Cosmos3Sequence(
            torch.randn(2, 2, 3, dtype=dtype, requires_grad=True),
            positions(2),
            torch.full((2, 2), 710.0),
            torch.ones(2, 2, dtype=torch.bool),
            domain_ids=torch.tensor([1, 3]),
        ),
    )
    reference = {
        **data,
        **{
            name: replace(data[name], sample=data[name].sample.detach().clone().requires_grad_())
            for name in ("vision", "sound", "action")
        },
    }
    weights = {
        name: value.detach().clone().requires_grad_() for name, value in model.state_dict().items()
    }
    output = model(**data)
    actual = dict(
        text=output.text.logits,
        **{name: getattr(output, name).prediction for name in ("vision", "sound", "action")},
    )
    expected = source_forward(c, weights, reference, model.rotary_emb.inv_freq)
    tolerance = (
        dict(atol=6e-6, rtol=7e-5) if dtype == torch.float32 else dict(atol=0.125, rtol=0.04)
    )
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], **tolerance, msg=name)
    if dtype != torch.float32:
        return
    factors = {name: torch.randn_like(value) / value.numel() for name, value in actual.items()}
    sum((value * factors[name]).sum() for name, value in actual.items()).backward()
    sum((value * factors[name]).sum() for name, value in expected.items()).backward()
    public = public_parameter_names(model)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, weights[public[name]].grad, atol=8e-6, rtol=2e-3, msg=name
        )
    for name in ("vision", "sound", "action"):
        torch.testing.assert_close(
            data[name].sample.grad, reference[name].sample.grad, atol=4e-6, rtol=5e-4, msg=name
        )


@pytest.mark.oracle
def test_models_cosmos3_understanding_real_transformers_subgraph_weights_gradients_cache():
    tf = pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.set_num_threads(1)
    torch.manual_seed(546)
    c = Cosmos3Config()
    native = Cosmos3MoT(c)
    config = tf.Qwen3VLTextConfig(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size,
        num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads,
        head_dim=c.head_dim,
        rms_norm_eps=c.rms_norm_eps,
        rope_parameters=dict(
            rope_type="default", rope_theta=c.rope_theta, mrope_section=list(c.rope_axes_dim)
        ),
    )
    config._attn_implementation = "eager"
    official = Qwen3VLTextModel(config)

    mapping = {}
    for name in official.state_dict():
        target = name
        for original, replacement in (
            ("q_proj", "to_q"),
            ("k_proj", "to_k"),
            ("v_proj", "to_v"),
            ("o_proj", "to_out"),
            ("q_norm", "norm_q"),
            ("k_norm", "norm_k"),
        ):
            target = target.replace(
                ".self_attn." + original + ".", ".self_attn." + replacement + "."
            )
        mapping[name] = target
    official.load_state_dict(
        {name: native.state_dict()[target] for name, target in mapping.items()}, strict=True
    )
    ids = torch.tensor([[1, 3, 5, 2], [1, 7, 9, 2]])
    positions = torch.stack((torch.arange(4) + 7, torch.arange(4) + 3, torch.arange(4) + 1))[
        :, None
    ].expand(-1, 2, -1)
    actual = native.forward_text(ids, understanding_positions=positions, output_hidden_states=True)
    expected = official(ids, position_ids=positions, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual.hidden_states[-1], expected, atol=4e-6, rtol=4e-5)
    factor = torch.randn_like(expected) / expected.numel()
    (actual.hidden_states[-1] * factor).sum().backward()
    (expected * factor).sum().backward()
    for name, parameter in official.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(native.named_parameters())[mapping[name]].grad,
            atol=5e-6,
            rtol=7e-4,
            msg=name,
        )
    prefix = native.forward_text(
        ids[:, :2], understanding_positions=positions[:, :, :2], use_cache=True
    )
    cache = official(ids[:, :2], position_ids=positions[:, :, :2], use_cache=True).past_key_values
    left = native.forward_text(
        ids[:, 2:],
        understanding_positions=positions[:, :, 2:],
        state=prefix.state,
        output_hidden_states=True,
    ).hidden_states[-1]
    right = official(
        ids[:, 2:], position_ids=positions[:, :, 2:], past_key_values=cache, use_cache=True
    ).last_hidden_state
    torch.testing.assert_close(left, right, atol=4e-6, rtol=4e-5)
