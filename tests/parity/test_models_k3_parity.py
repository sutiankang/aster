"""

Kimi K3 reference comparisons. KDA reference: MIT,
Copyright2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li; maintained by Moonshot."""

import pytest
import torch
import torch.nn.functional as F
from aster.models import KimiK3TextConfig, build_model
from aster.nn.kda import kda_scan, KDADecayGate


def _norm(x, weight, eps):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(
        x.dtype
    ) * weight


def _mix(prefix, snapshots, p, stem, c):
    if not snapshots:
        return prefix

    rows = torch.stack([*snapshots, prefix], 2)
    scores = F.linear(
        _norm(rows, p[stem + "_norm.weight"], c.rms_norm_eps), p[stem + "_proj.weight"]
    ).squeeze(-1)
    return (scores.float().softmax(-1)[..., None] * rows.float()).sum(2).to(prefix.dtype)


def _mlp(x, p, stem, c):
    g, u = F.linear(x, p[stem + "gate_up_proj.weight"]).float().chunk(2, -1)
    g = c.activation_situ_beta * torch.tanh(g / c.activation_situ_beta) * torch.sigmoid(g)
    u = c.activation_situ_linear_beta * torch.tanh(u / c.activation_situ_linear_beta)
    return F.linear((g * u).to(x.dtype), p[stem + "down_proj.weight"])


def _kda(q, k, v, gate, beta, initial=None):

    q, k = [
        x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)
    ]
    q = q * q.shape[-1] ** -0.5
    memory = (
        q.new_zeros(q.shape[0], q.shape[2], q.shape[-1], v.shape[-1])
        if initial is None
        else initial
    )
    values = []
    for t in range(q.shape[1]):
        memory = memory * gate[:, t, ..., None].exp()
        memory = memory + torch.einsum(
            "bhk,bhv->bhkv",
            beta[:, t, ..., None] * k[:, t],
            v[:, t] - (k[:, t, ..., None] * memory).sum(-2),
        )
        values.append(torch.einsum("bhk,bhkv->bhv", q[:, t], memory))
    return torch.stack(values, 1), memory


def _full(ids, p, c, inputs_embeds=None):
    x = F.embedding(ids, p["model.embed_tokens.weight"]) if inputs_embeds is None else inputs_embeds
    b, s, _ = x.shape
    bank = []
    for i in range(c.num_hidden_layers):
        stem = f"model.layers.{i}."
        query = _mix(x, bank, p, stem + "self_attention_res", c)
        boundary = i % c.attn_res_block_size == 0
        if boundary:
            bank.append(x)
        query = _norm(query, p[stem + "input_layernorm.weight"], c.rms_norm_eps)
        a = stem + "self_attn."
        if c.layer_types[i] == "kda":
            h, d = c.linear_num_heads, c.linear_head_dim
            qkv = F.linear(query, p[a + "qkv_proj.weight"]).transpose(1, 2)
            qkv = F.silu(
                F.conv1d(
                    F.pad(qkv, (c.linear_conv_kernel_dim - 1, 0)),
                    p[a + "qkv_conv1d.weight"],
                    groups=3 * h * d,
                )
            ).transpose(1, 2)
            q, k, v = [z.reshape(b, s, h, d) for z in qkv.chunk(3, -1)]
            f = F.linear(
                F.linear(query, p[a + "f_a_proj.weight"]), p[a + "f_b_proj.weight"]
            ).reshape(b, s, h, d)
            g = c.gate_lower_bound * torch.sigmoid(
                p[a + "decay_gate.A_log"].exp()[:, None]
                * (f + p[a + "decay_gate.dt_bias"].reshape(h, d))
            )
            beta = F.linear(query, p[a + "b_proj.weight"]).float().sigmoid()
            y, _ = _kda(q, k, v, g, beta)
            z = F.linear(query, p[a + "g_proj.weight"]).reshape(b, s, h, d)
            y = _norm(y, p[a + "o_norm.weight"], c.rms_norm_eps) * z.sigmoid()
            value = F.linear(y.flatten(-2), p[a + "o_proj.weight"])
        else:
            q = F.linear(
                _norm(
                    F.linear(query, p[a + "q_a_proj.weight"]),
                    p[a + "q_a_layernorm.weight"],
                    c.rms_norm_eps,
                ),
                p[a + "q_b_proj.weight"],
            )
            q = q.reshape(b, s, c.num_attention_heads, -1).transpose(1, 2)
            latent, shared = F.linear(query, p[a + "kv_a_proj_with_mqa.weight"]).split(
                (c.kv_lora_rank, c.qk_rope_head_dim), -1
            )
            latent = _norm(latent, p[a + "kv_a_layernorm.weight"], c.rms_norm_eps)
            k, v = (
                F.linear(latent, p[a + "kv_b_proj.weight"])
                .reshape(b, s, c.num_attention_heads, -1)
                .transpose(1, 2)
                .split((c.qk_nope_head_dim, c.v_head_dim), -1)
            )
            k = torch.cat((k, shared[:, None].expand(-1, c.num_attention_heads, -1, -1)), -1)
            scores = q @ k.transpose(-1, -2) / (c.qk_nope_head_dim + c.qk_rope_head_dim) ** 0.5
            scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), -torch.inf)
            y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, s, -1)
            value = F.linear(
                y * F.linear(query, p[a + "g_proj.weight"]).sigmoid(), p[a + "o_proj.weight"]
            )
        x = value if boundary else x + value
        query = _norm(
            _mix(x, bank, p, stem + "mlp_res", c),
            p[stem + "post_attention_layernorm.weight"],
            c.rms_norm_eps,
        )
        m = stem + "mlp."
        if i < c.first_k_dense_replace:
            value = _mlp(query, p, m, c)
        else:
            probabilities = F.linear(query, p[m + "gate.weight"]).float().sigmoid()

            selected = (
                (probabilities + p[m + "e_score_correction_bias"])
                .topk(c.num_experts_per_tok, sorted=False)
                .indices
            )
            weights = probabilities.gather(-1, selected)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * c.routed_scaling_factor
            latent = F.linear(query, p[m + "routed_expert_down_proj.weight"])
            experts = torch.stack(
                [_mlp(latent, p, m + f"experts.{j}.", c) for j in range(c.n_routed_experts)], -2
            )
            picked = experts.gather(
                -2, selected[..., None].expand(-1, -1, -1, c.routed_expert_hidden_size)
            )
            mix = (picked * weights[..., None]).sum(-2)
            value = F.linear(
                _norm(mix, p[m + "routed_expert_norm.weight"], c.rms_norm_eps),
                p[m + "routed_expert_up_proj.weight"],
            ) + _mlp(query, p, m + "shared_experts.", c)
        x = x + value
    x = _norm(_mix(x, bank, p, "model.output_attn_res", c), p["model.norm.weight"], c.rms_norm_eps)
    return F.linear(x, p["lm_head.weight"])


@pytest.mark.parametrize("block_size", [1, 2])
def test_models_k3_whole_functional_reference_all_gradients(block_size):
    torch.set_num_threads(1)
    torch.manual_seed(310)
    c = KimiK3TextConfig(attn_res_block_size=block_size)
    model = build_model(c)
    parameters = {n: p.detach().clone().requires_grad_() for n, p in model.named_parameters()}
    parameters.update({n: b.clone() for n, b in model.named_buffers()})
    tokens = torch.randint(0, 32, (2, 6))
    actual, expected = model(tokens).logits, _full(tokens, parameters, c)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
    target = torch.randn_like(actual)
    (actual * target).sum().backward()
    (expected * target).sum().backward()
    for name, p in model.named_parameters():
        if parameters[name].grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(
                p.grad, parameters[name].grad, atol=7e-5, rtol=7e-4, msg=name
            )


def test_models_k3_safe_gate_and_vector_recurrence_reference():
    torch.set_num_threads(1)
    torch.manual_seed(311)
    q, k, v = [torch.randn(2, 5, 3, 4, requires_grad=True) for _ in range(3)]
    alpha = torch.randn(2, 5, 3, 4, requires_grad=True)
    beta = torch.rand(2, 5, 3, requires_grad=True)
    gate = KDADecayGate(3, 4)
    log_decay = gate(alpha)
    assert bool((log_decay > -5).all()) and bool((log_decay < 0).all())
    actual, state = kda_scan(q, k, v, log_decay, beta)
    expected, final = _kda(q, k, v, log_decay, beta)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
    torch.testing.assert_close(state, final)
    inputs = (q, k, v, alpha, beta, *gate.parameters())
    target = torch.randn_like(actual)
    for a, b in zip(
        torch.autograd.grad((actual * target).sum(), inputs, retain_graph=True),
        torch.autograd.grad((expected * target).sum(), inputs),
    ):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
