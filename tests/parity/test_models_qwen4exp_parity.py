import math
import pytest
import torch
import torch.nn.functional as F
from aster.models import Qwen4ExpTextConfig, build_model
from aster.nn.delta import GatedDeltaNet, delta_public_parameter_name


def _norm(x, weight, eps, group=None, zero=True):
    original = x.shape
    y = x.float()
    if group:
        y = y.reshape(*original[:-1], -1, group)
    y = (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)).reshape(original)
    if zero:
        return (y * (weight.float() + 1)).to(x.dtype)
    return y.to(x.dtype) * weight


def _rope(x, positions, c):
    size = int(c.head_dim * c.partial_rotary_factor)
    inv = 1 / (c.rope.theta ** (torch.arange(0, size, 2, device=x.device).float() / size))
    phases = positions.float()[..., None] * inv
    mixed = phases[0].clone()
    for axis in (1, 2):
        mixed[..., axis : 3 * c.mrope_section[axis] : 3] = phases[
            axis, ..., axis : 3 * c.mrope_section[axis] : 3
        ]
    angles = torch.cat((mixed, mixed), -1)

    if x.ndim == 4:
        angles = angles.unsqueeze(2)
    first, second = x[..., :size].chunk(2, -1)
    return torch.cat(
        (
            x[..., :size] * angles.cos().to(x.dtype)
            + torch.cat((-second, first), -1) * angles.sin().to(x.dtype),
            x[..., size:],
        ),
        -1,
    )


def _hash_ids(tokens, state, p, c):
    rows = []
    for sequence in tokens.tolist():
        output = []
        for index in range(len(sequence)):
            previous_eos = max(
                (j for j in range(index) if sequence[j] == c.eos_token_id), default=-1
            )
            shifted = [
                sequence[index - offset] if index - offset > previous_eos else c.eos_token_id
                for offset in range(c.ngram_size)
            ]
            hashes = []
            for degree in range(2, c.ngram_size + 1):
                mixed = shifted[0] * int(state[p + "layer_multipliers"][0])
                for offset in range(1, degree):
                    mixed ^= shifted[offset] * int(state[p + "layer_multipliers"][offset])
                for head in range(
                    (degree - 2) * c.heads_per_ngram, (degree - 1) * c.heads_per_ngram
                ):
                    hashes.append(
                        mixed % int(state[p + "ngram_heads_vocab_sizes"][head])
                        + int(state[p + "ngram_heads_offsets"][head])
                    )
            output.append(hashes)
        rows.append(output)
    return torch.tensor(rows, device=tokens.device)


def _ple(hidden, tokens, state, p, c):
    lookup = _hash_ids(tokens, state, p + "ple_embedding.", c)
    x = F.embedding(lookup, state[p + "ple_embedding.ngram_embedding.weight"]).flatten(-2)
    key = F.linear(x, state[p + "key_proj.weight"])
    key = _norm(key, state[p + "norm_key.weight"], c.rms_norm_eps, c.hidden_size).unflatten(
        -1, (c.hc_count, c.hidden_size)
    )
    query = _norm(hidden, state[p + "norm_query.weight"], c.rms_norm_eps, c.hidden_size).unflatten(
        -1, (c.hc_count, c.hidden_size)
    )
    logits = (key * query).sum(-1, keepdim=True) / math.sqrt(c.hidden_size)
    logits = logits.abs().clamp_min(1e-6).sqrt() * logits.sign()
    value = (logits.sigmoid() * F.linear(x, state[p + "value_proj.weight"]).unsqueeze(-2)).flatten(
        -2
    )
    normal = _norm(value, state[p + "norm_conv.weight"], c.rms_norm_eps, c.hidden_size)
    conv = F.conv1d(
        normal.transpose(1, 2),
        state[p + "conv1d.weight"],
        padding=(c.ple_conv_kernel_size - 1) * c.ngram_size,
        dilation=c.ngram_size,
        groups=hidden.shape[-1],
    )[..., : hidden.shape[1]]
    return value + F.silu(conv).transpose(1, 2)


def _hyper(hidden, state, p, c, combine=True):
    normal = _norm(hidden, state[p + "hc_norm.weight"], c.rms_norm_eps, c.hidden_size)
    x = F.silu(F.linear(normal, state[p + "input_mix_weight_down.weight"]) / c.hc_count)
    gates = (
        F.linear(x, state[p + "input_mix_weight_up.weight"])
        .sigmoid()
        .unflatten(-1, (c.hc_count, c.hidden_size))
    )
    mixed = (normal.unflatten(-1, (c.hc_count, c.hidden_size)) * gates).mean(-2)
    if not combine:
        return mixed
    return mixed, 2 * (
        F.linear(normal, state[p + "block_inject_weight.weight"]) / c.hc_count
    ).sigmoid()


def _delta(hidden, state, p, c):
    b, s, _ = hidden.shape
    hk, hv, dk, dv = (
        c.linear_num_key_heads,
        c.linear_num_value_heads,
        c.linear_key_head_dim,
        c.linear_value_head_dim,
    )
    projected = F.linear(hidden, state[p + "in_proj_qkv.weight"]).transpose(1, 2)
    projected = F.silu(
        F.conv1d(
            projected,
            state[p + "conv1d.weight"],
            groups=projected.shape[1],
            padding=c.linear_conv_kernel_dim - 1,
        )[..., :s]
    ).transpose(1, 2)
    q, k, v = projected.split((hk * dk, hk * dk, hv * dv), -1)
    q, k = (x.reshape(b, s, hk, dk).repeat_interleave(hv // hk, 2) for x in (q, k))
    q, k = (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k))
    q, k, v = q.float() / math.sqrt(dk), k.float(), v.reshape(b, s, hv, dv).float()
    g = -state[p + "A_log"].float().exp() * F.softplus(
        F.linear(hidden, state[p + "in_proj_a.weight"]).float() + state[p + "dt_bias"]
    )
    beta = F.linear(hidden, state[p + "in_proj_b.weight"]).sigmoid()
    memory = hidden.new_zeros((b, hv, dk, dv), dtype=torch.float32)
    result = []
    for index in range(s):
        memory = memory * g[:, index].exp()[..., None, None]
        delta = (v[:, index] - torch.einsum("bhkv,bhk->bhv", memory, k[:, index])) * beta[
            :, index, :, None
        ]
        memory = memory + torch.einsum("bhk,bhv->bhkv", k[:, index], delta)
        result.append(torch.einsum("bhkv,bhk->bhv", memory, q[:, index]))
    result = torch.stack(result, 1).to(hidden.dtype)
    result = _norm(result, state[p + "norm.weight"], c.rms_norm_eps, zero=False)
    gate = F.linear(hidden, state[p + "in_proj_z.weight"]).reshape(b, s, hv, dv).float()
    result = result * (gate.sigmoid() if c.output_gate_type == "sigmoid" else F.silu(gate))
    return F.linear(result.to(hidden.dtype).flatten(-2), state[p + "out_proj.weight"])


def _qsa(hidden, positions, state, p, c):
    b, length, _ = hidden.shape
    ip = p + "indexer."
    q, raw = F.linear(hidden, state[ip + "index_qk_proj.weight"]).split(
        (c.indexer_n_heads * c.indexer_head_dim, c.indexer_head_dim), -1
    )
    q = _norm(
        q.reshape(b, length, c.indexer_n_heads, c.indexer_head_dim),
        state[ip + "q_layernorm.weight"],
        c.rms_norm_eps,
    )
    q = _rope(q, positions, c)
    mask = torch.zeros(b, length, length, dtype=torch.bool)
    all_scores = []
    for row in range(b):
        for query in range(length):
            count = (query + 1) // c.indexer_compress_ratio
            groups = torch.arange(count * c.indexer_compress_ratio).reshape(
                count, c.indexer_compress_ratio
            )
            if count:
                pooled = (
                    raw[row, : count * c.indexer_compress_ratio]
                    .reshape(count, c.indexer_compress_ratio, c.indexer_head_dim)
                    .float()
                    .mean(1)
                    .to(hidden.dtype)
                )
                pooled = _norm(pooled, state[ip + "k_layernorm.weight"], c.rms_norm_eps)
                keys = _rope(pooled, positions[:, row, groups[:, 0]], c)
                scores = (q[row, query].float() @ keys.float().T).relu().sum(0) / math.sqrt(
                    c.indexer_head_dim
                )
                indices = scores.topk(
                    min(count, c.indexer_budget // c.indexer_compress_ratio)
                ).indices
                mask[row, query, groups[indices].flatten()] = True
                all_scores.append(scores)
            mask[row, query, count * c.indexer_compress_ratio : query + 1] = True
    q, gate = (
        F.linear(hidden, state[p + "q_proj.weight"])
        .reshape(b, length, c.num_attention_heads, 2 * c.head_dim)
        .chunk(2, -1)
    )
    q = _rope(_norm(q, state[p + "q_norm.weight"], c.rms_norm_eps), positions, c).transpose(1, 2)
    k = F.linear(hidden, state[p + "k_proj.weight"]).reshape(
        b, length, c.num_key_value_heads, c.head_dim
    )
    k = _rope(_norm(k, state[p + "k_norm.weight"], c.rms_norm_eps), positions, c).transpose(1, 2)
    v = (
        F.linear(hidden, state[p + "v_proj.weight"])
        .reshape(b, length, c.num_key_value_heads, c.head_dim)
        .transpose(1, 2)
    )
    repeat = c.num_attention_heads // c.num_key_value_heads
    k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(c.head_dim)
    probabilities = (
        scores.masked_fill(~mask[:, None], -torch.inf).float().softmax(-1).to(hidden.dtype)
    )
    result = (probabilities @ v).transpose(1, 2).reshape(b, length, -1) * gate.reshape(
        b, length, -1
    ).sigmoid()
    return F.linear(result, state[p + "o_proj.weight"]), tuple(all_scores)


def _mlp(hidden, state, p):
    return F.linear(
        F.silu(F.linear(hidden, state[p + "gate_proj.weight"]))
        * F.linear(hidden, state[p + "up_proj.weight"]),
        state[p + "down_proj.weight"],
    )


def _moe(hidden, state, p, c):
    flat = hidden.flatten(0, 1)
    scores = F.linear(flat, state[p + "gate.weight"]).float().softmax(-1)
    weights, indices = scores.topk(c.num_experts_per_tok, -1)
    if c.norm_topk_prob:
        weights = weights / weights.sum(-1, keepdim=True)
    weights = weights.to(hidden.dtype)

    outputs = []
    for index, x in enumerate(flat):
        value = torch.zeros_like(x)
        for slot in range(c.num_experts_per_tok):
            expert = indices[index, slot]
            gate, up = F.linear(x, state[p + "experts.gate_up_proj"][expert]).chunk(2, -1)
            value = value + weights[index, slot] * F.linear(
                F.silu(gate) * up, state[p + "experts.down_proj"][expert]
            )
        outputs.append(value)
    result = (
        torch.stack(outputs)
        + _mlp(flat, state, p + "shared_expert.")
        * F.linear(flat, state[p + "shared_expert_gate.weight"]).sigmoid()
    )
    return result.reshape_as(hidden)


def source_forward(ids, positions, state, c, inputs_embeds=None):
    hidden = (
        F.embedding(ids, state["model.embed_tokens.weight"])
        if inputs_embeds is None
        else inputs_embeds
    ).repeat(1, 1, c.hc_count)
    indexes = []
    for i, kind in enumerate(c.layer_types):
        p = f"model.layers.{i}."
        if i + 1 in c.ple_layer_ids:
            hidden = hidden + _ple(hidden, ids, state, p + "ple.", c)
        mixed, injection = _hyper(hidden, state, p + "attn_hyper_connection.", c)
        if kind == "linear_attention":
            value = _delta(mixed, state, p + "linear_attn.", c)
        else:
            value, scores = _qsa(mixed, positions, state, p + "self_attn.", c)
            indexes.extend(scores)
        hidden = hidden + (value.unsqueeze(-2) * injection.unsqueeze(-1)).flatten(-2)
        mixed, injection = _hyper(hidden, state, p + "mlp_hyper_connection.", c)
        hidden = hidden + (
            _moe(mixed, state, p + "mlp.", c).unsqueeze(-2) * injection.unsqueeze(-1)
        ).flatten(-2)
    final = _hyper(hidden, state, "model.hyper_connection_mixer.", c, False)
    return F.linear(final, state["lm_head.weight"]), tuple(indexes)


def test_models_qwen4exp_fixed_source_complete_forward_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(411)
    c = Qwen4ExpTextConfig()
    native = build_model(c)
    torch.nn.init.normal_(native.model.layers[1].ple.conv1d.weight, std=0.03)
    weights = {
        name: x.detach().clone().requires_grad_(x.is_floating_point())
        for name, x in native.state_dict().items()
    }
    ids = torch.tensor([[1, 3, 2, 5, 7, 9, 11, 2], [1, 4, 6, 8, 10, 12, 13, 2]])
    positions = torch.arange(8)[None, None].expand(3, 2, -1).clone()
    positions[1, :, 3:] += 3
    positions[2, :, 2:] += 1
    output = native(ids, position_ids=positions)
    expected, reference_scores = source_forward(ids, positions, weights, c)
    torch.testing.assert_close(output.logits, expected, atol=4e-6, rtol=4e-5)
    native_scores = tuple(
        record[3] for layer in output.auxiliary["qsa_indexer"] for record in layer
    )
    assert len(native_scores) == len(reference_scores)
    for a, b in zip(native_scores, reference_scores):
        torch.testing.assert_close(a, b, atol=4e-6, rtol=5e-5)
    factor = torch.randn_like(expected)

    left = (output.logits * factor).sum() + sum(x.square().sum() * 0.03 for x in native_scores)
    right = (expected * factor).sum() + sum(x.square().sum() * 0.03 for x in reference_scores)
    left.backward()
    right.backward()
    for name, parameter in native.named_parameters():
        reference = weights[delta_public_parameter_name(name)]
        assert parameter.grad is not None and reference.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-4, rtol=2e-3, msg=name)


@pytest.mark.oracle
def test_models_qwen4exp_gdn_installed_official_subgraph():
    tf = pytest.importorskip("transformers")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    torch.set_num_threads(1)
    torch.manual_seed(412)
    c = Qwen4ExpTextConfig()
    names = (
        "hidden_size",
        "linear_num_value_heads",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "rms_norm_eps",
    )
    config = tf.Qwen3_5TextConfig(**{name: getattr(c, name) for name in names})
    oracle = Qwen3_5GatedDeltaNet(config, 0)
    oracle.norm.activation = "sigmoid"
    native = GatedDeltaNet(c, projection_layout="separate", output_gate="sigmoid")
    oracle.load_state_dict(native.state_dict(), strict=True)
    left = torch.randn(2, 9, c.hidden_size, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    actual = native(left)[0]
    expected = oracle(right)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=4e-5)
    factor = torch.randn_like(actual)
    (actual * factor).sum().backward()
    (expected * factor).sum().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=8e-4)
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad,
            dict(oracle.named_parameters())[delta_public_parameter_name(name)].grad,
            atol=3e-5,
            rtol=1e-3,
            msg=name,
        )
