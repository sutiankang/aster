import copy
import pytest
import torch
import torch.nn.functional as F
from aster.models import OCR2TextConfig, OCR2Config, build_model
from aster.nn.parameter_codec import public_parameter_names


def oracle_language(tf, c):
    from transformers.models.llama.modeling_llama import LlamaMLP

    config = tf.LlamaConfig(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size,
        num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads,
        max_position_embeddings=c.max_position_embeddings,
        rms_norm_eps=c.rms_norm_eps,
        rope_theta=c.rope.theta,
        tie_word_embeddings=c.tie_word_embeddings,
        attention_dropout=c.attention_dropout,
    )
    config._attn_implementation = "eager"

    class SourceMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(c.hidden_size, c.n_routed_experts, bias=False)
            routed = copy.deepcopy(config)
            routed.intermediate_size = c.moe_intermediate_size
            shared = copy.deepcopy(config)
            shared.intermediate_size = c.moe_intermediate_size * c.n_shared_experts
            self.experts = torch.nn.ModuleList(LlamaMLP(routed) for _ in range(c.n_routed_experts))
            self.shared_experts = LlamaMLP(shared)
            self.auxiliary = None

        def forward(self, hidden):
            b, length, width = hidden.shape
            logits = F.linear(hidden.float(), self.gate.weight.float()).reshape(
                -1, c.n_routed_experts
            )
            probabilities = logits.softmax(-1)
            weights, indices = probabilities.topk(c.num_experts_per_tok, dim=-1, sorted=False)
            if c.num_experts_per_tok > 1 and c.norm_topk_prob:
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            weights = weights * c.routed_scaling_factor
            expanded = hidden.reshape(-1, width).repeat_interleave(c.num_experts_per_tok, 0)
            result = torch.empty_like(expanded)
            chosen = indices.flatten()
            for i, expert in enumerate(self.experts):
                result[chosen == i] = expert(expanded[chosen == i])
            result = (
                (result.reshape(*weights.shape, width) * weights[..., None])
                .sum(1)
                .to(hidden.dtype)
                .reshape_as(hidden)
            )
            histogram = torch.zeros(b, c.n_routed_experts)
            histogram.scatter_add_(
                1, indices.reshape(b, -1), torch.ones(b, length * c.num_experts_per_tok)
            )
            histogram = histogram / (length * c.num_experts_per_tok / c.n_routed_experts)
            self.auxiliary = (
                (histogram * probabilities.reshape(b, length, -1).mean(1)).sum(-1).mean()
            )
            return result + self.shared_experts(hidden)

    model = tf.LlamaForCausalLM(config)
    for i, layer in enumerate(model.model.layers):
        if i >= c.first_k_dense_replace and i % c.moe_layer_freq == 0:
            layer.mlp = SourceMoE()
    return model


@pytest.mark.oracle
def test_models_ocr2_official_attention_independent_moe_all_gradients_cache():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(531)
    c = OCR2TextConfig()
    native = build_model(c)
    oracle = oracle_language(tf, c)
    oracle.load_state_dict(native.state_dict(), strict=True)
    tokens = torch.tensor([[1, 3, 5, 7, 9, 2], [1, 4, 6, 8, 10, 2]])
    actual, expected = native(tokens), oracle(tokens, use_cache=False)
    torch.testing.assert_close(actual.logits, expected.logits, atol=3e-6, rtol=4e-5)
    reference_aux = sum(
        layer.mlp.auxiliary for layer in oracle.model.layers if hasattr(layer.mlp, "auxiliary")
    )
    torch.testing.assert_close(actual.auxiliary["router_aux"].mean, reference_aux)
    factor = torch.randn_like(actual.logits)
    ((actual.logits * factor).sum() + 0.001 * actual.auxiliary["router_aux"].mean).backward()
    ((expected.logits * factor).sum() + 0.001 * reference_aux).backward()
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=4e-5, rtol=8e-4, msg=name
        )
    prefix = native(tokens[:, :3], use_cache=True)
    other = oracle(tokens[:, :3], use_cache=True)
    torch.testing.assert_close(
        native(tokens[:, 3:], state=prefix.state).logits,
        oracle(tokens[:, 3:], past_key_values=other.past_key_values, use_cache=True).logits,
        atol=3e-6,
        rtol=4e-5,
    )


@pytest.mark.oracle
def test_models_ocr2_full_visual_connection_independent_scatter_and_language():

    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(532)
    c = OCR2Config()
    native = build_model(c)
    reference_vision = copy.deepcopy(native.vision_encoder)
    reference_projector = copy.deepcopy(native.projector)
    reference_separator = native.separator.weight.detach().clone().requires_grad_()
    oracle = oracle_language(tf, c.text_config)
    oracle.load_state_dict(native.language_model.state_dict(), strict=True)
    count = c.vision_config.local_queries + c.vision_config.global_queries + 1
    tokens = torch.tensor([[1] + [28] * count + [3, 2]])
    global_pixels = torch.randn(1, 3, 32, 32, requires_grad=True)
    local_pixels = torch.randn(1, 3, 24, 24, requires_grad=True)
    global_ref = global_pixels.detach().clone().requires_grad_()
    local_ref = local_pixels.detach().clone().requires_grad_()
    actual = native(
        tokens,
        pixel_values=global_pixels,
        pixel_values_local=(local_pixels,),
        images_spatial_crop=torch.tensor([[1, 1]]),
    )
    local = reference_projector.layers(reference_vision(local_ref)).flatten(0, 1)
    global_ = reference_projector.layers(reference_vision(global_ref)).flatten(0, 1)
    values = torch.cat((local, global_, reference_separator[None]), 0)
    hidden = oracle.model.embed_tokens(tokens)
    embedded = torch.cat((hidden[:, :1], values[None], hidden[:, -2:]), 1)
    expected = oracle(inputs_embeds=embedded, use_cache=False).logits
    torch.testing.assert_close(actual.logits, expected, atol=4e-6, rtol=5e-5)
    factor = torch.randn_like(expected)
    (actual.logits * factor).sum().backward()
    (expected * factor).sum().backward()
    torch.testing.assert_close(global_pixels.grad, global_ref.grad, atol=1e-5, rtol=5e-4)
    torch.testing.assert_close(local_pixels.grad, local_ref.grad, atol=1e-5, rtol=5e-4)
    torch.testing.assert_close(
        native.separator.weight.grad, reference_separator.grad, atol=3e-5, rtol=5e-4
    )
    for name, p in native.language_model.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=4e-5, rtol=1e-3, msg=name
        )
