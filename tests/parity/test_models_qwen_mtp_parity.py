from dataclasses import replace
import importlib
import pytest
import torch
from torch import nn
from aster.models import QwenMTPConfig, Qwen35TextConfig, Qwen35MoETextConfig, build_model
from aster.nn.parameter_codec import public_parameter_names


def text_config(tf, c):
    fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rms_norm_eps",
        "attention_bias",
        "attention_dropout",
        "initializer_range",
        "tie_word_embeddings",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
    )
    values = {key: getattr(c, key) for key in fields}
    values.update(
        layer_types=list(c.layer_types),
        pad_token_id=None,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": c.rope.theta,
            "partial_rotary_factor": c.partial_rotary_factor,
            "mrope_section": list(c.mrope_section),
            "mrope_interleaved": True,
        },
    )
    if c.num_experts:
        values.update(
            {
                key: getattr(c, key)
                for key in (
                    "num_experts",
                    "num_experts_per_tok",
                    "moe_intermediate_size",
                    "shared_expert_intermediate_size",
                )
            }
        )
    result = (tf.Qwen3_5MoeTextConfig if c.num_experts else tf.Qwen3_5TextConfig)(**values)
    result._attn_implementation = "eager"
    return result


class OracleHead(nn.Module):
    def __init__(self, tf, c, embeddings, head):
        super().__init__()
        base = c.text_config
        name, prefix = ("qwen3_5_moe", "Qwen3_5Moe") if base.num_experts else ("qwen3_5", "Qwen3_5")
        module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        lc = replace(
            base,
            num_hidden_layers=c.num_mtp_layers,
            layer_types=("full_attention",) * c.num_mtp_layers,
        )
        self.reference_config = text_config(tf, lc)
        self.embed_tokens, self.lm_head = embeddings, head
        self.fc = nn.Linear(base.hidden_size * 2, base.hidden_size, bias=False)
        for field in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
            self.add_module(
                field, getattr(module, prefix + "RMSNorm")(base.hidden_size, base.rms_norm_eps)
            )
        self.layers = nn.ModuleList(
            getattr(module, prefix + "DecoderLayer")(self.reference_config, i)
            for i in range(c.num_mtp_layers)
        )
        self.rotary_emb = getattr(module, prefix + "TextRotaryEmbedding")(self.reference_config)

    def forward(self, ids, hidden, positions, mask, step, cache=None):

        x = self.fc(
            torch.cat(
                (
                    self.pre_fc_norm_embedding(self.embed_tokens(ids)),
                    self.pre_fc_norm_hidden(hidden),
                ),
                -1,
            )
        )
        index = step % len(self.layers)
        seen = 0 if cache is None else cache.get_seq_length(index)
        q = torch.arange(seen, seen + x.shape[1], device=x.device)
        k = torch.arange(seen + x.shape[1], device=x.device)
        allowed = (k[None] <= q[:, None])[None, None].expand(x.shape[0], 1, -1, -1)
        if mask is not None:
            allowed = allowed & mask[:, None, None].bool()
        additive = x.new_zeros(allowed.shape).masked_fill(~allowed, -torch.inf)
        x = self.layers[index](
            x,
            attention_mask=additive,
            position_ids=positions,
            position_embeddings=self.rotary_emb(x, positions),
            past_key_values=cache,
        )
        x = self.norm(x)
        return self.lm_head(x), x


@pytest.mark.oracle
@pytest.mark.parametrize("moe", [False, True])
def test_models_qwen_mtp_official_layers_full_gradient_and_draft_cache(moe):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(220)
    c = QwenMTPConfig(
        text_config=Qwen35MoETextConfig() if moe else Qwen35TextConfig(), num_mtp_layers=2
    )
    native = build_model(c).eval()
    oracle = nn.Module()
    oracle.backbone = (tf.Qwen3_5MoeForCausalLM if moe else tf.Qwen3_5ForCausalLM)(
        text_config(tf, c.text_config)
    ).eval()
    oracle.mtp = OracleHead(
        tf, c, oracle.backbone.get_input_embeddings(), oracle.backbone.lm_head
    ).eval()
    oracle.load_state_dict(native.state_dict(), strict=True)
    ids = torch.randint(1, 30, (2, 8))
    mask = torch.ones_like(ids)
    mask[1, -1] = 0
    expected = oracle.backbone(ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
    actual = native(ids, attention_mask=mask, mtp_depth=3)
    torch.testing.assert_close(actual.logits, expected.logits, atol=3e-6, rtol=4e-5)
    previous = expected.hidden_states[-1]
    reference_logits = []
    for step in range(3):
        offset = step + 1
        positions = torch.arange(offset, ids.shape[1])[None].expand(ids.shape[0], -1)
        logits, previous = oracle.mtp(
            ids[:, offset:], previous[:, :-1], positions, mask[:, offset:], step
        )
        reference_logits.append(logits)
        torch.testing.assert_close(
            actual.auxiliary["mtp_logits"][step], logits, atol=3e-6, rtol=4e-5
        )
    factors = [torch.randn_like(x) for x in reference_logits]
    sum(
        (value * coef).sum() for value, coef in zip(actual.auxiliary["mtp_logits"], factors)
    ).backward()
    sum((value * coef).sum() for value, coef in zip(reference_logits, factors)).backward()
    names = public_parameter_names(native)
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(oracle.named_parameters())[names[name]].grad,
            atol=8e-5,
            rtol=9e-4,
            msg=name,
        )

    hidden = torch.randn(2, 6, c.text_config.hidden_size)
    tokens = ids[:, 1:7]
    positions = torch.arange(1, 7)[None].expand(2, -1)
    ref_cache = tf.DynamicCache(config=oracle.mtp.reference_config)
    for step in (0, 1):
        ref_cache = tf.DynamicCache(config=oracle.mtp.reference_config)
        local = native.mtp(
            tokens[:, :3],
            hidden_states=hidden[:, :3],
            position_ids=positions[:, :3],
            spec_step_idx=step,
            use_cache=True,
        )
        oracle.mtp(tokens[:, :3], hidden[:, :3], positions[:, :3], None, step, ref_cache)
        local = native.mtp(
            tokens[:, 3:],
            hidden_states=hidden[:, 3:],
            position_ids=positions[:, 3:],
            spec_step_idx=step,
            state=local.state,
            use_cache=True,
        )
        logits, _ = oracle.mtp(
            tokens[:, 3:], hidden[:, 3:], positions[:, 3:], None, step, ref_cache
        )
        torch.testing.assert_close(local.logits, logits, atol=3e-6, rtol=4e-5)
