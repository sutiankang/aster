import pytest
import torch
from dataclasses import asdict
from aster.models import (
    build_model,
    LlamaConfig,
    Qwen2Config,
    Qwen3Config,
    MistralConfig,
    MixtralConfig,
    DeepSeekV3Config,
)
from aster.nn import RopeConfig
from aster.nn.parameter_codec import public_parameter_names


@pytest.mark.oracle
@pytest.mark.parametrize(
    "family, config",
    [
        ("Llama", LlamaConfig()),
        ("Qwen2", Qwen2Config()),
        ("Qwen3", Qwen3Config()),
        ("Mistral", MistralConfig(sliding_window=3)),
        ("Mixtral", MixtralConfig(sliding_window=3)),
        ("DeepseekV3", DeepSeekV3Config()),
        (
            "DeepseekV3",
            DeepSeekV3Config(q_lora_rank=None, n_routed_experts=8, n_group=2, topk_group=1),
        ),
    ],
)
def test_models_official_forward_gradient_cache(family, config):
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(9)
    values = asdict(config)
    values.pop("rope")
    values["head_dim"] = config.attention_head_dim
    if family == "DeepseekV3":
        values["head_dim"] = config.qk_rope_head_dim
        values["rope_interleave"] = config.rope.interleaved
    values["rope_theta"] = config.rope.theta
    values["pad_token_id"] = None
    official_config = getattr(transformers, family + "Config")(**values)
    official_config._attn_implementation = "eager"
    native = build_model(config)
    official = getattr(transformers, family + "ForCausalLM")(official_config)
    official.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 4, 5, 7, 2], [1, 6, 7, 8, 2]])
    left, right = native(ids).logits, official(ids, use_cache=False).logits
    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    coefficients = torch.randn_like(left)
    (left * coefficients).sum().backward()
    (right * coefficients).sum().backward()
    names = public_parameter_names(native)
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(official.named_parameters())[names[name]].grad,
            atol=2e-5,
            rtol=2e-4,
            msg=names[name],
        )
    native.eval()
    official.eval()
    state = native(ids[:, :3], use_cache=True).state
    cache = official(ids[:, :3], use_cache=True).past_key_values
    torch.testing.assert_close(
        native(ids[:, 3:], state=state).logits,
        official(ids[:, 3:], past_key_values=cache, use_cache=True).logits,
        atol=2e-6,
        rtol=2e-5,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("kind", ["linear", "llama3", "yarn"])
def test_models_official_rope_scaling(kind):
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    rope = RopeConfig(kind=kind, factor=4.0, original_max_position_embeddings=16)
    config = LlamaConfig(rope=rope)
    kwargs = {"rope_type": kind, "factor": 4.0}
    if kind == "llama3":
        kwargs.update(
            low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=16
        )
    elif kind == "yarn":
        kwargs.update(original_max_position_embeddings=16, beta_fast=32.0, beta_slow=1.0)
    values = asdict(config)
    values.pop("rope")
    values["head_dim"] = config.attention_head_dim
    reference = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(**values, rope_scaling=kwargs)
    ).eval()
    native = build_model(config).eval()
    reference.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 3, 4, 5]])
    positions = torch.tensor([[0, 3, 18, 44]])
    torch.testing.assert_close(
        native(ids, position_ids=positions).logits,
        reference(ids, position_ids=positions).logits,
        atol=2e-6,
        rtol=2e-5,
    )
