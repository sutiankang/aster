import json
import pytest
import torch
from torch import nn

from aster.models import build_model, LlamaConfig
from aster.inference import (
    PackedLinear,
    CalibrationData,
    quantize_linear,
    quantize_model,
    collect_calibration,
    save_optimized_model,
    load_optimized_model,
    FiniteJSONGrammar,
    ChatTemplate,
    ModelRunner,
    InferenceEngine,
    SamplingConfig,
)
from aster.data import ByteTokenizer


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_int4_is_packed_and_reference_matmul_is_honestly_labeled():
    torch.manual_seed(9)
    linear = nn.Linear(64, 32)
    packed = quantize_linear(linear, group_size=32)
    assert packed.packed_weight.dtype == torch.uint8
    assert packed.packed_weight.numel() == 64 * 32 // 2
    assert packed.stored_bytes < sum(p.numel() * p.element_size() for p in linear.parameters())
    inputs = torch.randn(4, 64)
    reference = torch.nn.functional.linear(
        inputs, packed.dequantized_weight(original_coordinates=True), packed.bias
    )
    torch.testing.assert_close(packed(inputs), reference)
    assert packed.compute_provider == "torch_float_dequant_reference"
    assert not list(packed.parameters())


def test_gptq_matches_independent_inverse_schur_column_oracle():
    linear = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[0.31, 0.91, -0.53]]))
    x = torch.tensor([[1.0, 1.0, 0.0], [2.0, 1.0, 0.1], [0.0, 2.0, 1.0], [1.0, -1.0, 3.0]])
    damping = 0.01
    packed = quantize_linear(
        linear,
        algorithm="gptq",
        group_size=3,
        calibration=CalibrationData(x, "fixed-calibration"),
        damping=damping,
    )
    h = x.t() @ x / len(x)
    h.diagonal().add_(damping * h.diag().mean())
    inverse = torch.linalg.inv(h)
    remaining = linear.weight.detach().clone()[0]
    scale = remaining.abs().max() / 7
    expected = []
    for _ in range(3):
        q = (remaining[0] / scale).round().clamp(-7, 7) * scale
        expected.append(q)
        error = remaining[0] - q
        remaining = remaining[1:] - error * inverse[0, 1:] / inverse[0, 0]
        inverse = inverse[1:, 1:] - inverse[1:, :1] @ inverse[:1, 1:] / inverse[0, 0]
    torch.testing.assert_close(
        packed.dequantized_weight()[0], torch.stack(expected), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("algorithm", ["smoothquant", "gptq", "awq_linear"])
def test_calibrated_algorithms_are_real_finite_transformations(algorithm):
    torch.manual_seed(7)
    linear = nn.Linear(12, 6)
    data = torch.randn(32, 12) * torch.arange(1, 13)
    packed = quantize_linear(
        linear,
        algorithm=algorithm,
        group_size=4,
        calibration=CalibrationData(data, "data-v1"),
        search_grid=4,
        clip_grid=3,
        act_order=algorithm == "gptq",
    )
    assert torch.isfinite(packed(data)).all()
    assert packed.calibration_fingerprint == "data-v1" and packed.algorithm == algorithm
    if algorithm == "smoothquant":
        assert not torch.allclose(packed.input_scale, torch.ones(12))
    if algorithm == "gptq":
        assert sorted(packed.permutation.tolist()) == list(range(12))
    with pytest.raises(ValueError):
        quantize_linear(linear, algorithm=algorithm)


def test_calibration_masks_and_optimized_export_reload(tmp_path):
    model = build_model(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    target = "model.layers.0.self_attn.q_proj"
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [3, 4, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
    }
    calibration = collect_calibration(
        model, [batch], targets=[target], dataset_fingerprint="real-token-fixture"
    )
    assert calibration[target].inputs.shape == (5, 16)
    optimized = quantize_model(
        model, targets=[target], group_size=8, algorithm="gptq", calibration=calibration
    )
    assert isinstance(model.get_submodule(target), nn.Linear)
    assert isinstance(optimized.get_submodule(target), PackedLinear)
    save_optimized_model(optimized, tmp_path, base_artifact_id="baseline-fixture")
    restored = load_optimized_model(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(optimized(**batch).logits, restored(**batch).logits)


def test_finite_json_grammar_constrains_actual_native_sampling():
    import asyncio

    async def exercise():
        tokenizer = ByteTokenizer()
        model = build_model(
            LlamaConfig(
                vocab_size=259,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        )
        grammar = FiniteJSONGrammar(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"ok": {"type": "boolean"}, "label": {"enum": ["甲", "乙"]}},
                "required": ["ok", "label"],
            },
            tokenizer,
        )
        engine = InferenceEngine(
            ModelRunner(model, policy_artifact_id="native-json", tokenizer=tokenizer)
        )
        handle = await engine.submit(
            [1, 8], SamplingConfig(max_new_tokens=64, temperature=0), grammar=grammar
        )
        result = await handle.collect()
        value = json.loads(result.text)
        assert type(value["ok"]) is bool and value["label"] in {"甲", "乙"}
        assert result.stop_reason == "grammar_complete"
        assert any(a != b for a, b in zip(result.raw_model_logprobs, result.behavior_logprobs))
        await engine.close()

    asyncio.run(exercise())


def test_grammar_rejects_unbounded_or_explosive_schema():
    tokenizer = ByteTokenizer()
    for schema in (
        {"type": "string"},
        {"type": "integer", "enum": ["wrong"]},
        {"type": "integer", "minimum": 0, "maximum": 99999},
        {"type": "object", "additionalProperties": True, "properties": {}},
    ):
        with pytest.raises(ValueError):
            FiniteJSONGrammar(schema, tokenizer)
    with pytest.raises(ValueError):
        ChatTemplate().render([{"role": "user", "content": [{"image_url": "http://untrusted"}]}])


def test_packed_artifact_keeps_bfloat16_weights_fp32_scales_and_tied_alias(tmp_path):
    model = build_model(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    ).to(torch.bfloat16)
    target = "model.layers.0.mlp.up_proj"
    packed = quantize_model(model, targets=[target], group_size=8)
    save_optimized_model(packed, tmp_path, base_artifact_id="bf16-native-fixture")
    loaded = load_optimized_model(tmp_path)
    assert loaded.model.embed_tokens.weight.dtype == torch.bfloat16
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert loaded.get_submodule(target).scales.dtype == torch.float32
    assert loaded.get_submodule(target).packed_weight.dtype == torch.uint8
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(loaded(ids).logits, packed(ids).logits, rtol=0, atol=0)


def test_quantization_does_not_replace_unknown_linear_subclass_semantics():
    class SpecialLinear(nn.Linear):
        def forward(self, x):
            return super().forward(x) * 2

    layer = SpecialLinear(4, 4)
    with pytest.raises(ValueError):
        quantize_linear(layer)
    with pytest.raises(ValueError):
        quantize_model(nn.Sequential(layer), targets=["0"])
