from copy import deepcopy
import json
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from aster.models import build_model, load_model, LlamaConfig, Qwen2Config, Qwen3Config
from aster.inference import PackedLinear, save_optimized_model, load_optimized_model
from aster.optimization import (
    QATLinear,
    grouped_fake_quantize,
    prepare_qat,
    configure_qat,
    convert_qat,
    mlp_importance,
    prune_mlp,
    CompileProvider,
    CUDAGraphProvider,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def small_model(kind=LlamaConfig):
    return build_model(
        kind(
            vocab_size=24,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    )


@pytest.mark.parametrize("bits", [4, 8])
def test_grouped_qat_matches_independent_torch_forward_and_clipped_ste(bits):
    torch.manual_seed(27)
    qmax = 2 ** (bits - 1) - 1
    weight = torch.randn(3, 7).mul(qmax).requires_grad_()

    with torch.no_grad():
        weight[0, :2] = torch.tensor([qmax + 0.2, qmax + 0.7])
    reference = weight.detach().clone().requires_grad_()
    scales = torch.ones(3, 2)
    scales[1] = 0.3
    actual = grouped_fake_quantize(weight, scales, bits=bits, group_size=4)
    padded = F.pad(reference, (0, 1)).reshape(6, 4)
    oracle = torch.fake_quantize_per_channel_affine(
        padded, scales.flatten(), torch.zeros(6, dtype=torch.int32), 0, -qmax, qmax
    ).reshape(3, 8)[:, :7]
    torch.testing.assert_close(actual, oracle)
    coefficients = torch.randn_like(actual)
    (actual * coefficients).sum().backward()
    (oracle * coefficients).sum().backward()
    torch.testing.assert_close(weight.grad, reference.grad, rtol=0, atol=0)
    assert weight.grad[0, 0] != 0 and weight.grad[0, 1] == 0


def test_observer_freeze_and_packed_export_keep_training_scales_exactly():
    source = nn.Linear(7, 3)
    qat = QATLinear(source, bits=4, group_size=4)
    qat.observer_enabled.fill_(False)
    original_scales = qat.scales.clone()
    with torch.no_grad():
        qat.weight.mul_(1.7)
    inputs = torch.randn(5, 7)
    result = qat(inputs)
    packed = qat.to_packed()
    torch.testing.assert_close(packed.scales, original_scales, rtol=0, atol=0)
    torch.testing.assert_close(packed(inputs), result, rtol=1e-6, atol=1e-6)
    assert (
        packed.algorithm == "qat_symmetric_weight_only"
        and packed.packed_weight.dtype == torch.uint8
    )
    assert source.weight.data_ptr() != qat.weight.data_ptr()
    qat.fake_quant_enabled.fill_(False)
    with pytest.raises(ValueError):
        qat.to_packed()


def test_qat_transform_rejects_aliases_and_roundtrips_native_packed_artifact(tmp_path):
    model = small_model()
    with pytest.raises(ValueError, match="Tied"):
        prepare_qat(model, targets=["lm_head"])
    target = "model.layers.0.mlp.up_proj"
    qat = prepare_qat(model, targets=[target], group_size=8)
    assert type(model.get_submodule(target)) is nn.Linear and isinstance(
        qat.get_submodule(target), QATLinear
    )
    configure_qat(qat, observe=False)
    ids = torch.tensor([[1, 3, 2]])
    converted = convert_qat(qat)
    save_optimized_model(converted, tmp_path, base_artifact_id="qat-fixture-v1")
    loaded = load_optimized_model(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(qat(ids).logits, converted(ids).logits)
        torch.testing.assert_close(converted(ids).logits, loaded(ids).logits)
    assert isinstance(loaded.get_submodule(target), PackedLinear)
    assert not any(p.requires_grad for p in converted.parameters())


@pytest.mark.parametrize("kind", [LlamaConfig, Qwen2Config, Qwen3Config])
def test_structural_pruning_matches_dense_mask_oracle_and_native_config_reload(kind, tmp_path):
    torch.manual_seed(19)
    model = small_model(kind).eval()
    scores = {index: torch.arange(24, dtype=torch.float32).roll(index) for index in range(2)}
    result = prune_mlp(
        model, intermediate_size=12, importance=scores, parent_artifact_id="unpruned-native"
    )
    masked = deepcopy(model)
    for index, layer in enumerate(masked.model.layers):
        removed = sorted(set(range(24)) - set(result.manifest["kept_channels"][str(index)]))
        with torch.no_grad():
            layer.mlp.down_proj.weight[:, removed] = 0
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(
            masked(ids).logits, result.model(ids).logits, atol=1e-6, rtol=1e-5
        )
    assert model.config.intermediate_size == 24 and result.model.config.intermediate_size == 12
    assert result.manifest["parameters_after"] < result.manifest["parameters_before"]
    assert result.model.lm_head.weight is result.model.model.embed_tokens.weight
    result.model.save_pretrained(tmp_path)
    loaded = load_model(tmp_path)
    assert loaded.config.intermediate_size == 12 and loaded.model.layers[
        0
    ].mlp.gate_proj.weight.shape == (12, 16)
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, result.model(ids).logits)
    assert json.loads((tmp_path / "config.json").read_text())["config"]["intermediate_size"] == 12


def test_activation_importance_uses_real_post_swiglu_and_valid_token_mask():
    model = small_model()
    batch = {"input_ids": torch.tensor([[1, 2, 0]]), "attention_mask": torch.tensor([[1, 1, 0]])}
    values = []
    handle = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(
        lambda module, args: values.append(args[0].detach())
    )
    score = mlp_importance(model, batches=[batch], dataset_fingerprint="explicit-token-fixture")
    handle.remove()
    expected = (
        model.model.layers[0].mlp.down_proj.weight.detach().norm(dim=0)
        * values[0][:, :2].square().mean((0, 1)).sqrt()
    )
    torch.testing.assert_close(score[0], expected)
    qat = prepare_qat(model, targets=["model.layers.0.mlp.up_proj"])
    with pytest.raises(ValueError, match="Prune before"):
        prune_mlp(qat, intermediate_size=12, parent_artifact_id="qat")


class TensorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 8)

    def forward(self, x):
        return F.silu(self.proj(x))


def test_real_torch_compile_aot_eager_bucket_lifecycle_and_immutable_policy():
    model = TensorBlock()
    provider = CompileProvider(
        model, policy_artifact_id="immutable-native-v1", backend="aot_eager", max_buckets=2
    )
    x = torch.randn(2, 4)
    bucket = provider.prepare("b2", {"x": x})
    assert bucket.status == "ready" and bucket.callable is not None
    actual = provider("b2", x=x)
    with torch.no_grad():
        torch.testing.assert_close(actual, model(x))
    changed = provider("b2", x=x + 1)
    with torch.no_grad():
        torch.testing.assert_close(changed, model(x + 1))
    assert bucket.calls == 2 and bucket.forward_seconds > 0
    assert provider.observation()["evidence_kind"] == "native_math_reference"
    with pytest.raises(ValueError, match="differs"):
        provider("b2", x=torch.randn(1, 4))
    with pytest.raises(ValueError):
        provider.prepare("b2", {"x": x})

    with torch.no_grad():
        model.proj.weight.add_(1)
    torch.testing.assert_close(provider("b2", x=x), actual)
    provider.model.proj.weight = nn.Parameter(
        provider.model.proj.weight.detach().clone(), requires_grad=False
    )
    with pytest.raises(RuntimeError, match="version changed"):
        provider("b2", x=x)
    provider.close()
    with pytest.raises(RuntimeError, match="closed"):
        provider("b2", x=x)


def test_compile_failure_is_sticky_and_has_no_silent_eager_fallback():
    class InvalidOutput(nn.Module):
        def forward(self, x):
            return {"value": x}

    provider = CompileProvider(
        InvalidOutput(), policy_artifact_id="invalid-fixture", backend="aot_eager"
    )
    with pytest.raises(ValueError):
        provider.prepare("failed", {"x": torch.ones(2)})
    assert provider.buckets["failed"].status == "failed"
    with pytest.raises(RuntimeError, match="not ready"):
        provider("failed", x=torch.ones(2))
    with pytest.raises(ValueError):
        CompileProvider(TensorBlock(), policy_artifact_id="x", atol=float("nan"))


def test_cuda_graph_has_real_device_gate_without_cpu_fallback():
    if torch.cuda.is_available():
        pytest.skip("CPU rejection check only; separate CUDA execution test covers GPU")
    with pytest.raises(RuntimeError, match="CUDA runtime"):
        CUDAGraphProvider(TensorBlock(), policy_artifact_id="gpu-only")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA Graph requires provisioned CUDA runtime/device; no numerical GPU claim on CPU",
)
def test_cuda_graph_real_replay_updates_inputs_and_does_not_alias_outputs():
    model = TensorBlock().cuda()
    provider = CUDAGraphProvider(model, policy_artifact_id="cuda-fixture-v1")
    x = torch.randn(2, 4, device="cuda")
    provider.prepare("b2", {"x": x})
    first = provider("b2", x=x)
    second = provider("b2", x=x + 1)
    with torch.no_grad():
        torch.testing.assert_close(first, model(x))
        torch.testing.assert_close(second, model(x + 1))
    assert first.data_ptr() != second.data_ptr()
    assert provider.buckets["b2"].calls == 2
    provider.close()
