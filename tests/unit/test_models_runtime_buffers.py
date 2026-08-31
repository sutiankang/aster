import hashlib
import torch
import pytest
from aster.core import atomic_json, read_json
from aster.models import LlamaConfig, build_model, load_model
from aster.models.serialization import semantic_buffers


def test_models_runtime_buffer_preserves_rounded_rope_and_legacy_format(tmp_path):
    torch.manual_seed(682)
    model = build_model(LlamaConfig()).bfloat16()
    path = tmp_path / "model"
    model.save_pretrained(path)
    restored = load_model(path)
    for name, value in model.named_buffers():
        torch.testing.assert_close(dict(restored.named_buffers())[name], value, atol=0, rtol=0)
    manifest = read_json(path / "config.json")
    manifest.pop("runtime_buffers")
    atomic_json(path / "config.json", manifest)
    legacy = load_model(path)

    assert next(legacy.buffers()).dtype == torch.float32


def test_models_runtime_buffer_hash_and_shape_guard(tmp_path):
    model = build_model(LlamaConfig())
    path = tmp_path / "model"
    model.save_pretrained(path)
    auxiliary = path / "runtime_buffers.pt"
    buffers = torch.load(auxiliary, weights_only=True)
    name = next(iter(buffers))
    buffers[name] = buffers[name][:-1]
    torch.save(buffers, auxiliary)
    with pytest.raises(ValueError, match="checksum"):
        load_model(path)
    manifest = read_json(path / "config.json")
    manifest["runtime_buffers"]["sha256"] = hashlib.sha256(auxiliary.read_bytes()).hexdigest()
    atomic_json(path / "config.json", manifest)
    with pytest.raises(ValueError, match="shape"):
        load_model(path)


def test_models_runtime_buffer_explicit_semantics_excludes_dynamic_cache():
    model = torch.nn.Module()
    model.register_buffer("cache", torch.arange(9), persistent=False)
    model.register_buffer("frequency", torch.tensor([1.0, 0.3]), persistent=False)
    model._aster_semantic_buffers = ("frequency",)
    assert set(semantic_buffers(model)) == {"frequency"}
    model._aster_semantic_buffers = ("missing",)
    with pytest.raises(ValueError, match="existing nonpersistent"):
        semantic_buffers(model)
