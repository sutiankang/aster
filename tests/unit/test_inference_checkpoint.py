import json
from dataclasses import asdict
import pytest
import torch

from aster.models import build_model, LlamaConfig
from aster.inference import native_config_from_hf, load_hf_safetensors


def fixture_checkpoint(path, *, tied=False, shards=True):
    safetensors = pytest.importorskip("safetensors.torch")
    c = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tied,
    )
    model = build_model(c).eval()
    config = asdict(c)
    config.pop("rope")
    config.update(
        model_type="llama",
        rope_theta=c.rope.theta,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
    )
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not tied or name != "lm_head.weight"
    }
    if shards:
        names = list(tensors)
        first, second = {n: tensors[n] for n in names[::2]}, {n: tensors[n] for n in names[1::2]}
        safetensors.save_file(first, path / "model-00001-of-00002.safetensors")
        safetensors.save_file(second, path / "model-00002-of-00002.safetensors")
        mapping = {
            **{n: "model-00001-of-00002.safetensors" for n in first},
            **{n: "model-00002-of-00002.safetensors" for n in second},
        }
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": mapping}), encoding="utf-8"
        )
    else:
        safetensors.save_file(tensors, path / "model.safetensors")
    return model


@pytest.mark.parametrize("tied,shards", [(False, True), (True, False)])
def test_native_streaming_checkpoint_exact_output_and_alias(tmp_path, tied, shards):
    torch.set_num_threads(1)
    model = fixture_checkpoint(tmp_path, tied=tied, shards=shards)
    loaded = load_hf_safetensors(tmp_path)
    ids = torch.tensor([[1, 3, 7, 9]])
    torch.testing.assert_close(loaded(ids).logits, model(ids).logits, atol=0, rtol=0)
    assert not any(p.is_meta for p in loaded.parameters())
    assert (
        loaded.load_report["largest_materialized_tensor_bytes"]
        < loaded.load_report["source_parameter_bytes"]
    )
    if tied:
        assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    with torch.no_grad():
        model.lm_head.weight.zero_()
    assert loaded.lm_head.weight.abs().sum() > 0
    with pytest.raises(ValueError, match="hash"):
        load_hf_safetensors(tmp_path, expected_hashes={"config.json": "0" * 64})


def test_checkpoint_bad_semantics_paths_and_tensor_shapes_rejected(tmp_path):
    fixture_checkpoint(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    with pytest.raises(ValueError, match="Unrecognized"):
        native_config_from_hf({**config, "fancy_attention": True})
    with pytest.raises(ValueError, match="Remote-code"):
        native_config_from_hf({**config, "auto_map": {"AutoModel": "remote.evil"}})
    with pytest.raises(ValueError, match="activation"):
        native_config_from_hf({**config, "hidden_act": "relu"})
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    first = next(iter(index["weight_map"]))
    index["weight_map"][first] = "../outside.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="relative"):
        load_hf_safetensors(tmp_path)


@pytest.mark.oracle
@pytest.mark.parametrize("family", ["Llama", "Qwen2", "Qwen3"])
def test_real_official_safetensors_export_into_native_loader(tmp_path, family):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    torch.set_num_threads(1)
    config = getattr(transformers, family + "Config")(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    official = getattr(transformers, family + "ForCausalLM")(config).eval()
    official.save_pretrained(tmp_path, max_shard_size="2KB")
    native = load_hf_safetensors(tmp_path)
    ids = torch.tensor([[1, 3, 7, 9]])
    torch.testing.assert_close(native(ids).logits, official(ids).logits, atol=3e-6, rtol=3e-5)
