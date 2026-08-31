from copy import deepcopy

import pytest
import torch
from torch import nn

from aster.models import LlamaConfig, build_model
from aster.core import LossTerm, atomic_json, read_json
from aster.training import Trainer
from aster.training.state import read_payload, write_payload
from aster.training.runtime_state import (
    runtime_buffers,
    runtime_descriptor,
    snapshot_runtime_state,
    validate_runtime_state,
    restore_runtime_state,
    apply_runtime_state,
)


class RuntimeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        frequencies = torch.tensor([0.3333333, 0.1234567])
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.register_buffer("shared_frequencies", frequencies, persistent=False)
        self.register_buffer("temporary_cache", torch.zeros(7), persistent=False)
        self._aster_semantic_buffers = ("frequencies", "shared_frequencies")


def test_runtime_state_restores_rounding_and_preserves_public_parameter_keys():
    torch.set_num_threads(1)
    config = LlamaConfig(
        vocab_size=17,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    trained = build_model(config).bfloat16().float()
    fresh = build_model(config)
    fresh.load_state_dict(trained.state_dict())
    assert any(
        not torch.equal(value, runtime_buffers(fresh)[name])
        for name, value in runtime_buffers(trained).items()
    )
    keys = set(fresh.state_dict())
    state = snapshot_runtime_state(trained)
    apply_runtime_state(fresh, state)
    assert set(fresh.state_dict()) == keys
    for name, value in runtime_buffers(fresh).items():
        assert torch.equal(value, runtime_buffers(trained)[name])
    ids = torch.tensor([[1, 5, 8, 9]])
    assert torch.equal(trained(ids).logits, fresh(ids).logits)


def test_buffers_are_explicit_and_aliases_are_not_split():
    source = RuntimeModule()
    state = snapshot_runtime_state(source)
    assert set(state["semantic_buffers"]) == {"frequencies", "shared_frequencies"}
    assert runtime_descriptor(source)["aliases"] == [["frequencies", "shared_frequencies"]]
    source.frequencies.zero_()
    source.temporary_cache.fill_(4)
    restore_runtime_state(source, state)
    assert source.frequencies is source.shared_frequencies
    assert torch.equal(source.temporary_cache, torch.full((7,), 4.0))
    target = RuntimeModule()
    state["semantic_buffers"] = {
        name: value.bfloat16() for name, value in state["semantic_buffers"].items()
    }
    apply_runtime_state(target, state)
    assert (
        target.frequencies is target.shared_frequencies
        and target.frequencies.dtype == torch.bfloat16
    )


@pytest.mark.parametrize("failure", ["shape", "dtype", "nan", "alias", "unknown", "missing"])
def test_runtime_preflight_has_no_partial_writes(failure):
    model = RuntimeModule()
    state = snapshot_runtime_state(model)
    before = deepcopy(model.state_dict())
    original = model.frequencies.clone()
    values = state["semantic_buffers"]
    if failure == "shape":
        values["frequencies"] = torch.zeros(3)
    elif failure == "dtype":
        values["frequencies"] = values["frequencies"].long()
    elif failure == "nan":
        values["frequencies"][0] = float("nan")
    elif failure == "alias":
        values["shared_frequencies"][0] += 1
    elif failure == "unknown":
        values["temporary_cache"] = torch.zeros(7)
    else:
        del values["shared_frequencies"]
    with pytest.raises(ValueError):
        restore_runtime_state(model, state)
    assert torch.equal(model.frequencies, original)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_runtime_owner_and_invalid_declaration_rejected():
    model = RuntimeModule()
    state = snapshot_runtime_state(model)
    model._aster_training_owned = True
    with pytest.raises(ValueError, match="Trainer-owned"):
        apply_runtime_state(model, state)
    del model._aster_training_owned
    model._aster_semantic_buffers = ("weight",)
    with pytest.raises(ValueError, match="nonpersistent"):
        validate_runtime_state(model, state)


def test_valid_checkpoint_can_recover_corrupted_current_values():
    model = RuntimeModule()
    saved = snapshot_runtime_state(model)
    model.frequencies.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        snapshot_runtime_state(model)

    restore_runtime_state(model, saved)
    assert torch.equal(model.frequencies, saved["semantic_buffers"]["frequencies"])


def language_objective(model, ids):
    output = model(ids).logits.float()
    loss = torch.nn.functional.cross_entropy(
        output[:, :-1].reshape(-1, output.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum"
    )
    return LossTerm(loss, torch.tensor(ids[:, 1:].numel(), dtype=torch.int64), "token")


def language_engine(zero=0, *, rounded=False, parallel=None):
    torch.manual_seed(123)
    torch.set_num_threads(1)
    config = LlamaConfig(
        vocab_size=17,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = build_model(config)
    if rounded:
        model.bfloat16().float()
    return Trainer(
        model,
        language_objective,
        zero_stage=zero,
        parallel=parallel,
        ema_decay=0.8,
        optimizer_factory=lambda p: torch.optim.AdamW(p, lr=0.001),
        offload_optimizer="cpu",
    )


@pytest.mark.parametrize("zero", [0, 1, 2, 3])
def test_fresh_native_portable_and_dense_deployment_preserve_semantic_state(zero, tmp_path):
    engine = language_engine(zero, rounded=True)
    ids = torch.tensor([[1, 5, 8, 9], [4, 2, 11, 6]])
    engine.step([ids])
    native = engine.save_checkpoint(tmp_path / "native")
    portable = engine.save_portable_checkpoint(tmp_path / "portable")
    runtime = engine.export_runtime_state()
    weights = engine.export_state_dict()
    fresh = language_engine(zero)
    assert any(
        not torch.equal(value, runtime_buffers(fresh.model)[name])
        for name, value in runtime["semantic_buffers"].items()
    )
    fresh.load_checkpoint(native)
    for name, value in runtime_buffers(fresh.model).items():
        assert torch.equal(value, runtime["semantic_buffers"][name])
    migrated = language_engine(0)
    migrated.load_portable_checkpoint(portable, seed=71)
    engine.step([ids])
    fresh.step([ids])
    migrated.step([ids])
    for name, value in engine.export_state_dict().items():
        assert torch.equal(value, fresh.export_state_dict()[name])
        torch.testing.assert_close(value, migrated.export_state_dict()[name], atol=2e-7, rtol=2e-6)
    for name, value in engine.export_state_dict(ema=True).items():
        assert torch.equal(value, fresh.export_state_dict(ema=True)[name])
    independent = build_model(engine.model.config)
    independent.load_state_dict(weights)
    apply_runtime_state(independent, runtime)
    independent.save_pretrained(tmp_path / "deployment")
    restored = type(independent).from_pretrained(tmp_path / "deployment")
    assert torch.equal(independent(ids).logits, restored(ids).logits)


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("failure", ["dtype", "missing", "nonfinite"])
def test_semantic_checkpoint_corruption_is_rejected_before_any_weights(portable, failure, tmp_path):
    engine = language_engine(3, rounded=True)
    ids = torch.tensor([[1, 5, 8, 9]])
    engine.step([ids])
    path = (engine.save_portable_checkpoint if portable else engine.save_checkpoint)(
        tmp_path / "original"
    )
    manifest = read_json(path)
    key = "entry" if portable else "entries"
    entry = manifest[key] if portable else manifest[key][0]
    payload = read_payload(path.parent, entry, trusted=False)
    saved = payload["roles"]["model"]
    if portable:
        name = next(name for name, record in saved["tensors"].items() if record.get("semantic"))
        values = saved["tensors"][name]
        if failure == "missing":
            del values["semantic"]
        elif failure == "dtype":
            values["value"] = values["value"].bfloat16()
        else:
            values["value"].fill_(float("nan"))
        for record in saved["tensors"].values():
            if record["parameter"]:
                record["value"].fill_(123.0)
    else:
        values = saved["runtime_state"]["semantic_buffers"]
        name = next(iter(values))
        if failure == "missing":
            del values[name]
        elif failure == "dtype":
            values[name] = values[name].bfloat16()
        else:
            values[name].fill_(float("nan"))
        for value in saved["model"].values():
            value.fill_(123.0)
    changed = write_payload(path.parent, "bad", payload)
    manifest[key] = changed if portable else [changed]
    atomic_json(tmp_path / "bad", manifest)
    before = engine.export_state_dict()
    with pytest.raises(ValueError, match="[Ss]emantic"):
        if portable:
            engine.load_portable_checkpoint(tmp_path / "bad", seed=4)
        else:
            engine.load_checkpoint(tmp_path / "bad")
    assert not engine._failed
    for name, value in engine.export_state_dict().items():
        assert torch.equal(value, before[name])
