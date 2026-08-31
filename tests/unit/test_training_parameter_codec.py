from copy import deepcopy

import pytest
import torch
from torch import nn

from aster.core import LossTerm
from aster.training import ParallelContext, Trainer
from aster.training.portable import logical_tensors


def _export(module, state, prefix, metadata):
    changes = {
        prefix + key: prefix + value
        for key, value in module._aster_parameter_key_map.items()
        if prefix + key in state
    }
    values = {destination: state[source] for source, destination in changes.items()}
    if any(destination in state and destination not in changes for destination in values):
        raise ValueError("Codec collision")
    for source in changes:
        del state[source]
    state.update(values)


def _import(module, state, prefix, metadata, strict, missing, unexpected, errors):
    for internal, public in module._aster_parameter_key_map.items():
        if prefix + public in state:
            if prefix + internal in state:
                raise ValueError("Both internal and public keys")
            state[prefix + internal] = state.pop(prefix + public)


class PublicLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.internal = nn.Linear(3, 2)
        self._aster_parameter_key_map = {"internal.weight": "weight", "internal.bias": "bias"}
        self.register_state_dict_post_hook(_export)
        self.register_load_state_dict_pre_hook(_import)

    def forward(self, value):
        return self.internal(value)


def _model():
    return nn.Sequential(PublicLinear(), nn.Tanh())


def _loss(model, data):
    values = (model(data) - 0.3).square()
    return LossTerm(values.sum(), torch.tensor(values.numel()), "elements")


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_explicit_parameter_codec_nested_target_ema_and_checkpoint(stage, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(837)
    engine = Trainer(_model(), _loss, zero_stage=stage, ema_decay=0.8, max_grad_norm=None)
    target = engine.clone_target("model", "target", factory=_model)
    prior = deepcopy(target.state_dict())
    entries = logical_tensors(engine.model, engine.parallel)
    assert {entry.name for entry in entries} == {"0.weight", "0.bias"}
    for entry in entries:
        assert entry.storage_name == (
            f"0.internal.shards.{0 if entry.name.endswith('weight') else 1}"
            if stage == 3
            else entry.name
        )
        assert entry.storage_name in engine.roles["model"].ema.shadow
    data = torch.randn(5, 3)
    engine.step([data])
    weights = engine.export_state_dict()
    ema = engine.export_state_dict(ema=True)
    for name, value in weights.items():
        torch.testing.assert_close(ema[name], 0.8 * prior[name] + 0.2 * value)
    engine.update_target("model", "target", 0.5)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, 0.5 * prior[name] + 0.5 * weights[name])
    dense = _model()
    dense.load_state_dict(weights, strict=True)
    torch.testing.assert_close(dense(data), engine.model(data))
    checkpoint = engine.save_checkpoint(tmp_path / "native")
    engine.step([data])
    expected = engine.export_state_dict(ema=True)
    engine.load_checkpoint(checkpoint)
    engine.step([data])
    for name, value in engine.export_state_dict(ema=True).items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)


@pytest.mark.parametrize("source_stage,target_stage", [(0, 3), (3, 0)])
def test_parameter_codec_portable_dense_shard_ema_and_optimizer_migration(
    source_stage, target_stage, tmp_path
):
    torch.manual_seed(718)
    source = Trainer(_model(), _loss, zero_stage=source_stage, ema_decay=0.9, max_grad_norm=None)
    data = torch.randn(4, 3)
    source.step([data])
    source.save_portable_checkpoint(tmp_path / "portable")
    target = Trainer(_model(), _loss, zero_stage=target_stage, ema_decay=0.9, max_grad_norm=None)
    target.load_portable_checkpoint(tmp_path / "portable", seed=83)
    source.step([data])
    target.step([data])
    for ema in (False, True):
        expected = source.export_state_dict(ema=ema)
        for name, value in target.export_state_dict(ema=ema).items():
            torch.testing.assert_close(value, expected[name], atol=3e-7, rtol=3e-6)


@pytest.mark.parametrize(
    "mapping",
    [
        [],
        {"missing.weight": "weight"},
        {"internal.weight": "internal.bias"},
        {"internal.weight": ""},
        {"internal.weight": "../weight"},
        {"internal.weight": "weight", "internal.bias": "weight"},
    ],
)
def test_invalid_parameter_codec_fails_without_guessing_or_collapsing_aliases(mapping):
    model = PublicLinear()
    model._aster_parameter_key_map = mapping
    with pytest.raises((TypeError, ValueError)):
        logical_tensors(model, ParallelContext())


def test_parameter_codec_requires_actual_dense_state_dict_hook_and_rejects_buffer_mapping():
    model = nn.Sequential(nn.Linear(3, 2))
    model._aster_parameter_key_map = {"0.weight": "weight"}
    with pytest.raises(ValueError, match="state_dict codec"):
        logical_tensors(model, ParallelContext())
    model._aster_parameter_key_map = {"marker": "public_marker"}
    model.register_buffer("marker", torch.ones(1))
    with pytest.raises(ValueError, match="buffers"):
        logical_tensors(model, ParallelContext())


def test_changed_codec_cannot_resume_same_native_shard_shapes(tmp_path):
    original = Trainer(_model(), zero_stage=3)
    checkpoint = original.save_checkpoint(tmp_path / "native")
    changed_model = _model()
    changed_model[0]._aster_parameter_key_map["internal.weight"] = "another_weight"
    changed = Trainer(changed_model, zero_stage=3)

    assert original.model.state_dict().keys() == changed.model.state_dict().keys()
    with pytest.raises(ValueError, match="checkpoint"):
        changed.load_checkpoint(checkpoint)


def test_distinct_aliases_keep_one_storage_owner_after_explicit_rename():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.left, self.right = nn.Linear(3, 2), nn.Linear(3, 2)
            self.right.weight = self.left.weight
            self._aster_parameter_key_map = {
                "left.weight": "shared_left",
                "right.weight": "shared_right",
            }
            self.register_state_dict_post_hook(_export)
            self.register_load_state_dict_pre_hook(_import)

        def forward(self, data):
            return self.left(data) + self.right(data)

    engine = Trainer(Tied(), zero_stage=3)
    entries = {entry.name: entry for entry in logical_tensors(engine.model, engine.parallel)}
    assert entries["shared_left"].tensor is entries["shared_right"].tensor
    assert len(engine.roles["model"].parameters) == 3
    exported = engine.export_state_dict()
    torch.testing.assert_close(exported["shared_left"], exported["shared_right"])
