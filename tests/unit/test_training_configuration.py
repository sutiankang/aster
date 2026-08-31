from dataclasses import dataclass, asdict, replace

import pytest
import torch
from torch import nn

from aster.training import Trainer


@dataclass(frozen=True)
class Configuration:
    architecture: str = "configured_mlp"
    dropout: float = 0.1
    log_std_min: float = -10.0

    def to_dict(self):
        return asdict(self)


class ConfiguredModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        return nn.functional.dropout(self.linear(value), self.config.dropout, self.training)


@pytest.mark.parametrize("zero_stage", [0, 3])
@pytest.mark.parametrize(
    "change", [{"dropout": 0.3}, {"log_std_min": -5.0}, {"architecture": "another_architecture"}]
)
def test_same_tensor_shapes_do_not_allow_changed_model_configuration(tmp_path, zero_stage, change):
    config = Configuration()
    first = Trainer(ConfiguredModel(config), zero_stage=zero_stage)
    path = first.save_checkpoint(tmp_path / "state")
    compatible = Trainer(ConfiguredModel(config), zero_stage=zero_stage)
    compatible.load_checkpoint(path)
    changed = Trainer(ConfiguredModel(replace(config, **change)), zero_stage=zero_stage)
    with pytest.raises(ValueError, match="checkpoint"):
        changed.load_checkpoint(path)


def test_unconfigured_pytorch_module_keeps_native_checkpoint_path(tmp_path):
    first = Trainer(nn.Sequential(nn.Linear(3, 2), nn.Tanh()))
    path = first.save_checkpoint(tmp_path / "state")
    restored = Trainer(nn.Sequential(nn.Linear(3, 2), nn.Tanh()))
    restored.load_checkpoint(path)
    assert first._identity()["roles"]["model"]["model_configuration"] is None
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[name])


def test_declared_configuration_must_be_explicit_and_finite_before_sharding():
    model = ConfiguredModel(Configuration(dropout=float("nan")))
    original = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError):
        Trainer(model, zero_stage=3)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])
    model.config = object()
    with pytest.raises(ValueError, match="to_dict"):
        Trainer(model)


@pytest.mark.parametrize("zero_stage", [0, 3])
@pytest.mark.parametrize("kind", ["groot", "distillation"])
def test_native_identity_rejects_real_objective_hyperparameter_change(tmp_path, zero_stage, kind):
    from aster.methods.groot import GrootFlowObjective
    from aster.methods.distillation import DistillationObjective

    def make(changed=False):
        objective = (
            GrootFlowObjective(noise_beta_alpha=2.0 if changed else 1.5)
            if kind == "groot"
            else DistillationObjective(nn.Linear(3, 2), temperature=2.0 if changed else 1.0)
        )
        engine = Trainer(nn.Linear(3, 2), objective, zero_stage=zero_stage)
        if kind == "distillation":
            engine.add_role("teacher", objective.teacher, trainable=False)
        return engine

    source = make()
    checkpoint = source.save_checkpoint(tmp_path / "checkpoint")
    compatible = make()
    compatible.load_checkpoint(checkpoint)
    incompatible = make(True)
    before = incompatible.export_state_dict()
    with pytest.raises(ValueError, match="checkpoint"):
        incompatible.load_checkpoint(checkpoint)
    for name, value in incompatible.export_state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_objective_codec_is_explicit_finite_json_and_plain_callable_is_compatible(tmp_path):
    class ExplicitObjective:
        def __init__(self, configuration):
            self.configuration = configuration

        def to_dict(self):
            return self.configuration

        def __call__(self, model, batch):
            raise AssertionError("Identity must not execute the objective")

    for invalid in (
        {"temperature": float("nan")},
        {"beta": float("inf")},
        {"tensor": torch.ones(1)},
        {1: "nonstring"},
        ["not_an_object"],
    ):
        model = nn.Linear(3, 2)
        original = {name: value.clone() for name, value in model.state_dict().items()}
        with pytest.raises(ValueError):
            Trainer(model, ExplicitObjective(invalid), zero_stage=3)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    configuration = {"pairs": ((1, 3),), "weight": 0.25, "enabled": True}
    engine = Trainer(nn.Linear(3, 2), ExplicitObjective(configuration))
    snapshot = engine._identity()["objective_configuration"]
    assert snapshot["configuration"]["pairs"] == [[1, 3]]
    configuration["weight"] = 0.5
    assert snapshot["configuration"]["weight"] == 0.25

    def old_objective(model, batch):
        return model(batch)

    first = Trainer(nn.Linear(3, 2), old_objective)
    checkpoint = first.save_checkpoint(tmp_path / "old")
    restored = Trainer(nn.Linear(3, 2), old_objective)
    restored.load_checkpoint(checkpoint)
    assert restored._identity()["objective_configuration"] is None
