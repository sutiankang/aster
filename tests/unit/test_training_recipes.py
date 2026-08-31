from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aster.recipes import TrainSettings
from aster.training import Trainer
from aster.training.recipes import RecipeSampler, RecipeState, recipe_context


class Records:
    fingerprint = "stable-test-records"

    def __len__(self):
        return 7

    def __getitem__(self, index):
        return index


def test_replica_sampler_disjoint_equal_epoch_and_exact_resume():
    samplers = [
        RecipeSampler(
            Records(), seed=41, context=SimpleNamespace(dp=SimpleNamespace(rank=rank, size=2))
        )
        for rank in range(2)
    ]
    for _ in range(3):
        rows = [sampler.take(9) for sampler in samplers]
        assert len(rows[0]) == len(rows[1]) == 3
        assert not set(rows[0]) & set(rows[1])
        assert samplers[0].dropped_per_epoch == 1
        for sampler in samplers:
            sampler.next_epoch()
    samplers[0].take(1)
    saved = samplers[0].state_dict()
    expected = samplers[0].take(2)
    samplers[0].load_state_dict(saved)
    assert samplers[0].take(2) == expected
    with pytest.raises(ValueError, match="divide"):
        RecipeSampler(
            Records(),
            seed=1,
            context=SimpleNamespace(dp=SimpleNamespace(rank=0, size=2)),
            tail="error",
        )
    with pytest.raises(ValueError, match="fewer records"):
        RecipeSampler(
            Records(), seed=1, context=SimpleNamespace(dp=SimpleNamespace(rank=0, size=8))
        )


@pytest.mark.parametrize(
    "options",
    [
        {"steps": True},
        {"seed": -1},
        {"zero_stage": 4},
        {"communication_overlap": True, "zero_stage": 2},
        {"replica_tail": "pad"},
        {"offload_parameters": "cpu"},
        {"max_consecutive_skips": -1},
        {"precision": "fp8"},
    ],
)
def test_recipe_settings_fail_fast(options):
    with pytest.raises(ValueError):
        TrainSettings(**options)


def test_recipe_identity_and_explicit_collective_optin(monkeypatch):
    state = RecipeState({"objective": "flow", "weight": 0.5})
    state.history.append({"step": 1, "loss": 0.4, "updated": True, "overflow": False})
    saved = state.state_dict()
    restored = RecipeState({"objective": "flow", "weight": 0.5})
    restored.load_state_dict(saved)
    assert restored.history == state.history and restored.history is not state.history
    with pytest.raises(ValueError, match="identity"):
        RecipeState({"objective": "flow", "weight": 0.6}).load_state_dict(saved)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="explicit parallel"):
        recipe_context()


@pytest.mark.parametrize("zero_stage", [0, 3])
def test_model_export_preserves_state_dict_persistent_buffers(zero_stage):
    model = nn.Sequential(nn.Linear(2, 2))
    model.register_buffer("persistent", torch.ones(2))
    model.register_buffer("cache", torch.arange(2.0), persistent=False)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    engine = Trainer(model, zero_stage=zero_stage, ema_decay=0.9)
    for ema in (False, True):
        exported = engine.export_state_dict(ema=ema)
        assert set(exported) == set(expected) and "cache" not in exported
        for key in expected:
            torch.testing.assert_close(exported[key], expected[key])


def test_cli_refuses_workflow_under_torchrun(monkeypatch, tmp_path, capsys):
    from aster.cli import main

    monkeypatch.setenv("WORLD_SIZE", "2")
    assert (
        main(
            [
                "train",
                str(tmp_path / "missing.json"),
                "--output",
                str(tmp_path / "run"),
                "--store",
                str(tmp_path / "store"),
            ]
        )
        == 2
    )
    assert "distributed-train" in capsys.readouterr().err


def test_zero3_parameter_metadata_is_not_an_empty_or_implicitly_gathered_tensor():
    engine = Trainer(nn.Linear(2, 3), zero_stage=3)
    unit = engine.model
    assert unit.weight.dtype == torch.float32 and unit.weight.shape == torch.Size((3, 2))
    assert unit.weight.device == torch.device("cpu") and unit.in_features == 2
    assert unit.gathers == 0 and unit.module.weight.numel() == 0
    with pytest.raises(RuntimeError, match="not a Tensor"):
        torch.nn.functional.linear(torch.ones(1, 2), unit.weight)
    with pytest.raises(RuntimeError, match="metadata"):
        unit.weight.data
    assert unit.gathers == 0


def test_shared_filesystem_probe_changes_nonce_without_changing_training_identity(tmp_path):
    from aster.core import read_json
    from aster.training import ParallelContext
    from aster.training.recipes import collective_run_directory

    context = ParallelContext()
    with collective_run_directory(context, tmp_path, "fixed-identity"):
        first = read_json(tmp_path / "distributed-input.json")
        assert (tmp_path / "run.lock").exists()
    with collective_run_directory(context, tmp_path, "fixed-identity"):
        second = read_json(tmp_path / "distributed-input.json")
    assert first["signature"] == second["signature"] == "fixed-identity"
    assert len(first["nonce"]) == 32 and first["nonce"] != second["nonce"]
    assert not (tmp_path / "run.lock").exists()
