from copy import deepcopy

import pytest
import torch

from aster.core import ArtifactStore, read_json
from aster.core.update_provenance import validate_successful_update_record
from aster.evaluation.edm_generation import publish_edm_generator
from aster.evaluation.interval_generation import publish_meanflow_generator
from aster.evaluation.generation_artifacts import verified_training_update
from aster.methods.generation import EDMObjective
from aster.methods.meanflow import MeanFlowObjective
from aster.models.generative import UNet2D, UNetConfig
from aster.models.interval_dit import IntervalDiT, IntervalDiTConfig
from aster.training import Trainer


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_pure_record_rejects_missing_stale_bool_and_config_mismatch_and_returns_copy():
    descriptor = {
        "class": "declared.Objective",
        "codec": "config_dict",
        "configuration": {"sigma": 0.6},
    }
    record = {
        "role": "model",
        "role_updates": 2,
        "phase": "override",
        "objective_configuration": descriptor,
    }
    result = validate_successful_update_record(record, descriptor, role_updates=2)
    result["objective_configuration"]["configuration"]["sigma"] = 0.9
    assert record["objective_configuration"]["configuration"]["sigma"] == 0.6
    for invalid in (
        None,
        {},
        {**record, "role_updates": True},
        {**record, "role_updates": 1},
        {**record, "objective_configuration": None},
        {**record, "role": "other"},
    ):
        with pytest.raises(ValueError):
            validate_successful_update_record(invalid, descriptor, role_updates=2)
    changed = deepcopy(descriptor)
    changed["configuration"]["sigma"] = 0.9
    with pytest.raises(ValueError, match="phase objective differs"):
        validate_successful_update_record(record, changed, role_updates=2)


def test_edm_actual_override_and_default_mutation_rejected_checkpoint_receipt_restored(tmp_path):
    torch.manual_seed(511)
    store = ArtifactStore(tmp_path / "store")
    model = UNet2D(
        UNetConfig(
            in_channels=3,
            model_channels=4,
            channel_mult=(1,),
            num_res_blocks=1,
            attention_levels=(),
            num_heads=1,
            prediction_type="edm_residual",
        )
    )
    engine = Trainer(model, EDMObjective(sigma_data=0.6), lr=0.003, ema_decay=0.8)
    batch = {
        "sample": torch.randn(2, 3, 4, 4),
        "sigma": torch.tensor([0.3, 0.8]),
        "noise": torch.randn(2, 3, 4, 4),
    }
    assert engine.step([batch]).updated
    engine.objective.sigma_data = 0.9
    with pytest.raises(RuntimeError, match="phase objective differs"):
        publish_edm_generator(engine, store, tmp_path / "changed-default")
    engine.objective.sigma_data = 0.6
    override = EDMObjective(sigma_data=0.8)
    assert engine.phase("edm_override", objective=override, microbatches=[batch]).updated
    assert (
        engine.last_successful_update()["objective_configuration"]["configuration"]["sigma_data"]
        == 0.8
    )
    with pytest.raises(RuntimeError, match="phase objective differs"):
        publish_edm_generator(engine, store, tmp_path / "incorrect-default")

    engine.objective = override
    artifact = publish_edm_generator(engine, store, tmp_path / "valid", ema=True)
    receipt = read_json(artifact.path / "successful_update.json")
    assert receipt["phase"] == "edm_override" and receipt["role_updates"] == 2
    assert (
        receipt
        == read_json(artifact.path / "generation_contract.json")["training"]["successful_update"]
    )
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    engine.phase("different_later", objective=EDMObjective(sigma_data=0.5), microbatches=[batch])
    engine.load_checkpoint(checkpoint, trusted=True)
    assert engine.last_successful_update() == receipt
    assert publish_edm_generator(engine, store, tmp_path / "restored", ema=True).id == artifact.id

    exposed = engine.last_successful_update()
    exposed["objective_configuration"]["configuration"]["sigma_data"] = 0.1
    assert verified_training_update(engine, override) == receipt


def test_meanflow_actual_phase_override_not_relabelled_as_default(tmp_path):
    torch.manual_seed(613)
    store = ArtifactStore(tmp_path / "store")
    config = IntervalDiTConfig(
        variant="meanflow",
        input_size=4,
        in_channels=3,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_classes=2,
    )
    objective = MeanFlowObjective(guidance=False, class_dropout=0.0)
    engine = Trainer(IntervalDiT(config), objective, lr=0.003)
    batch = {"sample": torch.randn(2, 3, 4, 4), "labels": torch.tensor([0, 1])}
    override = MeanFlowObjective(guidance=False, class_dropout=0.0, norm_power=0.5)
    assert engine.phase("meanflow_override", objective=override, microbatches=[batch]).updated
    with pytest.raises(RuntimeError, match="phase objective differs"):
        publish_meanflow_generator(engine, store, tmp_path / "wrong")
    engine.objective = override
    artifact = publish_meanflow_generator(engine, store, tmp_path / "valid")
    receipt = read_json(artifact.path / "generation_contract.json")["training"]["successful_update"]
    assert receipt == engine.last_successful_update() and receipt["phase"] == "meanflow_override"
