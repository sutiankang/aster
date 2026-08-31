from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from aster.core import LossTerm
from aster.training import Trainer
from aster.training.portable import optimizer_mapping


class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 3)
        self.head = nn.Linear(3, 5, bias=False)
        self.head.weight = self.embedding.weight
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0, 0.0],
                        [3.0, 4.0, 0.0],
                        [0.1, 0.2, 0.2],
                        [0.0, 0.0, 5.0],
                        [-4.0, 0.0, 0.0],
                    ]
                )
            )

    def forward(self, indices):
        return self.embedding(indices)


def objective(model, indices):
    values = model(indices).square()
    return LossTerm(values.sum(), torch.tensor(values.numel(), device=values.device), "elements")


def momentum_state(engine):
    optimizer, _, _ = optimizer_mapping(engine.roles["model"])
    return [
        deepcopy(
            optimizer._aster_state_loader(parameter)
            if hasattr(optimizer, "_aster_state_loader")
            else optimizer.state.get(parameter, {})
        )
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
@pytest.mark.parametrize("offload", ["none", "cpu", "nvme"])
def test_persistent_projection_updates_owner_not_moments_and_resumes(tmp_path, stage, offload):
    torch.set_num_threads(1)
    model = EmbeddingModel()
    reference = Trainer(deepcopy(model), objective, lr=0.02, max_grad_norm=None)
    engine = Trainer(
        model,
        objective,
        zero_stage=stage,
        offload_optimizer=offload,
        offload_directory=tmp_path / "disk" if offload == "nvme" else None,
        lr=0.02,
        max_grad_norm=None,
    )
    engine.register_embedding_projection("model", "embedding", max_norm=1.0)
    indices = torch.arange(5)
    reference.step([indices])
    engine.step([indices])
    before = engine.export_state_dict()["embedding.weight"]
    moments = momentum_state(engine)
    selected = torch.tensor([0, 2, 0])
    F.embedding(selected, reference.model.embedding.weight, max_norm=1.0)
    record = engine.project_embedding("model", "embedding", selected)
    assert record == {"rows": 2, "changed_rows": 1, "event": 1}
    actual = engine.export_state_dict()
    torch.testing.assert_close(
        actual["embedding.weight"], reference.model.embedding.weight, atol=2e-7, rtol=2e-6
    )
    torch.testing.assert_close(actual["head.weight"], actual["embedding.weight"], atol=0, rtol=0)
    torch.testing.assert_close(
        actual["embedding.weight"][[1, 2, 3, 4]], before[[1, 2, 3, 4]], atol=0, rtol=0
    )
    assert_tree_equal(momentum_state(engine), moments)

    reference.step([indices])
    engine.step([indices])
    torch.testing.assert_close(
        engine.export_state_dict()["embedding.weight"],
        reference.model.embedding.weight,
        atol=2e-7,
        rtol=2e-6,
    )
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    expected_event = engine.project_embedding("model", "embedding", torch.tensor([4]))
    engine.step([indices])
    expected = engine.export_state_dict()
    engine.load_checkpoint(checkpoint)
    assert engine.project_embedding("model", "embedding", torch.tensor([4])) == expected_event
    engine.step([indices])
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)


def test_projection_requires_explicit_policy_and_native_resume_preserves_policy(tmp_path):
    engine = Trainer(EmbeddingModel())
    with pytest.raises(ValueError, match="No embedding"):
        engine.project_embedding("model", "embedding", torch.tensor([0]))
    engine.register_embedding_projection("model", "embedding", max_norm=1.0)
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    changed = Trainer(EmbeddingModel())
    changed.register_embedding_projection("model", "embedding", max_norm=2.0)
    with pytest.raises(ValueError, match="checkpoint"):
        changed.load_checkpoint(checkpoint)
    before = engine.export_state_dict()
    for bad in (torch.tensor([-1]), torch.tensor([5]), torch.tensor([0.1])):
        with pytest.raises(ValueError, match="collectively"):
            engine.project_embedding("model", "embedding", bad)
    assert engine.project_embedding("model", "embedding", torch.empty(0, dtype=torch.long)) == {
        "rows": 0,
        "changed_rows": 0,
        "event": 1,
    }
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    automatic = Trainer(nn.Embedding(5, 3, max_norm=1.0))
    with pytest.raises(ValueError, match="max_norm=None"):
        automatic.register_embedding_projection("model", "", max_norm=1.0)


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_projection_preserves_unvisited_full_precision_master_values(stage):
    engine = Trainer(EmbeddingModel().to(torch.bfloat16), zero_stage=stage, offload_optimizer="cpu")
    engine.register_embedding_projection("model", "embedding", max_norm=1.0)
    _, owners, _ = optimizer_mapping(engine.roles["model"])
    owner = next(iter(owners.values()))
    assert owner.dtype == torch.float32

    with torch.no_grad():
        owner.reshape(5, 3)[4].add_(0.000123)
    untouched = owner.reshape(5, 3)[1:].clone()
    engine.project_embedding("model", "embedding", torch.tensor([0]))
    torch.testing.assert_close(owner.reshape(5, 3)[1:], untouched, atol=0, rtol=0)
    torch.testing.assert_close(
        owner.reshape(5, 3)[0],
        engine.export_state_dict()["embedding.weight"][0].float(),
        atol=0,
        rtol=0,
    )


def test_projection_frozen_embedding_inside_trainable_role_has_no_new_owner():
    model = nn.Sequential(nn.Embedding(3, 2), nn.Linear(2, 1))
    model[0].requires_grad_(False)
    with torch.no_grad():
        model[0].weight.fill_(2.0)

    engine = Trainer(model, offload_optimizer="cpu")
    count = len(engine.roles["model"].parameters)
    engine.register_embedding_projection("model", "0", max_norm=1.0)
    engine.project_embedding("model", "0", torch.tensor([0]))
    assert len(engine.roles["model"].parameters) == count
    assert engine.export_state_dict()["0.weight"][0].norm() <= 1.0


def test_projection_frozen_role_has_no_optimizer_under_zero3():
    engine = Trainer(nn.Linear(2, 1), zero_stage=3)
    embedding = nn.Embedding(3, 2)
    with torch.no_grad():
        embedding.weight.fill_(2.0)
    engine.add_role("frozen", embedding, trainable=False)
    engine.register_embedding_projection("frozen", "", max_norm=1.0)
    engine.project_embedding("frozen", "", torch.tensor([0]))
    assert engine.roles["frozen"].optimizer is None
    assert engine.export_state_dict(role="frozen")["weight"][0].norm() <= 1.0


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_runtime_ownership_marker_is_not_exported_but_survives_deepcopy(stage):
    engine = Trainer(EmbeddingModel(), zero_stage=stage, offload_optimizer="cpu")
    assert all(getattr(module, "_aster_training_owned", False) for module in engine.model.modules())
    copied = deepcopy(engine.model)
    assert all(getattr(module, "_aster_training_owned", False) for module in copied.modules())
    state = engine.export_state_dict()
    assert not any("_aster_training_owned" in name for name in state)
    deployed = EmbeddingModel()
    deployed.load_state_dict(state, strict=True)
    assert all(not getattr(module, "_aster_training_owned", False) for module in deployed.modules())
    frozen = engine.add_role("teacher", nn.Embedding(5, 3), trainable=False)
    assert frozen._aster_training_owned
