from copy import deepcopy

import pytest
import torch
from torch import nn

from aster.core import LossTerm
from aster.training import Trainer
from aster.training.portable import optimizer_mapping


def _model():
    return nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))


def _objective(model, batch):
    x, target = batch
    loss = (model(x) - target).square()
    return LossTerm(loss.sum(), torch.tensor(loss.numel()), "elements")


def _factory(parameters, *, algorithm=torch.optim.Adam, reverse=False, eps=1e-5):
    parameters = list(parameters)
    groups = [parameters[::2], parameters[1::2]]
    if reverse:
        groups.reverse()
    return algorithm(
        [
            {"params": groups[0], "lr": 0.02, "eps": eps, "weight_decay": 0.15},
            {"params": groups[1], "lr": 0.007, "eps": 1e-3, "weight_decay": 0.3},
        ],
        betas=(0.7, 0.91),
        amsgrad=True,
    )


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
@pytest.mark.parametrize("offload", ["none", "cpu", "nvme"])
def test_adam_factory_groups_owner_updates_resume_and_portable(tmp_path, stage, offload):
    torch.set_num_threads(1)
    torch.manual_seed(623)
    model = _model()
    dense = deepcopy(model)
    reference = _factory(dense.parameters())
    calls = []

    def factory(parameters):
        calls.append(tuple(parameters))
        return _factory(parameters)

    engine = Trainer(
        model,
        _objective,
        optimizer_factory=factory,
        zero_stage=stage,
        max_grad_norm=None,
        offload_optimizer=offload,
        offload_directory=tmp_path / "disk" if offload == "nvme" else None,
    )
    assert len(calls) == 1
    assert {id(p) for p in calls[0]} == {id(p) for p in engine.roles["model"].parameters}
    optimizer, _, _ = optimizer_mapping(engine.roles["model"])
    assert type(optimizer) is torch.optim.Adam
    assert [(g["lr"], g["eps"], g["weight_decay"]) for g in optimizer.param_groups] == [
        (0.02, 1e-5, 0.15),
        (0.007, 1e-3, 0.3),
    ]
    generator = torch.Generator().manual_seed(643)
    batch = (torch.randn(4, 2, generator=generator), torch.randn(4, 1, generator=generator))
    for _ in range(3):
        reference.zero_grad(set_to_none=True)
        _objective(dense, batch).mean.backward()
        reference.step()
        engine.step([batch])
        for name, value in engine.export_state_dict().items():
            torch.testing.assert_close(value, dense.state_dict()[name], atol=2e-7, rtol=2e-6)
    checkpoint = engine.save_checkpoint(tmp_path / "native")
    engine.step([batch])
    expected = engine.export_state_dict()
    engine.load_checkpoint(checkpoint)
    engine.step([batch])
    for name, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    portable = engine.save_portable_checkpoint(tmp_path / "portable")
    restored = Trainer(_model(), _objective, optimizer_factory=_factory, max_grad_norm=None)
    restored.load_portable_checkpoint(portable, seed=55)
    engine.step([batch])
    restored.step([batch])
    for name, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, restored.export_state_dict()[name], atol=2e-7, rtol=2e-6)


def test_coupled_l2_is_not_decoupled_adamw():

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    def zero(model, x):
        return LossTerm((model(x) * 0).sum(), torch.tensor(1), "sample")

    adam = Trainer(
        deepcopy(model),
        zero,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.1, weight_decay=0.2),
        max_grad_norm=None,
    )
    adamw = Trainer(
        deepcopy(model),
        zero,
        optimizer_factory=lambda p: torch.optim.AdamW(p, lr=0.1, weight_decay=0.2),
        max_grad_norm=None,
    )
    adam.step([torch.ones(1, 1)])
    adamw.step([torch.ones(1, 1)])
    assert abs(adam.model.weight.item() - 1.9) < 2e-7
    assert abs(adamw.model.weight.item() - 1.96) < 2e-7
    a, owners, _ = optimizer_mapping(adam.roles["model"])
    b, w_owners, _ = optimizer_mapping(adamw.roles["model"])
    assert a.state[next(iter(owners.values()))]["exp_avg"].abs().sum() > 0
    assert b.state[next(iter(w_owners.values()))]["exp_avg"].abs().sum() == 0


def test_factory_contract_and_parameter_group_identity(tmp_path):
    model = _model()
    with pytest.raises(ValueError, match="mutually exclusive"):
        Trainer(
            model,
            optimizer=torch.optim.Adam(model.parameters()),
            optimizer_factory=_factory,
            zero_stage=3,
        )
    assert model[0].weight.numel() == 6
    with pytest.raises(ValueError, match="callable"):
        Trainer(model, optimizer_factory=3, zero_stage=3)
    with pytest.raises(ValueError, match="native"):
        Trainer(_model(), optimizer_factory=lambda p: object())
    with pytest.raises(ValueError, match="全部可训练参数"):
        Trainer(_model(), optimizer_factory=lambda p: torch.optim.Adam(p[:-1]))
    with pytest.raises(ValueError, match="trainable role"):
        Trainer(_model()).add_role("teacher", _model(), trainable=False, optimizer_factory=_factory)
    first = Trainer(_model(), optimizer_factory=_factory, zero_stage=3)
    checkpoint = first.save_checkpoint(tmp_path / "native")
    for changed in (
        lambda p: _factory(p, reverse=True),
        lambda p: _factory(p, eps=1e-4),
        lambda p: _factory(p, algorithm=torch.optim.AdamW),
    ):
        other = Trainer(_model(), optimizer_factory=changed, zero_stage=3)
        with pytest.raises(ValueError, match="checkpoint"):
            other.load_checkpoint(checkpoint)
    role = first.add_role("policy", _model(), optimizer_factory=_factory)
    assert type(optimizer_mapping(first.roles["policy"])[0]) is torch.optim.Adam
    assert first.roles["policy"].model is role
