from copy import deepcopy
import pytest
import torch
from torch import nn
from aster.core import LossTerm
from aster.training import Trainer
from aster.training.portable import optimizer_mapping


def _objective(model, batch):
    x, target = batch
    values = (model(x) - target).square()
    return LossTerm(values.sum(), torch.tensor(values.numel(), dtype=torch.int64), "element")


def _factory(parameters):
    params = list(parameters)
    return torch.optim.RAdam(
        [
            {"params": params[::2], "lr": 0.013, "weight_decay": 0.12},
            {"params": params[1::2], "lr": 0.007, "eps": 0.0002},
        ],
        betas=(0.7, 0.91),
        decoupled_weight_decay=False,
    )


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
@pytest.mark.parametrize("offload", ["none", "cpu", "nvme"])
def test_native_radam_shards_offload_rectification_and_resume(stage, offload, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(641)
    make = lambda: nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    model = make()
    oracle = deepcopy(model)
    optimizer = _factory(oracle.parameters())
    engine = Trainer(
        model,
        _objective,
        optimizer_factory=_factory,
        zero_stage=stage,
        max_grad_norm=None,
        offload_optimizer=offload,
        offload_directory=tmp_path / "offload" if offload == "nvme" else None,
    )
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        _objective(oracle, batch).mean.backward()
        optimizer.step()
        engine.step([batch])
        for key, value in engine.export_state_dict().items():
            torch.testing.assert_close(value, oracle.state_dict()[key], rtol=3e-6, atol=2e-7)
    assert type(optimizer_mapping(engine.roles["model"])[0]) is torch.optim.RAdam
    path = engine.save_checkpoint(tmp_path / "native")
    engine.step([batch])
    expected = engine.export_state_dict()
    engine.load_checkpoint(path)
    engine.step([batch])
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    portable = engine.save_portable_checkpoint(tmp_path / "portable")
    restored = Trainer(make(), _objective, optimizer_factory=_factory, max_grad_norm=None)
    restored.load_portable_checkpoint(portable, seed=57)
    restored.step([batch])
    engine.step([batch])
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, restored.export_state_dict()[key], rtol=3e-6, atol=2e-7)
