from copy import deepcopy

import pytest
import torch
from torch import nn

from aster.core import LossTerm
from aster.training import Trainer


def objective(model, batch):
    return LossTerm(model(batch).sum(), torch.tensor(len(batch), dtype=torch.int64), "sample")


@pytest.mark.parametrize("zero", [0, 1, 2, 3])
@pytest.mark.parametrize("offload", ["none", "cpu", "nvme"])
def test_value_clip_adam_ownership_resume_and_portable(zero, offload, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(537)
    model = nn.Linear(2, 2)
    reference = deepcopy(model)
    factory = lambda params: torch.optim.Adam(params, lr=0.01, weight_decay=0.02)
    kwargs = {
        "zero_stage": zero,
        "offload_optimizer": offload,
        "optimizer_factory": factory,
        "accumulation_steps": 2,
        "max_grad_norm": None,
        "max_grad_value": 0.5,
    }
    if offload == "nvme":
        kwargs["offload_directory"] = tmp_path / "disk-state"
    engine = Trainer(model, objective, **kwargs)
    optimizer = factory(reference.parameters())
    batches = [torch.tensor([[10.0, 1.0]]), torch.tensor([[-3.0, 2.0], [-3.0, 2.0], [-3.0, 2.0]])]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        objective(reference, torch.cat(batches)).mean.backward()
        torch.nn.utils.clip_grad_value_(reference.parameters(), 0.5)
        optimizer.step()
        assert engine.step(batches).updated
        for key, value in engine.export_state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key], atol=2e-7, rtol=2e-6)
    native = engine.save_checkpoint(tmp_path / "native.json")
    portable = engine.save_portable_checkpoint(tmp_path / "portable.json")
    first = engine.step(batches)
    expected = engine.export_state_dict()
    engine.load_checkpoint(native)
    assert engine.step(batches) == first
    for key, value in engine.export_state_dict().items():
        assert torch.equal(value, expected[key])
    dense = Trainer(
        nn.Linear(2, 2),
        objective,
        optimizer_factory=factory,
        accumulation_steps=2,
        max_grad_norm=None,
        max_grad_value=0.5,
    )
    dense.load_portable_checkpoint(portable, seed=49)
    dense.step(batches)
    for key, value in dense.export_state_dict().items():
        assert torch.equal(value, expected[key])
    engine.max_grad_value = 0.25
    with pytest.raises(ValueError, match="checkpoint"):
        engine.load_checkpoint(native)
    with pytest.raises(ValueError, match="迁移"):
        engine.load_portable_checkpoint(portable, seed=49)


def test_norm_then_value_clipping_has_explicit_order():
    torch.set_num_threads(1)
    torch.manual_seed(829)
    model = nn.Linear(2, 1)
    reference = deepcopy(model)
    engine = Trainer(
        model,
        objective,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.1),
        max_grad_norm=2.0,
        max_grad_value=0.4,
    )
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    batch = torch.tensor([[3.0, 4.0]])
    objective(reference, batch).mean.backward()
    norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 2.0)
    torch.nn.utils.clip_grad_value_(reference.parameters(), 0.4)
    optimizer.step()
    result = engine.step([batch])
    assert result.grad_norm == pytest.approx(float(norm))
    for a, b in zip(engine.model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("value", [0.0, -1.0, True, float("nan"), float("inf"), "1"])
def test_invalid_value_clip_is_rejected(value):
    with pytest.raises(ValueError, match="max_grad_value"):
        Trainer(nn.Linear(2, 1), objective, max_grad_value=value)
