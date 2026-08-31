from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from aster.core.contracts import LossBundle, LossTerm
from aster.training import Trainer
from aster.training.sharding import Zero3Unit


def mse(model, batch):
    x, y = batch
    error = (model(x) - y).square()
    return LossTerm(error.sum(), torch.tensor(error.numel()), "elements")


def bundle(model, batch):
    x, y, mask = batch
    out = model(x)
    return LossBundle(
        (
            LossTerm(((out - y).square() * mask).sum(), mask.sum().detach(), "tokens", "mse", 0.7),
            LossTerm(out.abs().sum(), torch.tensor(out.shape[0]), "samples", "l1", 0.3),
        )
    )


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def test_multi_objective_independent_denominators_and_accumulation():
    torch.manual_seed(1)
    model = nn.Linear(2, 1)
    reference = deepcopy(model)
    trainer = Trainer(
        model,
        bundle,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.2),
        accumulation_steps=2,
        max_grad_norm=None,
    )
    batches = [
        (torch.randn(2, 2), torch.randn(2, 1), torch.ones(2, 1)),
        (torch.randn(5, 2), torch.randn(5, 1), torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0]])),
    ]
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.2)
    terms = [bundle(reference, batch).terms for batch in batches]
    loss = (
        0.7 * sum(t[0].numerator for t in terms) / 4 + 0.3 * sum(t[1].numerator for t in terms) / 7
    )
    loss.backward()
    optimizer.step()
    result = trainer.step(batches)
    assert (
        result.updated
        and result.terms["mse"]["denominator"] == 4
        and result.terms["l1"]["denominator"] == 7
    )
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b)


def test_zero_count_does_not_decay_or_step():
    model = nn.Linear(2, 1)
    before = deepcopy(model.state_dict())
    trainer = Trainer(model, lambda m, x: LossTerm(m(x).sum() * 0, torch.tensor(0), "tokens"))
    result = trainer.step([torch.randn(3, 2)])
    assert not result.updated and trainer.steps == 0
    assert_equal(before, model.state_dict())


def test_unused_parameter_not_weight_decayed():
    class WithUnused(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)
            self.unused = nn.Parameter(torch.ones(3))

        def forward(self, x):
            return self.linear(x)

    model = WithUnused()
    trainer = Trainer(
        model, mse, optimizer=torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.5)
    )
    trainer.step([(torch.ones(2, 1), torch.zeros(2, 1))])
    torch.testing.assert_close(model.unused, torch.ones(3), rtol=0, atol=0)
    assert model.unused.grad is None


def test_roles_ownership_freeze_preserves_input_gradient_and_restores_flags():
    actor = nn.Linear(1, 1, bias=False)
    critic = nn.Linear(1, 1, bias=False)
    trainer = Trainer(actor, lr=0.1, max_grad_norm=None)
    trainer.add_role("critic", critic)
    with pytest.raises(ValueError, match="tensor"):
        trainer.add_role("alias", actor)
    before_actor, before_critic = actor.weight.detach().clone(), critic.weight.detach().clone()

    def objective(model, x):
        value = critic(model(x))
        return LossTerm(-value.sum(), torch.tensor(value.numel()), "actions")

    trainer.phase(
        "actor", microbatches=[torch.ones(3, 1)], objective=objective, freeze_roles=("critic",)
    )
    assert not torch.equal(before_actor, actor.weight)
    assert torch.equal(before_critic, critic.weight) and critic.weight.requires_grad


def test_failure_restores_freeze_flags_and_requires_checkpoint():
    actor, critic = nn.Linear(1, 1), nn.Linear(1, 1)
    trainer = Trainer(actor)
    trainer.add_role("critic", critic)
    with pytest.raises(ValueError):
        trainer.phase(
            "bad", objective=lambda m, b: 1, microbatches=[None], freeze_roles=("critic",)
        )
    assert all(p.requires_grad for p in critic.parameters())
    with pytest.raises(RuntimeError):
        trainer.step([None])


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_zero_stages_match_unsharded_single_rank(stage):
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    original = deepcopy(model)
    reference = Trainer(original, mse, lr=0.02, max_grad_norm=0.6)
    trainer = Trainer(model, mse, lr=0.02, max_grad_norm=0.6, zero_stage=stage)
    batch = (torch.randn(7, 3), torch.randn(7, 2))
    for _ in range(3):
        expected = reference.step([batch])
        actual = trainer.step([batch])
        assert actual.loss == pytest.approx(expected.loss, abs=1e-6)
        assert actual.grad_norm == pytest.approx(expected.grad_norm, abs=1e-6)
        torch.testing.assert_close(
            trainer.model(batch[0]), reference.model(batch[0]), atol=1e-6, rtol=1e-5
        )
    if stage == 3:
        units = [unit for unit in trainer.model.modules() if isinstance(unit, Zero3Unit)]
        assert units and all(unit.gathers >= 6 and unit.releases >= 6 for unit in units)
        assert all(
            parameter.numel() == 0 for unit in units for parameter in unit.module.parameters()
        )


def test_complete_checkpoint_rng_ema_and_next_update(tmp_path):
    torch.manual_seed(22)
    model = nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.3), nn.Linear(4, 1))
    trainer = Trainer(model, mse, lr=0.01, ema_decay=0.8)
    batch = (torch.ones(3, 2), torch.zeros(3, 1))
    trainer.step([batch])
    checkpoint = trainer.save_checkpoint(tmp_path / "resume.json")
    expected_random = (random.random(), float(np.random.rand()), torch.rand(2))
    expected_result = trainer.step([batch])
    expected = deepcopy(model.state_dict())
    expected_ema = deepcopy(trainer.roles["model"].ema.state_dict())
    restored = Trainer(
        nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.3), nn.Linear(4, 1)),
        mse,
        lr=0.01,
        ema_decay=0.8,
    )
    restored.load_checkpoint(checkpoint)
    actual_random = (random.random(), float(np.random.rand()), torch.rand(2))
    assert_equal(expected_random, actual_random)
    assert restored.step([batch]) == expected_result
    assert_equal(expected, restored.model.state_dict())
    assert_equal(expected_ema, restored.roles["model"].ema.state_dict())


def test_checkpoint_hash_validation_and_configuration(tmp_path):
    trainer = Trainer(nn.Linear(1, 1), mse)
    path = trainer.save_checkpoint(tmp_path / "checkpoint.json")
    with pytest.raises(ValueError, match="配置"):
        Trainer(nn.Linear(1, 1), mse, accumulation_steps=2).load_checkpoint(path)
    entry = json.loads(path.read_text())["entries"][0]
    payload = path.parent / entry["file"]
    with payload.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="完整性"):
        trainer.load_checkpoint(path)


def test_precision_and_invalid_objectives_fail_fast():
    with pytest.raises(ValueError):
        Trainer(nn.Linear(1, 1), mse, precision="fp8")
    trainer = Trainer(nn.Linear(1, 1), mse, precision="bf16")
    assert trainer.step([(torch.ones(2, 1), torch.zeros(2, 1))]).updated
    with pytest.raises(ValueError):
        trainer.step([])


def test_nonfinite_loss_skips_optimizer_ema_scheduler():
    model = nn.Linear(1, 1)
    before = deepcopy(model.state_dict())
    trainer = Trainer(
        model,
        lambda m, x: LossTerm(m(x).sum() * float("nan"), torch.tensor(1), "sample"),
        ema_decay=0.9,
    )
    result = trainer.step([torch.ones(1, 1)])
    assert result.overflow and not result.updated and trainer.roles["model"].ema.updates == 0
    assert_equal(before, model.state_dict())


@pytest.mark.parametrize("heads", [1, 2])
def test_ring_single_group_attention_gradient_matches_sdpa(heads):
    from aster.training import ring_context_parallel_attention, Group

    torch.manual_seed(8)
    q = torch.randn(2, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, heads, 4, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, heads, 4, 3, dtype=torch.float64, requires_grad=True)
    actual = ring_context_parallel_attention(q, k, v, Group())
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2 // heads, 1), v.repeat_interleave(2 // heads, 1), is_causal=True
    )
    gradient = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), gradient, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (q, k, v), gradient)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_cpu_optimizer_offload_update_and_resume(tmp_path, stage):
    torch.manual_seed(9)
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    reference = Trainer(deepcopy(model), mse, lr=0.02, max_grad_norm=0.3)
    trainer = Trainer(
        model, mse, lr=0.02, zero_stage=stage, offload_optimizer="cpu", max_grad_norm=0.3
    )
    trainer.set_scheduler(
        lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    )
    reference.set_scheduler(
        lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    )
    batch = (torch.ones(3, 2), torch.zeros(3, 1))
    for _ in range(3):
        reference.step([batch])
        trainer.step([batch])
        torch.testing.assert_close(trainer.model(batch[0]), reference.model(batch[0]))
    path = trainer.save_checkpoint(tmp_path / f"cpu-{stage}.json")
    trainer.step([batch])
    expected = trainer.model(batch[0]).detach().clone()
    trainer.load_checkpoint(path)
    trainer.step([batch])
    torch.testing.assert_close(trainer.model(batch[0]), expected, atol=0, rtol=0)
    optimizer = trainer.roles["model"].optimizer
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    assert all(
        value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def test_portable_zero3_to_unsharded_preserves_optimizer_ema(tmp_path):
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    source = Trainer(model, mse, lr=0.01, zero_stage=3, ema_decay=0.8, offload_optimizer="cpu")
    batch = (torch.ones(3, 2), torch.zeros(3, 1))
    source.step([batch])
    source.step([batch])
    path = source.save_portable_checkpoint(tmp_path / "portable.json")
    target = Trainer(
        nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)), mse, lr=0.4, ema_decay=0.8
    )
    target.load_portable_checkpoint(path, seed=301)
    for a, b in zip(source.export_state_dict().values(), target.export_state_dict().values()):
        torch.testing.assert_close(a, b)
    source.step([batch])
    target.step([batch])
    for name, value in source.export_state_dict().items():
        torch.testing.assert_close(value, target.export_state_dict()[name], atol=1e-7, rtol=1e-6)
    for name, value in source.export_state_dict(ema=True).items():
        torch.testing.assert_close(
            value, target.export_state_dict(ema=True)[name], atol=1e-7, rtol=1e-6
        )
    assert not target.migration_record["exact_rank_rng_or_data_resume"]


def test_portable_refuses_silent_data_state_loss(tmp_path):
    class Stateful:
        def state_dict(self):
            return {"cursor": 3}

        def load_state_dict(self, state):
            pass

    trainer = Trainer(nn.Linear(1, 1), mse)
    trainer.register_state("data", Stateful())
    with pytest.raises(ValueError, match="静默丢弃"):
        trainer.save_portable_checkpoint(tmp_path / "portable.json")


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_target_snapshot_polyak_and_checkpoint(stage, tmp_path):
    factory = lambda: nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    torch.manual_seed(731)
    trainer = Trainer(factory(), mse, zero_stage=stage)
    target = trainer.clone_target("model", "target", factory=factory)
    before = deepcopy(target.state_dict())
    assert not any(p.requires_grad for p in target.parameters())
    assert trainer.roles["target"].optimizer is None
    trainer.step([(torch.ones(4, 2), torch.zeros(4, 1))])
    source = trainer.export_state_dict()
    trainer.update_target("model", "target", 0.9)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, before[key] * 0.9 + source[key] * 0.1)
    path = trainer.save_checkpoint(tmp_path / f"target-{stage}.json")
    expected = deepcopy(target.state_dict())
    trainer.update_target("model", "target", 0.0)
    trainer.load_checkpoint(path)
    assert_equal(expected, target.state_dict())


def test_target_buffer_policy_and_invalid_owner():
    factory = lambda: nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    trainer = Trainer(factory(), mse)
    target = trainer.clone_target("model", "target", factory=factory)
    with torch.no_grad():
        trainer.model[1].running_mean.fill_(3.0)
        trainer.model[1].num_batches_tracked.fill_(7)
    trainer.update_target("model", "target", 0.5, buffers="ema")
    torch.testing.assert_close(target[1].running_mean, torch.full((2,), 1.5))
    assert target[1].num_batches_tracked.item() == 7
    trainer.update_target("model", "target", 1.0, buffers="copy")
    torch.testing.assert_close(target[1].running_mean, torch.full((2,), 3.0))
    with pytest.raises(ValueError, match="冻结"):
        trainer.update_target("target", "model", 0.5)


@pytest.mark.parametrize("adjust_lr", ["original", "match_rms_adamw", "none"])
def test_native_muon_matches_torch_official(adjust_lr, tmp_path):
    from aster.training import Muon

    torch.manual_seed(729)
    native_model = nn.Linear(3, 5, bias=False)
    reference = deepcopy(native_model)
    native = Muon(native_model.parameters(), lr=0.03, weight_decay=0.1, adjust_lr=adjust_lr)
    official = torch.optim.Muon(
        reference.parameters(),
        lr=0.03,
        weight_decay=0.1,
        adjust_lr_fn=adjust_lr if adjust_lr != "none" else None,
    )

    if adjust_lr == "none":
        official.param_groups[0]["adjust_lr_fn"] = "no_scaling"
    trainer = Trainer(native_model, mse, optimizer=native)
    for _ in range(3):
        batch = (torch.randn(4, 3), torch.randn(4, 5))
        official.zero_grad()
        mse(reference, batch).mean.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        official.step()
        trainer.step([batch])
        torch.testing.assert_close(native_model.weight, reference.weight, rtol=1e-6, atol=1e-7)
    path = trainer.save_checkpoint(tmp_path / "muon.json")
    trainer.step([batch])
    expected = native_model.weight.detach().clone()
    trainer.load_checkpoint(path)
    trainer.step([batch])
    torch.testing.assert_close(native_model.weight, expected, rtol=0, atol=0)


def test_activation_recomputation_rng_and_multiple_objectives():
    from aster.training import checkpoint_activation

    class Block(nn.Module):
        def __init__(self, recompute):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.25), nn.Linear(3, 1))
            self.recompute = recompute
            self.calls = 0

        def function(self, x):
            self.calls += 1
            return self.net(x)

        def forward(self, x):
            return checkpoint_activation(self.function, x) if self.recompute else self.function(x)

    torch.manual_seed(317)
    original = Block(False)
    recomputed = Block(True)
    recomputed.load_state_dict(original.state_dict())
    reference = Trainer(original, bundle)
    native = Trainer(recomputed, bundle, activation_offload="cpu")
    batch = (torch.ones(4, 2), torch.zeros(4, 1), torch.ones(4, 1))
    rng = torch.random.get_rng_state()
    reference.step([batch])
    expected_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    native.step([batch])
    assert recomputed.calls > original.calls
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, recomputed.state_dict()[key])


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_disk_optimizer_real_eviction_exact_resume_and_portable(stage, tmp_path):
    from aster.training.offload import DiskOptimizer

    torch.manual_seed(924)
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    reference = Trainer(deepcopy(model), mse, lr=0.01)
    trainer = Trainer(
        model,
        mse,
        zero_stage=stage,
        lr=0.01,
        offload_optimizer="nvme",
        offload_directory=tmp_path / "offload",
    )
    batch = (torch.ones(4, 2), torch.zeros(4, 1))
    for _ in range(3):
        trainer.step([batch])
        reference.step([batch])
        torch.testing.assert_close(trainer.model(batch[0]), reference.model(batch[0]))
    wrapper = trainer.roles["model"].optimizer
    if not isinstance(wrapper, DiskOptimizer):
        wrapper = wrapper.optimizer
    assert wrapper.records and not wrapper.optimizer.state
    assert all((wrapper.directory / entry["file"]).is_file() for entry in wrapper.records.values())
    assert wrapper.peak_resident_state_elements <= 2 * max(p.numel() for p in wrapper.masters) + 1
    path = trainer.save_checkpoint(tmp_path / "disk.json")
    trainer.step([batch])
    expected = trainer.model(batch[0]).detach().clone()
    trainer.load_checkpoint(path)
    trainer.step([batch])
    torch.testing.assert_close(trainer.model(batch[0]), expected, rtol=0, atol=0)
    portable = trainer.save_portable_checkpoint(tmp_path / "portable.json")
    reference.load_portable_checkpoint(portable, seed=5)
    reference.step([batch])
    trainer.step([batch])
    torch.testing.assert_close(trainer.model(batch[0]), reference.model(batch[0]))
