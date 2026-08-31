from copy import deepcopy
from datetime import timedelta
import math
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.training import ParallelConfig, ParallelContext, Trainer
from aster.training.sharding import zero3_units
from aster.methods.offline import (
    ContinuousTwinQ,
    DeterministicActor,
    IQLActor,
    IQLMethod,
    StateValue,
    TD3Method,
)


def _adam(parameters):
    return torch.optim.Adam(parameters, lr=0.001, betas=(0.8, 0.97), eps=1e-6)


def _data():
    generator = torch.Generator().manual_seed(719)
    return {
        "observations": torch.randn(6, 3, generator=generator),
        "actions": torch.randn(6, 2, generator=generator).tanh(),
        "rewards": torch.randn(6, generator=generator),
        "next_observations": torch.randn(6, 3, generator=generator),
        "terminated": torch.tensor([False, True, False, False, False, True]),
        "truncated": torch.tensor([True, False, False, False, True, False]),
    }


def _split(data, rank, *, empty_rank=False):

    ranges = ((0, 1), (1, 2)) if rank == 0 else ((2, 3), (3, 6))
    if empty_rank:
        ranges = ((0, 0), (0, 0)) if rank == 0 else ((0, 2), (2, 6))
    return [{key: value[start:end] for key, value in data.items()} for start, end in ranges]


def _build(kind, stage, context):
    torch.manual_seed(483)
    actor = IQLActor(3, 2, 8) if kind == "iql" else DeterministicActor(3, 2, 8)
    critic, value = ContinuousTwinQ(3, 2, 8), StateValue(3, 8)
    reference = {
        "model": deepcopy(actor),
        "critic": deepcopy(critic),
        "target_critic": deepcopy(critic).requires_grad_(False),
    }
    if kind == "iql":
        reference["value"] = deepcopy(value)
    else:
        reference["target_actor"] = deepcopy(actor).requires_grad_(False)
    optimizers = {
        name: _adam(reference[name].parameters())
        for name in ("model", "critic", *(["value"] if kind == "iql" else []))
    }
    engine = Trainer(
        actor,
        parallel=context,
        zero_stage=stage,
        optimizer_factory=_adam,
        accumulation_steps=2,
        max_grad_norm=None,
        lr=0.001,
    )
    if kind == "iql":
        method = IQLMethod(
            engine, critic, value, critic_optimizer_factory=_adam, value_optimizer_factory=_adam
        )
    else:
        method = TD3Method(
            engine,
            critic,
            policy_noise=0.0,
            bc_alpha=2.5 if kind == "td3_bc" else None,
            critic_optimizer_factory=_adam,
        )
    return engine, method, reference, optimizers


def _dense_update(models, optimizers, kind, batch, step):

    obs, action, reward, following, done = (
        batch[key]
        for key in ("observations", "actions", "rewards", "next_observations", "terminated")
    )
    actor, critic, target_q = (models[key] for key in ("model", "critic", "target_critic"))
    losses = {}

    def update(role, loss):
        optimizers[role].zero_grad(set_to_none=True)
        loss.backward()
        optimizers[role].step()
        return float(loss.detach())

    def move_target(source, target):
        with torch.no_grad():
            for parameter, target_parameter in zip(
                models[source].parameters(), models[target].parameters()
            ):
                target_parameter.lerp_(parameter, 0.005)

    if kind == "iql":
        value = models["value"]
        with torch.no_grad():
            q = torch.minimum(*target_q(obs, action))
        difference = q - value(obs)
        losses["value"] = update(
            "value", (torch.where(difference > 0, 0.8, 0.2) * difference.square()).mean()
        )
        with torch.no_grad():
            weight = torch.exp(3.0 * (q - value(obs))).clamp_max(100.0)

        mean = actor.network(obs).tanh()
        log_std = actor.log_stds.clamp(-5.0, 2.0)
        log_probability = (
            -0.5 * ((action - mean) / log_std.exp()).square()
            - log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(-1)
        losses["actor"] = update("model", -(weight * log_probability).mean())
        with torch.no_grad():
            target = reward + 0.99 * (~done) * value(following)
    else:
        with torch.no_grad():
            following_action = models["target_actor"](following).clamp(-1, 1)
            target = reward + 0.99 * (~done) * torch.minimum(*target_q(following, following_action))
    q1, q2 = critic(obs, action)
    losses["critic"] = update("critic", ((q1 - target).square() + (q2 - target).square()).mean())
    if kind == "iql":
        move_target("critic", "target_critic")
    else:
        losses["actor"] = None
        if step % 2 == 0:
            critic.requires_grad_(False)
            try:
                predicted = actor(obs)
                q1, _ = critic(obs, predicted)
                objective = -q1.mean()
                if kind == "td3_bc":
                    objective = (
                        -2.5 / q1.detach().abs().mean() * q1.mean()
                        + (predicted - action).square().mean()
                    )
                losses["actor"] = update("model", objective)
            finally:
                critic.requires_grad_(True)
            move_target("model", "target_actor")
            move_target("critic", "target_critic")
    return losses


def _export(engine):
    return {
        name: engine.export_state_dict(role=name, only_rank_zero=False) for name in engine.roles
    }


def _assert_models(actual, expected, *, exact=False):
    for role, weights in actual.items():
        target = (
            expected[role].state_dict()
            if isinstance(expected[role], torch.nn.Module)
            else expected[role]
        )
        assert weights.keys() == target.keys()
        for name, tensor in weights.items():
            torch.testing.assert_close(
                tensor, target[name], atol=0 if exact else 5e-7, rtol=0 if exact else 3e-5
            )


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        context, data = ParallelContext(), _data()
        for kind in ("td3", "td3_bc", "iql"):
            for stage in range(4):
                engine, method, reference, optimizers = _build(kind, stage, context)
                batches = _split(data, rank)
                result = method.update(batches)
                expected = _dense_update(reference, optimizers, kind, data, 1)
                _assert_models(_export(engine), reference)
                checkpoint = engine.save_checkpoint(Path(output) / f"{kind}_{stage}")
                result = method.update(batches)
                expected = _dense_update(reference, optimizers, kind, data, 2)
                _assert_models(_export(engine), reference)
                for name, value in expected.items():
                    assert abs(result[name].loss - value) < 2e-6
                after, rng_next = _export(engine), torch.rand(5)
                engine.load_checkpoint(checkpoint)
                repeated = method.update(batches)
                _assert_models(_export(engine), after, exact=True)
                torch.testing.assert_close(torch.rand(5), rng_next, atol=0, rtol=0)
                assert method.updates == 2
                for name in result:
                    assert result[name].loss == repeated[name].loss
                if stage == 3:
                    units = zero3_units(engine.model)
                    assert units and all(unit.gathers > 0 and unit.releases > 0 for unit in units)
                    assert all(
                        parameter.numel() == 0
                        for unit in units
                        for parameter in unit.module.parameters()
                    )
                    assert sum(parameter.numel() for parameter in engine.model.parameters()) < sum(
                        parameter.numel() for parameter in reference["model"].parameters()
                    )

                malformed = deepcopy(batches)
                if rank == 1:
                    malformed[0]["next_observations"][0, 0] = float("nan")
                before_rng = torch.get_rng_state().clone()
                with pytest.raises(ValueError, match="collective preflight"):
                    method.update(malformed)
                _assert_models(_export(engine), after, exact=True)
                torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)
                with pytest.raises(ValueError, match="collective preflight"):
                    method.update(batches[:1] if rank == 1 else batches)
                assert not method._incomplete and method.updates == 2

                if stage == 3:
                    for step in (3, 4):
                        method.update(_split(data, rank, empty_rank=True))
                        _dense_update(reference, optimizers, kind, data, step)
                        _assert_models(_export(engine), reference)
                if kind == "iql" and stage == 3:
                    complete = engine.save_checkpoint(Path(output) / "before_failure")
                    original = engine.model.log_prob
                    if rank == 1:
                        engine.model.log_prob = lambda observations, actions: (
                            original(observations, actions) * float("nan")
                        )
                    with pytest.raises(RuntimeError, match="phase skipped"):
                        method.update(batches)
                    assert method._incomplete
                    with pytest.raises((RuntimeError, ValueError), match="incomplete"):
                        engine.save_checkpoint(Path(output) / "incomplete")
                    engine.model.log_prob = original
                    with pytest.raises(ValueError, match="incomplete"):
                        method.update(batches)
                    engine.load_checkpoint(complete)
                    method.update(batches)
                    _dense_update(reference, optimizers, kind, data, 5)
                    _assert_models(_export(engine), reference)

        fresh = Trainer(IQLActor(3, 2, 8), parallel=context, optimizer_factory=_adam)
        with pytest.raises(ValueError, match="ranks disagree"):
            IQLMethod(
                fresh, ContinuousTwinQ(3, 2, 8), StateValue(3, 8), gamma=0.9 if rank else 0.99
            )
        assert set(fresh.roles) == {"model"}
        for axis in ("tensor_parallel", "pipeline_parallel", "context_parallel", "gtp_remat"):
            unsupported = ParallelContext(ParallelConfig(**{axis: 2}))
            if axis == "pipeline_parallel":
                with pytest.raises(ValueError, match="PP"):
                    Trainer(IQLActor(3, 2, 8), parallel=unsupported, optimizer_factory=_adam)
                continue
            fresh = Trainer(IQLActor(3, 2, 8), parallel=unsupported, optimizer_factory=_adam)
            with pytest.raises(ValueError, match="DP/ZeRO0-3"):
                IQLMethod(fresh, ContinuousTwinQ(3, 2, 8), StateValue(3, 8))
            assert set(fresh.roles) == {"model"}
    finally:
        dist.destroy_process_group()


def test_real_dp2_offline_zero0_3_global_oracle_empty_rank_checkpoint_and_failure(tmp_path):
    temp_root = Path(tempfile.gettempdir()).resolve()
    if not str(temp_root).isascii():
        temp_root = Path("C:/Temp").resolve()
    rendezvous_dir = Path(tempfile.mkdtemp(prefix="aster_offline_", dir=temp_root)).resolve()
    try:
        mp.spawn(_worker, args=(str(rendezvous_dir / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if rendezvous_dir.parent == temp_root and rendezvous_dir.name.startswith("aster_offline_"):
            shutil.rmtree(rendezvous_dir)
