import copy
import pytest
import torch
from aster.training import Trainer
from aster.methods.offline import (
    DeterministicActor,
    ContinuousTwinQ,
    IQLActor,
    StateValue,
    TD3Method,
    IQLMethod,
    expectile_loss,
    advantage_weighted_bc,
)


def batch():
    return {
        "observations": torch.randn(4, 3),
        "actions": torch.randn(4, 2).tanh(),
        "rewards": torch.randn(4),
        "next_observations": torch.randn(4, 3),
        "terminated": torch.tensor([False, True, False, False]),
    }


def test_expectile_awr_and_iql_default_distribution():
    difference = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
    torch.testing.assert_close(expectile_loss(difference, 0.8), torch.tensor([0.8, 0.0, 7.2]))
    logp = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    advantage = torch.tensor([0.0, 1.0, 1000.0], requires_grad=True)
    values = advantage_weighted_bc(logp, advantage, inverse_temperature=1.0)
    torch.testing.assert_close(values, torch.tensor([1.0, 2 * torch.e, 300.0]))
    values.sum().backward()
    assert advantage.grad is None
    actor = IQLActor(3, 2, 8)
    observations = torch.randn(4, 3)
    actions = torch.ones(4, 2)
    distribution = actor.distribution(observations)
    reference = -0.5 * (
        (actions - distribution.mean).square() + torch.log(torch.tensor(2 * torch.pi))
    ).sum(-1)
    torch.testing.assert_close(actor.log_prob(observations, actions), reference)


def test_td3_delayed_target_and_full_batch_bc_scaling():
    torch.manual_seed(3)
    actor = DeterministicActor(3, 2, 8)
    critic = ContinuousTwinQ(3, 2, 8)
    left = Trainer(copy.deepcopy(actor), lr=0.001, max_grad_norm=None)
    right = Trainer(copy.deepcopy(actor), lr=0.001, max_grad_norm=None, accumulation_steps=2)
    a = TD3Method(left, copy.deepcopy(critic), policy_noise=0.0, bc_alpha=2.5)
    b = TD3Method(right, copy.deepcopy(critic), policy_noise=0.0, bc_alpha=2.5)
    data = batch()
    split = [
        {key: value[:1] for key, value in data.items()},
        {key: value[1:] for key, value in data.items()},
    ]
    before = copy.deepcopy(a.target_actor.state_dict())
    assert a.update([data])["actor"] is None
    assert b.update(split)["actor"] is None
    for key, value in before.items():
        torch.testing.assert_close(a.target_actor.state_dict()[key], value)
    assert a.update([data])["actor"].updated
    assert b.update(split)["actor"].updated
    for p, q in zip(left.model.parameters(), right.model.parameters()):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
    for key, value in before.items():
        torch.testing.assert_close(
            a.target_actor.state_dict()[key], 0.995 * value + 0.005 * left.model.state_dict()[key]
        )


def test_iql_roles_checkpoint_next_update(tmp_path):
    torch.manual_seed(8)

    def build():
        engine = Trainer(IQLActor(3, 2, 8), lr=0.001, max_grad_norm=None)
        return engine, IQLMethod(engine, ContinuousTwinQ(3, 2, 8), StateValue(3, 8))

    engine, method = build()
    data = batch()
    result = method.update([data])
    assert all(value.updated for value in result.values())
    engine.save_checkpoint(tmp_path / "checkpoint")
    second = method.update([data])
    restored, resumed = build()
    restored.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    recovered = resumed.update([data])
    assert resumed.updates == method.updates == 2
    for key in second:
        assert abs(second[key].loss - recovered[key].loss) < 1e-7
    for role in ("model", "value", "critic", "target_critic"):
        for p, q in zip(
            engine.roles[role].model.parameters(), restored.roles[role].model.parameters()
        ):
            torch.testing.assert_close(p, q)


def _adam(parameters):
    return torch.optim.Adam(parameters, lr=0.001, betas=(0.8, 0.97), eps=1e-6)


def _build(kind, stage=0, precision="fp32", accumulation=1):
    actor = IQLActor(3, 2, 8) if kind == "iql" else DeterministicActor(3, 2, 8)
    engine = Trainer(
        actor,
        optimizer_factory=_adam,
        lr=0.001,
        max_grad_norm=None,
        zero_stage=stage,
        precision=precision,
        accumulation_steps=accumulation,
    )
    critic = ContinuousTwinQ(3, 2, 8)
    if kind == "iql":
        method = IQLMethod(
            engine,
            critic,
            StateValue(3, 8),
            critic_optimizer_factory=_adam,
            value_optimizer_factory=_adam,
        )
    else:
        method = TD3Method(
            engine,
            critic,
            critic_optimizer_factory=_adam,
            bc_alpha=2.5 if kind == "td3_bc" else None,
        )
    return engine, method


def _export_all(engine):
    return {
        name: engine.export_state_dict(role=name, only_rank_zero=False) for name in engine.roles
    }


def _assert_export(actual, expected, *, exact=True):
    assert actual.keys() == expected.keys()
    for role, tensors in actual.items():
        assert tensors.keys() == expected[role].keys()
        for name, value in tensors.items():
            torch.testing.assert_close(
                value, expected[role][name], atol=0 if exact else 3e-7, rtol=0 if exact else 2e-5
            )


@pytest.mark.parametrize("kind", ["td3", "td3_bc", "iql"])
@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_all_offline_roles_zero_exact_stochastic_resume(kind, stage, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(125)
    engine, method = _build(kind, stage)
    data = batch()
    first = method.update([data])
    assert first["critic"].updated
    initial = _export_all(engine)
    checkpoint = engine.save_checkpoint(tmp_path / "native")
    second = method.update([data])
    expected = _export_all(engine)
    next_random = torch.rand(7)
    engine.load_checkpoint(checkpoint)
    _assert_export(_export_all(engine), initial)
    recovered = method.update([data])
    _assert_export(_export_all(engine), expected)
    torch.testing.assert_close(torch.rand(7), next_random, atol=0, rtol=0)
    for name, result in second.items():
        assert result is not None and result.updated
        assert recovered[name].loss == result.loss
    assert method.updates == 2
    if kind == "iql":
        assert expected["model"]["log_std.values"].abs().sum() > 0
    if stage == 3:
        from aster.training.sharding import zero3_units

        units = zero3_units(engine.model)
        assert units and all(unit.gathers > 0 and unit.releases > 0 for unit in units)
        assert all(
            parameter.numel() == 0 for unit in units for parameter in unit.module.parameters()
        )


@pytest.mark.parametrize("kind", ["td3", "td3_bc", "iql"])
@pytest.mark.parametrize("stage", [0, 3])
def test_offline_bfloat16_complete_multi_phase_resume(kind, stage, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(731)
    engine, method = _build(kind, stage, precision="bf16")
    data = batch()
    method.update([data])
    checkpoint = engine.save_checkpoint(tmp_path / "bf16")
    method.update([data])
    expected = _export_all(engine)
    engine.load_checkpoint(checkpoint)
    method.update([data])
    _assert_export(_export_all(engine), expected)


def test_offline_count_is_exact_integer_even_when_loss_is_bfloat16():
    from aster.methods.offline import _term

    term = _term(torch.ones(257, dtype=torch.bfloat16), "count")
    assert term.denominator.dtype == torch.int64 and term.denominator.item() == 257
    assert term.numerator.item() == 257 and term.mean.item() == 1


@pytest.mark.parametrize(
    "invalid", ["shape", "nonfinite", "discount", "truncated", "unknown", "dtype", "slots", "empty"]
)
def test_offline_invalid_input_rejected_without_parameter_rng_or_cursor_mutation(invalid):
    engine, method = _build("iql", stage=3)
    data = batch()
    batches = [data]
    if invalid == "shape":
        data["actions"] = torch.zeros(4, 3)
    if invalid == "nonfinite":
        data["next_observations"][0, 0] = float("nan")
    if invalid == "discount":
        data["discounts"] = torch.ones(4)
    if invalid == "truncated":
        data["truncated"] = torch.zeros(4)
    if invalid == "unknown":
        data["reward"] = data["rewards"]
    if invalid == "dtype":
        data["observations"] = data["observations"].double()
    if invalid == "slots":
        batches = []
    if invalid == "empty":
        batches = [{key: value[:0] for key, value in data.items()}]
    before, rng = _export_all(engine), torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        method.update(batches)
    _assert_export(_export_all(engine), before)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert not method._incomplete and method.updates == 0 and engine.steps == 0


def test_iql_half_round_overflow_is_not_checkpointable_and_requires_full_restore(tmp_path):
    engine, method = _build("iql", stage=3)
    data = batch()
    checkpoint = engine.save_checkpoint(tmp_path / "complete")
    original = engine.model.log_prob

    engine.model.log_prob = lambda observations, actions: (
        original(observations, actions) * float("nan")
    )
    with pytest.raises(RuntimeError, match="phase skipped"):
        method.update([data])
    assert engine.roles["value"].updates == 1 and engine.roles["model"].updates == 0
    assert method._incomplete and method.updates == 0
    with pytest.raises((ValueError, RuntimeError), match="incomplete"):
        engine.save_checkpoint(tmp_path / "invalid")
    engine.model.log_prob = original
    with pytest.raises(ValueError, match="incomplete"):
        method.update([data])
    engine.load_checkpoint(checkpoint)
    assert not method._incomplete
    assert all(result.updated for result in method.update([data]).values())


def test_terminal_mask_and_explicit_discount_match_but_time_limit_keeps_bootstrap():
    torch.manual_seed(741)
    left, a = _build("iql")
    torch.manual_seed(741)
    right, b = _build("iql")
    data = batch()
    data["truncated"] = torch.tensor([True, False, False, True])
    explicit = {**data, "discounts": 0.99 * (~data["terminated"]).float()}
    actual = a.update([data])
    expected = b.update([explicit])
    _assert_export(_export_all(left), _export_all(right))
    assert actual["critic"].loss == expected["critic"].loss
    wrong = {**data, "terminated": data["terminated"] | data["truncated"]}
    torch.manual_seed(741)
    other, c = _build("iql")
    assert c.update([wrong])["critic"].loss != actual["critic"].loss


def test_native_iql_same_shape_semantic_config_checkpoint_mismatch(tmp_path):
    engine, method = _build("iql")
    engine.save_checkpoint(tmp_path / "config")
    changed = Trainer(
        IQLActor(3, 2, 8, log_std_min=-4.0), optimizer_factory=_adam, lr=0.001, max_grad_norm=None
    )
    IQLMethod(
        changed,
        ContinuousTwinQ(3, 2, 8),
        StateValue(3, 8),
        critic_optimizer_factory=_adam,
        value_optimizer_factory=_adam,
    )
    with pytest.raises(ValueError, match="配置"):
        changed.load_checkpoint(tmp_path / "config")
