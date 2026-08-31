from copy import deepcopy
import math

import pytest
import torch
import torch.nn.functional as F

from aster.methods.conservative import (
    CQLPolicyConfig,
    CQLPolicy,
    CQLTwinQ,
    CQLMethod,
    conservative_gap,
)
from aster.training import Trainer


def transition_batch(size=4):
    return dict(
        observations=torch.randn(size, 3),
        next_observations=torch.randn(size, 3),
        actions=torch.randn(size, 2).tanh(),
        rewards=torch.randn(size),
        terminated=torch.arange(size) == 1,
        truncated=torch.arange(size) == size - 1,
    )


def build(*, stage=0, precision="fp32", accumulation=1, **settings):
    engine = Trainer(
        CQLPolicy(CQLPolicyConfig(3, 2, hidden=8)),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.001),
        max_grad_norm=None,
        zero_stage=stage,
        precision=precision,
        accumulation_steps=accumulation,
    )
    return engine, CQLMethod(engine, CQLTwinQ(3, 2, 8), num_random=3, **settings)


def reference_policy(policy, obs, noise, actions=None):
    # Independently spell out the official policy algebra, rather than invoking its helpers.
    hidden = F.relu(F.linear(obs, policy.hidden[0].weight, policy.hidden[0].bias))
    hidden = F.relu(F.linear(hidden, policy.hidden[2].weight, policy.hidden[2].bias))
    mean = F.linear(hidden, policy.mean.weight, policy.mean.bias)
    log_std = F.linear(hidden, policy.log_std.weight, policy.log_std.bias).clamp(-5, 2)
    if actions is None:
        raw = mean + log_std.exp() * noise
        action = raw.tanh()
    else:
        mean = mean.clamp(-9, 9)
        action = actions
        raw = torch.log((1 + action).clamp_min(1e-6) / (1 - action).clamp_min(1e-6)) / 2
    logp = torch.distributions.Normal(mean, log_std.exp()).log_prob(raw)
    logp = (logp - torch.log(1 - action * action + 1e-6)).sum(-1)
    return action, logp


def test_cql_policy_sampling_bc_boundary_and_gradients():
    torch.manual_seed(91)
    policy = CQLPolicy(CQLPolicyConfig(3, 2, hidden=8))
    reference = deepcopy(policy)
    observations, noise = torch.randn(5, 3), torch.randn(5, 2)
    actual = policy(observations, noise=noise)
    expected = reference_policy(reference, observations, noise)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)
    sum(x.sum() for x in actual).backward()
    sum(x.sum() for x in expected).backward()
    for p, q in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=2e-5)
    with torch.no_grad():
        policy.mean.bias.fill_(12)
    endpoint = torch.tensor([[1.0, -1.0]]).expand(5, -1)
    expected = reference_policy(policy, observations, noise, actions=endpoint)[1]
    torch.testing.assert_close(policy.log_prob(observations, endpoint), expected)
    assert torch.isfinite(expected).all()
    with pytest.raises(ValueError, match="noise"):
        policy(observations, deterministic=True, noise=noise)


@pytest.mark.parametrize("version", [2, 3])
def test_conservative_gap_official_sum_not_logmean_and_density_detached(version):
    torch.manual_seed(17)
    values = [torch.randn(3, requires_grad=True)] + [
        torch.randn(3, 4, requires_grad=True) for _ in range(5)
    ]
    data, random, current, nxt, logp, next_logp = values
    actual = conservative_gap(*values, action_dim=2, temperature=0.7, weight=2.5, version=version)
    scores = (
        torch.cat((random + math.log(4), nxt - next_logp.detach(), current - logp.detach()), 1)
        if version == 3
        else torch.cat((random, data[:, None], nxt, current), 1)
    )
    expected = 2.5 * (0.7 * torch.log(torch.exp(scores / 0.7).sum(1)) - data)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert logp.grad is None and next_logp.grad is None
    assert all(value.grad is not None for value in values[:4])


@pytest.mark.parametrize(
    "stage,precision",
    [(0, "fp32"), (1, "fp32"), (2, "fp32"), (3, "fp32"), (0, "bf16"), (3, "bf16")],
)
def test_cql_multirole_adam_and_exact_next_update(stage, precision, tmp_path):
    torch.manual_seed(11)
    engine, method = build(
        stage=stage, precision=precision, lagrange=True, deterministic_backup=False
    )
    data = transition_batch()
    first = method.update([data])
    assert len(first) == 4 and all(result.updated for result in first.values())
    engine.save_checkpoint(tmp_path / "complete")
    second = method.update([data])
    restored, resumed = build(
        stage=stage, precision=precision, lagrange=True, deterministic_backup=False
    )
    restored.load_checkpoint(tmp_path / "complete", trusted=True)
    recovered = resumed.update([data])
    assert resumed.updates == method.updates == 2
    for phase in second:
        assert second[phase].loss == recovered[phase].loss
    for role in engine.roles:
        left, right = engine.export_state_dict(role=role), restored.export_state_dict(role=role)
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


@pytest.mark.parametrize("version,max_backup,bc", [(2, True, 0), (3, False, 3)])
def test_cql_fixed_entropy_backup_and_bc_branches(version, max_backup, bc):
    torch.manual_seed(21)
    engine, method = build(
        automatic_entropy=False, max_q_backup=max_backup, policy_eval_start=bc, version=version
    )
    before = engine.export_state_dict(role="model")
    result = method.update([transition_batch()])
    assert set(result) == {"cql_actor", "cql_critic"}
    assert any(
        not torch.equal(value, engine.export_state_dict(role="model")[key])
        for key, value in before.items()
    )


def test_cql_preflight_and_incomplete_update_reject_checkpoint(tmp_path, monkeypatch):
    engine, method = build()
    before = engine.export_state_dict(role="model")
    bad = transition_batch()
    bad["actions"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="preflight"):
        method.update([bad])
    assert engine.steps == 0 and not method._incomplete
    for key, value in before.items():
        torch.testing.assert_close(engine.export_state_dict(role="model")[key], value)

    def fail(*args, **kwargs):
        raise RuntimeError("injected role failure")

    monkeypatch.setattr(engine, "phase", fail)
    with pytest.raises(RuntimeError, match="injected"):
        method.update([transition_batch()])
    with pytest.raises(RuntimeError, match="incomplete"):
        method.state_dict()
    with pytest.raises(ValueError, match="Restore"):
        method.update([transition_batch()])


def test_cql_complete_update_matches_independent_simultaneous_gradient_reference():
    """Compare one real multi-role Adam step, not just two copies of CQLMethod.

    The reference forms actor/Q/dual gradients at old weights before applying independent
    optimizers. This checks that our reordered phases preserve the official old-Q actor
    gradient, old-policy proposal distribution, and pre-dual-step Q multiplier.
    """
    torch.manual_seed(16)
    engine, method = build(lagrange=True, deterministic_backup=False, target_action_gap=0.7)
    policy, critic, target = (
        deepcopy(engine.model),
        deepcopy(method.critic),
        deepcopy(method.target),
    )
    log_alpha, log_dual = torch.nn.Parameter(torch.zeros(())), torch.nn.Parameter(torch.zeros(()))
    optimizers = [torch.optim.Adam(module.parameters(), lr=0.001) for module in (policy, critic)]
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=0.001)
    dual_optimizer = torch.optim.Adam([log_dual], lr=0.001)
    data = transition_batch()
    obs, nxt, count = data["observations"], data["next_observations"], len(data["observations"])
    rng = torch.get_rng_state()
    noise = torch.randn_like(data["actions"])
    action, actor_logp = reference_policy(policy, obs, noise)

    def proposals(x, number):
        repeated = x[:, None].expand(-1, number, -1).reshape(count * number, -1)
        with torch.no_grad():
            a, lp = reference_policy(policy, repeated, torch.randn(count * number, 2))
        return a.reshape(count, number, 2), lp.reshape(count, number)

    current, current_logp = proposals(obs, 3)
    next_actions, next_logp = proposals(nxt, 3)
    backup, backup_logp = proposals(nxt, 1)
    random_actions = torch.empty_like(current).uniform_(-1, 1)
    alpha_loss = -(log_alpha * (actor_logp.detach() - 2)).mean()
    alpha_loss.backward()
    alpha_optimizer.step()
    alpha = log_alpha.detach().exp()
    with torch.no_grad():
        tq1, tq2 = target(nxt, backup[:, 0])
        target_value = data["rewards"] + 0.99 * (~data["terminated"]) * (
            torch.minimum(tq1, tq2) - alpha * backup_logp[:, 0]
        )
    actor_q = critic(obs, action)
    actor_loss = (alpha * actor_logp - torch.minimum(*actor_q)).mean()
    data_q = critic(obs, data["actions"])

    def values(a):
        repeated = obs[:, None].expand(-1, 3, -1).reshape(count * 3, -1)
        return tuple(x.reshape(count, 3) for x in critic(repeated, a.reshape(-1, 2)))

    random_q, current_q, next_q = values(random_actions), values(current), values(next_actions)
    gaps = []
    for i in range(2):
        scores = torch.cat(
            (random_q[i] + math.log(4), next_q[i] - next_logp, current_q[i] - current_logp), 1
        )
        gaps.append(torch.logsumexp(scores, 1) - data_q[i])
    dual_loss = -log_dual.exp() * (0.5 * (gaps[0].detach() + gaps[1].detach()).mean() - 0.7)
    critic_loss = sum(
        (q - target_value).square().mean() for q in data_q
    ) + log_dual.detach().exp() * (gaps[0].mean() + gaps[1].mean() - 1.4)
    parameters = [list(policy.parameters()), list(critic.parameters()), [log_dual]]
    gradients = [
        torch.autograd.grad(loss, params, retain_graph=True)
        for loss, params in zip((actor_loss, critic_loss, dual_loss), parameters)
    ]
    for optimizer, params, grads in zip((*optimizers, dual_optimizer), parameters, gradients):
        for parameter, gradient in zip(params, grads):
            parameter.grad = gradient
        optimizer.step()
    torch.set_rng_state(rng)
    method.update([data])
    for expected, actual in ((policy, engine.model), (critic, method.critic)):
        for p, q in zip(expected.parameters(), actual.parameters()):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(log_alpha, method.alpha())
    torch.testing.assert_close(log_dual, method.multiplier())
    for old, updated, actual in zip(
        target.parameters(), critic.parameters(), method.target.parameters()
    ):
        torch.testing.assert_close(actual, 0.99 * old + 0.01 * updated)


def test_cql_offline_reward_learning_and_local_policy_artifact(tmp_path):
    """Fixed offline bandit data; no target action is supplied to the actor loss."""
    from aster.models import load_model

    torch.set_num_threads(1)
    torch.manual_seed(26)
    policy = CQLPolicy(CQLPolicyConfig(3, 2, hidden=32))
    engine = Trainer(
        policy, optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.003), max_grad_norm=None
    )
    method = CQLMethod(
        engine,
        CQLTwinQ(3, 2, 32),
        num_random=8,
        automatic_entropy=False,
        fixed_alpha=0.01,
        conservative_weight=0.1,
        critic_lr=0.003,
    )
    obs, action = torch.zeros(128, 3), torch.empty(128, 2).uniform_(-1, 1)
    ideal = torch.tensor([0.5, -0.4]).expand_as(action)
    batch = dict(
        observations=obs,
        next_observations=obs,
        actions=action,
        rewards=-(action - ideal).square().sum(-1),
        terminated=torch.ones(128, dtype=torch.bool),
    )
    with torch.no_grad():
        before = (policy(obs, deterministic=True)[0] - ideal).square().mean()
    for _ in range(200):
        method.update([batch])
    with torch.no_grad():
        selected = policy(obs, deterministic=True)[0]
        assert (selected - ideal).square().mean() < before * 0.25
    policy.save_pretrained(tmp_path / "policy")
    restored = load_model(tmp_path / "policy")
    torch.testing.assert_close(restored(obs, deterministic=True)[0], selected, atol=0, rtol=0)


def test_cql_does_not_silently_ignore_per_or_unknown_fields():
    engine, method = build()
    batch = transition_batch()
    original_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="uniform replay"):
        method.update([{**batch, "importance_weights": torch.tensor([1.0, 0.5, 0.4, 0.9])}])
    with pytest.raises(ValueError, match="Unsupported"):
        method.update([{**batch, "discount": torch.ones(4)}])
    assert engine.steps == 0
    torch.testing.assert_close(original_rng, torch.get_rng_state(), rtol=0, atol=0)
    assert method.update([{**batch, "importance_weights": torch.ones(4)}])["cql_actor"].updated
