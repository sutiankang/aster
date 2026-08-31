"""Continuous-action CQL and SAC updates through the shared multi-role trainer."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..core import LossTerm
from ..models.conservative import CQLPolicyConfig, CQLPolicy, CQLTwinQ


class LogCoefficient(nn.Module):
    """A trainable scalar role whose value and optimizer state belong to the trainer."""

    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(1))

    def forward(self):
        return self.value[0]


def conservative_gap(
    data_q,
    random_q,
    current_q,
    next_q,
    current_logp,
    next_logp,
    *,
    action_dim,
    temperature=1.0,
    weight=1.0,
    version=3,
):
    """Return the per-transition conservative Q penalty; gradients flow to Q values,
    not to proposal-action probability densities."""
    if (
        data_q.ndim != 1
        or random_q.ndim != 2
        or random_q.shape[0] != len(data_q)
        or random_q.shape[1] < 1
        or any(
            value.shape != random_q.shape for value in (current_q, next_q, current_logp, next_logp)
        )
    ):
        raise ValueError("CQL candidate tensors must be aligned B,N arrays")
    if (
        type(action_dim) is not int
        or action_dim < 1
        or temperature <= 0
        or weight < 0
        or not math.isfinite(temperature)
        or not math.isfinite(weight)
        or version not in (2, 3)
    ):
        raise ValueError("Invalid CQL conservative objective")
    if version == 3:
        scores = torch.cat(
            (
                random_q + action_dim * math.log(2),
                next_q - next_logp.detach(),
                current_q - current_logp.detach(),
            ),
            -1,
        )
    else:
        scores = torch.cat((random_q, data_q[:, None], next_q, current_q), -1)
    return weight * (
        temperature * torch.logsumexp(scores.float() / temperature, -1) - data_q.float()
    )


def _transition_loss(values, name):
    return LossTerm(
        values.sum(),
        torch.tensor(values.numel(), dtype=torch.int64, device=values.device),
        "transition",
        name,
    )


class CQLMethod:
    """CQL multi-role updates through the same trainer across supported DP/ZeRO layouts."""

    def __init__(
        self,
        engine,
        critic: CQLTwinQ,
        *,
        discount=0.99,
        reward_scale=1.0,
        tau=0.01,
        temperature=1.0,
        conservative_weight=1.0,
        version=3,
        num_random=10,
        deterministic_backup=True,
        max_q_backup=False,
        max_backup_actions=10,
        automatic_entropy=True,
        target_entropy=None,
        fixed_alpha=1.0,
        lagrange=False,
        target_action_gap=0.0,
        policy_eval_start=0,
        critic_lr=1e-3,
        coefficient_lr=1e-3,
    ):

        declaration = {
            key: value for key, value in locals().items() if key not in ("self", "engine", "critic")
        }
        declaration["critic_dimensions"] = getattr(critic, "dimensions", None)
        declarations = engine.parallel.world.gather_objects(declaration)
        if any(item != declaration for item in declarations):
            raise ValueError("CQL ranks must declare identical method settings and role dimensions")
        c = engine.model.config
        if not isinstance(c, CQLPolicyConfig) or critic.dimensions[:2] != (
            c.observation_dim,
            c.action_dim,
        ):
            raise ValueError("CQL requires matching native stochastic policy and continuous twin Q")
        groups = engine.parallel
        if any(
            getattr(groups, name).size > 1 for name in ("tp", "pp", "cp", "gtp_remat", "ep", "etp")
        ):
            raise ValueError(
                "CQL vector MLP supports DP/ZeRO; other parallel layouts need explicit model partitioning"
            )
        settings = dict(
            discount=discount,
            reward_scale=reward_scale,
            tau=tau,
            temperature=temperature,
            conservative_weight=conservative_weight,
            version=version,
            num_random=num_random,
            deterministic_backup=deterministic_backup,
            max_q_backup=max_q_backup,
            max_backup_actions=max_backup_actions,
            automatic_entropy=automatic_entropy,
            target_entropy=-float(c.action_dim) if target_entropy is None else target_entropy,
            fixed_alpha=fixed_alpha,
            lagrange=lagrange,
            target_action_gap=target_action_gap,
            policy_eval_start=policy_eval_start,
            critic_lr=critic_lr,
            coefficient_lr=coefficient_lr,
        )
        if not all(math.isfinite(value) for value in settings.values()):
            raise ValueError("CQL settings must be finite")
        if (
            not 0 <= discount <= 1
            or not 0 < tau <= 1
            or min(temperature, critic_lr, coefficient_lr) <= 0
            or min(conservative_weight, fixed_alpha) < 0
            or version not in (2, 3)
            or any(type(v) is not int or v < 1 for v in (num_random, max_backup_actions))
            or type(policy_eval_start) is not int
            or policy_eval_start < 0
            or any(
                type(v) is not bool
                for v in (deterministic_backup, max_q_backup, automatic_entropy, lagrange)
            )
        ):
            raise ValueError("Invalid CQL update settings")
        self.engine, self.settings = engine, settings
        self.critic = engine.add_role(
            "cql_critic",
            critic,
            optimizer_factory=lambda parameters: torch.optim.Adam(parameters, lr=critic_lr),
        )
        self.target = engine.clone_target(
            "cql_critic", "cql_target", factory=lambda: CQLTwinQ(*critic.dimensions)
        )
        coefficient_factory = lambda parameters: torch.optim.Adam(parameters, lr=coefficient_lr)
        self.alpha = (
            engine.add_role("cql_alpha", LogCoefficient(), optimizer_factory=coefficient_factory)
            if automatic_entropy
            else None
        )
        self.multiplier = (
            engine.add_role("cql_lagrange", LogCoefficient(), optimizer_factory=coefficient_factory)
            if lagrange
            else None
        )
        self.updates, self._incomplete = 0, False
        engine.register_state("cql_method", self)

    def _autocast(self):
        return torch.autocast(
            self.engine.device.type,
            dtype=torch.bfloat16 if self.engine.precision == "bf16" else torch.float16,
            enabled=self.engine.precision != "fp32",
        )

    def _preflight(self, microbatches):
        error, batches = None, None
        try:
            if self._incomplete:
                raise RuntimeError(
                    "Restore a complete CQL checkpoint before retrying a partial update"
                )
            batches = list(microbatches)
            if len(batches) != self.engine.accumulation_steps:
                raise ValueError(
                    "CQL needs exactly one transition microbatch per accumulation slot"
                )
            c = self.engine.model.config
            for batch in batches:
                unknown = set(batch) - {
                    "observations",
                    "next_observations",
                    "actions",
                    "rewards",
                    "terminated",
                    "truncated",
                    "discounts",
                    "importance_weights",
                    "replay_indices",
                    "replay_versions",
                }
                if unknown:
                    raise ValueError(f"Unsupported CQL transition fields: {sorted(unknown)}")
                obs, next_obs, action, reward, terminal = (
                    batch[k]
                    for k in (
                        "observations",
                        "next_observations",
                        "actions",
                        "rewards",
                        "terminated",
                    )
                )
                if (
                    obs.ndim != 2
                    or obs.shape != (len(obs), c.observation_dim)
                    or len(obs) < 1
                    or next_obs.shape != obs.shape
                    or action.shape != (len(obs), c.action_dim)
                    or reward.shape != (len(obs),)
                    or terminal.shape != (len(obs),)
                    or terminal.dtype != torch.bool
                ):
                    raise ValueError(
                        "CQL requires aligned nonempty vector transitions and boolean terminated"
                    )
                for value in (obs, next_obs, action, reward):
                    if (
                        value.dtype != torch.float32
                        or value.device != self.engine.device
                        or not torch.isfinite(value).all()
                    ):
                        raise ValueError(
                            "Replay tensors must be finite FP32 on the Trainer device before autocast"
                        )
                if terminal.device != self.engine.device or action.abs().max() > 1:
                    raise ValueError(
                        "CQL actions must be normalized to [-1,1] on the Trainer device"
                    )
                # time-limit truncation is metadata, NOT an absorbing terminal for TD bootstrap.
                truncated = batch.get("truncated")
                if truncated is not None and (
                    truncated.shape != terminal.shape
                    or truncated.dtype != torch.bool
                    or truncated.device != terminal.device
                ):
                    raise ValueError("truncated must be a separate aligned boolean vector")
                if "discounts" in batch:
                    raise ValueError(
                        "CQL uses declared scalar discount; pre-discounted/n-step returns need an explicit objective"
                    )
                importance = batch.get("importance_weights")
                if importance is not None and (
                    importance.shape != reward.shape
                    or importance.device != reward.device
                    or not importance.is_floating_point()
                    or not torch.isfinite(importance).all()
                    or not (importance == 1).all()
                ):
                    raise ValueError(
                        "CQL currently requires uniform replay weights; proposal-density correction is not PER weighting"
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = self.engine.parallel.world.gather_objects(error)
        if any(errors):
            raise ValueError("CQL collective preflight failed: " + str(errors))
        return batches

    @staticmethod
    def _candidate_q(critic, observations, actions):
        count, candidates, width = actions.shape
        repeated = observations[:, None].expand(-1, candidates, -1).reshape(count * candidates, -1)
        return tuple(
            q.reshape(count, candidates) for q in critic(repeated, actions.reshape(-1, width))
        )

    def _proposals(self, policy, observations, count):
        repeated = (
            observations[:, None].expand(-1, count, -1).reshape(len(observations) * count, -1)
        )
        actions, logp = policy(repeated)
        return actions.reshape(len(observations), count, -1), logp.reshape(len(observations), count)

    def _gaps(self, critic, batch):
        obs = batch["observations"]
        data = critic(obs, batch["actions"])
        random_q = self._candidate_q(critic, obs, batch["random_actions"])
        current_q = self._candidate_q(critic, obs, batch["current_actions"])
        # next-state proposals still evaluate Q(s,a), not Q(s',a).
        next_q = self._candidate_q(critic, obs, batch["next_actions"])
        gaps = tuple(
            conservative_gap(
                data[i],
                random_q[i],
                current_q[i],
                next_q[i],
                batch["current_logp"],
                batch["next_logp"],
                action_dim=self.engine.model.config.action_dim,
                temperature=self.settings["temperature"],
                weight=self.settings["conservative_weight"],
                version=self.settings["version"],
            )
            for i in range(2)
        )
        return data, gaps

    def update(self, microbatches):
        batches = self._preflight(microbatches)
        self._incomplete = True
        policy, s, results = self.engine.model, self.settings, {}
        prepared = []
        # Freeze all stochastic proposals before any role changes. Explicit cached noise makes
        # temperature and actor see the very same sample, not two independent policy draws.
        with torch.no_grad(), self._autocast():
            for batch in batches:
                obs, nxt = batch["observations"], batch["next_observations"]
                noise = torch.randn_like(batch["actions"])
                _, logp = policy(obs, noise=noise)
                current, current_logp = self._proposals(policy, obs, s["num_random"])
                next_actions, next_logp = self._proposals(policy, nxt, s["num_random"])
                backup, backup_logp = self._proposals(
                    policy, nxt, s["max_backup_actions"] if s["max_q_backup"] else 1
                )
                prepared.append(
                    {
                        **batch,
                        "actor_noise": noise,
                        "actor_logp": logp,
                        "current_actions": current,
                        "current_logp": current_logp,
                        "next_actions": next_actions,
                        "next_logp": next_logp,
                        "backup_actions": backup,
                        "backup_logp": backup_logp,
                        "random_actions": torch.empty_like(current).uniform_(-1, 1),
                    }
                )

        def phase(name, role, objective, freeze=()):
            result = self.engine.phase(
                name, role=role, objective=objective, microbatches=prepared, freeze_roles=freeze
            )
            if not result.updated:
                raise RuntimeError(
                    f"{name} did not update; restore the last complete CQL checkpoint"
                )
            results[name] = result

        if self.alpha is not None:
            phase(
                "cql_temperature",
                "cql_alpha",
                lambda coefficient, b: _transition_loss(
                    -coefficient() * (b["actor_logp"] + s["target_entropy"]), "entropy_temperature"
                ),
            )
        with torch.no_grad(), self._autocast():
            alpha = (
                self.alpha().exp()
                if self.alpha is not None
                else torch.tensor(s["fixed_alpha"], device=self.engine.device)
            )
            multiplier = (
                self.multiplier().exp().clamp(0, 1e6) if self.multiplier is not None else 1.0
            )
            for batch in prepared:
                q1, q2 = self._candidate_q(
                    self.target, batch["next_observations"], batch["backup_actions"]
                )
                target_q = (
                    torch.minimum(q1.max(-1).values, q2.max(-1).values)
                    if s["max_q_backup"]
                    else torch.minimum(q1[:, 0], q2[:, 0])
                )
                if not s["max_q_backup"] and not s["deterministic_backup"]:
                    target_q = target_q - alpha * batch["backup_logp"][:, 0]
                batch["target"] = (
                    s["reward_scale"] * batch["rewards"]
                    + s["discount"] * (~batch["terminated"]) * target_q
                )
                if self.multiplier is not None:
                    _, gaps = self._gaps(self.critic, batch)
                    batch["gap_mean"] = 0.5 * (gaps[0] + gaps[1]) - s["target_action_gap"]
        if self.multiplier is not None:
            phase(
                "cql_dual",
                "cql_lagrange",
                lambda coefficient, b: _transition_loss(
                    -coefficient().exp().clamp(0, 1e6) * b["gap_mean"], "conservative_dual"
                ),
            )

        def actor_objective(model, batch):
            action, logp = model(batch["observations"], noise=batch["actor_noise"])
            if self.updates + 1 < s["policy_eval_start"]:
                value = model.log_prob(batch["observations"], batch["actions"])
            else:
                q1, q2 = self.critic(batch["observations"], action)
                value = torch.minimum(q1, q2)
            return _transition_loss(alpha * logp - value, "cql_actor")

        phase("cql_actor", "model", actor_objective, freeze=("cql_critic",))

        def critic_objective(critic, batch):
            data, gaps = self._gaps(critic, batch)
            error = (data[0].float() - batch["target"]).square() + (
                data[1].float() - batch["target"]
            ).square()
            conservative = gaps[0] + gaps[1]
            if self.multiplier is not None:
                # CQL uses the multiplier value BEFORE its dual optimizer update in this Q step.
                conservative = multiplier * (conservative - 2 * s["target_action_gap"])
            return _transition_loss(error + conservative, "cql_critic")

        phase("cql_critic", "cql_critic", critic_objective, freeze=("model",))
        self.engine.update_target("cql_critic", "cql_target", 1 - s["tau"])
        self.updates += 1
        self._incomplete = False
        return results

    def state_dict(self):
        if self._incomplete:
            raise RuntimeError("Cannot checkpoint an incomplete CQL multi-role update")
        return {
            "settings": self.settings,
            "policy_config": self.engine.model.config.to_dict(),
            "critic_dimensions": list(self.critic.dimensions),
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        if (
            state.get("settings") != self.settings
            or state.get("policy_config") != self.engine.model.config.to_dict()
            or state.get("critic_dimensions") != list(self.critic.dimensions)
            or type(state.get("updates")) is not int
            or state["updates"] < 0
        ):
            raise ValueError("CQL configuration or update state mismatch")
        self.updates, self._incomplete = state["updates"], False
