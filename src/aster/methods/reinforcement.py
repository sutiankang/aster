"""RL objectives and multi-role updates owned by the shared trainer."""

from __future__ import annotations
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..core import LossTerm, LossBundle
from .supervised import (
    sequence_logprobs,
    native_causal_config,
    preflight_causal_microbatches,
    supervision_mask,
)


def mlp(incoming, outgoing, hidden=128):
    return nn.Sequential(
        nn.Linear(incoming, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, outgoing),
    )


def generalized_advantage(
    rewards, values, next_values, terminated, truncated, *, gamma=0.99, lam=0.95, valid=None
):
    """Compute GAE for [B,T] trajectories.

    Terminal transitions do not bootstrap. Time-limit truncations bootstrap from
    the last observation but stop advantage propagation into the next episode."""
    if (
        not all(x.shape == rewards.shape for x in (values, next_values, terminated, truncated))
        or rewards.ndim != 2
        or not 0 <= gamma <= 1
        or not 0 <= lam <= 1
    ):
        raise ValueError("Invalid GAE inputs")
    valid = torch.ones_like(terminated, dtype=torch.bool) if valid is None else valid
    advantages = torch.zeros_like(rewards)
    following = torch.zeros_like(rewards[:, 0])
    for index in reversed(range(rewards.shape[1])):
        delta = (
            rewards[:, index]
            + gamma * (~terminated[:, index]) * next_values[:, index]
            - values[:, index]
        )
        continuation = ~(terminated[:, index] | truncated[:, index])
        following = (delta + gamma * lam * continuation * following) * valid[:, index]
        advantages[:, index] = following
    return advantages, advantages + values


def lambda_returns(rewards, values, discounts, *, lam=0.95):
    if (
        rewards.ndim != 2
        or discounts.shape != rewards.shape
        or values.shape != (len(rewards), rewards.shape[1] + 1)
        or not 0 <= lam <= 1
    ):
        raise ValueError("Invalid lambda return shapes")
    result = torch.empty_like(rewards)
    following = values[:, -1]
    for t in reversed(range(rewards.shape[1])):
        following = rewards[:, t] + discounts[:, t] * (
            (1 - lam) * values[:, t + 1] + lam * following
        )
        result[:, t] = following
    return result


class CategoricalActorCritic(nn.Module):
    def __init__(self, observation_dim, actions, hidden=64):
        super().__init__()
        self.policy, self.value = (
            mlp(observation_dim, actions, hidden),
            mlp(observation_dim, 1, hidden),
        )

    def forward(self, observations):
        return self.policy(observations), self.value(observations).squeeze(-1)

    def act(self, observations, *, deterministic=False):
        logits, value = self(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), value


class PPOObjective(nn.Module):
    def __init__(
        self,
        *,
        clip_ratio=0.2,
        value_clip=0.2,
        value_weight=0.5,
        entropy_weight=0.01,
        normalize_advantages=True,
    ):
        super().__init__()
        if (
            clip_ratio <= 0
            or value_clip is not None
            and value_clip <= 0
            or value_weight < 0
            or entropy_weight < 0
        ):
            raise ValueError("Invalid PPO configuration")
        (
            self.clip_ratio,
            self.value_clip,
            self.value_weight,
            self.entropy_weight,
            self.normalize_advantages,
        ) = clip_ratio, value_clip, value_weight, entropy_weight, normalize_advantages

    def config_dict(self):
        return {
            "type": "ppo",
            "clip_ratio": self.clip_ratio,
            "value_clip": self.value_clip,
            "value_weight": self.value_weight,
            "entropy_weight": self.entropy_weight,
            "normalize_advantages": self.normalize_advantages,
        }

    def forward(self, model, batch):
        logits, values = model(batch["observations"])
        distribution = torch.distributions.Categorical(logits=logits)
        log_probs = distribution.log_prob(batch["actions"])
        old = batch["old_log_probs"].detach()
        advantages = batch["advantages"].detach()
        valid = batch.get("valid", torch.ones_like(old, dtype=torch.bool))
        if self.normalize_advantages:
            selected = advantages.masked_select(valid)
            if len(selected) > 1:
                advantages = (advantages - selected.mean()) / selected.std(correction=1).clamp_min(
                    1e-8
                )
        ratio = (log_probs - old).exp()
        policy = -torch.minimum(
            ratio * advantages, ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        )
        value_error = (values - batch["returns"]).square()
        if self.value_clip is not None:
            clipped = batch["old_values"] + (values - batch["old_values"]).clamp(
                -self.value_clip, self.value_clip
            )
            value_error = torch.maximum(value_error, (clipped - batch["returns"]).square())
        count = valid.sum().to(values)
        return LossBundle(
            (
                LossTerm(policy.masked_select(valid).sum(), count, "transition", "policy"),
                LossTerm(
                    0.5 * value_error.masked_select(valid).sum(),
                    count,
                    "transition",
                    "value",
                    self.value_weight,
                ),
                LossTerm(
                    -distribution.entropy().masked_select(valid).sum(),
                    count,
                    "transition",
                    "entropy",
                    self.entropy_weight,
                ),
            )
        )


class DQNObjective(nn.Module):
    def __init__(self, target, *, gamma=0.99, double=True):
        super().__init__()
        self.target = target.eval().requires_grad_(False)
        self.gamma, self.double = gamma, double

    def forward(self, model, batch):
        prediction = (
            model(batch["observations"]).gather(-1, batch["actions"].long()[..., None]).squeeze(-1)
        )
        with torch.no_grad():
            next_q = self.target(batch["next_observations"])
            next_value = (
                next_q.gather(
                    -1, model(batch["next_observations"]).argmax(-1, keepdim=True)
                ).squeeze(-1)
                if self.double
                else next_q.max(-1).values
            )
            discounts = batch.get("discounts", self.gamma * (~batch["terminated"]))
            target = batch["rewards"] + discounts * next_value
        losses = F.smooth_l1_loss(prediction, target, reduction="none") * batch.get(
            "importance_weights", 1.0
        )
        return LossTerm(losses.sum(), losses.new_tensor(len(losses)), "transition", "td_error")


class GaussianActor(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden=128,
        *,
        action_low=-1.0,
        action_high=1.0,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()
        if action_low >= action_high or log_std_min >= log_std_max:
            raise ValueError("Invalid actor support")
        self.network = mlp(observation_dim, action_dim * 2, hidden)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.register_buffer(
            "action_scale", torch.full((action_dim,), (action_high - action_low) / 2)
        )
        self.register_buffer(
            "action_bias", torch.full((action_dim,), (action_high + action_low) / 2)
        )

    def forward(self, observations, *, deterministic=False):
        mean, raw_std = self.network(observations).chunk(2, -1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            raw_std.tanh() + 1
        )
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        squashed = raw.tanh()
        action = squashed * self.action_scale + self.action_bias

        log_jacobian = 2 * (math.log(2) - raw - F.softplus(-2 * raw)) + self.action_scale.log()
        logp = (distribution.log_prob(raw) - log_jacobian).sum(-1)
        return action, logp

    def log_prob(self, observations, actions):
        mean, raw_std = self.network(observations).chunk(2, -1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            raw_std.tanh() + 1
        )
        normalized = ((actions - self.action_bias) / self.action_scale).clamp(-1 + 1e-6, 1 - 1e-6)
        raw = torch.atanh(normalized)
        log_jacobian = 2 * (math.log(2) - raw - F.softplus(-2 * raw)) + self.action_scale.log()
        return (torch.distributions.Normal(mean, log_std.exp()).log_prob(raw) - log_jacobian).sum(
            -1
        )


class TwinQ(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden=128):
        super().__init__()
        self.dimensions = (observation_dim, action_dim, hidden)
        self.q1, self.q2 = (
            mlp(observation_dim + action_dim, 1, hidden),
            mlp(observation_dim + action_dim, 1, hidden),
        )

    def forward(self, observations, actions):
        inputs = torch.cat((observations, actions), -1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)


class EntropyTemperature(nn.Module):
    def __init__(self, initial=0.2):
        super().__init__()
        if initial <= 0:
            raise ValueError("Entropy temperature must be positive")
        self.log_alpha = nn.Parameter(torch.tensor(math.log(initial)))

    def forward(self):
        return self.log_alpha.exp()


class SACMethod:
    """Coordinate critic, actor, and temperature updates with the shared trainer.
    Frozen critic parameters must still transmit input gradients to the actor.
    Polyak target updates follow successful optimizer updates only."""

    def __init__(
        self,
        engine,
        critic,
        *,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy=True,
        target_entropy=None,
    ):
        if not 0 <= gamma <= 1 or not 0 < tau <= 1:
            raise ValueError("Invalid SAC parameters")
        self.engine, self.gamma, self.tau = engine, gamma, tau
        dimensions = critic.dimensions
        self.critic = engine.add_role("critic", critic)
        self.target = engine.clone_target(
            "critic", "target_critic", factory=lambda: TwinQ(*dimensions)
        )
        self.temperature = engine.add_role(
            "temperature", EntropyTemperature(alpha), trainable=automatic_entropy
        )
        self.automatic_entropy = automatic_entropy
        self.target_entropy = (
            -len(engine.model.action_scale) if target_entropy is None else target_entropy
        )
        self.updates = 0
        engine.register_state("sac_method", self)

    def update(self, microbatches):
        batches = list(microbatches)
        actor = self.engine.model

        def critic_objective(critic, batch):
            with torch.no_grad():
                action, logp = actor(batch["next_observations"])
                q1, q2 = self.target(batch["next_observations"], action)
                discounts = batch.get("discounts", self.gamma * (~batch["terminated"]))
                target = batch["rewards"] + discounts * (
                    torch.minimum(q1, q2) - self.temperature() * logp
                )
            q1, q2 = critic(batch["observations"], batch["actions"])
            values = ((q1 - target).square() + (q2 - target).square()) * batch.get(
                "importance_weights", 1.0
            )
            return LossTerm(
                values.sum(), values.new_tensor(len(values)), "transition", "sac_critic"
            )

        critic_result = self.engine.phase(
            "critic",
            role="critic",
            objective=critic_objective,
            microbatches=batches,
            freeze_roles=("model", "temperature"),
        )

        def actor_objective(policy, batch):
            actions, logp = policy(batch["observations"])

            q1, q2 = self.critic(batch["observations"], actions)
            values = self.temperature().detach() * logp - torch.minimum(q1, q2)
            return LossTerm(values.sum(), values.new_tensor(len(values)), "transition", "sac_actor")

        actor_result = self.engine.phase(
            "actor",
            objective=actor_objective,
            microbatches=batches,
            freeze_roles=("critic", "temperature"),
        )
        temperature_result = None
        if self.automatic_entropy:

            def temperature_objective(module, batch):
                with torch.no_grad():
                    _, logp = actor(batch["observations"])
                values = -module.log_alpha * (logp + self.target_entropy)
                return LossTerm(
                    values.sum(),
                    values.new_tensor(len(values)),
                    "transition",
                    "entropy_temperature",
                )

            temperature_result = self.engine.phase(
                "temperature",
                role="temperature",
                objective=temperature_objective,
                microbatches=batches,
                freeze_roles=("model", "critic"),
            )
        if critic_result.updated:
            self.engine.update_target("critic", "target_critic", 1 - self.tau)
        self.updates += 1
        return {"critic": critic_result, "actor": actor_result, "temperature": temperature_result}

    def state_dict(self):
        return {
            "gamma": self.gamma,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
            "automatic_entropy": self.automatic_entropy,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        if any(
            state[key] != getattr(self, key)
            for key in ("gamma", "tau", "target_entropy", "automatic_entropy")
        ):
            raise ValueError("SAC method configuration mismatch")
        self.updates = state["updates"]


def group_relative_advantages(rewards, group_ids, *, normalize=True, epsilon=1e-4):
    if rewards.ndim != 1 or group_ids.shape != rewards.shape or not torch.isfinite(rewards).all():
        raise ValueError("Invalid grouped rewards")
    result = torch.empty_like(rewards)
    for group in group_ids.unique():
        selected = group_ids == group
        values = rewards[selected]
        if len(values) < 2:
            raise ValueError("GRPO needs at least two completions per prompt")
        result[selected] = (values - values.mean()) / (
            values.std(correction=1) + epsilon if normalize else 1.0
        )
    return result


class GRPOObjective(nn.Module):
    def __init__(
        self,
        *,
        clip_low=0.2,
        clip_high=0.2,
        kl_weight=0.04,
        reduction="sequence",
        max_completion_length=None,
    ):
        super().__init__()
        if (
            min(clip_low, clip_high, kl_weight) < 0
            or reduction not in {"sequence", "token", "constant"}
            or reduction == "constant"
            and not max_completion_length
        ):
            raise ValueError("Invalid GRPO reduction")
        (
            self.clip_low,
            self.clip_high,
            self.kl_weight,
            self.reduction,
            self.max_completion_length,
        ) = clip_low, clip_high, kl_weight, reduction, max_completion_length

    def config_dict(self):

        return {
            "type": "grpo",
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
            "kl_weight": self.kl_weight,
            "reduction": self.reduction,
            "max_completion_length": self.max_completion_length,
        }

    @torch.no_grad()
    def preflight_microbatches(self, policy, batches):
        """Validate the complete accumulation window and all model roles used by this
        objective before forward or sharded parameter communication."""
        if (
            any(
                not math.isfinite(x) or x < 0
                for x in (self.clip_low, self.clip_high, self.kl_weight)
            )
            or self.reduction not in {"sequence", "token", "constant"}
            or (
                self.reduction == "constant"
                and (type(self.max_completion_length) is not int or self.max_completion_length < 1)
            )
        ):
            raise ValueError("Invalid declared GRPO clipping/reduction configuration")
        batches = preflight_causal_microbatches(policy, batches)
        if native_causal_config(policy) is None:
            return batches
        for batch in batches:
            labels = batch.get("labels", batch.get("input_ids"))
            valid = supervision_mask(batch, labels)[:, 1:]
            for name, shape in (
                ("old_behavior_log_probs", valid.shape),
                ("reference_log_probs", valid.shape),
                ("advantages", (len(valid),)),
            ):
                value = batch.get(name)
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != shape
                    or value.device != labels.device
                    or not value.is_floating_point()
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(
                        name
                        + " must be finite floating data aligned to the actual next-token trajectory"
                    )
            lengths = valid.sum(-1)
            if (lengths == 0).any():
                raise ValueError("GRPO empty completions require explicit rejection accounting")
            if self.reduction == "constant" and (lengths > self.max_completion_length).any():
                raise ValueError("Completion exceeds fixed Dr.GRPO denominator")
        return batches

    def forward(self, policy, batch):
        logp, valid = sequence_logprobs(policy, batch)
        old = batch["old_behavior_log_probs"].detach()
        if old.shape != logp.shape or batch["reference_log_probs"].shape != logp.shape:
            raise ValueError("Rollout log-probs must exactly align sampled completion tokens")
        advantage = batch["advantages"].detach()[:, None]
        ratio = (logp - old).exp()
        surrogate = torch.minimum(
            ratio * advantage, ratio.clamp(1 - self.clip_low, 1 + self.clip_high) * advantage
        )
        difference = batch["reference_log_probs"].detach() - logp
        kl = difference.exp() - difference - 1
        losses = (-surrogate + self.kl_weight * kl) * valid
        lengths = valid.sum(-1)
        if (lengths == 0).any():
            raise ValueError("GRPO empty completions require explicit rejection accounting")
        if self.reduction == "sequence":
            numerator, denominator, unit = (
                (losses.sum(-1) / lengths).sum(),
                len(losses),
                "completion",
            )
        elif self.reduction == "token":
            numerator, denominator, unit = losses.sum(), valid.sum(), "token"
        else:
            if (lengths > self.max_completion_length).any():
                raise ValueError("Completion exceeds fixed Dr.GRPO denominator")
            numerator, denominator, unit = (
                losses.sum(),
                len(losses) * self.max_completion_length,
                "completion_slot",
            )
        return LossTerm(
            numerator,
            numerator.new_tensor(denominator)
            if not isinstance(denominator, torch.Tensor)
            else denominator.to(numerator),
            unit,
            "grpo",
        )
