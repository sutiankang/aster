"""MuZero unroll supervision and n-step targets through the shared trainer."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from ..core import LossTerm, LossBundle
from ..models.muzero import support_targets


def nstep_value_targets(rewards, values, terminated, *, discount=0.997, n_steps=10, truncated=None):
    """Reward r[t] belongs to s[t] -> s[t+1]. Bootstrap truncations from their final
    observation, but do not bootstrap terminated episodes."""
    if (
        rewards.ndim != 2
        or values.shape != (len(rewards), rewards.shape[1] + 1)
        or terminated.shape != rewards.shape
        or terminated.dtype != torch.bool
        or type(n_steps) is not int
        or n_steps < 1
        or not 0 <= discount <= 1
        or not math.isfinite(discount)
    ):
        raise ValueError("Invalid MuZero n-step trajectory arrays/settings")
    truncated = torch.zeros_like(terminated) if truncated is None else truncated
    if truncated.shape != rewards.shape or truncated.dtype != torch.bool:
        raise ValueError("truncated must be separate B,T boolean reset metadata")
    if not all(torch.isfinite(x).all() for x in (rewards, values)) or any(
        x.device != rewards.device for x in (values, terminated, truncated)
    ):
        raise ValueError("Trajectory tensors must be finite and share a device")
    targets = torch.empty_like(values)
    targets[:, -1] = values[:, -1]
    for start in range(rewards.shape[1]):
        active = torch.ones(len(rewards), dtype=torch.bool, device=rewards.device)
        result = rewards.new_zeros(len(rewards))
        weight = 1.0
        end = min(start + n_steps, rewards.shape[1])
        for step in range(start, end):
            result += weight * rewards[:, step] * active
            weight *= discount

            active = active & ~terminated[:, step]
            cutoff = active & truncated[:, step]
            result += weight * values[:, step + 1] * cutoff
            active = active & ~cutoff
        targets[:, start] = result + weight * values[:, end] * active
    return targets


class MuZeroObjective(nn.Module):
    def __init__(self, *, value_weight=1.0, policy_weight=1.0, reward_weight=1.0):
        super().__init__()
        self.weights = dict(value=value_weight, policy=policy_weight, reward=reward_weight)
        if not all(math.isfinite(x) and x >= 0 for x in self.weights.values()) or not any(
            self.weights.values()
        ):
            raise ValueError("MuZero head weights must be finite, nonnegative and not all zero")

    def config_dict(self):
        return dict(
            type="muzero_unroll",
            weights=self.weights,
            gradient_normalization="per_K_heads",
            reward_coordinates="transition_into_next_state",
        )

    def validate(self, model, batch):
        obs, actions, policy, value, reward = (
            batch[key]
            for key in (
                "observations",
                "actions",
                "policy_targets",
                "value_targets",
                "reward_targets",
            )
        )
        if actions.ndim != 2 or min(actions.shape) < 1:
            raise ValueError("MuZero needs nonempty B,K action unrolls")
        count, horizon = actions.shape
        if (
            obs.shape != (count, model.config.observation_dim)
            or policy.shape != (count, horizon + 1, model.config.num_actions)
            or value.shape != (count, horizon + 1)
            or reward.shape != (count, horizon)
            or actions.dtype not in (torch.int32, torch.int64)
            or (actions < 0).any()
            or (actions >= model.config.num_actions).any()
        ):
            raise ValueError("MuZero target coordinates/dimensions/actions do not align")
        if not all(
            x.is_floating_point() and torch.isfinite(x).all() for x in (obs, policy, value, reward)
        ):
            raise ValueError("MuZero observations and targets must be finite floating tensors")
        valid = batch.get("valid", torch.ones_like(value, dtype=torch.bool))
        if valid.shape != value.shape or valid.dtype != torch.bool or not valid[:, 0].all():
            raise ValueError("MuZero valid mask must be B,K+1 with valid initial states")
        if ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError("MuZero validity cannot cross an episode/reset boundary")
        reward_valid = batch.get("reward_valid", valid[:, :-1])
        if (
            reward_valid.shape != reward.shape
            or reward_valid.dtype != torch.bool
            or reward_valid.device != obs.device
            or (reward_valid & ~valid[:, :-1]).any()
            or ((~reward_valid[:, :-1]) & reward_valid[:, 1:]).any()
        ):
            raise ValueError(
                "MuZero reward_valid must mark a prefix of actually observed transitions"
            )
        sums = policy.sum(-1)
        if (policy < 0).any() or not torch.allclose(
            sums[valid], torch.ones_like(sums[valid]), atol=1e-5, rtol=1e-5
        ):
            raise ValueError("Each valid search policy target must be a probability distribution")
        importance = batch.get("importance_weights", value.new_ones(count))
        if (
            importance.shape != (count,)
            or not torch.isfinite(importance).all()
            or (importance < 0).any()
        ):
            raise ValueError("Replay importance weights must be finite nonnegative B values")
        if any(x.device != obs.device for x in (actions, policy, value, reward, valid, importance)):
            raise ValueError("Prepare all MuZero trajectory tensors on one device")
        return valid, importance

    def forward(self, model, batch):
        valid, importance = self.validate(model, batch)
        horizon = batch["actions"].shape[1]
        predictions = model(batch["observations"], batch["actions"])
        losses = {key: [] for key in self.weights}
        for step, prediction in enumerate(predictions):
            value_target = support_targets(
                batch["value_targets"][:, step],
                model.config.support_size,
                model.config.transform_epsilon,
            )
            losses["value"].append(
                -(value_target * F.log_softmax(prediction.value_logits.float(), -1)).sum(-1)
            )
            losses["policy"].append(
                -(
                    batch["policy_targets"][:, step]
                    * F.log_softmax(prediction.prior_logits.float(), -1)
                ).sum(-1)
            )
            if step:
                reward_target = support_targets(
                    batch["reward_targets"][:, step - 1],
                    model.config.support_size,
                    model.config.transform_epsilon,
                )
                losses["reward"].append(
                    -(reward_target * F.log_softmax(prediction.reward_logits.float(), -1)).sum(-1)
                )
        terms = []
        for name, sequence in losses.items():
            errors = torch.stack(sequence, 1)
            mask = valid if name != "reward" else batch.get("reward_valid", valid[:, :-1])
            # Reward at a terminal transition is still observed even if next state is padded.
            # All heads scale by K, not separate valid-step means that change episode weights.
            numerator = (errors * mask * importance[:, None]).sum() / horizon
            terms.append(
                LossTerm(
                    numerator,
                    torch.tensor(len(valid), dtype=torch.int64, device=valid.device),
                    "trajectory",
                    name,
                    self.weights[name],
                )
            )
        return LossBundle(tuple(terms))


class MuZeroMethod:
    """Bind target models, replay, and search RNG to shared checkpoint boundaries."""

    def __init__(self, engine, objective=None):
        self.engine = engine
        self.objective = objective or MuZeroObjective()
        if any(
            getattr(engine.parallel.config, key, 1) > 1
            for key in (
                "tensor_parallel",
                "pipeline_parallel",
                "context_parallel",
                "gtp_remat",
                "expert_parallel",
                "expert_tensor_parallel",
            )
        ):
            raise ValueError(
                "MuZero vector learner currently uses DP/ZeRO, not implicit model sharding"
            )
        self.updates = 0
        engine.register_state("muzero_method", self)

    def update(self, microbatches):
        error, batches, layout = None, None, None
        try:
            batches = list(microbatches)
            if len(batches) != self.engine.accumulation_steps:
                raise ValueError("MuZero unroll microbatches must fill the accumulation window")
            for batch in batches:
                self.objective.validate(self.engine.model, batch)
                if batch["observations"].device != self.engine.device:
                    raise ValueError(
                        "Move MuZero unroll tensors to the Trainer device before update"
                    )
            layout = [batch["actions"].shape[1] for batch in batches]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        declarations = self.engine.parallel.world.gather_objects((error, layout))
        if any(value[0] for value in declarations) or any(
            value[1] != layout for value in declarations
        ):
            raise ValueError("MuZero collective preflight/horizon mismatch: " + str(declarations))
        result = self.engine.phase("muzero", objective=self.objective, microbatches=batches)
        if result.updated:
            self.updates += 1
        return result

    def state_dict(self):
        return {
            "objective": self.objective.config_dict(),
            "model": self.engine.model.config.to_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        if (
            state.get("objective") != self.objective.config_dict()
            or state.get("model") != self.engine.model.config.to_dict()
            or type(state.get("updates")) is not int
            or state["updates"] < 0
        ):
            raise ValueError("MuZero method checkpoint configuration mismatch")
        self.updates = state["updates"]
