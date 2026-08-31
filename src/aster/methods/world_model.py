"""World-model reconstruction, reward, continuation, KL, and imagined control objectives."""

import torch
from torch import nn
import torch.nn.functional as F
from ..core import LossTerm, LossBundle
from ..models.world import symlog, two_hot


class WorldModelObjective(nn.Module):
    def __init__(self, *, dynamics_weight=0.5, representation_weight=0.1, free_nats=1.0):
        super().__init__()
        if min(dynamics_weight, representation_weight, free_nats) < 0:
            raise ValueError("Negative world-model weight")
        self.dynamics_weight, self.representation_weight, self.free_nats = (
            dynamics_weight,
            representation_weight,
            free_nats,
        )

    def config_dict(self):
        return {
            "type": "rssm",
            "dynamics_weight": self.dynamics_weight,
            "representation_weight": self.representation_weight,
            "free_nats": self.free_nats,
        }

    def forward(self, model, batch):
        result = model(batch["observations"], batch["actions"], batch["is_first"])
        valid = batch.get("valid", torch.ones_like(batch["is_first"])).bool()
        post, prior = (
            model.log_probs(result["state"].logits),
            model.log_probs(result["prior_logits"]),
        )
        dynamics = (
            (post.detach().exp() * (post.detach() - prior)).sum((-1, -2)).clamp_min(self.free_nats)
        )
        representation = (
            (post.exp() * (post - prior.detach())).sum((-1, -2)).clamp_min(self.free_nats)
        )
        reconstruction = (
            (result["reconstruction_symlog"] - symlog(batch["observations"])).square().sum(-1)
        )
        targets = two_hot(symlog(batch["rewards"]), model.reward_support)
        reward = -(targets * result["reward_logits"].log_softmax(-1)).sum(-1)

        continuation = F.binary_cross_entropy_with_logits(
            result["continue_logits"], (~batch["terminated"]).to(reward.dtype), reduction="none"
        )
        values = [
            ("reconstruction", reconstruction, 1.0),
            ("reward", reward, 1.0),
            ("continue", continuation, 1.0),
            ("dynamics", dynamics, self.dynamics_weight),
            ("representation", representation, self.representation_weight),
        ]
        return LossBundle(
            tuple(
                LossTerm(
                    value.masked_select(valid).sum(),
                    valid.sum().to(value),
                    "transition",
                    name,
                    weight,
                )
                for name, value, weight in values
            )
        )


@torch.no_grad()
def cem_plan(
    world,
    initial,
    *,
    horizon=8,
    population=128,
    elites=16,
    iterations=4,
    discount=0.99,
    action_low=-1.0,
    action_high=1.0,
    generator=None,
):
    """Replan after each real observation and execute only the first selected action."""
    if (
        initial.deter.shape[0] != 1
        or not 1 <= elites <= population
        or min(horizon, iterations) < 1
        or not 0 < discount <= 1
        or action_low >= action_high
    ):
        raise ValueError("Invalid CEM planning configuration")
    d = world.config.action_dim
    mean = initial.deter.new_full((horizon, d), (action_low + action_high) / 2)
    std = initial.deter.new_full((horizon, d), (action_high - action_low) / 2)
    indices = torch.zeros(population, device=mean.device, dtype=torch.long)
    state = initial.reorder(indices)
    for _ in range(iterations):
        candidates = (
            mean
            + std * torch.randn(population, horizon, d, device=mean.device, generator=generator)
        ).clamp(action_low, action_high)
        imagined = world.imagine(state, candidates)
        predictions = world.predictions(imagined)
        continuation = predictions["continue_logits"].sigmoid() * discount
        weights = torch.cat(
            (torch.ones(population, 1, device=mean.device), continuation[:, :-1]), 1
        ).cumprod(1)
        scores = (weights * predictions["reward"]).sum(1)
        chosen = candidates[scores.topk(elites).indices]
        mean, std = (
            chosen.mean(0),
            chosen.std(0, correction=0).clamp_min(0.05 * (action_high - action_low)),
        )
    return mean[0], {"plan": mean, "predicted_return": float(scores.max())}


class ImaginedActorCritic:
    """Train actor/value roles inside an RSSM using explicitly supplied environment replay."""

    def __init__(
        self,
        engine,
        actor,
        value,
        *,
        horizon=8,
        discount=0.99,
        lam=0.95,
        entropy_weight=1e-3,
        actor_gradient="reinforce",
    ):
        if (
            horizon < 1
            or not 0 < discount <= 1
            or not 0 <= lam <= 1
            or entropy_weight < 0
            or actor_gradient not in {"reinforce", "dynamics"}
        ):
            raise ValueError("Invalid imagined actor-critic settings")
        self.engine = engine
        self.actor = engine.add_role("actor", actor)
        self.value = engine.add_role("value", value)
        self.horizon, self.discount, self.lam, self.entropy_weight, self.actor_gradient = (
            horizon,
            discount,
            lam,
            entropy_weight,
            actor_gradient,
        )
        self.updates = 0
        engine.register_state("imagined_actor_critic", self)

    def update(self, microbatches):
        from ..models.world import RSSMState
        from .reinforcement import lambda_returns

        batches = list(microbatches)
        world = self.engine.model
        world_result = self.engine.phase(
            "world", objective=WorldModelObjective(), microbatches=batches
        )
        starts = []
        with torch.no_grad():
            for batch in batches:
                sequence, _, _ = world.observe(
                    batch["observations"], batch["actions"], batch["is_first"]
                )
                valid = (
                    batch.get("valid", torch.ones_like(batch["is_first"])).bool()
                    & ~batch["terminated"]
                )
                if not valid.any():
                    raise ValueError("No nonterminal states available for imagination")
                state = RSSMState(
                    sequence.deter[valid].detach(),
                    sequence.stochastic[valid].detach(),
                    sequence.logits[valid].detach(),
                    sequence.config_key,
                )
                starts.append({"state": state})
        imagined_targets = []

        def actor_objective(actor, batch):
            state = batch["state"]
            features = [state.features]
            rewards, discounts, logps = [], [], []
            for _ in range(self.horizon):
                action, entropy_logp = actor(state.features.detach())
                policy_logp = (
                    actor.log_prob(state.features.detach(), action.detach())
                    if self.actor_gradient == "reinforce"
                    else entropy_logp
                )
                state, _ = world.step(state, action)
                prediction = world.predictions(state)
                features.append(state.features)
                rewards.append(prediction["reward"])
                discounts.append(prediction["continue_logits"].sigmoid() * self.discount)
                logps.append(policy_logp)
            features = torch.stack(features, 1)
            rewards, discounts, logps = map(
                lambda values: torch.stack(values, 1), (rewards, discounts, logps)
            )
            values = self.value(features).squeeze(-1)
            returns = lambda_returns(rewards, values, discounts, lam=self.lam)
            weights = (
                torch.cat((torch.ones_like(discounts[:, :1]), discounts[:, :-1]), 1)
                .cumprod(1)
                .detach()
            )
            if self.actor_gradient == "reinforce":
                advantages = (returns - values[:, :-1]).detach()
                objective = -(advantages * logps) - self.entropy_weight * (-logps)
            else:
                objective = -returns + self.entropy_weight * logps
            loss = (weights * objective).sum()
            imagined_targets.append(
                {
                    "features": features[:, :-1].detach(),
                    "returns": returns.detach(),
                    "weights": weights,
                }
            )
            return LossTerm(
                loss, loss.new_tensor(weights.numel()), "imagined_transition", "imagined_actor"
            )

        actor_result = self.engine.phase(
            "imagined_actor",
            role="actor",
            objective=actor_objective,
            microbatches=starts,
            freeze_roles=("model", "value"),
        )

        def value_objective(value, batch):
            errors = (
                0.5
                * (value(batch["features"]).squeeze(-1) - batch["returns"]).square()
                * batch["weights"]
            )
            return LossTerm(
                errors.sum(),
                errors.new_tensor(errors.numel()),
                "imagined_transition",
                "imagined_value",
            )

        value_result = self.engine.phase(
            "imagined_value",
            role="value",
            objective=value_objective,
            microbatches=imagined_targets,
            freeze_roles=("model", "actor"),
        )
        self.updates += 1
        return {"world": world_result, "actor": actor_result, "value": value_result}

    def state_dict(self):
        return {
            key: getattr(self, key)
            for key in ("horizon", "discount", "lam", "entropy_weight", "actor_gradient", "updates")
        }

    def load_state_dict(self, state):
        if any(state[key] != getattr(self, key) for key in state if key != "updates"):
            raise ValueError("Imagined method settings changed")
        self.updates = state["updates"]
