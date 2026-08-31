"""Native TD3, TD3+BC, and IQL with shared multi-role checkpoint ownership."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from collections.abc import Mapping
import torch
from torch import nn
from ..core import LossTerm


def relu_network(inputs, outputs, hidden):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, outputs),
    )


@dataclass(frozen=True)
class _MLPConfig:
    kind: str
    observation_dim: int
    action_dim: int
    hidden: int
    max_action: float = 1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    def to_dict(self):
        return asdict(self)


def _dimensions(*values):
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("MLP dimensions must be positive integers")


class DeterministicActor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden=256, max_action=1.0):
        super().__init__()
        _dimensions(observation_dim, action_dim, hidden)
        if not math.isfinite(max_action) or max_action <= 0:
            raise ValueError("Invalid actor action support")
        self.config = _MLPConfig(
            "td3_actor", observation_dim, action_dim, hidden, max_action=max_action
        )
        self.dimensions = (observation_dim, action_dim, hidden, max_action)
        self.network = relu_network(observation_dim, action_dim, hidden)
        self.max_action = max_action

    def forward(self, observations):
        return self.max_action * self.network(observations).tanh()


class ContinuousTwinQ(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden=256):
        super().__init__()
        _dimensions(observation_dim, action_dim, hidden)
        self.config = _MLPConfig("twin_q", observation_dim, action_dim, hidden)
        self.dimensions = (observation_dim, action_dim, hidden)
        self.q1, self.q2 = (
            relu_network(observation_dim + action_dim, 1, hidden),
            relu_network(observation_dim + action_dim, 1, hidden),
        )

    def forward(self, observations, actions):
        inputs = torch.cat((observations, actions), -1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)


class _StateIndependentLogStd(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.values = nn.Parameter(torch.zeros(action_dim))

    def forward(self):
        return self.values


class IQLActor(nn.Module):
    """Use a tanh mean and state-independent standard deviation. The Normal
    distribution itself is not tanh-transformed."""

    def __init__(
        self, observation_dim, action_dim, hidden=256, *, log_std_min=-5.0, log_std_max=2.0
    ):
        super().__init__()
        _dimensions(observation_dim, action_dim, hidden)
        if (
            not math.isfinite(log_std_min)
            or not math.isfinite(log_std_max)
            or log_std_min >= log_std_max
        ):
            raise ValueError("Finite ordered IQL log standard-deviation bounds required")
        self.config = _MLPConfig(
            "iql_actor",
            observation_dim,
            action_dim,
            hidden,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self.network = relu_network(observation_dim, action_dim, hidden)
        self.log_std = _StateIndependentLogStd(action_dim)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    @property
    def log_stds(self):

        return self.log_std.values

    def distribution(self, observations):
        mean = self.network(observations).tanh().float()
        std = self.log_std().float().clamp(self.log_std_min, self.log_std_max).exp()
        return torch.distributions.Normal(mean, std)

    def log_prob(self, observations, actions):
        return self.distribution(observations).log_prob(actions).sum(-1)

    def forward(self, observations, *, deterministic=False):
        distribution = self.distribution(observations)
        return (distribution.mean if deterministic else distribution.sample()).clamp(-1, 1)


class StateValue(nn.Module):
    def __init__(self, observation_dim, hidden=256):
        super().__init__()
        _dimensions(observation_dim, hidden)
        self.config = _MLPConfig("value", observation_dim, 0, hidden)
        self.network = relu_network(observation_dim, 1, hidden)

    def forward(self, observations):
        return self.network(observations).squeeze(-1)


def expectile_loss(difference, expectile=0.8):
    if not 0 < expectile < 1:
        raise ValueError("Expectile must lie in (0,1)")
    return torch.where(difference > 0, expectile, 1 - expectile) * difference.square()


def advantage_weighted_bc(log_prob, advantage, *, inverse_temperature=3.0, max_weight=100.0):
    if (
        not all(math.isfinite(value) for value in (inverse_temperature, max_weight))
        or inverse_temperature < 0
        or max_weight <= 0
        or log_prob.shape != advantage.shape
    ):
        raise ValueError("Invalid AWR inputs")

    weights = (inverse_temperature * advantage.detach()).clamp_max(math.log(max_weight)).exp()
    return -weights * log_prob


def _term(values, name):
    if values.ndim != 1:
        raise ValueError("Each continuous-control batch must have one scalar loss per transition")

    return LossTerm(
        values.float().sum(),
        torch.tensor(len(values), device=values.device, dtype=torch.int64),
        "transition",
        name,
    )


def _validate_batch(batch, *, observation_dim=None, action_dim=None, max_action=None, device=None):

    required = {"observations", "actions", "rewards", "next_observations", "terminated"}
    if (
        not isinstance(batch, Mapping)
        or not required <= batch.keys()
        or set(batch) - required - {"truncated", "discounts"}
    ):
        raise ValueError("Offline batch has missing or unknown transition fields")
    if any(not isinstance(value, torch.Tensor) for value in batch.values()):
        raise TypeError("Every offline transition field must be a Tensor")
    if device is not None and any(value.device != device for value in batch.values()):
        raise ValueError("Prepare every transition tensor on the Trainer device before update")
    obs, actions = batch["observations"], batch["actions"]
    if obs.ndim != 2 or actions.ndim != 2:
        raise ValueError("Vector MLP expects observations[B,O] and actions[B,A]")
    count = len(obs)
    if (
        batch["rewards"].shape != (count,)
        or batch["terminated"].shape != (count,)
        or batch["terminated"].dtype != torch.bool
    ):
        raise ValueError(
            "Rewards/terminated must be aligned B vectors; truncation is not termination"
        )
    if batch["next_observations"].shape != obs.shape or len(actions) != count:
        raise ValueError("Transition observations/actions do not align")
    if (
        observation_dim is not None
        and obs.shape[1] != observation_dim
        or action_dim is not None
        and actions.shape[1] != action_dim
    ):
        raise ValueError("Replay feature dimensions do not match policy/critic configuration")
    for key in ("observations", "actions", "rewards", "next_observations"):
        if not batch[key].is_floating_point() or not torch.isfinite(batch[key]).all():
            raise ValueError(f"{key} must contain finite floating values")
    if max_action is not None and (actions.abs() > max_action).any():
        raise ValueError("Replay action lies outside the declared policy support")
    if "truncated" in batch and (
        batch["truncated"].shape != (count,) or batch["truncated"].dtype != torch.bool
    ):
        raise ValueError("truncated must be an aligned boolean B vector")
    if "discounts" in batch:
        discount = batch["discounts"]
        if (
            discount.shape != (count,)
            or not discount.is_floating_point()
            or not torch.isfinite(discount).all()
            or ((discount < 0) | (discount > 1)).any()
        ):
            raise ValueError("Explicit discounts must be finite B vectors in [0,1]")
        if (discount[batch["terminated"]] != 0).any():
            raise ValueError("A terminated transition cannot have a nonzero bootstrap discount")


def _collective_check(engine, error, declaration, name):
    records = engine.parallel.world.gather_objects((error, declaration))
    if any(item[0] for item in records):
        raise ValueError(f"{name} collective preflight failed: {[item[0] for item in records]}")
    if any(item[1] != declaration for item in records):
        raise ValueError(f"{name} ranks disagree on settings/update cursor/microbatch call order")


def _topology(engine):
    if any(
        getattr(engine.parallel, axis).size != 1
        for axis in ("tp", "pp", "cp", "gtp_remat", "ep", "etp")
    ):
        raise ValueError(
            "Native vector offline methods support DP/ZeRO0-3 only; TP/PP/SP/CP/EP/GTP need explicit model-parallel policy providers"
        )


def _role_contract(engine):

    declaration = (
        engine.precision,
        engine.zero_stage,
        engine.accumulation_steps,
        engine.max_grad_norm,
        {name: role.optimizer_identity for name, role in engine.roles.items() if role.trainable},
    )
    _collective_check(engine, None, declaration, "Offline role ownership")


class _OfflineMethod:
    """Coordinate multiple phases without pretending they form a rollback transaction.
    After a partial update fails, restore the last complete checkpoint."""

    def _preflight(self, microbatches):
        error, declaration, batches = None, None, None
        try:
            if self._incomplete:
                raise RuntimeError(
                    "Previous multi-role update incomplete; restore last complete checkpoint"
                )
            if any(getattr(self, key) != value for key, value in self.settings.items()):
                raise ValueError(
                    "Do not mutate offline settings in-place; construct an explicit new method"
                )
            _topology(self.engine)
            batches = list(microbatches)
            if len(batches) != self.engine.accumulation_steps:
                raise ValueError("One microbatch is required for every accumulation slot")
            config = self.engine.model.config
            allowed_dtypes = {next(self.engine.model.parameters()).dtype}
            if self.engine.precision != "fp32":
                allowed_dtypes.add(
                    torch.bfloat16 if self.engine.precision == "bf16" else torch.float16
                )
            for batch in batches:
                _validate_batch(
                    batch,
                    observation_dim=config.observation_dim,
                    action_dim=config.action_dim,
                    max_action=config.max_action,
                    device=self.engine.device,
                )
                if any(
                    batch[key].dtype not in allowed_dtypes
                    for key in ("observations", "next_observations", "actions")
                ):
                    raise ValueError(
                        "Observation/action dtype must match model storage or explicit autocast precision"
                    )
            declaration = (self.settings, self.updates, len(batches))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _collective_check(self.engine, error, declaration, type(self).__name__)

        counts = self.engine.replica_group.gather_objects(
            sum(len(batch["actions"]) for batch in batches)
        )
        if sum(counts) == 0:
            raise ValueError("Offline update needs at least one global transition")
        return batches

    def _require_update(self, result, name):
        if not result.updated:
            raise RuntimeError(
                f"{name} phase skipped during multi-role update; restore last complete checkpoint"
            )

    def state_dict(self):
        if self._incomplete:
            raise RuntimeError("Cannot checkpoint an incomplete offline multi-role update")
        if any(getattr(self, key) != value for key, value in self.settings.items()):
            raise ValueError(
                "Offline settings changed outside the declared checkpoint configuration"
            )
        return {"schema_version": 1, "settings": dict(self.settings), "updates": self.updates}

    def load_state_dict(self, state):
        if (
            set(state) != {"schema_version", "settings", "updates"}
            or state["schema_version"] != 1
            or state["settings"] != self.settings
        ):
            raise ValueError("Offline method configuration mismatch")
        if type(state["updates"]) is not int or state["updates"] < 0:
            raise ValueError("Invalid offline update cursor")
        self.updates, self._incomplete = state["updates"], False


class TD3Method(_OfflineMethod):
    """Twin-Q learning with target-action smoothing and delayed policy updates.
    An explicit bc_alpha enables TD3+BC without changing the Q objective."""

    def __init__(
        self,
        engine,
        critic,
        *,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        bc_alpha=None,
        critic_optimizer=None,
        critic_optimizer_factory=None,
    ):
        error, declaration = None, None
        self.settings = dict(
            gamma=gamma,
            tau=tau,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_delay=policy_delay,
            bc_alpha=bc_alpha,
        )
        try:
            if (
                not all(math.isfinite(value) for value in (gamma, tau, policy_noise, noise_clip))
                or not 0 <= gamma <= 1
                or not 0 < tau <= 1
                or min(policy_noise, noise_clip) < 0
            ):
                raise ValueError("Invalid finite TD3 settings")
            if (
                type(policy_delay) is not int
                or policy_delay < 1
                or bc_alpha is not None
                and (not math.isfinite(bc_alpha) or bc_alpha <= 0)
            ):
                raise ValueError("Invalid TD3 policy delay/BC coefficient")
            _topology(engine)
            if type(engine.model) is not DeterministicActor or type(critic) is not ContinuousTwinQ:
                raise TypeError(
                    "TD3 currently requires its native DeterministicActor/ContinuousTwinQ providers"
                )
            if engine.model.dimensions[:2] != critic.dimensions[:2]:
                raise ValueError("Actor/critic dimensions differ")
            if critic_optimizer is not None and critic_optimizer_factory is not None:
                raise ValueError("Specify optimizer or factory, not both")
            declaration = (self.settings, engine.model.config.to_dict(), critic.config.to_dict())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _collective_check(engine, error, declaration, "TD3 constructor")
        self.engine, self.gamma, self.tau = engine, gamma, tau
        self.policy_noise, self.noise_clip, self.policy_delay, self.bc_alpha = (
            policy_noise,
            noise_clip,
            policy_delay,
            bc_alpha,
        )

        if critic_optimizer is None and critic_optimizer_factory is None:
            critic_optimizer_factory = lambda parameters: torch.optim.Adam(parameters, lr=engine.lr)
        self.critic = engine.add_role(
            "critic", critic, optimizer=critic_optimizer, optimizer_factory=critic_optimizer_factory
        )
        _role_contract(engine)
        self.target_critic = engine.clone_target(
            "critic", "target_critic", factory=lambda: ContinuousTwinQ(*critic.dimensions)
        )
        self.target_actor = engine.clone_target(
            "model", "target_actor", factory=lambda: DeterministicActor(*engine.model.dimensions)
        )
        self.updates, self._incomplete = 0, False
        engine.register_state("td3_method", self)

    def update(self, microbatches):
        batches = self._preflight(microbatches)
        self._incomplete = True

        def critic_loss(critic, batch):
            with torch.no_grad():
                noise = (
                    (torch.randn_like(batch["actions"]) * self.policy_noise).clamp(
                        -self.noise_clip, self.noise_clip
                    )
                    if self.policy_noise
                    else 0.0
                )
                action = (self.target_actor(batch["next_observations"]) + noise).clamp(
                    -self.target_actor.max_action, self.target_actor.max_action
                )
                q1, q2 = self.target_critic(batch["next_observations"], action)
                target = batch["rewards"].float() + batch.get(
                    "discounts", self.gamma * (~batch["terminated"])
                ) * torch.minimum(q1.float(), q2.float())
            q1, q2 = critic(batch["observations"], batch["actions"])
            return _term(
                (q1.float() - target).square() + (q2.float() - target).square(), "td3_critic"
            )

        critic_result = self.engine.phase(
            "td3_critic",
            role="critic",
            objective=critic_loss,
            microbatches=batches,
            freeze_roles=("model",),
        )
        result = {"critic": critic_result, "actor": None}
        self._require_update(critic_result, "td3_critic")
        next_update = self.updates + 1
        if next_update % self.policy_delay:
            self.updates, self._incomplete = next_update, False
            return result
        scale = 1.0
        if self.bc_alpha is not None:
            with (
                torch.no_grad(),
                torch.autocast(
                    self.engine.device.type,
                    dtype=torch.bfloat16 if self.engine.precision == "bf16" else torch.float16,
                    enabled=self.engine.precision != "fp32",
                ),
            ):
                statistics = torch.zeros(2, device=self.engine.device, dtype=torch.float64)
                for batch in batches:
                    q1, _ = self.critic(
                        batch["observations"], self.engine.model(batch["observations"])
                    )
                    statistics += torch.stack(
                        (q1.abs().double().sum(), q1.new_tensor(q1.numel(), dtype=torch.float64))
                    )
                self.engine.replica_group.all_reduce(statistics)
                if not torch.isfinite(statistics).all() or statistics[0] <= 0 or statistics[1] <= 0:
                    raise ValueError(
                        "TD3+BC undefined Q scaling at zero/nonfinite |Q|; restore last complete checkpoint"
                    )
                scale = (self.bc_alpha * statistics[1] / statistics[0]).float()

        def actor_loss(policy, batch):
            actions = policy(batch["observations"])
            q1, _ = self.critic(batch["observations"], actions)
            values = -scale * q1.float()
            if self.bc_alpha is not None:
                values = values + (actions.float() - batch["actions"].float()).square().mean(-1)
            return _term(values, "td3_actor")

        result["actor"] = self.engine.phase(
            "td3_actor", objective=actor_loss, microbatches=batches, freeze_roles=("critic",)
        )
        self._require_update(result["actor"], "td3_actor")
        self.engine.update_target("model", "target_actor", 1 - self.tau)
        self.engine.update_target("critic", "target_critic", 1 - self.tau)
        self.updates, self._incomplete = next_update, False
        return result


class IQLMethod(_OfflineMethod):
    """Apply expectile value fitting, advantage-weighted behavior cloning, Q learning,
    and target EMA in the declared update order."""

    def __init__(
        self,
        engine,
        critic,
        value,
        *,
        gamma=0.99,
        tau=0.005,
        expectile=0.8,
        inverse_temperature=3.0,
        max_weight=100.0,
        critic_optimizer=None,
        critic_optimizer_factory=None,
        value_optimizer=None,
        value_optimizer_factory=None,
    ):
        error, declaration = None, None
        self.settings = dict(
            gamma=gamma,
            tau=tau,
            expectile=expectile,
            inverse_temperature=inverse_temperature,
            max_weight=max_weight,
        )
        try:
            if (
                not all(
                    math.isfinite(item)
                    for item in (gamma, tau, expectile, inverse_temperature, max_weight)
                )
                or not 0 <= gamma <= 1
                or not 0 < tau <= 1
                or not 0 < expectile < 1
                or inverse_temperature < 0
                or max_weight <= 0
            ):
                raise ValueError("Invalid finite IQL settings")
            _topology(engine)
            if (
                type(engine.model) is not IQLActor
                or type(critic) is not ContinuousTwinQ
                or type(value) is not StateValue
            ):
                raise TypeError(
                    "IQL currently requires its native IQLActor/ContinuousTwinQ/StateValue providers"
                )
            config = engine.model.config
            if (config.observation_dim, config.action_dim) != critic.dimensions[
                :2
            ] or config.observation_dim != value.config.observation_dim:
                raise ValueError("Actor/critic/value dimensions differ")
            if (
                critic_optimizer is not None
                and critic_optimizer_factory is not None
                or value_optimizer is not None
                and value_optimizer_factory is not None
            ):
                raise ValueError("Specify optimizer or factory, not both")
            declaration = (
                self.settings,
                config.to_dict(),
                critic.config.to_dict(),
                value.config.to_dict(),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _collective_check(engine, error, declaration, "IQL constructor")
        self.engine = engine
        self.gamma, self.tau, self.expectile = gamma, tau, expectile
        self.inverse_temperature, self.max_weight = inverse_temperature, max_weight
        if critic_optimizer is None and critic_optimizer_factory is None:
            critic_optimizer_factory = lambda parameters: torch.optim.Adam(parameters, lr=engine.lr)
        if value_optimizer is None and value_optimizer_factory is None:
            value_optimizer_factory = lambda parameters: torch.optim.Adam(parameters, lr=engine.lr)
        self.critic = engine.add_role(
            "critic", critic, optimizer=critic_optimizer, optimizer_factory=critic_optimizer_factory
        )
        self.value = engine.add_role(
            "value", value, optimizer=value_optimizer, optimizer_factory=value_optimizer_factory
        )
        _role_contract(engine)
        self.target_critic = engine.clone_target(
            "critic", "target_critic", factory=lambda: ContinuousTwinQ(*critic.dimensions)
        )
        self.updates, self._incomplete = 0, False
        engine.register_state("iql_method", self)

    def update(self, microbatches):
        batches = self._preflight(microbatches)
        self._incomplete = True

        def value_loss(value, batch):
            with torch.no_grad():
                q1, q2 = self.target_critic(batch["observations"], batch["actions"])
                target = torch.minimum(q1.float(), q2.float())
            return _term(
                expectile_loss(target - value(batch["observations"]).float(), self.expectile),
                "iql_value",
            )

        result = {
            "value": self.engine.phase(
                "iql_value",
                role="value",
                objective=value_loss,
                microbatches=batches,
                freeze_roles=("model", "critic"),
            )
        }
        self._require_update(result["value"], "iql_value")

        def actor_loss(actor, batch):
            with torch.no_grad():
                q1, q2 = self.target_critic(batch["observations"], batch["actions"])
                advantage = (
                    torch.minimum(q1.float(), q2.float())
                    - self.value(batch["observations"]).float()
                )
            values = advantage_weighted_bc(
                actor.log_prob(batch["observations"], batch["actions"]),
                advantage,
                inverse_temperature=self.inverse_temperature,
                max_weight=self.max_weight,
            )
            return _term(values, "iql_actor")

        result["actor"] = self.engine.phase(
            "iql_actor",
            objective=actor_loss,
            microbatches=batches,
            freeze_roles=("critic", "value"),
        )
        self._require_update(result["actor"], "iql_actor")

        def critic_loss(critic, batch):
            with torch.no_grad():
                target = (
                    batch["rewards"].float()
                    + batch.get("discounts", self.gamma * (~batch["terminated"]))
                    * self.value(batch["next_observations"]).float()
                )
            q1, q2 = critic(batch["observations"], batch["actions"])
            return _term(
                (q1.float() - target).square() + (q2.float() - target).square(), "iql_critic"
            )

        result["critic"] = self.engine.phase(
            "iql_critic",
            role="critic",
            objective=critic_loss,
            microbatches=batches,
            freeze_roles=("model", "value"),
        )
        self._require_update(result["critic"], "iql_critic")
        self.engine.update_target("critic", "target_critic", 1 - self.tau)
        self.updates += 1
        self._incomplete = False
        return result
