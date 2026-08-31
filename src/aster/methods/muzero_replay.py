"""MuZero search, reanalysis, overlapping trajectory windows, and prioritized replay."""

from dataclasses import dataclass
import hashlib
import math

import torch

from ..core.serialization import canonical_json
from ..data.replay import ReplayBuffer
from ..models.muzero import MuZeroConfig, MuZeroModel
from ..planning.mcts import muzero_policy, gumbel_muzero_policy
from .muzero import nstep_value_targets


def _digest(configuration, tensors):
    digest = hashlib.sha256(canonical_json(configuration).encode("utf-8"))
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(canonical_json([name, str(value.dtype), list(value.shape)]).encode("utf-8"))

        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MuZeroEpisode:
    observations: torch.Tensor
    actions: torch.Tensor  # [T]
    rewards: torch.Tensor  # [T]
    terminated: torch.Tensor  # [T]，absorbing terminal
    truncated: torch.Tensor
    invalid_actions: torch.Tensor | None = None

    def validate(self, config):
        length = self.actions.numel()
        if (
            self.actions.ndim != 1
            or length < 1
            or self.actions.dtype != torch.int64
            or self.observations.shape != (length + 1, config.observation_dim)
            or self.rewards.shape != (length,)
            or self.terminated.shape != (length,)
            or self.truncated.shape != (length,)
        ):
            raise ValueError("MuZero episode needs aligned T transitions and T+1 observations")
        tensors = (self.observations, self.actions, self.rewards, self.terminated, self.truncated)
        if any(value.device.type != "cpu" for value in tensors):
            raise ValueError(
                "Commit replay episodes on CPU; planning moves observations explicitly"
            )
        if (
            self.observations.dtype != torch.float32
            or self.rewards.dtype != torch.float32
            or not torch.isfinite(self.observations).all()
            or not torch.isfinite(self.rewards).all()
            or self.terminated.dtype != torch.bool
            or self.truncated.dtype != torch.bool
        ):
            raise ValueError("Episode observations/rewards must be finite FP32, boundaries boolean")
        if (self.actions < 0).any() or (self.actions >= config.num_actions).any():
            raise ValueError("Episode contains out-of-range actions")
        boundary = self.terminated | self.truncated
        if boundary[:-1].any() or not boundary[-1]:
            raise ValueError("Commit one complete episode, with exactly its final reset boundary")
        if self.invalid_actions is not None:
            invalid = self.invalid_actions
            if (
                invalid.shape != (length + 1, config.num_actions)
                or invalid.dtype != torch.bool
                or invalid.device.type != "cpu"
            ):
                raise ValueError("Invalid-action masks must be CPU boolean T+1,A")
            if invalid[torch.arange(length), self.actions].any():
                raise ValueError("Recorded episode executed an invalid action")
            count = length if self.terminated[-1] else length + 1
            if invalid[:count].all(-1).any():
                raise ValueError("Every nonterminal state needs at least one legal action")
        return length

    def fingerprint(self):
        return _digest(
            {"type": "muzero_episode_v1"},
            {name: value for name, value in vars(self).items() if value is not None},
        )


@dataclass(frozen=True)
class MuZeroAnalysis:
    episode_id: str
    model_id: str
    policy_targets: torch.Tensor
    search_values: torch.Tensor
    algorithm: str


class MuZeroSearch:
    """Search against a fixed model identity with resumable random state."""

    def __init__(
        self,
        config,
        weights,
        *,
        seed=0,
        device="cpu",
        algorithm="muzero",
        num_simulations=32,
        search_options=None,
    ):
        if not isinstance(config, MuZeroConfig) or algorithm not in {"muzero", "gumbel_muzero"}:
            raise ValueError(
                "MuZeroSearch needs a native model configuration and explicit algorithm"
            )
        if type(num_simulations) is not int or num_simulations < 1:
            raise ValueError("Search simulation budget must be positive")
        self.model = MuZeroModel(config).to(device).eval().requires_grad_(False)
        self.model.load_state_dict(weights, strict=True)
        self.algorithm, self.num_simulations = algorithm, num_simulations
        self.options = dict(search_options or {})
        reserved = {"root", "recurrent_fn", "generator", "num_simulations", "invalid_actions"}
        if reserved & self.options.keys():
            raise ValueError("Search options cannot replace planner inputs/RNG/budget")
        canonical_json(self.options)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.model_id = self._fingerprint()

    @classmethod
    def from_trainer(cls, trainer, **options):

        weights = trainer.export_state_dict(only_rank_zero=False)
        return cls(trainer.model.config, weights, **options)

    def _fingerprint(self):
        return _digest(self.model.config.to_dict(), self.model.state_dict())

    def config_dict(self):
        return dict(
            type="muzero_search_v1",
            model=self.model.config.to_dict(),
            algorithm=self.algorithm,
            num_simulations=self.num_simulations,
            options=self.options,
        )

    @torch.no_grad()
    def plan(self, observations, *, invalid_actions=None):
        if self.model.training or self._fingerprint() != self.model_id:
            raise ValueError(
                "Search snapshot changed; use explicit refresh, not external weight mutation"
            )
        device = next(self.model.parameters()).device
        if (
            observations.device != device
            or observations.dtype != torch.float32
            or not torch.isfinite(observations).all()
        ):
            raise ValueError("Search observations must be finite FP32 on the snapshot device")
        search = muzero_policy if self.algorithm == "muzero" else gumbel_muzero_policy
        return search(
            self.model.search_root(observations),
            self.model.search_step,
            num_simulations=self.num_simulations,
            generator=self.generator,
            invalid_actions=invalid_actions,
            **self.options,
        )

    @torch.no_grad()
    def reanalyze(self, episode):
        length = episode.validate(self.model.config)

        count = length if episode.terminated[-1] else length + 1
        device = next(self.model.parameters()).device
        invalid = (
            None if episode.invalid_actions is None else episode.invalid_actions[:count].to(device)
        )
        result = self.plan(episode.observations[:count].to(device), invalid_actions=invalid)
        policy = torch.zeros(length + 1, self.model.config.num_actions)
        values = torch.zeros(length + 1)
        policy[:count] = result.action_weights.cpu()
        values[:count] = result.search_tree.summary()["value"].cpu()
        return MuZeroAnalysis(episode.fingerprint(), self.model_id, policy, values, self.algorithm)

    def refresh(self, trainer):
        if trainer.model.config.to_dict() != self.model.config.to_dict():
            raise ValueError("Cannot refresh planner with a different model architecture")
        weights = trainer.export_state_dict(only_rank_zero=False)
        self.model.load_state_dict(weights, strict=True)
        self.model_id = self._fingerprint()

    def state_dict(self):
        if self._fingerprint() != self.model_id:
            raise ValueError("Cannot checkpoint a mutated search snapshot")
        return dict(
            config=self.config_dict(),
            model_id=self.model_id,
            weights={k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            rng=self.generator.get_state().clone(),
        )

    def load_state_dict(self, state):
        if (
            state["config"] != self.config_dict()
            or _digest(self.model.config.to_dict(), state["weights"]) != state["model_id"]
        ):
            raise ValueError("MuZero search checkpoint identity mismatch")

        check = torch.Generator(device=self.generator.device)
        check.set_state(state["rng"])
        expected = self.model.state_dict()
        if state["weights"].keys() != expected.keys() or any(
            state["weights"][k].shape != v.shape or state["weights"][k].dtype != v.dtype
            for k, v in expected.items()
        ):
            raise ValueError("MuZero search checkpoint weight schema mismatch")
        self.model.load_state_dict(state["weights"], strict=True)
        self.generator.set_state(state["rng"])
        self.model_id = state["model_id"]


class MuZeroReplay:
    """Store overlapping fixed-unroll windows with search-weight identity in shared
    prioritized replay."""

    def __init__(
        self, config, *, capacity=10000, unroll_steps=5, td_steps=10, seed=0, priority_alpha=0.6
    ):
        if not isinstance(config, MuZeroConfig) or any(
            type(x) is not int or x < 1 for x in (unroll_steps, td_steps)
        ):
            raise ValueError("MuZero replay needs a model config and positive unroll/TD horizons")
        self.config, self.unroll_steps, self.td_steps = config, unroll_steps, td_steps
        self.buffer = ReplayBuffer(capacity, seed=seed, priority_alpha=priority_alpha)

    def add_episode(self, episode, analysis):
        length = episode.validate(self.config)
        if not isinstance(analysis, MuZeroAnalysis) or analysis.episode_id != episode.fingerprint():
            raise ValueError("Search analysis does not belong to this episode")
        policies, values = analysis.policy_targets, analysis.search_values
        valid_count = length if episode.terminated[-1] else length + 1
        if (
            policies.shape != (length + 1, self.config.num_actions)
            or values.shape != (length + 1,)
            or policies.device.type != "cpu"
            or values.device.type != "cpu"
            or not torch.isfinite(policies).all()
            or not torch.isfinite(values).all()
            or (policies < 0).any()
            or not torch.allclose(policies[:valid_count].sum(-1), torch.ones(valid_count))
        ):
            raise ValueError("Invalid reanalysis policy/value targets")
        if (
            len(analysis.model_id) != 64
            or any(x not in "0123456789abcdef" for x in analysis.model_id)
            or analysis.algorithm not in {"muzero", "gumbel_muzero"}
        ):
            raise ValueError("Reanalysis needs a weight fingerprint and known search algorithm")
        if episode.terminated[-1] and (values[-1] != 0 or policies[-1].any()):
            raise ValueError(
                "Absorbing terminal analysis must have zero value and no search policy"
            )
        targets = nstep_value_targets(
            episode.rewards[None],
            values[None],
            episode.terminated[None],
            truncated=episode.truncated[None],
            discount=self.config.discount,
            n_steps=self.td_steps,
        )[0]
        fingerprint = torch.tensor(list(bytes.fromhex(analysis.model_id)), dtype=torch.uint8)
        horizon = self.unroll_steps
        for start in range(length):
            observed = min(horizon, length - start)
            states = min(horizon + 1, valid_count - start)
            actions = torch.zeros(horizon, dtype=torch.int64)
            reward = torch.zeros(horizon)
            policy = torch.zeros(horizon + 1, self.config.num_actions)
            value = torch.zeros(horizon + 1)
            actions[:observed] = episode.actions[start : start + observed]
            reward[:observed] = episode.rewards[start : start + observed]
            policy[:states] = policies[start : start + states]
            value[:states] = targets[start : start + states]
            item = dict(
                observations=episode.observations[start],
                actions=actions,
                reward_targets=reward,
                policy_targets=policy,
                value_targets=value,
                valid=torch.arange(horizon + 1) < states,
                reward_valid=torch.arange(horizon) < observed,
                search_model_id=fingerprint,
            )
            self.buffer.add(item, priority=float(abs(targets[start] - values[start])))

    def sample(self, batch_size, *, beta=0.4, device="cpu"):
        return self.buffer.sample(batch_size, beta=beta, device=device)

    def update_priorities(self, batch, errors):
        self.buffer.update_priorities(
            batch["replay_indices"], batch["replay_versions"], errors.detach().abs().cpu()
        )

    def state_dict(self):
        return dict(
            config=self.config.to_dict(),
            unroll_steps=self.unroll_steps,
            td_steps=self.td_steps,
            buffer=self.buffer.state_dict(),
        )

    def load_state_dict(self, state):
        if (
            state["config"] != self.config.to_dict()
            or state["unroll_steps"] != self.unroll_steps
            or state["td_steps"] != self.td_steps
        ):
            raise ValueError("MuZero replay checkpoint configuration mismatch")
        self.buffer.load_state_dict(state["buffer"])
