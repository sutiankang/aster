"""TD-MPC2 latent, reward, value, and policy-prior learning with MPPI control."""

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F

from ..core import LossTerm, LossBundle
from ..models.tdmpc2 import QEnsemble


class RunningValueScale:
    """Start at 1; EMA-update the 5th-to-95th percentile range with a lower bound of 1."""

    def __init__(self, tau=0.01):
        if not 0 < tau <= 1:
            raise ValueError("Running-scale decay must be in (0,1]")
        self.tau, self.value = tau, 1.0

    def update(self, values):
        values = values.detach().float().reshape(-1)
        if not values.numel() or not torch.isfinite(values).all():
            raise ValueError("Cannot estimate value scale from empty/nonfinite values")
        percentiles = torch.quantile(values, values.new_tensor([0.05, 0.95]))
        target = max(1.0, float(percentiles[1] - percentiles[0]))
        self.value += self.tau * (target - self.value)

    def state_dict(self):
        return {"tau": self.tau, "value": self.value}

    def load_state_dict(self, state):
        if state["tau"] != self.tau or not 1 <= state["value"] < float("inf"):
            raise ValueError("Invalid TD-MPC2 running scale checkpoint")
        self.value = state["value"]


class TDMPC2Method:
    def __init__(
        self,
        engine,
        policy,
        *,
        policy_optimizer=None,
        policy_optimizer_factory=None,
        discount=0.99,
        rho=0.5,
        tau=0.01,
        consistency_weight=20.0,
        reward_weight=0.1,
        value_weight=0.1,
        termination_weight=1.0,
        entropy_weight=1e-4,
    ):
        if not 0 < discount <= 1 or not 0 < rho <= 1 or not 0 < tau <= 1:
            raise ValueError("Invalid TD-MPC2 discount/horizon decay/target averaging")
        if (
            min(consistency_weight, reward_weight, value_weight, termination_weight, entropy_weight)
            < 0
        ):
            raise ValueError("Objective coefficients must be nonnegative")
        if not all(
            math.isfinite(value)
            for value in (
                discount,
                rho,
                tau,
                consistency_weight,
                reward_weight,
                value_weight,
                termination_weight,
                entropy_weight,
            )
        ):
            raise ValueError("TD-MPC2 coefficients must be finite before any role update")
        self.engine = engine
        c = engine.model.config
        if (
            policy.config.feature_dim != c.latent_dim + c.task_dim
            or policy.config.action_dim != c.action_dim
        ):
            raise ValueError("World and policy prior dimensions do not align")
        if c.task_dim:
            engine.register_embedding_projection(
                "model", "task_embedding", max_norm=1.0, norm_type=2.0
            )
        self.policy = engine.add_role(
            "policy_prior",
            policy,
            optimizer=policy_optimizer,
            optimizer_factory=policy_optimizer_factory,
        )
        self.target = engine.clone_target(
            "model", "target_q", factory=lambda: QEnsemble(c), source_path="q_heads"
        )
        self.settings = dict(
            discount=discount,
            rho=rho,
            tau=tau,
            consistency_weight=consistency_weight,
            reward_weight=reward_weight,
            value_weight=value_weight,
            termination_weight=termination_weight,
            entropy_weight=entropy_weight,
        )
        self.scale = RunningValueScale(tau)
        self.updates, self._incomplete = 0, False
        engine.register_state("tdmpc2", self)

    def _validate(self, batch):
        obs, action, reward, terminal = (
            batch[key] for key in ("observations", "actions", "rewards", "terminated")
        )
        if (
            action.ndim != 3
            or obs.shape[:2] != (len(action), action.shape[1] + 1)
            or reward.shape != action.shape[:2]
            or terminal.shape != reward.shape
        ):
            raise ValueError("TD-MPC2 expects obs[B,H+1,...], action[B,H,D], reward/terminal[B,H]")
        if action.shape[-1] != self.engine.model.config.action_dim or terminal.dtype != torch.bool:
            raise ValueError("Invalid action dimension or terminal type")
        if min(action.shape[:2]) < 1:
            raise ValueError("TD-MPC2 requires nonempty batches and horizons")
        c = self.engine.model.config
        expected_observation = (
            (c.observation_dim,) if c.observation_kind == "state" else (c.image_channels, 64, 64)
        )
        if obs.shape[2:] != expected_observation:
            raise ValueError(
                "Observation tail dimensions do not match the declared TD-MPC2 encoder"
            )
        if (
            not action.is_floating_point()
            or not reward.is_floating_point()
            or (c.observation_kind == "state" and not obs.is_floating_point())
        ):
            raise ValueError("State/actions/rewards must be floating tensors")
        if c.observation_kind == "rgb" and (obs.min() < 0 or obs.max() > 255):
            raise ValueError("RGB replay must be explicitly in [0,255]")
        tensors = [value for value in batch.values() if isinstance(value, torch.Tensor)]
        if any(value.device != self.engine.device for value in tensors):
            raise ValueError(
                "Prepare every replay tensor on the Trainer device before a multi-phase TD-MPC2 update"
            )
        task = batch.get("task_ids")
        if c.task_dim:
            if task is None or task.shape != obs.shape[:2] or task.dtype != torch.long:
                raise ValueError("Multitask windows require aligned B,H+1 integer task IDs")
            if (
                (task < 0).any()
                or (task >= len(c.action_dimensions)).any()
                or not (task == task[:, :1]).all()
            ):
                raise ValueError("A replay window belongs to one valid task per episode")
        elif task is not None:
            raise ValueError("Single-task replay must not carry implicit task semantics")
        if "valid" in batch and not batch["valid"].all():
            raise ValueError(
                "Sample complete within-episode windows; padded windows need a separate objective"
            )
        if terminal[:, :-1].any():
            raise ValueError("A training window cannot cross a terminal/reset boundary")
        truncated = batch.get("truncated")
        if truncated is not None and (
            truncated.shape != terminal.shape
            or truncated.dtype != torch.bool
            or truncated[:, :-1].any()
        ):
            raise ValueError(
                "A window cannot cross a time-limit reset; truncation still permits TD bootstrap at its end"
            )
        if (
            not all(torch.isfinite(value).all() for value in (obs, action, reward))
            or action.abs().max() > 1
        ):
            raise ValueError(
                "Finite observations/rewards and normalized actions in [-1,1] required"
            )

    def update(self, microbatches):
        batches, error, layout = None, None, None
        try:
            if self._incomplete:
                raise RuntimeError(
                    "Previous multi-role update incomplete; restore its last complete checkpoint"
                )
            batches = list(microbatches)
            if len(batches) != self.engine.accumulation_steps:
                raise ValueError("One complete window microbatch is required per accumulation slot")
            for batch in batches:
                self._validate(batch)
            layout = [batch["actions"].shape[1] for batch in batches]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        declarations = self.engine.parallel.world.gather_objects((error, layout))
        if any(item[0] for item in declarations):
            raise ValueError(
                "TD-MPC2 collective preflight failed: " + str([item[0] for item in declarations])
            )
        if any(item[1] != layout for item in declarations):
            raise ValueError("TD-MPC2 ranks must agree on microbatch horizon/call order")
        world, settings = self.engine.model, self.settings
        self._incomplete = True
        task_rows = (
            torch.cat([batch["task_ids"].reshape(-1) for batch in batches])
            if world.config.task_dim
            else None
        )
        if task_rows is not None:
            self.engine.project_embedding("model", "task_embedding", task_rows)
        prepared = []
        world.eval()
        with (
            torch.no_grad(),
            torch.autocast(
                self.engine.device.type,
                dtype=torch.bfloat16 if self.engine.precision == "bf16" else torch.float16,
                enabled=self.engine.precision != "fp32",
            ),
        ):
            for batch in batches:
                task = batch.get("task_ids")
                next_task = None if task is None else task[:, 1:]
                target_z = world.encode(batch["observations"][:, 1:], next_task)
                action, _ = self.policy(
                    world.condition(target_z, next_task),
                    action_mask=world.action_mask(target_z, next_task),
                )
                target_q = world.q(
                    target_z, action, next_task, ensemble=self.target, reduction="min"
                )
                target = batch["rewards"] + settings["discount"] * (~batch["terminated"]) * target_q
                prepared.append({**batch, "target_z": target_z, "td_target": target})
        latents = []

        def world_objective(model, batch):
            action, target_z = batch["actions"], batch["target_z"]
            task = batch.get("task_ids")
            horizon = action.shape[1]
            current_task = None if task is None else task[:, 0]
            z = model.encode(batch["observations"][:, 0], current_task)
            sequence = [z]
            consistency = z.new_zeros(())
            for step in range(horizon):
                current_task = None if task is None else task[:, step]
                z = model.next(z, action[:, step], current_task)
                sequence.append(z)
                consistency = (
                    consistency
                    + (z.float() - target_z[:, step].float()).square().mean(-1).sum()
                    * settings["rho"] ** step
                )
            zs = torch.stack(sequence, 1)
            tasks = None if task is None else task[:, :-1]
            reward_logits = model.reward(zs[:, :-1], action, tasks)
            q_logits = model.q(zs[:, :-1], action, tasks)
            weights = action.new_tensor(settings["rho"]) ** torch.arange(
                horizon, device=action.device
            )
            reward_error = model.value_loss(reward_logits, batch["rewards"])
            value_error = model.value_loss(
                q_logits, batch["td_target"].expand(model.config.num_q, -1, -1)
            )
            count = torch.tensor(
                len(action) * horizon, dtype=torch.int64, device=consistency.device
            )
            terms = [
                LossTerm(
                    consistency, count, "transition", "consistency", settings["consistency_weight"]
                ),
                LossTerm(
                    (reward_error * weights).sum(),
                    count,
                    "transition",
                    "reward",
                    settings["reward_weight"],
                ),
                LossTerm(
                    (value_error * weights).sum(),
                    count * model.config.num_q,
                    "q_transition",
                    "value",
                    settings["value_weight"],
                ),
            ]
            if model.config.episodic:
                logits = model.termination(zs[:, 1:])
                errors = F.binary_cross_entropy_with_logits(
                    logits, batch["terminated"].to(logits), reduction="sum"
                )
                terms.append(
                    LossTerm(
                        errors, count, "transition", "termination", settings["termination_weight"]
                    )
                )
            latents.append({"latents": zs.detach(), "task_ids": task})
            return LossBundle(tuple(terms))

        world_result = self.engine.phase(
            "tdmpc2_world",
            objective=world_objective,
            microbatches=prepared,
            freeze_roles=("policy_prior",),
        )
        if not world_result.updated:
            raise RuntimeError(
                "TD-MPC2 world phase skipped; restore before another multi-role update"
            )
        if task_rows is not None:
            self.engine.project_embedding("model", "task_embedding", task_rows)

        actor_batches, start_values = [], []
        world.train()
        with (
            torch.no_grad(),
            torch.autocast(
                self.engine.device.type,
                dtype=torch.bfloat16 if self.engine.precision == "bf16" else torch.float16,
                enabled=self.engine.precision != "fp32",
            ),
        ):
            for batch in latents:
                z, task = batch["latents"], batch["task_ids"]
                features = world.condition(z, task).detach()
                mask = world.action_mask(z, task)
                noise = torch.randn((*z.shape[:2], world.config.action_dim), device=z.device)
                action, _ = self.policy(features, action_mask=mask, noise=noise)
                rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state(z.device) if z.is_cuda else None
                logits = world.q(z, action, task)
                indices = torch.randperm(world.config.num_q, device=z.device)[:2]
                values = world.decode_value(logits[indices]).mean(0)
                start_values.append(values[:, 0])
                actor_batches.append(
                    {
                        **batch,
                        "features": features,
                        "mask": mask,
                        "noise": noise,
                        "q_indices": indices,
                        "q_rng": rng,
                        "q_cuda_rng": cuda_rng,
                    }
                )
        local_values = torch.cat(start_values).cpu().tolist()
        gathered = self.engine.replica_group.gather_objects(local_values)
        self.scale.update(torch.tensor([value for shard in gathered for value in shard]))

        def policy_objective(policy, batch):
            z, task = batch["latents"], batch["task_ids"]
            action, info = policy(
                batch["features"], action_mask=batch["mask"], noise=batch["noise"]
            )

            with torch.random.fork_rng(devices=[z.device] if z.is_cuda else []):
                torch.set_rng_state(batch["q_rng"])
                if z.is_cuda:
                    torch.cuda.set_rng_state(batch["q_cuda_rng"], z.device)
                logits = world.q(z, action, task)
            values = world.decode_value(logits[batch["q_indices"]]).mean(0) / self.scale.value
            weights = values.new_tensor(settings["rho"]) ** torch.arange(
                values.shape[1], device=values.device
            )
            errors = -(values + settings["entropy_weight"] * info["scaled_entropy"]) * weights
            return LossTerm(
                errors.sum(),
                torch.tensor(errors.numel(), dtype=torch.int64, device=errors.device),
                "latent_slot",
                "policy_prior",
            )

        policy_result = self.engine.phase(
            "tdmpc2_policy",
            role="policy_prior",
            objective=policy_objective,
            microbatches=actor_batches,
            freeze_roles=("model",),
        )
        if not policy_result.updated:
            raise RuntimeError(
                "TD-MPC2 policy phase skipped after world update; restore complete checkpoint"
            )
        self.engine.update_target("model", "target_q", 1 - settings["tau"], source_path="q_heads")
        self.updates += 1
        self._incomplete = False
        world.eval()
        return {"world": world_result, "policy": policy_result}

    def state_dict(self):
        if self._incomplete:
            raise RuntimeError("Cannot checkpoint an incomplete TD-MPC2 multi-role update")
        return {
            "settings": self.settings,
            "scale": self.scale.state_dict(),
            "updates": self.updates,
            "world_config": self.engine.model.config.to_dict(),
            "policy_config": self.policy.config.to_dict(),
        }

    def load_state_dict(self, state):
        if (
            state["settings"] != self.settings
            or state.get("world_config") != self.engine.model.config.to_dict()
            or state.get("policy_config") != self.policy.config.to_dict()
        ):
            raise ValueError("TD-MPC2 update settings changed")
        self.scale.load_state_dict(state["scale"])
        self.updates, self._incomplete = state["updates"], False


@dataclass(frozen=True)
class MPPIConfig:
    horizon: int = 5
    population: int = 128
    elites: int = 16
    policy_trajectories: int = 16
    iterations: int = 4
    temperature: float = 0.5
    min_std: float = 0.05
    max_std: float = 2.0
    discount: float = 0.99

    def __post_init__(self):
        if (
            min(self.horizon, self.population, self.elites, self.iterations) < 1
            or self.elites > self.population
            or not 0 <= self.policy_trajectories < self.population
        ):
            raise ValueError("Invalid MPPI planning budget")
        if (
            not 0 < self.min_std <= self.max_std
            or self.temperature <= 0
            or not 0 < self.discount <= 1
        ):
            raise ValueError("Invalid MPPI sampling/value parameters")


class TDMPC2Planner:
    """Inject policy trajectories into elite-weighted MPPI and retain a warm-start mean."""

    def __init__(self, world, policy, config=None, *, task_projection=None):
        self.world, self.policy, self.config = world, policy, config or MPPIConfig()
        if task_projection is not None and not callable(task_projection):
            raise TypeError("Task projection must be an explicit callable over integer row IDs")
        if (
            world.config.task_dim
            and getattr(world, "_aster_training_owned", False)
            and task_projection is None
        ):
            raise ValueError(
                "Training-owned task embeddings require a Trainer projection callback or an independently exported inference model"
            )
        self.task_projection = task_projection
        self.previous_mean = None

    @torch.no_grad()
    def plan(self, observation, *, first=False, eval_mode=False, task_id=None, generator=None):
        c, world = self.config, self.world
        world.eval()
        self.policy.eval()
        if len(observation) != 1:
            raise ValueError(
                "One planner state belongs to one environment; batch environments own separate planners"
            )
        if world.config.task_dim:
            if type(task_id) is not int or not 0 <= task_id < len(world.config.action_dimensions):
                raise ValueError("Multitask planning needs one valid integer task_id")
        elif task_id is not None:
            raise ValueError("Single-task planning cannot accept an undeclared task ID")
        task = (
            None
            if task_id is None
            else torch.full((1,), task_id, device=observation.device, dtype=torch.long)
        )
        if task is not None:
            if self.task_projection is not None:
                self.task_projection(task)
            else:
                if getattr(world, "_aster_training_owned", False) or not isinstance(
                    world.task_embedding, nn.Embedding
                ):
                    raise ValueError(
                        "Local task projection cannot mutate training-owned/sharded parameters"
                    )
                table = world.task_embedding.weight
                rows = table[task].float()
                norms = rows.norm(dim=-1, keepdim=True)
                projected = torch.where(norms > 1, rows / (norms + 1e-7), rows)
                table.index_copy_(0, task, projected.to(table))
        initial = world.encode(observation, task, generator=generator)
        task_population = None if task is None else task.expand(c.population)
        mask = world.action_mask(initial, task)[0]
        mean = initial.new_zeros(c.horizon, world.config.action_dim)
        if not first and self.previous_mean is not None:
            mean[:-1] = self.previous_mean.to(mean)[1:]
        std = torch.full_like(mean, c.max_std)
        actions = initial.new_empty(c.horizon, c.population, world.config.action_dim)
        if c.policy_trajectories:
            z = initial.expand(c.policy_trajectories, -1)
            ptask = None if task is None else task.expand(c.policy_trajectories)
            for step in range(c.horizon):
                action, _ = self.policy(
                    world.condition(z, ptask),
                    action_mask=world.action_mask(z, ptask),
                    generator=generator,
                )
                actions[step, : c.policy_trajectories] = action
                z = world.next(z, action, ptask)
        for _ in range(c.iterations):
            noise = torch.randn(
                (c.horizon, c.population - c.policy_trajectories, world.config.action_dim),
                device=mean.device,
                generator=generator,
            )
            actions[:, c.policy_trajectories :] = (mean[:, None] + std[:, None] * noise).clamp(
                -1, 1
            )
            actions *= mask
            z = initial.expand(c.population, -1)
            values, alive = initial.new_zeros(c.population), initial.new_ones(c.population)
            for step in range(c.horizon):
                values += (
                    c.discount**step
                    * alive
                    * world.decode_value(world.reward(z, actions[step], task_population))
                )
                z = world.next(z, actions[step], task_population)
                if world.config.episodic:
                    alive *= world.termination(z).sigmoid() <= 0.5
            action, _ = self.policy(
                world.condition(z, task_population),
                action_mask=world.action_mask(z, task_population),
                generator=generator,
            )
            values += (
                c.discount**c.horizon
                * alive
                * world.q(z, action, task_population, reduction="avg", generator=generator)
            )
            if not torch.isfinite(values).all():
                raise ValueError(
                    "Nonfinite predicted return; refusing to turn it into a valid control action"
                )
            elite_values, indices = values.topk(c.elites)
            elite_actions = actions[:, indices]
            weights = (c.temperature * (elite_values - elite_values.max())).softmax(0)
            mean = (elite_actions * weights[None, :, None]).sum(1)
            std = (
                ((elite_actions - mean[:, None]).square() * weights[None, :, None])
                .sum(1)
                .sqrt()
                .clamp(c.min_std, c.max_std)
            )
            mean, std = mean * mask, std * mask
        index = torch.multinomial(weights, 1, generator=generator).item()
        action = elite_actions[0, index]
        if not eval_mode:
            action = action + std[0] * torch.randn(
                action.shape, device=action.device, generator=generator
            )
        self.previous_mean = mean.detach().clone()
        return action.clamp(-1, 1), {
            "mean": mean,
            "std": std,
            "elite_values": elite_values,
            "model_based_not_environment_return": True,
        }

    def state_dict(self):
        return {"config": asdict(self.config), "previous_mean": self.previous_mean}

    def load_state_dict(self, state):
        if state["config"] != asdict(self.config):
            raise ValueError("Planner budget or sampling semantics changed")
        self.previous_mean = state["previous_mean"]
