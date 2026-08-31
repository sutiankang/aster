"""PlaNet episode collection, contiguous replay chunks, and resumable MPC interaction."""

from collections import deque
from copy import deepcopy
import math
import torch

from ..models.planet import PlaNetConfig, PlaNetWorldModel, PlaNetState
from ..planning.planet import planet_cem_plan
from .planet import PlaNetObjective, preprocess_planet_images


def planet_chunk_offsets(length, chunk_length, *, generator, num_chunks=None):
    """Use the declared random contiguous-chunk scheme, not independently sampled windows."""
    if (
        type(length) is not int
        or type(chunk_length) is not int
        or chunk_length < 1
        or length < chunk_length
    ):
        raise ValueError("PlaNet episode is shorter than one training chunk")
    count = max(1, length // chunk_length - 1) if num_chunks is None else num_chunks
    if type(count) is not int or count < 1 or count * chunk_length > length:
        raise ValueError("Invalid PlaNet number of contiguous chunks")
    offset = int(torch.randint(length - count * chunk_length + 1, (), generator=generator))
    return tuple(offset + index * chunk_length for index in range(count))


class PlaNetReplay:
    """Rotate sampled episodes, then consume contiguous chunks from the selected episode."""

    def __init__(self, config, *, sequence_length=50, seed=0, max_episodes=None, num_chunks=None):
        if (
            not isinstance(config, PlaNetConfig)
            or type(sequence_length) is not int
            or sequence_length < 1
            or type(seed) is not int
        ):
            raise ValueError("Invalid PlaNet replay configuration")
        if max_episodes is not None and (type(max_episodes) is not int or max_episodes < 1):
            raise ValueError("PlaNet replay capacity must be positive or None")
        if num_chunks is not None and (type(num_chunks) is not int or num_chunks < 1):
            raise ValueError("PlaNet num_chunks must be positive or None")
        self.config = config
        self.settings = dict(
            model=config.to_dict(),
            sequence_length=sequence_length,
            max_episodes=max_episodes,
            num_chunks=num_chunks,
            sampler="reload_permutation_contiguous_random_offset",
            image_bits=5,
        )
        self.rng = torch.Generator().manual_seed(seed)
        self.episodes, self.order, self.pending = {}, deque(), deque()
        self.insertions = self.samples = 0

    def _validate_episode(self, episode):
        if set(episode) != {
            "observations",
            "previous_actions",
            "rewards",
            "is_first",
            "terminated",
            "truncated",
        }:
            raise ValueError(
                "PlaNet episode requires explicit aligned observation/action/reward/end fields"
            )
        obs = episode["observations"]
        if (
            not isinstance(obs, torch.Tensor)
            or obs.ndim != 1 + len(self.config.observation_shape)
            or obs.shape[1:] != self.config.observation_shape
            or len(obs) < 2
        ):
            raise ValueError("Invalid PlaNet episode observation shape")
        expected_dtype = torch.float32 if self.config.observation_dim else torch.uint8
        if obs.dtype != expected_dtype or not torch.isfinite(obs).all():
            raise ValueError(
                "Store PlaNet vector float32 or raw uint8 pixels, not dequantized cached images"
            )
        for key, shape, dtype in (
            ("previous_actions", (len(obs), self.config.action_dim), torch.float32),
            ("rewards", (len(obs),), torch.float32),
            ("is_first", (len(obs),), torch.bool),
            ("terminated", (len(obs),), torch.bool),
            ("truncated", (len(obs),), torch.bool),
        ):
            value = episode[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.dtype != dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid PlaNet episode {key}")
        if (
            not episode["is_first"][0]
            or episode["is_first"][1:].any()
            or episode["previous_actions"][0].any()
            or episode["rewards"][0] != 0
        ):
            raise ValueError(
                "PlaNet episode reset row must have is_first and zero previous-action/reward"
            )
        done = episode["terminated"] | episode["truncated"]
        if done[:-1].any() or not done[-1]:
            raise ValueError("PlaNet replay accepts complete episodes with one final end boundary")
        if (episode["previous_actions"].abs() > 1).any():
            raise ValueError("PlaNet replay actions must use normalized [-1,1] coordinates")

    def add(self, episode):
        self._validate_episode(episode)

        copied = {key: value.detach().cpu().clone() for key, value in episode.items()}
        self.episodes[self.insertions] = copied
        self.insertions += 1
        capacity = self.settings["max_episodes"]
        if capacity is not None and len(self.episodes) > capacity:
            self.episodes.pop(min(self.episodes))

    def _fill(self):
        length = self.settings["sequence_length"]
        minimum = length * (self.settings["num_chunks"] or 1)
        while not self.pending:
            if not self.order:
                eligible = [
                    key
                    for key, episode in self.episodes.items()
                    if len(episode["observations"]) >= minimum
                ]
                if not eligible:
                    raise ValueError(
                        "No complete PlaNet episode is long enough for configured chunks"
                    )
                permutation = torch.randperm(len(eligible), generator=self.rng).tolist()
                self.order.extend(eligible[index] for index in permutation)
            key = self.order.popleft()
            if key not in self.episodes:
                continue
            episode = self.episodes[key]
            offsets = planet_chunk_offsets(
                len(episode["observations"]),
                length,
                generator=self.rng,
                num_chunks=self.settings["num_chunks"],
            )
            for offset in offsets:
                self.pending.append(
                    {
                        name: value[offset : offset + length].clone()
                        for name, value in episode.items()
                    }
                )

    def sample(self, batch_size, *, device="cpu"):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("PlaNet replay batch size must be positive")
        rows = []
        for _ in range(batch_size):
            self._fill()
            rows.append(self.pending.popleft())
        result = {
            key: torch.stack([row[key] for row in rows])
            for key in ("observations", "previous_actions", "rewards", "is_first")
        }
        if not self.config.observation_dim:
            result["observations"] = preprocess_planet_images(
                result["observations"], generator=self.rng
            )
        self.samples += batch_size
        return {key: value.to(device) for key, value in result.items()}

    def state_dict(self):
        return dict(
            settings=deepcopy(self.settings),
            insertions=self.insertions,
            samples=self.samples,
            episodes=deepcopy(self.episodes),
            order=list(self.order),
            pending=list(deepcopy(self.pending)),
            rng=self.rng.get_state().clone(),
        )

    def load_state_dict(self, state):
        if state["settings"] != self.settings or any(
            type(state[key]) is not int or state[key] < 0 for key in ("insertions", "samples")
        ):
            raise ValueError("PlaNet replay settings/counters differ")
        for key, episode in state["episodes"].items():
            if type(key) is not int or not 0 <= key < state["insertions"]:
                raise ValueError("Invalid PlaNet replay episode identity")
            self._validate_episode(episode)
        capacity = self.settings["max_episodes"]
        if capacity is not None and len(state["episodes"]) > capacity:
            raise ValueError("PlaNet replay exceeds capacity")
        if any(
            type(key) is not int or not 0 <= key < state["insertions"] for key in state["order"]
        ):
            raise ValueError("Invalid PlaNet replay pending order")
        for row in state["pending"]:
            if set(row) != {
                "observations",
                "previous_actions",
                "rewards",
                "is_first",
                "terminated",
                "truncated",
            } or any(len(v) != self.settings["sequence_length"] for v in row.values()):
                raise ValueError("Invalid PlaNet replay pending chunk")
        rng = torch.Generator()
        rng.set_state(state["rng"])
        self.episodes = deepcopy(state["episodes"])
        self.order = deque(state["order"])
        self.pending = deque(deepcopy(state["pending"]))
        self.insertions, self.samples, self.rng = state["insertions"], state["samples"], rng


class PlaNetLoop:
    """Collect observations and learn from replay with checkpointed environment and belief state."""

    def __init__(
        self,
        engine,
        environment,
        replay,
        *,
        batch_size=50,
        seed=0,
        action_noise=0.3,
        max_episode_steps=1000,
        planner=None,
        state_name="planet_loop",
    ):
        error = None
        try:
            if (
                not isinstance(engine.model.config, PlaNetConfig)
                or not isinstance(replay, PlaNetReplay)
                or engine.model.config != replay.config
            ):
                raise ValueError("PlaNet loop model/replay differ")
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
                raise ValueError("PlaNet loop currently requires pure DP model providers")
            if any(
                not callable(getattr(environment, name, None))
                for name in ("config_dict", "reset", "step", "state_dict", "load_state_dict")
            ):
                raise ValueError(
                    "PlaNet environment must provide an explicit restorable simulator protocol"
                )
            if (
                any(
                    type(value) is not int or value < 1 for value in (batch_size, max_episode_steps)
                )
                or type(seed) is not int
                or not math.isfinite(action_noise)
                or action_noise < 0
            ):
                raise ValueError("Invalid PlaNet collection/training settings")
            options = dict(
                horizon=12,
                population=1000,
                elites=100,
                iterations=10,
                action_low=-1.0,
                action_high=1.0,
            )
            if set(planner or {}) - set(options):
                raise ValueError("Unknown PlaNet planner option")
            options.update(planner or {})
            if (
                any(
                    type(options[k]) is not int or options[k] < 1
                    for k in ("horizon", "population", "elites")
                )
                or type(options["iterations"]) is not int
                or options["iterations"] < 0
                or options["elites"] > options["population"]
            ):
                raise ValueError("Invalid PlaNet planner dimensions")
            if (
                not all(math.isfinite(options[k]) for k in ("action_low", "action_high"))
                or not -1 <= options["action_low"] < options["action_high"] <= 1
            ):
                raise ValueError("PlaNet planner bounds must use normalized [-1,1] actions")

            deepcopy(environment.state_dict())
            if (
                not isinstance(engine.objective, PlaNetObjective)
                or engine.objective.settings["sequence_length"]
                != replay.settings["sequence_length"]
            ):
                raise ValueError("PlaNet loop requires a matching default objective")
        except Exception as exc:
            error = str(exc)
        errors = engine.parallel.world.gather_objects(error)
        if any(errors):
            raise ValueError(f"Invalid PlaNet loop setup: {errors}")
        self.engine, self.environment, self.replay = engine, environment, replay
        self.settings = dict(
            batch_size=batch_size,
            action_noise=action_noise,
            max_episode_steps=max_episode_steps,
            planner=options,
            environment=deepcopy(environment.config_dict()),
            model=engine.model.config.to_dict(),
        )
        self.rng = torch.Generator(device=engine.device).manual_seed(seed)
        self.active = None
        self.environment_steps = self.episodes = self.updates = 0
        self.failed = False
        self.role_name = state_name + "_world"

        def factory():
            with torch.random.fork_rng(devices=[]):
                return PlaNetWorldModel(engine.model.config)

        self.world = engine.clone_target("model", self.role_name, factory=factory)
        engine.register_state(state_name, self)

    def refresh_world(self):
        errors = self.engine.parallel.world.gather_objects(self.active is not None or self.failed)
        if any(errors):
            raise ValueError("Refresh PlaNet world only between episodes on all ranks")
        self.engine.update_target("model", self.role_name, 0.0)

    def _observation(self, observation):
        value = torch.as_tensor(observation).detach().cpu().clone()
        c = self.engine.model.config
        if (
            value.shape != c.observation_shape
            or value.dtype != (torch.float32 if c.observation_dim else torch.uint8)
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                "Simulator observation differs from explicit PlaNet vector/raw-pixel configuration"
            )
        return value

    @torch.no_grad()
    def collect_steps(self, steps, *, random=False):
        if self.failed or type(steps) is not int or steps < 1 or type(random) is not bool:
            raise ValueError("Invalid PlaNet collection boundary/step count")
        returns = []
        c = self.engine.model.config
        try:
            for _ in range(steps):
                if self.active is None:
                    first = self._observation(self.environment.reset())
                    self.active = dict(
                        observations=[first],
                        previous_actions=[torch.zeros(c.action_dim, dtype=torch.float32)],
                        rewards=[torch.tensor(0.0, dtype=torch.float32)],
                        is_first=[True],
                        terminated=[False],
                        truncated=[False],
                        state=self.world.initial(1, device=self.engine.device),
                        random=random,
                    )
                if self.active["random"] != random:
                    raise ValueError("PlaNet collection policy cannot change mid-episode")
                active = self.active
                if random:
                    action = (
                        2
                        * torch.rand(
                            c.action_dim,
                            device=self.engine.device,
                            dtype=torch.float32,
                            generator=self.rng,
                        )
                        - 1
                    )
                else:
                    observation = active["observations"][-1].to(self.engine.device)
                    if not c.observation_dim:
                        observation = preprocess_planet_images(observation, generator=self.rng)
                    previous = active["previous_actions"][-1].to(self.engine.device)
                    first = torch.tensor(
                        [[len(active["observations"]) == 1]],
                        dtype=torch.bool,
                        device=self.engine.device,
                    )
                    posterior, _ = self.world.observe(
                        observation[None, None],
                        previous[None, None],
                        first,
                        initial=active["state"],
                        generator=self.rng,
                    )
                    active["state"] = posterior.map(lambda value: value[:, -1].detach())
                    action, _ = planet_cem_plan(
                        self.world, active["state"], generator=self.rng, **self.settings["planner"]
                    )
                    action = action[0] + self.settings["action_noise"] * torch.randn(
                        c.action_dim, device=self.engine.device, generator=self.rng
                    )
                    action = action.clamp(-1, 1)
                observation, reward, terminated, truncated, _ = self.environment.step(
                    action.cpu().clone()
                )
                observation = self._observation(observation)
                if (
                    type(terminated) is not bool
                    or type(truncated) is not bool
                    or not math.isfinite(float(reward))
                ):
                    raise ValueError(
                        "Simulator must return finite reward and separate bool terminated/truncated"
                    )
                truncated = truncated or (
                    len(active["observations"]) >= self.settings["max_episode_steps"]
                    and not terminated
                )
                active["observations"].append(observation)
                active["previous_actions"].append(action.cpu().clone())
                active["rewards"].append(torch.tensor(float(reward), dtype=torch.float32))
                active["is_first"].append(False)
                active["terminated"].append(terminated)
                active["truncated"].append(truncated)
                self.environment_steps += 1
                if terminated or truncated:
                    episode = {
                        key: torch.stack(active[key])
                        if key in {"observations", "previous_actions", "rewards"}
                        else torch.tensor(active[key], dtype=torch.bool)
                        for key in (
                            "observations",
                            "previous_actions",
                            "rewards",
                            "is_first",
                            "terminated",
                            "truncated",
                        )
                    }
                    self.replay.add(episode)
                    returns.append(float(episode["rewards"].sum()))
                    self.episodes += 1
                    self.active = None
        except Exception:
            self.failed = True
            raise
        return dict(
            episode_returns=returns,
            environment_steps=self.environment_steps,
            episodes=self.episodes,
        )

    def train_step(self):
        if self.failed:
            raise RuntimeError("Restore a complete checkpoint after failed collection")
        previous = deepcopy(self.replay.state_dict())
        batches, error = [], None
        try:
            batches = [
                self.replay.sample(self.settings["batch_size"], device=self.engine.device)
                for _ in range(self.engine.accumulation_steps)
            ]
        except Exception as exc:
            error = str(exc)
        errors = self.engine.parallel.world.gather_objects(error)
        if any(errors):
            self.replay.load_state_dict(previous)
            raise ValueError(f"Cannot sample PlaNet replay on all ranks: {errors}")
        try:
            result = self.engine.step(batches)
        except Exception:
            self.replay.load_state_dict(previous)
            if self.engine._failed:
                self.failed = True
            raise
        if result.updated:
            self.updates += 1
        else:
            self.replay.load_state_dict(previous)
        return result

    def state_dict(self):
        if self.failed:
            raise RuntimeError("Cannot certify a checkpoint after partial simulator failure")
        active = deepcopy(self.active)
        if active is not None:
            state = active["state"]
            active["state"] = dict(
                mean=state.mean.cpu(),
                stddev=state.stddev.cpu(),
                sample=state.sample.cpu(),
                belief=state.belief.cpu(),
                config_key=state.config_key,
            )
        return dict(
            settings=deepcopy(self.settings),
            environment=deepcopy(self.environment.state_dict()),
            replay=self.replay.state_dict(),
            active=active,
            environment_steps=self.environment_steps,
            episodes=self.episodes,
            updates=self.updates,
            rng=self.rng.get_state(),
        )

    def load_state_dict(self, state):
        if state["settings"] != self.settings or any(
            type(state[name]) is not int or state[name] < 0
            for name in ("environment_steps", "episodes", "updates")
        ):
            raise ValueError("PlaNet loop checkpoint configuration/counters differ")
        active = deepcopy(state["active"])
        if active is not None:
            encoded = active["state"]
            active["state"] = PlaNetState(
                **{
                    key: value.to(self.engine.device) if isinstance(value, torch.Tensor) else value
                    for key, value in encoded.items()
                }
            )
            self.world._check_state(active["state"], 1)
        rng = torch.Generator(device=self.engine.device)
        rng.set_state(state["rng"])
        self.replay.load_state_dict(state["replay"])
        self.environment.load_state_dict(deepcopy(state["environment"]))
        self.active, self.rng = active, rng
        self.environment_steps, self.episodes, self.updates = (
            state["environment_steps"],
            state["episodes"],
            state["updates"],
        )
        self.failed = False
