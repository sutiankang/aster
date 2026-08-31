"""World Models controller search in native imagined environments."""

from copy import deepcopy
import hashlib
import math
import torch

from ..core.serialization import canonical_json
from ..models.vmc import MDNRNN, MDNRNNConfig, VMCController, VMCControllerConfig, sample_mdn
from ..optimization.evolution import CMAES


def _fingerprint(model, mean, logvar):
    digest = hashlib.sha256(canonical_json(model.config.to_dict()).encode("utf-8"))
    tensors = dict(model.state_dict(), initial_mean=mean, initial_logvar=logvar)
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(canonical_json([name, list(value.shape), str(value.dtype)]).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def vmc_population_returns(
    world,
    controller_config,
    candidates,
    initial_mean,
    initial_logvar,
    *,
    episodes=16,
    horizon=2100,
    temperature=1.25,
    seed=0,
):

    c = world.config
    if not isinstance(c, MDNRNNConfig) or not isinstance(controller_config, VMCControllerConfig):
        raise ValueError("VMC evaluation needs a native MDN and explicit controller configuration")
    if (c.latent_size, c.hidden_size, c.action_dim) != (
        controller_config.latent_size,
        controller_config.hidden_size,
        controller_config.action_dim,
    ):
        raise ValueError("VMC controller and dynamics dimensions differ")
    if (
        any(type(value) is not int or value < 1 for value in (episodes, horizon))
        or type(seed) is not int
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Invalid VMC rollout protocol")
    if (
        candidates.ndim != 3
        or candidates.shape[1:] != (controller_config.feature_size, controller_config.action_dim)
        or not len(candidates)
        or not torch.isfinite(candidates).all()
    ):
        raise ValueError("VMC candidates require finite [P,features,actions] weights")
    if (
        initial_mean.ndim != 2
        or initial_mean.shape != initial_logvar.shape
        or not len(initial_mean)
        or initial_mean.shape[-1] != c.latent_size
        or any(
            not value.is_floating_point() or not torch.isfinite(value).all()
            for value in (initial_mean, initial_logvar)
        )
    ):
        raise ValueError("VMC initial state distribution is invalid")
    parameter = next(world.parameters())
    device = parameter.device
    population = len(candidates)
    rng = torch.Generator(device=device).manual_seed(seed)
    means, logvars = (
        initial_mean.to(device=device, dtype=torch.float32),
        initial_logvar.to(device=device, dtype=torch.float32),
    )

    weights = candidates.to(device=device, dtype=torch.float32).repeat_interleave(episodes, dim=0)
    indices = torch.randint(len(means), (population * episodes,), device=device, generator=rng)
    latent = means[indices] + (0.5 * logvars[indices]).exp() * torch.randn(
        population * episodes, c.latent_size, device=device, generator=rng
    )
    state = world.initial(population * episodes, device=device)
    restart = torch.ones(population * episodes, dtype=torch.bool, device=device)
    alive = torch.ones_like(restart)
    returns = torch.zeros(population * episodes, dtype=torch.float64, device=device)
    termination_step = torch.full(
        (population * episodes,), horizon, dtype=torch.int64, device=device
    )
    mode = world.training
    try:
        world.eval()
        for step in range(horizon):
            features = torch.cat(
                (latent, state.cell, state.hidden)
                if controller_config.include_cell
                else (latent, state.hidden),
                -1,
            )
            actions = torch.bmm(features[:, None], weights).squeeze(1).tanh()
            prediction = world(latent[:, None], actions[:, None], restart[:, None], state=state)
            next_latent, next_restart = sample_mdn(
                prediction, temperature=temperature, generator=rng
            )

            returns = returns + alive
            done = next_restart[:, 0] & alive
            termination_step = torch.where(done, step + 1, termination_step)
            alive = alive & ~done
            latent, restart, state = next_latent[:, 0], next_restart[:, 0], prediction.state
        return returns.reshape(population, episodes).mean(1).cpu(), dict(
            episode_returns=returns.reshape(population, episodes).cpu(),
            terminated=(~alive).reshape(population, episodes).cpu(),
            truncated=alive.reshape(population, episodes).cpu(),
            lengths=termination_step.reshape(population, episodes).cpu(),
            generated_model_steps=horizon,
            seed=seed,
        )
    finally:
        world.train(mode)


class VMCControllerSearch:
    def __init__(
        self,
        engine,
        initial_mean,
        initial_logvar,
        *,
        controller_config=None,
        population=64,
        episodes=16,
        horizon=2100,
        temperature=1.25,
        sigma=0.1,
        seed=0,
        weight_decay=0.0,
    ):
        if not isinstance(engine.model.config, MDNRNNConfig):
            raise ValueError("Controller search requires a native MDN Trainer")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Controller weight decay must be finite and nonnegative")
        c = engine.model.config
        self.controller_config = controller_config or VMCControllerConfig(
            c.latent_size, c.hidden_size, c.action_dim
        )
        weights = engine.export_state_dict(only_rank_zero=False)
        with torch.random.fork_rng(devices=[]):
            self.world = MDNRNN(c)
        self.world.load_state_dict(weights)
        self.world.to(engine.device).eval().requires_grad_(False)
        self.initial_mean = initial_mean.detach().cpu().clone()
        self.initial_logvar = initial_logvar.detach().cpu().clone()
        self.settings = dict(
            controller=self.controller_config.to_dict(),
            episodes=episodes,
            horizon=horizon,
            temperature=temperature,
            seed=seed,
            weight_decay=weight_decay,
            dynamics_and_initial_sha256=_fingerprint(
                self.world, self.initial_mean, self.initial_logvar
            ),
            dynamics_updates=engine.roles["model"].updates,
        )
        self.evolution = CMAES(
            torch.zeros(self.controller_config.feature_size * self.controller_config.action_dim),
            sigma=sigma,
            population=population,
            seed=seed,
        )
        self.generations = 0

    def step(self):
        if self.evolution.pending is not None:
            candidates = self.evolution.pending.clone()
        else:
            candidates = self.evolution.ask()
        weights = candidates.reshape(
            len(candidates), self.controller_config.feature_size, self.controller_config.action_dim
        )
        scores, diagnostics = vmc_population_returns(
            self.world,
            self.controller_config,
            weights,
            self.initial_mean,
            self.initial_logvar,
            episodes=self.settings["episodes"],
            horizon=self.settings["horizon"],
            temperature=self.settings["temperature"],
            seed=self.settings["seed"] + self.generations,
        )

        cost = -scores + self.settings["weight_decay"] * candidates.square().mean(-1)
        outcome = self.evolution.tell(cost)
        self.generations += 1
        return dict(
            outcome,
            generation=self.generations,
            mean_return=float(scores.mean()),
            best_return=float(scores.max()),
            diagnostics=diagnostics,
        )

    def controller(self, *, best=True):
        with torch.random.fork_rng(devices=[]):
            controller = VMCController(self.controller_config)
        parameters = self.evolution.best if best else self.evolution.mean
        with torch.no_grad():
            controller.weight.copy_(parameters.reshape_as(controller.weight))
        return controller.eval()

    def state_dict(self):

        return dict(
            settings=deepcopy(self.settings),
            generations=self.generations,
            evolution=self.evolution.state_dict(),
        )

    def load_state_dict(self, state):
        if (
            state["settings"] != self.settings
            or type(state["generations"]) is not int
            or state["generations"] < 0
        ):
            raise ValueError("VMC search dynamics/data/protocol identity changed")
        if state["evolution"]["evaluations"] != state["generations"] * self.evolution.population:
            raise ValueError("VMC search generation clock differs from CMA state")
        self.evolution.load_state_dict(state["evolution"])
        self.generations = state["generations"]
