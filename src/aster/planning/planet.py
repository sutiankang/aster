"""PlaNet CEM-MPC: replan after observation and execute only the first action."""

import math
import torch

from ..models.planet import PlaNetConfig


@torch.no_grad()
def planet_cem_plan(
    world,
    initial,
    *,
    horizon=12,
    population=1000,
    elites=100,
    iterations=10,
    action_low=-1.0,
    action_high=1.0,
    generator=None,
):
    if not isinstance(world.config, PlaNetConfig):
        raise ValueError("PlaNet CEM requires its Gaussian RSSM")
    if (
        any(type(v) is not int or v < 1 for v in (horizon, population, elites))
        or type(iterations) is not int
        or iterations < 0
        or elites > population
    ):
        raise ValueError("Invalid PlaNet CEM population/horizon/iterations")
    if not all(math.isfinite(v) for v in (action_low, action_high)) or not action_low < action_high:
        raise ValueError("Invalid PlaNet CEM action bounds")
    b, device = initial.sample.shape[0], initial.sample.device
    world._check_state(initial, b)
    mean = torch.zeros(
        b, horizon, world.config.action_dim, device=device, dtype=initial.sample.dtype
    )
    stddev = torch.ones_like(mean)
    indices = torch.arange(b, device=device).repeat_interleave(population)

    starts = initial.reorder(indices)
    mode = world.training
    scores = None
    try:
        world.eval()
        for _ in range(iterations):
            proposals = (
                mean[:, None]
                + stddev[:, None]
                * torch.randn(
                    b,
                    population,
                    horizon,
                    world.config.action_dim,
                    device=device,
                    dtype=mean.dtype,
                    generator=generator,
                )
            ).clamp(action_low, action_high)
            predicted = world.imagine(starts, proposals.flatten(0, 1), generator=generator)

            rewards = world.reward_head(predicted.features).squeeze(-1)
            scores = rewards.sum(-1).reshape(b, population)
            if not torch.isfinite(scores).all():
                raise ValueError("PlaNet CEM produced nonfinite predicted returns")
            chosen = scores.topk(elites, dim=1, sorted=False).indices
            best = proposals.gather(
                1, chosen[..., None, None].expand(b, elites, horizon, world.config.action_dim)
            )
            mean = best.mean(1)
            stddev = (best.var(1, correction=0) + 1e-6).sqrt()

        return mean[:, 0], dict(
            plan=mean,
            stddev=stddev,
            predicted_best_return=None if scores is None else scores.max(1).values,
            model_rollout_steps=iterations * horizon,
            trajectories=iterations * b * population,
        )
    finally:
        world.train(mode)
