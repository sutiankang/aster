"""PlaNet likelihoods, Gaussian KL, overshooting, and pixel preprocessing."""

import math
import torch
from torch import nn

from ..core import LossBundle, LossTerm
from ..models.planet import PlaNetConfig


def preprocess_planet_images(images, *, bits=5, generator=None):
    if (
        not isinstance(images, torch.Tensor)
        or images.dtype != torch.uint8
        or type(bits) is not int
        or not 1 <= bits <= 8
    ):
        raise ValueError("PlaNet preprocessing expects uint8 pixels and 1..8 bits")
    bins = 2**bits
    quantized = torch.floor(images.float() / (2 ** (8 - bits))) / bins

    return (
        quantized + torch.rand(images.shape, device=images.device, generator=generator) / bins - 0.5
    )


def postprocess_planet_images(images, *, bits=5):
    if (
        not images.is_floating_point()
        or not torch.isfinite(images).all()
        or type(bits) is not int
        or not 1 <= bits <= 8
    ):
        raise ValueError("Invalid PlaNet image output/bits")
    return (
        (torch.floor((images + 0.5) * (2**bits)) * (256 / (2**bits))).clamp(0, 255).to(torch.uint8)
    )


def gaussian_state_kl(posterior, prior, *, stop_posterior=False):
    """Compute KL(q || p), summing latent coordinates while preserving batch/time axes."""
    qm, qs = posterior.mean.float(), posterior.stddev.float()
    pm, ps = prior.mean.float(), prior.stddev.float()
    if stop_posterior:
        qm, qs = qm.detach(), qs.detach()
    return (torch.log(ps / qs) + (qs.square() + (qm - pm).square()) / (2 * ps.square()) - 0.5).sum(
        -1
    )


class PlaNetObjective(nn.Module):
    def __init__(
        self,
        *,
        sequence_length=50,
        free_nats=3.0,
        image_weight=1.0,
        reward_weight=10.0,
        divergence_weight=1.0,
        overshooting_distance=0,
        overshooting_weight=0.0,
        stop_overshooting_posterior=True,
    ):
        super().__init__()
        if (
            type(sequence_length) is not int
            or sequence_length < 1
            or type(overshooting_distance) is not int
            or not 0 <= overshooting_distance <= sequence_length
        ):
            raise ValueError("Invalid PlaNet sequence/overshooting length")
        if any(
            not math.isfinite(v) or v < 0
            for v in (
                free_nats,
                image_weight,
                reward_weight,
                divergence_weight,
                overshooting_weight,
            )
        ):
            raise ValueError("PlaNet loss weights/free_nats must be finite and nonnegative")
        if overshooting_weight and overshooting_distance < 2:
            raise ValueError("Latent overshooting requires distance >=2")
        if type(stop_overshooting_posterior) is not bool:
            raise ValueError("Overshooting posterior stop flag must be boolean")
        self.settings = dict(
            sequence_length=sequence_length,
            free_nats=free_nats,
            image_weight=image_weight,
            reward_weight=reward_weight,
            divergence_weight=divergence_weight,
            overshooting_distance=overshooting_distance,
            overshooting_weight=overshooting_weight,
            stop_overshooting_posterior=stop_overshooting_posterior,
        )

    def config_dict(self):
        return dict(
            type="planet",
            **self.settings,
            normalization="valid_transitions_and_valid_overshooting_pairs",
            previous_action_alignment=True,
            likelihood="unit_gaussian_with_constant",
        )

    def _validate(self, model, batch):
        c, length = model.config, self.settings["sequence_length"]
        if not isinstance(c, PlaNetConfig):
            raise ValueError("PlaNetObjective requires the continuous Gaussian PlaNet model")
        required = {"observations", "previous_actions", "is_first", "rewards"}
        if not required <= set(batch) or set(batch) - required - {
            "valid",
            "prior_noise",
            "posterior_noise",
            "overshooting_noise",
        }:
            raise ValueError(
                "Unknown/missing PlaNet batch fields; actions must be previous_actions"
            )
        obs, actions, first, rewards = (
            batch[k] for k in ("observations", "previous_actions", "is_first", "rewards")
        )
        if (
            not isinstance(obs, torch.Tensor)
            or obs.ndim != 2 + len(c.observation_shape)
            or tuple(obs.shape[1:]) != (length, *c.observation_shape)
            or not len(obs)
        ):
            raise ValueError(
                "PlaNet observations must match the explicit sequence length and observation shape"
            )
        b, device = len(obs), obs.device
        shapes = {
            "observations": obs.shape,
            "previous_actions": (b, length, c.action_dim),
            "rewards": (b, length),
        }
        for key, shape in shapes.items():
            value = batch[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.device != device
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"PlaNet {key} must be finite floating-point, aligned shape/device"
                )
        for key in ("is_first", "valid"):
            value = batch.get(key)
            if value is not None and (
                not isinstance(value, torch.Tensor)
                or value.shape != (b, length)
                or value.dtype != torch.bool
                or value.device != device
            ):
                raise ValueError(f"PlaNet {key} must be aligned bool [B,T]")
        valid = batch.get("valid", torch.ones_like(first))
        if (~valid[:, 0]).any() or ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError("PlaNet valid mask must be a nonempty contiguous prefix per sequence")
        for key in ("prior_noise", "posterior_noise", "overshooting_noise"):
            if key in batch:
                shape = (
                    (b, length, c.state_size)
                    if key != "overshooting_noise"
                    else (b, length, self.settings["overshooting_distance"], c.state_size)
                )
                noise = batch[key]
                if (
                    not isinstance(noise, torch.Tensor)
                    or noise.shape != shape
                    or noise.device != device
                    or not noise.is_floating_point()
                    or not torch.isfinite(noise).all()
                ):
                    raise ValueError(
                        f"PlaNet {key} must have its declared finite latent-noise shape"
                    )

    def preflight_microbatches(self, model, batches):

        for batch in batches:
            self._validate(model, batch)
        return batches

    def _overshooting(self, model, result, batch, valid):
        """Roll forward from the previous posterior time and compare distances 2 through D."""
        posterior = result["state"]
        b, t = valid.shape
        initial = model.initial(b, device=valid.device, dtype=posterior.sample.dtype)

        names = ("mean", "stddev", "sample", "belief")
        from ..models.planet import PlaNetState

        state = PlaNetState(
            *(
                torch.cat(
                    (getattr(initial, key)[:, None], getattr(posterior, key)[:, :-1]), 1
                ).flatten(0, 1)
                for key in names
            ),
            posterior.config_key,
        )
        active = valid.clone()
        total = posterior.mean.sum() * 0
        count = torch.zeros((), dtype=torch.int64, device=valid.device)
        for offset in range(self.settings["overshooting_distance"]):
            width = t - offset
            action = torch.zeros_like(batch["previous_actions"])
            first = torch.ones_like(batch["is_first"])
            action[:, :width] = batch["previous_actions"][:, offset:]
            first[:, :width] = batch["is_first"][:, offset:]
            step_valid = torch.zeros_like(valid)
            step_valid[:, :width] = valid[:, offset:]
            active = active & step_valid
            if offset:
                active = active & ~first
            keep = (~first).flatten()[:, None]
            state = state.map(lambda value: value * keep)
            noise = batch.get("overshooting_noise")
            noise = None if noise is None else noise[:, :, offset].flatten(0, 1)
            state = model.transition(state, action.flatten(0, 1) * keep, noise=noise)
            if offset:
                target = posterior.map(lambda value: value[:, offset:].flatten(0, 1))
                predicted = state.map(
                    lambda value: value.reshape(b, t, -1)[:, :width].flatten(0, 1)
                )
                kl = gaussian_state_kl(
                    target, predicted, stop_posterior=self.settings["stop_overshooting_posterior"]
                )
                kl = (kl - self.settings["free_nats"]).clamp_min(0).reshape(b, width)
                mask = active[:, :width]
                total = total + kl.masked_select(mask).sum()
                count = count + mask.sum(dtype=torch.int64)
        return LossTerm(
            total, count, "valid_latent_pair", "overshooting", self.settings["overshooting_weight"]
        )

    def forward(self, model, batch):
        self._validate(model, batch)
        result = model(
            batch["observations"],
            batch["previous_actions"],
            batch["is_first"],
            prior_noise=batch.get("prior_noise"),
            posterior_noise=batch.get("posterior_noise"),
        )
        valid = batch.get("valid", torch.ones_like(batch["is_first"]))
        count = valid.sum(dtype=torch.int64)

        image = (
            (
                0.5 * (result["reconstruction"].float() - batch["observations"].float()).square()
                + 0.5 * math.log(2 * math.pi)
            )
            .flatten(2)
            .sum(-1)
        )
        reward = 0.5 * (
            result["reward"].float() - batch["rewards"].float()
        ).square() + 0.5 * math.log(2 * math.pi)
        divergence = (
            gaussian_state_kl(result["state"], result["prior"]) - self.settings["free_nats"]
        ).clamp_min(0)
        terms = [
            LossTerm(
                value.masked_select(valid).sum(),
                count,
                "valid_transition",
                name,
                self.settings[f"{name}_weight"],
            )
            for name, value in (("image", image), ("reward", reward), ("divergence", divergence))
        ]
        if self.settings["overshooting_weight"]:
            terms.append(self._overshooting(model, result, batch, valid))
        return LossBundle(tuple(terms))
