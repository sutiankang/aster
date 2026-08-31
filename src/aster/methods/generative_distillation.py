"""Consistency, distribution-matching, and Drifting objectives for generative acceleration."""

from __future__ import annotations
import math
import torch
from torch import nn
from ..core import LossTerm
from .generation import expand, mean_flat, edm_denoise, sample_flow


def consistency_prediction(model, noisy, sigma, condition=None, *, sigma_data=0.5, sigma_min=0.002):
    if sigma_data <= 0 or sigma_min <= 0 or (sigma < sigma_min).any():
        raise ValueError("Invalid consistency sigma")
    t = expand(sigma, noisy)
    offset = t - sigma_min
    c_skip = sigma_data**2 / (offset.square() + sigma_data**2)
    c_out = sigma_data * offset / (t.square() + sigma_data**2).sqrt()
    output = model(noisy / (t.square() + sigma_data**2).sqrt(), sigma.log() / 4, condition)
    if output.prediction_type != "consistency_residual":
        raise ValueError("Consistency boundary wrapper requires consistency_residual")
    residual = output.prediction
    return c_skip * noisy + c_out * residual


class ConsistencyDistillationObjective(nn.Module):
    """Use the teacher ODE for neighboring times and the EMA target for the
    boundary-consistent mapping; both targets are explicitly frozen."""

    def __init__(self, teacher, target, *, sigma_data=0.5, sigma_min=0.002, loss="pseudo_huber"):
        super().__init__()
        if loss not in {"mse", "pseudo_huber"}:
            raise ValueError("Unknown consistency loss")
        self.teacher, self.target = (
            teacher.eval().requires_grad_(False),
            target.eval().requires_grad_(False),
        )
        self.sigma_data, self.sigma_min, self.loss = sigma_data, sigma_min, loss

    def forward(self, model, batch):
        clean, high, low = batch["sample"], batch["sigma_high"], batch["sigma_low"]
        if (
            high.shape != (len(clean),)
            or low.shape != high.shape
            or not (high > low).all()
            or (low < self.sigma_min).any()
        ):
            raise ValueError("Need ordered adjacent consistency levels")
        noise = batch.get("noise")
        if noise is None:
            noise = torch.randn_like(clean)
        condition = batch.get("condition")
        x = clean + expand(high, clean) * noise
        self.teacher.eval()
        self.target.eval()
        with torch.no_grad():
            first = (
                x - edm_denoise(self.teacher, x, high, condition, sigma_data=self.sigma_data)
            ) / expand(high, x)
            euler = x + expand(low - high, x) * first
            second = (
                euler - edm_denoise(self.teacher, euler, low, condition, sigma_data=self.sigma_data)
            ) / expand(low, x)
            next_x = x + expand(low - high, x) * (first + second) / 2
            target = consistency_prediction(
                self.target,
                next_x,
                low,
                condition,
                sigma_data=self.sigma_data,
                sigma_min=self.sigma_min,
            )
        predicted = consistency_prediction(
            model, x, high, condition, sigma_data=self.sigma_data, sigma_min=self.sigma_min
        )
        squared = (predicted.float() - target.float()).square()
        if self.loss == "pseudo_huber":
            constant = 0.00054 * math.sqrt(clean[0].numel())
            squared = (squared + constant**2).sqrt() - constant
        values = mean_flat(squared)
        return LossTerm(values.sum(), values.new_tensor(len(values)), "sample", "consistency")


def dmd_surrogate(generated, real_denoised, fake_denoised, *, normalize=True, epsilon=1e-6):
    """Construct the clean-space DMD2 score-difference surrogate gradient."""
    if (
        generated.shape != real_denoised.shape
        or generated.shape != fake_denoised.shape
        or epsilon <= 0
    ):
        raise ValueError("DMD tensors must align")
    with torch.no_grad():
        gradient = fake_denoised.float() - real_denoised.float()
        if normalize:
            denominator = mean_flat(
                (generated.detach().float() - real_denoised.float()).abs()
            ).clamp_min(epsilon)
            gradient = gradient / expand(denominator, gradient)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite DMD target; do not silently replace with zeros")
        target = generated.detach().float() - gradient
    losses = 0.5 * mean_flat((generated.float() - target).square())
    return LossTerm(losses.sum(), losses.new_tensor(len(losses)), "sample", "distribution_matching")


class DMDGeneratorObjective(nn.Module):
    def __init__(self, real_score, fake_score, *, sigma_data=0.5, generator_time=1.0):
        super().__init__()
        self.real_score, self.fake_score = real_score, fake_score
        self.sigma_data, self.generator_time = sigma_data, generator_time

    def forward(self, generator, batch):
        noise, sigma, condition = batch["noise"], batch["sigma"], batch.get("condition")
        output = generator(noise, noise.new_full((len(noise),), self.generator_time), condition)
        if output.prediction_type != "x0":
            raise ValueError("DMD generator must emit direct clean samples")
        generated = output.prediction
        with torch.no_grad():
            perturbation = batch.get("score_noise")
            if perturbation is None:
                perturbation = torch.randn_like(generated)
            noisy = generated.detach() + expand(sigma, generated) * perturbation
            real = edm_denoise(self.real_score, noisy, sigma, condition, sigma_data=self.sigma_data)
            fake = edm_denoise(self.fake_score, noisy, sigma, condition, sigma_data=self.sigma_data)
        return dmd_surrogate(generated, real, fake)


def drifting_loss(
    generated,
    positive,
    negative=None,
    *,
    radii=(0.02, 0.05, 0.2),
    generated_weights=None,
    positive_weights=None,
    negative_weights=None,
    statistics_group=None,
):
    """Apply the Drifting feature-force formula to [batch, samples, features]."""
    if (
        generated.ndim != 3
        or positive.ndim != 3
        or generated.shape[::2] != positive.shape[::2]
        or min(generated.shape) < 1
        or positive.shape[1] < 1
    ):
        raise ValueError("Drifting expects nonempty aligned BND features")
    if not radii or any(not math.isfinite(r) or r <= 0 for r in radii):
        raise ValueError("Drifting radii must be positive")
    b, n, d = generated.shape
    negative = generated[:, :0].detach() if negative is None else negative
    if negative.ndim != 3 or negative.shape[::2] != generated.shape[::2]:
        raise ValueError("Negative feature dimensions must match")
    tensors = (generated, negative, positive)
    if any(
        value.device != generated.device
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
        for value in tensors
    ):
        raise ValueError("Drifting features must be finite floating tensors on one device")
    supplied = (generated_weights, negative_weights, positive_weights)
    weights = []
    for value, weight in zip(tensors, supplied):
        weight = torch.ones(value.shape[:2], device=value.device) if weight is None else weight
        if (
            weight.shape != value.shape[:2]
            or weight.device != value.device
            or not torch.isfinite(weight).all()
            or (weight < 0).any()
        ):
            raise ValueError("Invalid drifting weights")
        weights.append(weight.float())
    if (weights[-1].sum(-1) <= 0).any():
        raise ValueError("Each group needs positive mass")
    with torch.no_grad():
        current = generated.detach().float()
        targets = torch.cat((current, negative.detach().float(), positive.detach().float()), dim=1)
        mass = torch.cat(weights, dim=1)
        squared = (
            current.square().sum(-1, keepdim=True)
            + targets.square().sum(-1)[:, None]
            - 2 * (current @ targets.transpose(-1, -2))
        )
        distance = squared.clamp_min(1e-8).sqrt()

        statistics = torch.stack(
            (
                (distance * mass[:, None]).sum(),
                distance.new_tensor(distance.numel()),
                mass.sum(),
                mass.new_tensor(mass.numel()),
            )
        )
        if statistics_group is not None:
            statistics_group.all_reduce(statistics)
        scale = (statistics[0] / statistics[1]) / (statistics[2] / statistics[3])
        coordinate_scale = (scale / math.sqrt(d)).clamp_min(1e-3)
        centers, points = current / coordinate_scale, targets / coordinate_scale
        normalized = distance / scale.clamp_min(1e-3)
        normalized[:, torch.arange(n), torch.arange(n)] += 100.0
        total_force = torch.zeros_like(centers)
        info = {"scale": scale}
        split = n + negative.shape[1]
        for radius in radii:
            logits = -normalized / radius
            affinity = (logits.softmax(-1) * logits.softmax(-2)).clamp_min(1e-6).sqrt() * mass[
                :, None
            ]
            negative_affinity, positive_affinity = affinity[:, :, :split], affinity[:, :, split:]
            coefficients = torch.cat(
                (
                    -negative_affinity * positive_affinity.sum(-1, keepdim=True),
                    positive_affinity * negative_affinity.sum(-1, keepdim=True),
                ),
                dim=-1,
            )
            force = coefficients @ points - coefficients.sum(-1, keepdim=True) * centers
            magnitude_parts = torch.stack((force.square().sum(), force.new_tensor(force.numel())))
            if statistics_group is not None:
                statistics_group.all_reduce(magnitude_parts)
            magnitude = magnitude_parts[0] / magnitude_parts[1]
            total_force += force / magnitude.clamp_min(1e-8).sqrt()
            info[f"force_energy_{radius}"] = magnitude
        target = centers + total_force
    values = (generated.float() / coordinate_scale - target).square().mean((-1, -2))
    return LossTerm(
        values.sum(), torch.tensor(b, device=values.device, dtype=torch.int64), "group", "drifting"
    ), info


class DriftingObjective(nn.Module):
    """A pre-grouped feature objective; queueing, guidance sampling, and distributed
    statistics belong to DriftingMethod."""

    def __init__(self, encoder=None, *, radii=(0.02, 0.05, 0.2), generator_time=1.0):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False) if encoder is not None else None
        self.radii, self.generator_time = tuple(radii), generator_time

    def forward(self, generator, batch):
        noise = batch["noise"]
        groups = batch["positive"].shape[0]
        output = generator(
            noise, noise.new_full((len(noise),), self.generator_time), batch.get("condition")
        )
        if output.prediction_type != "x0":
            raise ValueError("Drifting objective requires a direct x0 generator")
        generated = output.prediction
        if len(generated) % groups:
            raise ValueError("Generated samples must partition into condition groups")
        positive = batch["positive"]
        if self.encoder is not None:
            self.encoder.eval()

            generated = self.encoder(generated)
            with torch.no_grad():
                positive = self.encoder(positive.flatten(0, 1)).reshape(
                    groups, positive.shape[1], -1
                )
        generated = generated.reshape(groups, len(noise) // groups, -1)
        positive = positive.reshape(groups, positive.shape[1], -1)
        return drifting_loss(
            generated,
            positive,
            batch.get("negative_features"),
            radii=self.radii,
            generated_weights=batch.get("generated_weights"),
            positive_weights=batch.get("positive_weights"),
            negative_weights=batch.get("negative_weights"),
        )[0]


@torch.no_grad()
def reflow_pairs(
    teacher, noise, *, condition=None, steps=50, solver="heun", direction="noise_to_data"
):
    """Preserve paired source/target endpoints so subsequent flow training uses the
    same coupling rather than independently shuffled noise and data."""
    endpoint = sample_flow(
        teacher, noise, steps=steps, solver=solver, direction=direction, condition=condition
    )
    return {"noise": noise.detach(), "sample": endpoint.detach(), "condition": condition}


class DMDMethod:
    """Coordinate generator, frozen real score, and online fake-score fitting.
    This is a multi-phase update, not an isolated KL term."""

    def __init__(
        self, engine, real_score, fake_score, *, sigma_data=0.5, generator_time=1.0, fake_updates=1
    ):
        from .generation import EDMObjective

        if type(fake_updates) is not int or fake_updates < 1:
            raise ValueError("DMD requires positive integer fake-score update count")
        if (
            any(
                type(v) not in {int, float} or not math.isfinite(v)
                for v in (sigma_data, generator_time)
            )
            or sigma_data <= 0
        ):
            raise ValueError("DMD requires finite time and positive score scale")
        self.engine, self.fake_updates, self.generator_time = engine, fake_updates, generator_time
        self.sigma_data, self._round_in_progress = sigma_data, False
        self.real_score = engine.add_role("real_score", real_score, trainable=False)
        self.fake_score = engine.add_role("fake_score", fake_score)
        self.generator_objective = DMDGeneratorObjective(
            self.real_score, self.fake_score, sigma_data=sigma_data, generator_time=generator_time
        )
        self.fake_objective = EDMObjective(sigma_data=sigma_data)
        self.updates = 0
        engine.register_state("dmd_method", self)

    def update(self, microbatches):

        if self._round_in_progress:
            raise RuntimeError("DMD round is incomplete; restore a complete checkpoint")
        self.state_dict()
        batches = list(microbatches)
        fake_results = []
        if len(batches) != self.engine.accumulation_steps:
            raise ValueError("DMD microbatches must match accumulation_steps")
        self._round_in_progress = True
        for _ in range(self.fake_updates):
            generated_batches = []
            with torch.no_grad():
                for batch in batches:
                    noise = batch["noise"]
                    condition = batch.get("condition")
                    generated = self.engine.model(
                        noise, noise.new_full((len(noise),), self.generator_time), condition
                    ).prediction
                    generated_batches.append({"sample": generated.detach(), "condition": condition})
            fake_result = self.engine.phase(
                "fake_score",
                role="fake_score",
                objective=self.fake_objective,
                microbatches=generated_batches,
                freeze_roles=("model",),
            )
            if not fake_result.updated:
                raise RuntimeError("DMD fake-score phase skipped; restore a complete checkpoint")
            fake_results.append(fake_result)
        generator_result = self.engine.phase(
            "generator",
            objective=self.generator_objective,
            microbatches=batches,
            freeze_roles=("fake_score", "real_score"),
        )
        if not generator_result.updated:
            raise RuntimeError("DMD generator phase skipped; restore a complete checkpoint")
        self.updates += 1
        self._round_in_progress = False
        return {"fake_score": fake_results, "generator": generator_result}

    def state_dict(self):
        if self._round_in_progress:
            raise RuntimeError("Cannot save/export an incomplete DMD round")
        if (
            self.generator_objective.generator_time != self.generator_time
            or self.generator_objective.sigma_data != self.sigma_data
            or self.fake_objective.sigma_data != self.sigma_data
        ):
            raise ValueError("DMD objective controls changed during training")
        return {
            "schema_version": 1,
            "fake_updates": self.fake_updates,
            "generator_time": self.generator_time,
            "sigma_data": self.sigma_data,
            "fake_score_objective": self.fake_objective.config_dict(),
            "updates": self.updates,
            "complete_round": True,
        }

    def load_state_dict(self, state):
        if (
            set(state)
            != {
                "schema_version",
                "fake_updates",
                "generator_time",
                "sigma_data",
                "fake_score_objective",
                "updates",
                "complete_round",
            }
            or state["schema_version"] != 1
            or state["complete_round"] is not True
        ):
            raise ValueError("DMD checkpoint lacks a complete-round semantic contract")
        if (state["fake_updates"], state["generator_time"], state["sigma_data"]) != (
            self.fake_updates,
            self.generator_time,
            self.sigma_data,
        ):
            raise ValueError("DMD method settings changed")
        if type(state["updates"]) is not int or state["updates"] < 0:
            raise ValueError("Invalid completed DMD round count")
        if (
            self.generator_objective.generator_time != self.generator_time
            or self.generator_objective.sigma_data != self.sigma_data
            or self.fake_objective.sigma_data != self.sigma_data
        ):
            raise ValueError("DMD objective controls changed during restore")
        if state["fake_score_objective"] != self.fake_objective.config_dict():
            raise ValueError("DMD fake-score objective settings changed")
        self.updates = state["updates"]
        self._round_in_progress = False
