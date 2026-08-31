"""Score-SDE objectives and native few-step diffusion integrators."""

from dataclasses import dataclass
import math
import torch
from torch import nn
from ..core import LossTerm, FieldOutput
from .generation import expand, mean_flat, randn_like


@dataclass(frozen=True)
class SDE:
    kind: str = "vp"
    beta_min: float = 0.1
    beta_max: float = 20.0
    sigma_min: float = 0.01
    sigma_max: float = 50.0

    def __post_init__(self):
        if (
            self.kind not in {"vp", "subvp", "ve"}
            or not 0 < self.beta_min <= self.beta_max
            or not 0 < self.sigma_min < self.sigma_max
        ):
            raise ValueError("Invalid score-SDE configuration")

    def marginal(self, time):
        if ((time < 0) | (time > 1)).any():
            raise ValueError("Continuous SDE time must be in [0,1]")
        if self.kind == "ve":
            return torch.ones_like(time), self.sigma_min * (self.sigma_max / self.sigma_min) ** time
        integrated = self.beta_min * time + 0.5 * (self.beta_max - self.beta_min) * time.square()
        signal = (-0.5 * integrated).exp()
        variance = -torch.expm1(-integrated)
        return signal, variance if self.kind == "subvp" else variance.sqrt()

    def coefficients(self, sample, time):
        if self.kind == "ve":
            _, sigma = self.marginal(time)
            return torch.zeros_like(sample), sigma * math.sqrt(
                2 * math.log(self.sigma_max / self.sigma_min)
            )
        beta = self.beta_min + time * (self.beta_max - self.beta_min)
        drift = -0.5 * expand(beta, sample) * sample
        if self.kind == "subvp":
            integrated = (
                self.beta_min * time + 0.5 * (self.beta_max - self.beta_min) * time.square()
            )
            diffusion = (beta * (-torch.expm1(-2 * integrated))).sqrt()
        else:
            diffusion = beta.sqrt()
        return drift, diffusion


class ScoreSDEObjective(nn.Module):
    def __init__(self, sde=SDE(), *, epsilon=1e-5, likelihood_weighting=False):
        super().__init__()
        if not 0 < epsilon < 1:
            raise ValueError("SDE loss excludes singular t=0")
        self.sde, self.epsilon, self.likelihood_weighting = sde, epsilon, likelihood_weighting

    def forward(self, model, batch):
        clean = batch["sample"]
        time = batch.get("time")
        noise = batch.get("noise")
        if time is None:
            time = torch.rand(len(clean), device=clean.device) * (1 - self.epsilon) + self.epsilon
        if (time < self.epsilon).any():
            raise ValueError("Score time falls outside trained SDE domain")
        if noise is None:
            noise = torch.randn_like(clean)
        signal, std = self.sde.marginal(time)
        noisy = expand(signal, clean) * clean + expand(std, clean) * noise
        output = model(noisy, time, batch.get("condition"))
        if output.prediction_type != "score":
            raise ValueError("Continuous score loss needs score parameterization")
        if self.likelihood_weighting:
            _, diffusion = self.sde.coefficients(noisy, time)
            losses = (
                mean_flat((output.prediction + noise / expand(std, noisy)).square())
                * diffusion.square()
            )
        else:
            losses = mean_flat((output.prediction * expand(std, noisy) + noise).square())
        return LossTerm(losses.sum(), losses.new_tensor(len(losses)), "sample", "score_sde")


@torch.no_grad()
def sample_score_sde(
    model,
    noise,
    *,
    sde=SDE(),
    steps=1000,
    epsilon=1e-3,
    probability_flow=False,
    corrector_steps=0,
    snr=0.16,
    condition=None,
    generator=None,
):
    if steps < 1 or not 0 < epsilon < 1 or corrector_steps < 0 or snr <= 0:
        raise ValueError("Invalid SDE solver settings")
    sample = noise.clone() * (sde.sigma_max if sde.kind == "ve" else 1.0)
    times = torch.linspace(1.0, epsilon, steps + 1, device=sample.device, dtype=sample.dtype)

    def score(x, t):
        result = model(x, t, condition)
        if result.prediction_type != "score":
            raise ValueError("SDE solver expects score output")
        return result.prediction

    for high, low in zip(times[:-1], times[1:]):
        t = high.expand(len(sample))
        for _ in range(corrector_steps):
            gradient = score(sample, t)
            perturbation = randn_like(sample, generator)
            gradient_norm = gradient.flatten(1).norm(dim=-1).mean().clamp_min(1e-8)
            noise_norm = perturbation.flatten(1).norm(dim=-1).mean()
            step = 2 * (snr * noise_norm / gradient_norm).square()
            sample = sample + step * gradient + (2 * step).sqrt() * perturbation
        drift, diffusion = sde.coefficients(sample, t)
        reverse_drift = drift - expand(diffusion.square(), sample) * score(sample, t) * (
            0.5 if probability_flow else 1.0
        )
        dt = low - high
        mean = sample + dt * reverse_drift
        sample = (
            mean
            if probability_flow
            else mean + expand(diffusion, sample) * (-dt).sqrt() * randn_like(sample, generator)
        )
    return mean


@torch.no_grad()
def sample_dpmpp_2m(denoiser, noise, sigmas, *, condition=None):
    """Integrate denoised predictions in -log(sigma) time and reuse the previous prediction."""
    if (
        sigmas.ndim != 1
        or len(sigmas) < 3
        or sigmas[-1] != 0
        or not torch.isfinite(sigmas).all()
        or not (sigmas[:-1] > sigmas[1:]).all()
    ):
        raise ValueError("DPM++ sigmas must strictly decrease to zero")
    sample = noise * sigmas[0].to(noise)
    previous = previous_step = None
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        clean = denoiser(sample, sigma.to(sample).expand(len(sample)), condition)
        if clean.shape != sample.shape:
            raise ValueError("DPM++ denoiser must return clean sample")
        if next_sigma == 0:
            sample = clean
        else:
            step = (sigma.log() - next_sigma.log()).to(sample)
            estimate = clean
            if previous is not None:
                ratio = previous_step / step
                estimate = (1 + 1 / (2 * ratio)) * clean - previous / (2 * ratio)
            sample = (next_sigma / sigma).to(sample) * sample - (-step).expm1() * estimate
            previous_step = step
        previous = clean
    return sample


class ProgressiveDistillationObjective(nn.Module):
    """Invert the transport equation so one student step matches two deterministic
    teacher DDIM steps."""

    def __init__(self, teacher, schedule):
        super().__init__()
        self.teacher, self.schedule = teacher.eval().requires_grad_(False), schedule

    def forward(self, model, batch):
        clean, high, middle, low = (
            batch["sample"],
            batch["time_high"],
            batch["time_middle"],
            batch["time_low"],
        )
        if not (high > middle).all() or not (middle > low).all() or (low < -1).any():
            raise ValueError("Progressive distillation needs ordered high>middle>low>=-1")
        noise = batch.get("noise")
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = self.schedule.noise(clean, high, noise)
        condition = batch.get("condition")

        def transport(x, t, next_t):
            output = self.teacher(x, self.schedule.model_time(t), condition)
            predicted, eps = self.schedule.clean_and_noise(output, x, t)
            a = self.schedule.at("alpha_bar", next_t.clamp_min(0), x)
            a = torch.where(expand(next_t == -1, x), torch.ones_like(a), a)
            return a.sqrt() * predicted + (1 - a).sqrt() * eps

        self.teacher.eval()
        with torch.no_grad():
            intermediate = transport(noisy, high, middle)
            endpoint = transport(intermediate, middle, low)
            a = self.schedule.at("alpha_bar", high, noisy)
            next_a = self.schedule.at("alpha_bar", low.clamp_min(0), noisy)
            next_a = torch.where(expand(low == -1, noisy), torch.ones_like(next_a), next_a)
            ratio = ((1 - next_a) / (1 - a)).sqrt()
            denominator = next_a.sqrt() - ratio * a.sqrt()
            if (denominator.abs() < 1e-8).any():
                raise ValueError("Degenerate progressive transport interval")
            target = (endpoint - ratio * noisy) / denominator
        output = model(noisy, self.schedule.model_time(high), condition)
        predicted, _ = self.schedule.clean_and_noise(output, noisy, high)
        losses = mean_flat((predicted - target).square())
        return LossTerm(
            losses.sum(), losses.new_tensor(len(losses)), "sample", "progressive_distillation"
        )
