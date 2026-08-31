"""Diffusion and flow objectives with explicit prediction types and time directions."""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..core import FieldOutput, LossTerm, LossBundle


def expand(value, sample):
    return value.reshape(-1, *([1] * (sample.ndim - 1)))


def mean_flat(value):
    return value.flatten(1).mean(1)


def randn_like(value, generator=None):
    return torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)


def guided_field(
    model, sample, time, condition=None, *, guidance_scale=1.0, negative_condition=None, rescale=0.0
):
    """Combine conditional and unconditional predictions in the same parameterization.
    Guidance rescaling is an explicit approximate transform."""
    if not math.isfinite(guidance_scale) or not 0 <= rescale <= 1:
        raise ValueError("Invalid guidance configuration")
    positive = model(sample, time, condition)
    if guidance_scale == 1:
        return positive
    negative = model(sample, time, negative_condition)
    if negative.prediction_type != positive.prediction_type:
        raise ValueError("CFG prediction types must match")
    guided = negative.prediction + guidance_scale * (positive.prediction - negative.prediction)
    if rescale:
        dims = tuple(range(1, guided.ndim))
        scaled = (
            guided
            * positive.prediction.std(dims, keepdim=True, correction=0)
            / guided.std(dims, keepdim=True, correction=0).clamp_min(1e-6)
        )
        guided = rescale * scaled + (1 - rescale) * guided
    return FieldOutput(guided, positive.prediction_type)


class DiffusionSchedule(nn.Module):
    def __init__(self, betas, *, timestep_map=None):
        super().__init__()
        betas = torch.as_tensor(betas, dtype=torch.float64)
        if (
            betas.ndim != 1
            or len(betas) < 2
            or not torch.isfinite(betas).all()
            or not ((betas > 0) & (betas < 1)).all()
        ):
            raise ValueError("Need at least two beta values strictly in (0,1)")
        cumulative = (1 - betas).cumprod(0)
        previous = F.pad(cumulative[:-1], (1, 0), value=1)
        posterior = betas * (1 - previous) / (1 - cumulative)
        values = {
            "betas": betas,
            "alpha_bar": cumulative,
            "previous_alpha_bar": previous,
            "posterior_variance": posterior,
            "posterior_log_variance": torch.cat((posterior[1:2], posterior[1:])).log(),
            "posterior_x0": betas * previous.sqrt() / (1 - cumulative),
            "posterior_xt": (1 - previous) * (1 - betas).sqrt() / (1 - cumulative),
        }
        for name, value in values.items():
            self.register_buffer(name, value)
        mapping = (
            torch.arange(len(betas))
            if timestep_map is None
            else torch.as_tensor(timestep_map, dtype=torch.long)
        )
        if mapping.shape != betas.shape or not (mapping[1:] > mapping[:-1]).all():
            raise ValueError("Timestep map must be strictly increasing")
        self.register_buffer("timestep_map", mapping)

    @classmethod
    def create(cls, steps=1000, name="cosine"):
        if steps < 2:
            raise ValueError("At least two diffusion steps are required")
        if name == "linear":
            betas = torch.linspace(
                0.0001 * 1000 / steps, 0.02 * 1000 / steps, steps, dtype=torch.float64
            )
        elif name == "cosine":
            t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
            a = ((t + 0.008) / 1.008 * math.pi / 2).cos().square()
            betas = (1 - a[1:] / a[:-1]).clamp(max=0.999)
        else:
            raise ValueError("Unknown beta schedule")
        return cls(betas)

    def __len__(self):
        return len(self.betas)

    def at(self, name, time, sample):
        if (
            time.shape != (sample.shape[0],)
            or time.dtype != torch.long
            or (time < 0).any()
            or (time >= len(self)).any()
        ):
            raise ValueError("Invalid discrete timestep")
        return expand(getattr(self, name).to(sample.device)[time].to(sample.dtype), sample)

    def model_time(self, time):
        return self.timestep_map.to(time.device)[time]

    def noise(self, clean, time, noise):
        a = self.at("alpha_bar", time, clean)
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    def clean_and_noise(self, output, noisy, time):
        a = self.at("alpha_bar", time, noisy)
        signal, sigma = a.sqrt(), (1 - a).sqrt()
        value = output.prediction
        if output.prediction_type == "epsilon":
            clean = (noisy - sigma * value) / signal
        elif output.prediction_type == "x0":
            clean = value
        elif output.prediction_type == "v":
            clean = signal * noisy - sigma * value
        elif output.prediction_type == "score":
            clean = (noisy + (1 - a) * value) / signal
        else:
            raise ValueError(
                "Discrete VP diffusion does not accept flow velocity without a path conversion"
            )
        return clean, (noisy - signal * clean) / sigma

    def posterior(self, clean, noisy, time):
        return (
            self.at("posterior_x0", time, noisy) * clean
            + self.at("posterior_xt", time, noisy) * noisy
        )

    def respaced(self, timesteps):
        indices = torch.as_tensor(timesteps, dtype=torch.long, device=self.alpha_bar.device)
        if (
            len(indices) < 2
            or (indices < 0).any()
            or (indices >= len(self)).any()
            or not (indices[1:] > indices[:-1]).all()
        ):
            raise ValueError("Respacing requires sorted distinct original steps")
        selected = self.alpha_bar[indices]
        beta = 1 - selected / F.pad(selected[:-1], (1, 0), value=1)

        return DiffusionSchedule(beta, timestep_map=self.timestep_map[indices])


def _discretized_nll(clean, mean, log_variance):
    inverse_std = (-0.5 * log_variance).exp()

    def cdf(x):
        return 0.5 * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))

    upper, lower = (
        cdf((clean - mean + 1 / 255) * inverse_std),
        cdf((clean - mean - 1 / 255) * inverse_std),
    )
    logp = torch.where(
        clean < -0.999,
        upper.clamp_min(1e-12).log(),
        torch.where(
            clean > 0.999,
            (1 - lower).clamp_min(1e-12).log(),
            (upper - lower).clamp_min(1e-12).log(),
        ),
    )
    return -mean_flat(logp) / math.log(2)


class DiffusionObjective(nn.Module):
    def __init__(
        self,
        schedule,
        *,
        min_snr_gamma=None,
        learned_variance=False,
        vb_weight=1.0,
        offset_noise=0.0,
    ):
        super().__init__()
        self.schedule = schedule
        if min_snr_gamma is not None and min_snr_gamma <= 0:
            raise ValueError("Min-SNR gamma must be positive")
        if vb_weight < 0 or offset_noise < 0:
            raise ValueError("Invalid objective weights")
        self.min_snr_gamma, self.learned_variance, self.vb_weight, self.offset_noise = (
            min_snr_gamma,
            learned_variance,
            vb_weight,
            offset_noise,
        )

    def config_dict(self):
        return {
            "type": "diffusion",
            "min_snr_gamma": self.min_snr_gamma,
            "learned_variance": self.learned_variance,
            "vb_weight": self.vb_weight,
            "offset_noise": self.offset_noise,
            "betas": self.schedule.betas.tolist(),
            "timestep_map": self.schedule.timestep_map.tolist(),
        }

    def forward(self, model, batch):
        clean = batch["sample"]
        time = batch.get("time")
        if time is None:
            time = torch.randint(len(self.schedule), (len(clean),), device=clean.device)
        noise = batch.get("noise")
        if noise is None:
            noise = torch.randn_like(clean)
            if self.offset_noise:
                noise = noise + self.offset_noise * torch.randn(
                    (*clean.shape[:2], *([1] * (clean.ndim - 2))),
                    device=clean.device,
                    dtype=clean.dtype,
                )
        noisy = self.schedule.noise(clean, time, noise)
        output = model(noisy, self.schedule.model_time(time), batch.get("condition"))
        variance_values = None
        if self.learned_variance:
            if output.prediction.shape[1] != 2 * clean.shape[1]:
                raise ValueError("Learned variance needs twice the output channels")
            mean_values, variance_values = output.prediction.chunk(2, dim=1)
            output = FieldOutput(mean_values, output.prediction_type)
        if output.prediction.shape != clean.shape:
            raise ValueError("Diffusion prediction shape mismatch")
        a = self.schedule.at("alpha_bar", time, clean)
        targets = {
            "epsilon": noise,
            "x0": clean,
            "v": a.sqrt() * noise - (1 - a).sqrt() * clean,
            "score": -noise / (1 - a).sqrt(),
        }
        if output.prediction_type not in targets:
            raise ValueError("VP target parameterization mismatch")
        errors = mean_flat(
            (output.prediction.float() - targets[output.prediction_type].float()).square()
        )
        if self.min_snr_gamma is not None:
            snr = (a / (1 - a)).flatten()
            clipped = snr.clamp(max=self.min_snr_gamma)
            if output.prediction_type == "epsilon":
                weight = clipped / snr
            elif output.prediction_type == "v":
                weight = clipped / (snr + 1)
            elif output.prediction_type == "x0":
                weight = clipped
            else:
                raise ValueError("Min-SNR score weighting must be explicitly derived")
            errors = errors * weight
        terms = [LossTerm(errors.sum(), errors.new_tensor(errors.numel()), "sample", "denoising")]
        if variance_values is not None:
            estimated, _ = self.schedule.clean_and_noise(
                FieldOutput(output.prediction.detach(), output.prediction_type), noisy, time
            )
            true_mean = self.schedule.posterior(clean, noisy, time)
            model_mean = self.schedule.posterior(estimated, noisy, time)
            true_log = self.schedule.at("posterior_log_variance", time, clean)
            fraction = (variance_values + 1) / 2
            model_log = (
                fraction * self.schedule.at("betas", time, clean).log() + (1 - fraction) * true_log
            )
            kl = 0.5 * (
                -1
                + model_log
                - true_log
                + (true_log - model_log).exp()
                + (true_mean - model_mean).square() * (-model_log).exp()
            )
            vb = torch.where(
                time == 0,
                _discretized_nll(clean, model_mean, model_log),
                mean_flat(kl) / math.log(2),
            )
            terms.append(
                LossTerm(vb.sum(), vb.new_tensor(len(vb)), "sample", "vlb", self.vb_weight)
            )
        return terms[0] if len(terms) == 1 else LossBundle(tuple(terms))


@torch.no_grad()
def sample_diffusion(
    model,
    noise,
    schedule,
    *,
    method="ddim",
    eta=0.0,
    condition=None,
    guidance_scale=1.0,
    clip_clean=False,
    learned_variance=False,
    generator=None,
):
    if method not in {"ddpm", "ddim"} or not 0 <= eta <= 1:
        raise ValueError("Invalid diffusion sampler")
    x = noise.clone()
    for index in reversed(range(len(schedule))):
        t = torch.full((len(x),), index, device=x.device, dtype=torch.long)
        output = guided_field(
            model, x, schedule.model_time(t), condition, guidance_scale=guidance_scale
        )
        variance = None
        if learned_variance:
            prediction, variance = output.prediction.chunk(2, 1)
            output = FieldOutput(prediction, output.prediction_type)
        clean, eps = schedule.clean_and_noise(output, x, t)
        if clip_clean:
            clean = clean.clamp(-1, 1)
            a = schedule.at("alpha_bar", t, x)
            eps = (x - a.sqrt() * clean) / (1 - a).sqrt()
        if method == "ddpm":
            logvar = schedule.at("posterior_log_variance", t, x)
            if variance is not None:
                logvar = (variance + 1) / 2 * schedule.at("betas", t, x).log() + (
                    1 - variance
                ) / 2 * logvar
            x = schedule.posterior(clean, x, t)
            if index:
                x = x + (logvar * 0.5).exp() * randn_like(x, generator)
        else:
            a, previous = schedule.at("alpha_bar", t, x), schedule.at("previous_alpha_bar", t, x)
            sigma = eta * ((1 - previous) / (1 - a) * (1 - a / previous)).clamp_min(0).sqrt()
            x = previous.sqrt() * clean + (1 - previous - sigma.square()).clamp_min(0).sqrt() * eps
            if index and eta:
                x = x + sigma * randn_like(x, generator)
    return x


def edm_denoise(model, sample, sigma, condition=None, *, sigma_data=0.5):
    if sigma_data <= 0 or (sigma <= 0).any():
        raise ValueError("EDM sigma must be positive")
    s = expand(sigma, sample)
    denominator = s.square() + sigma_data**2
    output = model(sample / denominator.sqrt(), sigma.log() / 4, condition)
    if output.prediction_type != "edm_residual":
        raise ValueError("EDM preconditioning requires edm_residual, not epsilon/x0/flow velocity")
    prediction = output.prediction
    if prediction.shape != sample.shape:
        raise ValueError("EDM residual must match sample")
    return sigma_data**2 / denominator * sample + s * sigma_data / denominator.sqrt() * prediction


class EDMObjective(nn.Module):
    def __init__(self, *, sigma_data=0.5, log_mean=-1.2, log_std=1.2):
        super().__init__()
        if sigma_data <= 0 or log_std <= 0:
            raise ValueError("Invalid EDM distribution")
        self.sigma_data, self.log_mean, self.log_std = sigma_data, log_mean, log_std

    def config_dict(self):
        return {
            "type": "edm",
            "sigma_data": self.sigma_data,
            "log_mean": self.log_mean,
            "log_std": self.log_std,
        }

    def forward(self, model, batch):
        clean = batch["sample"]
        sigma = batch.get("sigma")
        if sigma is None:
            sigma = (
                torch.randn(len(clean), device=clean.device) * self.log_std + self.log_mean
            ).exp()
        noise = batch.get("noise")
        if noise is None:
            noise = torch.randn_like(clean)
        result = edm_denoise(
            model,
            clean + expand(sigma, clean) * noise,
            sigma,
            batch.get("condition"),
            sigma_data=self.sigma_data,
        )
        weight = (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square()
        losses = weight * mean_flat((result.float() - clean.float()).square())
        return LossTerm(losses.sum(), losses.new_tensor(len(losses)), "sample", "edm")


def karras_sigmas(steps, *, sigma_min=0.002, sigma_max=80.0, rho=7.0, device="cpu"):
    if steps < 2 or not 0 < sigma_min < sigma_max or rho <= 0:
        raise ValueError("Invalid EDM schedule")
    ramp = torch.linspace(0, 1, steps, device=device, dtype=torch.float64)
    schedule = (
        sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ).pow(rho)
    return F.pad(schedule, (0, 1))


@torch.no_grad()
def sample_edm(
    model,
    noise,
    sigmas,
    *,
    condition=None,
    sigma_data=0.5,
    churn=0.0,
    churn_min=0.0,
    churn_max=float("inf"),
    noise_scale=1.0,
    generator=None,
):
    if (
        sigmas.ndim != 1
        or len(sigmas) < 3
        or sigmas[-1] != 0
        or not (sigmas[:-1] > sigmas[1:]).all()
        or churn < 0
    ):
        raise ValueError("EDM schedule must decrease to zero")
    x = noise * sigmas[0].to(noise)
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        gamma = (
            min(churn / (len(sigmas) - 1), math.sqrt(2) - 1)
            if churn_min <= sigma <= churn_max
            else 0.0
        )
        augmented = sigma * (1 + gamma)
        x = x + (augmented.square() - sigma.square()).clamp_min(0).sqrt().to(
            x
        ) * noise_scale * randn_like(x, generator)
        t = augmented.to(x).expand(len(x))
        next_t = next_sigma.to(x).expand(len(x))
        slope = (x - edm_denoise(model, x, t, condition, sigma_data=sigma_data)) / expand(t, x)
        proposal = x + (next_sigma - augmented).to(x) * slope
        if next_sigma > 0:
            slope2 = (
                proposal - edm_denoise(model, proposal, next_t, condition, sigma_data=sigma_data)
            ) / expand(next_t, x)
            x = x + (next_sigma - augmented).to(x) * (slope + slope2) / 2
        else:
            x = proposal
    return x


@dataclass(frozen=True)
class FlowPath:
    name: str = "linear"
    direction: str = "noise_to_data"

    def __post_init__(self):
        if self.name not in {"linear", "cosine"} or self.direction not in {
            "noise_to_data",
            "data_to_noise",
        }:
            raise ValueError("Unsupported flow path")

    def sample(self, data, noise, time):
        if (
            data.shape != noise.shape
            or time.shape != (len(data),)
            or ((time < 0) | (time > 1)).any()
        ):
            raise ValueError("Invalid flow path inputs")
        start, end = (noise, data) if self.direction == "noise_to_data" else (data, noise)
        t = expand(time, data)
        if self.name == "linear":
            return (1 - t) * start + t * end, end - start
        phase = t * math.pi / 2
        return phase.cos() * start + phase.sin() * end, math.pi / 2 * (
            -phase.sin() * start + phase.cos() * end
        )


class FlowObjective(nn.Module):
    def __init__(
        self, path=FlowPath(), *, time_distribution="uniform", logit_mean=0.0, logit_std=1.0
    ):
        super().__init__()
        if time_distribution not in {"uniform", "logit_normal"} or logit_std <= 0:
            raise ValueError("Invalid flow time distribution")
        self.path, self.time_distribution, self.logit_mean, self.logit_std = (
            path,
            time_distribution,
            logit_mean,
            logit_std,
        )

    def config_dict(self):
        return {
            "type": "flow_matching",
            "path": self.path.name,
            "direction": self.path.direction,
            "time_distribution": self.time_distribution,
            "logit_mean": self.logit_mean,
            "logit_std": self.logit_std,
        }

    def forward(self, model, batch):
        data = batch["sample"]
        time = batch.get("time")
        noise = batch.get("noise")
        if time is None:
            time = (
                torch.rand(len(data), device=data.device)
                if self.time_distribution == "uniform"
                else (
                    torch.randn(len(data), device=data.device) * self.logit_std + self.logit_mean
                ).sigmoid()
            )
        if noise is None:
            noise = torch.randn_like(data)
        noisy, target = self.path.sample(data, noise, time)
        output = model(noisy, time, batch.get("condition"))
        if output.prediction_type != "velocity" or output.prediction.shape != target.shape:
            raise ValueError("Flow matching requires matching path velocity")
        losses = mean_flat((output.prediction.float() - target.float()).square())
        return LossTerm(losses.sum(), losses.new_tensor(len(losses)), "sample", "flow_matching")


@torch.no_grad()
def sample_flow(
    model,
    noise,
    *,
    steps=20,
    solver="heun",
    direction="noise_to_data",
    shift=1.0,
    condition=None,
    guidance_scale=1.0,
):
    if (
        steps < 1
        or solver not in {"euler", "heun", "rk4"}
        or direction not in {"noise_to_data", "data_to_noise"}
        or shift <= 0
    ):
        raise ValueError("Invalid flow sampler")
    times = torch.linspace(0, 1, steps + 1, device=noise.device, dtype=noise.dtype)
    times = shift * times / (1 + (shift - 1) * times)
    if direction == "data_to_noise":
        times = times.flip(0)

    def field(x, time):
        output = guided_field(
            model, x, time.expand(len(x)), condition, guidance_scale=guidance_scale
        )
        if output.prediction_type != "velocity":
            raise ValueError("Flow ODE expects velocity")
        return output.prediction

    x = noise.clone()
    for t, next_t in zip(times[:-1], times[1:]):
        dt = next_t - t
        k1 = field(x, t)
        if solver == "euler":
            x = x + dt * k1
        elif solver == "heun":
            x = x + dt * (k1 + field(x + dt * k1, next_t)) / 2
        else:
            k2 = field(x + dt * k1 / 2, t + dt / 2)
            k3 = field(x + dt * k2 / 2, t + dt / 2)
            k4 = field(x + dt * k3, next_t)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


class AutoencoderObjective(nn.Module):
    def __init__(self, *, kl_weight=1e-6, perceptual=None, perceptual_weight=0.0):
        super().__init__()
        if kl_weight < 0 or perceptual_weight < 0 or perceptual_weight and perceptual is None:
            raise ValueError("Invalid VAE losses")
        self.kl_weight, self.perceptual_weight = kl_weight, perceptual_weight
        self.perceptual = (
            perceptual.eval().requires_grad_(False) if perceptual is not None else None
        )

    def config_dict(self):
        return {
            "type": "kl_vae",
            "kl_weight": self.kl_weight,
            "perceptual_weight": self.perceptual_weight,
        }

    def forward(self, model, batch):
        clean = batch["sample"]
        reconstruction, posterior = model(clean)
        reconstruction_loss = mean_flat((reconstruction - clean).abs())
        if self.perceptual is not None:
            self.perceptual.eval()

            with torch.no_grad():
                target = self.perceptual(clean)
            reconstruction_loss = reconstruction_loss + self.perceptual_weight * mean_flat(
                (self.perceptual(reconstruction) - target).square()
            )
        count = reconstruction_loss.new_tensor(len(clean))
        return LossBundle(
            (
                LossTerm(reconstruction_loss.sum(), count, "sample", "reconstruction"),
                LossTerm(posterior.kl().sum(), count, "sample", "kl", self.kl_weight),
            )
        )
