"""Native video codecs and conditional fields using the shared training and sampling stack."""

import math

import torch
from torch import nn

from ..core import FieldOutput, LossTerm
from ..models.video_vae import WanVideoVAE
from ..models.video_world import WanVideoDiT
from .generation import FlowPath


class WanVideoObjective(nn.Module):
    def __init__(
        self, *, time_distribution="logit_normal", logit_mean=0.0, logit_std=1.0, shift=1.0
    ):
        super().__init__()
        if (
            time_distribution not in {"uniform", "logit_normal"}
            or not all(math.isfinite(x) for x in (logit_mean, logit_std, shift))
            or min(logit_std, shift) <= 0
        ):
            raise ValueError("Invalid video flow time distribution")
        self.time_distribution, self.logit_mean, self.logit_std, self.shift = (
            time_distribution,
            logit_mean,
            logit_std,
            shift,
        )

    def config_dict(self):
        return {
            "type": "wan_video_flow",
            "direction": "data_to_noise",
            "time_distribution": self.time_distribution,
            "logit_mean": self.logit_mean,
            "logit_std": self.logit_std,
            "shift": self.shift,
        }

    def forward(self, model, batch):
        data, noise, time = batch["sample"], batch.get("noise"), batch.get("time")
        if data.ndim != 5 or min(data.shape) < 1:
            raise ValueError("Video objective expects nonempty B,C,T,H,W scaled latents")
        if noise is None:
            noise = torch.randn_like(data)
        if time is None:
            time = (
                torch.rand(len(data), device=data.device)
                if self.time_distribution == "uniform"
                else (
                    torch.randn(len(data), device=data.device) * self.logit_std + self.logit_mean
                ).sigmoid()
            )
            time = self.shift * time / (1 + (self.shift - 1) * time)
        noisy, target = FlowPath(direction="data_to_noise").sample(data, noise, time)
        result = model(noisy, time, batch["condition"])
        if (
            not isinstance(result, FieldOutput)
            or result.prediction_type != "velocity"
            or result.prediction.shape != data.shape
        ):
            raise ValueError("Wan requires d x/d sigma with the same video latent layout")
        valid = batch.get("valid_mask", torch.ones_like(data, dtype=torch.bool))
        if valid.shape != data.shape or valid.dtype != torch.bool:
            raise ValueError("Video valid_mask must explicitly match every latent element")
        error = (result.prediction.float() - target.float()).square()
        return LossTerm(
            error.masked_select(valid).sum(),
            valid.sum(dtype=torch.int64),
            "latent_element",
            "wan_flow",
        )


def image_video_condition(vae, first_frame, frames, *, last_frame=None):

    if not isinstance(vae, WanVideoVAE) or first_frame.ndim != 4 or first_frame.shape[1] != 3:
        raise ValueError("Image conditioning needs native WanVideoVAE and B,3,H,W")
    stride = vae.config.temporal_stride
    if type(frames) is not int or frames < 1 or (frames - 1) % stride:
        raise ValueError("Condition video length must be 1+k*temporal_stride")
    b, _, h, w = first_frame.shape
    if any(n % vae.config.spatial_stride for n in (h, w)):
        raise ValueError("Image dimensions must match the VAE stride")
    video = first_frame.new_zeros(b, 3, frames, h, w)
    video[:, :, 0] = first_frame
    observed = first_frame.new_zeros(
        b, frames, h // vae.config.spatial_stride, w // vae.config.spatial_stride
    )
    observed[:, 0] = 1
    if last_frame is not None:
        if last_frame.shape != first_frame.shape or frames == 1:
            raise ValueError("Last frame needs matching shape and a distinct endpoint")
        video[:, :, -1], observed[:, -1] = last_frame, 1
    latent = vae.latent(video, sample=False)

    observed = torch.cat((observed[:, :1].expand(-1, stride, -1, -1), observed[:, 1:]), 1)
    mask = observed.reshape(b, latent.shape[2], stride, *observed.shape[-2:]).transpose(1, 2)
    return torch.cat((mask, latent), 1)


@torch.no_grad()
def sample_video_latents(
    model,
    noise,
    condition,
    *,
    steps=30,
    solver="heun",
    shift=5.0,
    negative_condition=None,
    guidance_scale=1.0,
):

    if (
        type(steps) is not int
        or steps < 1
        or solver not in {"euler", "heun"}
        or not math.isfinite(shift)
        or shift <= 0
    ):
        raise ValueError("Invalid video flow solver or shifted schedule")
    if (
        not math.isfinite(guidance_scale)
        or guidance_scale < 0
        or (guidance_scale != 1 and negative_condition is None)
    ):
        raise ValueError("Guided video generation needs explicit negative conditioning")
    if model.training:
        raise ValueError("Video inference requires eval mode")
    sigmas = torch.linspace(1, 0, steps + 1, device=noise.device, dtype=torch.float32)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    def velocity(value, sigma):
        time = sigma.expand(len(value))
        result = model(value, time, condition)
        if result.prediction_type != "velocity" or result.prediction.shape != value.shape:
            raise ValueError("Sampler field does not match video velocity layout")
        if guidance_scale != 1:
            negative = model(value, time, negative_condition)
            if negative.prediction_type != "velocity" or negative.prediction.shape != value.shape:
                raise ValueError("Negative velocity layout mismatch")
            return negative.prediction + guidance_scale * (result.prediction - negative.prediction)
        return result.prediction

    value = noise.clone()
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        dt = (next_sigma - sigma).to(value)
        first = velocity(value, sigma)
        proposal = value + dt * first
        value = (
            proposal
            if solver == "euler"
            else value + dt * (first + velocity(proposal, next_sigma)) / 2
        )
    return value


class VideoGenerationPipeline(nn.Module):
    def __init__(self, field, vae):
        super().__init__()
        if not isinstance(field, WanVideoDiT) or not isinstance(vae, WanVideoVAE):
            raise TypeError("This pipeline composes native Wan video components")
        if field.config.latent_channels != vae.config.latent_channels:
            raise ValueError("Video field/VAE latent channels differ")
        if (
            field.config.image_conditioned
            and field.config.condition_channels
            != vae.config.latent_channels + vae.config.temporal_stride
        ):
            raise ValueError("Image-conditioned field must consume mask and scaled VAE channels")
        self.field, self.vae = field, vae

    @torch.no_grad()
    def training_batch(self, video, text, *, image_features=None, last_frame=False):
        if self.vae.training:
            raise ValueError("Frozen latent preparation requires VAE eval mode")
        condition = {"text": text}
        if self.field.config.image_conditioned:
            if image_features is None or bool(last_frame) != self.field.config.first_last_frames:
                raise ValueError("Declare the matching image-conditioning variant and features")
            condition.update(
                image_features=image_features,
                video_condition=image_video_condition(
                    self.vae,
                    video[:, :, 0],
                    video.shape[2],
                    last_frame=video[:, :, -1] if last_frame else None,
                ),
            )
        elif image_features is not None or last_frame:
            raise ValueError("Text-only pipeline cannot ignore image features")
        return {"sample": self.vae.latent(video, sample=False), "condition": condition}

    @torch.no_grad()
    def generate(self, noise, condition, **sampling):
        if self.training or self.field.training or self.vae.training:
            raise ValueError("Generation pipeline must be in eval mode")
        latent = sample_video_latents(self.field, noise, condition, **sampling)
        return self.vae.decode(latent, scaled=True)
