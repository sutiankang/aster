"""World Models VAE and mixture-density recurrent objectives."""

import math
import torch
from torch import nn
import torch.nn.functional as F

from ..core import LossBundle, LossTerm
from ..models.vmc import VMCVAEConfig, MDNRNNConfig


class VMCVAEObjective(nn.Module):
    def __init__(self, *, kl_tolerance=0.5):
        super().__init__()
        if not math.isfinite(kl_tolerance) or kl_tolerance < 0:
            raise ValueError("VMC KL tolerance must be finite and nonnegative")
        self.kl_tolerance = kl_tolerance

    def config_dict(self):
        return dict(
            type="vmc_vae",
            kl_tolerance=self.kl_tolerance,
            image_range="zero_one",
            reconstruction="pixel_sum_mse",
        )

    def _validate(self, model, batch):
        c, images = model.config, batch["images"]
        if not isinstance(c, VMCVAEConfig) or set(batch) - {"images", "noise"}:
            raise ValueError("Invalid VMC VAE model/batch fields")
        if (
            not isinstance(images, torch.Tensor)
            or images.ndim != 4
            or images.shape[1:] != (c.image_channels, 64, 64)
            or len(images) < 1
            or not images.is_floating_point()
            or not torch.isfinite(images).all()
            or images.min() < 0
            or images.max() > 1
        ):
            raise ValueError("VMC VAE requires finite float [0,1] BCHW images")
        if "noise" in batch:
            noise = batch["noise"]
            if (
                noise.shape != (len(images), c.latent_size)
                or noise.device != images.device
                or not torch.isfinite(noise).all()
            ):
                raise ValueError("Invalid VMC VAE latent noise")

    def preflight_microbatches(self, model, batches):
        for batch in batches:
            self._validate(model, batch)
        return batches

    def forward(self, model, batch):
        self._validate(model, batch)
        output = model(batch["images"], noise=batch.get("noise"))
        reconstruction = (
            (output.reconstruction.float() - batch["images"].float()).square().flatten(1).sum(1)
        )
        kl = -0.5 * (1 + output.logvar - output.mean.square() - output.logvar.exp()).sum(-1)

        kl = kl.clamp_min(self.kl_tolerance * model.config.latent_size)
        count = torch.tensor(len(kl), dtype=torch.int64, device=kl.device)
        return LossBundle(
            (
                LossTerm(reconstruction.sum(), count, "image", "reconstruction"),
                LossTerm(kl.sum(), count, "image", "kl"),
            )
        )


@torch.no_grad()
def encode_vmc_episodes(vae, images, *, chunk_size=128):
    """Store posterior mu/logvar and resample latent z during dynamics training."""
    if (
        not isinstance(vae.config, VMCVAEConfig)
        or type(chunk_size) is not int
        or chunk_size < 1
        or images.ndim != 5
        or min(images.shape[:2]) < 1
    ):
        raise ValueError("Invalid VMC episode encoding configuration")
    if (
        not images.is_floating_point()
        or not torch.isfinite(images).all()
        or images.min() < 0
        or images.max() > 1
    ):
        raise ValueError("VMC episode images must be finite [0,1] floats")
    mode = vae.training
    try:
        vae.eval()
        means, logvars = [], []
        for chunk in images.flatten(0, 1).split(chunk_size):
            mean, logvar = vae.encode(chunk)
            means.append(mean)
            logvars.append(logvar)
        return dict(
            mean=torch.cat(means).reshape(*images.shape[:2], vae.config.latent_size),
            logvar=torch.cat(logvars).reshape(*images.shape[:2], vae.config.latent_size),
        )
    finally:
        vae.train(mode)


class MDNRNNObjective(nn.Module):
    def __init__(self, *, sequence_length=500, restart_factor=10.0):
        super().__init__()
        if (
            type(sequence_length) is not int
            or sequence_length < 2
            or not math.isfinite(restart_factor)
            or restart_factor <= 0
        ):
            raise ValueError("Invalid MDN sequence length/restart factor")
        self.sequence_length, self.restart_factor = sequence_length, restart_factor

    def config_dict(self):
        return dict(
            type="vmc_mdn",
            sequence_length=self.sequence_length,
            restart_factor=self.restart_factor,
            action_alignment="current",
            latent_loss="mean_independent_coordinate_mixture_nll",
        )

    def _validate(self, model, batch):
        c = model.config
        if not isinstance(c, MDNRNNConfig) or set(batch) - {
            "latents",
            "mean",
            "logvar",
            "noise",
            "actions",
            "restart",
            "valid",
        }:
            raise ValueError("Invalid MDN model/batch fields")
        direct = "latents" in batch
        if direct == ("mean" in batch or "logvar" in batch) or ("noise" in batch and direct):
            raise ValueError("Provide MDN latents OR mean/logvar with optional noise")
        latent = batch["latents"] if direct else batch["mean"]
        if (
            not isinstance(latent, torch.Tensor)
            or latent.ndim != 3
            or latent.shape[1:] != (self.sequence_length, c.latent_size)
            or len(latent) < 1
        ):
            raise ValueError("MDN sequence length/latent dimension differs")
        b, t = latent.shape[:2]
        floating = (
            {"latents": latent} if direct else {key: batch[key] for key in ("mean", "logvar")}
        )
        if "noise" in batch:
            floating["noise"] = batch["noise"]
        for name, value in floating.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != latent.shape
                or value.device != latent.device
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"MDN {name} must be finite and match latent shape/device")
        actions, restart = batch["actions"], batch["restart"]
        if (
            actions.shape != (b, t, c.action_dim)
            or actions.device != latent.device
            or not actions.is_floating_point()
            or not torch.isfinite(actions).all()
        ):
            raise ValueError("MDN actions must be finite aligned [B,T,A]")
        for name in ("restart", "valid"):
            value = batch.get(name)
            if value is not None and (
                value.shape != (b, t) or value.dtype != torch.bool or value.device != latent.device
            ):
                raise ValueError(f"MDN {name} must be bool [B,T]")
        valid = batch.get("valid", torch.ones_like(restart))
        if not valid[:, :2].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError(
                "MDN valid mask must be a contiguous prefix containing at least two frames"
            )

    def preflight_microbatches(self, model, batches):
        for batch in batches:
            self._validate(model, batch)
        return batches

    def forward(self, model, batch):
        return self.loss_and_state(model, batch)[0]

    def loss_and_state(self, model, batch, *, state=None):
        """Accept and return recurrent state explicitly; the default forward treats each
        window independently rather than carrying global hidden state."""
        self._validate(model, batch)
        if "latents" in batch:
            latent = batch["latents"]
        else:
            noise = batch.get("noise")
            if noise is None:
                noise = torch.randn_like(batch["mean"])

            latent = batch["mean"] + (0.5 * batch["logvar"]).exp() * noise
        latent = latent.detach()
        output = model(
            latent[:, :-1], batch["actions"][:, :-1], batch["restart"][:, :-1], state=state
        )
        target = latent[:, 1:, :, None]
        lognormal = (
            -0.5 * ((target - output.mean) * (-output.logstd).exp()).square()
            - output.logstd
            - 0.5 * math.log(2 * math.pi)
        )
        nll = -torch.logsumexp(output.logmix + lognormal, -1)
        valid = batch.get("valid", torch.ones_like(batch["restart"]))[:, 1:]
        restart = batch["restart"][:, 1:].to(output.restart_logits)
        restart_loss = F.binary_cross_entropy_with_logits(
            output.restart_logits, restart, reduction="none"
        )
        restart_loss = restart_loss * (1 + restart * (self.restart_factor - 1))
        count = valid.sum(dtype=torch.int64)

        return (
            LossBundle(
                (
                    LossTerm(
                        nll[valid].sum(),
                        count * model.config.latent_size,
                        "latent_coordinate",
                        "mixture_nll",
                    ),
                    LossTerm(restart_loss[valid].sum(), count, "transition", "restart"),
                )
            ),
            output.state,
        )
