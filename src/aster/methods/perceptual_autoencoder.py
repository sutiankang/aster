"""Perceptual VAE objectives with frozen feature roles and preserved input gradients."""

import math
import torch
from torch import nn

from ..core import LossTerm, LossBundle
from ..models.generative import AutoencoderConfig
from ..models.perceptual import LPIPS


class PerceptualAutoencoderObjective(nn.Module):
    def __init__(
        self,
        perceptual,
        *,
        kl_weight=1e-6,
        perceptual_weight=1.0,
        logvar=0.0,
        sample_posterior=True,
        pixel_reduction="sum",
    ):
        super().__init__()
        if not isinstance(perceptual, LPIPS) or perceptual.config.spatial:
            raise ValueError(
                "Perceptual VAE requires native per-image LPIPS, not a generic feature encoder"
            )
        if not bool(perceptual.weights_loaded) and not perceptual.config.allow_untrained:
            raise ValueError(
                "Load frozen perceptual weights before constructing the training objective"
            )
        if (
            any(not math.isfinite(x) for x in (kl_weight, perceptual_weight, logvar))
            or min(kl_weight, perceptual_weight) < 0
            or abs(logvar) > 30
        ):
            raise ValueError("Invalid perceptual autoencoder objective")
        if type(sample_posterior) is not bool or pixel_reduction not in {"sum", "mean"}:
            raise ValueError(
                "Declare posterior sampling and sum/mean pixel normalization explicitly"
            )
        self.perceptual = perceptual.eval().requires_grad_(False)
        self.kl_weight, self.perceptual_weight, self.logvar = kl_weight, perceptual_weight, logvar
        self.sample_posterior, self.pixel_reduction = sample_posterior, pixel_reduction
        self.perceptual_identity = perceptual.weight_identity()

    def config_dict(self):
        return dict(
            type="perceptual_vae",
            kl_weight=self.kl_weight,
            perceptual_weight=self.perceptual_weight,
            logvar=self.logvar,
            sample_posterior=self.sample_posterior,
            pixel_reduction=self.pixel_reduction,
            perceptual_config=self.perceptual.config.to_dict(),
            perceptual_weights=self.perceptual_identity,
            input_range="minus_one_to_one",
            perceptual_precision="fp32_frozen",
            adversarial=False,
        )

    def _validate(self, model, batch):
        if (
            not isinstance(model.config, AutoencoderConfig)
            or model.config.in_channels != 3
            or set(batch) - {"sample", "posterior_noise"}
        ):
            raise ValueError(
                "Perceptual VAE expects native RGB autoencoder and explicit sample/noise tensors"
            )
        images = batch["sample"]
        divisor = 2 ** (len(model.config.channel_mult) - 1)
        minimum = 16 if self.perceptual.config.backbone == "vgg" else 31
        if (
            images.ndim != 4
            or images.shape[0] < 1
            or images.shape[1] != 3
            or min(images.shape[-2:]) < minimum
            or any(x % divisor for x in images.shape[-2:])
        ):
            raise ValueError(
                "Perceptual VAE image dimensions do not satisfy encoder and perceptual feature grids"
            )
        if (
            not images.is_floating_point()
            or not torch.isfinite(images).all()
            or images.abs().max() > 1
        ):
            raise ValueError("Training images must be finite RGB explicitly normalized to [-1,1]")
        if images.device != self.perceptual.shift.device:
            raise ValueError("Frozen perceptual role and images must share device")
        noise = batch.get("posterior_noise")
        if noise is not None:
            shape = (
                len(images),
                model.config.latent_channels,
                images.shape[-2] // divisor,
                images.shape[-1] // divisor,
            )
            if (
                not self.sample_posterior
                or noise.shape != shape
                or not noise.is_floating_point()
                or noise.device != images.device
                or not torch.isfinite(noise).all()
            ):
                raise ValueError(
                    "Explicit posterior noise must match the raw, unscaled latent grid"
                )

    def preflight_microbatches(self, model, batches):
        if self.perceptual.weight_identity() != self.perceptual_identity:
            raise ValueError("Frozen perceptual weights changed outside the bound objective")
        if next(self.perceptual.parameters()).dtype != torch.float32 or any(
            p.requires_grad for p in self.perceptual.parameters()
        ):
            raise ValueError("Training requires the bound perceptual role frozen in FP32")
        for batch in batches:
            self._validate(model, batch)
        return batches

    def reconstruct(self, model, batch):
        """Share one posterior sample between reconstruction and auxiliary objectives."""
        self._validate(model, batch)
        clean = batch["sample"]
        posterior = model.encode(clean)
        if not self.sample_posterior:
            z = posterior.mode()
        elif "posterior_noise" in batch:
            z = posterior.mean + (0.5 * posterior.logvar).exp() * batch["posterior_noise"].to(
                posterior.mean
            )
        else:
            z = posterior.sample()
        reconstruction = model.decode(z)
        return reconstruction, posterior

    def loss_from_reconstruction(self, clean, reconstruction, posterior):
        """Preserve the reconstruction graph through a frozen perceptual network;
        freezing weights does not mean detaching its image input."""
        self.perceptual.eval()

        distance = self.perceptual(clean.detach().float(), reconstruction.float())
        error = (clean.float() - reconstruction.float()).abs() + self.perceptual_weight * distance
        nll = error * math.exp(-self.logvar) + self.logvar
        per_sample = (
            nll.flatten(1).sum(1) if self.pixel_reduction == "sum" else nll.flatten(1).mean(1)
        )
        count = torch.tensor(len(clean), dtype=torch.int64, device=clean.device)
        return LossBundle(
            (
                LossTerm(per_sample.sum(), count, "sample", "perceptual_nll"),
                LossTerm(posterior.kl().sum(), count, "sample", "kl", self.kl_weight),
            )
        )

    def forward(self, model, batch):
        reconstruction, posterior = self.reconstruct(model, batch)
        return self.loss_from_reconstruction(batch["sample"], reconstruction, posterior)


class PerceptualAutoencoderMethod:
    """Keep frozen feature-network weights in the same checkpoint lifecycle as the
    generator, optimizer, and RNG."""

    def __init__(
        self, engine, perceptual, *, state_name="perceptual_autoencoder", **objective_settings
    ):
        error = None
        try:
            if any(
                getattr(engine.parallel.config, key, 1) != 1
                for key in (
                    "tensor_parallel",
                    "pipeline_parallel",
                    "context_parallel",
                    "gtp_remat",
                    "expert_parallel",
                    "expert_tensor_parallel",
                )
            ):
                raise ValueError("Perceptual autoencoder currently admits pure DP/ZeRO providers")
            if (
                not isinstance(engine.model.config, AutoencoderConfig)
                or engine.model.config.in_channels != 3
            ):
                raise ValueError("Perceptual training requires an RGB AutoencoderKL")
            if (
                not isinstance(state_name, str)
                or not state_name
                or state_name in engine.states
                or state_name + "_metric" in engine.roles
            ):
                raise ValueError("Perceptual method state/role name must be unused")
            objective = PerceptualAutoencoderObjective(perceptual, **objective_settings)
            declaration = objective.config_dict()
        except Exception as exc:
            error, declaration = f"{type(exc).__name__}: {exc}", None
        declarations = engine.parallel.world.gather_objects((error, declaration))
        if any(item[0] for item in declarations) or any(
            item[1] != declaration for item in declarations
        ):
            raise ValueError(
                "Perceptual setup differs across ranks or is invalid: " + str(declarations)
            )
        self.engine, self.state_name = engine, state_name
        self.perceptual = engine.add_role(state_name + "_metric", perceptual, trainable=False)
        self.objective, self.updates = objective, 0
        engine.register_state(state_name, self)

    def update(self, microbatches):
        result = self.engine.phase(
            self.state_name, objective=self.objective, microbatches=microbatches
        )
        if result.updated:
            self.updates += 1
        return result

    def state_dict(self):
        if self.perceptual.weight_identity() != self.objective.perceptual_identity:
            raise ValueError(
                "Cannot certify a checkpoint after external perceptual weight mutation"
            )
        return dict(configuration=self.objective.config_dict(), updates=self.updates)

    def load_state_dict(self, state):
        if (
            state["configuration"] != self.objective.config_dict()
            or type(state["updates"]) is not int
            or state["updates"] < 0
        ):
            raise ValueError("Perceptual method checkpoint differs")
        if self.perceptual.weight_identity() != self.objective.perceptual_identity:
            raise ValueError("Restored perceptual role differs from its content identity")
        self.updates = state["updates"]
