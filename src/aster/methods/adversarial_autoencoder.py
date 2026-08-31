"""KL-VAE, LPIPS, and PatchGAN training through shared generator/discriminator roles."""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..core import LossBundle, LossTerm
from ..models.adversarial import ActNorm2d, PatchDiscriminator
from ..models.generative import AutoencoderConfig
from .perceptual_autoencoder import PerceptualAutoencoderObjective


class AdversarialGeneratorObjective(nn.Module):
    def __init__(self, reconstruction, discriminator, *, disc_factor=1.0, active=True):
        super().__init__()
        self.reconstruction, self.discriminator = reconstruction, discriminator
        self.disc_factor, self.active = disc_factor, active

    def config_dict(self):
        return dict(
            type="adversarial_vae_generator",
            reconstruction=self.reconstruction.config_dict(),
            discriminator=self.discriminator.config.to_dict(),
            disc_factor=self.disc_factor,
            active=self.active,
            adaptive_weight="global_gradient_ratio_decoder_last_weight",
        )

    def preflight_microbatches(self, model, batches):
        self.reconstruction.preflight_microbatches(model, batches)
        for batch in batches:
            self.discriminator._validate(batch["sample"])
        return batches

    def forward(self, model, batch):
        reconstruction, posterior = self.reconstruction.reconstruct(model, batch)
        terms = self.reconstruction.loss_from_reconstruction(
            batch["sample"], reconstruction, posterior
        ).terms
        logits = self.discriminator(reconstruction).float()

        adversarial = LossTerm(
            -logits.sum(),
            logits.new_tensor(logits.numel()),
            "patch",
            "g_loss",
            self.disc_factor if self.active else 0.0,
        )
        return LossBundle((*terms, adversarial))


class AdversarialDiscriminatorObjective(nn.Module):
    def __init__(self, generator, reconstruction, *, loss="hinge", disc_factor=1.0, active=True):
        super().__init__()
        if loss not in {"hinge", "vanilla"}:
            raise ValueError("Discriminator loss must be hinge or vanilla")
        self.generator, self.reconstruction = generator, reconstruction
        self.loss, self.disc_factor, self.active = loss, disc_factor, active

    def config_dict(self):
        return dict(
            type="adversarial_vae_discriminator",
            loss=self.loss,
            disc_factor=self.disc_factor,
            active=self.active,
            fake_source="reconstructed_after_generator_update",
            reconstruction=self.reconstruction.config_dict(),
        )

    def preflight_microbatches(self, model, batches):
        self.reconstruction.preflight_microbatches(self.generator, batches)
        for batch in batches:
            model._validate(batch["sample"])
        return batches

    def forward(self, model, batch):
        with torch.no_grad():
            reconstruction, _ = self.reconstruction.reconstruct(self.generator, batch)
        real, fake = model(batch["sample"].detach()).float(), model(reconstruction.detach()).float()
        if self.loss == "hinge":
            value = 0.5 * (F.relu(1 - real).sum() + F.relu(1 + fake).sum())
        else:
            value = 0.5 * (F.softplus(-real).sum() + F.softplus(fake).sum())

        factor = self.disc_factor if self.active else 0.0
        return LossTerm(value * factor, real.new_tensor(real.numel()), "patch", "discriminator")


@dataclass(frozen=True)
class AdversarialAutoencoderResult:
    generator: object
    discriminator: object | None
    updates: int

    @property
    def updated(self):
        return (
            self.generator.updated and self.discriminator is not None and self.discriminator.updated
        )


class AdversarialAutoencoderMethod:
    """Coordinate generator and discriminator phases. A partially completed update
    cannot be saved as a completed method checkpoint."""

    def __init__(
        self,
        engine,
        perceptual,
        discriminator,
        *,
        discriminator_optimizer_factory=None,
        disc_start=0,
        disc_factor=1.0,
        disc_weight=1.0,
        disc_loss="hinge",
        state_name="adversarial_autoencoder",
        **reconstruction_settings,
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
                raise ValueError("Adversarial autoencoder currently admits pure DP/ZeRO providers")
            if (
                not isinstance(engine.model.config, AutoencoderConfig)
                or engine.model.config.in_channels != 3
            ):
                raise ValueError("Adversarial autoencoder requires native RGB AutoencoderKL")
            if (
                not isinstance(discriminator, PatchDiscriminator)
                or discriminator.config.in_channels != 3
                or discriminator.config.normalization != "actnorm"
            ):
                raise ValueError(
                    "Adversarial Method requires native RGB PatchGAN with explicit initialized ActNorm"
                )
            if any(
                not bool(module.initialized)
                for module in discriminator.modules()
                if isinstance(module, ActNorm2d)
            ):
                raise ValueError(
                    "Initialize discriminator using global calibration before role/optimizer creation"
                )
            if (
                type(disc_start) is not int
                or disc_start < 0
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in (disc_factor, disc_weight)
                )
            ):
                raise ValueError("Invalid discriminator warmup/factor/weight")
            if disc_loss not in {"hinge", "vanilla"}:
                raise ValueError("Discriminator loss must be hinge or vanilla")
            if (
                not isinstance(state_name, str)
                or not state_name
                or state_name in engine.states
                or any(
                    state_name + suffix in engine.roles for suffix in ("_metric", "_discriminator")
                )
            ):
                raise ValueError("Adversarial method state and role names must be unused")
            if discriminator_optimizer_factory is not None and not callable(
                discriminator_optimizer_factory
            ):
                raise ValueError("Discriminator optimizer factory must be callable")
            reconstruction = PerceptualAutoencoderObjective(perceptual, **reconstruction_settings)
            last_weight = f"decoder.{len(engine.model.decoder) - 1}.weight"
            declaration = dict(
                type="adversarial_autoencoder",
                disc_start=disc_start,
                disc_factor=disc_factor,
                disc_weight=disc_weight,
                disc_loss=disc_loss,
                state_name=state_name,
                last_weight=last_weight,
                reconstruction=reconstruction.config_dict(),
                discriminator=discriminator.config.to_dict(),
                warmup_clock="completed_method_iterations",
                normalization="global_window",
                generator_then_discriminator=True,
            )
        except Exception as exc:
            error, declaration = f"{type(exc).__name__}: {exc}", None
        declarations = engine.parallel.world.gather_objects((error, declaration))
        if any(item[0] for item in declarations) or any(
            item[1] != declaration for item in declarations
        ):
            raise ValueError(
                "Adversarial setup differs across ranks or is invalid: " + str(declarations)
            )
        self.engine, self.state_name, self.configuration = engine, state_name, declaration
        self.metric_role, self.discriminator_role = (
            state_name + "_metric",
            state_name + "_discriminator",
        )
        self.perceptual = engine.add_role(self.metric_role, perceptual, trainable=False)
        if discriminator_optimizer_factory is None:
            discriminator_optimizer_factory = lambda parameters: torch.optim.Adam(
                parameters, lr=engine.lr, betas=(0.5, 0.9)
            )
        self.discriminator = engine.add_role(
            self.discriminator_role,
            discriminator,
            optimizer_factory=discriminator_optimizer_factory,
        )
        self.generator_objective = AdversarialGeneratorObjective(
            reconstruction, self.discriminator, disc_factor=disc_factor
        )
        self.discriminator_objective = AdversarialDiscriminatorObjective(
            engine.model, reconstruction, loss=disc_loss, disc_factor=disc_factor
        )
        self.policy_name = state_name + "_adaptive"
        engine.register_gradient_ratio(
            self.policy_name,
            role="model",
            reference_term="perceptual_nll",
            target_term="g_loss",
            parameter=last_weight,
            eps=1e-4,
            min_ratio=0.0,
            max_ratio=1e4,
            multiplier=disc_weight,
        )
        self.updates, self._incomplete = 0, False
        engine.register_state(state_name, self)

    def update(self, microbatches):
        batches = list(microbatches)
        error = None
        try:
            if self._incomplete:
                raise RuntimeError("Incomplete adversarial transaction; restore a full checkpoint")
            if len(batches) != self.engine.accumulation_steps:
                raise ValueError("Adversarial microbatch count must match accumulation_steps")
            self.generator_objective.preflight_microbatches(self.engine.model, batches)
            self.discriminator_objective.preflight_microbatches(self.discriminator, batches)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        declarations = self.engine.parallel.world.gather_objects((error, self.updates))
        if any(item[0] for item in declarations) or any(
            item[1] != self.updates for item in declarations
        ):
            raise ValueError("Adversarial whole-window preflight failed: " + str(declarations))
        active = self.updates >= self.configuration["disc_start"]
        self.generator_objective.active = self.discriminator_objective.active = active
        generated = self.engine.phase(
            self.state_name + "_generator",
            objective=self.generator_objective,
            microbatches=batches,
            freeze_roles=(self.discriminator_role,),
        )
        if not generated.updated:
            return AdversarialAutoencoderResult(generated, None, self.updates)
        self._incomplete = True
        judged = self.engine.phase(
            self.state_name + "_discriminator",
            role=self.discriminator_role,
            objective=self.discriminator_objective,
            microbatches=batches,
            freeze_roles=("model",),
        )
        if not judged.updated:
            raise RuntimeError(
                "Generator updated but discriminator did not; restore a full checkpoint before continuing"
            )
        self.updates += 1
        self._incomplete = False
        return AdversarialAutoencoderResult(generated, judged, self.updates)

    def state_dict(self):
        if self._incomplete:
            raise RuntimeError("Cannot checkpoint an incomplete adversarial transaction")
        if (
            self.perceptual.weight_identity()
            != self.generator_objective.reconstruction.perceptual_identity
        ):
            raise ValueError("Frozen perceptual identity changed")
        return dict(configuration=self.configuration, updates=self.updates)

    def load_state_dict(self, state):
        if (
            state["configuration"] != self.configuration
            or type(state["updates"]) is not int
            or state["updates"] < 0
        ):
            raise ValueError("Adversarial method checkpoint differs")
        if (
            self.perceptual.weight_identity()
            != self.generator_objective.reconstruction.perceptual_identity
        ):
            raise ValueError("Restored perceptual identity differs")
        self.updates, self._incomplete = state["updates"], False
