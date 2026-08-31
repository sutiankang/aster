"""Joint multimodal latent training and sampling with shared flow paths and solvers."""

from dataclasses import dataclass, replace
import math
import torch
from torch import nn
import torch.nn.functional as F

from aster.core import FieldOutput, LossTerm, LossBundle
from .generation import FlowPath, sample_flow
from .supervised import token_targets


class Wan22AutoencoderObjective(nn.Module):
    def __init__(self, *, sequence_length=5, kl_weight=1e-6, sample_posterior=True):
        super().__init__()
        if type(sequence_length) is not int or sequence_length < 1 or (sequence_length - 1) % 4:
            raise ValueError("Wan2.2 training sequence_length must be 1+4k")
        if not math.isfinite(kl_weight) or kl_weight < 0 or type(sample_posterior) is not bool:
            raise ValueError("Invalid Wan2.2 objective settings")
        self.sequence_length, self.kl_weight, self.sample_posterior = (
            sequence_length,
            kl_weight,
            sample_posterior,
        )

    def config_dict(self):
        return dict(
            type="wan22_vae",
            sequence_length=self.sequence_length,
            kl_weight=self.kl_weight,
            sample_posterior=self.sample_posterior,
            reconstruction="unclipped_mean_l1",
        )

    def _validate(self, model, batch):
        from aster.models.wan22_vae import Wan22VAEConfig

        x = batch["sample"]
        if (
            not isinstance(model.config, Wan22VAEConfig)
            or x.ndim != 5
            or min(x.shape) < 1
            or x.shape[1] != 3
            or not x.is_floating_point()
            or not torch.isfinite(x).all()
        ):
            raise ValueError("Wan2.2 objective needs configured finite RGB BCTHW input")
        if x.shape[2] != self.sequence_length or any(
            n % model.config.spatial_stride for n in x.shape[-2:]
        ):
            raise ValueError(
                "Wan2.2 fixed sequence_length/spatial stride mismatch before parameter gathers"
            )

    def preflight_microbatches(self, model, batches):
        for batch in batches:
            self._validate(model, batch)
        return batches

    def forward(self, model, batch):
        self._validate(model, batch)
        clean = batch["sample"]
        reconstruction, posterior = model(
            clean, sample_posterior=self.sample_posterior, clip_output=False
        )
        errors = (reconstruction.float() - clean.float()).abs().flatten(1).mean(1)

        mean, logvar = posterior.mean.float(), posterior.logvar.float()
        kl = 0.5 * (mean.square() + torch.expm1(logvar) - logvar).flatten(1).sum(1)
        count = torch.tensor(len(clean), dtype=torch.int64, device=clean.device)
        return LossBundle(
            (
                LossTerm(errors.sum(), count, "sample", "reconstruction"),
                LossTerm(kl.sum(), count, "sample", "kl", self.kl_weight),
            )
        )


class Cosmos3AudioAutoencoderObjective(nn.Module):
    def __init__(self, *, kl_weight=1e-4, sample_posterior=True):
        super().__init__()
        if not math.isfinite(kl_weight) or kl_weight < 0 or type(sample_posterior) is not bool:
            raise ValueError("Invalid AVAE2 objective configuration")
        self.kl_weight, self.sample_posterior = kl_weight, sample_posterior

    def config_dict(self):
        return dict(
            type="cosmos3_avae2",
            kl_weight=self.kl_weight,
            sample_posterior=self.sample_posterior,
            reconstruction="unclipped_waveform_l1",
            kl_definition="source_twice_gaussian_kl_per_latent_frame",
        )

    def _validate(self, model, batch):
        from aster.models.cosmos3_audio import Cosmos3AudioConfig

        c, x = model.config, batch["sample"]
        if not isinstance(c, Cosmos3AudioConfig) or not c.encoder_enabled or c.normalize_volume:
            raise ValueError(
                "AVAE2 training requires a full codec and explicit normalize_volume=False preprocessing contract"
            )
        if (
            x.ndim != 3
            or min(x.shape) < 1
            or x.shape[1] != c.dec_out_channels
            or not x.is_floating_point()
            or not torch.isfinite(x).all()
        ):
            raise ValueError(
                "AVAE2 training requires finite waveform [B,C,N] before parameter gathers"
            )
        length = (x.shape[-1] + (-x.shape[-1]) % c.hop_size) // c.enc_hop_length
        for stride in c.enc_strides:
            pad = math.ceil(stride / 2)
            if (c.padding_mode == "reflect" and pad >= length) or (
                c.padding_mode == "circular" and pad > length
            ):
                raise ValueError(
                    "AVAE2 waveform is too short for its declared convolution padding mode"
                )
            length = (length + 2 * pad - 2 * stride) // stride + 1

    def preflight_microbatches(self, model, batches):
        for batch in batches:
            self._validate(model, batch)
        return batches

    def forward(self, model, batch):
        self._validate(model, batch)
        clean = batch["sample"]
        output, posterior = model(
            clean, sample_posterior=self.sample_posterior, force_pad=True, clip_output=False
        )
        errors = (output[..., : clean.shape[-1]].float() - clean.float()).abs()

        std = F.softplus(posterior.scale.float()) + 1e-4
        logvar = 2 * std.log()
        kl = (posterior.mean.float().square() + torch.expm1(logvar) - logvar).sum(1)
        return LossBundle(
            (
                LossTerm(
                    errors.sum(),
                    torch.tensor(errors.numel(), dtype=torch.int64, device=clean.device),
                    "audio_scalar",
                    "reconstruction",
                ),
                LossTerm(
                    kl.sum(),
                    torch.tensor(kl.numel(), dtype=torch.int64, device=clean.device),
                    "acoustic_frame",
                    "kl",
                    self.kl_weight,
                ),
            )
        )


@dataclass
class Cosmos3VideoOutput:
    video: torch.Tensor
    latents: dict[str, torch.Tensor]


class Cosmos3VideoPipeline:
    def __init__(self, model, vae, *, temporal_margin=15000.0, base_fps=24.0):
        from aster.models.cosmos3 import Cosmos3Config
        from aster.models.cosmos3_vlm import Cosmos3VLMConfig
        from aster.models.wan22_vae import Wan22VAEConfig

        if not isinstance(model.config, (Cosmos3Config, Cosmos3VLMConfig)) or not isinstance(
            vae.config, Wan22VAEConfig
        ):
            raise ValueError("Cosmos3 video requires native MoT and verified Wan2.2 codec")
        self.has_visual_understanding = isinstance(model.config, Cosmos3VLMConfig)
        self.mot_config = model.config.mot if self.has_visual_understanding else model.config
        if self.mot_config.latent_channel != vae.config.z_dim:
            raise ValueError("Cosmos3 and video codec latent channels differ")
        if (
            not math.isfinite(temporal_margin)
            or temporal_margin < 0
            or not math.isfinite(base_fps)
            or base_fps <= 0
        ):
            raise ValueError("Invalid Cosmos3 cross-modality position metadata")
        self.model, self.vae, self.temporal_margin, self.base_fps = (
            model,
            vae,
            temporal_margin,
            base_fps,
        )

    @torch.no_grad()
    def encode_video(self, video):

        if (
            video.ndim != 5
            or not video.is_floating_point()
            or not torch.isfinite(video).all()
            or (video.abs() > 1).any()
        ):
            raise ValueError("Cosmos3 codec input must be finite normalized RGB video in [-1,1]")
        parameter = next(self.vae.parameters())
        modes = {m: m.training for m in self.vae.modules()}
        try:
            self.vae.eval()
            with torch.autocast(parameter.device.type, enabled=False):
                return self.vae.latent(video.to(parameter), sample=False).to(video)
        finally:
            for module, mode in modes.items():
                module.training = mode

    @torch.no_grad()
    def decode_video(self, latent):
        parameter = next(self.vae.parameters())
        modes = {m: m.training for m in self.vae.modules()}
        try:
            self.vae.eval()
            with torch.autocast(parameter.device.type, enabled=False):
                return self.vae.decode(latent.to(parameter), scaled=True, clip_output=True).to(
                    latent
                )
        finally:
            for module, mode in modes.items():
                module.training = mode

    def _inputs(self, model_inputs, latent, *, fps, noisy_frames, timesteps=None):
        from aster.models.cosmos3 import Cosmos3Vision, cosmos3_positions

        inputs = dict(model_inputs)
        if any(
            key in inputs for key in ("vision", "state", "inputs_embeds", "understanding_positions")
        ):
            raise ValueError(
                "Video token-prefix pipeline takes IDs, not implicit packed/cached multimodal positions"
            )
        ids = inputs["input_ids"]
        mask = inputs.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
        if (
            ids.ndim != 2
            or ids.shape[0] != len(latent)
            or mask.shape != ids.shape
            or mask.dtype != torch.bool
            or not mask.any(-1).all()
        ):
            raise ValueError(
                "Cosmos3 video prefix requires aligned nonempty bool-masked token rows"
            )
        if ids.device != latent.device or mask.device != latent.device or ids.dtype != torch.int64:
            raise ValueError("Cosmos3 video token IDs, mask and latent must share device")
        if (
            latent.ndim != 5
            or latent.shape[1] != self.mot_config.latent_channel
            or min(latent.shape) < 1
        ):
            raise ValueError("Cosmos3 video latent shape is invalid")
        if (
            noisy_frames.dtype != torch.bool
            or noisy_frames.shape != (len(latent), latent.shape[2])
            or noisy_frames.device != latent.device
        ):
            raise ValueError("Cosmos3 noisy frame mask must be bool[B,T]")
        p = self.mot_config.latent_patch_size
        grid = (latent.shape[2], math.ceil(latent.shape[3] / p), math.ceil(latent.shape[4] / p))

        lengths = mask.sum(-1)
        if self.has_visual_understanding:
            from aster.models.qwen_vl import multimodal_positions

            config = self.model.config
            modalities = torch.where(
                ids == config.image_token_id, 1, torch.where(ids == config.video_token_id, 2, 0)
            )
            _, deltas = multimodal_positions(
                ids,
                modalities,
                config.vision_config.spatial_merge_size,
                inputs.get("image_grid_thw"),
                inputs.get("video_grid_thw"),
                mask,
            )
            lengths = lengths + deltas[:, 0]
        positions = torch.cat(
            tuple(
                cosmos3_positions(
                    grid,
                    temporal_offset=int(length) + self.temporal_margin,
                    fps=fps,
                    base_fps=self.base_fps,
                    temporal_compression=self.vae.config.temporal_stride,
                    device=latent.device,
                )
                for length in lengths.tolist()
            ),
            1,
        )
        inputs["attention_mask"] = mask
        if not self.has_visual_understanding:
            inputs["understanding_positions"] = (mask.long().cumsum(-1) - 1).masked_fill(~mask, 0)
        if timesteps is None:
            timesteps = latent.new_zeros(len(latent), latent.shape[2])
        inputs["vision"] = Cosmos3Vision(latent, positions, timesteps, noisy_frames)
        return inputs

    def training_batch(
        self, video, model_inputs, *, fps=24.0, noisy_frames=None, timesteps=None, noise=None
    ):

        latent = self.encode_video(video)
        if noisy_frames is None:
            noisy_frames = torch.ones(
                len(latent), latent.shape[2], dtype=torch.bool, device=latent.device
            )
        batch = dict(
            model_inputs=self._inputs(
                model_inputs, latent, fps=fps, noisy_frames=noisy_frames, timesteps=timesteps
            )
        )
        if noise is not None:
            batch["noise"] = noise
        return batch

    @torch.no_grad()
    def generate(
        self,
        model_inputs,
        noise,
        *,
        fps=24.0,
        condition_video=None,
        steps=20,
        solver="heun",
        shift=10.0,
    ):

        if noise.ndim != 5 or not noise.is_floating_point() or not torch.isfinite(noise).all():
            raise ValueError("Cosmos3 generation requires explicit finite BCTHW latent noise")
        latent = noise.clone()
        noisy = torch.ones(len(noise), noise.shape[2], dtype=torch.bool, device=noise.device)
        if condition_video is not None:
            condition = self.encode_video(condition_video).to(noise)
            if (
                condition.shape[:2] != noise.shape[:2]
                or condition.shape[-2:] != noise.shape[-2:]
                or condition.shape[2] > noise.shape[2]
            ):
                raise ValueError(
                    "Condition video latent cannot differ in B/C/H/W or exceed generation length"
                )
            latent[:, :, : condition.shape[2]] = condition
            noisy[:, : condition.shape[2]] = False
        inputs = self._inputs(model_inputs, latent, fps=fps, noisy_frames=noisy)
        latents = sample_cosmos3(self.model, inputs, steps=steps, solver=solver, shift=shift)
        return Cosmos3VideoOutput(self.decode_video(latents["vision"]), latents)


@dataclass
class Cosmos3AudioVideoOutput:
    video: torch.Tensor
    sound: torch.Tensor
    latents: dict[str, torch.Tensor]
    sampling_rate: int
    fps: float


class Cosmos3AudioVideoPipeline(Cosmos3VideoPipeline):
    def __init__(self, model, vae, audio_codec, **kwargs):
        super().__init__(model, vae, **kwargs)
        from aster.models.cosmos3_audio import Cosmos3AudioConfig

        if (
            not isinstance(audio_codec.config, Cosmos3AudioConfig)
            or self.mot_config.sound_dim != audio_codec.config.vocoder_input_dim
        ):
            raise ValueError(
                "Cosmos3 joint sound requires the actual AVAE2 and matching acoustic latent width"
            )
        self.audio_codec = audio_codec

    @torch.no_grad()
    def encode_audio(self, waveform, *, sampling_rate, sample_posterior=False, generator=None):
        if sampling_rate != self.audio_codec.config.sampling_rate:
            raise ValueError("AVAE2 sample rate mismatch; resampling is an explicit data stage")
        parameter = next(self.audio_codec.parameters())
        modes = {m: m.training for m in self.audio_codec.modules()}
        try:
            self.audio_codec.eval()
            with torch.autocast(parameter.device.type, enabled=False):
                posterior = self.audio_codec.encode(waveform.to(parameter), force_pad=True)
                latent = posterior.sample(generator) if sample_posterior else posterior.mode()
                return latent.transpose(1, 2).to(waveform)
        finally:
            for module, mode in modes.items():
                module.training = mode

    @torch.no_grad()
    def decode_audio(self, latent):
        parameter = next(self.audio_codec.parameters())
        modes = {m: m.training for m in self.audio_codec.modules()}
        try:
            self.audio_codec.eval()
            with torch.autocast(parameter.device.type, enabled=False):
                return self.audio_codec.decode(latent.transpose(1, 2).to(parameter)).to(latent)
        finally:
            for module, mode in modes.items():
                module.training = mode

    def _sound_inputs(self, inputs, latent, *, timesteps=None):
        from aster.models.cosmos3 import Cosmos3Sequence, cosmos3_positions

        if (
            "sound" in inputs
            or latent.ndim != 3
            or latent.shape[0] != inputs["input_ids"].shape[0]
            or latent.shape[-1] != self.mot_config.sound_dim
        ):
            raise ValueError("Cosmos3 sound latent needs explicit aligned BTD acoustic layout")
        if (
            min(latent.shape) < 1
            or not latent.is_floating_point()
            or latent.device != inputs["input_ids"].device
            or not torch.isfinite(latent).all()
        ):
            raise ValueError("Cosmos3 sound latent must be finite, nonempty and colocated")
        offsets = inputs["vision"].positions[0, :, 0]
        c = self.audio_codec.config
        positions = torch.cat(
            tuple(
                cosmos3_positions(
                    (latent.shape[1], 1, 1),
                    temporal_offset=float(offset),
                    fps=c.sampling_rate / c.hop_size,
                    base_fps=self.base_fps,
                    temporal_compression=1,
                    device=latent.device,
                )
                for offset in offsets.tolist()
            ),
            1,
        )
        if timesteps is None:
            timesteps = latent.new_zeros(latent.shape[:2])
        result = dict(inputs)
        result["sound"] = Cosmos3Sequence(
            latent,
            positions,
            timesteps,
            torch.ones(latent.shape[:2], dtype=torch.bool, device=latent.device),
        )
        return result

    def training_batch(
        self,
        video,
        audio,
        model_inputs,
        *,
        sampling_rate,
        fps=24.0,
        noisy_frames=None,
        timesteps=None,
        audio_timesteps=None,
        noise=None,
    ):
        batch = super().training_batch(
            video,
            model_inputs,
            fps=fps,
            noisy_frames=noisy_frames,
            timesteps=timesteps,
            noise=noise,
        )
        batch["model_inputs"] = self._sound_inputs(
            batch["model_inputs"],
            self.encode_audio(audio, sampling_rate=sampling_rate),
            timesteps=audio_timesteps,
        )
        return batch

    @torch.no_grad()
    def generate(
        self,
        model_inputs,
        noise,
        *,
        sound_noise,
        fps=24.0,
        condition_video=None,
        steps=20,
        solver="heun",
        shift=10.0,
    ):
        if (
            noise.ndim != 5
            or not noise.is_floating_point()
            or not torch.isfinite(noise).all()
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("Cosmos3 joint sampling requires finite explicit video noise and FPS")
        frames = (noise.shape[2] - 1) * self.vae.config.temporal_stride + 1
        c = self.audio_codec.config
        samples = int(frames / fps * c.sampling_rate)
        expected = math.ceil(samples / c.hop_size)
        if sound_noise.ndim != 3 or sound_noise.shape[1] != expected:
            raise ValueError(
                "Cosmos3 sound-noise length must match video duration rounded to the codec hop"
            )
        latent = noise.clone()
        noisy = torch.ones(len(noise), noise.shape[2], dtype=torch.bool, device=noise.device)
        if condition_video is not None:
            condition = self.encode_video(condition_video).to(noise)
            if (
                condition.shape[:2] != noise.shape[:2]
                or condition.shape[-2:] != noise.shape[-2:]
                or condition.shape[2] > noise.shape[2]
            ):
                raise ValueError("Condition video latent has an incompatible shape")
            latent[:, :, : condition.shape[2]] = condition
            noisy[:, : condition.shape[2]] = False
        inputs = self._inputs(model_inputs, latent, fps=fps, noisy_frames=noisy)
        inputs = self._sound_inputs(inputs, sound_noise)
        latents = sample_cosmos3(self.model, inputs, steps=steps, solver=solver, shift=shift)
        return Cosmos3AudioVideoOutput(
            self.decode_video(latents["vision"]),
            self.decode_audio(latents["sound"]),
            latents,
            c.sampling_rate,
            fps,
        )


def _frame_mask(field, name):
    mask = field.noisy_frames
    if field.valid_frames is not None:
        mask = mask & field.valid_frames
    return mask[:, None, :, None, None] if name == "vision" else mask[..., None]


class Cosmos3FlowObjective(nn.Module):
    def __init__(
        self,
        *,
        text_weight=0.0,
        vision_weight=1.0,
        sound_weight=1.0,
        action_weight=1.0,
        time_distribution="uniform",
        logit_mean=0.0,
        logit_std=1.0,
    ):
        super().__init__()
        self.weights = dict(
            text=text_weight, vision=vision_weight, sound=sound_weight, action=action_weight
        )
        if any(not math.isfinite(x) or x < 0 for x in self.weights.values()) or not any(
            self.weights.values()
        ):
            raise ValueError(
                "Cosmos3 objective weights must be finite/nonnegative with an active term"
            )
        if (
            time_distribution not in {"provided", "uniform", "logit_normal"}
            or not math.isfinite(logit_mean)
            or not math.isfinite(logit_std)
            or logit_std <= 0
        ):
            raise ValueError("Invalid Cosmos3 flow time distribution")
        self.time_distribution, self.logit_mean, self.logit_std = (
            time_distribution,
            logit_mean,
            logit_std,
        )
        self.path = FlowPath(direction="data_to_noise")

    def config_dict(self):
        return dict(
            type="cosmos3_flow",
            **{name + "_weight": value for name, value in self.weights.items()},
            time_distribution=self.time_distribution,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
        )

    def forward(self, model, batch):
        inputs = dict(batch["model_inputs"])
        if inputs.get("state") is not None or inputs.get("use_cache", False):
            raise ValueError(
                "Cosmos3 training recomputes the joint graph without cached detached conditions"
            )
        inputs.pop("use_cache", None)
        fields = {
            name: inputs[name]
            for name in ("vision", "sound", "action")
            if inputs.get(name) is not None
        }
        first = next(iter(fields.values()), None)
        times = None
        if self.time_distribution != "provided" and first is not None:
            data = first.sample
            times = (
                torch.rand(len(data), device=data.device)
                if self.time_distribution == "uniform"
                else (
                    torch.randn(len(data), device=data.device) * self.logit_std + self.logit_mean
                ).sigmoid()
            )
        targets = {}
        for name, field in fields.items():
            data = field.sample
            noise = batch.get("noise", {}).get(name)
            if noise is None:
                noise = torch.randn_like(data)
            if noise.shape != data.shape or noise.device != data.device:
                raise ValueError("Cosmos3 noise must match each modality tensor")
            frame_times = (
                field.timesteps / 1000
                if times is None
                else times[:, None].expand_as(field.timesteps)
            )

            data_frames = data.transpose(1, 2) if name == "vision" else data
            noise_frames = noise.transpose(1, 2) if name == "vision" else noise
            noisy, target = self.path.sample(
                data_frames.flatten(0, 1), noise_frames.flatten(0, 1), frame_times.flatten()
            )
            noisy, target = noisy.reshape_as(data_frames), target.reshape_as(data_frames)
            if name == "vision":
                noisy, target = noisy.transpose(1, 2), target.transpose(1, 2)
            mask = _frame_mask(field, name)
            inputs[name] = replace(
                field, sample=torch.where(mask, noisy, data), timesteps=frame_times * 1000
            )
            targets[name] = target
        output = model(**inputs, use_cache=False)
        terms = []
        for name, field in fields.items():
            if not self.weights[name]:
                continue
            prediction = getattr(output, name)
            if (
                prediction is None
                or prediction.prediction_type != "velocity"
                or prediction.prediction.shape != field.sample.shape
            ):
                raise ValueError(
                    "Cosmos3 joint flow output must preserve each modality shape/velocity"
                )
            mask = _frame_mask(field, name).expand_as(field.sample)
            error = (prediction.prediction.float() - targets[name].float()).square()
            terms.append(
                LossTerm(
                    error.masked_select(mask).sum(),
                    mask.sum(dtype=torch.int64),
                    name + "_scalar",
                    name + "_flow",
                    self.weights[name],
                )
            )
        if self.weights["text"]:
            logits, targets_text, mask = token_targets(batch, output.text.logits)
            safe_targets = targets_text.masked_fill(~mask, 0)
            values = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), safe_targets.flatten(), reduction="none"
            ).reshape_as(safe_targets)
            terms.append(
                LossTerm(
                    values.masked_select(mask).sum(),
                    mask.sum(dtype=torch.int64),
                    "token",
                    "ce",
                    self.weights["text"],
                )
            )
        if not terms:
            raise ValueError("Cosmos3 batch has no active supervised modality")
        return LossBundle(tuple(terms))


class Cosmos3VisualFlowObjective(Cosmos3FlowObjective):
    def __init__(self, *, visual_prefill="image", **kwargs):
        super().__init__(**kwargs)
        if visual_prefill not in {"image", "video", "none"}:
            raise ValueError("Invalid Cosmos3 visual_prefill contract")
        self.visual_prefill = visual_prefill

    def config_dict(self):
        return dict(
            super().config_dict(), type="cosmos3_visual_flow", visual_prefill=self.visual_prefill
        )

    def _validate(self, model, batch):
        from aster.models.cosmos3_vlm import Cosmos3VLMConfig
        from aster.models.cosmos3 import Cosmos3MoT
        from aster.models.qwen_vl import _validated_grids, multimodal_positions

        if not isinstance(model.config, Cosmos3VLMConfig):
            raise ValueError("Visual flow requires the actual Cosmos3 Qwen wrapper")
        inputs = batch["model_inputs"]
        ids = inputs["input_ids"]
        c = model.config
        allowed = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "vision",
            "sound",
            "action",
            "state",
            "use_cache",
            "output_hidden_states",
        }
        if set(inputs) - allowed:
            raise ValueError("Cosmos3 visual training has unsupported model input keys")
        if inputs.get("state") is not None or inputs.get("use_cache", False):
            raise ValueError("Cosmos3 visual training cannot reuse a detached prefill cache")
        if (
            ids.ndim != 2
            or min(ids.shape) < 1
            or ids.dtype != torch.int64
            or ((ids < 0) | (ids >= c.mot.vocab_size)).any()
        ):
            raise ValueError("Cosmos3 visual training requires valid int64[B,S] token IDs")
        mask = inputs.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
        if (
            mask.shape != ids.shape
            or mask.dtype != torch.bool
            or mask.device != ids.device
            or not mask.any(-1).all()
        ):
            raise ValueError("Cosmos3 visual training mask must contain a valid token per row")
        actual = []
        for kind, pixels_key, grid_key in (
            ("image", "pixel_values", "image_grid_thw"),
            ("video", "pixel_values_videos", "video_grid_thw"),
        ):
            pixels, grid = inputs.get(pixels_key), inputs.get(grid_key)
            if (pixels is None) != (grid is None):
                raise ValueError("Cosmos3 prefill pixel/grid pair must be complete")
            if pixels is None:
                continue
            actual.append(kind)
            grids = _validated_grids(grid, c.vision_config)
            width = (
                c.vision_config.in_channels
                * c.vision_config.temporal_patch_size
                * c.vision_config.patch_size**2
            )
            if (
                pixels.shape != (sum(t * h * w for t, h, w in grids), width)
                or not pixels.is_floating_point()
                or not torch.isfinite(pixels).all()
            ):
                raise ValueError(
                    "Cosmos3 visual training patches do not match the declared grid/feature width"
                )
            if kind == "image" and any(t != 1 for t, _, _ in grids):
                raise ValueError("Cosmos3 image prefill requires T=1 grids")
        if actual != ([] if self.visual_prefill == "none" else [self.visual_prefill]):
            raise ValueError("Cosmos3 visual_prefill must be identical before parameter gathers")
        modalities = torch.where(
            ids == c.image_token_id, 1, torch.where(ids == c.video_token_id, 2, 0)
        )
        if ((modalities != 0) & ~mask).any():
            raise ValueError("Cosmos3 visual placeholders cannot be padded away")
        for row in ids:
            for visual_id in (c.image_token_id, c.video_token_id):
                for index in torch.nonzero(row == visual_id).flatten().tolist():
                    if (index == 0 or row[index - 1] != visual_id) and (
                        index == 0 or row[index - 1] != c.vision_start_token_id
                    ):
                        raise ValueError("Cosmos3 visual spans need an explicit vision_start token")
                    if (index + 1 == len(row) or row[index + 1] != visual_id) and (
                        index + 1 == len(row) or row[index + 1] != c.vision_end_token_id
                    ):
                        raise ValueError("Cosmos3 visual spans need an explicit vision_end token")
        multimodal_positions(
            ids,
            modalities,
            c.vision_config.spatial_merge_size,
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            mask,
        )

        for name in ("vision", "sound", "action"):
            value = inputs.get(name)
            if value is None:
                continue
            x = value.sample
            width = c.mot.latent_channel if name == "vision" else getattr(c.mot, name + "_dim")
            if (
                width is None
                or not isinstance(x, torch.Tensor)
                or not x.is_floating_point()
                or x.device != ids.device
                or min(x.shape) < 1
            ):
                raise ValueError(
                    "Cosmos3 generation field is disabled or has invalid tensor metadata"
                )
            if name == "vision":
                if x.ndim != 5 or x.shape[:2] != (len(ids), width):
                    raise ValueError("Cosmos3 vision field must be aligned BCTHW")
                length = x.shape[2]
                p = c.mot.latent_patch_size
                positions_length = length * math.ceil(x.shape[3] / p) * math.ceil(x.shape[4] / p)
            else:
                if x.ndim != 3 or x.shape[0] != len(ids) or x.shape[-1] != width:
                    raise ValueError("Cosmos3 sequence field must be aligned BTD")
                length = positions_length = x.shape[1]
                domain = value.domain_ids
                if name == "sound" and domain is not None:
                    raise ValueError("Sound cannot choose action embodiment domains")
                if name == "action" and (
                    domain is None
                    or domain.shape != (len(ids),)
                    or domain.dtype != torch.long
                    or domain.device != ids.device
                    or ((domain < 0) | (domain >= c.mot.num_embodiment_domains)).any()
                ):
                    raise ValueError(
                        "Cosmos3 action domain IDs differ from configured embodiment table"
                    )
            Cosmos3MoT._field_mask(value, len(ids), length, ids.device)
            Cosmos3MoT._positions(value.positions, len(ids), positions_length, ids.device)
            noise = batch.get("noise", {}).get(name)
            if noise is not None and (
                noise.shape != x.shape or noise.device != x.device or not noise.is_floating_point()
            ):
                raise ValueError("Cosmos3 explicit modality noise shape/device/dtype is invalid")

    def preflight_microbatches(self, model, batches):
        for batch in batches:
            self._validate(model, batch)
        return batches

    def forward(self, model, batch):
        self._validate(model, batch)
        return super().forward(model, batch)


@torch.no_grad()
def sample_cosmos3(
    model, model_inputs, *, steps=20, solver="heun", shift=10.0, reuse_understanding=True
):

    inputs = dict(model_inputs)
    inputs.pop("use_cache", None)
    fields = {
        name: inputs.pop(name)
        for name in ("vision", "sound", "action")
        if inputs.get(name) is not None
    }
    if not fields:
        raise ValueError("Cosmos3 sampling needs at least one latent modality")
    modes = {module: module.training for module in model.modules()}
    sizes = {name: field.sample[0].numel() for name, field in fields.items()}
    initial = torch.cat(tuple(field.sample.flatten(1) for field in fields.values()), -1)
    try:
        model.eval()
        if reuse_understanding:
            prefix = model.forward_text(**inputs, use_cache=True)
            inputs = dict(state=prefix.state)

        def joint_field(flattened, time, condition):
            del condition
            current, start = dict(inputs), 0
            for name, field in fields.items():
                end = start + sizes[name]
                values = flattened[:, start:end].reshape_as(field.sample)
                current[name] = replace(
                    field, sample=values, timesteps=time[:, None].expand_as(field.timesteps) * 1000
                )
                start = end
            output = model(**current, use_cache=False)
            combined = torch.cat(
                tuple(getattr(output, name).prediction.flatten(1) for name in fields), -1
            )
            return FieldOutput(combined, "velocity")

        final = sample_flow(
            joint_field, initial, steps=steps, solver=solver, shift=shift, direction="data_to_noise"
        )
        result, start = {}, 0
        for name, field in fields.items():
            end = start + sizes[name]
            result[name] = final[:, start:end].reshape_as(field.sample)
            start = end
        return result
    finally:
        for module, mode in modes.items():
            module.training = mode
