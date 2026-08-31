"""Native latent encoding, conditional-field training, and decoding pipelines."""

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
from pathlib import Path
import re
import torch
from torch import nn
from .core import atomic_json, read_json, digest_json, file_digest
from .core.update_provenance import validate_successful_update_record
from .models import load_model
from .methods.generation import (
    FlowPath,
    FlowObjective,
    DiffusionSchedule,
    sample_flow,
    sample_diffusion,
    sample_edm,
    karras_sigmas,
)


@dataclass(frozen=True)
class LatentPipelineConfig:
    method: str = "flow"
    solver: str = "heun"
    steps: int = 20
    direction: str = "noise_to_data"
    shift: float = 1.0
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    diffusion_steps: int = 1000
    diffusion_schedule: str = "cosine"
    learned_variance: bool = False
    eta: float = 0.0
    clip_clean: bool = False

    def __post_init__(self):
        if (
            self.method not in {"flow", "diffusion", "edm"}
            or type(self.steps) is not int
            or self.steps < 1
        ):
            raise ValueError("Invalid latent sampling configuration")
        for value in (self.shift, self.sigma_data, self.sigma_min, self.sigma_max):
            if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                raise ValueError("Latent scales must be finite and positive")
        if self.sigma_min >= self.sigma_max or self.direction not in {
            "noise_to_data",
            "data_to_noise",
        }:
            raise ValueError("Invalid noise range/direction")
        if (
            type(self.diffusion_steps) is not int
            or self.diffusion_steps < 2
            or self.diffusion_schedule not in {"linear", "cosine"}
        ):
            raise ValueError("Invalid original diffusion schedule declaration")
        if (
            type(self.learned_variance) is not bool
            or type(self.clip_clean) is not bool
            or type(self.eta) not in {float, int}
            or not math.isfinite(self.eta)
            or not 0 <= self.eta <= 1
        ):
            raise ValueError("Invalid diffusion variance/clip/eta controls")
        if self.method != "diffusion" and (
            self.learned_variance or self.clip_clean or self.eta != 0
        ):
            raise ValueError("Diffusion controls cannot be silently applied to flow/EDM")
        if self.method == "diffusion" and self.solver == "ddpm" and self.eta != 0:
            raise ValueError("Eta is a DDIM control, not DDPM")
        allowed = {"flow": {"euler", "heun", "rk4"}, "diffusion": {"ddpm", "ddim"}, "edm": {"heun"}}
        if self.solver not in allowed[self.method]:
            raise ValueError("Solver does not match the field parameterization")
        if self.method == "diffusion" and not 2 <= self.steps <= self.diffusion_steps:
            raise ValueError("Respaced diffusion needs between two and original training steps")
        if self.method == "edm" and self.steps < 2:
            raise ValueError("EDM requires at least two positive sigma steps")


def _model_fingerprint(model):

    tensors = {}
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        tensors[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(
                value.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    return digest_json({"config": model.config.to_dict(), "tensors": tensors})


def _schedule_from_record(record):
    if not isinstance(record, dict) or set(record) != {"betas", "timestep_map"}:
        raise ValueError("Original schedule needs complete betas and timestep_map")
    betas, mapping = record["betas"], record["timestep_map"]
    if not isinstance(betas, list) or any(
        type(v) not in {int, float} or not math.isfinite(v) for v in betas
    ):
        raise ValueError("Invalid original beta values")
    if not isinstance(mapping, list) or any(type(v) is not int or v < 0 for v in mapping):
        raise ValueError("Model time map must contain nonnegative integer coordinates")
    return DiffusionSchedule(betas, timestep_map=mapping)


class LatentGenerationPipeline:
    def __init__(
        self,
        autoencoder,
        field,
        config=LatentPipelineConfig(),
        *,
        conditioning_metadata=None,
        diffusion_schedule=None,
    ):
        if type(config) is not LatentPipelineConfig:
            raise ValueError("Typed latent pipeline config required")
        if autoencoder.config.latent_channels != field.config.in_channels:
            raise ValueError("VAE latent channels and field input channels differ")
        self.autoencoder, self.field, self.config = autoencoder.eval(), field.eval(), config
        self.conditioning_metadata = dict(conditioning_metadata or {"kind": "unconditional"})
        digest_json(self.conditioning_metadata)
        expected = {
            "flow": {"velocity"},
            "diffusion": {"epsilon", "x0", "v", "score"},
            "edm": {"edm_residual"},
        }
        if field.config.prediction_type not in expected[config.method]:
            raise ValueError("Pipeline method differs from field prediction parameterization")
        channels = field.config.out_channels or field.config.in_channels
        if channels != field.config.in_channels * (2 if config.learned_variance else 1):
            raise ValueError("Field output channels differ from learned variance declaration")
        self._original = None
        self._indices = None
        if config.method == "diffusion":
            original = (
                DiffusionSchedule.create(config.diffusion_steps, config.diffusion_schedule)
                if diffusion_schedule is None
                else diffusion_schedule
            )
            if type(original) is not DiffusionSchedule:
                raise ValueError("Explicit original schedule must be native DiffusionSchedule")
            record = {
                "betas": original.betas.detach().cpu().tolist(),
                "timestep_map": original.timestep_map.detach().cpu().tolist(),
            }
            checked = _schedule_from_record(record)
            if len(checked) != config.diffusion_steps:
                raise ValueError(
                    "Declared original diffusion length differs from actual training chain"
                )

            self._original = copy.deepcopy(record)
            self._indices = tuple(
                torch.linspace(0, len(checked) - 1, config.steps).round().long().tolist()
            )
        elif diffusion_schedule is not None:
            raise ValueError("Only discrete diffusion accepts an original beta schedule")
        self._source = {
            "kind": "caller_config" if diffusion_schedule is None else "caller_explicit_schedule",
            "training_semantics_bound": False,
        }
        self._configuration_id = digest_json(asdict(config))
        self._model_config_ids = (
            digest_json(autoencoder.config.to_dict()),
            digest_json(field.config.to_dict()),
        )

    @property
    def sampling_binding(self):

        return copy.deepcopy(
            {
                "schema_version": 1,
                "source": self._source,
                "original_schedule": self._original,
                "original_schedule_id": None
                if self._original is None
                else digest_json(self._original),
                "selected_training_indices": None if self._indices is None else list(self._indices),
                "configuration_id": self._configuration_id,
                "model_config_ids": list(self._model_config_ids),
            }
        )

    def _check_binding(self):
        if digest_json(asdict(self.config)) != self._configuration_id or self._model_config_ids != (
            digest_json(self.autoencoder.config.to_dict()),
            digest_json(self.field.config.to_dict()),
        ):
            raise ValueError(
                "Pipeline configuration/model identity changed; construct a new pipeline explicitly"
            )
        source = self._source
        if source["kind"] != "verified_training_artifacts":
            return
        objective = source["training_objective"]
        inner = objective.get("objective", {})
        update = source.get("successful_update")
        validate_successful_update_record(
            update,
            {
                "class": "aster.pipelines.LatentFieldObjective",
                "codec": "config_dict",
                "configuration": objective,
            },
            role_updates=update.get("role_updates") if isinstance(update, dict) else None,
        )
        if (
            objective.get("type") != "latent_field"
            or objective.get("encoder_identity") != source["autoencoder_artifact_id"]
        ):
            raise ValueError(
                "Training latent encoder identity differs from the actual decoder artifact"
            )
        if self.config.method == "diffusion":
            if (
                inner.get("type") != "diffusion"
                or type(inner.get("learned_variance")) is not bool
                or inner["learned_variance"] != self.config.learned_variance
                or {"betas": inner.get("betas"), "timestep_map": inner.get("timestep_map")}
                != self._original
            ):
                raise ValueError(
                    "Pipeline original schedule/variance differs from the trained diffusion objective"
                )
        elif self.config.method == "flow":
            if (
                inner.get("type") != "flow_matching"
                or inner.get("direction") != self.config.direction
            ):
                raise ValueError("Pipeline flow direction differs from the trained objective")
        elif inner.get("type") != "edm" or inner.get("sigma_data") != self.config.sigma_data:
            raise ValueError("Pipeline EDM preconditioning differs from the trained objective")
        if (
            _model_fingerprint(self.field) != source["field_weight_fingerprint"]
            or _model_fingerprint(self.autoencoder) != source["autoencoder_weight_fingerprint"]
        ):
            raise ValueError("Artifact-bound pipeline weights changed since source verification")

    @classmethod
    def from_artifacts(
        cls,
        store,
        autoencoder_artifact_id,
        field_artifact_id,
        config,
        *,
        device="cpu",
        conditioning_metadata=None,
    ):

        auto_artifact, field_artifact = (
            store.get(autoencoder_artifact_id, verify=True),
            store.get(field_artifact_id, verify=True),
        )

        def load(artifact):
            paths = [
                path
                for path in (artifact.path, artifact.path / "model")
                if (path / "config.json").is_file()
            ]
            if len(paths) != 1:
                raise ValueError("Artifact must have exactly one native model layout")
            return load_model(paths[0]), paths[0].relative_to(artifact.path).as_posix()

        with torch.random.fork_rng(devices=[]):
            autoencoder, auto_path = load(auto_artifact)
            field, field_path = load(field_artifact)
        objective_path = field_artifact.path / "objective.json"
        objective = read_json(objective_path)
        if (
            not isinstance(objective, dict)
            or objective.get("type") != "latent_field"
            or objective.get("encoder_identity") != autoencoder_artifact_id
        ):
            raise ValueError(
                "Artifact pipeline requires a training latent encoder identity matching the decoder"
            )
        update_path = field_artifact.path / "successful_update.json"
        if not update_path.is_file():
            raise ValueError(
                "Training-bound pipeline needs actual successful objective provenance; legacy declarations require explicit caller-owned reconstruction"
            )
        actual_update = read_json(update_path)
        inner = objective.get("objective", {})
        original = (
            _schedule_from_record(
                {"betas": inner.get("betas"), "timestep_map": inner.get("timestep_map")}
            )
            if config.method == "diffusion"
            else None
        )
        pipeline = cls(
            autoencoder.to(device),
            field.to(device),
            config,
            conditioning_metadata=conditioning_metadata,
            diffusion_schedule=original,
        )
        pipeline._source = {
            "kind": "verified_training_artifacts",
            "training_semantics_bound": True,
            "autoencoder_artifact_id": autoencoder_artifact_id,
            "field_artifact_id": field_artifact_id,
            "autoencoder_model_layout": auto_path,
            "field_model_layout": field_path,
            "objective_file_sha256": file_digest(objective_path),
            "training_objective": objective,
            "successful_update": actual_update,
            "successful_update_file_sha256": file_digest(update_path),
            "field_weight_fingerprint": _model_fingerprint(field),
            "autoencoder_weight_fingerprint": _model_fingerprint(autoencoder),
        }
        pipeline._check_binding()
        store.get(autoencoder_artifact_id, verify=True)
        store.get(field_artifact_id, verify=True)
        return pipeline

    @torch.no_grad()
    def encode(self, images, *, sample=False, generator=None):
        return self.autoencoder.latent(images, sample=sample, generator=generator)

    @torch.no_grad()
    def decode(self, latent):
        return self.autoencoder.decode(latent, scaled=True)

    @torch.no_grad()
    def sample(self, noise, *, condition=None, guidance_scale=1.0, generator=None):
        self._check_binding()
        c = self.config
        if noise.ndim != 4 or noise.shape[1] != self.autoencoder.config.latent_channels:
            raise ValueError("Noise must be BCHW in the saved latent coordinate system")
        self.autoencoder.eval()
        self.field.eval()
        if c.method == "flow":
            latent = sample_flow(
                self.field,
                noise,
                steps=c.steps,
                solver=c.solver,
                direction=c.direction,
                shift=c.shift,
                condition=condition,
                guidance_scale=guidance_scale,
            )
        elif c.method == "diffusion":
            original = _schedule_from_record(self._original)
            schedule = (
                original
                if self._indices == tuple(range(len(original)))
                else original.respaced(self._indices)
            )
            latent = sample_diffusion(
                self.field,
                noise,
                schedule.to(noise.device),
                method=c.solver,
                condition=condition,
                guidance_scale=guidance_scale,
                generator=generator,
                learned_variance=c.learned_variance,
                eta=c.eta,
                clip_clean=c.clip_clean,
            )
        else:
            if guidance_scale != 1.0:
                raise ValueError(
                    "EDM pipeline CFG requires explicit conditioned preconditioning; no silent ignore"
                )
            sigmas = karras_sigmas(
                c.steps, sigma_min=c.sigma_min, sigma_max=c.sigma_max, device=noise.device
            )
            latent = sample_edm(
                self.field,
                noise,
                sigmas,
                condition=condition,
                sigma_data=c.sigma_data,
                generator=generator,
            )
        return self.decode(latent)

    def save_pretrained(self, directory):
        self._check_binding()
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Latent pipeline export must be new")
        self.autoencoder.save_pretrained(directory / "autoencoder")
        self.field.save_pretrained(directory / "field")
        binding = self.sampling_binding
        atomic_json(
            directory / "pipeline.json",
            {
                "schema_version": 2,
                "config": asdict(self.config),
                "conditioning": self.conditioning_metadata,
                "sampling_binding": binding,
                "sampling_binding_id": digest_json(binding),
                "latent_transform": {
                    "scaling_factor": self.autoencoder.config.scaling_factor,
                    "shift_factor": self.autoencoder.config.shift_factor,
                },
            },
        )

    @classmethod
    def from_pretrained(cls, directory, *, device="cpu"):
        directory = Path(directory)
        metadata = read_json(directory / "pipeline.json")
        if metadata["schema_version"] != 2:
            raise ValueError(
                "Legacy/unrecognized pipeline lacks a bound original schedule; rebuild explicitly from caller config or training artifacts"
            )
        if set(metadata) != {
            "schema_version",
            "config",
            "conditioning",
            "sampling_binding",
            "sampling_binding_id",
            "latent_transform",
        }:
            raise ValueError("Unknown pipeline metadata fields")
        binding = metadata["sampling_binding"]
        if digest_json(binding) != metadata["sampling_binding_id"]:
            raise ValueError("Pipeline schedule/source binding checksum mismatch")
        with torch.random.fork_rng(devices=[]):
            autoencoder = load_model(directory / "autoencoder").to(device)
            field = load_model(directory / "field").to(device)
        expected = {
            "scaling_factor": autoencoder.config.scaling_factor,
            "shift_factor": autoencoder.config.shift_factor,
        }
        if metadata["latent_transform"] != expected:
            raise ValueError("Pipeline and autoencoder latent coordinate systems differ")
        original = binding["original_schedule"]
        pipeline = cls(
            autoencoder,
            field,
            LatentPipelineConfig(**metadata["config"]),
            conditioning_metadata=metadata["conditioning"],
            diffusion_schedule=None if original is None else _schedule_from_record(original),
        )
        source = binding["source"]
        if not isinstance(source, dict) or source.get("kind") not in {
            "caller_config",
            "caller_explicit_schedule",
            "verified_training_artifacts",
        }:
            raise ValueError(
                "Pipeline source must distinguish caller intent from verified training facts"
            )
        if source["kind"] == "verified_training_artifacts":
            required = {
                "kind",
                "training_semantics_bound",
                "autoencoder_artifact_id",
                "field_artifact_id",
                "autoencoder_model_layout",
                "field_model_layout",
                "objective_file_sha256",
                "training_objective",
                "field_weight_fingerprint",
                "autoencoder_weight_fingerprint",
                "successful_update",
                "successful_update_file_sha256",
            }
            if (
                set(source) != required
                or source["training_semantics_bound"] is not True
                or any(
                    not isinstance(source[name], str)
                    or not re.fullmatch("[a-f0-9]{64}", source[name])
                    for name in (
                        "autoencoder_artifact_id",
                        "field_artifact_id",
                        "objective_file_sha256",
                        "field_weight_fingerprint",
                        "autoencoder_weight_fingerprint",
                        "successful_update_file_sha256",
                    )
                )
            ):
                raise ValueError("Invalid verified training source identities")
        elif source != {"kind": source["kind"], "training_semantics_bound": False}:
            raise ValueError("Caller-owned configuration cannot claim verified training evidence")
        pipeline._source = copy.deepcopy(source)
        if pipeline.sampling_binding != binding:
            raise ValueError(
                "Saved pipeline schedule/model/respacing differs from its configuration"
            )
        pipeline._check_binding()
        return pipeline


class LatentFieldObjective(nn.Module):
    def __init__(self, autoencoder, objective=None, *, encoder_identity, sample_posterior=True):
        super().__init__()
        if not encoder_identity:
            raise ValueError("Latent training requires encoder artifact identity")
        self.autoencoder = autoencoder.eval().requires_grad_(False)
        self.objective = objective if objective is not None else FlowObjective(FlowPath())
        self.encoder_identity, self.sample_posterior = encoder_identity, sample_posterior

    def config_dict(self):
        return {
            "type": "latent_field",
            "encoder_identity": self.encoder_identity,
            "sample_posterior": self.sample_posterior,
            "objective": self.objective.config_dict(),
        }

    def forward(self, model, batch):
        if "latent" in batch:
            if batch.get("encoder_identity") != self.encoder_identity:
                raise ValueError("Cached latent belongs to a different encoder artifact")
            sample = batch["latent"]
        else:
            self.autoencoder.eval()
            with torch.no_grad():
                sample = self.autoencoder.latent(batch["pixels"], sample=self.sample_posterior)
        payload = {
            key: value
            for key, value in batch.items()
            if key not in {"pixels", "latent", "encoder_identity"}
        }
        payload["sample"] = sample
        return self.objective(model, payload)
