"""One-forward Drifting image generation from native trained artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import math
from pathlib import Path
import re
import time

import torch
from PIL import Image

from ..core import atomic_json, digest_json, file_digest, read_json
from .generation_artifacts import load_native_artifact_model, _role_fingerprint
from .generative import (
    GenerationCase,
    ImageFile,
    MediaManifest,
    MediaSample,
    quantize_image,
    runtime_environment,
    _image_identity,
    _merge_image_shards,
)


@dataclass(frozen=True)
class DriftingSamplingPlan:
    cases: tuple[GenerationCase, ...]
    noise_shape: tuple[int, int, int]
    cfg_scale: float = 1.0
    temperature: float = 1.0
    quantization: str = "minus_one_one_stylegan"

    def __post_init__(self):
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "noise_shape", tuple(self.noise_shape))
        if not self.cases or any(
            not isinstance(case, GenerationCase)
            or type(case.condition) is not int
            or case.condition < 0
            for case in self.cases
        ):
            raise ValueError("Drifting cases require explicit nonnegative integer class labels")
        if (
            len({case.id for case in self.cases}) != len(self.cases)
            or len(self.noise_shape) != 3
            or any(type(x) is not int or x < 1 for x in self.noise_shape)
        ):
            raise ValueError("Drifting needs unique cases and explicit CHW noise shape")
        if (
            any(
                type(x) not in {int, float} or not math.isfinite(x)
                for x in (self.cfg_scale, self.temperature)
            )
            or self.cfg_scale < 1
            or self.temperature <= 0
        ):
            raise ValueError(
                "Drifting guidance embedding is >=1 and noise temperature must be positive"
            )
        if self.quantization not in {"minus_one_one_stylegan", "zero_one_round"}:
            raise ValueError("Unknown Drifting image quantization rule")

    @property
    def id(self):
        return digest_json(asdict(self))

    @property
    def cohort_id(self):

        return digest_json(
            {"cases": [asdict(case) for case in self.cases], "quantization": self.quantization}
        )


def _noise_semantics(config):
    return {
        "continuous": "fp32_standard_normal_times_temperature",
        "discrete": "uniform_int_after_continuous_same_generator"
        if config.noise_classes
        else "none",
        "noise_classes": config.noise_classes,
        "noise_coords": config.noise_coords,
    }


def publish_drifting_generator(method, store, directory, *, ema=False, parents=()):

    from ..methods.drifting import DriftingMethod
    from ..models.drifting import DriftingGenerator
    from ..training.sharding import Zero3Unit
    from ..training.recipes import agree, collective_local, leader_call

    if type(method) is not DriftingMethod or type(ema) is not bool:
        raise ValueError("Export requires native DriftingMethod and explicit bool EMA selection")
    engine = method.engine
    context = engine.parallel
    parents = tuple(parents)

    def preflight():
        if engine.states.get("drifting_method") is not method:
            raise ValueError("Unregistered Drifting lifecycle")
        state = method.state_dict()
        module = engine.roles["model"].model
        module = module.module if isinstance(module, Zero3Unit) else module
        if (
            type(module) is not DriftingGenerator
            or state["updates"] < 1
            or engine.roles["model"].updates < 1
        ):
            raise ValueError("Publish only a genuinely updated native Drifting generator")
        if any(parameter.dtype != torch.float32 for parameter in module.parameters()):
            raise ValueError("Drifting deployment currently requires FP32 parameter storage")
        if ema and engine.roles["model"].ema is None:
            raise ValueError("Requested EMA does not exist in this Trainer")

        return module.config, {
            "method_updates": state["updates"],
            "settings": state["settings"],
            "generator_updates": engine.roles["model"].updates,
            "weight_selection": "ema" if ema else "model",
            "ema_decay": engine.roles["model"].ema.decay if ema else None,
            "trainer_precision": engine.precision,
            "parallel": context.to_dict(),
            "deployment_only": True,
        }

    config, training = collective_local(context, preflight, "Validate Drifting deployment boundary")
    agree(
        context,
        {"directory": str(Path(directory).absolute()), "training": training, "parents": parents},
        "Drifting deployment identity",
    )
    tensors = engine.export_state_dict(ema=ema)

    def publish():
        for parent in parents:
            store.get(parent, verify=True)
        target = Path(directory).absolute()
        target.mkdir(parents=True, exist_ok=False)

        with torch.random.fork_rng(devices=[]):
            model = DriftingGenerator(config)
        model.load_state_dict(tensors, strict=True, assign=True)
        model.save_pretrained(target / "model")
        contract = {
            "schema_version": 1,
            "method": "drifting",
            "prediction_type": "x0",
            "conditioning_semantics": "guidance_embedding",
            "noise_semantics": _noise_semantics(config),
            "model_weight_fingerprint": _role_fingerprint(config.to_dict(), tensors),
            "training": training,
        }
        atomic_json(target / "generation_contract.json", contract)
        artifact = store.publish(
            target,
            kind="native_drifting_generator",
            metadata={
                "method": "drifting",
                "conditioning_semantics": "guidance_embedding",
                "generation_contract_id": digest_json(contract),
                "training_checkpoint_included": False,
            },
            parents=parents,
        )
        return artifact.id

    artifact_id = leader_call(context, publish, "Publish Drifting generator")
    return collective_local(
        context, lambda: store.get(artifact_id, verify=True), "Verify published Drifting artifact"
    )


def _binding(artifact, model, layout, plan):
    contract = read_json(artifact.path / "generation_contract.json")
    fields = {
        "schema_version",
        "method",
        "prediction_type",
        "conditioning_semantics",
        "noise_semantics",
        "model_weight_fingerprint",
        "training",
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != fields
        or contract["schema_version"] != 1
        or contract["method"] != "drifting"
        or contract["prediction_type"] != "x0"
        or contract["conditioning_semantics"] != "guidance_embedding"
    ):
        raise ValueError(
            "Drifting needs its own guidance-embedding contract, never DMD generator_time"
        )
    if contract["noise_semantics"] != _noise_semantics(model.config):
        raise ValueError("Drifting continuous/discrete noise contract differs from the model")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError(
            "Drifting sampling supports FP32 stored weights; no silent dtype conversion"
        )
    if contract["model_weight_fingerprint"] != _role_fingerprint(
        model.config.to_dict(), model.state_dict()
    ):
        raise ValueError("Drifting model weights differ from their deployed snapshot")
    training = contract["training"]
    if (
        not isinstance(training, dict)
        or any(
            type(training.get(key)) is not int or training[key] < 1
            for key in ("method_updates", "generator_updates")
        )
        or training.get("weight_selection") not in {"model", "ema"}
    ):
        raise ValueError("Drifting contract lacks completed training/weight-selection evidence")
    settings = training.get("settings")
    if not isinstance(settings, dict) or settings.get("model") != model.config.to_dict():
        raise ValueError("Drifting training settings are not bound to this model configuration")
    if (artifact.path / "objective.json").exists():
        raise ValueError("Drifting cannot also declare a VP/flow time objective")
    if plan.noise_shape != (
        model.config.in_channels,
        model.config.input_size,
        model.config.input_size,
    ):
        raise ValueError("Drifting plan noise geometry differs from its fixed native model")
    return {
        "schema_version": 1,
        "policy_artifact_id": artifact.id,
        "model_relative_path": layout,
        "sampling_mode": "drifting_direct",
        "conditioning_semantics": "guidance_embedding",
        "generation_contract": contract,
        "generation_contract_sha256": file_digest(artifact.path / "generation_contract.json"),
        "cfg_scale": plan.cfg_scale,
        "temperature": plan.temperature,
        "inside_declared_training_cfg_range": settings["cfg_min"]
        <= plan.cfg_scale
        <= settings["cfg_max"],
        "training_semantics_bound": True,
        "forward_calls_per_successful_sample": 1,
    }


def _sources():
    root = Path(__file__).resolve().parents[1]
    names = (
        "evaluation/drifting_generation.py",
        "evaluation/generative.py",
        "evaluation/generation_artifacts.py",
        "models/drifting.py",
        "models/generative.py",
        "models/serialization.py",
        "models/config.py",
        "models/__init__.py",
    )
    return {name: file_digest(root / name) for name in names}


def generate_drifting_shard(
    store,
    policy_artifact_id,
    plan,
    output_directory,
    *,
    rank=0,
    world_size=1,
    decoder_artifact_id=None,
    device="cpu",
):

    from ..models.drifting import DriftingGenerator
    from ..models.generative import AutoencoderKL

    if (
        not isinstance(plan, DriftingSamplingPlan)
        or type(rank) is not int
        or type(world_size) is not int
        or not 0 <= rank < world_size
    ):
        raise ValueError("A typed Drifting plan and valid shard are required")
    artifact = store.get(policy_artifact_id, verify=True)
    model, layout = load_native_artifact_model(artifact)
    if type(model) is not DriftingGenerator:
        raise ValueError("Drifting producer only implements the native DriftingGenerator")
    binding = _binding(artifact, model, layout, plan)
    model = model.eval().to(device)
    decoder = None
    parents = (policy_artifact_id,)
    binding["decoder"] = None
    if decoder_artifact_id is not None:
        decoder, decoder_path = load_native_artifact_model(
            store.get(decoder_artifact_id, verify=True)
        )
        if type(decoder) is not AutoencoderKL or any(
            p.dtype != torch.float32 for p in decoder.parameters()
        ):
            raise ValueError(
                "Drifting latent decoding requires a separately pinned FP32 native KL-VAE"
            )
        decoder = decoder.eval().to(device)
        parents += (decoder_artifact_id,)
        binding["decoder"] = {
            "artifact_id": decoder_artifact_id,
            "model_relative_path": decoder_path,
        }
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    assigned = [
        (index, case) for index, case in enumerate(plan.cases) if index % world_size == rank
    ]
    samples, inputs = [], []
    started = time.monotonic()
    for index, case in assigned:
        input_record = {
            "id": case.id,
            "seed": case.seed,
            "label": case.condition,
            "continuous_noise_sha256": None,
            "noise_labels": None,
        }
        try:
            if (
                case.condition
                >= binding["generation_contract"]["training"]["settings"]["num_classes"]
            ):
                raise ValueError("Class label is outside the declared trained classes")
            generator = torch.Generator(device=device).manual_seed(case.seed)
            noise = (
                torch.randn(
                    (1, *plan.noise_shape), generator=generator, device=device, dtype=torch.float32
                )
                * plan.temperature
            )
            input_record["continuous_noise_sha256"] = hashlib.sha256(
                noise.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            labels = torch.tensor([case.condition], device=device, dtype=torch.int64)
            condition = labels
            if model.config.noise_classes:
                discrete = torch.randint(
                    model.config.noise_classes,
                    (1, model.config.noise_coords),
                    generator=generator,
                    device=device,
                )
                condition = {"labels": labels, "noise_labels": discrete}
                input_record["noise_labels"] = discrete.cpu().tolist()[0]
            with torch.no_grad():
                value = model(
                    noise,
                    torch.tensor([plan.cfg_scale], device=device, dtype=torch.float32),
                    condition,
                )
                if value.prediction_type != "x0":
                    raise ValueError("Drifting must emit direct samples")
                output = (
                    value.prediction
                    if decoder is None
                    else decoder.decode(value.prediction, scaled=True)
                )
            pixels = quantize_image(output[0], plan.quantization)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            relative = f"{index:08d}-{digest_json(case.id)[:16]}.png"
            with (root / relative).open("xb") as stream:
                stream.write(buffer.getvalue())
            samples.append(
                MediaSample(
                    case.id,
                    (ImageFile(relative, *_image_identity(root / relative)),),
                    seed=case.seed,
                )
            )
        except Exception as error:
            samples.append(MediaSample(case.id, (), "error", case.seed, type(error).__name__))
        inputs.append(input_record)
    for parent in parents:
        store.get(parent, verify=True)
    manifest = MediaManifest(
        "images",
        "native_drifting_generated",
        plan.id,
        "generation",
        "producer_artifact_terms",
        plan.cohort_id,
        tuple(case.id for _, case in assigned),
        tuple(samples),
        parents,
    )
    manifest.save(root)
    atomic_json(
        root / "shard.json",
        {
            "schema_version": 1,
            "plan": asdict(plan),
            "plan_id": plan.id,
            "rank": rank,
            "world_size": world_size,
            "manifest_id": manifest.id,
            "producer_artifacts": parents,
            "native_producer_sources": _sources(),
            "sampling_binding": binding,
            "sampling_binding_id": digest_json(binding),
            "sample_inputs": inputs,
            "environment": runtime_environment(device),
            "end_to_end_seconds_including_io": time.monotonic() - started,
        },
    )
    return manifest


def _check_inputs(record, manifest, plan):
    values = record.get("sample_inputs")
    if (
        not isinstance(values, list)
        or tuple(value.get("id") for value in values) != manifest.expected_ids
    ):
        raise ValueError(
            "Drifting input evidence must cover every ordered sample including failures"
        )
    cases = {case.id: case for case in plan.cases}
    noise = record["sampling_binding"]["generation_contract"]["noise_semantics"]
    for sample, value in zip(manifest.samples, values):
        if value["seed"] != cases[sample.id].seed or value["label"] != cases[sample.id].condition:
            raise ValueError("Drifting sampled input identity differs from its plan")
        if sample.status != "ok":
            continue
        if (
            not isinstance(value.get("continuous_noise_sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", value["continuous_noise_sha256"]) is None
        ):
            raise ValueError("Drifting successful sample lacks actual continuous noise identity")
        discrete = value.get("noise_labels")
        if noise["noise_classes"]:
            if (
                not isinstance(discrete, list)
                or len(discrete) != noise["noise_coords"]
                or any(type(x) is not int or not 0 <= x < noise["noise_classes"] for x in discrete)
            ):
                raise ValueError("Drifting discrete noise evidence is invalid")
        elif discrete is not None:
            raise ValueError("Unexpected discrete noise evidence")


def merge_drifting_shards(shard_directories, plan, output_directory):
    if not isinstance(plan, DriftingSamplingPlan):
        raise ValueError("Expected DriftingSamplingPlan")
    roots = tuple(Path(path) for path in shard_directories)
    by_id = {}
    for root in roots:
        record, manifest = read_json(root / "shard.json"), MediaManifest.load(root)
        _check_inputs(record, manifest, plan)
        for value in record["sample_inputs"]:
            if value["id"] in by_id:
                raise ValueError("Duplicate Drifting input evidence")
            by_id[value["id"]] = value
    manifest = _merge_image_shards(
        roots, plan, output_directory, dataset_id="native_drifting_generated"
    )
    path = Path(output_directory) / "generation.json"
    record = read_json(path)
    record["sample_inputs"] = [by_id[case.id] for case in plan.cases]
    atomic_json(path, record)
    return manifest


def drifting_generation_record(root, manifest):
    root = Path(root)
    path = root / ("generation.json" if (root / "generation.json").exists() else "shard.json")
    record = read_json(path)
    values = dict(record["plan"])
    values["cases"] = tuple(GenerationCase(**case) for case in values["cases"])
    plan = DriftingSamplingPlan(**values)
    if (
        manifest.dataset_id != "native_drifting_generated"
        or record["plan_id"] != plan.id
        or manifest.revision != plan.id
        or manifest.cohort_id != plan.cohort_id
        or manifest.expected_ids != tuple(case.id for case in plan.cases)
    ):
        raise ValueError("Drifting generation record is not the full fixed cohort")
    binding = record.get("sampling_binding")
    if (
        not isinstance(binding, dict)
        or record.get("sampling_binding_id") != digest_json(binding)
        or not manifest.producer_artifacts
        or binding.get("policy_artifact_id") != manifest.producer_artifacts[0]
        or binding.get("conditioning_semantics") != "guidance_embedding"
    ):
        raise ValueError("Drifting generation binding is missing or inconsistent")
    if (
        binding.get("sampling_mode") != "drifting_direct"
        or binding.get("cfg_scale") != plan.cfg_scale
        or binding.get("temperature") != plan.temperature
    ):
        raise ValueError("Drifting guidance/temperature differs from its actual sampling binding")
    if not record.get("native_producer_sources") or any(
        not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
        for value in record["native_producer_sources"].values()
    ):
        raise ValueError("Drifting producer source identity is missing")
    if any(sample.seed != case.seed for sample, case in zip(manifest.samples, plan.cases)):
        raise ValueError("Drifting seed identity differs from its plan")
    _check_inputs(record, manifest, plan)
    return record
