"""Consistency-model artifacts, stochastic few-step sampling, and image records."""

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
from ..methods.consistency import (
    ConsistencyConfig,
    ConsistencyMethod,
    sample_consistency,
    _fingerprint,
)
from ..models.generative import UNet2D, DiT, AutoencoderKL
from .generation_artifacts import load_native_artifact_model, _role_fingerprint
from .generative import (
    GenerationCase,
    ImageFile,
    MediaManifest,
    MediaSample,
    _image_identity,
    _merge_image_shards,
    quantize_image,
    runtime_environment,
)


def _sha(value):
    return isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) is not None


def _tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class ConsistencySamplingPlan:
    cases: tuple[GenerationCase, ...]
    noise_shape: tuple[int, int, int]
    sigmas: tuple[float, ...]
    clip_denoised: bool = True
    quantization: str = "minus_one_one_stylegan"

    def __post_init__(self):
        for name in ("cases", "noise_shape", "sigmas"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            not self.cases
            or any(not isinstance(c, GenerationCase) for c in self.cases)
            or len({c.id for c in self.cases}) != len(self.cases)
        ):
            raise ValueError("Consistency generation requires a fixed unique case set")
        if len(self.noise_shape) != 3 or any(type(n) is not int or n < 1 for n in self.noise_shape):
            raise ValueError("Consistency generation requires positive CHW noise dimensions")
        if (
            not self.sigmas
            or any(
                type(s) not in {int, float} or not math.isfinite(s) or s <= 0 for s in self.sigmas
            )
            or any(a <= b for a, b in zip(self.sigmas, self.sigmas[1:]))
        ):
            raise ValueError(
                "Actual model-evaluation sigmas must be positive, finite and strictly decreasing"
            )
        if type(self.clip_denoised) is not bool or self.quantization not in {
            "minus_one_one_stylegan",
            "zero_one_round",
        }:
            raise ValueError("Clipping and PNG quantization must be explicitly declared")

    @property
    def sampler(self):
        return "consistency"

    @property
    def id(self):
        return digest_json(asdict(self))

    @property
    def cohort_id(self):
        return digest_json(
            {"cases": [asdict(case) for case in self.cases], "quantization": self.quantization}
        )


def _preconditioning(config):
    return {
        "sigma_min": config.sigma_min,
        "sigma_data": config.sigma_data,
        "time_scale": config.time_scale,
        "time_semantics": "student_time_scale_times_log_sigma",
        "input_noise": "unit_fp32_standard_normal",
        "multistep": "denoise_then_independent_noise_std_sqrt_next_sigma_squared_minus_sigma_min_squared",
    }


def publish_consistency_generator(
    method, store, directory, *, sampling_role=None, teacher_artifact_id=None, parents=()
):

    from ..models import build_model
    from ..training.sharding import Zero3Unit
    from ..training.recipes import collective_local, agree, leader_call

    if type(method) is not ConsistencyMethod:
        raise ValueError("Expected the native ConsistencyMethod lifecycle")
    engine = method.engine
    context = engine.parallel
    parents = tuple(parents)

    def preflight():
        if engine._busy or engine._failed or engine.states.get("consistency_method") is not method:
            raise ValueError(
                "Publishing requires the registered successful idle consistency boundary"
            )
        if any(getattr(context, name).size != 1 for name in ("tp", "pp", "cp", "gtp_remat")) or any(
            getattr(context.config, key, 1) != 1
            for key in ("expert_parallel", "expert_tensor_parallel")
        ):
            raise ValueError("Consistency deployment currently supports DP/ZeRO only, not EP/ETP")
        state = method.state_dict()
        declaration = method.export_config()
        if declaration["completed_updates"] < 1:
            raise ValueError("Publish only a genuinely updated consistency model")
        role_name = declaration["sampling_role"] if sampling_role is None else sampling_role
        if role_name not in {"model", "consistency_ema"} or role_name not in engine.roles:
            raise ValueError(
                "Only the student model or an existing sampling EMA can be a generator, never target/teacher"
            )
        owned_model = engine.model if role_name == "model" else method.sampling_model
        if engine.roles[role_name].model is not owned_model:
            raise ValueError("Selected generator is not the method/Trainer-owned role")
        module = engine.roles[role_name].model
        module = module.module if isinstance(module, Zero3Unit) else module
        if (
            type(module) not in {UNet2D, DiT}
            or module.config.prediction_type != "consistency_residual"
        ):
            raise ValueError(
                "Only the native UNet2D/DiT consistency-residual deployment path is implemented"
            )
        if any(p.dtype != torch.float32 for p in module.parameters()):
            raise ValueError(
                "Consistency deployment requires FP32 parameter storage, not a silent dtype conversion"
            )
        teacher_binding = None
        if teacher_artifact_id is not None:
            if method.config.mode != "cd":
                raise ValueError("Only CD has an EDM teacher artifact")
            source = store.get(teacher_artifact_id, verify=True)

            with torch.random.fork_rng(devices=[]):
                teacher, layout = load_native_artifact_model(source)
            if (
                type(teacher) is not type(method.teacher)
                or teacher.config.to_dict() != method.teacher.config.to_dict()
                or _fingerprint(teacher) != declaration["teacher_sha256"]
            ):
                raise ValueError(
                    "Declared teacher artifact differs from the actual frozen CD teacher"
                )
            teacher_binding = {
                "artifact_id": source.id,
                "model_relative_path": layout,
                "teacher_weight_sha256": declaration["teacher_sha256"],
            }
        return (
            module.config,
            role_name,
            declaration,
            teacher_binding,
            {
                "method_updates": state["updates"],
                "initial_role_updates": state["initial_role_updates"],
                "generator_updates": engine.roles["model"].updates,
                "trainer_precision": engine.precision,
                "parallel": context.to_dict(),
                "deployment_only": True,
            },
        )

    config, role, declaration, teacher_binding, training = collective_local(
        context, preflight, "Validate consistency deployment"
    )
    lineage = tuple(
        dict.fromkeys(parents + (() if teacher_artifact_id is None else (teacher_artifact_id,)))
    )
    agree(
        context,
        {
            "directory": str(Path(directory).absolute()),
            "role": role,
            "declaration": declaration,
            "model": config.to_dict(),
            "training": training,
            "teacher": teacher_binding,
            "parents": lineage,
        },
        "Consistency deployment identity",
    )
    tensors = engine.export_state_dict(role=role)

    def publish():
        for parent in lineage:
            store.get(parent, verify=True)
        target = Path(directory).absolute()
        target.mkdir(parents=True, exist_ok=False)
        with torch.random.fork_rng(devices=[]):
            model = build_model(config)
        model.load_state_dict(tensors, strict=True, assign=True)
        model.save_pretrained(target / "model")
        contract = {
            "schema_version": 1,
            "method": "consistency",
            "mode": method.config.mode,
            "prediction_type": "consistency_residual",
            "model": config.to_dict(),
            "method_declaration": declaration,
            "sampling_role": role,
            "role_weight_fingerprint": _role_fingerprint(config.to_dict(), tensors),
            "teacher_artifact": teacher_binding,
            "training": training,
            "preconditioning": _preconditioning(method.config),
        }
        atomic_json(target / "consistency.json", declaration)
        atomic_json(target / "generation_contract.json", contract)
        return store.publish(
            target,
            kind="native_consistency_generator",
            metadata={
                "method": "consistency",
                "mode": method.config.mode,
                "sampling_role": role,
                "generation_contract_id": digest_json(contract),
                "training_checkpoint_included": False,
            },
            parents=lineage,
        ).id

    identity = leader_call(context, publish, "Publish consistency generator")
    return collective_local(
        context, lambda: store.get(identity, verify=True), "Verify consistency artifact"
    )


def _geometry(config, shape):
    divisor = (
        2 ** (len(config.channel_mult) - 1)
        if hasattr(config, "channel_mult")
        else config.patch_size
    )
    if (
        shape[0] != config.in_channels
        or config.out_channels not in {None, config.in_channels}
        or any(n % divisor for n in shape[-2:])
    ):
        raise ValueError(
            "Consistency residual/noise geometry must match channels and downsampling/patch divisor"
        )


def consistency_condition(model, case, device):

    config, value = model.config, case.condition
    if value is None:
        return None
    if config.condition_dim:
        if not isinstance(value, tuple) or len(value) != config.condition_dim:
            raise ValueError("Consistency condition vector differs from the trained model width")
        return torch.tensor([value], device=device, dtype=torch.float32)
    if config.num_classes:
        if type(value) is not int or not 0 <= value < config.num_classes:
            raise ValueError(
                "Consistency label is outside genuine trained classes; use None for null"
            )
        return torch.tensor([value], device=device, dtype=torch.int64)
    raise ValueError("Unconditional consistency model does not accept a condition")


def consistency_binding(artifact, model, layout, plan):
    if (
        not isinstance(plan, ConsistencySamplingPlan)
        or type(model) not in {UNet2D, DiT}
        or model.config.prediction_type != "consistency_residual"
    ):
        raise ValueError("Consistency requires its typed plan and native residual model")
    path = artifact.path / "generation_contract.json"
    contract = read_json(path)
    fields = {
        "schema_version",
        "method",
        "mode",
        "prediction_type",
        "model",
        "method_declaration",
        "sampling_role",
        "role_weight_fingerprint",
        "teacher_artifact",
        "training",
        "preconditioning",
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != fields
        or contract["schema_version"] != 1
        or contract["method"] != "consistency"
        or contract["prediction_type"] != "consistency_residual"
        or digest_json(contract["model"]) != digest_json(model.config.to_dict())
    ):
        raise ValueError("Invalid artifact-bound consistency generator contract")
    declaration = read_json(artifact.path / "consistency.json")
    if (
        declaration != contract["method_declaration"]
        or set(declaration)
        != {
            "schema_version",
            "method",
            "training",
            "completed_updates",
            "teacher_sha256",
            "sampling_role",
        }
        or declaration["schema_version"] != 1
        or declaration["method"] != "consistency"
    ):
        raise ValueError("Consistency lifecycle declaration differs from the generation contract")
    config = ConsistencyConfig(**declaration["training"])
    if (
        config.to_dict() != declaration["training"]
        or contract["mode"] != config.mode
        or contract["preconditioning"] != _preconditioning(config)
    ):
        raise ValueError(
            "Consistency mode/preconditioning/time units differ from the trained method"
        )
    if declaration["sampling_role"] != (
        "model" if config.sampling_ema is None else "consistency_ema"
    ):
        raise ValueError(
            "Declared default sampling role differs from the training EMA configuration"
        )
    selected = contract["sampling_role"]
    if (
        selected not in {"model", "consistency_ema"}
        or selected == "consistency_ema"
        and config.sampling_ema is None
    ):
        raise ValueError("Teacher/target/nonexistent EMA cannot be a consistency generator")
    if (
        type(declaration["completed_updates"]) is not int
        or not 1 <= declaration["completed_updates"] <= config.total_steps
    ):
        raise ValueError("Consistency export lacks completed training evidence")
    if (config.mode == "cd" and not _sha(declaration["teacher_sha256"])) or (
        config.mode != "cd" and declaration["teacher_sha256"] is not None
    ):
        raise ValueError("Only CD declares an actual frozen EDM teacher")
    teacher = contract["teacher_artifact"]
    if teacher is not None and (
        config.mode != "cd"
        or not isinstance(teacher, dict)
        or set(teacher) != {"artifact_id", "model_relative_path", "teacher_weight_sha256"}
        or teacher["artifact_id"] not in artifact.parents
        or teacher["teacher_weight_sha256"] != declaration["teacher_sha256"]
    ):
        raise ValueError("CD teacher artifact identity/lineage differs from the verified teacher")
    training = contract["training"]
    if (
        not isinstance(training, dict)
        or type(training.get("initial_role_updates")) is not int
        or training["initial_role_updates"] < 0
        or any(
            type(training.get(key)) is not int or training[key] < 1
            for key in ("method_updates", "generator_updates")
        )
        or training["method_updates"] != declaration["completed_updates"]
        or training["generator_updates"]
        != training["initial_role_updates"] + declaration["completed_updates"]
    ):
        raise ValueError("Consistency student/method completed-update clocks differ")
    if any(p.dtype != torch.float32 for p in model.parameters()) or contract[
        "role_weight_fingerprint"
    ] != _role_fingerprint(model.config.to_dict(), model.state_dict()):
        raise ValueError("Consistency deployed weights changed or use unsupported stored dtype")
    if (artifact.path / "objective.json").exists():
        raise ValueError("Consistency cannot also declare ordinary EDM/VP/flow objective.json")
    _geometry(model.config, plan.noise_shape)
    if plan.sigmas[0] != config.sigma_max or plan.sigmas[-1] < config.sigma_min:
        raise ValueError(
            "Consistency generation must begin at trained sigma_max and stay >=sigma_min"
        )
    return {
        "schema_version": 1,
        "policy_artifact_id": artifact.id,
        "model_relative_path": layout,
        "sampling_mode": "consistency",
        "mode": config.mode,
        "generation_contract": contract,
        "generation_contract_sha256": file_digest(path),
        "consistency_declaration_sha256": file_digest(artifact.path / "consistency.json"),
        "preconditioning": _preconditioning(config),
        "sampling_role": selected,
        "training_semantics_bound": True,
        "actual_evaluation_sigmas": list(plan.sigmas),
        "clip_denoised": plan.clip_denoised,
        "forward_calls_per_successful_sample": len(plan.sigmas),
    }


def sample_consistency_plan(model, plan, noise, condition, generator, binding):
    controls = binding["preconditioning"]
    return sample_consistency(
        model,
        noise,
        plan.sigmas,
        condition=condition,
        generator=generator,
        sigma_min=controls["sigma_min"],
        sigma_data=controls["sigma_data"],
        time_scale=controls["time_scale"],
        clip_denoised=plan.clip_denoised,
    )


def _sources():
    base = Path(__file__).resolve().parents[1]
    names = (
        "evaluation/consistency_generation.py",
        "evaluation/generative.py",
        "evaluation/generation_artifacts.py",
        "methods/consistency.py",
        "models/generative.py",
        "models/serialization.py",
        "models/__init__.py",
    )
    return {name: file_digest(base / name) for name in names}


def generate_consistency_shard(
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

    if (
        not isinstance(plan, ConsistencySamplingPlan)
        or type(rank) is not int
        or type(world_size) is not int
        or not 0 <= rank < world_size
    ):
        raise ValueError("Typed consistency plan and valid shard required")
    artifact = store.get(policy_artifact_id, verify=True)
    with torch.random.fork_rng(devices=[]):
        model, layout = load_native_artifact_model(artifact)
    binding = consistency_binding(artifact, model, layout, plan)
    model = model.eval().to(device)
    decoder = None
    parents = (policy_artifact_id,)
    binding["decoder"] = None
    if decoder_artifact_id is not None:
        with torch.random.fork_rng(devices=[]):
            decoder, decoder_path = load_native_artifact_model(
                store.get(decoder_artifact_id, verify=True)
            )
        if type(decoder) is not AutoencoderKL or any(
            p.dtype != torch.float32 for p in decoder.parameters()
        ):
            raise ValueError("Consistency latent decoding requires a pinned FP32 native KL-VAE")
        decoder = decoder.eval().to(device)
        parents += (decoder_artifact_id,)
        binding["decoder"] = {
            "artifact_id": decoder_artifact_id,
            "model_relative_path": decoder_path,
        }
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    assigned = [(i, case) for i, case in enumerate(plan.cases) if i % world_size == rank]
    counter, samples, inputs = [0], [], []
    started = time.monotonic()
    source_versions = _sources()
    hook = model.register_forward_pre_hook(lambda *_: counter.__setitem__(0, counter[0] + 1))
    try:
        with torch.no_grad():
            for index, case in assigned:
                evidence = {
                    "id": case.id,
                    "seed": case.seed,
                    "condition_id": digest_json(case.condition),
                    "initial_noise_sha256": None,
                    "rng_before_sampling_sha256": None,
                    "rng_after_sampling_sha256": None,
                    "nfe": 0,
                }
                counter[0] = 0
                try:
                    condition = consistency_condition(model, case, device)
                    generator = torch.Generator(device=device).manual_seed(case.seed)
                    noise = torch.randn(
                        (1, *plan.noise_shape),
                        generator=generator,
                        device=device,
                        dtype=torch.float32,
                    )
                    evidence["initial_noise_sha256"] = _tensor_hash(noise)
                    evidence["rng_before_sampling_sha256"] = _tensor_hash(generator.get_state())
                    output = sample_consistency_plan(
                        model, plan, noise, condition, generator, binding
                    )
                    evidence["rng_after_sampling_sha256"] = _tensor_hash(generator.get_state())
                    evidence["nfe"] = counter[0]
                    if counter[0] != len(plan.sigmas):
                        raise ValueError("Observed consistency NFE differs from actual sigma calls")
                    if decoder is not None:
                        output = decoder.decode(output, scaled=True)
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
                    evidence["nfe"] = counter[0]
                    samples.append(
                        MediaSample(case.id, (), "error", case.seed, type(error).__name__)
                    )
                inputs.append(evidence)
    finally:
        hook.remove()
    for parent in parents:
        store.get(parent, verify=True)
    if source_versions != _sources():
        raise RuntimeError("Native consistency source changed during sampling")
    manifest = MediaManifest(
        "images",
        "native_consistency_generated",
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
            "native_producer_sources": source_versions,
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
    cases = {case.id: case for case in plan.cases}
    if (
        not isinstance(values, list)
        or tuple(value.get("id") for value in values) != manifest.expected_ids
    ):
        raise ValueError("Consistency input evidence must include the full ordered outcome set")
    for sample, value in zip(manifest.samples, values):
        case = cases[sample.id]
        if (
            value["seed"] != case.seed
            or value["condition_id"] != digest_json(case.condition)
            or type(value.get("nfe")) is not int
            or value["nfe"] < 0
        ):
            raise ValueError("Consistency sampled condition/seed/NFE identity differs")
        if sample.status == "ok" and (
            value["nfe"] != len(plan.sigmas)
            or any(
                not _sha(value.get(name))
                for name in (
                    "initial_noise_sha256",
                    "rng_before_sampling_sha256",
                    "rng_after_sampling_sha256",
                )
            )
        ):
            raise ValueError("Consistency successful sample lacks actual noise/RNG/NFE evidence")


def merge_consistency_shards(shard_directories, plan, output_directory):
    if not isinstance(plan, ConsistencySamplingPlan):
        raise ValueError("Expected ConsistencySamplingPlan")
    roots = tuple(Path(path) for path in shard_directories)
    inputs = {}
    for root in roots:
        record, manifest = read_json(root / "shard.json"), MediaManifest.load(root)
        _check_inputs(record, manifest, plan)
        for value in record["sample_inputs"]:
            if value["id"] in inputs:
                raise ValueError("Duplicate consistency input evidence")
            inputs[value["id"]] = value
    manifest = _merge_image_shards(
        roots, plan, output_directory, dataset_id="native_consistency_generated"
    )
    path = Path(output_directory) / "generation.json"
    record = read_json(path)
    record["sample_inputs"] = [inputs[case.id] for case in plan.cases]
    atomic_json(path, record)
    return manifest


def consistency_generation_record(root, manifest):
    root = Path(root)
    record = read_json(
        root / ("generation.json" if (root / "generation.json").exists() else "shard.json")
    )
    values = dict(record["plan"])
    values["cases"] = tuple(GenerationCase(**case) for case in values["cases"])
    plan = ConsistencySamplingPlan(**values)
    if (
        manifest.dataset_id != "native_consistency_generated"
        or record["plan_id"] != plan.id
        or manifest.revision != plan.id
        or manifest.cohort_id != plan.cohort_id
        or manifest.expected_ids != tuple(c.id for c in plan.cases)
    ):
        raise ValueError("Consistency generation is not the full fixed cohort")
    binding = record.get("sampling_binding")
    if (
        not isinstance(binding, dict)
        or record.get("sampling_binding_id") != digest_json(binding)
        or not manifest.producer_artifacts
        or binding.get("policy_artifact_id") != manifest.producer_artifacts[0]
    ):
        raise ValueError("Consistency generation lacks the actual artifact binding")
    if (
        binding.get("sampling_mode") != "consistency"
        or binding.get("actual_evaluation_sigmas") != list(plan.sigmas)
        or binding.get("clip_denoised") != plan.clip_denoised
        or binding.get("forward_calls_per_successful_sample") != len(plan.sigmas)
    ):
        raise ValueError("Consistency sampler controls/NFE differ from the plan")
    if not record.get("native_producer_sources") or any(
        not _sha(value) for value in record["native_producer_sources"].values()
    ):
        raise ValueError("Consistency producer source identity is missing")
    if any(sample.seed != case.seed for sample, case in zip(manifest.samples, plan.cases)):
        raise ValueError("Consistency sample seed differs")
    _check_inputs(record, manifest, plan)
    return record
