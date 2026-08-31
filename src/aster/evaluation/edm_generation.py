"""EDM teacher artifacts and Heun baselines under the shared image-evaluation protocol."""

from dataclasses import asdict, dataclass
import io
import math
from pathlib import Path
import re
import time

import torch
from PIL import Image

from ..core import atomic_json, digest_json, file_digest, read_json
from ..methods.generation import EDMObjective, sample_edm
from ..models.generative import UNet2D, DiT, AutoencoderKL
from .generation_artifacts import (
    load_native_artifact_model,
    _role_fingerprint,
    verified_training_update,
    validate_successful_update_record,
)
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


@dataclass(frozen=True)
class EDMSamplingPlan:
    cases: tuple[GenerationCase, ...]
    noise_shape: tuple[int, int, int]
    sigmas: tuple[float, ...]
    churn: float = 0.0
    churn_min: float = 0.0
    churn_max: float | None = None
    noise_scale: float = 1.0
    quantization: str = "minus_one_one_stylegan"

    def __post_init__(self):
        for name in ("cases", "noise_shape", "sigmas"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            not self.cases
            or any(not isinstance(case, GenerationCase) for case in self.cases)
            or len({case.id for case in self.cases}) != len(self.cases)
        ):
            raise ValueError("EDM requires a complete fixed unique case set")
        if len(self.noise_shape) != 3 or any(type(n) is not int or n < 1 for n in self.noise_shape):
            raise ValueError("EDM requires positive CHW noise dimensions")
        if (
            len(self.sigmas) < 3
            or any(
                type(s) not in {float, int} or not math.isfinite(s) or s < 0 for s in self.sigmas
            )
            or self.sigmas[-1] != 0
            or any(a <= b for a, b in zip(self.sigmas, self.sigmas[1:]))
        ):
            raise ValueError(
                "EDM needs at least two positive sigma levels followed by terminal zero, strictly decreasing"
            )
        if any(
            type(v) not in {float, int} or not math.isfinite(v) or v < 0
            for v in (self.churn, self.churn_min, self.noise_scale)
        ):
            raise ValueError("EDM churn/noise controls must be finite and nonnegative")
        if self.churn_max is not None and (
            type(self.churn_max) not in {float, int}
            or not math.isfinite(self.churn_max)
            or self.churn_max < self.churn_min
        ):
            raise ValueError("Use None for an unbounded churn upper limit, never JSON infinity")
        if self.quantization not in {"minus_one_one_stylegan", "zero_one_round"}:
            raise ValueError("Explicit PNG quantization required")

    @property
    def sampler(self):
        return "edm_heun"

    @property
    def id(self):
        return digest_json(asdict(self))

    @property
    def cohort_id(self):
        return digest_json(
            {"cases": [asdict(case) for case in self.cases], "quantization": self.quantization}
        )


def edm_nfe(plan):
    return 2 * (len(plan.sigmas) - 1) - 1


def _objective_controls(objective):
    if not isinstance(objective, dict):
        raise ValueError("EDM needs an actual serialized training objective")
    decoder = None
    inner = objective
    if objective.get("type") == "latent_field":
        decoder = objective.get("encoder_identity")
        inner = objective.get("objective")
        if not isinstance(decoder, str) or not re.fullmatch("[a-f0-9]{64}", decoder):
            raise ValueError("EDM latent training requires an immutable encoder artifact identity")
    if (
        not isinstance(inner, dict)
        or set(inner) != {"type", "sigma_data", "log_mean", "log_std"}
        or inner["type"] != "edm"
    ):
        raise ValueError("EDM Heun cannot reinterpret a VP/flow/consistency training objective")
    if (
        any(
            type(inner[name]) not in {float, int} or not math.isfinite(inner[name])
            for name in ("sigma_data", "log_mean", "log_std")
        )
        or min(inner["sigma_data"], inner["log_std"]) <= 0
    ):
        raise ValueError("Invalid EDM training noise/preconditioning controls")
    return inner, decoder


def _preconditioning(sigma_data):
    return {
        "sigma_data": sigma_data,
        "time_scale": 0.25,
        "time_semantics": "quarter_log_sigma",
        "input_noise": "unit_fp32_standard_normal_times_first_sigma",
        "state_dtype": "torch.float32",
        "sigma_schedule_dtype": "torch.float64",
        "sigma_rounding": "none_continuous_native_model",
        "solver": "heun_positive_endpoints_terminal_euler",
        "churn_rng": "one_noise_draw_per_interval_including_zero_churn",
    }


def publish_edm_generator(engine, store, directory, *, ema=False, parents=()):

    from ..models import build_model
    from ..pipelines import LatentFieldObjective
    from ..training.sharding import Zero3Unit
    from ..training.recipes import collective_local, agree, leader_call

    context = engine.parallel
    parents = tuple(parents)

    def preflight():
        if (
            type(ema) is not bool
            or engine._busy
            or engine._failed
            or engine.roles["model"].updates < 1
        ):
            raise ValueError("EDM export requires a genuinely updated successful idle Trainer")
        if any(getattr(context, key).size != 1 for key in ("tp", "pp", "cp", "gtp_remat")) or any(
            getattr(context.config, key, 1) != 1
            for key in ("expert_parallel", "expert_tensor_parallel")
        ):
            raise ValueError("EDM deployment supports native DP/ZeRO, not TP/PP/CP/EP/ETP/GTP")
        module = engine.roles["model"].model
        module = module.module if isinstance(module, Zero3Unit) else module
        if (
            type(module) not in {UNet2D, DiT}
            or module.config.prediction_type != "edm_residual"
            or any(p.dtype != torch.float32 for p in module.parameters())
        ):
            raise ValueError("EDM exporter requires FP32 stored native UNet2D/DiT residual weights")
        objective = engine.objective
        inner = objective.objective if type(objective) is LatentFieldObjective else objective
        if type(inner) is not EDMObjective:
            raise ValueError("EDM exporter requires the actual native EDMObjective")
        declaration = objective.config_dict()
        _, decoder = _objective_controls(declaration)
        if ema and engine.roles["model"].ema is None:
            raise ValueError("Requested EDM sampling EMA does not exist")
        if decoder is not None:
            with torch.random.fork_rng(devices=[]):
                actual_decoder, _ = load_native_artifact_model(store.get(decoder, verify=True))
            if type(actual_decoder) is not AutoencoderKL or _role_fingerprint(
                actual_decoder.config.to_dict(), actual_decoder.state_dict()
            ) != _role_fingerprint(
                objective.autoencoder.config.to_dict(), objective.autoencoder.state_dict()
            ):
                raise ValueError(
                    "Declared EDM latent encoder artifact differs from the actual frozen training encoder"
                )
        actual_update = verified_training_update(engine, objective)
        return (
            module.config,
            declaration,
            decoder,
            {
                "updates": engine.roles["model"].updates,
                "successful_update": actual_update,
                "precision": engine.precision,
                "parallel": context.to_dict(),
                "deployment_only": True,
            },
        )

    config, objective, decoder_id, training = collective_local(
        context, preflight, "Validate EDM deployment"
    )
    lineage = tuple(dict.fromkeys(parents + (() if decoder_id is None else (decoder_id,))))
    agree(
        context,
        {
            "directory": str(Path(directory).absolute()),
            "model": config.to_dict(),
            "objective": objective,
            "ema": ema,
            "training": training,
            "parents": lineage,
        },
        "EDM deployment identity",
    )
    tensors = engine.export_state_dict(ema=ema)

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
            "method": "edm",
            "sampling_role": "ema" if ema else "model",
            "model": config.to_dict(),
            "training_objective": objective,
            "training": training,
            "role_weight_fingerprint": _role_fingerprint(config.to_dict(), tensors),
        }
        atomic_json(target / "objective.json", objective)
        atomic_json(target / "generation_contract.json", contract)
        atomic_json(target / "successful_update.json", training["successful_update"])
        return store.publish(
            target,
            kind="native_edm_generator",
            metadata={
                "method": "edm",
                "generation_contract_id": digest_json(contract),
                "training_checkpoint_included": False,
            },
            parents=lineage,
        ).id

    identity = leader_call(context, publish, "Publish EDM generator")
    return collective_local(
        context, lambda: store.get(identity, verify=True), "Verify EDM deployment"
    )


def edm_binding(artifact, model, layout, plan):
    if (
        not isinstance(plan, EDMSamplingPlan)
        or type(model) not in {UNet2D, DiT}
        or model.config.prediction_type != "edm_residual"
    ):
        raise ValueError("EDM sampling requires its typed Heun plan and native EDM residual model")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("EDM reference production requires FP32 parameter storage")
    config = model.config
    divisor = (
        2 ** (len(config.channel_mult) - 1)
        if hasattr(config, "channel_mult")
        else config.patch_size
    )
    if (
        plan.noise_shape[0] != config.in_channels
        or (config.out_channels or config.in_channels) != config.in_channels
        or any(n % divisor for n in plan.noise_shape[-2:])
    ):
        raise ValueError("EDM residual/noise geometry differs from the trained model")
    path = artifact.path / "objective.json"
    objective = read_json(path)
    inner, decoder_id = _objective_controls(objective)
    descriptor = {
        "class": "aster.pipelines.LatentFieldObjective"
        if objective["type"] == "latent_field"
        else "aster.methods.generation.EDMObjective",
        "codec": "config_dict",
        "configuration": objective,
    }
    update_path = artifact.path / "successful_update.json"
    actual_update = read_json(update_path) if update_path.is_file() else None
    if update_path.is_file():
        validate_successful_update_record(
            actual_update,
            descriptor,
            role_updates=actual_update.get("role_updates")
            if isinstance(actual_update, dict)
            else None,
        )
    contract_path = artifact.path / "generation_contract.json"
    contract = read_json(contract_path) if contract_path.is_file() else None
    if contract is not None:
        required = {
            "schema_version",
            "method",
            "sampling_role",
            "model",
            "training_objective",
            "training",
            "role_weight_fingerprint",
        }
        if (
            not isinstance(contract, dict)
            or set(contract) != required
            or contract["schema_version"] != 1
            or contract["method"] != "edm"
            or contract["sampling_role"] not in {"model", "ema"}
            or digest_json(contract["model"]) != digest_json(config.to_dict())
            or contract["training_objective"] != objective
        ):
            raise ValueError(
                "EDM generation contract differs from the actual model/training objective"
            )
        if (
            not isinstance(contract["training"], dict)
            or type(contract["training"].get("updates")) is not int
            or contract["training"]["updates"] < 1
        ):
            raise ValueError("EDM export lacks completed-update evidence")
        validate_successful_update_record(
            contract["training"].get("successful_update"),
            descriptor,
            role_updates=contract["training"]["updates"],
        )
        if actual_update != contract["training"]["successful_update"]:
            raise ValueError(
                "EDM actual successful objective record differs from its generation contract"
            )
        if contract["role_weight_fingerprint"] != _role_fingerprint(
            config.to_dict(), model.state_dict()
        ):
            raise ValueError("EDM weights differ from the selected role snapshot")
    return {
        "schema_version": 1,
        "policy_artifact_id": artifact.id,
        "model_relative_path": layout,
        "sampling_mode": "edm_heun",
        "training_semantics_bound": True,
        "training_objective": objective,
        "objective_file_sha256": file_digest(path),
        "generation_contract": contract,
        "generation_contract_sha256": file_digest(contract_path) if contract is not None else None,
        "training_evidence": "successful_trainer_export"
        if contract is not None
        else "artifact_serialized_objective_not_training_history_proof",
        "successful_update": actual_update,
        "successful_update_file_sha256": file_digest(update_path)
        if actual_update is not None
        else None,
        "actual_successful_objective_bound": actual_update is not None,
        "preconditioning": _preconditioning(inner["sigma_data"]),
        "required_decoder_artifact_id": decoder_id,
        "actual_sigmas_including_terminal_zero": list(plan.sigmas),
        "churn": plan.churn,
        "churn_min": plan.churn_min,
        "churn_max": plan.churn_max,
        "noise_scale": plan.noise_scale,
        "forward_calls_per_successful_sample": edm_nfe(plan),
    }


def edm_condition(model, case, device):
    value, config = case.condition, model.config
    if value is None:
        return None
    if config.condition_dim:
        if not isinstance(value, tuple) or len(value) != config.condition_dim:
            raise ValueError("EDM condition vector differs from model width")
        return torch.tensor([value], device=device, dtype=torch.float32)
    if config.num_classes:
        if type(value) is not int or not 0 <= value < config.num_classes:
            raise ValueError("EDM label is outside genuine trained classes")
        return torch.tensor([value], device=device, dtype=torch.int64)
    raise ValueError("Unconditional EDM model cannot consume an invented condition")


def sample_edm_plan(model, plan, noise, condition, generator, binding):
    return sample_edm(
        model,
        noise,
        torch.tensor(plan.sigmas, dtype=torch.float64, device=noise.device),
        condition=condition,
        sigma_data=binding["preconditioning"]["sigma_data"],
        churn=plan.churn,
        churn_min=plan.churn_min,
        churn_max=float("inf") if plan.churn_max is None else plan.churn_max,
        noise_scale=plan.noise_scale,
        generator=generator,
    )


def _tensor_hash(tensor):
    import hashlib

    return hashlib.sha256(
        tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _sources():
    root = Path(__file__).resolve().parents[1]
    names = (
        "evaluation/edm_generation.py",
        "evaluation/generative.py",
        "evaluation/generation_artifacts.py",
        "methods/generation.py",
        "models/generative.py",
        "models/serialization.py",
        "models/config.py",
        "models/__init__.py",
        "core/update_provenance.py",
    )
    return {name: file_digest(root / name) for name in names}


def generate_edm_shard(
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
        not isinstance(plan, EDMSamplingPlan)
        or type(rank) is not int
        or type(world_size) is not int
        or not 0 <= rank < world_size
    ):
        raise ValueError("Typed EDM plan and valid shard required")
    artifact = store.get(policy_artifact_id, verify=True)
    with torch.random.fork_rng(devices=[]):
        model, layout = load_native_artifact_model(artifact)
    binding = edm_binding(artifact, model, layout, plan)
    if (
        binding["required_decoder_artifact_id"] is not None
        and decoder_artifact_id != binding["required_decoder_artifact_id"]
    ):
        raise ValueError("EDM decoder differs from the encoder actually bound by latent training")
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
            raise ValueError("EDM decoding requires a pinned FP32 native KL-VAE")
        decoder = decoder.eval().to(device)
        parents += (decoder_artifact_id,)
        binding["decoder"] = {
            "artifact_id": decoder_artifact_id,
            "model_relative_path": decoder_path,
        }
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    assigned = [(i, case) for i, case in enumerate(plan.cases) if i % world_size == rank]
    samples, inputs, counter = [], [], [0]
    versions = _sources()
    start = time.monotonic()
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
                    condition = edm_condition(model, case, device)
                    generator = torch.Generator(device=device).manual_seed(case.seed)
                    noise = torch.randn(
                        (1, *plan.noise_shape),
                        device=device,
                        dtype=torch.float32,
                        generator=generator,
                    )
                    evidence["initial_noise_sha256"] = _tensor_hash(noise)
                    evidence["rng_before_sampling_sha256"] = _tensor_hash(generator.get_state())
                    output = sample_edm_plan(model, plan, noise, condition, generator, binding)
                    evidence["rng_after_sampling_sha256"] = _tensor_hash(generator.get_state())
                    evidence["nfe"] = counter[0]
                    if counter[0] != edm_nfe(plan):
                        raise ValueError(
                            "Observed EDM forward count differs from actual Heun steps"
                        )
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
    if versions != _sources():
        raise RuntimeError("Native EDM source changed during sampling")
    manifest = MediaManifest(
        "images",
        "native_edm_generated",
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
            "native_producer_sources": versions,
            "sampling_binding": binding,
            "sampling_binding_id": digest_json(binding),
            "sample_inputs": inputs,
            "environment": runtime_environment(device),
            "end_to_end_seconds_including_io": time.monotonic() - start,
        },
    )
    return manifest


def _check_inputs(record, manifest, plan):
    inputs = record.get("sample_inputs")
    cases = {case.id: case for case in plan.cases}
    if (
        not isinstance(inputs, list)
        or tuple(item.get("id") for item in inputs) != manifest.expected_ids
    ):
        raise ValueError("EDM evidence needs the complete ordered outcome set")
    for sample, item in zip(manifest.samples, inputs):
        case = cases[sample.id]
        if (
            item["seed"] != case.seed
            or item["condition_id"] != digest_json(case.condition)
            or type(item["nfe"]) is not int
            or item["nfe"] < 0
        ):
            raise ValueError("EDM seed/condition/NFE changed")
        if sample.status == "ok" and (
            item["nfe"] != edm_nfe(plan)
            or any(
                not isinstance(item.get(key), str) or not re.fullmatch("[a-f0-9]{64}", item[key])
                for key in (
                    "initial_noise_sha256",
                    "rng_before_sampling_sha256",
                    "rng_after_sampling_sha256",
                )
            )
        ):
            raise ValueError("Successful EDM sample lacks actual noise/RNG/NFE evidence")


def merge_edm_shards(shard_directories, plan, output_directory):
    if not isinstance(plan, EDMSamplingPlan):
        raise ValueError("Typed EDM plan required")
    roots = tuple(Path(path) for path in shard_directories)
    inputs = {}
    for root in roots:
        record, manifest = read_json(root / "shard.json"), MediaManifest.load(root)
        _check_inputs(record, manifest, plan)
        for value in record["sample_inputs"]:
            if value["id"] in inputs:
                raise ValueError("Duplicate EDM input evidence")
            inputs[value["id"]] = value
    manifest = _merge_image_shards(roots, plan, output_directory, dataset_id="native_edm_generated")
    path = Path(output_directory) / "generation.json"
    record = read_json(path)
    record["sample_inputs"] = [inputs[case.id] for case in plan.cases]
    atomic_json(path, record)
    return manifest


def edm_generation_record(root, manifest):
    root = Path(root)
    record = read_json(
        root / ("generation.json" if (root / "generation.json").exists() else "shard.json")
    )
    values = dict(record["plan"])
    values["cases"] = tuple(GenerationCase(**case) for case in values["cases"])
    plan = EDMSamplingPlan(**values)
    if (
        manifest.dataset_id != "native_edm_generated"
        or record["plan_id"] != plan.id
        or manifest.revision != plan.id
        or manifest.cohort_id != plan.cohort_id
        or manifest.expected_ids != tuple(case.id for case in plan.cases)
    ):
        raise ValueError("EDM generation is not the entire fixed cohort")
    binding = record.get("sampling_binding")
    if (
        not isinstance(binding, dict)
        or record.get("sampling_binding_id") != digest_json(binding)
        or not manifest.producer_artifacts
        or binding.get("policy_artifact_id") != manifest.producer_artifacts[0]
    ):
        raise ValueError("EDM artifact binding missing")
    if (
        binding.get("sampling_mode") != plan.sampler
        or binding.get("actual_sigmas_including_terminal_zero") != list(plan.sigmas)
        or binding.get("forward_calls_per_successful_sample") != edm_nfe(plan)
        or any(
            binding.get(key) != getattr(plan, key)
            for key in ("churn", "churn_min", "churn_max", "noise_scale")
        )
    ):
        raise ValueError("EDM sampler controls/NFE differ from the declared plan")
    if not record.get("native_producer_sources") or any(
        not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value)
        for value in record["native_producer_sources"].values()
    ):
        raise ValueError("EDM source identity missing")
    if any(sample.seed != case.seed for sample, case in zip(manifest.samples, plan.cases)):
        raise ValueError("EDM seed changed")
    _check_inputs(record, manifest, plan)
    return record


def validate_consistency_teacher_baseline(
    store, teacher_id, student_id, teacher_plan, student_plan
):

    from .consistency_generation import consistency_binding, ConsistencySamplingPlan
    from ..methods.consistency import _fingerprint

    if not isinstance(teacher_plan, EDMSamplingPlan) or not isinstance(
        student_plan, ConsistencySamplingPlan
    ):
        raise ValueError("Teacher and student need distinct typed samplers")
    if (
        teacher_plan.cohort_id != student_plan.cohort_id
        or teacher_plan.noise_shape != student_plan.noise_shape
        or teacher_plan.sigmas[0] != student_plan.sigmas[0]
    ):
        raise ValueError("Teacher/student must share cohort, initial unit-noise geometry and sigma")
    teacher_artifact, student_artifact = (
        store.get(teacher_id, verify=True),
        store.get(student_id, verify=True),
    )
    with torch.random.fork_rng(devices=[]):
        teacher, teacher_path = load_native_artifact_model(teacher_artifact)
        student, student_path = load_native_artifact_model(student_artifact)
    teacher_binding = edm_binding(teacher_artifact, teacher, teacher_path, teacher_plan)
    student_binding = consistency_binding(student_artifact, student, student_path, student_plan)
    contract = student_binding["generation_contract"]
    parent = contract["teacher_artifact"]
    training = contract["method_declaration"]["training"]
    if (
        contract["mode"] != "cd"
        or parent is None
        or parent["artifact_id"] != teacher_id
        or _fingerprint(teacher) != parent["teacher_weight_sha256"]
    ):
        raise ValueError("Baseline is not the artifact-bound actual frozen CD teacher")
    if (
        training["teacher_time_scale"] != 0.25
        or training["sigma_data"] != teacher_binding["preconditioning"]["sigma_data"]
    ):
        raise ValueError("CD teacher preconditioning differs from the EDM baseline runtime")
    return {
        "teacher_artifact_id": teacher_id,
        "student_artifact_id": student_id,
        "cohort_id": teacher_plan.cohort_id,
        "teacher_plan_id": teacher_plan.id,
        "student_plan_id": student_plan.id,
        "teacher_binding_id": digest_json(teacher_binding),
        "student_binding_id": digest_json(student_binding),
        "quality_and_performance_evaluated": False,
    }
