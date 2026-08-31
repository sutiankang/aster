"""Artifact-bound MeanFlow and Shortcut sampling with distinct interval conventions."""

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
from ..models.interval_dit import IntervalDiT
from ..methods.meanflow import MeanFlowObjective, sample_meanflow
from ..methods.shortcut import ShortcutMethod, sample_shortcut
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
    quantize_image,
    runtime_environment,
    _image_identity,
    _merge_image_shards,
)


def _common_plan(plan):
    object.__setattr__(plan, "cases", tuple(plan.cases))
    object.__setattr__(plan, "noise_shape", tuple(plan.noise_shape))
    if not plan.cases or any(
        not isinstance(c, GenerationCase) or type(c.condition) is not int or c.condition < 0
        for c in plan.cases
    ):
        raise ValueError("Interval generation requires explicit nonnegative integer class labels")
    if (
        len({c.id for c in plan.cases}) != len(plan.cases)
        or len(plan.noise_shape) != 3
        or any(type(n) is not int or n < 1 for n in plan.noise_shape)
    ):
        raise ValueError("Unique cases and a positive CHW noise shape are required")
    if plan.quantization not in {"minus_one_one_stylegan", "zero_one_round"}:
        raise ValueError("Unknown image quantization rule")


class _PlanIdentity:
    @property
    def id(self):
        return digest_json(asdict(self))

    @property
    def cohort_id(self):

        return digest_json(
            {"cases": [asdict(c) for c in self.cases], "quantization": self.quantization}
        )


@dataclass(frozen=True)
class MeanFlowSamplingPlan(_PlanIdentity):
    cases: tuple[GenerationCase, ...]
    noise_shape: tuple[int, int, int]
    timesteps: tuple[float, ...] = (1.0, 0.0)
    quantization: str = "minus_one_one_stylegan"

    def __post_init__(self):
        _common_plan(self)
        object.__setattr__(self, "timesteps", tuple(self.timesteps))
        t = self.timesteps
        if (
            len(t) < 2
            or any(type(x) not in {int, float} or not math.isfinite(x) for x in t)
            or t[0] != 1
            or t[-1] != 0
            or any(a <= b for a, b in zip(t, t[1:]))
        ):
            raise ValueError("MeanFlow times must strictly decrease from noise=1 to data=0")

    @property
    def sampler(self):
        return "meanflow"


@dataclass(frozen=True)
class ShortcutSamplingPlan(_PlanIdentity):
    cases: tuple[GenerationCase, ...]
    noise_shape: tuple[int, int, int]
    steps: int = 1
    guidance_scale: float = 1.0
    quantization: str = "minus_one_one_stylegan"

    def __post_init__(self):
        _common_plan(self)
        if type(self.steps) is not int or self.steps < 1 or self.steps & (self.steps - 1):
            raise ValueError("Shortcut sampling steps must be a positive power of two")
        if (
            type(self.guidance_scale) not in {int, float}
            or not math.isfinite(self.guidance_scale)
            or self.guidance_scale < 0
        ):
            raise ValueError("Shortcut inference guidance must be finite and nonnegative")

    @property
    def sampler(self):
        return "shortcut"


def _semantics(variant):
    return (
        {
            "time_direction": "data_to_noise",
            "interval_semantics": "duration",
            "inference_guidance": "training_embedded_no_two_prediction_interpolation",
        }
        if variant == "meanflow"
        else {
            "time_direction": "noise_to_data",
            "interval_semantics": "negative_log2_step",
            "inference_guidance": "zero_null_only_one_conditional_other_two_predictions",
        }
    )


def publish_meanflow_generator(engine, store, directory, *, ema=False, parents=()):

    return _publish(engine, None, store, directory, ema=ema, parents=parents)


def publish_shortcut_generator(method, store, directory, *, ema=False, parents=()):

    if type(method) is not ShortcutMethod:
        raise ValueError("Expected native ShortcutMethod")
    return _publish(method.engine, method, store, directory, ema=ema, parents=parents)


def _publish(engine, method, store, directory, *, ema, parents):
    from ..training.sharding import Zero3Unit
    from ..training.recipes import collective_local, agree, leader_call

    context = engine.parallel
    parents = tuple(parents)

    def preflight():
        if type(ema) is not bool or engine._busy or engine._failed:
            raise ValueError(
                "Only an idle successful Trainer boundary and bool EMA selection can publish"
            )
        if any(getattr(context, name).size != 1 for name in ("tp", "pp", "cp", "gtp_remat")) or any(
            getattr(context.config, key, 1) != 1
            for key in ("expert_parallel", "expert_tensor_parallel")
        ):
            raise ValueError("Interval deployment currently supports native DP/ZeRO only")
        module = engine.roles["model"].model
        module = module.module if isinstance(module, Zero3Unit) else module
        variant = "meanflow" if method is None else "shortcut"
        if (
            type(module) is not IntervalDiT
            or module.config.variant != variant
            or engine.roles["model"].updates < 1
        ):
            raise ValueError("Publish a genuinely updated matching native IntervalDiT")
        if any(p.dtype != torch.float32 for p in module.parameters()):
            raise ValueError("Interval deployment currently requires FP32 parameter storage")
        if ema and engine.roles["model"].ema is None:
            raise ValueError("Requested Trainer EMA does not exist")
        if method is None:
            if type(engine.objective) is not MeanFlowObjective:
                raise ValueError("MeanFlow export requires the actual native MeanFlowObjective")
            objective = engine.objective.config_dict()
            actual_update = verified_training_update(engine, engine.objective)
            method_updates = engine.roles["model"].updates
        else:
            if engine.states.get("shortcut_method") is not method:
                raise ValueError("Shortcut lifecycle is not registered")
            state = method.state_dict()
            if state["updates"] < 1 or state["settings"]["model"] != module.config.to_dict():
                raise ValueError("Shortcut training identity does not match the deployed model")
            objective = state["settings"]
            method_updates = state["updates"]
        training = {
            "objective": objective,
            "method_updates": method_updates,
            "generator_updates": engine.roles["model"].updates,
            "weight_selection": "ema" if ema else "model",
            "ema_decay": engine.roles["model"].ema.decay if ema else None,
            "trainer_precision": engine.precision,
            "parallel": context.to_dict(),
            "deployment_only": True,
        }
        if method is None:
            training["successful_update"] = actual_update
        return module.config, training

    config, training = collective_local(context, preflight, "Validate interval deployment boundary")
    agree(
        context,
        {
            "directory": str(Path(directory).absolute()),
            "parents": parents,
            "model": config.to_dict(),
            "training": training,
        },
        "Interval deployment identity",
    )
    tensors = engine.export_state_dict(ema=ema)

    def publish():
        for parent in parents:
            store.get(parent, verify=True)
        target = Path(directory).absolute()
        target.mkdir(parents=True, exist_ok=False)

        with torch.random.fork_rng(devices=[]):
            model = IntervalDiT(config)
        model.load_state_dict(tensors, strict=True, assign=True)
        model.save_pretrained(target / "model")
        contract = {
            "schema_version": 1,
            "method": config.variant,
            "prediction_type": "average_velocity",
            **_semantics(config.variant),
            "model": config.to_dict(),
            "training": training,
            "model_weight_fingerprint": _role_fingerprint(config.to_dict(), tensors),
        }
        atomic_json(target / "generation_contract.json", contract)
        return store.publish(
            target,
            kind="native_interval_generator",
            metadata={
                "method": config.variant,
                "generation_contract_id": digest_json(contract),
                "training_checkpoint_included": False,
            },
            parents=parents,
        ).id

    identity = leader_call(context, publish, "Publish interval generator")
    return collective_local(
        context, lambda: store.get(identity, verify=True), "Verify interval generator"
    )


def interval_nfe(plan):
    if isinstance(plan, MeanFlowSamplingPlan):
        return len(plan.timesteps) - 1
    if isinstance(plan, ShortcutSamplingPlan):
        return plan.steps * (1 if plan.guidance_scale in (0, 1) else 2)
    raise ValueError("Expected a typed interval sampling plan")


def interval_binding(artifact, model, layout, plan):
    if (
        type(model) is not IntervalDiT
        or not isinstance(plan, (MeanFlowSamplingPlan, ShortcutSamplingPlan))
        or model.config.variant != plan.sampler
    ):
        raise ValueError("Interval sampler and native architecture variant differ")
    contract = read_json(artifact.path / "generation_contract.json")
    fields = {
        "schema_version",
        "method",
        "prediction_type",
        "time_direction",
        "interval_semantics",
        "inference_guidance",
        "model",
        "training",
        "model_weight_fingerprint",
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != fields
        or contract["schema_version"] != 1
        or contract["method"] != plan.sampler
        or contract["prediction_type"] != "average_velocity"
        or contract["model"] != model.config.to_dict()
        or any(contract[key] != value for key, value in _semantics(plan.sampler).items())
    ):
        raise ValueError(
            "Invalid interval generation contract: duration/log2/time direction are not aliases"
        )
    if (artifact.path / "objective.json").exists():
        raise ValueError("Interval deployment cannot also declare VP/ordinary flow objective.json")
    if any(p.dtype != torch.float32 for p in model.parameters()) or contract[
        "model_weight_fingerprint"
    ] != _role_fingerprint(model.config.to_dict(), model.state_dict()):
        raise ValueError("Interval deployed weights differ or use unsupported stored dtype")
    training = contract["training"]
    if (
        not isinstance(training, dict)
        or any(
            type(training.get(key)) is not int or training[key] < 1
            for key in ("method_updates", "generator_updates")
        )
        or training.get("weight_selection") not in {"model", "ema"}
    ):
        raise ValueError("Interval contract needs completed training and weight-selection identity")
    objective = training["objective"]
    if plan.sampler == "meanflow":
        values = dict(objective)
        for name in ("type", "time_direction", "jvp"):
            values.pop(name)
        if MeanFlowObjective(**values).config_dict() != objective:
            raise ValueError("MeanFlow objective configuration differs from native semantics")
        validate_successful_update_record(
            training.get("successful_update"),
            {
                "class": "aster.methods.meanflow.MeanFlowObjective",
                "codec": "config_dict",
                "configuration": objective,
            },
            role_updates=training["generator_updates"],
        )
    else:
        base = objective.get("base_steps")
        if (
            objective.get("model") != model.config.to_dict()
            or type(base) is not int
            or base < 2
            or base & (base - 1)
        ):
            raise ValueError("Shortcut contract lacks matching training levels/model")
        if plan.steps > base:
            raise ValueError("Shortcut plan exceeds the explicitly trained base_steps")
        if objective.get("endpoint_sigma") != 1e-5 or objective.get("target_clip") != 4.0:
            raise ValueError("Shortcut training path/bootstrap convention differs")
    if plan.noise_shape != (
        model.config.in_channels,
        model.config.input_size,
        model.config.input_size,
    ):
        raise ValueError("Interval noise geometry differs from its fixed native model")
    return {
        "schema_version": 1,
        "policy_artifact_id": artifact.id,
        "model_relative_path": layout,
        "sampling_mode": plan.sampler,
        **_semantics(plan.sampler),
        "generation_contract": contract,
        "generation_contract_sha256": file_digest(artifact.path / "generation_contract.json"),
        "training_semantics_bound": True,
        "forward_calls_per_successful_sample": interval_nfe(plan),
    }


def sample_interval(model, plan, noise, labels):

    if isinstance(plan, MeanFlowSamplingPlan):
        return sample_meanflow(model, noise, labels=labels, timesteps=plan.timesteps)
    if isinstance(plan, ShortcutSamplingPlan):
        return sample_shortcut(
            model, noise, labels=labels, steps=plan.steps, guidance_scale=plan.guidance_scale
        )
    raise ValueError("Unsupported interval plan")


def _sources():
    base = Path(__file__).resolve().parents[1]
    names = (
        "evaluation/interval_generation.py",
        "evaluation/generative.py",
        "evaluation/generation_artifacts.py",
        "models/interval_dit.py",
        "models/serialization.py",
        "models/__init__.py",
        "methods/meanflow.py",
        "methods/shortcut.py",
        "core/update_provenance.py",
    )
    return {name: file_digest(base / name) for name in names}


def generate_interval_shard(
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

    from ..models.generative import AutoencoderKL

    if (
        not isinstance(plan, (MeanFlowSamplingPlan, ShortcutSamplingPlan))
        or type(rank) is not int
        or type(world_size) is not int
        or not 0 <= rank < world_size
    ):
        raise ValueError("Typed interval plan and valid shard required")
    artifact = store.get(policy_artifact_id, verify=True)
    model, layout = load_native_artifact_model(artifact)
    binding = interval_binding(artifact, model, layout, plan)
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
            raise ValueError("Interval latent decoding requires a pinned FP32 native KL-VAE")
        decoder = decoder.eval().to(device)
        parents += (decoder_artifact_id,)
        binding["decoder"] = {
            "artifact_id": decoder_artifact_id,
            "model_relative_path": decoder_path,
        }
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    assigned = [(i, c) for i, c in enumerate(plan.cases) if i % world_size == rank]
    samples, inputs, counter = [], [], [0]
    started = time.monotonic()
    hook = model.register_forward_pre_hook(lambda *_: counter.__setitem__(0, counter[0] + 1))
    try:
        with torch.no_grad():
            for index, case in assigned:
                evidence = {
                    "id": case.id,
                    "seed": case.seed,
                    "label": case.condition,
                    "noise_sha256": None,
                    "nfe": None,
                }
                counter[0] = 0
                try:
                    if case.condition >= model.config.num_classes:
                        raise ValueError("Class is outside the trained label range")
                    rng = torch.Generator(device=device).manual_seed(case.seed)
                    noise = torch.randn(
                        (1, *plan.noise_shape), generator=rng, device=device, dtype=torch.float32
                    )
                    evidence["noise_sha256"] = hashlib.sha256(
                        noise.cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                    labels = torch.tensor([case.condition], device=device, dtype=torch.int64)
                    output = sample_interval(model, plan, noise, labels)
                    evidence["nfe"] = counter[0]
                    if counter[0] != interval_nfe(plan):
                        raise ValueError("Unexpected native interval forward call count")
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
    manifest = MediaManifest(
        "images",
        "native_interval_generated",
        plan.id,
        "generation",
        "producer_artifact_terms",
        plan.cohort_id,
        tuple(c.id for _, c in assigned),
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
    evidence = record.get("sample_inputs")
    cases = {c.id: c for c in plan.cases}
    if (
        not isinstance(evidence, list)
        or tuple(x.get("id") for x in evidence) != manifest.expected_ids
    ):
        raise ValueError("Interval sample input evidence must include the ordered full sample set")
    for sample, value in zip(manifest.samples, evidence):
        if (
            value.get("seed") != cases[sample.id].seed
            or value.get("label") != cases[sample.id].condition
            or type(value.get("nfe")) is not int
            or value["nfe"] < 0
        ):
            raise ValueError("Interval input seed/label/NFE evidence is inconsistent")
        if sample.status == "ok" and (
            value["nfe"] != interval_nfe(plan)
            or not isinstance(value.get("noise_sha256"), str)
            or re.fullmatch("[a-f0-9]{64}", value["noise_sha256"]) is None
        ):
            raise ValueError("Successful interval generation lacks its real noise/NFE evidence")


def merge_interval_shards(shard_directories, plan, output_directory):
    if not isinstance(plan, (MeanFlowSamplingPlan, ShortcutSamplingPlan)):
        raise ValueError("Expected typed interval plan")
    roots = tuple(Path(p) for p in shard_directories)
    by_id = {}
    for root in roots:
        record, manifest = read_json(root / "shard.json"), MediaManifest.load(root)
        _check_inputs(record, manifest, plan)
        for value in record["sample_inputs"]:
            if value["id"] in by_id:
                raise ValueError("Duplicate interval input evidence")
            by_id[value["id"]] = value
    manifest = _merge_image_shards(
        roots, plan, output_directory, dataset_id="native_interval_generated"
    )
    path = Path(output_directory) / "generation.json"
    record = read_json(path)
    record["sample_inputs"] = [by_id[c.id] for c in plan.cases]
    atomic_json(path, record)
    return manifest


def interval_plan_from_record(record):
    values = dict(record["plan"])
    values["cases"] = tuple(GenerationCase(**c) for c in values["cases"])
    variant = record["sampling_binding"]["sampling_mode"]
    if variant == "meanflow":
        return MeanFlowSamplingPlan(**values)
    if variant == "shortcut":
        return ShortcutSamplingPlan(**values)
    raise ValueError("Unknown interval sampling mode")


def interval_generation_record(root, manifest):
    root = Path(root)
    record = read_json(
        root / ("generation.json" if (root / "generation.json").exists() else "shard.json")
    )
    plan = interval_plan_from_record(record)
    if (
        manifest.dataset_id != "native_interval_generated"
        or record["plan_id"] != plan.id
        or manifest.revision != plan.id
        or manifest.cohort_id != plan.cohort_id
        or manifest.expected_ids != tuple(c.id for c in plan.cases)
    ):
        raise ValueError("Interval generation must cover the full planned cohort")
    binding = record["sampling_binding"]
    if (
        record.get("sampling_binding_id") != digest_json(binding)
        or not manifest.producer_artifacts
        or binding.get("policy_artifact_id") != manifest.producer_artifacts[0]
        or binding.get("forward_calls_per_successful_sample") != interval_nfe(plan)
    ):
        raise ValueError("Interval artifact/NFE binding differs from the plan")
    if any(binding.get(key) != value for key, value in _semantics(plan.sampler).items()):
        raise ValueError("Interval sampler time/conditioning semantics changed")
    if not record.get("native_producer_sources") or any(
        not isinstance(value, str) or re.fullmatch("[a-f0-9]{64}", value) is None
        for value in record["native_producer_sources"].values()
    ):
        raise ValueError("Interval producer lacks source identity")
    if any(sample.seed != case.seed for sample, case in zip(manifest.samples, plan.cases)):
        raise ValueError("Interval sample seed differs from plan")
    _check_inputs(record, manifest, plan)
    return record
