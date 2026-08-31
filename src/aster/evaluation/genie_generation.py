"""Genie video artifacts, paired controllability, and bounded resource measurements."""

from dataclasses import asdict, dataclass
import io
import math
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from PIL import Image

from ..core import atomic_json, digest_json, file_digest, read_json
from ..methods.genie_artifact_training import (
    GenieVideoCorpus,
    TokenizedGenieData,
    load_trained_genie,
    tensor_identity,
)
from ..planning.genie import generate_genie_video
from .genie_world import paired_delta_psnr
from .generative import (
    ImageFile,
    MediaSample,
    MediaManifest,
    _image_identity,
    _under,
    _sha,
    quantize_image,
    runtime_environment,
    DistributionProtocol,
    evaluate_media_directories,
)
from .generation_performance import GenerationBenchmarkSettings
from .protocol import ComparisonProtocol, EvaluationRun, EvaluationRecord


@dataclass(frozen=True)
class GenieGenerationCase:
    id: str
    source_id: str
    seed: int

    def __post_init__(self):
        if (
            not all(isinstance(x, str) and x for x in (self.id, self.source_id))
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63 - 1
        ):
            raise ValueError(
                "Genie case requires explicit identity/source and a seed with a separate seed+1 action stream"
            )


@dataclass(frozen=True)
class GenieSamplingPlan:
    cases: tuple[GenieGenerationCase, ...]
    video_artifact_id: str
    time_index: int = 4
    steps: int = 25
    token_temperature: float = 1.0
    choice_temperature: float = 2.0
    mask_order: str = "confidence"
    mse_floor: float = 1e-12
    quantization: str = "zero_one_round"

    def __post_init__(self):
        object.__setattr__(self, "cases", tuple(self.cases))
        if (
            not self.cases
            or any(not isinstance(case, GenieGenerationCase) for case in self.cases)
            or len({c.id for c in self.cases}) != len(self.cases)
            or not _sha(self.video_artifact_id)
        ):
            raise ValueError(
                "Genie plan fixes an immutable corpus and a unique complete case population"
            )
        if (
            type(self.time_index) is not int
            or self.time_index < 1
            or type(self.steps) is not int
            or self.steps < 1
        ):
            raise ValueError(
                "Genie horizon and actual MaskGIT iteration count must be positive integers"
            )
        if (
            any(
                type(x) not in {float, int} or not math.isfinite(x)
                for x in (self.token_temperature, self.choice_temperature, self.mse_floor)
            )
            or self.token_temperature <= 0
            or self.choice_temperature < 0
            or not 0 < self.mse_floor <= 1
        ):
            raise ValueError("Invalid categorical/Gumbel temperature or finite PSNR floor")
        if self.mask_order not in {"confidence", "random"} or self.quantization != "zero_one_round":
            raise ValueError("Unsupported Genie masking or pixel quantization protocol")

    @property
    def id(self):
        return digest_json(asdict(self))

    @property
    def cohort_id(self):

        return digest_json(
            {
                "cases": [asdict(c) for c in self.cases],
                "video_artifact_id": self.video_artifact_id,
                "time_index": self.time_index,
                "mse_floor": self.mse_floor,
                "quantization": self.quantization,
                "conditioning": "first_frame_pixels_and_model_inferred_or_random_action_codes",
            }
        )

    @property
    def frame_indices(self):
        return tuple(range(self.time_index + 1))


def _plan(value):
    return GenieSamplingPlan(
        **{**value, "cases": tuple(GenieGenerationCase(**case) for case in value["cases"])}
    )


def _sources():
    root = Path(__file__).resolve().parents[1]
    names = (
        "evaluation/genie_generation.py",
        "evaluation/genie_world.py",
        "evaluation/generative.py",
        "methods/genie_artifact_training.py",
        "methods/genie.py",
        "planning/genie.py",
        "models/genie.py",
        "models/serialization.py",
        "models/config.py",
        "models/__init__.py",
        "core/update_provenance.py",
    )
    return {name: file_digest(root / name) for name in names}


def _environment(device):

    return {
        **runtime_environment(device),
        "torch_num_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "host_identity": digest_json(platform.node()),
    }


def _prepare(store, world_id, plan, device):
    if not isinstance(plan, GenieSamplingPlan):
        raise TypeError("Use a typed native Genie sampling plan")
    world, contract = load_trained_genie(store, world_id, world=True)
    objective = contract["objective"]
    data = TokenizedGenieData(store, objective["trace_artifact_id"])
    expected = {
        "type": "bound_genie_world",
        "trace_artifact_id": data.artifact_id,
        "tokenizer_artifact_id": data.tokenizer_artifact_id,
        "video_artifact_id": data.video_artifact_id,
        "normalization": data.corpus.spec.normalization,
        "objective": objective["objective"],
    }
    if objective != expected:
        raise ValueError("World last actual objective differs from tokenization lineage")
    tokenizer, codec = load_trained_genie(store, data.tokenizer_artifact_id)
    corpus = GenieVideoCorpus(store, plan.video_artifact_id)
    tc, wc = tokenizer.config, world.config
    if (tc.num_codes, tc.spatial_tokens, tc.max_frames) != (
        wc.dynamics.vocab_size,
        wc.dynamics.spatial_tokens,
        wc.dynamics.max_frames,
    ):
        raise ValueError("World/tokenizer code geometries differ")
    if (
        corpus.shape[1:] != (3, tc.image_height, tc.image_width)
        or (wc.action.image_channels, wc.action.image_height, wc.action.image_width)
        != corpus.shape[1:]
        or tc.image_channels != 3
    ):
        raise ValueError("PNG generation currently requires aligned native RGB video geometry")
    if not plan.time_index < min(tc.max_frames, wc.action.max_frames, corpus.shape[0]) or any(
        case.source_id not in corpus.ids for case in plan.cases
    ):
        raise ValueError(
            "Planned Genie horizon/source is outside the complete fixed corpus/context"
        )
    training_pixels = {
        digest_json(tensor_identity(data.corpus.load(key)["video"])) for key in data.corpus.ids
    }
    overlapping = [
        key
        for key in corpus.ids
        if digest_json(tensor_identity(corpus.load(key)["video"])) in training_pixels
    ]
    binding = {
        "world_artifact_id": world_id,
        "tokenizer_artifact_id": data.tokenizer_artifact_id,
        "training_trace_artifact_id": data.artifact_id,
        "training_video_artifact_id": data.video_artifact_id,
        "evaluation_video_artifact_id": plan.video_artifact_id,
        "world_contract": contract,
        "tokenizer_contract": codec,
        "evaluation_spec": asdict(corpus.spec),
        "normalization": corpus.spec.normalization,
        "exact_training_pixel_overlap_ids": overlapping,
        "near_duplicate_or_external_contamination_checked": False,
        "action_semantics": "each_world_infers_its_own_latent_codes_not_shared_physical_action_labels",
        "pixels_visible_to_generation": "only_frame_zero",
        "future_pixels_used_for_action_inference": True,
    }
    return tokenizer.to(device), world.to(device), corpus, binding


@torch.no_grad()
def _sample_pair(tokenizer, world, row, case, plan, device):
    video = row["video"][None].to(device)
    if not row["valid"][: plan.time_index + 1].all():
        raise ValueError("Planned evaluation includes missing/invalid source frames")
    counters = {"dynamics": 0, "tokenizer_encodes": 0, "tokenizer_decodes": 0, "action_encodes": 0}

    def increment(name):
        def hook(*_):
            counters[name] += 1

        return hook

    handles = [
        module.register_forward_hook(increment(name))
        for name, module in (
            ("dynamics", world.dynamics),
            ("tokenizer_encodes", tokenizer.encoder),
            ("tokenizer_decodes", tokenizer.decoder),
            ("action_encodes", world.action_model.encoder),
        )
    ]
    try:
        inferred_actions = world.action_model.encode(video[:, : plan.time_index + 1]).indices
        action_rng = torch.Generator(device=device).manual_seed(case.seed + 1)
        random_actions = torch.randint(
            world.config.action.num_codes,
            inferred_actions.shape,
            generator=action_rng,
            device=device,
        )
        outputs, diagnostics = {}, {}
        for branch, actions in (("inferred", inferred_actions), ("random", random_actions)):
            output, info = generate_genie_video(
                tokenizer,
                world,
                video[:, :1],
                actions,
                generator=torch.Generator(device=device).manual_seed(case.seed),
                steps=plan.steps,
                token_temperature=plan.token_temperature,
                choice_temperature=plan.choice_temperature,
                mask_order=plan.mask_order,
            )
            outputs[branch] = output
            outputs[branch + "_tokens"] = info["tokens"]
            outputs[branch + "_actions"] = actions
            diagnostics[branch] = {key: value for key, value in info.items() if key != "tokens"}
    finally:
        for handle in handles:
            handle.remove()
    if (
        counters
        != {
            "dynamics": 2 * plan.time_index * plan.steps,
            "tokenizer_encodes": 2,
            "tokenizer_decodes": 2,
            "action_encodes": 1,
        }
        or sum(info["model_calls"] for info in diagnostics.values()) != counters["dynamics"]
    ):
        raise RuntimeError("Actual native forward counts differ from fixed Genie protocol")
    if any(
        not torch.isfinite(outputs[branch]).all()
        or outputs[branch].min() < 0
        or outputs[branch].max() > 1
        for branch in ("inferred", "random")
    ):
        raise ValueError("Generated Genie pixels are not finite [0,1]")
    return outputs, {"actual_calls": counters, "branches": diagnostics}


def _write_envelope(path, record):
    atomic_json(path, {"id": digest_json(record), "record": record})


def _read_envelope(path):
    saved = read_json(path)
    if set(saved) != {"id", "record"} or saved["id"] != digest_json(saved["record"]):
        raise ValueError("Genie evidence digest differs")
    return saved["record"]


def generate_genie_shard(
    store, world_artifact_id, plan, directory, *, rank=0, world_size=1, device="cpu"
):

    if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
        raise ValueError("Invalid Genie generation rank")
    tokenizer, world, corpus, binding = _prepare(store, world_artifact_id, plan, device)
    sources = _sources()
    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    for branch in ("inferred", "random"):
        (root / branch).mkdir()
    selected = [
        (index, case) for index, case in enumerate(plan.cases) if index % world_size == rank
    ]
    samples = {"inferred": [], "random": []}
    outcomes = []
    parents = tuple(
        dict.fromkeys(
            (
                world_artifact_id,
                binding["tokenizer_artifact_id"],
                binding["training_trace_artifact_id"],
                plan.video_artifact_id,
            )
        )
    )
    for index, case in selected:
        outcome = {"id": case.id, "status": "error", "error": None}
        created = []
        try:
            outputs, info = _sample_pair(
                tokenizer, world, corpus.load(case.source_id), case, plan, device
            )
            tensors = {key: value.detach().cpu().contiguous() for key, value in outputs.items()}
            name = f"{index:08d}-{digest_json(case.id)[:16]}"
            encoded = {}
            for branch in samples:
                encoded[branch] = []
                for frame in plan.frame_indices:
                    stream = io.BytesIO()
                    Image.fromarray(
                        quantize_image(tensors[branch][0, frame], plan.quantization)
                    ).save(stream, format="PNG")
                    encoded[branch].append((f"{name}-f{frame:06d}.png", stream.getvalue()))
            with (root / (name + ".pt")).open("xb") as stream:
                created.append(root / (name + ".pt"))
                torch.save(tensors, stream)
            committed = {}
            for branch in samples:
                files = []
                for relative, payload in encoded[branch]:
                    with (root / branch / relative).open("xb") as stream:
                        created.append(root / branch / relative)
                        stream.write(payload)
                    files.append(ImageFile(relative, *_image_identity(root / branch / relative)))
                committed[branch] = MediaSample(
                    case.id,
                    tuple(files),
                    seed=case.seed,
                    frame_indices=plan.frame_indices,
                    fps=corpus.spec.fps,
                )
            outcome.update(
                status="ok",
                tensor_file=name + ".pt",
                tensor_file_sha256=file_digest(root / (name + ".pt")),
                tensors={key: tensor_identity(value) for key, value in tensors.items()},
                diagnostics=info,
            )
            for branch in samples:
                samples[branch].append(committed[branch])
        except Exception as error:
            for path in created:
                path.unlink(missing_ok=True)
            outcome = {"id": case.id, "status": "error", "error": None}
            outcome["error"] = type(error).__name__
            for branch in samples:
                samples[branch].append(
                    MediaSample(
                        case.id,
                        (),
                        "error",
                        case.seed,
                        type(error).__name__,
                        plan.frame_indices,
                        corpus.spec.fps,
                    )
                )
        outcomes.append(outcome)
    corpus.verify()
    for parent in parents:
        store.get(parent, verify=True)
    if sources != _sources():
        raise RuntimeError("Genie source changed during generation")
    manifests = {}
    for branch, values in samples.items():
        manifest = MediaManifest(
            "video_frames",
            "native_genie_" + branch,
            plan.id,
            "generation",
            "producer_artifact_terms",
            plan.cohort_id,
            tuple(case.id for _, case in selected),
            tuple(values),
            parents,
        )
        manifest.save(root / branch)
        manifests[branch] = manifest.id
    record = {
        "schema_version": 1,
        "plan": asdict(plan),
        "plan_id": plan.id,
        "cohort_id": plan.cohort_id,
        "rank": rank,
        "world_size": world_size,
        "binding": binding,
        "binding_id": digest_json(binding),
        "native_producer_sources": sources,
        "environment": _environment(device),
        "outcomes": outcomes,
        "manifests": manifests,
    }
    _write_envelope(root / "genie_shard.json", record)
    return record


def _check_record(root, record, *, complete):
    plan = _plan(record["plan"])
    if (
        record.get("schema_version") != 1
        or record["plan_id"] != plan.id
        or record["cohort_id"] != plan.cohort_id
        or record["binding_id"] != digest_json(record["binding"])
    ):
        raise ValueError("Genie plan/cohort/model binding differs")
    rank, size = record.get("rank", 0), record["world_size"]
    if type(size) is not int or size < 1 or type(rank) is not int or not 0 <= rank < size:
        raise ValueError("Malformed Genie generation rank")
    cases = (
        plan.cases
        if complete
        else tuple(case for i, case in enumerate(plan.cases) if i % size == rank)
    )
    if tuple(row["id"] for row in record["outcomes"]) != tuple(case.id for case in cases):
        raise ValueError("Genie outcomes omit/reorder planned cases")
    if not record.get("native_producer_sources") or any(
        not _sha(value) for value in record["native_producer_sources"].values()
    ):
        raise ValueError("Genie native producer source identity missing")
    binding = record["binding"]
    if binding["evaluation_video_artifact_id"] != plan.video_artifact_id:
        raise ValueError("Genie evaluation corpus differs")
    for branch in ("inferred", "random"):
        manifest = MediaManifest.load(Path(root) / branch).verify(
            Path(root) / branch, require_complete=False
        )
        if (
            manifest.id != record["manifests"][branch]
            or manifest.dataset_id != "native_genie_" + branch
            or manifest.cohort_id != plan.cohort_id
            or manifest.revision != plan.id
            or manifest.expected_ids != tuple(case.id for case in cases)
        ):
            raise ValueError("Genie branch manifest differs")
        expected_parents = tuple(
            dict.fromkeys(
                (
                    binding["world_artifact_id"],
                    binding["tokenizer_artifact_id"],
                    binding["training_trace_artifact_id"],
                    plan.video_artifact_id,
                )
            )
        )
        if manifest.producer_artifacts != expected_parents:
            raise ValueError("Genie branch model/codec/trace lineage differs")
        for sample, case, outcome in zip(manifest.samples, cases, record["outcomes"]):
            if (
                sample.status != outcome["status"]
                or sample.seed != case.seed
                or sample.frame_indices != plan.frame_indices
                or sample.fps != binding["evaluation_spec"]["fps"]
            ):
                raise ValueError("Genie branch outcome/seed/frames/FPS differs")
            config = binding["tokenizer_contract"]["model"]
            if any(
                (image.height, image.width) != (config["image_height"], config["image_width"])
                for image in sample.files
            ):
                raise ValueError("Genie PNG geometry differs from its native codec contract")
    for outcome in record["outcomes"]:
        if outcome["status"] == "ok":
            expected_calls = {
                "dynamics": 2 * plan.time_index * plan.steps,
                "tokenizer_encodes": 2,
                "tokenizer_decodes": 2,
                "action_encodes": 1,
            }
            if outcome["diagnostics"]["actual_calls"] != expected_calls:
                raise ValueError("Genie saved NFE differs from the native execution protocol")
            path = _under(root, outcome["tensor_file"])
            if file_digest(path) != outcome["tensor_file_sha256"]:
                raise ValueError("Genie raw trajectory bytes changed")
            values = torch.load(path, map_location="cpu", weights_only=True)
            if (
                set(values)
                != {
                    branch + suffix
                    for branch in ("inferred", "random")
                    for suffix in ("", "_tokens", "_actions")
                }
                or {key: tensor_identity(value) for key, value in values.items()}
                != outcome["tensors"]
            ):
                raise ValueError("Genie raw trajectory numeric identity differs")
            config = binding["tokenizer_contract"]["model"]
            for branch in ("inferred", "random"):
                if (
                    values[branch].shape
                    != (1, plan.time_index + 1, 3, config["image_height"], config["image_width"])
                    or values[branch].dtype != torch.float32
                    or not torch.isfinite(values[branch]).all()
                    or values[branch].min() < 0
                    or values[branch].max() > 1
                ):
                    raise ValueError(
                        "Genie raw pixel layout/normalization differs from its native producer"
                    )
                if (
                    values[branch + "_tokens"].dtype != torch.int64
                    or values[branch + "_tokens"].shape
                    != (
                        1,
                        plan.time_index + 1,
                        config["image_height"] * config["image_width"] // config["patch_size"] ** 2,
                    )
                    or (values[branch + "_tokens"] < 0).any()
                    or (values[branch + "_tokens"] >= config["num_codes"]).any()
                ):
                    raise ValueError("Genie raw code indices differ from its codec")
                if (
                    values[branch + "_actions"].dtype != torch.int64
                    or values[branch + "_actions"].shape != (1, plan.time_index)
                    or (values[branch + "_actions"] < 0).any()
                    or (
                        values[branch + "_actions"]
                        >= binding["world_contract"]["model"]["action"]["num_codes"]
                    ).any()
                ):
                    raise ValueError("Genie raw latent actions differ from its world model")
                sample = next(
                    sample
                    for sample in MediaManifest.load(Path(root) / branch).samples
                    if sample.id == outcome["id"]
                )
                for frame, image_file in zip(plan.frame_indices, sample.files):
                    with Image.open(_under(Path(root) / branch, image_file.path)) as image:
                        actual = np.asarray(image)
                    if not np.array_equal(
                        actual, quantize_image(values[branch][0, frame], plan.quantization)
                    ):
                        raise ValueError(
                            "Genie PNG pixels differ from the actual raw trajectory/quantization"
                        )
    return plan


def merge_genie_shards(directories, plan, directory):
    shards = []
    ranks = set()
    for directory_in in directories:
        source = Path(directory_in)
        record = _read_envelope(source / "genie_shard.json")
        if _check_record(source, record, complete=False).id != plan.id or record["rank"] in ranks:
            raise ValueError("Mixed/duplicate Genie shards")
        shards.append((source, record))
        ranks.add(record["rank"])
    if (
        not shards
        or ranks != set(range(len(shards)))
        or any(record["world_size"] != len(shards) for _, record in shards)
    ):
        raise ValueError("All Genie ranks, including empty ranks, must be present")
    same = ("binding", "binding_id", "native_producer_sources", "environment")
    initial = shards[0][1]
    if any(any(record[key] != initial[key] for key in same) for _, record in shards):
        raise ValueError("Mixed Genie model/source/environment shards")
    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    for branch in ("inferred", "random"):
        (root / branch).mkdir()
    outcomes = {}
    samples = {"inferred": {}, "random": {}}
    for source, record in shards:
        for row in record["outcomes"]:
            outcomes[row["id"]] = row
            if row["status"] == "ok":
                shutil.copyfile(_under(source, row["tensor_file"]), root / row["tensor_file"])
        for branch in samples:
            for sample in MediaManifest.load(source / branch).samples:
                samples[branch][sample.id] = sample
                for frame in sample.files:
                    shutil.copyfile(_under(source / branch, frame.path), root / branch / frame.path)
    manifests = {}
    for branch in samples:
        old = MediaManifest.load(shards[0][0] / branch)
        merged = MediaManifest(
            old.kind,
            old.dataset_id,
            plan.id,
            old.split,
            old.license_id,
            plan.cohort_id,
            tuple(case.id for case in plan.cases),
            tuple(samples[branch][case.id] for case in plan.cases),
            old.producer_artifacts,
        )
        merged.verify(root / branch, require_complete=False).save(root / branch)
        manifests[branch] = merged.id
    result = {
        "schema_version": 1,
        "plan": asdict(plan),
        "plan_id": plan.id,
        "cohort_id": plan.cohort_id,
        "world_size": len(shards),
        **{key: initial[key] for key in same},
        "outcomes": [outcomes[case.id] for case in plan.cases],
        "manifests": manifests,
    }
    _write_envelope(root / "genie_generation.json", result)
    return result


def _record(root):
    root = Path(root)
    merged = (root / "genie_generation.json").exists()
    record = _read_envelope(root / ("genie_generation.json" if merged else "genie_shard.json"))
    if not merged and record["world_size"] != 1:
        raise ValueError("A partial Genie shard cannot enter full-cohort evaluation")
    _check_record(root, record, complete=True)
    return record


def genie_generation_record(branch_root, manifest):

    root = Path(branch_root)
    branch = root.name
    if branch not in ("inferred", "random"):
        raise ValueError("Select the explicit Genie trajectory branch")
    record = _record(root.parent)
    if record["manifests"][branch] != manifest.id:
        raise ValueError("Genie FVD manifest is not the actual produced branch")
    return record


def publish_genie_generation(store, directory):

    record = _record(directory)
    manifest = MediaManifest.load(Path(directory) / "inferred")
    return store.publish(
        directory,
        kind="native_genie_generated_cohort",
        metadata={
            "generation_record_id": digest_json(record),
            "cohort_id": record["cohort_id"],
            "manifests": record["manifests"],
            "public_quality_evaluated": False,
        },
        parents=manifest.producer_artifacts,
    )


def evaluate_genie_controls(store, directory):

    root = Path(directory)
    record = _record(root)
    plan = _plan(record["plan"])
    _, _, corpus, binding = _prepare(store, record["binding"]["world_artifact_id"], plan, "cpu")
    if binding != record["binding"]:
        raise ValueError("Genie evaluation artifact binding differs from production")
    metric = f"delta_psnr_t{plan.time_index}"
    protocol = ComparisonProtocol(
        "native_genie_controllability",
        plan.video_artifact_id,
        "native_paired_delta_psnr",
        file_digest(Path(__file__).with_name("genie_world.py")),
        {
            "cohort_id": plan.cohort_id,
            "frame_indices": plan.frame_indices,
            "fps": corpus.spec.fps,
            "mse_floor": plan.mse_floor,
            "metric_pixels": "original_float32_not_quantized_png",
            "conditioning": binding["action_semantics"],
        },
        tuple(case.id for case in plan.cases),
        metric,
        True,
        10 * math.log10(plan.mse_floor),
    )
    run = EvaluationRun(
        protocol,
        binding["world_artifact_id"],
        environment=record["environment"],
        transforms=(
            {
                "sampling_plan_id": plan.id,
                "native_sources": record["native_producer_sources"],
                "binding_id": record["binding_id"],
            },
        ),
    )
    for case, outcome in zip(plan.cases, record["outcomes"]):
        if outcome["status"] != "ok":
            run.add(EvaluationRecord(case.id, "error", error=outcome["error"]))
            continue
        values = torch.load(
            _under(root, outcome["tensor_file"]), weights_only=True, map_location="cpu"
        )
        reference = corpus.load(case.source_id)["video"][None, plan.time_index]
        delta, direct, random = paired_delta_psnr(
            reference,
            values["inferred"][:, plan.time_index],
            values["random"][:, plan.time_index],
            mse_floor=plan.mse_floor,
        )
        run.add(
            EvaluationRecord(
                case.id,
                "ok",
                {
                    metric: float(delta[0]),
                    "inferred_psnr": float(direct[0]),
                    "random_psnr": float(random[0]),
                },
                details={
                    "actual_calls": outcome["diagnostics"]["actual_calls"],
                    "raw_tensor_sha256": outcome["tensor_file_sha256"],
                    "public_quality_evaluated": False,
                },
            )
        )
    return run.finalize()


def benchmark_genie_sampler(
    store,
    world_artifact_id,
    plan,
    directory,
    *,
    settings=GenerationBenchmarkSettings(),
    device="cpu",
):

    tokenizer, world, corpus, binding = _prepare(store, world_artifact_id, plan, device)
    sources = _sources()
    torch_device = torch.device(device)
    trials = []
    warmup_errors = []

    rows = {
        case.id: {key: value.to(device) for key, value in corpus.load(case.source_id).items()}
        for case in plan.cases
    }

    def synchronize():
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

    for repetition in range(-settings.warmup_repetitions, settings.repetitions):
        for case in plan.cases:
            trial = {"case_id": case.id, "repetition": repetition, "status": "error", "error": None}
            try:
                synchronize()
                if torch_device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(torch_device)
                start = time.perf_counter()
                values, info = _sample_pair(tokenizer, world, rows[case.id], case, plan, device)
                synchronize()
                elapsed = time.perf_counter() - start
                peak = (
                    torch.cuda.max_memory_allocated(torch_device)
                    if torch_device.type == "cuda"
                    else None
                )
                trial.update(
                    status="ok",
                    paired_generation_seconds=elapsed,
                    actual_calls=info["actual_calls"],
                    cuda_peak_allocated_bytes=peak,
                    tensors={key: tensor_identity(value) for key, value in values.items()},
                )
            except Exception as error:
                trial["error"] = type(error).__name__
            if repetition >= 0:
                trials.append(trial)
            elif trial["status"] != "ok":
                warmup_errors.append(trial)
    corpus.verify()
    for artifact_id in (
        world_artifact_id,
        binding["tokenizer_artifact_id"],
        binding["training_trace_artifact_id"],
    ):
        store.get(artifact_id, verify=True)
    if sources != _sources():
        raise RuntimeError("Genie benchmark source changed")
    report = {
        "schema_version": 1,
        "plan": asdict(plan),
        "plan_id": plan.id,
        "cohort_id": plan.cohort_id,
        "binding": binding,
        "binding_id": digest_json(binding),
        "native_producer_sources": sources,
        "environment": _environment(device),
        "settings": asdict(settings),
        "trials": trials,
        "warmup_errors": warmup_errors,
        "status": "ok"
        if not warmup_errors and all(t["status"] == "ok" for t in trials)
        else "error",
        "timing_scope": "LAM_inference_plus_two_native_video_generations_excludes_load_trace_io_metric",
        "memory_scope": "absolute_torch_cuda_allocator_peak_including_resident_models_or_unavailable_on_cpu",
        "hardware_isolation": "host_asserted_not_os_enforced"
        if settings.isolated_hardware_asserted
        else "development_only",
    }
    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    _write_envelope(root / "benchmark.json", report)
    return report


@dataclass(frozen=True)
class GenieFVDResources:
    protocol: DistributionProtocol
    reference_root: str
    source_root: str
    weights_path: str
    grant: object


def evaluate_genie_fvd(
    directory, output_directory, *, resources=None, branch="inferred", device="cpu"
):

    record = _record(directory)
    plan = _plan(record["plan"])
    if branch not in ("inferred", "random"):
        raise ValueError("Choose the actual Genie FVD branch")
    if resources is None:
        report = {
            "status": "not_evaluated",
            "reason": "approved_pinned_official_I3D_and_reference_corpus_unavailable",
            "metrics": {},
            "cohort_id": plan.cohort_id,
            "branch": branch,
            "expected_ids": [case.id for case in plan.cases],
        }
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=False)
        _write_envelope(root / "not_evaluated.json", report)
        return report
    if not isinstance(resources, GenieFVDResources):
        raise TypeError("Provide explicit GenieFVDResources")
    protocol = resources.protocol
    if (
        protocol.metrics != ("fvd_styleganv_i3d",)
        or protocol.generated_cohort_id != plan.cohort_id
        or protocol.frame_indices != plan.frame_indices
        or protocol.fps != record["binding"]["evaluation_spec"]["fps"]
    ):
        raise ValueError("FVD protocol differs from fixed Genie clip/cohort/FPS")
    return evaluate_media_directories(
        protocol,
        resources.reference_root,
        Path(directory) / branch,
        source_root=resources.source_root,
        weights_path=resources.weights_path,
        grant=resources.grant,
        output_directory=output_directory,
        device=device,
    )


def _performance_for_generation(path, generation):
    report = _read_envelope(path)
    plan = _plan(generation["plan"])
    settings = GenerationBenchmarkSettings(**report["settings"])
    if (
        report.get("timing_scope")
        != "LAM_inference_plus_two_native_video_generations_excludes_load_trace_io_metric"
        or report.get("memory_scope")
        != "absolute_torch_cuda_allocator_peak_including_resident_models_or_unavailable_on_cpu"
    ):
        raise ValueError("Genie benchmark measured a different timing/memory scope")
    if report["hardware_isolation"] != (
        "host_asserted_not_os_enforced"
        if settings.isolated_hardware_asserted
        else "development_only"
    ):
        raise ValueError("Genie hardware-isolation declaration differs")
    for key in (
        "plan_id",
        "cohort_id",
        "binding",
        "binding_id",
        "native_producer_sources",
        "environment",
    ):
        if report[key] != generation[key]:
            raise ValueError(
                "Genie quality and performance used different artifacts/plan/software/environment"
            )
    if (
        _plan(report["plan"]).id != plan.id
        or report.get("warmup_errors")
        or report["status"] != "ok"
    ):
        raise ValueError("Genie benchmark did not complete every warmup/measured case")
    expected = [
        (repetition, case.id) for repetition in range(settings.repetitions) for case in plan.cases
    ]
    if [(row["repetition"], row["case_id"]) for row in report["trials"]] != expected:
        raise ValueError("Genie benchmark omitted/reordered/duplicated a measured trial")
    outputs = {row["id"]: row for row in generation["outcomes"]}
    for row in report["trials"]:
        reference = outputs[row["case_id"]]
        if (
            row["status"] != "ok"
            or reference["status"] != "ok"
            or row["tensors"] != reference["tensors"]
            or row["actual_calls"] != reference["diagnostics"]["actual_calls"]
        ):
            raise ValueError("Genie measured output/NFE differs from quality trajectory")
        if (
            type(row["paired_generation_seconds"]) not in {float, int}
            or not math.isfinite(row["paired_generation_seconds"])
            or row["paired_generation_seconds"] <= 0
        ):
            raise ValueError("Genie performance duration is missing/nonfinite/nonpositive")
        memory = row["cuda_peak_allocated_bytes"]
        if memory is not None and (type(memory) is not int or memory < 1):
            raise ValueError("Invalid real CUDA allocator measurement")
    return report


def _cluster_ci(values, repetitions, confidence, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid complete-cohort comparison values")
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(1)
    low, high = np.quantile(estimates, ((1 - confidence) / 2, (1 + confidence) / 2))
    return {
        "mean": float(values.mean()),
        "low": float(low),
        "high": float(high),
        "independent_cohorts": len(values),
        "confidence": confidence,
    }


def compare_genie_cohorts(
    store,
    pairs,
    directory,
    *,
    resources=None,
    max_delta_psnr_regression=0.0,
    max_fvd_regression=0.0,
    minimum_latency_improvement=0.05,
    maximum_memory_ratio=None,
    repetitions=1000,
    confidence=0.95,
    seed=0,
):

    pairs = tuple(pairs)
    numeric = (
        max_delta_psnr_regression,
        max_fvd_regression,
        minimum_latency_improvement,
        confidence,
    )
    if (
        not pairs
        or any(type(x) not in {int, float} or not math.isfinite(x) for x in numeric)
        or min(numeric[:3]) < 0
        or minimum_latency_improvement >= 1
        or not 0 < confidence < 1
        or type(repetitions) is not int
        or repetitions < 100
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("Invalid Genie joint quality/resource gate controls")
    if maximum_memory_ratio is not None and (
        type(maximum_memory_ratio) not in {int, float}
        or not math.isfinite(maximum_memory_ratio)
        or maximum_memory_ratio <= 0
    ):
        raise ValueError("Invalid memory regression limit")
    if resources is not None and len(resources) != len(pairs):
        raise ValueError("Each cohort requires its own fixed FVD protocol/grant")
    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    seeds = set()
    quality, latency, fvd = [], [], []
    unevaluated = []
    common = None
    rejected = []
    official_protocol = None
    if len(pairs) < 3:
        unevaluated.append("at_least_three_independent_complete_cohorts_required")
    for index, pair in enumerate(pairs):
        if set(pair) != {"baseline", "candidate", "baseline_benchmark", "candidate_benchmark"}:
            raise ValueError("Explicit paired generation/benchmark paths are required")
        baseline, candidate = _record(pair["baseline"]), _record(pair["candidate"])
        bp, cp = _plan(baseline["plan"]), _plan(candidate["plan"])
        if (
            bp.cohort_id != cp.cohort_id
            or baseline["native_producer_sources"] != candidate["native_producer_sources"]
            or baseline["environment"] != candidate["environment"]
        ):
            raise ValueError("Genie comparison changed fixed cohort/software/environment")
        current = {
            "baseline": baseline["binding_id"],
            "candidate": candidate["binding_id"],
            "environment": baseline["environment"],
            "source": baseline["native_producer_sources"],
            "data": bp.video_artifact_id,
            "baseline_sampling": {
                key: value for key, value in baseline["plan"].items() if key != "cases"
            },
            "candidate_sampling": {
                key: value for key, value in candidate["plan"].items() if key != "cases"
            },
        }
        if common is None:
            common = current
        elif common != current:
            raise ValueError("Genie multi-cohort gate mixed models/data/sampling controls")
        cohort_seeds = {case.seed for case in bp.cases}
        if seeds & cohort_seeds:
            raise ValueError("Independent Genie cohorts cannot reuse generation seeds")
        seeds |= cohort_seeds
        base_run, candidate_run = (
            evaluate_genie_controls(store, pair["baseline"]),
            evaluate_genie_controls(store, pair["candidate"]),
        )
        if any(
            record["binding"]["exact_training_pixel_overlap_ids"]
            or record["binding"]["evaluation_spec"]["split"].lower()
            not in {"test", "validation", "evaluation"}
            for record in (baseline, candidate)
        ):
            unevaluated.append("held_out_nonoverlapping_evaluation_corpus_required")
        if base_run.protocol.id != candidate_run.protocol.id:
            raise ValueError("Genie paired control protocols differ")
        if any(
            row.status != "ok" for run in (base_run, candidate_run) for row in run.records.values()
        ):
            rejected.append("failed_generation_in_full_population")
        quality.append(float(candidate_run.scores().mean() - base_run.scores().mean()))
        base_performance = _performance_for_generation(pair["baseline_benchmark"], baseline)
        cand_performance = _performance_for_generation(pair["candidate_benchmark"], candidate)
        if not all(
            report["settings"]["isolated_hardware_asserted"]
            for report in (base_performance, cand_performance)
        ):
            unevaluated.append("hardware_isolation_not_asserted")
        base_seconds = np.mean(
            [row["paired_generation_seconds"] for row in base_performance["trials"]]
        )
        cand_seconds = np.mean(
            [row["paired_generation_seconds"] for row in cand_performance["trials"]]
        )
        latency.append(float(1 - cand_seconds / base_seconds))
        base_nfe = {row["actual_calls"]["dynamics"] for row in base_performance["trials"]}
        cand_nfe = {row["actual_calls"]["dynamics"] for row in cand_performance["trials"]}
        if max(cand_nfe) > min(base_nfe):
            rejected.append("dynamics_NFE_regression")
        if maximum_memory_ratio is not None:
            base_mem = [row["cuda_peak_allocated_bytes"] for row in base_performance["trials"]]
            cand_mem = [row["cuda_peak_allocated_bytes"] for row in cand_performance["trials"]]
            if any(value is None for value in base_mem + cand_mem):
                unevaluated.append("actual_CUDA_memory_measurement_unavailable")
            elif max(cand_mem) / max(base_mem) > maximum_memory_ratio:
                rejected.append("CUDA_memory_regression")
        resource = None if resources is None else resources[index]
        if resource is not None:
            if not isinstance(resource, GenieFVDResources):
                raise TypeError("Each resource must have an explicit fixed official FVD protocol")
            signature = {
                key: value
                for key, value in resource.protocol.to_dict().items()
                if key not in {"generated_cohort_id", "expected_generated_ids"}
            }
            if official_protocol is None:
                official_protocol = signature
            elif official_protocol != signature:
                raise ValueError(
                    "Genie FVD cohorts cannot mix reference populations/extractors/preprocessing"
                )
        base_fvd = evaluate_genie_fvd(
            pair["baseline"], root / f"{index:04d}_baseline_fvd", resources=resource
        )
        cand_fvd = evaluate_genie_fvd(
            pair["candidate"], root / f"{index:04d}_candidate_fvd", resources=resource
        )
        if all(value["status"] == "ok" for value in (base_fvd, cand_fvd)):
            fvd.append(
                base_fvd["metrics"]["fvd_styleganv_i3d"]["value"]
                - cand_fvd["metrics"]["fvd_styleganv_i3d"]["value"]
            )
        else:
            unevaluated.append("official_FVD_not_successfully_evaluated")
        rows.append(
            {
                "cohort_id": bp.cohort_id,
                "baseline_plan_id": bp.id,
                "candidate_plan_id": cp.id,
                "baseline_control": base_run.summary(),
                "candidate_control": candidate_run.summary(),
                "baseline_nfe": sorted(base_nfe),
                "candidate_nfe": sorted(cand_nfe),
                "baseline_benchmark_id": digest_json(base_performance),
                "candidate_benchmark_id": digest_json(cand_performance),
                "baseline_fvd_id": digest_json(base_fvd),
                "candidate_fvd_id": digest_json(cand_fvd),
            }
        )
    comparisons = {
        "delta_psnr_improvement": _cluster_ci(quality, repetitions, confidence, seed),
        "fractional_latency_improvement": _cluster_ci(latency, repetitions, confidence, seed + 1),
        "fvd_improvement": _cluster_ci(fvd, repetitions, confidence, seed + 2)
        if len(fvd) == len(pairs)
        else None,
    }
    if comparisons["delta_psnr_improvement"]["low"] < -max_delta_psnr_regression:
        rejected.append("paired_control_quality_regression")
    if comparisons["fractional_latency_improvement"]["low"] < minimum_latency_improvement:
        rejected.append("real_latency_improvement_not_demonstrated")
    if (
        comparisons["fvd_improvement"] is not None
        and comparisons["fvd_improvement"]["low"] < -max_fvd_regression
    ):
        rejected.append("official_FVD_quality_regression")
    report = {
        "schema_version": 1,
        "status": "reject" if rejected else ("not_evaluated" if unevaluated else "promote"),
        "reasons": sorted(set(rejected)),
        "unevaluated": sorted(set(unevaluated)),
        "cohorts": rows,
        "comparison": comparisons,
        "controls": {
            "max_delta_psnr_regression": max_delta_psnr_regression,
            "max_fvd_regression": max_fvd_regression,
            "minimum_latency_improvement": minimum_latency_improvement,
            "maximum_memory_ratio": maximum_memory_ratio,
            "bootstrap_repetitions": repetitions,
            "confidence": confidence,
            "seed": seed,
        },
        "aggregation": "paired_independent_complete_cohorts_not_per_video_FVD",
        "automatically_deployed": False,
    }
    _write_envelope(root / "gate.json", report)
    return report
