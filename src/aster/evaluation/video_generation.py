"""Native Wan video generation and frame artifacts under a shared FVD protocol."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import io
import math
from pathlib import Path
import shutil
import time

from PIL import Image
import torch

from ..core import atomic_json, digest_json, file_digest, read_json
from .generative import (
    ImageFile,
    MediaSample,
    MediaManifest,
    _sha,
    _under,
    _image_identity,
    quantize_image,
    runtime_environment,
)


_CONDITION_KEYS = {"text", "text_lengths", "image_features", "video_condition"}


def _tensor_conditions(branches):

    if (
        not isinstance(branches, dict)
        or not set(branches) <= {"positive", "negative"}
        or "positive" not in branches
    ):
        raise ValueError("Condition case requires positive and optional negative tensor branches")
    tensors = {}
    for branch, condition in branches.items():
        if (
            not isinstance(condition, dict)
            or "text" not in condition
            or not set(condition) <= _CONDITION_KEYS
        ):
            raise ValueError("Unknown/missing video condition tensor field")
        for name, tensor in condition.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.layout != torch.strided
                or tensor.device.type == "meta"
                or tensor.numel() == 0
            ):
                raise ValueError("Video conditions must be real, nonempty dense tensors")
            if name == "text_lengths":
                if (
                    tensor.shape != (1,)
                    or tensor.dtype not in {torch.int32, torch.int64}
                    or bool((tensor < 0).any())
                ):
                    raise ValueError("One original text length is required per condition case")
            elif (
                not tensor.is_floating_point()
                or tensor.ndim < 1
                or tensor.shape[0] != 1
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(
                    "Condition features must be finite floating tensors with batch size one"
                )
            elif name in {"text", "image_features"} and tensor.ndim != 3:
                raise ValueError("Text/image features are [1,L,D], not token IDs or raw pixels")
            elif name == "video_condition" and tensor.ndim != 5:
                raise ValueError("Video conditioning is [1,C,T,H,W]")
            tensors[branch + "." + name] = tensor.detach().cpu().contiguous().clone()
        if (
            "text_lengths" in condition
            and int(condition["text_lengths"][0]) > condition["text"].shape[1]
        ):
            raise ValueError("Original text length exceeds stored text features")
    return tensors


def publish_video_conditions(
    store,
    cases_by_key,
    output_directory,
    *,
    source_artifact_ids=(),
    declared_encoder_artifact_id=None,
    declared_processor_artifact_id=None,
):

    if not cases_by_key or any(not isinstance(key, str) or not key for key in cases_by_key):
        raise ValueError("Provide an explicit nonempty condition key set")
    source_artifact_ids = tuple(source_artifact_ids)
    parents = source_artifact_ids + tuple(
        x for x in (declared_encoder_artifact_id, declared_processor_artifact_id) if x is not None
    )
    parents = tuple(dict.fromkeys(parents))
    for artifact_id in parents:
        store.get(artifact_id, verify=True)
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    entries = {}
    for key, branches in cases_by_key.items():
        tensors = _tensor_conditions(branches)
        name = digest_json(key) + ".pt"
        with (root / name).open("xb") as stream:
            torch.save(tensors, stream)
        entries[key] = {
            "path": name,
            "sha256": file_digest(root / name),
            "tensors": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in tensors.items()
            },
        }
    provenance = {
        "origin": "caller_provided_numeric_tensors",
        "encoder_execution_verified": False,
        "source_artifact_ids": list(source_artifact_ids),
        "declared_encoder_artifact_id": declared_encoder_artifact_id,
        "declared_processor_artifact_id": declared_processor_artifact_id,
        "text_semantics": "stored_features_not_automatically_official_T5_or_CLIP",
    }
    manifest = {
        "schema_version": 1,
        "format": "video_condition_tensors_v1",
        "entries": entries,
        "provenance": provenance,
    }
    atomic_json(root / "conditions.json", manifest)
    return store.publish(
        root,
        kind="video_condition_tensors_v1",
        metadata={"conditions_manifest_id": digest_json(manifest), "provenance": provenance},
        parents=parents,
    )


class VideoConditionBundle:
    def __init__(self, store, artifact_id):
        artifact = store.get(artifact_id, verify=True)
        manifest = read_json(artifact.path / "conditions.json")
        if (
            artifact.kind != "video_condition_tensors_v1"
            or set(manifest) != {"schema_version", "format", "entries", "provenance"}
            or manifest["schema_version"] != 1
            or manifest["format"] != artifact.kind
        ):
            raise ValueError("Not a supported tensor-only video condition artifact")
        if artifact.metadata.get("conditions_manifest_id") != digest_json(manifest):
            raise ValueError("Condition manifest differs from artifact metadata")
        provenance = manifest["provenance"]
        if (
            artifact.metadata.get("provenance") != provenance
            or provenance.get("origin") != "caller_provided_numeric_tensors"
            or provenance.get("encoder_execution_verified") is not False
        ):
            raise ValueError("Unrecognized/overclaimed condition provenance")
        parents = provenance["source_artifact_ids"] + [
            v
            for v in (
                provenance["declared_encoder_artifact_id"],
                provenance["declared_processor_artifact_id"],
            )
            if v is not None
        ]
        if tuple(dict.fromkeys(parents)) != artifact.parents:
            raise ValueError("Condition source lineage differs from stored artifact parents")
        for parent in artifact.parents:
            store.get(parent, verify=True)
        if not manifest["entries"]:
            raise ValueError("Empty condition set")
        self.artifact_id, self.root, self.entries = artifact_id, artifact.path, manifest["entries"]
        self.provenance = provenance

    def load_case(self, key, *, device="cpu", dtype=torch.float32):
        entry = self.entries[key]
        path = _under(self.root, entry["path"])
        if file_digest(path) != entry["sha256"]:
            raise ValueError("Video condition tensor bytes changed")
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensors, dict) or set(tensors) != set(entry["tensors"]):
            raise ValueError("Condition file tensor key set differs from manifest")
        branches = {}
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor) or entry["tensors"][name] != {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }:
                raise ValueError("Condition tensor dtype/shape differs from manifest")
            parts = name.split(".")
            if len(parts) != 2:
                raise ValueError("Malformed video condition tensor name")
            branches.setdefault(parts[0], {})[parts[1]] = value
        checked = _tensor_conditions(branches)

        for name, value in checked.items():
            branch, field = name.split(".")
            branches[branch][field] = value.to(
                device=device, dtype=dtype if value.is_floating_point() else value.dtype
            ).clone()
        return branches


@dataclass(frozen=True)
class VideoGenerationCase:
    id: str
    seed: int
    condition_key: str

    def __post_init__(self):
        if (
            not all(isinstance(value, str) and value for value in (self.id, self.condition_key))
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("Video generation needs explicit sample/condition identity and seed")


@dataclass(frozen=True)
class VideoSamplingPlan:
    cases: tuple[VideoGenerationCase, ...]
    condition_artifact_id: str
    latent_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int]
    fps: float
    steps: int = 30
    solver: str = "heun"
    shift: float = 5.0
    guidance_scale: float = 1.0
    quantization: str = "minus_one_one_stylegan"

    def __post_init__(self):
        for field in ("cases", "latent_shape", "output_shape"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if (
            not self.cases
            or len({case.id for case in self.cases}) != len(self.cases)
            or not _sha(self.condition_artifact_id)
        ):
            raise ValueError("Video cohort needs unique cases and a condition artifact hash")
        if (
            len(self.latent_shape) != 4
            or len(self.output_shape) != 3
            or any(type(x) is not int or x < 1 for x in (*self.latent_shape, *self.output_shape))
        ):
            raise ValueError("Declare latent CTHW and decoded THW")
        if type(self.steps) is not int or self.steps < 1 or self.solver not in {"euler", "heun"}:
            raise ValueError("Only native Euler/Heun video flow sampling is implemented")
        if (
            any(
                type(x) not in {int, float} or not math.isfinite(x)
                for x in (self.fps, self.shift, self.guidance_scale)
            )
            or min(self.fps, self.shift) <= 0
            or self.guidance_scale < 0
        ):
            raise ValueError("Invalid frame rate, shift or guidance")
        if self.quantization not in {"minus_one_one_stylegan", "zero_one_round"}:
            raise ValueError("Unknown video frame quantization")

    @property
    def id(self):
        return digest_json(asdict(self))

    @property
    def cohort_id(self):
        return digest_json(
            {
                "cases": [asdict(case) for case in self.cases],
                "condition_artifact_id": self.condition_artifact_id,
                "output_shape": self.output_shape,
                "fps": self.fps,
                "quantization": self.quantization,
            }
        )

    @property
    def frame_indices(self):
        return tuple(range(self.output_shape[0]))


def _sources():
    root = Path(__file__).resolve().parents[1]
    names = (
        "evaluation/video_generation.py",
        "evaluation/generative.py",
        "methods/video_generation.py",
        "models/video_world.py",
        "models/video_vae.py",
        "models/serialization.py",
        "models/config.py",
        "models/__init__.py",
    )
    return {name: file_digest(root / name) for name in names}


def generate_video_shard(
    store,
    field_artifact_id,
    vae_artifact_id,
    plan,
    output_directory,
    *,
    rank=0,
    world_size=1,
    device="cpu",
):

    from ..models import load_model
    from ..models.video_world import WanVideoDiT
    from ..models.video_vae import WanVideoVAE
    from ..methods.video_generation import VideoGenerationPipeline

    if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
        raise ValueError("Invalid video generation rank")
    field = load_model(store.get(field_artifact_id, verify=True).path).eval().to(device)
    vae = load_model(store.get(vae_artifact_id, verify=True).path).eval().to(device)
    if type(field) is not WanVideoDiT or type(vae) is not WanVideoVAE:
        raise TypeError("Video sampler only supports native Wan field/VAE artifacts")
    if any(
        parameter.dtype != torch.float32
        for model in (field, vae)
        for parameter in model.parameters()
    ):
        raise ValueError("This verified video producer currently requires FP32 model weights")
    pipeline = VideoGenerationPipeline(field, vae).eval()
    conditions = VideoConditionBundle(store, plan.condition_artifact_id)
    expected_shape = (
        1 + (plan.latent_shape[1] - 1) * vae.config.temporal_stride,
        plan.latent_shape[2] * vae.config.spatial_stride,
        plan.latent_shape[3] * vae.config.spatial_stride,
    )
    if (
        plan.latent_shape[0] != field.config.latent_channels
        or plan.output_shape != expected_shape
        or any(n % p for n, p in zip(plan.latent_shape[1:], field.config.patch_size))
    ):
        raise ValueError("Video shape differs from native field patch/VAE stride geometry")
    if any(case.condition_key not in conditions.entries for case in plan.cases):
        raise ValueError("A planned condition key is missing from the immutable condition artifact")
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    selected = [(i, case) for i, case in enumerate(plan.cases) if i % world_size == rank]
    started, outcomes = time.monotonic(), []
    for index, case in selected:
        try:
            branches = conditions.load_case(case.condition_key, device=device)
            generator = torch.Generator(device=device).manual_seed(case.seed)
            noise = torch.randn(
                (1, *plan.latent_shape), device=device, dtype=torch.float32, generator=generator
            )
            with torch.no_grad():
                video = pipeline.generate(
                    noise,
                    branches["positive"],
                    steps=plan.steps,
                    solver=plan.solver,
                    shift=plan.shift,
                    guidance_scale=plan.guidance_scale,
                    negative_condition=branches.get("negative"),
                )
            if video.shape != (1, 3, *plan.output_shape) or not torch.isfinite(video).all():
                raise ValueError(
                    "Native decoded video differs from declared finite RGB output geometry"
                )

            encoded = []
            for frame in plan.frame_indices:
                buffer = io.BytesIO()
                Image.fromarray(quantize_image(video[0, :, frame], plan.quantization)).save(
                    buffer, format="PNG"
                )
                encoded.append(buffer.getvalue())
            files = []
            for frame, data in zip(plan.frame_indices, encoded):
                name = f"{index:08d}-{digest_json(case.id)[:16]}-f{frame:06d}.png"
                with (root / name).open("xb") as stream:
                    stream.write(data)
                files.append(ImageFile(name, *_image_identity(root / name)))
            outcomes.append(
                MediaSample(
                    case.id,
                    tuple(files),
                    seed=case.seed,
                    frame_indices=plan.frame_indices,
                    fps=plan.fps,
                )
            )
        except Exception as error:
            outcomes.append(
                MediaSample(
                    case.id,
                    (),
                    "error",
                    case.seed,
                    type(error).__name__,
                    plan.frame_indices,
                    plan.fps,
                )
            )
    parents = (field_artifact_id, vae_artifact_id, plan.condition_artifact_id)
    manifest = MediaManifest(
        "video_frames",
        "native_wan_video_generated",
        plan.id,
        "generation",
        "producer_artifact_terms",
        plan.cohort_id,
        tuple(case.id for _, case in selected),
        tuple(outcomes),
        parents,
    )
    manifest.save(root)
    atomic_json(
        root / "video_shard.json",
        {
            "schema_version": 1,
            "plan": asdict(plan),
            "plan_id": plan.id,
            "rank": rank,
            "world_size": world_size,
            "manifest_id": manifest.id,
            "producer_artifacts": list(parents),
            "condition_provenance": conditions.provenance,
            "native_producer_sources": _sources(),
            "environment": runtime_environment(device),
            "condition_cast": "float_features_to_float32_preserve_integer_lengths",
            "end_to_end_seconds_including_io": time.monotonic() - started,
        },
    )
    return manifest


def _plan(values):
    data = dict(values)
    data["cases"] = tuple(VideoGenerationCase(**case) for case in data["cases"])
    return VideoSamplingPlan(**data)


def merge_video_shards(shard_directories, plan, output_directory):

    shards, seen = [], set()
    for directory in shard_directories:
        root = Path(directory)
        record = read_json(root / "video_shard.json")
        manifest = MediaManifest.load(root).verify(root, require_complete=False)
        rank, size = record["rank"], record["world_size"]
        if type(rank) is not int or type(size) is not int or not 0 <= rank < size or rank in seen:
            raise ValueError("Invalid/duplicate video shard rank")
        expected_cases = tuple(case for i, case in enumerate(plan.cases) if i % size == rank)
        if (
            record["plan_id"] != plan.id
            or _plan(record["plan"]).id != plan.id
            or manifest.revision != plan.id
            or manifest.cohort_id != plan.cohort_id
            or manifest.id != record["manifest_id"]
        ):
            raise ValueError("Video shard plan/manifest identity mismatch")
        if (
            manifest.dataset_id != "native_wan_video_generated"
            or manifest.kind != "video_frames"
            or manifest.expected_ids != tuple(case.id for case in expected_cases)
        ):
            raise ValueError("Video shard sample set mismatch")
        if (
            any(
                sample.seed != case.seed
                or sample.frame_indices != plan.frame_indices
                or sample.fps != plan.fps
                for sample, case in zip(manifest.samples, expected_cases)
            )
            or list(manifest.producer_artifacts) != record["producer_artifacts"]
        ):
            raise ValueError("Video shard seed/frame/FPS/producer mismatch")
        shards.append((root, record, manifest))
        seen.add(rank)
    if (
        not shards
        or seen != set(range(len(shards)))
        or any(record["world_size"] != len(shards) for _, record, _ in shards)
    ):
        raise ValueError("All video generation ranks must be present")
    initial = shards[0][1]
    same_fields = (
        "producer_artifacts",
        "environment",
        "native_producer_sources",
        "condition_provenance",
        "condition_cast",
    )
    if any(
        any(record[field] != initial[field] for field in same_fields) for _, record, _ in shards
    ):
        raise ValueError("Mixed video models/conditions/software across generation ranks")
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    outcomes = {}
    for source, _, manifest in shards:
        for sample in manifest.samples:
            if sample.id in outcomes:
                raise ValueError("Duplicate generated video case")
            outcomes[sample.id] = sample
            for frame in sample.files:
                shutil.copyfile(_under(source, frame.path), root / frame.path)
    manifest = MediaManifest(
        "video_frames",
        "native_wan_video_generated",
        plan.id,
        "generation",
        "producer_artifact_terms",
        plan.cohort_id,
        tuple(case.id for case in plan.cases),
        tuple(outcomes[case.id] for case in plan.cases),
        tuple(initial["producer_artifacts"]),
    )
    manifest.verify(root, require_complete=False).save(root)
    atomic_json(
        root / "video_generation.json",
        {
            "schema_version": 1,
            "plan": asdict(plan),
            "plan_id": plan.id,
            "world_size": len(shards),
            "shard_manifest_ids": [
                manifest.id for _, _, manifest in sorted(shards, key=lambda s: s[1]["rank"])
            ],
            **{field: initial[field] for field in same_fields},
        },
    )
    return manifest


def video_generation_record(root, manifest):

    root = Path(root)
    path = root / (
        "video_generation.json" if (root / "video_generation.json").exists() else "video_shard.json"
    )
    record = read_json(path)
    plan = _plan(record["plan"])
    if (
        record["plan_id"] != plan.id
        or manifest.revision != plan.id
        or manifest.cohort_id != plan.cohort_id
    ):
        raise ValueError("Native video plan provenance mismatch")
    if manifest.expected_ids != tuple(case.id for case in plan.cases):
        raise ValueError("A video shard is not the full planned cohort")
    if (
        list(manifest.producer_artifacts) != record["producer_artifacts"]
        or len(manifest.producer_artifacts) != 3
        or manifest.producer_artifacts[2] != plan.condition_artifact_id
    ):
        raise ValueError("Native video field/VAE/condition lineage mismatch")
    if not record.get("native_producer_sources") or any(
        not _sha(value) for value in record["native_producer_sources"].values()
    ):
        raise ValueError("Native video source identities are missing")
    for sample, case in zip(manifest.samples, plan.cases):
        if (
            sample.seed != case.seed
            or sample.frame_indices != plan.frame_indices
            or sample.fps != plan.fps
        ):
            raise ValueError("Video sample seed/frame/FPS differs from generation plan")
        if sample.status == "ok" and any(
            (frame.height, frame.width) != plan.output_shape[1:] for frame in sample.files
        ):
            raise ValueError("Video frame geometry differs from generation plan")
    return record
