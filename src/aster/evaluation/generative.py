"""Native media generation, complete sample manifests, and local feature-based evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.metadata
import io
import math
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import threading
import time

import numpy as np
from PIL import Image
import torch

from ..core import atomic_json, digest_json, file_digest, read_json
from .generation_artifacts import (
    load_native_artifact_model,
    resolve_image_sampling,
    publish_dmd_generator,
)


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_NUMPY_RANDOM_LOCK = threading.Lock()
_WEIGHT_URLS = {
    "cleanfid_inception": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt",
    "styleganv_i3d": "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1",
}


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _regular(path):

    path = Path(path).absolute()
    for node in (path, *path.parents):
        attrs = node.stat(follow_symlinks=False)
        if node.is_symlink() or getattr(attrs, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024
        ):
            raise ValueError("Media/source paths cannot contain links or reparse points")
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Expected a regular local file")
    return path


def _under(root, relative):
    relative = PurePosixPath(relative)
    if (
        relative.is_absolute()
        or any(part in {"..", "."} or ":" in part or "\\" in part for part in relative.parts)
        or not relative.parts
    ):
        raise ValueError("Media paths must be portable relative paths")
    root = Path(root).absolute()
    path = _regular(root.joinpath(*relative.parts))
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Media path escapes root")
    return path


def _image_identity(path):
    path = _regular(path)
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ValueError("Only explicit PNG/JPEG/WebP media files are supported")
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1 or image.width * image.height > 100_000_000:
            raise ValueError("Animated/oversized images require a separate protocol")

        if image.mode != "RGB":
            raise ValueError("Image manifest requires explicit RGB uint8 encoding")
        image.load()
        width, height = image.size
    return file_digest(path), width, height


@dataclass(frozen=True)
class ImageFile:
    path: str
    sha256: str
    width: int
    height: int

    def __post_init__(self):
        if not _sha(self.sha256) or any(
            type(x) is not int or x < 1 for x in (self.width, self.height)
        ):
            raise ValueError("Image dimensions/hash must be explicit")
        p = PurePosixPath(self.path)
        if (
            p.is_absolute()
            or not p.parts
            or any(x == ".." or ":" in x or "\\" in x for x in p.parts)
        ):
            raise ValueError("Invalid relative image path")


@dataclass(frozen=True)
class MediaSample:
    id: str
    files: tuple[ImageFile, ...]
    status: str = "ok"
    seed: int | None = None
    error: str | None = None
    frame_indices: tuple[int, ...] = ()
    fps: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "frame_indices", tuple(self.frame_indices))
        if not isinstance(self.id, str) or not self.id or self.status not in {"ok", "error"}:
            raise ValueError("Each sample needs identity and an explicit outcome")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError("Seed must be a nonnegative signed 64-bit integer")
        if self.status == "ok" and (not self.files or self.error is not None):
            raise ValueError("Successful media sample needs files and no error")
        if self.status == "error" and (self.files or not self.error):
            raise ValueError(
                "Failed samples retain their ID, not partial output pretending to be valid"
            )
        if self.frame_indices:
            if (
                self.fps is None
                or type(self.fps) not in {float, int}
                or not math.isfinite(self.fps)
                or self.fps <= 0
            ):
                raise ValueError("Video clips need their original frame rate")
            if (
                any(type(x) is not int or x < 0 for x in self.frame_indices)
                or tuple(sorted(set(self.frame_indices))) != self.frame_indices
            ):
                raise ValueError("Video frame indices must be unique and increasing")
            if self.status == "ok" and len(self.frame_indices) != len(self.files):
                raise ValueError("Video frame manifest is incomplete")
        elif self.fps is not None:
            raise ValueError("Frame rate without a frame selection is ambiguous")


@dataclass(frozen=True)
class MediaManifest:
    kind: str
    dataset_id: str
    revision: str
    split: str
    license_id: str
    cohort_id: str
    expected_ids: tuple[str, ...]
    samples: tuple[MediaSample, ...]
    producer_artifacts: tuple[str, ...] = ()

    def __post_init__(self):
        for field in ("expected_ids", "samples", "producer_artifacts"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if self.kind not in {"images", "video_frames"} or not all(
            (self.dataset_id, self.revision, self.split, self.license_id)
        ):
            raise ValueError("Dataset/split/license and media kind are mandatory")
        if self.revision.lower() in {"main", "master", "latest"} or not _sha(self.cohort_id):
            raise ValueError("Use a fixed data revision and cohort digest")
        if tuple(s.id for s in self.samples) != self.expected_ids or len(
            set(self.expected_ids)
        ) != len(self.expected_ids):
            raise ValueError("Sample outcomes must cover the exact ordered expected ID set")
        if any(not _sha(x) for x in self.producer_artifacts) or len(
            set(self.producer_artifacts)
        ) != len(self.producer_artifacts):
            raise ValueError("Producer lineage must contain unique artifact hashes")
        paths = [f.path for s in self.samples for f in s.files]
        if len(set(paths)) != len(paths):
            raise ValueError("Media files cannot be reused under multiple sample/frame identities")
        for sample in self.samples:
            if self.kind == "images" and (
                sample.frame_indices
                or sample.fps is not None
                or (sample.status == "ok" and len(sample.files) != 1)
            ):
                raise ValueError("Image samples have exactly one image, not a clip")
            if self.kind == "video_frames" and not sample.frame_indices:
                raise ValueError("Every video outcome must retain its planned frame selection")

    @property
    def id(self):
        return digest_json(asdict(self))

    def save(self, root):
        atomic_json(Path(root) / "media.json", {"manifest_id": self.id, "manifest": asdict(self)})

    @classmethod
    def load(cls, root):
        saved = read_json(Path(root) / "media.json")
        data = dict(saved["manifest"])
        data["samples"] = tuple(
            MediaSample(**{**s, "files": tuple(ImageFile(**f) for f in s["files"])})
            for s in data["samples"]
        )
        result = cls(**data)
        if set(saved) != {"manifest_id", "manifest"} or result.id != saved["manifest_id"]:
            raise ValueError("Media manifest identity mismatch")
        return result

    def verify(self, root, *, require_complete=True):
        root = Path(root).absolute()
        listed = set()
        for sample in self.samples:
            if sample.status != "ok":
                if require_complete:
                    raise ValueError(
                        "Failed generation remains in the cohort; no distribution score is valid"
                    )
                continue
            for image in sample.files:
                path = _under(root, image.path)
                if _image_identity(path) != (image.sha256, image.width, image.height):
                    raise ValueError("Media bytes/dimensions differ from the frozen manifest")
                listed.add(image.path)
        actual = set()
        for path in root.rglob("*"):
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError("Media tree contains a redirect")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                _regular(path)
                actual.add(path.relative_to(root).as_posix())
        if listed != actual:
            raise ValueError("Media directory has missing or unlisted images")
        return self


def image_directory_manifest(
    root, *, dataset_id, revision, split, license_id, files_by_id, cohort_id=None
):
    """Bind the full sample-ID/file mapping before evaluation; do not truncate or filter bad images."""
    samples = []
    for sample_id, relative in files_by_id.items():
        identity = _image_identity(_under(root, relative))
        samples.append(MediaSample(sample_id, (ImageFile(relative, *identity),)))
    cohort_id = cohort_id or digest_json(
        {"dataset": dataset_id, "revision": revision, "split": split, "ids": list(files_by_id)}
    )
    return MediaManifest(
        "images",
        dataset_id,
        revision,
        split,
        license_id,
        cohort_id,
        tuple(files_by_id),
        tuple(samples),
    ).verify(root)


def video_directory_manifest(
    root,
    *,
    dataset_id,
    revision,
    split,
    license_id,
    frames_by_id,
    frame_indices,
    fps,
    cohort_id=None,
):

    samples = []
    for sample_id, relatives in frames_by_id.items():
        files = tuple(
            ImageFile(relative, *_image_identity(_under(root, relative))) for relative in relatives
        )
        samples.append(MediaSample(sample_id, files, frame_indices=tuple(frame_indices), fps=fps))
    cohort_id = cohort_id or digest_json(
        {
            "dataset": dataset_id,
            "revision": revision,
            "split": split,
            "ids": list(frames_by_id),
            "frame_indices": list(frame_indices),
            "fps": fps,
        }
    )
    return MediaManifest(
        "video_frames",
        dataset_id,
        revision,
        split,
        license_id,
        cohort_id,
        tuple(frames_by_id),
        tuple(samples),
    ).verify(root)


@dataclass(frozen=True)
class GenerationCase:
    id: str
    seed: int
    condition: tuple[float, ...] | int | None = None

    def __post_init__(self):
        if (
            not isinstance(self.id, str)
            or not self.id
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("Generation case needs a fixed ID and seed")
        if self.condition is not None and type(self.condition) is not int:
            object.__setattr__(self, "condition", tuple(self.condition))
            if any(type(v) not in {int, float} or not math.isfinite(v) for v in self.condition):
                raise ValueError("Vector conditioning must be finite")


@dataclass(frozen=True)
class ImageSamplingPlan:
    cases: tuple[GenerationCase, ...]
    noise_shape: tuple[int, int, int]
    sampler: str = "flow_heun"
    steps: int = 20
    respacing_indices: tuple[int, ...] | None = None
    eta: float = 0.0
    guidance_scale: float = 1.0
    flow_direction: str = "noise_to_data"
    flow_shift: float = 1.0
    clip_clean: bool = False
    learned_variance: bool = False
    quantization: str = "minus_one_one_stylegan"

    def __post_init__(self):
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "noise_shape", tuple(self.noise_shape))
        if self.respacing_indices is not None:
            object.__setattr__(self, "respacing_indices", tuple(self.respacing_indices))
            if (
                any(type(x) is not int or x < 0 for x in self.respacing_indices)
                or tuple(sorted(set(self.respacing_indices))) != self.respacing_indices
            ):
                raise ValueError("Respacing indices must be sorted distinct nonnegative integers")
        if (
            not self.cases
            or len({c.id for c in self.cases}) != len(self.cases)
            or len(self.noise_shape) != 3
            or any(type(x) is not int or x < 1 for x in self.noise_shape)
        ):
            raise ValueError("Generation requires unique cases and CHW noise shape")
        if (
            self.sampler not in {"flow_euler", "flow_heun", "flow_rk4", "ddim", "ddpm", "direct_x0"}
            or type(self.steps) is not int
            or self.steps < 1
        ):
            raise ValueError("Unsupported native sampling algorithm")
        if self.flow_direction not in {"noise_to_data", "data_to_noise"}:
            raise ValueError("Sampling time/parameterization must be explicit")
        if self.quantization not in {"minus_one_one_stylegan", "zero_one_round"}:
            raise ValueError("Unsupported image quantization")
        if (
            any(
                type(x) not in {int, float} or not math.isfinite(x)
                for x in (self.eta, self.guidance_scale, self.flow_shift)
            )
            or self.eta < 0
            or self.flow_shift <= 0
        ):
            raise ValueError("Invalid sampler controls")
        if type(self.clip_clean) is not bool or type(self.learned_variance) is not bool:
            raise ValueError("Sampler flags must be booleans")
        if self.sampler.startswith("flow_") and (
            self.eta != 0
            or self.clip_clean
            or self.learned_variance
            or self.respacing_indices is not None
        ):
            raise ValueError("Diffusion-only controls cannot silently affect a flow protocol")
        if not self.sampler.startswith("flow_") and (
            self.flow_shift != 1 or self.flow_direction != "noise_to_data"
        ):
            raise ValueError("Flow-only controls cannot silently affect a diffusion protocol")
        if self.sampler == "ddpm" and self.eta != 0:
            raise ValueError("DDPM does not use DDIM eta")
        if self.sampler in {"ddpm", "ddim"} and (
            self.steps < 2
            or (self.respacing_indices is not None and len(self.respacing_indices) != self.steps)
        ):
            raise ValueError("Diffusion steps must match at least two selected training marginals")
        if self.sampler == "direct_x0" and (
            self.steps != 1
            or self.eta != 0
            or self.guidance_scale != 1
            or self.clip_clean
            or self.learned_variance
            or self.respacing_indices is not None
        ):
            raise ValueError(
                "Direct x0 is exactly one artifact-bound generator call, without ODE/VP/CFG controls"
            )

    @property
    def id(self):
        return digest_json(asdict(self))

    @property
    def cohort_id(self):

        return digest_json(
            {"cases": [asdict(case) for case in self.cases], "quantization": self.quantization}
        )


def quantize_image(sample, rule):
    """Evaluate the saved uint8 pixels; float-to-pixel rounding is part of the metric protocol."""
    if sample.ndim != 3 or sample.shape[0] != 3 or not torch.isfinite(sample).all():
        raise ValueError("Decoded output must be a finite RGB CHW image")
    if rule == "minus_one_one_stylegan":
        pixels = (sample.detach().float() * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    elif rule == "zero_one_round":
        pixels = (sample.detach().float().clamp(0, 1) * 255 + 0.5).floor().to(torch.uint8)
    else:
        raise ValueError("Unknown output quantization rule")
    return pixels.permute(1, 2, 0).cpu().numpy()


def generate_image_shard(
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

    from ..models.generative import UNet2D, DiT, AutoencoderKL
    from ..methods.generation import sample_diffusion, sample_flow

    if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
        raise ValueError("Invalid generation shard")
    artifact = store.get(policy_artifact_id, verify=True)
    model, model_path = load_native_artifact_model(artifact)
    model = model.eval().to(device)
    if type(model) not in {UNet2D, DiT}:
        raise ValueError(
            "This producer implements native UNet2D/DiT, not external generator wrappers"
        )
    decoder = None
    parents = (policy_artifact_id,)
    if decoder_artifact_id is not None:
        decoder, decoder_path = load_native_artifact_model(
            store.get(decoder_artifact_id, verify=True)
        )
        decoder = decoder.eval().to(device)
        if type(decoder) is not AutoencoderKL:
            raise ValueError(
                "Latent image decoding requires the native, separately pinned AutoencoderKL"
            )
        parents += (decoder_artifact_id,)

    schedule, binding = resolve_image_sampling(artifact, model, model_path, plan)
    binding["decoder"] = (
        None
        if decoder is None
        else {"artifact_id": decoder_artifact_id, "model_relative_path": decoder_path}
    )
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    assigned = [(i, c) for i, c in enumerate(plan.cases) if i % world_size == rank]
    dtype = next(model.parameters()).dtype
    samples = []
    started = time.monotonic()
    for index, case in assigned:
        try:
            generator = torch.Generator(device=device).manual_seed(case.seed)
            noise = torch.randn(
                (1, *plan.noise_shape), generator=generator, device=device, dtype=dtype
            )
            condition = None
            if type(case.condition) is int:
                condition = torch.tensor([case.condition], device=device, dtype=torch.long)
            elif case.condition is not None:
                condition = torch.tensor([case.condition], device=device, dtype=dtype)
            with torch.no_grad():
                if plan.sampler.startswith("flow_"):
                    output = sample_flow(
                        model,
                        noise,
                        steps=plan.steps,
                        solver=plan.sampler[5:],
                        direction=plan.flow_direction,
                        shift=plan.flow_shift,
                        condition=condition,
                        guidance_scale=plan.guidance_scale,
                    )
                elif plan.sampler == "direct_x0":
                    value = model(
                        noise,
                        noise.new_full((1,), binding["generation_contract"]["generator_time"]),
                        condition,
                    )
                    if value.prediction_type != "x0" or value.prediction.shape != noise.shape:
                        raise ValueError("Direct generator must emit one matching clean sample")
                    output = value.prediction
                else:
                    output = sample_diffusion(
                        model,
                        noise,
                        schedule,
                        method=plan.sampler,
                        eta=plan.eta,
                        condition=condition,
                        guidance_scale=plan.guidance_scale,
                        clip_clean=plan.clip_clean,
                        learned_variance=plan.learned_variance,
                        generator=generator,
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
            samples.append(MediaSample(case.id, (), "error", case.seed, type(error).__name__))

    for parent in parents:
        store.get(parent, verify=True)
    manifest = MediaManifest(
        "images",
        "native_generated",
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
            "native_producer_sources": _producer_sources(),
            "sampling_binding": binding,
            "sampling_binding_id": digest_json(binding),
            "environment": runtime_environment(device),
            "end_to_end_seconds_including_io": time.monotonic() - started,
        },
    )
    return manifest


def merge_image_shards(shard_directories, plan, output_directory):
    """Require the complete planned shard set and retain failures; never hide errors by resampling."""
    return _merge_image_shards(
        shard_directories, plan, output_directory, dataset_id="native_generated"
    )


def _merge_image_shards(shard_directories, plan, output_directory, *, dataset_id):

    shards, seen = [], set()
    for directory in shard_directories:
        root = Path(directory)
        shard = read_json(root / "shard.json")
        manifest = MediaManifest.load(root).verify(root, require_complete=False)
        if manifest.dataset_id != dataset_id:
            raise ValueError("Generation shards belong to a different native producer family")
        rank, size = shard["rank"], shard["world_size"]
        if type(rank) is not int or type(size) is not int or not 0 <= rank < size or rank in seen:
            raise ValueError("Duplicate/invalid rank in generation shards")
        if (
            shard["plan_id"] != plan.id
            or digest_json(shard["plan"]) != plan.id
            or manifest.cohort_id != plan.cohort_id
            or shard["manifest_id"] != manifest.id
        ):
            raise ValueError("Generation shard plan/manifest mismatch")
        if list(manifest.producer_artifacts) != shard["producer_artifacts"]:
            raise ValueError("Generation shard lineage differs from media lineage")
        binding = shard.get("sampling_binding")
        if (
            not isinstance(binding, dict)
            or shard.get("sampling_binding_id") != digest_json(binding)
            or binding.get("policy_artifact_id") != manifest.producer_artifacts[0]
        ):
            raise ValueError("Generation shard sampling binding is missing or inconsistent")
        if binding.get("sampling_mode") != (
            "drifting_direct" if dataset_id == "native_drifting_generated" else plan.sampler
        ):
            raise ValueError("Generation shard sampler binding differs from the typed plan")
        expected = tuple(c.id for i, c in enumerate(plan.cases) if i % size == rank)
        if manifest.expected_ids != expected or any(
            s.seed != plan.cases[i].seed
            for i, s in zip(range(rank, len(plan.cases), size), manifest.samples)
        ):
            raise ValueError("Shard omitted/reordered/reseeded samples")
        shards.append((root, shard, manifest))
        seen.add(rank)
    if (
        not shards
        or any(s[1]["world_size"] != len(shards) for s in shards)
        or seen != set(range(len(shards)))
    ):
        raise ValueError("All generation ranks, including empty ranks, must be present")
    parents = shards[0][2].producer_artifacts
    if any(
        m.producer_artifacts != parents
        or s["environment"] != shards[0][1]["environment"]
        or s["native_producer_sources"] != shards[0][1]["native_producer_sources"]
        or s["sampling_binding_id"] != shards[0][1]["sampling_binding_id"]
        for _, s, m in shards
    ):
        raise ValueError("Cannot mix model versions or generation environments across shards")
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    samples = {}
    for source, _, manifest in shards:
        for sample in manifest.samples:
            if sample.id in samples:
                raise ValueError("Duplicate generated case")
            samples[sample.id] = sample
            for image in sample.files:
                shutil.copyfile(_under(source, image.path), root / image.path)
    merged = MediaManifest(
        "images",
        dataset_id,
        plan.id,
        "generation",
        "producer_artifact_terms",
        plan.cohort_id,
        tuple(c.id for c in plan.cases),
        tuple(samples[c.id] for c in plan.cases),
        parents,
    )
    merged.verify(root, require_complete=False).save(root)
    atomic_json(
        root / "generation.json",
        {
            "plan": asdict(plan),
            "plan_id": plan.id,
            "world_size": len(shards),
            "shard_manifest_ids": [m.id for _, _, m in sorted(shards, key=lambda s: s[1]["rank"])],
            "native_producer_sources": shards[0][1]["native_producer_sources"],
            "sampling_binding": shards[0][1]["sampling_binding"],
            "sampling_binding_id": shards[0][1]["sampling_binding_id"],
            "environment": shards[0][1]["environment"],
        },
    )
    return merged


def _producer_sources():
    base = Path(__file__).resolve().parents[1]

    names = (
        "evaluation/generative.py",
        "evaluation/generation_artifacts.py",
        "models/generative.py",
        "models/serialization.py",
        "models/config.py",
        "models/__init__.py",
        "methods/generation.py",
        "core/update_provenance.py",
    )
    return {name: file_digest(base / name) for name in names}


def _generation_record(root, manifest):
    if manifest.dataset_id == "native_wan21_teacache":
        from .wan_teacache import wan_cache_generation_record

        return wan_cache_generation_record(root, manifest)
    if manifest.dataset_id in {"native_genie_inferred", "native_genie_random"}:
        from .genie_generation import genie_generation_record

        return genie_generation_record(root, manifest)
    if manifest.dataset_id == "native_edm_generated":
        from .edm_generation import edm_generation_record

        return edm_generation_record(root, manifest)
    if manifest.dataset_id == "native_consistency_generated":
        from .consistency_generation import consistency_generation_record

        return consistency_generation_record(root, manifest)
    if manifest.dataset_id == "native_wan_video_generated":
        from .video_generation import video_generation_record

        return video_generation_record(root, manifest)
    if manifest.dataset_id == "native_drifting_generated":
        from .drifting_generation import drifting_generation_record

        return drifting_generation_record(root, manifest)
    if manifest.dataset_id == "native_interval_generated":
        from .interval_generation import interval_generation_record

        return interval_generation_record(root, manifest)
    if manifest.dataset_id != "native_generated":
        return None
    root = Path(root)
    path = root / ("generation.json" if (root / "generation.json").exists() else "shard.json")
    record = read_json(path)
    values = dict(record["plan"])
    values["cases"] = tuple(GenerationCase(**case) for case in values["cases"])
    plan = ImageSamplingPlan(**values)
    if (
        record["plan_id"] != plan.id
        or manifest.revision != plan.id
        or manifest.cohort_id != plan.cohort_id
    ):
        raise ValueError("Native media plan provenance mismatch")
    cases = {case.id: case for case in plan.cases}
    if manifest.expected_ids != tuple(cases):
        raise ValueError("A generation shard is not the full planned evaluation cohort")
    if any(
        sample.id not in cases or sample.seed != cases[sample.id].seed
        for sample in manifest.samples
    ):
        raise ValueError("Native media seed provenance mismatch")
    if (
        not manifest.producer_artifacts
        or not record.get("native_producer_sources")
        or any(not _sha(value) for value in record["native_producer_sources"].values())
    ):
        raise ValueError("Native media is missing producer source/weight lineage")
    binding = record.get("sampling_binding")
    if (
        not isinstance(binding, dict)
        or record.get("sampling_binding_id") != digest_json(binding)
        or binding.get("policy_artifact_id") != manifest.producer_artifacts[0]
    ):
        raise ValueError("Native media is missing its artifact-bound sampling semantics")
    if binding.get("sampling_mode") != plan.sampler:
        raise ValueError("Native media sampler differs from its actual binding")
    return record


def source_tree_hash(root):

    root = Path(root).absolute()
    for path in root.rglob("*"):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("Source tree cannot contain redirected directories/files")
    files = {
        path.relative_to(root).as_posix(): file_digest(_regular(path))
        for path in sorted(root.rglob("*.py"))
    }
    if not files:
        raise ValueError("Feature extractor source tree is empty")
    return digest_json(files)


@dataclass(frozen=True)
class ExtractorPin:
    provider: str
    revision: str
    version: str
    source_sha256: str
    weights_sha256: str
    weights_source: str
    license_id: str
    dependencies: tuple[tuple[str, str], ...]

    def __post_init__(self):
        object.__setattr__(self, "dependencies", tuple(tuple(pair) for pair in self.dependencies))
        if self.provider not in _WEIGHT_URLS or not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("Extractor needs a supported provider and full source commit")
        if (
            not self.version
            or not self.license_id
            or not _sha(self.source_sha256)
            or not _sha(self.weights_sha256)
        ):
            raise ValueError("Extractor weights/source/version/license must all be fixed")
        if self.weights_source != _WEIGHT_URLS[self.provider]:
            raise ValueError("This metric protocol requires its documented official weight source")
        expected = {"torch", "numpy", "Pillow"}
        if self.provider == "cleanfid_inception":
            expected |= {"clean-fid", "scipy", "torchvision"}
        if (
            {key for key, _ in self.dependencies} != expected
            or len(self.dependencies) != len(expected)
            or any(not value for _, value in self.dependencies)
        ):
            raise ValueError("All extractor/preprocessing dependency versions must be pinned")
        if (
            self.provider == "cleanfid_inception"
            and dict(self.dependencies)["clean-fid"] != self.version
        ):
            raise ValueError("clean-fid package version differs from extractor version")

    @property
    def id(self):
        return digest_json(asdict(self))

    def verify(self, source_root, weights_path):
        if (
            source_tree_hash(source_root) != self.source_sha256
            or file_digest(_regular(weights_path)) != self.weights_sha256
        ):
            raise ValueError("Feature extractor source/weights hash mismatch")
        required = (
            ("fid.py", "inception_torchscript.py", "resize.py", "utils.py")
            if self.provider == "cleanfid_inception"
            else ("frechet_video_distance.py", "metric_utils.py")
        )
        for name in required:
            _under(source_root, name)
        if (
            self.provider == "cleanfid_inception"
            and Path(weights_path).name != "inception-2015-12-05.pt"
        ):
            raise ValueError("clean-fid local loader uses the exact inception archive filename")
        for name, expected in self.dependencies:
            if importlib.metadata.version(name) != expected:
                raise ValueError("Pinned extractor dependency version mismatch: " + name)


def record_local_extractor(
    provider, *, revision, source_root, weights_path, license_id, version=None
):

    if provider not in _WEIGHT_URLS:
        raise ValueError("Unsupported extractor provider")
    names = {"torch", "numpy", "Pillow"}
    if provider == "cleanfid_inception":
        names |= {"clean-fid", "scipy", "torchvision"}
    dependencies = tuple((name, importlib.metadata.version(name)) for name in sorted(names))
    if provider == "cleanfid_inception":
        version = version or dict(dependencies)["clean-fid"]
    if not version:
        raise ValueError("Unpackaged I3D extractor requires an explicit release/version label")
    pin = ExtractorPin(
        provider,
        revision,
        version,
        source_tree_hash(source_root),
        file_digest(_regular(weights_path)),
        _WEIGHT_URLS[provider],
        license_id,
        dependencies,
    )
    pin.verify(source_root, weights_path)
    return pin


def runtime_environment(device="cpu"):
    device = torch.device(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "Pillow": importlib.metadata.version("Pillow"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.machine(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


@dataclass(frozen=True)
class DistributionProtocol:
    reference_manifest_id: str
    generated_cohort_id: str
    extractor: ExtractorPin
    expected_generated_ids: tuple[str, ...]
    metrics: tuple[str, ...] = ("fid_clean", "kid_clean")
    batch_size: int = 16
    kid_subsets: int = 100
    kid_subset_size: int = 1000
    kid_seed: int = 0
    frame_indices: tuple[int, ...] = ()
    fps: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "frame_indices", tuple(self.frame_indices))
        object.__setattr__(self, "expected_generated_ids", tuple(self.expected_generated_ids))
        if not _sha(self.reference_manifest_id) or not _sha(self.generated_cohort_id):
            raise ValueError(
                "Distribution comparison binds the full reference and candidate sampling cohort"
            )
        if (
            len(self.expected_generated_ids) < 2
            or len(set(self.expected_generated_ids)) != len(self.expected_generated_ids)
            or any(not isinstance(value, str) or not value for value in self.expected_generated_ids)
        ):
            raise ValueError("Distribution protocol must fix the complete generated sample IDs")
        if (
            any(type(x) is not int or x < 1 for x in (self.batch_size, self.kid_subsets))
            or type(self.kid_subset_size) is not int
            or self.kid_subset_size < 2
        ):
            raise ValueError("Feature batch/KID sampling sizes are invalid")
        if type(self.kid_seed) is not int or not 0 <= self.kid_seed < 2**32:
            raise ValueError("KID numpy seed is a uint32")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("Choose explicit unique distribution metrics")
        if self.extractor.provider == "cleanfid_inception":
            if (
                not set(self.metrics) <= {"fid_clean", "kid_clean"}
                or self.frame_indices
                or self.fps is not None
            ):
                raise ValueError("Image Inception and video I3D protocols cannot be interchanged")
        elif (
            self.metrics != ("fvd_styleganv_i3d",)
            or len(self.frame_indices) < 10
            or type(self.fps) not in {int, float}
            or not math.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("I3D FVD requires an explicit >=10-frame clip and FPS protocol")
        if (
            any(type(x) is not int or x < 0 for x in self.frame_indices)
            or tuple(sorted(set(self.frame_indices))) != self.frame_indices
        ):
            raise ValueError("Invalid video frame indices")

    def to_dict(self):
        data = asdict(self)

        data["preprocessing"] = (
            {
                "decode": "Pillow_RGB8_no_EXIF_or_ICC",
                "resize": "PIL_per_channel_float32_bicubic_299",
                "resize_quantization": "none_clip_0_255",
                "network_scale": "(x-128)/128",
                "features": "Inception_pool_2048",
                "covariance_ddof": 1,
            }
            if self.extractor.provider == "cleanfid_inception"
            else {
                "decode": "Pillow_RGB8_no_EXIF_or_ICC",
                "layout": "BCTHW_uint8",
                "resize": "inside_pinned_I3D_archive",
                "detector_kwargs": {"rescale": True, "resize": True, "return_features": True},
                "features": "I3D_Kinetics400_pre_softmax_logits",
                "covariance_ddof": 0,
                "distance_solver": "float64_symmetric_PSD_eigh_not_TensorFlow_sqrtm",
            }
        )
        data["aggregation"] = "distribution_level_no_per_image_accuracy_or_CI"
        return data

    @property
    def id(self):
        return digest_json(self.to_dict())


def _features(value, count, dimension):
    array = np.asarray(value)
    if (
        array.shape != (count, dimension)
        or array.dtype.kind not in "f"
        or not np.isfinite(array).all()
    ):
        raise ValueError("Feature extractor returned wrong shape/type/nonfinite values")
    return array


def _clean_features(root, manifest, model, fid, protocol, device):
    return _features(
        fid.get_files_features(
            [str(_under(root, s.files[0].path)) for s in manifest.samples],
            model=model,
            num_workers=0,
            batch_size=protocol.batch_size,
            device=torch.device(device),
            mode="clean",
            verbose=False,
        ),
        len(manifest.samples),
        2048,
    )


def _video_features(root, manifest, model, protocol, device):
    rows = []
    for start in range(0, len(manifest.samples), protocol.batch_size):
        clips = []
        for sample in manifest.samples[start : start + protocol.batch_size]:
            if sample.frame_indices != protocol.frame_indices or sample.fps != protocol.fps:
                raise ValueError("Video clip selection/FPS differs from fixed protocol")
            frames = []
            for image in sample.files:
                with Image.open(_under(root, image.path)) as frame:
                    frames.append(
                        torch.from_numpy(np.array(frame.convert("RGB"), copy=True)).permute(2, 0, 1)
                    )

            clips.append(torch.stack(frames, dim=1))
        with torch.inference_mode():
            features = model(
                torch.stack(clips).to(device), rescale=True, resize=True, return_features=True
            )
        rows.append(features.detach().cpu().numpy())
    return _features(np.concatenate(rows), len(manifest.samples), 400)


def population_frechet_distance(real, generated):
    """Use population covariance (ddof=0) for the declared StyleGAN-V convention."""
    real, generated = np.asarray(real, np.float64), np.asarray(generated, np.float64)
    if (
        real.ndim != 2
        or generated.ndim != 2
        or real.shape[1] != generated.shape[1]
        or min(len(real), len(generated)) < 2
        or not np.isfinite(real).all()
        or not np.isfinite(generated).all()
    ):
        raise ValueError("Need matching finite feature matrices and >=2 clips")
    mean_r, mean_g = real.mean(0), generated.mean(0)
    centered_r, centered_g = real - mean_r, generated - mean_g
    cov_r, cov_g = centered_r.T @ centered_r / len(real), centered_g.T @ centered_g / len(generated)
    values, vectors = np.linalg.eigh((cov_r + cov_r.T) / 2)
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    middle = root @ cov_g @ root
    distance = (
        np.square(mean_r - mean_g).sum()
        + np.trace(cov_r)
        + np.trace(cov_g)
        - 2 * np.sqrt(np.maximum(np.linalg.eigvalsh((middle + middle.T) / 2), 0)).sum()
    )
    if not math.isfinite(distance) or distance < -1e-5:
        raise FloatingPointError("Invalid numerical distribution distance")
    return max(float(distance), 0.0)


def evaluate_media_directories(
    protocol,
    reference_root,
    generated_root,
    *,
    source_root,
    weights_path,
    grant,
    output_directory,
    device="cpu",
):

    grant.require(protocol, "official_evaluator")
    grant.require(protocol, "torchscript_execution")
    reference, generated = MediaManifest.load(reference_root), MediaManifest.load(generated_root)
    if (
        reference.id != protocol.reference_manifest_id
        or generated.cohort_id != protocol.generated_cohort_id
        or generated.expected_ids != protocol.expected_generated_ids
    ):
        raise ValueError("Media datasets differ from fixed distribution protocol")
    kind = "images" if protocol.extractor.provider == "cleanfid_inception" else "video_frames"
    if (
        reference.kind != kind
        or generated.kind != kind
        or min(len(reference.samples), len(generated.samples)) < 2
    ):
        raise ValueError("Distribution metric requires >=2 samples of the expected media kind")
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "protocol_id": protocol.id,
        "protocol": protocol.to_dict(),
        "status": "error",
        "metrics": {},
        "error": None,
        "reference_manifest_id": reference.id,
        "generated_manifest_id": generated.id,
        "producer_artifacts": list(generated.producer_artifacts),
        "expected_reference_samples": len(reference.samples),
        "expected_generated_samples": len(generated.samples),
        "failed_reference_ids": [s.id for s in reference.samples if s.status != "ok"],
        "failed_generated_ids": [s.id for s in generated.samples if s.status != "ok"],
        "reference_ids": list(reference.expected_ids),
        "generated_ids": list(generated.expected_ids),
        "feature_files": {},
        "environment": runtime_environment(device),
        "uncertainty": None,
        "generation": None,
        "provenance_verification": "caller_pinned_bytes_not_independent_official_origin_certification",
        "benchmark_claim": "metric_on_declared_cohort_not_automatic_public_leaderboard_submission",
    }
    started = time.monotonic()
    try:
        reference.verify(reference_root)
        generated.verify(generated_root)
        report["generation"] = _generation_record(generated_root, generated)
        if kind == "video_frames" and any(
            sample.frame_indices != protocol.frame_indices or sample.fps != protocol.fps
            for manifest in (reference, generated)
            for sample in manifest.samples
        ):
            raise ValueError("Video selection/FPS mismatch must fail before loading executable I3D")
        protocol.extractor.verify(source_root, weights_path)
        grant.require(protocol, "official_evaluator")
        grant.require(protocol, "torchscript_execution")
        if kind == "images":
            fid = importlib.import_module("cleanfid.fid")
            inception = importlib.import_module("cleanfid.inception_torchscript")
            for module in (fid, inception):
                if Path(module.__file__).resolve().parent != Path(source_root).resolve():
                    raise ValueError(
                        "Imported clean-fid module differs from pinned source directory"
                    )

            model = (
                inception.InceptionV3W(
                    str(Path(weights_path).parent), download=False, resize_inside=False
                )
                .eval()
                .to(device)
            )
            real = _clean_features(reference_root, reference, model, fid, protocol, device)
            fake = _clean_features(generated_root, generated, model, fid, protocol, device)
            scores = {}
            if "fid_clean" in protocol.metrics:
                scores["fid_clean"] = float(fid.fid_from_feats(real, fake))
            if "kid_clean" in protocol.metrics:
                with _NUMPY_RANDOM_LOCK:
                    state = np.random.get_state()
                    try:
                        np.random.seed(protocol.kid_seed)
                        scores["kid_clean"] = float(
                            fid.kernel_distance(
                                real,
                                fake,
                                num_subsets=protocol.kid_subsets,
                                max_subset_size=protocol.kid_subset_size,
                            )
                        )
                    finally:
                        np.random.set_state(state)
        else:
            model = torch.jit.load(str(weights_path), map_location=device).eval()
            real = _video_features(reference_root, reference, model, protocol, device)
            fake = _video_features(generated_root, generated, model, protocol, device)
            scores = {"fvd_styleganv_i3d": population_frechet_distance(real, fake)}
        if not all(math.isfinite(value) for value in scores.values()):
            raise FloatingPointError("Non-finite metric cannot enter a promotion report")
        reference.verify(reference_root)
        generated.verify(generated_root)
        protocol.extractor.verify(source_root, weights_path)
        grant.require(protocol, "official_evaluator")
        for name, features in (("reference", real), ("generated", fake)):
            path = root / (name + ".features.npy")
            np.save(path, features, allow_pickle=False)
            report["feature_files"][path.name] = {
                "sha256": file_digest(path),
                "shape": list(features.shape),
                "dtype": str(features.dtype),
            }
        report["metrics"] = {
            name: {"value": value, "higher_is_better": False, "unit": "distance", "ci": None}
            for name, value in scores.items()
        }
        report["actual_reference_samples"], report["actual_generated_samples"] = (
            len(real),
            len(fake),
        )
        report["status"] = "ok"
    except Exception as error:
        report["metrics"], report["error"] = {}, type(error).__name__
    report["evaluation_wall_seconds_including_io"] = time.monotonic() - started
    atomic_json(root / "report.json", {"report_id": digest_json(report), "report": report})
    return report
