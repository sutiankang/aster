"""Wan cache calibration and paired error, FVD, and runtime comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ..core import atomic_json, digest_json, file_digest, read_json
from ..models import load_model
from ..models.video_world import WanVideoDiT
from ..models.video_vae import WanVideoVAE
from ..optimization.wan_teacache import (
    WanCacheSampler,
    WanCacheCalibration,
    WanTeaCacheSettings,
    WanTeaCacheSession,
    calibrate_wan_teacache,
    run_wan_teacache_session,
    tensor_fingerprint,
)
from .video_generation import VideoConditionBundle, VideoSamplingPlan, _plan
from .generative import (
    MediaManifest,
    MediaSample,
    ImageFile,
    _under,
    _image_identity,
    quantize_image,
    runtime_environment,
    DistributionProtocol,
    evaluate_media_directories,
)
from .generation_performance import GenerationBenchmarkSettings, _BENCHMARK_LOCK
from .generation_gate import complete_cohort_interval


def _sources():
    base = Path(__file__).resolve().parents[1]
    return {
        name: file_digest(base / name)
        for name in (
            "evaluation/wan_teacache.py",
            "evaluation/video_generation.py",
            "evaluation/generative.py",
            "evaluation/generation_gate.py",
            "evaluation/generation_performance.py",
            "optimization/wan_teacache.py",
            "models/video_world.py",
            "models/video_vae.py",
            "models/serialization.py",
            "models/config.py",
            "models/__init__.py",
        )
    }


def _environment(device):
    import platform
    import socket

    return {
        **runtime_environment(device),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "processor": platform.processor(),
        "host_identity_hash": digest_json(socket.gethostname()),
    }


def _envelope(path, report):
    atomic_json(path, {"report_id": digest_json(report), "report": report})


def _read_envelope(path):
    value = read_json(path)
    if set(value) != {"report_id", "report"} or digest_json(value["report"]) != value["report_id"]:
        raise ValueError("Wan cache record bytes/schema identity mismatch")
    return value["report"]


def _sampler(plan):
    return WanCacheSampler(plan.steps, plan.solver, plan.shift, plan.guidance_scale)


def _load(store, artifact_id, device):
    artifact = store.get(artifact_id, verify=True)

    with torch.random.fork_rng(devices=[]):
        model = load_model(artifact.path).eval().to(device)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Wan cache bridge currently requires FP32 storage")
    return model


def publish_wan_cache_calibration(
    store, field_artifact_id, plan, directory, *, mode="default", degree=2, ridge=1e-6, device="cpu"
):

    if not isinstance(plan, VideoSamplingPlan):
        raise TypeError("A fixed native video plan is required")
    sources = _sources()
    model = _load(store, field_artifact_id, device)
    if type(model) is not WanVideoDiT:
        raise ValueError("Only native Wan2.1 has this verified cache adapter")
    conditions = VideoConditionBundle(store, plan.condition_artifact_id)

    def cases():
        for case in plan.cases:
            yield {
                "id": case.id,
                "noise": torch.randn(
                    (1, *plan.latent_shape),
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(case.seed),
                ),
                **conditions.load_case(case.condition_key, device=device),
            }

    calibration = calibrate_wan_teacache(
        model,
        cases(),
        policy_artifact_id=field_artifact_id,
        dataset_fingerprint=plan.id,
        sampler=_sampler(plan),
        mode=mode,
        degree=degree,
        ridge=ridge,
    )
    if sources != _sources():
        raise ValueError("Native calibration sources changed while executing")
    store.get(field_artifact_id, verify=True)
    store.get(plan.condition_artifact_id, verify=True)
    report = {
        "schema_version": 1,
        "calibration": calibration.to_dict(),
        "calibration_id": calibration.id,
        "plan": asdict(plan),
        "plan_id": plan.id,
        "fit_degree": degree,
        "fit_ridge": ridge,
        "native_producer_sources": sources,
        "environment": _environment(device),
        "condition_provenance": conditions.provenance,
        "official_coefficients_used": False,
    }
    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    _envelope(root / "calibration.json", report)
    return store.publish(
        root,
        kind="native_wan21_teacache_calibration",
        metadata={"report_id": digest_json(report)},
        parents=(field_artifact_id, plan.condition_artifact_id),
    )


def load_wan_cache_calibration(store, artifact_id):
    artifact = store.get(artifact_id, verify=True)
    report = _read_envelope(artifact.path / "calibration.json")
    if artifact.kind != "native_wan21_teacache_calibration" or artifact.metadata.get(
        "report_id"
    ) != digest_json(report):
        raise ValueError("Not an intact native Wan measured calibration artifact")
    fitted, plan = WanCacheCalibration.from_dict(report["calibration"]), _plan(report["plan"])
    if (
        fitted.id != report["calibration_id"]
        or fitted.dataset_fingerprint != plan.id
        or report["plan_id"] != plan.id
        or fitted.sampler != _sampler(plan)
        or fitted.latent_shape != (1, *plan.latent_shape)
        or fitted.case_ids != tuple(c.id for c in plan.cases)
        or report["official_coefficients_used"] is not False
        or artifact.parents != (fitted.policy_artifact_id, plan.condition_artifact_id)
        or report["native_producer_sources"] != _sources()
    ):
        raise ValueError("Calibration policy/data/sampler/source identity changed")

    degree, ridge = report["fit_degree"], report["fit_ridge"]
    if (
        type(degree) is not int
        or not 0 <= degree <= 4
        or type(ridge) not in {int, float}
        or not math.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("Invalid recorded calibration fit controls")
    x = torch.tensor([v["probe_distance"] for v in fitted.measurements], dtype=torch.float64)
    y = torch.tensor([v["residual_distance"] for v in fitted.measurements], dtype=torch.float64)
    design = torch.vander(x, N=degree + 1)
    actual = torch.linalg.solve(
        design.T @ design + ridge * torch.eye(degree + 1, dtype=x.dtype), design.T @ y
    )
    if tuple(actual.tolist()) != fitted.coefficients:
        raise ValueError("Calibration coefficients do not match actual recorded fit")
    for parent in artifact.parents:
        store.get(parent, verify=True)
    return fitted, report


def _prepare(store, field_id, vae_id, plan, calibration_id, cache_settings, device):
    if not isinstance(plan, VideoSamplingPlan) or (calibration_id is None) != (
        cache_settings is None
    ):
        raise ValueError(
            "Provide fixed video plan and either full reference or calibration plus cache settings"
        )
    if cache_settings is not None and not isinstance(cache_settings, WanTeaCacheSettings):
        raise TypeError("Typed cache settings required")
    field, vae = _load(store, field_id, device), _load(store, vae_id, device)
    if type(field) is not WanVideoDiT or type(vae) is not WanVideoVAE:
        raise ValueError("Only native Wan2.1 field/VAE supported")
    shape = (
        1 + (plan.latent_shape[1] - 1) * vae.config.temporal_stride,
        plan.latent_shape[2] * vae.config.spatial_stride,
        plan.latent_shape[3] * vae.config.spatial_stride,
    )
    if (
        plan.output_shape != shape
        or field.config.latent_channels != vae.config.latent_channels
        or plan.latent_shape[0] != field.config.latent_channels
    ):
        raise ValueError("Native cache field/VAE geometry mismatch")
    if any(n % p for n, p in zip(plan.latent_shape[1:], field.config.patch_size)):
        raise ValueError("Invalid native Wan patch geometry")
    conditions = VideoConditionBundle(store, plan.condition_artifact_id)
    if any(case.condition_key not in conditions.entries for case in plan.cases):
        raise ValueError("Missing planned condition key")
    calibration, evidence = (
        (None, None)
        if calibration_id is None
        else load_wan_cache_calibration(store, calibration_id)
    )
    if calibration is not None and (
        calibration.policy_artifact_id != field_id
        or calibration.sampler != _sampler(plan)
        or calibration.mode != cache_settings.mode
    ):
        raise ValueError("Cache calibration model/solver/clock/probe mismatch")
    if evidence is not None and evidence["environment"] != _environment(device):
        raise ValueError(
            "Calibration numeric execution environment differs; recalibrate on this target"
        )
    binding = {
        "field_artifact_id": field_id,
        "vae_artifact_id": vae_id,
        "condition_artifact_id": plan.condition_artifact_id,
        "calibration_artifact_id": calibration_id,
        "cache_settings": asdict(cache_settings) if cache_settings else None,
        "calibration_id": calibration.id if calibration is not None else None,
        "calibration_plan": evidence["plan"] if evidence else None,
        "condition_provenance": conditions.provenance,
        "method": "approximate_residual_cache" if calibration_id else "full_native_reference",
        "scope": "batch_one_FP32_native_Wan21_Euler_Heun_not_official_UniPC_or_DPMpp",
    }
    return field, vae, conditions, calibration, binding


def _session(field, conditions, case, plan, calibration, settings, policy_id, device):
    branches = conditions.load_case(case.condition_key, device=device)
    session = WanTeaCacheSession(
        field,
        policy_artifact_id=policy_id,
        sampler=_sampler(plan),
        condition=branches["positive"],
        negative_condition=branches.get("negative"),
        calibration=calibration,
        settings=settings,
    )
    noise = torch.randn(
        (1, *plan.latent_shape),
        device=device,
        generator=torch.Generator(device=device).manual_seed(case.seed),
    )
    return session, noise


def _check_video(video, plan):
    if (
        video.shape != (1, 3, *plan.output_shape)
        or video.dtype != torch.float32
        or not torch.isfinite(video).all()
    ):
        raise ValueError("Native Wan VAE output differs from declared finite FP32 RGB geometry")


def generate_wan_cache_cohort(
    store,
    field_artifact_id,
    vae_artifact_id,
    plan,
    directory,
    *,
    calibration_artifact_id=None,
    cache_settings=None,
    device="cpu",
):

    sources = _sources()
    field, vae, conditions, calibration, binding = _prepare(
        store,
        field_artifact_id,
        vae_artifact_id,
        plan,
        calibration_artifact_id,
        cache_settings,
        device,
    )
    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    samples, records = [], []
    for index, case in enumerate(plan.cases):
        session = None
        created = []
        try:
            session, noise = _session(
                field,
                conditions,
                case,
                plan,
                calibration,
                cache_settings,
                field_artifact_id,
                device,
            )
            with torch.inference_mode():
                video = vae.decode(run_wan_teacache_session(session, noise), scaled=True)
            _check_video(video, plan)
            payloads = []
            for frame in plan.frame_indices:
                stream = io.BytesIO()
                Image.fromarray(quantize_image(video[0, :, frame], plan.quantization)).save(
                    stream, format="PNG"
                )
                payloads.append((f"{index:08d}-f{frame:06d}.png", stream.getvalue()))
            stream = io.BytesIO()
            torch.save({"video": video.cpu()}, stream)
            raw_name = f"{index:08d}-raw.pt"
            payloads.append((raw_name, stream.getvalue()))
            for name, data in payloads:
                path = root / name
                with path.open("xb") as out:
                    out.write(data)
                created.append(path)
            files = tuple(
                ImageFile(name, *_image_identity(root / name)) for name, _ in payloads[:-1]
            )
            samples.append(
                MediaSample(
                    case.id, files, seed=case.seed, frame_indices=plan.frame_indices, fps=plan.fps
                )
            )
            records.append(
                {
                    "id": case.id,
                    "status": "ok",
                    "error": None,
                    "raw_path": raw_name,
                    "raw_sha256": file_digest(root / raw_name),
                    "video_fingerprint": tensor_fingerprint(video),
                    "observation": session.observation(),
                }
            )
        except Exception as error:
            for path in created:
                path.unlink()
            samples.append(
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
            records.append(
                {
                    "id": case.id,
                    "status": "error",
                    "error": type(error).__name__,
                    "raw_path": None,
                    "raw_sha256": None,
                    "video_fingerprint": None,
                    "observation": session.observation() if session else None,
                }
            )
        finally:
            if session is not None:
                session.close()
    parents = (field_artifact_id, vae_artifact_id, plan.condition_artifact_id) + (
        (calibration_artifact_id,) if calibration_artifact_id else ()
    )
    manifest = MediaManifest(
        "video_frames",
        "native_wan21_teacache",
        plan.id,
        "generation",
        "producer_artifact_terms",
        plan.cohort_id,
        tuple(c.id for c in plan.cases),
        tuple(samples),
        parents,
    )
    manifest.save(root)
    if sources != _sources():
        raise ValueError("Native Wan producer sources changed during generation")
    for parent in parents:
        store.get(parent, verify=True)
    report = {
        "schema_version": 1,
        "plan": asdict(plan),
        "plan_id": plan.id,
        "manifest_id": manifest.id,
        "binding": binding,
        "binding_id": digest_json(binding),
        "native_producer_sources": sources,
        "environment": _environment(device),
        "records": records,
    }
    _envelope(root / "wan_cache_generation.json", report)
    return manifest


def wan_cache_generation_record(directory, manifest=None):
    root = Path(directory)
    report = _read_envelope(root / "wan_cache_generation.json")
    plan = _plan(report["plan"])
    actual = MediaManifest.load(root).verify(root, require_complete=False)
    if manifest is not None and manifest.id != actual.id:
        raise ValueError("Mismatched Wan cache media manifest")
    manifest = actual
    binding = report["binding"]
    parents = (
        binding["field_artifact_id"],
        binding["vae_artifact_id"],
        plan.condition_artifact_id,
    ) + ((binding["calibration_artifact_id"],) if binding["calibration_artifact_id"] else ())
    if (
        manifest.id != report["manifest_id"]
        or manifest.dataset_id != "native_wan21_teacache"
        or manifest.revision != plan.id
        or report["plan_id"] != plan.id
        or manifest.cohort_id != plan.cohort_id
        or report["binding_id"] != digest_json(binding)
        or manifest.producer_artifacts != parents
        or binding["condition_artifact_id"] != plan.condition_artifact_id
        or manifest.expected_ids != tuple(c.id for c in plan.cases)
        or [row["id"] for row in report["records"]] != list(manifest.expected_ids)
    ):
        raise ValueError("Wan cache plan/lineage/full population identity differs")
    for row, sample, case in zip(report["records"], manifest.samples, plan.cases):
        if (
            row["status"] != sample.status
            or sample.seed != case.seed
            or sample.fps != plan.fps
            or sample.frame_indices != plan.frame_indices
        ):
            raise ValueError("Wan cache sample metadata differs from fixed plan")
        if row["status"] != "ok":
            continue
        raw_path = _under(root, row["raw_path"])
        if file_digest(raw_path) != row["raw_sha256"]:
            raise ValueError("Raw video bytes changed")
        data = torch.load(raw_path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or set(data) != {"video"}:
            raise ValueError("Unexpected raw video fields")
        video = data["video"]
        _check_video(video, plan)
        if tensor_fingerprint(video) != row["video_fingerprint"]:
            raise ValueError("Raw video numeric identity changed")
        for frame, image in zip(plan.frame_indices, sample.files):
            with Image.open(_under(root, image.path)) as opened:
                if not np.array_equal(
                    np.asarray(opened), quantize_image(video[0, :, frame], plan.quantization)
                ):
                    raise ValueError("PNG pixels do not quantize the measured raw video")
        observe = row["observation"]
        rounds = len(_sampler(plan).evaluation_times("cpu"))
        branches = _sampler(plan).branches
        expected = [(i, b) for i in range(rounds) for b in branches]
        if (
            [(v["round"], v["branch"]) for v in observe["trace"]] != expected
            or observe["field_calls"] != len(expected)
            or observe["full_backbone_calls"] + observe["reused_backbone_calls"] != len(expected)
            or observe["reused_backbone_calls"] != sum(v["reused"] for v in observe["trace"])
            or observe["audit_backbone_calls"] != sum(v["audit"] for v in observe["trace"])
        ):
            raise ValueError("Cache trace/NFE differs from actual full sampling clock")
        if (
            observe["policy_artifact_id"] != binding["field_artifact_id"]
            or observe["settings"] != binding["cache_settings"]
            or observe["sampler"] != asdict(_sampler(plan))
            or observe["calibration_id"] != binding["calibration_id"]
        ):
            raise ValueError("Cache observation used a different policy/settings")
    return report


def benchmark_wan_cache_cohort(
    store,
    field_artifact_id,
    vae_artifact_id,
    plan,
    directory,
    *,
    calibration_artifact_id=None,
    cache_settings=None,
    settings=GenerationBenchmarkSettings(),
    device="cpu",
):
    sources = _sources()
    field, vae, conditions, calibration, binding = _prepare(
        store,
        field_artifact_id,
        vae_artifact_id,
        plan,
        calibration_artifact_id,
        cache_settings,
        device,
    )
    device = torch.device(device)
    trials, warmups = [], []
    with _BENCHMARK_LOCK:
        for warm, repetitions in (
            (True, settings.warmup_repetitions),
            (False, settings.repetitions),
        ):
            for repetition in range(repetitions):
                for case in plan.cases:
                    session = None
                    try:
                        session, noise = _session(
                            field,
                            conditions,
                            case,
                            plan,
                            calibration,
                            cache_settings,
                            field_artifact_id,
                            device,
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                            torch.cuda.reset_peak_memory_stats(device)
                        started = time.perf_counter()
                        with torch.inference_mode():
                            video = vae.decode(
                                run_wan_teacache_session(session, noise), scaled=True
                            )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        elapsed = time.perf_counter() - started
                        memory = (
                            torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda"
                            else None
                        )
                        _check_video(video, plan)
                        row = {
                            "id": case.id,
                            "repetition": repetition,
                            "status": "ok",
                            "error": None,
                            "latency_seconds": elapsed,
                            "cuda_peak_allocated_bytes": memory,
                            "video_fingerprint": tensor_fingerprint(video),
                            "observation": session.observation(),
                        }
                    except Exception as error:
                        row = {
                            "id": case.id,
                            "repetition": repetition,
                            "status": "error",
                            "error": type(error).__name__,
                            "latency_seconds": None,
                            "cuda_peak_allocated_bytes": None,
                            "video_fingerprint": None,
                            "observation": session.observation() if session else None,
                        }
                    finally:
                        if session is not None:
                            session.close()
                    (warmups if warm else trials).append(row)
    if sources != _sources():
        raise ValueError("Performance sources changed during measurement")
    report = {
        "schema_version": 1,
        "status": "ok" if all(r["status"] == "ok" for r in warmups + trials) else "error",
        "plan": asdict(plan),
        "plan_id": plan.id,
        "binding": binding,
        "binding_id": digest_json(binding),
        "native_producer_sources": sources,
        "environment": _environment(device),
        "settings": asdict(settings),
        "latency_scope": "synchronized_native_sampler_with_cache_audits_plus_VAE_excludes_loading_identity_scan_initial_RNG_and_IO",
        "memory_scope": "CUDA_allocator_absolute_peak_including_resident_weights_and_both_branch_caches_CPU_unavailable",
        "nfe_scope": "field_evaluations_and_full_backbone_calls_separately_not_solver_step_reduction",
        "warmups": warmups,
        "trials": trials,
    }
    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    _envelope(root / "benchmark.json", report)
    return report


@dataclass(frozen=True)
class WanCacheFVDResources:
    protocol: DistributionProtocol
    reference_root: str | Path
    source_root: str | Path
    weights_path: str | Path
    grant: object


def evaluate_wan_cache_fvd(directory, output_directory, *, resources=None, device="cpu"):
    report = wan_cache_generation_record(directory)
    plan = _plan(report["plan"])
    if resources is None:
        result = {
            "status": "not_evaluated",
            "metrics": {},
            "reason": "approved_pinned_official_I3D_source_weights_reference_and_grant_required",
            "cohort_id": plan.cohort_id,
            "generation_id": digest_json(report),
        }
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=False)
        _envelope(root / "report.json", result)
        return result
    if not isinstance(resources, WanCacheFVDResources):
        raise TypeError("Typed pinned official FVD resources required")
    protocol = resources.protocol
    if (
        protocol.generated_cohort_id != plan.cohort_id
        or protocol.expected_generated_ids != tuple(c.id for c in plan.cases)
        or protocol.metrics != ("fvd_styleganv_i3d",)
        or protocol.frame_indices != plan.frame_indices
        or protocol.fps != plan.fps
    ):
        raise ValueError(
            "Official FVD protocol changed the actual cache cohort/frame/FPS population"
        )
    return evaluate_media_directories(
        protocol,
        resources.reference_root,
        directory,
        source_root=resources.source_root,
        weights_path=resources.weights_path,
        grant=resources.grant,
        output_directory=output_directory,
        device=device,
    )


def _performance(path, generation):
    report = _read_envelope(Path(path) / "benchmark.json")
    for key in (
        "plan_id",
        "plan",
        "binding",
        "binding_id",
        "native_producer_sources",
        "environment",
    ):
        if report[key] != generation[key]:
            raise ValueError(
                "Performance did not execute the exact quality model/cohort/cache/source/environment"
            )
    if report["native_producer_sources"] != _sources():
        raise ValueError("Current native performance implementation changed")
    if (
        report["latency_scope"]
        != "synchronized_native_sampler_with_cache_audits_plus_VAE_excludes_loading_identity_scan_initial_RNG_and_IO"
    ):
        raise ValueError("Unrecognized performance timing boundary")
    if (
        report["memory_scope"]
        != "CUDA_allocator_absolute_peak_including_resident_weights_and_both_branch_caches_CPU_unavailable"
        or report["nfe_scope"]
        != "field_evaluations_and_full_backbone_calls_separately_not_solver_step_reduction"
    ):
        raise ValueError("Unrecognized memory/NFE measurement scope")
    settings = GenerationBenchmarkSettings(**report["settings"])
    plan = _plan(report["plan"])
    by_id = {r["id"]: r for r in generation["records"]}
    for name, repetitions in (
        ("warmups", settings.warmup_repetitions),
        ("trials", settings.repetitions),
    ):
        rows = report[name]
        if [(r["repetition"], r["id"]) for r in rows] != [
            (i, c.id) for i in range(repetitions) for c in plan.cases
        ]:
            raise ValueError("Incomplete/reordered performance repetition population")
        for row in rows:
            if row["status"] != "ok":
                continue
            if (
                row["video_fingerprint"] != by_id[row["id"]]["video_fingerprint"]
                or row["observation"] != by_id[row["id"]]["observation"]
            ):
                raise ValueError(
                    "Performance and quality used different actual output/cache decisions"
                )
            if (
                type(row["latency_seconds"]) not in {int, float}
                or not math.isfinite(row["latency_seconds"])
                or row["latency_seconds"] <= 0
            ):
                raise ValueError("Invalid measured native latency")
            peak = row["cuda_peak_allocated_bytes"]
            if peak is not None and (type(peak) is not int or peak <= 0):
                raise ValueError("Invalid actual allocator peak")
    return report


def compare_wan_cache_cohorts(
    pairs,
    directory,
    *,
    resources=None,
    maximum_pixel_rmse=0.02,
    maximum_fvd_regression=0.0,
    minimum_latency_improvement=0.05,
    maximum_memory_ratio=None,
    repetitions=1000,
    confidence=0.95,
    seed=0,
):

    if not pairs or resources is not None and len(resources) != len(pairs):
        raise ValueError("Explicit aligned cohort pairs/resources required")
    numeric = (maximum_pixel_rmse, maximum_fvd_regression, minimum_latency_improvement)
    if (
        any(type(v) not in {int, float} or not math.isfinite(v) or v < 0 for v in numeric)
        or not 0 < minimum_latency_improvement < 1
    ):
        raise ValueError(
            "Finite quality budgets and a strictly positive real resource improvement required"
        )
    if (
        type(repetitions) is not int
        or repetitions < 200
        or not 0 < confidence < 1
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        raise ValueError("Invalid full-cohort bootstrap controls")
    if maximum_memory_ratio is not None and (
        type(maximum_memory_ratio) not in {int, float}
        or not math.isfinite(maximum_memory_ratio)
        or maximum_memory_ratio <= 0
    ):
        raise ValueError("Invalid memory limit")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    errors, missing, rows, used_seeds = [], [], [], set()
    latency_base, latency_cache, fvd_base, fvd_cache = [], [], [], []
    common, official = None, None
    if len(pairs) < 3:
        missing.append("three_independent_complete_cohorts_required")
    for i, pair in enumerate(pairs):
        if set(pair) != {"baseline", "candidate", "baseline_benchmark", "candidate_benchmark"}:
            raise ValueError("Incomplete comparison paths")
        base, candidate = (wan_cache_generation_record(pair[k]) for k in ("baseline", "candidate"))
        plan = _plan(base["plan"])
        if any(
            base[k] != candidate[k]
            for k in ("plan_id", "plan", "native_producer_sources", "environment")
        ):
            raise ValueError("Cache comparison changed model cohort/software/solver")
        bb, cb = base["binding"], candidate["binding"]
        if (
            bb["method"] != "full_native_reference"
            or cb["method"] != "approximate_residual_cache"
            or any(
                bb[k] != cb[k]
                for k in (
                    "field_artifact_id",
                    "vae_artifact_id",
                    "condition_artifact_id",
                    "condition_provenance",
                )
            )
        ):
            raise ValueError(
                "Require same native field/VAE/conditions with cache as the only optimization"
            )
        current = (
            base["binding_id"],
            candidate["binding_id"],
            {k: v for k, v in base["plan"].items() if k != "cases"},
            base["native_producer_sources"],
            base["environment"],
        )
        if common is None:
            common = current
        elif current != common:
            raise ValueError("Multi-cohort comparison mixed cache/model/data/solver profiles")
        seeds = {c.seed for c in plan.cases}
        if len(seeds) != len(plan.cases) or seeds & used_seeds:
            raise ValueError("Cohorts must have distinct fixed generation seeds")
        used_seeds |= seeds
        calibration = _plan(cb["calibration_plan"])
        if seeds & {c.seed for c in calibration.cases} or {c.id for c in plan.cases} & {
            c.id for c in calibration.cases
        }:
            errors.append("quality_cohort_reuses_calibration_cases_or_seeds")
        failed = any(r["status"] != "ok" for record in (base, candidate) for r in record["records"])
        if failed:
            errors.append("failed_generation_in_full_population")
        rmse = []
        for a, b in zip(base["records"], candidate["records"]):
            if a["status"] != "ok" or b["status"] != "ok":
                continue
            one = torch.load(_under(Path(pair["baseline"]), a["raw_path"]), weights_only=True)[
                "video"
            ]
            two = torch.load(_under(Path(pair["candidate"]), b["raw_path"]), weights_only=True)[
                "video"
            ]
            rmse.append(float((one.double() - two.double()).square().mean().sqrt()))
            if b["observation"]["guard_failed"]:
                errors.append("cache_runtime_error_guard_failed")
            if b["observation"]["full_backbone_calls"] > a["observation"]["full_backbone_calls"]:
                errors.append("full_backbone_NFE_regression")
        if rmse and max(rmse) > maximum_pixel_rmse:
            errors.append("paired_pixel_RMSE_regression")
        performances = [
            _performance(pair[k], record)
            for k, record in (("baseline_benchmark", base), ("candidate_benchmark", candidate))
        ]
        if any(
            p["status"] != "ok" or any(r["status"] != "ok" for r in p["warmups"] + p["trials"])
            for p in performances
        ):
            errors.append("failed_performance_population")
        else:
            latency_base.append(
                float(np.mean([r["latency_seconds"] for r in performances[0]["trials"]]))
            )
            latency_cache.append(
                float(np.mean([r["latency_seconds"] for r in performances[1]["trials"]]))
            )
        if not all(p["settings"]["isolated_hardware_asserted"] for p in performances):
            missing.append("hardware_isolation_not_asserted")
        if maximum_memory_ratio is not None:
            m = [[r["cuda_peak_allocated_bytes"] for r in p["trials"]] for p in performances]
            if any(v is None for values in m for v in values):
                missing.append("real_CUDA_peak_unavailable")
            elif max(m[1]) > max(m[0]) * maximum_memory_ratio:
                errors.append("CUDA_memory_regression")
        resource = None if resources is None else resources[i]
        if resource is not None:
            signature = {
                k: v
                for k, v in resource.protocol.to_dict().items()
                if k not in {"generated_cohort_id", "expected_generated_ids"}
            }
            if official is None:
                official = signature
            elif official != signature:
                raise ValueError("FVD cohorts mixed reference/extractor/preprocessing")
        full_quality = evaluate_wan_cache_fvd(
            pair["baseline"], root / f"{i:04d}_baseline_fvd", resources=resource
        )
        cache_quality = evaluate_wan_cache_fvd(
            pair["candidate"], root / f"{i:04d}_candidate_fvd", resources=resource
        )
        if full_quality["status"] == cache_quality["status"] == "ok":
            fvd_base.append(full_quality["metrics"]["fvd_styleganv_i3d"]["value"])
            fvd_cache.append(cache_quality["metrics"]["fvd_styleganv_i3d"]["value"])
        else:
            missing.append("official_FVD_not_successfully_evaluated")
        rows.append(
            {
                "cohort_id": plan.cohort_id,
                "baseline_generation_id": digest_json(base),
                "candidate_generation_id": digest_json(candidate),
                "maximum_paired_pixel_rmse": max(rmse) if rmse else None,
                "baseline_fvd_id": digest_json(full_quality),
                "candidate_fvd_id": digest_json(cache_quality),
                "performance_ids": [digest_json(p) for p in performances],
            }
        )
    comparisons = {}
    if len(latency_base) == len(pairs) and len(pairs) >= 3:
        comparisons["latency"] = complete_cohort_interval(
            latency_base,
            latency_cache,
            relative=True,
            confidence=confidence,
            repetitions=repetitions,
            seed=seed,
        )
        if comparisons["latency"]["low"] < minimum_latency_improvement:
            errors.append("real_latency_improvement_not_demonstrated")
    if len(fvd_base) == len(pairs) and len(pairs) >= 3:
        comparisons["fvd"] = complete_cohort_interval(
            fvd_base, fvd_cache, confidence=confidence, repetitions=repetitions, seed=seed + 1
        )
        if comparisons["fvd"]["low"] < -maximum_fvd_regression:
            errors.append("official_FVD_regression")
    report = {
        "schema_version": 1,
        "status": "reject" if errors else ("not_evaluated" if missing else "promote"),
        "reasons": sorted(set(errors)),
        "unevaluated": sorted(set(missing)),
        "cohorts": rows,
        "comparison": comparisons,
        "controls": {
            "maximum_pixel_rmse": maximum_pixel_rmse,
            "maximum_fvd_regression": maximum_fvd_regression,
            "minimum_latency_improvement": minimum_latency_improvement,
            "maximum_memory_ratio": maximum_memory_ratio,
            "repetitions": repetitions,
            "confidence": confidence,
            "seed": seed,
        },
        "aggregation": "paired_independent_full_cohorts_not_per_video_FVD",
        "automatically_deployed": False,
    }
    _envelope(root / "gate.json", report)
    return report
