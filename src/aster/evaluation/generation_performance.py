"""Synchronized sampler timing, actual model-call counts, and CUDA memory measurements."""

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import platform
import threading
import time

import torch

from ..core import atomic_json, digest_json, file_digest
from ..methods.generation import sample_diffusion, sample_flow
from ..models.generative import UNet2D, DiT, AutoencoderKL
from .generation_artifacts import load_native_artifact_model, resolve_image_sampling
from .generative import ImageSamplingPlan, runtime_environment


_BENCHMARK_LOCK = threading.Lock()


@dataclass(frozen=True)
class GenerationBenchmarkSettings:
    warmup_repetitions: int = 1
    repetitions: int = 5
    isolated_hardware_asserted: bool = False

    def __post_init__(self):
        if (
            type(self.warmup_repetitions) is not int
            or self.warmup_repetitions < 1
            or type(self.repetitions) is not int
            or self.repetitions < 2
        ):
            raise ValueError("Generation benchmarks need >=1 warmup and >=2 measured repetitions")
        if type(self.isolated_hardware_asserted) is not bool:
            raise ValueError(
                "Hardware isolation is an explicit host assertion, not inferred from a fast measurement"
            )


def expected_nfe(plan):

    from .drifting_generation import DriftingSamplingPlan
    from .interval_generation import MeanFlowSamplingPlan, ShortcutSamplingPlan, interval_nfe
    from .consistency_generation import ConsistencySamplingPlan
    from .edm_generation import EDMSamplingPlan, edm_nfe

    if isinstance(plan, EDMSamplingPlan):
        return edm_nfe(plan)
    if isinstance(plan, ConsistencySamplingPlan):
        return len(plan.sigmas)
    if isinstance(plan, DriftingSamplingPlan):
        return 1
    if isinstance(plan, (MeanFlowSamplingPlan, ShortcutSamplingPlan)):
        return interval_nfe(plan)
    calls = {"flow_euler": 1, "flow_heun": 2, "flow_rk4": 4, "ddim": 1, "ddpm": 1, "direct_x0": 1}[
        plan.sampler
    ]
    return calls * plan.steps * (1 if plan.guidance_scale == 1 else 2)


def _prepare(store, policy_id, plan, decoder_id, device):
    from .drifting_generation import DriftingSamplingPlan, _binding
    from ..models.drifting import DriftingGenerator
    from .interval_generation import MeanFlowSamplingPlan, ShortcutSamplingPlan, interval_binding
    from .consistency_generation import ConsistencySamplingPlan, consistency_binding
    from .edm_generation import EDMSamplingPlan, edm_binding

    artifact = store.get(policy_id, verify=True)
    model, layout = load_native_artifact_model(artifact)
    schedule = None
    if isinstance(plan, EDMSamplingPlan):
        binding = edm_binding(artifact, model, layout, plan)
        if (
            binding["required_decoder_artifact_id"] is not None
            and decoder_id != binding["required_decoder_artifact_id"]
        ):
            raise ValueError("EDM benchmark decoder differs from the trained encoder artifact")
    elif isinstance(plan, ConsistencySamplingPlan):
        binding = consistency_binding(artifact, model, layout, plan)
    elif isinstance(plan, DriftingSamplingPlan):
        if type(model) is not DriftingGenerator:
            raise ValueError("Drifting performance needs its native generator")
        binding = _binding(artifact, model, layout, plan)
    elif isinstance(plan, (MeanFlowSamplingPlan, ShortcutSamplingPlan)):
        binding = interval_binding(artifact, model, layout, plan)
    elif isinstance(plan, ImageSamplingPlan):
        if type(model) not in {UNet2D, DiT}:
            raise ValueError("Unsupported native image benchmark architecture")
        schedule, binding = resolve_image_sampling(artifact, model, layout, plan)
    else:
        raise ValueError("Only implemented native image generation sampling plans are accepted")
    decoder, decoder_binding = None, None
    if decoder_id is not None:
        decoder, decoder_path = load_native_artifact_model(store.get(decoder_id, verify=True))
        if type(decoder) is not AutoencoderKL:
            raise ValueError("Benchmark decoder must be the pinned native KL-VAE")
        if isinstance(
            plan,
            (
                DriftingSamplingPlan,
                MeanFlowSamplingPlan,
                ShortcutSamplingPlan,
                ConsistencySamplingPlan,
                EDMSamplingPlan,
            ),
        ) and any(p.dtype != torch.float32 for p in decoder.parameters()):
            raise ValueError("This native producer requires FP32 stored decoder weights")
        decoder = decoder.eval().to(device)
        decoder_binding = {"artifact_id": decoder_id, "model_relative_path": decoder_path}
    binding["decoder"] = decoder_binding
    return model.eval().to(device), decoder, schedule, binding


def _inputs(model, plan, case, device, binding):
    from .drifting_generation import DriftingSamplingPlan
    from .interval_generation import MeanFlowSamplingPlan, ShortcutSamplingPlan
    from .consistency_generation import ConsistencySamplingPlan, consistency_condition
    from .edm_generation import EDMSamplingPlan, edm_condition

    generator = torch.Generator(device=device).manual_seed(case.seed)
    dtype = next(model.parameters()).dtype
    noise = torch.randn((1, *plan.noise_shape), generator=generator, device=device, dtype=dtype)
    if isinstance(plan, EDMSamplingPlan):
        condition = edm_condition(model, case, device)
    elif isinstance(plan, ConsistencySamplingPlan):
        condition = consistency_condition(model, case, device)
    elif isinstance(plan, DriftingSamplingPlan):
        noise = noise * plan.temperature
        if case.condition >= binding["generation_contract"]["training"]["settings"]["num_classes"]:
            raise ValueError("Class is outside the declared Drifting training classes")
        labels = torch.tensor([case.condition], dtype=torch.int64, device=device)
        condition = (
            labels
            if not model.config.noise_classes
            else {
                "labels": labels,
                "noise_labels": torch.randint(
                    model.config.noise_classes,
                    (1, model.config.noise_coords),
                    generator=generator,
                    device=device,
                ),
            }
        )
    elif isinstance(plan, (MeanFlowSamplingPlan, ShortcutSamplingPlan)):
        if case.condition >= model.config.num_classes:
            raise ValueError("Class is outside the trained interval label range")
        condition = torch.tensor([case.condition], dtype=torch.int64, device=device)
    elif type(case.condition) is int:
        condition = torch.tensor([case.condition], dtype=torch.int64, device=device)
    elif case.condition is None:
        condition = None
    else:
        condition = torch.tensor([case.condition], dtype=dtype, device=device)
    return noise, condition, generator


def _run(model, decoder, schedule, binding, plan, noise, condition, generator):
    from .drifting_generation import DriftingSamplingPlan
    from .interval_generation import MeanFlowSamplingPlan, ShortcutSamplingPlan, sample_interval
    from .consistency_generation import ConsistencySamplingPlan, sample_consistency_plan
    from .edm_generation import EDMSamplingPlan, sample_edm_plan

    if isinstance(plan, EDMSamplingPlan):
        output = sample_edm_plan(model, plan, noise, condition, generator, binding)
    elif isinstance(plan, ConsistencySamplingPlan):
        output = sample_consistency_plan(model, plan, noise, condition, generator, binding)
    elif isinstance(plan, DriftingSamplingPlan):
        value = model(noise, noise.new_tensor([plan.cfg_scale]), condition)
        if value.prediction_type != "x0":
            raise ValueError("Drifting benchmark requires x0")
        output = value.prediction
    elif isinstance(plan, (MeanFlowSamplingPlan, ShortcutSamplingPlan)):
        output = sample_interval(model, plan, noise, condition)
    elif plan.sampler.startswith("flow_"):
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
            noise, noise.new_tensor([binding["generation_contract"]["generator_time"]]), condition
        )
        if value.prediction_type != "x0":
            raise ValueError("DMD benchmark requires x0")
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
    return output if decoder is None else decoder.decode(output, scaled=True)


def _native_sources(plan):

    from .drifting_generation import DriftingSamplingPlan, _sources as drifting_sources
    from .interval_generation import (
        MeanFlowSamplingPlan,
        ShortcutSamplingPlan,
        _sources as interval_sources,
    )
    from .consistency_generation import ConsistencySamplingPlan, _sources as consistency_sources
    from .edm_generation import EDMSamplingPlan, _sources as edm_sources
    from .generative import _producer_sources

    if isinstance(plan, DriftingSamplingPlan):
        return drifting_sources()
    if isinstance(plan, EDMSamplingPlan):
        return edm_sources()
    if isinstance(plan, ConsistencySamplingPlan):
        return consistency_sources()
    if isinstance(plan, (MeanFlowSamplingPlan, ShortcutSamplingPlan)):
        return interval_sources()
    if isinstance(plan, ImageSamplingPlan):
        return _producer_sources()
    raise ValueError("Unsupported native producer provenance")


def benchmark_image_sampler(
    store,
    policy_artifact_id,
    plan,
    settings,
    output_directory,
    *,
    decoder_artifact_id=None,
    device="cpu",
):

    if not isinstance(settings, GenerationBenchmarkSettings):
        raise ValueError("Typed benchmark settings are required")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU/CUDA benchmark timing is implemented")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA hardware is unavailable")
    if not _BENCHMARK_LOCK.acquire(blocking=False):
        raise RuntimeError("Concurrent benchmark would invalidate allocator/timing observations")
    try:
        model, decoder, schedule, binding = _prepare(
            store, policy_artifact_id, plan, decoder_artifact_id, device
        )
        root = Path(output_directory).absolute()
        root.mkdir(parents=True, exist_ok=False)
        environment = runtime_environment(device)
        environment.update(
            host_fingerprint=digest_json(platform.node()),
            cpu_count=os.cpu_count(),
            torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
        )
        report = {
            "schema_version": 1,
            "kind": "native_generation_performance",
            "status": "ok",
            "candidate_artifact_id": policy_artifact_id,
            "cohort_id": plan.cohort_id,
            "plan_id": plan.id,
            "plan": asdict(plan),
            "expected_ids": [case.id for case in plan.cases],
            "settings": asdict(settings),
            "sampling_binding": binding,
            "environment": environment,
            "native_producer_sources": _native_sources(plan),
            "measurement_source_sha256": file_digest(Path(__file__)),
            "latency_scope": "synchronized_sampler_plus_optional_vae_no_loading_rng_creation_or_image_io",
            "nfe_scope": "actual_native_field_forward_calls_excludes_vae_decode",
            "memory_scope": "cuda_peak_allocated_absolute_includes_resident_model"
            if device.type == "cuda"
            else "cpu_torch_allocator_peak_unavailable",
            "qualification": "host_asserted_isolation"
            if settings.isolated_hardware_asserted
            else "development_not_promotion_evidence",
            "warmups": [],
            "records": [],
        }
        counter = [0]
        hook = model.register_forward_pre_hook(lambda *_: counter.__setitem__(0, counter[0] + 1))
        try:
            with torch.no_grad():
                for measured, repetitions in (
                    (False, settings.warmup_repetitions),
                    (True, settings.repetitions),
                ):
                    for repetition in range(repetitions):
                        for case in plan.cases:
                            row = {
                                "sample_id": case.id,
                                "repetition": repetition,
                                "status": "error",
                                "error": None,
                                "latency_seconds": None,
                                "nfe": None,
                                "cuda_peak_allocated_bytes": None,
                            }
                            try:
                                noise, condition, generator = _inputs(
                                    model, plan, case, device, binding
                                )
                                counter[0] = 0
                                if device.type == "cuda":
                                    torch.cuda.synchronize(device)
                                    torch.cuda.reset_peak_memory_stats(device)
                                start = time.perf_counter_ns()
                                output = _run(
                                    model,
                                    decoder,
                                    schedule,
                                    binding,
                                    plan,
                                    noise,
                                    condition,
                                    generator,
                                )
                                if device.type == "cuda":
                                    torch.cuda.synchronize(device)
                                elapsed = (time.perf_counter_ns() - start) / 1e9
                                peak = (
                                    int(torch.cuda.max_memory_allocated(device))
                                    if device.type == "cuda"
                                    else None
                                )
                                if (
                                    output.ndim != 4
                                    or output.shape[0] != 1
                                    or output.shape[1] != 3
                                    or not torch.isfinite(output).all()
                                ):
                                    raise ValueError(
                                        "Benchmark must produce the same finite RGB output contract as image evaluation"
                                    )
                                if counter[0] != expected_nfe(plan) or elapsed <= 0:
                                    raise ValueError("Unexpected actual forward count/timer result")
                                row.update(
                                    status="ok",
                                    latency_seconds=elapsed,
                                    nfe=counter[0],
                                    cuda_peak_allocated_bytes=peak,
                                    output_shape=list(output.shape),
                                    output_sha256=hashlib.sha256(
                                        output.float().cpu().contiguous().numpy().tobytes()
                                    ).hexdigest(),
                                )

                                del output, noise, condition, generator
                            except Exception as error:
                                row["error"] = type(error).__name__
                                report["status"] = "error"
                            report["records" if measured else "warmups"].append(row)
        finally:
            hook.remove()
        store.get(policy_artifact_id, verify=True)
        if decoder_artifact_id is not None:
            store.get(decoder_artifact_id, verify=True)
        if _native_sources(plan) != report["native_producer_sources"]:
            raise RuntimeError("Native producer source changed during measurement")
        atomic_json(root / "report.json", {"report_id": digest_json(report), "report": report})
        return report
    finally:
        _BENCHMARK_LOCK.release()
