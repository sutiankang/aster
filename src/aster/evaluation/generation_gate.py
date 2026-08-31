"""Joint quality and performance gates over complete, predeclared generation cohorts."""

from dataclasses import asdict, dataclass
import importlib
import math
from pathlib import Path
import re

import numpy as np

from ..core import atomic_json, digest_json, file_digest, read_json
from .generative import DistributionProtocol, ExtractorPin, _NUMPY_RANDOM_LOCK
from .generation_performance import GenerationBenchmarkSettings, expected_nfe


class _MissingEvidence(Exception):
    pass


@dataclass(frozen=True)
class GenerationGateProtocol:
    baseline_artifact_id: str
    candidate_artifact_id: str
    cohort_ids: tuple[str, ...]
    quality_max_regression: tuple[tuple[str, float], ...] = (("fid_clean", 0.0), ("kid_clean", 0.0))
    resource_max_relative_regression: tuple[tuple[str, float], ...] = (
        ("latency_seconds", 0.05),
        ("nfe", 0.0),
    )
    required_relative_improvements: tuple[tuple[str, float], ...] = (("latency_seconds", 0.05),)
    confidence: float = 0.95
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "cohort_ids", tuple(self.cohort_ids))
        if any(
            not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
            for value in (self.baseline_artifact_id, self.candidate_artifact_id, *self.cohort_ids)
        ):
            raise ValueError("Gate model/cohort identities must be immutable SHA256 values")
        if not self.cohort_ids or len(set(self.cohort_ids)) != len(self.cohort_ids):
            raise ValueError("Unique fixed cohorts are required")
        resources = {"latency_seconds", "nfe", "cuda_peak_allocated_bytes"}
        for field, names in (
            ("quality_max_regression", {"fid_clean", "kid_clean"}),
            ("resource_max_relative_regression", resources),
            ("required_relative_improvements", resources),
        ):
            pairs = tuple(tuple(pair) for pair in getattr(self, field))
            object.__setattr__(self, field, pairs)
            if not pairs or any(len(pair) != 2 for pair in pairs):
                raise ValueError("Explicit metric thresholds are required")
            if len({pair[0] for pair in pairs}) != len(pairs) or any(
                name not in names
                or type(value) not in {float, int}
                or not math.isfinite(value)
                or value < 0
                for name, value in pairs
            ):
                raise ValueError("Invalid gate metrics/tolerances")
        if any(not 0 < value < 1 for name, value in self.required_relative_improvements):
            raise ValueError("A real strictly positive resource improvement is mandatory")
        if (
            type(self.confidence) not in {float, int}
            or not 0 < self.confidence < 1
            or type(self.bootstrap_repetitions) is not int
            or self.bootstrap_repetitions < 200
            or type(self.bootstrap_seed) is not int
            or not 0 <= self.bootstrap_seed < 2**32
        ):
            raise ValueError("Invalid cohort bootstrap controls")

    @property
    def id(self):
        return digest_json(asdict(self))


def complete_cohort_interval(
    baseline, candidate, *, relative=False, confidence=0.95, repetitions=2000, seed=0
):
    """Treat each input as an aggregate from an independent complete cohort,
    not as a per-image approximation to FID."""
    baseline, candidate = (
        np.asarray(baseline, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
    )
    if (
        baseline.ndim != 1
        or baseline.shape != candidate.shape
        or len(baseline) < 3
        or not np.isfinite(baseline).all()
        or not np.isfinite(candidate).all()
    ):
        raise ValueError(
            "At least three finite independent complete-cohort aggregates are required"
        )
    if repetitions < 200 or not 0 < confidence < 1 or (relative and (baseline <= 0).any()):
        raise ValueError("Invalid aggregate interval controls")
    differences = (baseline - candidate) / baseline if relative else baseline - candidate
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [
            differences[rng.integers(0, len(differences), len(differences))].mean()
            for _ in range(repetitions)
        ]
    )
    low, high = np.quantile(estimates, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {
        "improvement": float(differences.mean()),
        "low": float(low),
        "high": float(high),
        "confidence": confidence,
        "independent_complete_cohorts": len(differences),
        "unit": "relative_fraction" if relative else "absolute_metric_units",
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
    }


def _read_report(directory):
    value = read_json(Path(directory) / "report.json")
    if set(value) != {"report_id", "report"} or digest_json(value["report"]) != value["report_id"]:
        raise ValueError("Evidence checksum/schema mismatch")
    return value["report"], value["report_id"]


def _quality(directory, cohort_id, artifact_id, requested):
    report, identity = _read_report(directory)
    if report.get("failed_reference_ids") or report.get("failed_generated_ids"):
        raise ValueError("Failed samples cannot be removed from a quality cohort")
    if report.get("status") != "ok":
        if report.get("error") in {
            "FileNotFoundError",
            "ModuleNotFoundError",
            "ImportError",
            "PermissionError",
        }:
            raise _MissingEvidence("official_quality_evaluation_unavailable")
        raise ValueError("Quality evaluation failed")
    values = dict(report["protocol"])
    values.pop("preprocessing")
    values.pop("aggregation")
    values["extractor"] = ExtractorPin(**values["extractor"])
    protocol = DistributionProtocol(**values)
    if (
        report["protocol_id"] != protocol.id
        or digest_json(report["protocol"]) != protocol.id
        or protocol.extractor.provider != "cleanfid_inception"
    ):
        raise ValueError("Only the declared clean-fid image protocol is currently supported")
    if (
        protocol.generated_cohort_id != cohort_id
        or report["producer_artifacts"][0] != artifact_id
        or not set(requested) <= set(protocol.metrics)
    ):
        raise ValueError("Quality cohort/model/metric identity differs from the gate")
    if (
        report["actual_generated_samples"] != len(protocol.expected_generated_ids)
        or report["expected_generated_samples"] != len(protocol.expected_generated_ids)
        or report["generated_ids"] != list(protocol.expected_generated_ids)
    ):
        raise ValueError("Generated quality sample set is incomplete or reordered")
    if report["actual_reference_samples"] != report["expected_reference_samples"] or report[
        "actual_reference_samples"
    ] != len(report["reference_ids"]):
        raise ValueError("Reference quality sample set is incomplete")
    generation = report["generation"]
    if (
        not generation
        or generation["plan_id"] != digest_json(generation["plan"])
        or generation["sampling_binding"]["policy_artifact_id"] != artifact_id
    ):
        raise ValueError("Quality report lacks native generation identity")
    cases = generation["plan"]["cases"]
    if tuple(case["id"] for case in cases) != protocol.expected_generated_ids:
        raise ValueError("Quality report is not the complete planned native cohort")
    if (
        digest_json({"cases": cases, "quantization": generation["plan"]["quantization"]})
        != cohort_id
    ):
        raise ValueError(
            "Quality cohort hash does not describe the actual cases/seeds/conditioning"
        )
    arrays = {}
    if set(report["feature_files"]) != {"reference.features.npy", "generated.features.npy"}:
        raise ValueError("Both complete official feature matrices are required")
    for key, count in (
        ("reference", report["actual_reference_samples"]),
        ("generated", report["actual_generated_samples"]),
    ):
        name = key + ".features.npy"
        path = Path(directory) / name
        metadata = report["feature_files"][name]
        if path.is_symlink() or file_digest(path) != metadata["sha256"]:
            raise ValueError("Quality feature bytes changed")
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if (
            array.shape != (count, 2048)
            or list(array.shape) != metadata["shape"]
            or str(array.dtype) != metadata["dtype"]
            or array.dtype.kind != "f"
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                "Quality features differ from the real Inception 2048-dimension protocol"
            )
        arrays[key] = array
    return report, identity, protocol, arrays


def _official_scores(fid, protocol, arrays):
    scores = {}
    if "fid_clean" in protocol.metrics:
        scores["fid_clean"] = float(fid.fid_from_feats(arrays["reference"], arrays["generated"]))
    if "kid_clean" in protocol.metrics:
        with _NUMPY_RANDOM_LOCK:
            state = np.random.get_state()
            try:
                np.random.seed(protocol.kid_seed)
                scores["kid_clean"] = float(
                    fid.kernel_distance(
                        arrays["reference"],
                        arrays["generated"],
                        num_subsets=protocol.kid_subsets,
                        max_subset_size=protocol.kid_subset_size,
                    )
                )
            finally:
                np.random.set_state(state)
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("Non-finite official aggregate score")
    return scores


def _performance(directory, quality, cohort_id, artifact_id, resources):
    report, identity = _read_report(directory)
    if report.get("kind") != "native_generation_performance" or report.get("status") != "ok":
        raise ValueError("Performance run failed or has a different measurement kind")
    if report.get("qualification") != "host_asserted_isolation":
        raise _MissingEvidence("performance_hardware_isolation_not_asserted")
    settings = GenerationBenchmarkSettings(**report["settings"])
    if not settings.isolated_hardware_asserted:
        raise _MissingEvidence("performance_hardware_isolation_not_asserted")
    generation = quality["generation"]
    from . import generation_performance

    if report["measurement_source_sha256"] != file_digest(Path(generation_performance.__file__)):
        raise ValueError("Performance measurement implementation is not the current verified codec")
    if (
        report["latency_scope"]
        != "synchronized_sampler_plus_optional_vae_no_loading_rng_creation_or_image_io"
        or report["nfe_scope"] != "actual_native_field_forward_calls_excludes_vae_decode"
    ):
        raise ValueError("Performance timing/NFE scopes differ from the supported measurement")
    if any(
        report["environment"].get(key) != value for key, value in generation["environment"].items()
    ):
        raise ValueError(
            "Performance uses a different execution environment from quality generation"
        )
    if (
        report["candidate_artifact_id"] != artifact_id
        or report["cohort_id"] != cohort_id
        or report["plan_id"] != generation["plan_id"]
        or digest_json(report["plan"]) != report["plan_id"]
    ):
        raise ValueError("Performance did not execute the evaluated model/sampler/cohort")
    if report["sampling_binding"] != generation["sampling_binding"]:
        raise ValueError(
            "Performance and quality generation used different model/decoder contracts"
        )
    if (
        not report.get("native_producer_sources")
        or report["native_producer_sources"] != generation["native_producer_sources"]
    ):
        raise ValueError(
            "Performance and quality generation used different native producer source versions"
        )
    from .generative import GenerationCase, ImageSamplingPlan
    from .drifting_generation import DriftingSamplingPlan
    from .interval_generation import interval_plan_from_record
    from .consistency_generation import ConsistencySamplingPlan
    from .edm_generation import EDMSamplingPlan

    values = dict(report["plan"])
    values["cases"] = tuple(GenerationCase(**case) for case in values["cases"])
    mode = report["sampling_binding"]["sampling_mode"]
    if mode == "edm_heun":
        plan = EDMSamplingPlan(**values)
    elif mode == "consistency":
        plan = ConsistencySamplingPlan(**values)
    elif mode in {"meanflow", "shortcut"}:
        plan = interval_plan_from_record(report)
    else:
        plan = (
            DriftingSamplingPlan(**values)
            if mode == "drifting_direct"
            else ImageSamplingPlan(**values)
        )
    if report["expected_ids"] != [case.id for case in plan.cases]:
        raise ValueError("Performance expected sample set changed")
    for field, repeat_count in (
        ("warmups", settings.warmup_repetitions),
        ("records", settings.repetitions),
    ):
        rows = report[field]
        if [(row["repetition"], row["sample_id"]) for row in rows] != [
            (repeat, case.id) for repeat in range(repeat_count) for case in plan.cases
        ]:
            raise ValueError("Performance matrix has missing/duplicate/reordered trials")
        if any(row["status"] != "ok" or row["error"] is not None for row in rows):
            raise ValueError("Failed warmup/trial cannot enter the gate")
        for row in rows:
            if (
                type(row["nfe"]) is not int
                or row["nfe"] != expected_nfe(plan)
                or type(row["latency_seconds"]) not in {int, float}
                or not math.isfinite(row["latency_seconds"])
                or row["latency_seconds"] <= 0
            ):
                raise ValueError("Performance metrics must be finite real native measurements")
    aggregates = {}
    for name in resources:
        values = [row.get(name) for row in report["records"]]
        if any(value is None for value in values):
            raise _MissingEvidence("required_resource_not_measured:" + name)
        if any(
            type(value) not in {int, float} or not math.isfinite(value) or value <= 0
            for value in values
        ):
            raise ValueError("Invalid resource measurement")
        if (
            name == "cuda_peak_allocated_bytes"
            and report["memory_scope"] != "cuda_peak_allocated_absolute_includes_resident_model"
        ):
            raise ValueError("CPU RSS/Python memory cannot be re-labelled as CUDA peak allocation")
        aggregates[name] = float(
            max(values) if name == "cuda_peak_allocated_bytes" else np.mean(values)
        )
    return report, identity, aggregates


def _decide(protocol, quality, resources):
    failures = []
    for name, tolerance in protocol.quality_max_regression:
        if quality[name]["low"] < -tolerance:
            failures.append("quality_noninferiority:" + name)
    for name, tolerance in protocol.resource_max_relative_regression:
        if resources[name]["low"] < -tolerance:
            failures.append("resource_regression:" + name)
    for name, required in protocol.required_relative_improvements:
        if resources[name]["low"] < required:
            failures.append("insufficient_resource_improvement:" + name)
    return failures


def evaluate_generation_gate(
    protocol,
    quality_pairs,
    performance_pairs,
    *,
    output_directory,
    source_root=None,
    weights_path=None,
    grant=None,
):

    if not isinstance(protocol, GenerationGateProtocol):
        raise ValueError("Typed generation gate protocol required")
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "protocol_id": protocol.id,
        "protocol": asdict(protocol),
        "status": "not_evaluated",
        "passed": False,
        "reasons": [],
        "quality": {},
        "resources": {},
        "evidence": [],
        "statistical_unit": "independent_complete_generation_cohort_not_individual_image",
        "source_sha256": file_digest(Path(__file__)),
    }
    try:
        if set(quality_pairs) != set(protocol.cohort_ids) or set(performance_pairs) != set(
            protocol.cohort_ids
        ):
            raise _MissingEvidence("missing_declared_cohort_evidence")
        if len(protocol.cohort_ids) < 3:
            raise _MissingEvidence("at_least_three_independent_complete_cohorts_required")
        if source_root is None or weights_path is None or grant is None:
            raise _MissingEvidence("official_resources_or_authorization_missing")
        if not Path(source_root).is_dir() or not Path(weights_path).is_file():
            raise _MissingEvidence("official_local_resources_missing")
        grant.require(protocol, "official_metric_recompute")
        quality_values = {name: [[], []] for name, _ in protocol.quality_max_regression}
        resources = set(dict(protocol.resource_max_relative_regression)) | set(
            dict(protocol.required_relative_improvements)
        )
        resource_values = {name: [[], []] for name in sorted(resources)}
        common_quality = common_reference = common_performance = None
        seen_seeds = set()
        feature_paths = []
        report_paths = []
        for cohort in protocol.cohort_ids:
            if len(quality_pairs[cohort]) != 2 or len(performance_pairs[cohort]) != 2:
                raise ValueError("Each cohort needs baseline and candidate evidence")
            pair_seed_set = None
            for side, artifact_id in enumerate(
                (protocol.baseline_artifact_id, protocol.candidate_artifact_id)
            ):
                qdir, pdir = quality_pairs[cohort][side], performance_pairs[cohort][side]
                quality, qid, metric_protocol, arrays = _quality(
                    qdir, cohort, artifact_id, quality_values
                )
                metric_protocol.extractor.verify(source_root, weights_path)
                grant.require(protocol, "official_metric_recompute")
                fid = importlib.import_module("cleanfid.fid")
                if Path(fid.__file__).resolve().parent != Path(source_root).resolve():
                    raise ValueError("Imported official metric code differs from pinned source")
                controls = metric_protocol.to_dict()
                controls.pop("generated_cohort_id")
                controls.pop("expected_generated_ids")
                signature = digest_json(
                    {"controls": controls, "generated_count": quality["actual_generated_samples"]}
                )
                reference = quality["feature_files"]["reference.features.npy"]["sha256"]
                if common_quality is None:
                    common_quality, common_reference = signature, reference
                if signature != common_quality or reference != common_reference:
                    raise ValueError(
                        "Quality repetitions use different reference/features/protocol or sample counts"
                    )
                seeds = {case["seed"] for case in quality["generation"]["plan"]["cases"]}
                if side == 0:
                    if seeds & seen_seeds:
                        raise ValueError(
                            "Repeated complete cohorts must use disjoint generation seeds"
                        )
                    seen_seeds |= seeds
                    pair_seed_set = seeds
                elif seeds != pair_seed_set:
                    raise ValueError("Baseline/candidate cohort seeds differ")
                scores = _official_scores(fid, metric_protocol, arrays)
                for name in quality_values:
                    saved = quality["metrics"][name]
                    if (
                        saved["higher_is_better"] is not False
                        or saved["unit"] != "distance"
                        or type(saved["value"]) not in {int, float}
                        or not math.isfinite(saved["value"])
                        or not math.isclose(
                            saved["value"], scores[name], rel_tol=1e-6, abs_tol=1e-6
                        )
                    ):
                        raise ValueError(
                            "Reported public score differs from recomputed official feature aggregate"
                        )
                    quality_values[name][side].append(scores[name])
                performance, pid, aggregates = _performance(
                    pdir, quality, cohort, artifact_id, resources
                )
                performance_signature = digest_json(
                    {
                        name: performance[name]
                        for name in (
                            "settings",
                            "environment",
                            "measurement_source_sha256",
                            "latency_scope",
                            "nfe_scope",
                            "memory_scope",
                        )
                    }
                )
                if common_performance is None:
                    common_performance = performance_signature
                if performance_signature != common_performance:
                    raise ValueError("Performance hardware/settings/timing implementation differ")
                for name in resource_values:
                    resource_values[name][side].append(aggregates[name])
                report["evidence"].append(
                    {
                        "cohort_id": cohort,
                        "artifact_id": artifact_id,
                        "quality_report_id": qid,
                        "performance_report_id": pid,
                    }
                )
                feature_paths.extend(
                    (Path(qdir) / name, metadata["sha256"])
                    for name, metadata in quality["feature_files"].items()
                )
                report_paths.extend(((qdir, qid), (pdir, pid)))
        comparisons = (
            len(protocol.quality_max_regression)
            + len(protocol.resource_max_relative_regression)
            + len(protocol.required_relative_improvements)
        )
        confidence = 1 - (1 - protocol.confidence) / comparisons
        arguments = {
            "confidence": confidence,
            "repetitions": protocol.bootstrap_repetitions,
            "seed": protocol.bootstrap_seed,
        }
        report["quality"] = {
            name: complete_cohort_interval(*values, **arguments)
            for name, values in quality_values.items()
        }
        report["resources"] = {
            name: complete_cohort_interval(*values, relative=True, **arguments)
            for name, values in resource_values.items()
        }
        for path, expected in feature_paths:
            if file_digest(path) != expected:
                raise ValueError("Feature evidence changed during the gate")
        for directory, expected in report_paths:
            if _read_report(directory)[1] != expected:
                raise ValueError("Quality/performance report changed during the gate")
        metric_protocol.extractor.verify(source_root, weights_path)
        grant.require(protocol, "official_metric_recompute")
        report["reasons"] = _decide(protocol, report["quality"], report["resources"])
        report["passed"] = not report["reasons"]
        report["status"] = "promote" if report["passed"] else "reject"
        report["effective_per_comparison_confidence"] = confidence
    except (
        _MissingEvidence,
        FileNotFoundError,
        ModuleNotFoundError,
        ImportError,
        PermissionError,
    ) as error:
        report["status"], report["reasons"] = (
            "not_evaluated",
            [str(error) if isinstance(error, _MissingEvidence) else type(error).__name__],
        )
    except (ValueError, KeyError, TypeError, IndexError, FloatingPointError) as error:
        report["status"], report["reasons"] = "reject", [type(error).__name__]
    atomic_json(root / "report.json", {"report_id": digest_json(report), "report": report})
    return report
