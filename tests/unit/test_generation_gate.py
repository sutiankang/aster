from dataclasses import replace
import os
import time

import numpy as np
import pytest
import torch

from aster.core import ArtifactStore, atomic_json, digest_json, read_json
from aster.evaluation.generative import (
    GenerationCase,
    ImageSamplingPlan,
    generate_image_shard,
    _generation_record,
)
from aster.evaluation.generation_performance import (
    GenerationBenchmarkSettings,
    benchmark_image_sampler,
    expected_nfe,
)
from aster.evaluation.generation_gate import (
    GenerationGateProtocol,
    complete_cohort_interval,
    evaluate_generation_gate,
    _decide,
    _read_report,
    _performance,
    _MissingEvidence,
)
from aster.methods.generation import FlowObjective
from aster.models.generative import UNet2D, UNetConfig
from aster.training import Trainer


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def policy(tmp_path):
    torch.manual_seed(111)
    model = UNet2D(
        UNetConfig(
            in_channels=3,
            model_channels=8,
            channel_mult=(1,),
            num_res_blocks=1,
            attention_levels=(),
            num_heads=2,
            prediction_type="velocity",
        )
    )
    objective = FlowObjective()
    engine = Trainer(model, objective, lr=0.003)
    assert engine.step([{"sample": torch.randn(2, 3, 4, 4)}]).updated
    model.save_pretrained(tmp_path / "model")
    atomic_json(tmp_path / "model" / "objective.json", objective.config_dict())
    store = ArtifactStore(tmp_path / "store")
    return store, store.publish(
        tmp_path / "model",
        kind="native_field",
        metadata={"evidence": "trained_local_fixture_not_public_benchmark"},
    )


def gate(**kwargs):
    return GenerationGateProtocol(
        "a" * 64,
        "b" * 64,
        tuple(digest_json(i) for i in range(3)),
        bootstrap_repetitions=300,
        **kwargs,
    )


def test_complete_cohort_bootstrap_is_not_pseudo_per_image_fid():

    actual = complete_cohort_interval([10.0, 11.0, 12.0], [8.0, 9.0, 10.0], repetitions=300)
    assert actual["low"] == actual["high"] == actual["improvement"] == 2.0
    assert actual["independent_complete_cohorts"] == 3
    relative = complete_cohort_interval(
        [10.0, 20.0, 30.0], [8.0, 16.0, 24.0], relative=True, repetitions=300
    )
    assert relative["low"] == pytest.approx(0.2) and relative["high"] == pytest.approx(0.2)
    first = complete_cohort_interval([10.0, 12.0, 15.0], [7.0, 10.0, 12.0], repetitions=300, seed=5)
    assert first == complete_cohort_interval(
        [10.0, 12.0, 15.0], [7.0, 10.0, 12.0], repetitions=300, seed=5
    )
    for bad in ([1.0, 2.0], [1.0, float("nan"), 3.0]):
        with pytest.raises(ValueError):
            complete_cohort_interval(bad, bad)
    with pytest.raises(ValueError):
        complete_cohort_interval([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], relative=True)


def test_joint_decision_requires_quality_and_actual_positive_resource_gain():
    protocol = gate()
    quality = {"fid_clean": {"low": 0.0}, "kid_clean": {"low": 0.0}}
    resources = {"latency_seconds": {"low": 0.15}, "nfe": {"low": 0.8}}
    assert _decide(protocol, quality, resources) == []
    assert _decide(protocol, {**quality, "fid_clean": {"low": -0.01}}, resources) == [
        "quality_noninferiority:fid_clean"
    ]
    assert "insufficient_resource_improvement:latency_seconds" in _decide(
        protocol, quality, {**resources, "latency_seconds": {"low": 0.0}}
    )
    assert "resource_regression:nfe" in _decide(
        protocol, quality, {**resources, "nfe": {"low": -0.01}}
    )
    with pytest.raises(ValueError):
        gate(required_relative_improvements=(("latency_seconds", 0.0),))
    with pytest.raises(ValueError):
        gate(quality_max_regression=(("fid_clean", float("nan")),))


def test_public_gate_missing_official_resources_never_promotes(tmp_path):
    protocol = gate()
    pairs = {
        cohort: (tmp_path / "baseline", tmp_path / "candidate") for cohort in protocol.cohort_ids
    }
    report = evaluate_generation_gate(
        protocol, pairs, pairs, output_directory=tmp_path / "no-resources"
    )
    assert report["status"] == "not_evaluated" and not report["passed"] and not report["quality"]
    assert report["reasons"] == ["official_resources_or_authorization_missing"]
    assert digest_json(_read_report(tmp_path / "no-resources")[0]) == digest_json(report)
    one = replace(protocol, cohort_ids=protocol.cohort_ids[:1])
    report = evaluate_generation_gate(
        one,
        {one.cohort_ids[0]: pairs[one.cohort_ids[0]]},
        {one.cohort_ids[0]: pairs[one.cohort_ids[0]]},
        output_directory=tmp_path / "one-cohort",
    )
    assert report["status"] == "not_evaluated" and "three" in report["reasons"][0]
    report = evaluate_generation_gate(
        protocol, {}, pairs, output_directory=tmp_path / "missing-cohort"
    )
    assert report["status"] == "not_evaluated" and not report["passed"]


@pytest.mark.parametrize(
    "sampler,guidance,nfe", [("flow_euler", 1.0, 2), ("flow_heun", 1.0, 4), ("flow_rk4", 2.0, 16)]
)
def test_actual_native_performance_reports_forward_counts_and_sync_scope(
    tmp_path, sampler, guidance, nfe
):
    store, artifact = policy(tmp_path)
    plan = ImageSamplingPlan(
        (GenerationCase("a", 55), GenerationCase("b", 56)),
        (3, 4, 4),
        sampler=sampler,
        steps=2,
        guidance_scale=guidance,
    )
    settings = GenerationBenchmarkSettings(warmup_repetitions=1, repetitions=2)
    report = benchmark_image_sampler(store, artifact.id, plan, settings, tmp_path / "performance")
    assert report["status"] == "ok" and len(report["records"]) == 4 and len(report["warmups"]) == 2
    assert expected_nfe(plan) == nfe and all(
        row["nfe"] == nfe and row["latency_seconds"] > 0 for row in report["records"]
    )
    assert all(row["cuda_peak_allocated_bytes"] is None for row in report["records"])
    assert report["memory_scope"] == "cpu_torch_allocator_peak_unavailable"
    assert "no_loading_rng_creation_or_image_io" in report["latency_scope"]
    assert report["qualification"] == "development_not_promotion_evidence"
    assert report["records"][0]["output_sha256"] == report["records"][2]["output_sha256"]
    images = generate_image_shard(store, artifact.id, plan, tmp_path / "images")
    generation = _generation_record(tmp_path / "images", images)
    assert report["sampling_binding"] == generation["sampling_binding"]
    assert report["native_producer_sources"] == generation["native_producer_sources"]
    assert all(
        report["environment"][key] == value for key, value in generation["environment"].items()
    )

    quality = {"generation": {"plan_id": plan.id, "sampling_binding": report["sampling_binding"]}}
    with pytest.raises(_MissingEvidence, match="isolation"):
        _performance(
            tmp_path / "performance", quality, plan.cohort_id, artifact.id, {"latency_seconds"}
        )


def test_performance_failure_keeps_all_trials_instead_of_average_survivors(tmp_path):
    store, artifact = policy(tmp_path)
    plan = ImageSamplingPlan((GenerationCase("a", 1), GenerationCase("b", 2)), (2, 4, 4), steps=1)
    report = benchmark_image_sampler(
        store, artifact.id, plan, GenerationBenchmarkSettings(repetitions=2), tmp_path / "failed"
    )
    assert (
        report["status"] == "error" and len(report["records"]) == 4 and len(report["warmups"]) == 2
    )
    assert all(
        row["status"] == "error" and row["latency_seconds"] is None and row["nfe"] is None
        for row in report["records"]
    )
    with pytest.raises(ValueError):
        GenerationBenchmarkSettings(warmup_repetitions=0)
    with pytest.raises(ValueError):
        GenerationBenchmarkSettings(repetitions=1)


def test_report_integrity_is_checked_not_trusted_pass_boolean(tmp_path):
    tmp_path.joinpath("report.json").write_text(
        '{"report_id":"invalid","report":{"passed":true}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum"):
        _read_report(tmp_path)


@pytest.mark.parametrize("family", ["dmd", "drifting"])
def test_actual_single_step_performance_matches_image_consumer_binding(tmp_path, family):
    store = ArtifactStore(tmp_path / "store")
    torch.manual_seed(173)
    if family == "dmd":
        from aster.methods.generative_distillation import DMDMethod
        from aster.evaluation.generative import publish_dmd_generator

        config = UNetConfig(
            in_channels=3,
            model_channels=8,
            channel_mult=(1,),
            num_res_blocks=1,
            attention_levels=(),
            num_heads=2,
            prediction_type="x0",
        )
        engine = Trainer(UNet2D(config), lr=0.003)
        method = DMDMethod(
            engine,
            UNet2D(replace(config, prediction_type="edm_residual")),
            UNet2D(replace(config, prediction_type="edm_residual")),
        )
        method.update([{"noise": torch.randn(2, 3, 4, 4), "sigma": torch.tensor([0.5, 1.0])}])
        artifact = publish_dmd_generator(method, store, tmp_path / "export")
        plan = ImageSamplingPlan((GenerationCase("x", 4),), (3, 4, 4), sampler="direct_x0", steps=1)
        producer = generate_image_shard
    else:
        from aster.models.drifting import DriftingConfig, DriftingGenerator
        from aster.methods.drifting import DriftingMethod, SpatialFeatureStatistics
        from aster.evaluation.drifting_generation import (
            DriftingSamplingPlan,
            publish_drifting_generator,
            generate_drifting_shard,
        )

        config = DriftingConfig(
            input_size=4,
            in_channels=3,
            out_channels=3,
            hidden_size=16,
            cond_dim=16,
            num_heads=2,
            num_layers=1,
            num_classes=2,
            noise_classes=5,
            noise_coords=2,
        )
        engine = Trainer(DriftingGenerator(config), lr=0.003)
        method = DriftingMethod(
            engine,
            SpatialFeatureStatistics(patch_sizes=(), use_mean=False, use_std=False),
            feature_identity="local_pixel_fixture",
            positive_capacity=3,
            negative_capacity=4,
            positive_samples=2,
            negative_samples=2,
            generated_samples=2,
        )
        method.update([{"samples": torch.randn(2, 3, 4, 4), "labels": torch.tensor([0, 1])}])
        artifact = publish_drifting_generator(method, store, tmp_path / "export")
        plan = DriftingSamplingPlan((GenerationCase("x", 4, 0),), (3, 4, 4), cfg_scale=2.0)
        producer = generate_drifting_shard
    report = benchmark_image_sampler(
        store, artifact.id, plan, GenerationBenchmarkSettings(repetitions=2), tmp_path / "perf"
    )
    images = producer(store, artifact.id, plan, tmp_path / "images")
    images.verify(tmp_path / "images")
    assert report["status"] == "ok" and all(row["nfe"] == 1 for row in report["records"])
    assert (
        report["sampling_binding"]
        == _generation_record(tmp_path / "images", images)["sampling_binding"]
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA hardware; CPU is not CUDA allocator validation"
)
def test_cuda_sync_and_allocator_peak_are_real_hardware_measurements(tmp_path):
    store, artifact = policy(tmp_path)
    plan = ImageSamplingPlan((GenerationCase("gpu", 22),), (3, 4, 4), steps=1)
    report = benchmark_image_sampler(
        store,
        artifact.id,
        plan,
        GenerationBenchmarkSettings(repetitions=2),
        tmp_path / "gpu",
        device="cuda",
    )
    assert report["status"] == "ok"
    assert all(row["cuda_peak_allocated_bytes"] > 0 for row in report["records"])


@pytest.mark.skipif(
    not os.environ.get("ASTER_APPROVED_GENERATION_GATE"),
    reason="No approved public cohort/official feature resources; never fake a promotion",
)
def test_real_approved_official_quality_and_performance_gate(tmp_path):
    from aster.evaluation.suites import EvaluationGrant

    config = read_json(os.environ["ASTER_APPROVED_GENERATION_GATE"])
    assert config["approved_local_execution"] is True
    protocol = GenerationGateProtocol(**config["protocol"])
    report = evaluate_generation_gate(
        protocol,
        config["quality_pairs"],
        config["performance_pairs"],
        source_root=config["source_root"],
        weights_path=config["weights_path"],
        output_directory=tmp_path / "official-gate",
        grant=EvaluationGrant(
            protocol.id,
            ("official_metric_recompute",),
            time.monotonic() + config["deadline_seconds"],
        ),
    )
    assert report["status"] == config["expected_status"]
    assert report["status"] in {"promote", "reject"}
