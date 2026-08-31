from dataclasses import replace
import copy
import importlib.metadata
import time

import numpy as np
from PIL import Image
import pytest
import torch

from aster.core import ArtifactStore, atomic_json, digest_json, read_json
from aster.models.video_world import WanVideoConfig, WanVideoDiT
from aster.models.video_vae import WanVAEConfig, WanVideoVAE
from aster.methods.video_generation import WanVideoObjective
from aster.training import Trainer
from aster.evaluation.video_generation import (
    VideoSamplingPlan,
    VideoGenerationCase,
    publish_video_conditions,
)
from aster.evaluation.generative import (
    _generation_record,
    MediaManifest,
    DistributionProtocol,
    ExtractorPin,
)
from aster.evaluation.suites import EvaluationGrant
from aster.evaluation.generation_performance import GenerationBenchmarkSettings
from aster.optimization.wan_teacache import WanTeaCacheSettings
from aster.evaluation.wan_teacache import (
    publish_wan_cache_calibration,
    load_wan_cache_calibration,
    generate_wan_cache_cohort,
    benchmark_wan_cache_cohort,
    wan_cache_generation_record,
    evaluate_wan_cache_fvd,
    compare_wan_cache_cohorts,
)
from aster.evaluation.wan_teacache import WanCacheFVDResources


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trained_components(root, negative=True):
    torch.manual_seed(962)
    store = ArtifactStore(root / "store")
    field = WanVideoDiT(
        WanVideoConfig(
            latent_channels=2,
            hidden_size=12,
            intermediate_size=24,
            num_heads=2,
            num_layers=2,
            text_dim=4,
            text_length=3,
            frequency_dim=4,
        )
    )
    positive = {"text": torch.randn(1, 2, 4)}
    batch = {
        "sample": torch.randn(1, 2, 6, 2, 2),
        "condition": positive,
        "noise": torch.randn(1, 2, 6, 2, 2),
        "time": torch.tensor([0.6]),
    }
    engine = Trainer(field, WanVideoObjective(), lr=0.002)
    for _ in range(3):
        assert engine.step([batch]).updated
    field.eval()
    field.save_pretrained(root / "field")
    policy = store.publish(
        root / "field",
        kind="native_wan_video",
        metadata={"test": "actual_tiny_native_updates_not_pretrained_quality"},
    )
    vae = WanVideoVAE(
        WanVAEConfig(
            base_channels=4,
            latent_channels=2,
            channel_mult=(1, 2),
            temporal_downsample=(True,),
            num_res_blocks=1,
            latent_mean=(0.3, -0.1),
            latent_std=(0.8, 1.4),
        )
    ).eval()
    vae.save_pretrained(root / "vae")
    decoder = store.publish(
        root / "vae", kind="native_wan_vae", metadata={"test": "untrained_tiny_decoder"}
    )
    branches = {"positive": positive}
    if negative:
        branches["negative"] = {"text": torch.randn(1, 2, 4)}
    conditions = publish_video_conditions(store, {"fixture_prompt": branches}, root / "conditions")
    plan = VideoSamplingPlan(
        tuple(VideoGenerationCase(f"calibration-{i}", 10 + i, "fixture_prompt") for i in range(2)),
        conditions.id,
        (2, 6, 2, 2),
        (11, 4, 4),
        fps=24.0,
        steps=5,
        solver="euler",
        guidance_scale=2.0,
    )
    return store, policy.id, decoder.id, plan


def test_trained_artifact_calibration_png_performance_public_FVD_unavailable(tmp_path):
    store, policy, vae, train = trained_components(tmp_path)
    before = torch.get_rng_state().clone()
    calibration = publish_wan_cache_calibration(store, policy, train, tmp_path / "calibration")
    assert torch.equal(before, torch.get_rng_state())
    fitted, receipt = load_wan_cache_calibration(store, calibration.id)
    assert len(fitted.measurements) == 16 and receipt["official_coefficients_used"] is False
    plan = replace(
        train,
        cases=tuple(
            VideoGenerationCase(f"evaluation-{i}", 90 + i, "fixture_prompt") for i in range(2)
        ),
    )
    settings = WanTeaCacheSettings(threshold=1e8, maximum_relative_output_error=1e8)
    base_dir, cache_dir = tmp_path / "baseline", tmp_path / "candidate"
    baseline = generate_wan_cache_cohort(store, policy, vae, plan, base_dir)
    candidate = generate_wan_cache_cohort(
        store,
        policy,
        vae,
        plan,
        cache_dir,
        calibration_artifact_id=calibration.id,
        cache_settings=settings,
    )
    assert all(s.status == "ok" for m in (baseline, candidate) for s in m.samples)
    assert (
        baseline.cohort_id == candidate.cohort_id
        and baseline.producer_artifacts[:3] == candidate.producer_artifacts[:3]
    )
    b = wan_cache_generation_record(base_dir)
    c = _generation_record(cache_dir, candidate)
    assert c["binding"]["condition_provenance"]["encoder_execution_verified"] is False
    assert c["binding"]["calibration_artifact_id"] == calibration.id
    assert all(
        r["observation"]["field_calls"] == 10 and r["observation"]["full_backbone_calls"] == 4
        for r in c["records"]
    )
    assert all(r["observation"]["full_backbone_calls"] == 10 for r in b["records"])
    for directory, kwargs in (
        (tmp_path / "baseline_perf", {}),
        (
            tmp_path / "candidate_perf",
            {"calibration_artifact_id": calibration.id, "cache_settings": settings},
        ),
    ):
        report = benchmark_wan_cache_cohort(
            store,
            policy,
            vae,
            plan,
            directory,
            settings=GenerationBenchmarkSettings(repetitions=2),
            **kwargs,
        )
        assert (
            report["status"] == "ok" and len(report["trials"]) == 4 and len(report["warmups"]) == 2
        )
        assert all(
            r["latency_seconds"] > 0 and r["cuda_peak_allocated_bytes"] is None
            for r in report["trials"]
        )
    public = evaluate_wan_cache_fvd(cache_dir, tmp_path / "public_fvd")
    assert public["status"] == "not_evaluated" and public["metrics"] == {}

    pin = ExtractorPin(
        "styleganv_i3d",
        "1" * 40,
        "fixture-no-official-resources",
        "2" * 64,
        "3" * 64,
        "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1",
        "fixture-no-weight",
        tuple((name, importlib.metadata.version(name)) for name in ("Pillow", "numpy", "torch")),
    )
    protocol = DistributionProtocol(
        baseline.id,
        plan.cohort_id,
        pin,
        candidate.expected_ids,
        metrics=("fvd_styleganv_i3d",),
        frame_indices=plan.frame_indices,
        fps=plan.fps,
    )
    grant = EvaluationGrant(
        protocol.id, ("official_evaluator", "torchscript_execution"), time.monotonic() + 30
    )
    attempted = evaluate_wan_cache_fvd(
        cache_dir,
        tmp_path / "official_resource_preflight",
        resources=WanCacheFVDResources(
            protocol, base_dir, tmp_path / "no-source", tmp_path / "no-weight", grant
        ),
    )
    assert (
        attempted["status"] == "error"
        and attempted["metrics"] == {}
        and attempted["generation"]["plan_id"] == plan.id
    )
    assert not (tmp_path / "no-weight").exists()
    pair = {
        "baseline": base_dir,
        "candidate": cache_dir,
        "baseline_benchmark": tmp_path / "baseline_perf",
        "candidate_benchmark": tmp_path / "candidate_perf",
    }
    gate = compare_wan_cache_cohorts(
        [pair], tmp_path / "gate", maximum_pixel_rmse=1e8, maximum_memory_ratio=1.1
    )
    assert gate["status"] == "not_evaluated"
    assert "official_FVD_not_successfully_evaluated" in gate["unevaluated"]
    assert "hardware_isolation_not_asserted" in gate["unevaluated"]
    assert "real_CUDA_peak_unavailable" in gate["unevaluated"]
    assert not gate["automatically_deployed"]
    published = store.publish(
        cache_dir,
        kind="wan_cached_generated_video",
        metadata={"manifest_id": candidate.id},
        parents=candidate.producer_artifacts,
    )
    assert wan_cache_generation_record(published.path)["plan_id"] == plan.id

    path = tmp_path / "candidate_perf" / "benchmark.json"
    envelope = read_json(path)
    envelope["report"]["trials"].pop()
    envelope["report_id"] = digest_json(envelope["report"])
    atomic_json(path, envelope)
    with pytest.raises(ValueError, match="population"):
        compare_wan_cache_cohorts([pair], tmp_path / "bad_gate")


def test_rehashed_wrong_coefficients_sampler_and_public_png_raw_mismatch_rejected(tmp_path):
    store, policy, vae, plan = trained_components(tmp_path)
    calibration = publish_wan_cache_calibration(store, policy, plan, tmp_path / "calibration")
    envelope = read_json(calibration.path / "calibration.json")
    envelope["report"]["calibration"]["coefficients"] = [0.0]
    envelope["report"]["calibration_id"] = digest_json(envelope["report"]["calibration"])
    envelope["report_id"] = digest_json(envelope["report"])
    wrong = tmp_path / "wrong_coefficients"
    wrong.mkdir()
    atomic_json(wrong / "calibration.json", envelope)
    forged = store.publish(
        wrong,
        kind=calibration.kind,
        metadata={"report_id": envelope["report_id"]},
        parents=calibration.parents,
    )
    with pytest.raises(ValueError, match="coefficients"):
        load_wan_cache_calibration(store, forged.id)
    with pytest.raises(ValueError, match="solver"):
        generate_wan_cache_cohort(
            store,
            policy,
            vae,
            replace(plan, solver="heun"),
            tmp_path / "wrong_solver",
            calibration_artifact_id=calibration.id,
            cache_settings=WanTeaCacheSettings(),
        )
    directory = tmp_path / "quality"
    manifest = generate_wan_cache_cohort(store, policy, vae, plan, directory)

    image = directory / manifest.samples[0].files[0].path
    with Image.open(image) as opened:
        pixels = np.array(opened)
    pixels[0, 0, 0] ^= 1
    Image.fromarray(pixels).save(image)
    from aster.evaluation.generative import ImageFile, _image_identity

    replacement = ImageFile(manifest.samples[0].files[0].path, *_image_identity(image))
    first = replace(manifest.samples[0], files=(replacement, *manifest.samples[0].files[1:]))
    modified = replace(manifest, samples=(first, *manifest.samples[1:]))
    modified.save(directory)
    envelope = read_json(directory / "wan_cache_generation.json")
    envelope["report"]["manifest_id"] = modified.id
    envelope["report_id"] = digest_json(envelope["report"])
    atomic_json(directory / "wan_cache_generation.json", envelope)
    with pytest.raises(ValueError, match="PNG pixels"):
        wan_cache_generation_record(directory)


def test_missing_cfg_branch_failure_population_is_not_dropped(tmp_path):
    store, policy, vae, plan = trained_components(tmp_path, negative=False)
    manifest = generate_wan_cache_cohort(store, policy, vae, plan, tmp_path / "failed")
    assert len(manifest.samples) == len(plan.cases) and all(
        s.status == "error" for s in manifest.samples
    )
    report = wan_cache_generation_record(tmp_path / "failed")
    assert [r["id"] for r in report["records"]] == [c.id for c in plan.cases]
    perf = benchmark_wan_cache_cohort(
        store,
        policy,
        vae,
        plan,
        tmp_path / "perf",
        settings=GenerationBenchmarkSettings(repetitions=2),
    )
    assert perf["status"] == "error" and len(perf["trials"]) == 4
    assert all(r["latency_seconds"] is None and r["status"] == "error" for r in perf["trials"])


def test_three_real_complete_cohorts_aggregate_latency_but_cannot_promote_without_FVD(tmp_path):
    store, policy, vae, training = trained_components(tmp_path)
    calibration = publish_wan_cache_calibration(store, policy, training, tmp_path / "calibration")
    cache = WanTeaCacheSettings(threshold=1e8, maximum_relative_output_error=1e8)
    pairs = []
    for cohort in range(3):
        plan = replace(
            training,
            cases=tuple(
                VideoGenerationCase(
                    f"heldout-{cohort}-{i}", 100 + cohort * 10 + i, "fixture_prompt"
                )
                for i in range(2)
            ),
        )
        pair = {
            name: tmp_path / f"{cohort}-{name}"
            for name in ("baseline", "candidate", "baseline_benchmark", "candidate_benchmark")
        }
        for name, kwargs in (
            ("baseline", {}),
            ("candidate", {"calibration_artifact_id": calibration.id, "cache_settings": cache}),
        ):
            generate_wan_cache_cohort(store, policy, vae, plan, pair[name], **kwargs)
            benchmark_wan_cache_cohort(
                store,
                policy,
                vae,
                plan,
                pair[name + "_benchmark"],
                settings=GenerationBenchmarkSettings(repetitions=2),
                **kwargs,
            )
        pairs.append(pair)
    report = compare_wan_cache_cohorts(
        pairs, tmp_path / "joint_gate", maximum_pixel_rmse=1e8, repetitions=200
    )
    assert (
        report["status"] != "promote"
        and "official_FVD_not_successfully_evaluated" in report["unevaluated"]
    )
    assert report["comparison"]["latency"]["independent_complete_cohorts"] == 3
    assert "fvd" not in report["comparison"]
    assert len(report["cohorts"]) == 3 and all(
        r["maximum_paired_pixel_rmse"] >= 0 for r in report["cohorts"]
    )
    with pytest.raises(ValueError, match="distinct"):
        compare_wan_cache_cohorts(
            [pairs[0], pairs[0], pairs[1]], tmp_path / "repeated_seeds", repetitions=200
        )
