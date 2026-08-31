from dataclasses import replace
import importlib.metadata
import time

import numpy as np
from PIL import Image
import pytest
import torch

from aster.core import ArtifactStore, atomic_json, read_json
from aster.models.video_world import WanVideoConfig, WanVideoDiT
from aster.models.video_vae import WanVAEConfig, WanVideoVAE
from aster.methods.video_generation import VideoGenerationPipeline, image_video_condition
from aster.evaluation.generative import (
    DistributionProtocol,
    ExtractorPin,
    MediaManifest,
    evaluate_media_directories,
    quantize_image,
)
from aster.evaluation.suites import EvaluationGrant
from aster.evaluation.video_generation import (
    VideoConditionBundle,
    VideoGenerationCase,
    VideoSamplingPlan,
    generate_video_shard,
    merge_video_shards,
    publish_video_conditions,
    video_generation_record,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def components(root, *, image=False, negative=True):
    torch.manual_seed(233)
    store = ArtifactStore(root / "store")
    field = WanVideoDiT(
        WanVideoConfig(
            latent_channels=2,
            hidden_size=12,
            intermediate_size=24,
            num_heads=2,
            num_layers=1,
            text_dim=4,
            text_length=3,
            frequency_dim=4,
            condition_channels=4 if image else 0,
            image_conditioned=image,
            image_dim=3,
        )
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
    )
    with torch.no_grad():
        field.head.head.weight.normal_(std=0.05)
    field.eval()
    vae.eval()
    field.save_pretrained(root / "field")
    vae.save_pretrained(root / "vae")
    field_artifact = store.publish(
        root / "field", kind="native_wan_video", metadata={"evidence": "fixture"}
    )
    vae_artifact = store.publish(
        root / "vae", kind="native_wan_vae", metadata={"evidence": "fixture"}
    )
    positive = {"text": torch.randn(1, 2, 4), "text_lengths": torch.tensor([2])}
    if image:
        positive.update(
            image_features=torch.randn(1, 2, 3),
            video_condition=image_video_condition(vae, torch.randn(1, 3, 4, 4), 11),
        )
    branches = {"positive": positive}
    if negative:
        branches["negative"] = {**positive, "text": torch.zeros_like(positive["text"])}
    condition = publish_video_conditions(store, {"prompt1": branches}, root / "conditions")
    plan = VideoSamplingPlan(
        tuple(VideoGenerationCase(f"clip{i}", 80 + i, "prompt1") for i in range(3)),
        condition.id,
        (2, 6, 2, 2),
        (11, 4, 4),
        24.0,
        steps=2,
        solver="euler",
        guidance_scale=2.0 if negative else 1.0,
    )
    return store, field_artifact, vae_artifact, condition, plan, field, vae, branches


def unavailable_i3d():
    return ExtractorPin(
        "styleganv_i3d",
        "1" * 40,
        "fixture-not-official",
        "2" * 64,
        "3" * 64,
        "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1",
        "fixture-no-weight",
        tuple((name, importlib.metadata.version(name)) for name in ("Pillow", "numpy", "torch")),
    )


def test_native_video_shards_vae_decode_condition_lineage_and_fvd_preflight(tmp_path):
    store, field_id, vae_id, condition, plan, field, vae, branches = components(tmp_path)
    single = generate_video_shard(store, field_id.id, vae_id.id, plan, tmp_path / "single")
    assert all(s.status == "ok" for s in single.samples)
    assert all(
        len(s.files) == 11 and s.frame_indices == tuple(range(11)) and s.fps == 24.0
        for s in single.samples
    )
    rank_directories = [tmp_path / f"rank-{rank}" for rank in range(2)]
    for rank, directory in enumerate(rank_directories):
        generate_video_shard(
            store, field_id.id, vae_id.id, plan, directory, rank=rank, world_size=2
        )
    merged = merge_video_shards(rank_directories[::-1], plan, tmp_path / "merged")
    assert merged.samples == single.samples and merged.producer_artifacts == (
        field_id.id,
        vae_id.id,
        condition.id,
    )
    assert len({sample.files[0].sha256 for sample in merged.samples}) == 3
    merged.verify(tmp_path / "merged")
    artifact = store.publish(
        tmp_path / "merged",
        kind="generated_video_frames",
        metadata={"manifest_id": merged.id},
        parents=merged.producer_artifacts,
    )
    assert MediaManifest.load(artifact.path).verify(artifact.path).id == merged.id
    generation = video_generation_record(artifact.path, merged)
    assert generation["condition_provenance"]["encoder_execution_verified"] is False
    assert generation["plan"]["condition_artifact_id"] == condition.id
    with torch.no_grad():
        pipeline = VideoGenerationPipeline(field, vae).eval()
        noise = torch.randn((1, *plan.latent_shape), generator=torch.Generator().manual_seed(80))
        expected = pipeline.generate(
            noise,
            branches["positive"],
            negative_condition=branches["negative"],
            steps=plan.steps,
            solver=plan.solver,
            shift=plan.shift,
            guidance_scale=plan.guidance_scale,
        )
    for index, frame in enumerate(merged.samples[0].files):
        with Image.open(artifact.path / frame.path) as image:
            np.testing.assert_array_equal(
                np.asarray(image), quantize_image(expected[0, :, index], plan.quantization)
            )
    protocol = DistributionProtocol(
        single.id,
        plan.cohort_id,
        unavailable_i3d(),
        merged.expected_ids,
        metrics=("fvd_styleganv_i3d",),
        frame_indices=plan.frame_indices,
        fps=plan.fps,
    )
    grant = EvaluationGrant(
        protocol.id, ("official_evaluator", "torchscript_execution"), time.monotonic() + 30
    )
    report = evaluate_media_directories(
        protocol,
        tmp_path / "single",
        artifact.path,
        source_root=tmp_path / "no-code",
        weights_path=tmp_path / "no-i3d",
        grant=grant,
        output_directory=tmp_path / "fvd",
    )
    assert report["status"] == "error" and report["metrics"] == {}
    assert report["generation"]["plan_id"] == plan.id and report["expected_generated_samples"] == 3
    assert not (tmp_path / "no-i3d").exists()


def test_video_condition_artifact_is_tensor_only_and_does_not_claim_encoder_execution(tmp_path):
    store, _, _, condition, _, _, _, branches = components(tmp_path)
    bundle = VideoConditionBundle(store, condition.id)
    loaded = bundle.load_case("prompt1")
    torch.testing.assert_close(loaded["positive"]["text"], branches["positive"]["text"])
    loaded["positive"]["text"].zero_()
    assert bundle.load_case("prompt1")["positive"]["text"].abs().sum() > 0
    assert (
        condition.metadata["provenance"]["text_semantics"]
        == "stored_features_not_automatically_official_T5_or_CLIP"
    )
    with pytest.raises(ValueError, match="Unknown/missing"):
        publish_video_conditions(
            store,
            {"x": {"positive": {"text": torch.zeros(1, 2, 4), "callback": lambda: 1}}},
            tmp_path / "bad-key",
        )
    with pytest.raises(ValueError):
        publish_video_conditions(
            store,
            {"x": {"positive": {"text": torch.tensor(float("nan"))}}},
            tmp_path / "bad-tensor",
        )
    provenance_root = tmp_path / "source"
    provenance_root.mkdir()
    atomic_json(provenance_root / "prompt.json", {"prompt": "fixture"})
    source = store.publish(provenance_root, kind="condition_input", metadata={})
    declared = publish_video_conditions(
        store, {"x": branches}, tmp_path / "with-source", source_artifact_ids=(source.id,)
    )
    assert VideoConditionBundle(store, declared.id).provenance["source_artifact_ids"] == [source.id]


def test_video_cfg_failure_preserves_every_clip_and_wrong_shapes_are_rejected(tmp_path):
    store, field, vae, _, plan, _, _, _ = components(tmp_path, negative=False)
    bad = replace(plan, guidance_scale=3.0)
    manifest = generate_video_shard(store, field.id, vae.id, bad, tmp_path / "failed")
    assert len(manifest.samples) == 3 and all(
        sample.status == "error" for sample in manifest.samples
    )
    assert all(
        sample.frame_indices == plan.frame_indices and not sample.files
        for sample in manifest.samples
    )
    merged = merge_video_shards([tmp_path / "failed"], bad, tmp_path / "merged-failed")
    merged.verify(tmp_path / "merged-failed", require_complete=False)
    with pytest.raises(ValueError, match="Failed"):
        merged.verify(tmp_path / "merged-failed")
    with pytest.raises(ValueError, match="geometry"):
        generate_video_shard(
            store,
            field.id,
            vae.id,
            replace(plan, output_shape=(12, 4, 4)),
            tmp_path / "wrong-geometry",
        )
    assert not (tmp_path / "wrong-geometry").exists()
    with pytest.raises(ValueError, match="condition key"):
        generate_video_shard(
            store,
            field.id,
            vae.id,
            replace(plan, cases=(VideoGenerationCase("bad", 4, "absent"),)),
            tmp_path / "missing-key",
        )


def test_video_image_conditions_and_cross_rank_identity_are_actual(tmp_path):
    store, field, vae, _, plan, _, _, _ = components(tmp_path, image=True)
    directories = [tmp_path / f"rank-{rank}" for rank in range(2)]
    for rank, path in enumerate(directories):
        result = generate_video_shard(store, field.id, vae.id, plan, path, rank=rank, world_size=2)
        assert all(sample.status == "ok" for sample in result.samples)
    with pytest.raises(ValueError, match="All video"):
        merge_video_shards(directories[:1], plan, tmp_path / "missing-rank")
    with pytest.raises(ValueError, match="duplicate"):
        merge_video_shards([directories[0], directories[0]], plan, tmp_path / "duplicate-rank")
    partial = MediaManifest.load(directories[0])
    with pytest.raises(ValueError, match="full planned cohort"):
        video_generation_record(directories[0], partial)
    record = read_json(directories[0] / "video_shard.json")
    record["plan"]["fps"] = 30.0
    atomic_json(directories[0] / "video_shard.json", record)
    with pytest.raises(ValueError, match="plan/manifest"):
        merge_video_shards(directories, plan, tmp_path / "wrong-fps")


def test_video_cohort_allows_distillation_without_erasing_conditions_or_fps(tmp_path):
    _, _, _, _, plan, _, _, _ = components(tmp_path)
    optimized = replace(plan, steps=1, solver="heun", guidance_scale=1.0)
    assert optimized.id != plan.id and optimized.cohort_id == plan.cohort_id
    assert replace(plan, fps=30.0).cohort_id != plan.cohort_id
    assert replace(plan, condition_artifact_id="f" * 64).cohort_id != plan.cohort_id
    with pytest.raises(ValueError):
        replace(plan, solver="unipc")
    with pytest.raises(ValueError):
        replace(plan, fps=True)
