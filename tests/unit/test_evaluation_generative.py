from dataclasses import asdict, replace
import importlib.metadata
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from aster.core import ArtifactStore, atomic_json, digest_json, file_digest, read_json
from aster.evaluation.generative import (
    DistributionProtocol,
    ExtractorPin,
    GenerationCase,
    ImageSamplingPlan,
    MediaManifest,
    evaluate_media_directories,
    generate_image_shard,
    image_directory_manifest,
    merge_image_shards,
    population_frechet_distance,
    quantize_image,
    source_tree_hash,
    video_directory_manifest,
    _video_features,
)
from aster.evaluation.suites import EvaluationGrant
from aster.models.generative import (
    AutoencoderKL,
    AutoencoderConfig,
    DiT,
    DiTConfig,
    UNet2D,
    UNetConfig,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def image_set(root, count=3, cohort_id=None):
    root.mkdir()
    names = {}
    for index in range(count):
        name = f"{index}.png"
        Image.fromarray(np.full((4, 6, 3), index * 30, dtype=np.uint8)).save(root / name)
        names[f"case-{index}"] = name
    manifest = image_directory_manifest(
        root,
        dataset_id="local_fixture_not_benchmark",
        revision="fixed-v1",
        split="test",
        license_id="fixture-generated",
        files_by_id=names,
        cohort_id=cohort_id,
    )
    manifest.save(root)
    return manifest


def fake_pin(provider="cleanfid_inception"):

    dependencies = {
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "Pillow": importlib.metadata.version("Pillow"),
    }
    if provider == "cleanfid_inception":
        dependencies.update(
            {
                "clean-fid": "fixture-unavailable",
                "scipy": "fixture-unavailable",
                "torchvision": "fixture-unavailable",
            }
        )
    url = (
        "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
        if provider == "cleanfid_inception"
        else "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
    )
    return ExtractorPin(
        provider,
        "1" * 40,
        "fixture-unavailable",
        "2" * 64,
        "3" * 64,
        url,
        "fixture-not-official-provenance",
        tuple(sorted(dependencies.items())),
    )


def permit(protocol):
    return EvaluationGrant(
        protocol.id, ("official_evaluator", "torchscript_execution"), time.monotonic() + 60
    )


def test_image_manifest_full_ids_bytes_modes_and_extra_files(tmp_path):
    root = tmp_path / "images"
    manifest = image_set(root)
    assert MediaManifest.load(root) == manifest
    assert manifest.verify(root).id == manifest.id
    Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(root / "unlisted.png")
    with pytest.raises(ValueError, match="unlisted"):
        manifest.verify(root)
    with pytest.raises(ValueError):
        image_directory_manifest(
            root,
            dataset_id="d",
            revision="main",
            split="test",
            license_id="fixture",
            files_by_id={"x": "0.png"},
        )
    with pytest.raises(ValueError):
        image_directory_manifest(
            root,
            dataset_id="d",
            revision="fixed",
            split="test",
            license_id="fixture",
            files_by_id={"x": "../outside.png"},
        )
    Image.fromarray(np.ones((4, 6, 3), np.uint8)).save(root / "0.png")
    with pytest.raises(ValueError, match="bytes"):
        manifest.verify(root)
    gray = tmp_path / "gray"
    gray.mkdir()
    Image.fromarray(np.zeros((4, 4), np.uint8)).save(gray / "x.png")
    with pytest.raises(ValueError, match="RGB"):
        image_directory_manifest(
            gray,
            dataset_id="d",
            revision="v1",
            split="test",
            license_id="fixture",
            files_by_id={"x": "x.png"},
        )


def test_generation_quantization_is_explicit_and_finite():
    image = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).reshape(1, 1, 5).expand(3, -1, -1)
    assert quantize_image(image, "minus_one_one_stylegan")[0, :, 0].tolist() == [
        0,
        0,
        128,
        255,
        255,
    ]
    image = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 3).expand(3, -1, -1)
    assert quantize_image(image, "zero_one_round")[0, :, 0].tolist() == [0, 128, 255]
    with pytest.raises(ValueError):
        quantize_image(torch.full((3, 4, 4), float("nan")), "zero_one_round")
    with pytest.raises(ValueError):
        quantize_image(torch.zeros(4, 4, 4), "zero_one_round")


def publish_native(store, directory, kind="dit"):
    torch.manual_seed(38)
    if kind == "dit":
        model = DiT(DiTConfig(in_channels=3, hidden_size=16, num_layers=1, num_heads=2))

        with torch.no_grad():
            model.output.weight.normal_(std=0.1)
            model.blocks[0].ada[-1].weight.normal_(std=0.1)
    else:
        model = UNet2D(
            UNetConfig(
                in_channels=3,
                model_channels=8,
                channel_mult=(1,),
                num_res_blocks=1,
                attention_levels=(),
                num_heads=1,
                prediction_type="epsilon",
            )
        )
        with torch.no_grad():
            model.output[-1].weight.normal_(std=0.02)
    model.save_pretrained(directory)
    if kind != "dit":
        from aster.methods.generation import DiffusionObjective, DiffusionSchedule

        atomic_json(
            directory / "objective.json",
            DiffusionObjective(DiffusionSchedule.create(17)).config_dict(),
        )
    return store.publish(
        directory, kind="native_field_model", metadata={"evidence": "local_fixture_not_benchmark"}
    )


def test_native_sampling_rank_partition_same_images_and_artifact_lineage(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    policy = publish_native(store, tmp_path / "model")
    plan = ImageSamplingPlan(
        tuple(GenerationCase(f"sample{i}", 101 + i) for i in range(5)), (3, 4, 4), steps=2
    )
    single = generate_image_shard(store, policy.id, plan, tmp_path / "single")
    parts = [tmp_path / f"rank{rank}" for rank in range(2)]
    for rank, directory in enumerate(parts):
        generate_image_shard(store, policy.id, plan, directory, rank=rank, world_size=2)
    merged = merge_image_shards(parts[::-1], plan, tmp_path / "merged")
    assert single.samples == merged.samples and all(s.status == "ok" for s in merged.samples)
    assert len({s.files[0].sha256 for s in merged.samples}) > 1
    assert merged.producer_artifacts == (policy.id,)
    artifact = store.publish(
        tmp_path / "merged",
        kind="generated_images",
        metadata={"manifest_id": merged.id},
        parents=merged.producer_artifacts,
    )
    assert MediaManifest.load(artifact.path).verify(artifact.path).id == merged.id
    with pytest.raises(ValueError, match="All generation ranks"):
        merge_image_shards(parts[:1], plan, tmp_path / "incomplete")
    with pytest.raises(ValueError, match="Duplicate"):
        merge_image_shards([parts[0], parts[0]], plan, tmp_path / "duplicated")
    shard = read_json(parts[0] / "shard.json")
    shard["plan"]["steps"] = 3
    atomic_json(parts[0] / "shard.json", shard)
    with pytest.raises(ValueError, match="plan"):
        merge_image_shards(parts, plan, tmp_path / "changed-plan")


def test_native_ddim_sampler_runs_and_differs_from_flow_parameters(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    policy = publish_native(store, tmp_path / "unet", "unet")
    plan = ImageSamplingPlan(
        (GenerationCase("a", 1), GenerationCase("b", 2)), (3, 4, 4), sampler="ddim", steps=3
    )
    result = generate_image_shard(store, policy.id, plan, tmp_path / "ddim")
    assert all(s.status == "ok" for s in result.samples)
    result.verify(tmp_path / "ddim")
    with pytest.raises(ValueError):
        replace(plan, flow_shift=2.0)
    with pytest.raises(ValueError):
        replace(plan, sampler="flow_heun", eta=0.1)


def test_latent_generator_uses_native_decoder_and_explicit_scale_shift(tmp_path):
    from aster.methods.generation import sample_flow

    store = ArtifactStore(tmp_path / "store")
    torch.manual_seed(811)
    model = DiT(DiTConfig(in_channels=2, hidden_size=16, num_layers=1, num_heads=2)).eval()
    decoder = AutoencoderKL(
        AutoencoderConfig(
            in_channels=3,
            latent_channels=2,
            base_channels=4,
            channel_mult=(1,),
            num_res_blocks=1,
            scaling_factor=0.5,
            shift_factor=0.7,
        )
    ).eval()
    model.save_pretrained(tmp_path / "model")
    decoder.save_pretrained(tmp_path / "decoder")
    policy = store.publish(tmp_path / "model", kind="native_field_model", metadata={})
    vae = store.publish(tmp_path / "decoder", kind="native_decoder", metadata={})
    plan = ImageSamplingPlan((GenerationCase("latent-case", 198),), (2, 4, 4), steps=2)
    manifest = generate_image_shard(
        store, policy.id, plan, tmp_path / "generated", decoder_artifact_id=vae.id
    )
    assert manifest.producer_artifacts == (policy.id, vae.id) and manifest.samples[0].status == "ok"
    noise = torch.randn((1, 2, 4, 4), generator=torch.Generator().manual_seed(198))
    with torch.no_grad():
        latent = sample_flow(model, noise, steps=2)
        expected = quantize_image(decoder.decode(latent, scaled=True)[0], plan.quantization)
        incorrect = quantize_image(decoder.decode(latent, scaled=False)[0], plan.quantization)
    with Image.open(tmp_path / "generated" / manifest.samples[0].files[0].path) as image:
        np.testing.assert_array_equal(np.asarray(image), expected)
    assert not np.array_equal(expected, incorrect)


def test_single_rank_cannot_impersonate_full_generated_sample_set(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    policy = publish_native(store, tmp_path / "model")
    plan = ImageSamplingPlan(tuple(GenerationCase(str(i), i) for i in range(4)), (3, 4, 4), steps=1)
    partial = generate_image_shard(
        store, policy.id, plan, tmp_path / "partial", rank=0, world_size=2
    )
    reference = image_set(tmp_path / "reference", 2)
    protocol = DistributionProtocol(
        reference.id, plan.cohort_id, fake_pin(), tuple(c.id for c in plan.cases)
    )
    with pytest.raises(ValueError, match="fixed distribution protocol"):
        evaluate_media_directories(
            protocol,
            tmp_path / "reference",
            tmp_path / "partial",
            source_root=tmp_path / "missing",
            weights_path=tmp_path / "missing-weight",
            grant=permit(protocol),
            output_directory=tmp_path / "report",
        )

    bad_protocol = replace(protocol, expected_generated_ids=partial.expected_ids)
    report = evaluate_media_directories(
        bad_protocol,
        tmp_path / "reference",
        tmp_path / "partial",
        source_root=tmp_path / "missing",
        weights_path=tmp_path / "missing-weight",
        grant=permit(bad_protocol),
        output_directory=tmp_path / "bad-report",
    )
    assert (
        report["error"] == "ValueError" and report["generation"] is None and not report["metrics"]
    )


def test_failed_generation_retains_entire_cohort_and_no_distribution_score(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    policy = publish_native(store, tmp_path / "model")

    plan = ImageSamplingPlan((GenerationCase("a", 1), GenerationCase("b", 2)), (2, 4, 4), steps=1)
    manifest = generate_image_shard(store, policy.id, plan, tmp_path / "failed")
    assert manifest.expected_ids == ("a", "b") and all(
        s.status == "error" for s in manifest.samples
    )
    merged = merge_image_shards([tmp_path / "failed"], plan, tmp_path / "merged")
    merged.verify(tmp_path / "merged", require_complete=False)
    with pytest.raises(ValueError, match="Failed"):
        merged.verify(tmp_path / "merged")
    reference = image_set(tmp_path / "reference", 2)
    protocol = DistributionProtocol(reference.id, plan.cohort_id, fake_pin(), merged.expected_ids)
    report = evaluate_media_directories(
        protocol,
        tmp_path / "reference",
        tmp_path / "merged",
        source_root=tmp_path / "missing-code",
        weights_path=tmp_path / "missing-weights",
        grant=permit(protocol),
        output_directory=tmp_path / "report",
    )
    assert (
        report["status"] == "error"
        and report["metrics"] == {}
        and report["failed_generated_ids"] == ["a", "b"]
    )
    assert report["expected_generated_samples"] == 2 and report["feature_files"] == {}
    assert read_json(tmp_path / "report" / "report.json")["report_id"] == digest_json(report)


def test_optimized_sampling_is_comparable_without_erasing_actual_solver_identity():
    plan = ImageSamplingPlan((GenerationCase("a", 1), GenerationCase("b", 2)), (3, 4, 4), steps=50)
    optimized = replace(plan, steps=4, sampler="flow_euler")
    assert optimized.id != plan.id and optimized.cohort_id == plan.cohort_id
    assert replace(plan, quantization="zero_one_round").cohort_id != plan.cohort_id
    assert (
        replace(plan, cases=(GenerationCase("a", 3), GenerationCase("b", 2))).cohort_id
        != plan.cohort_id
    )


def test_missing_official_weight_never_downloads_and_authorization_precedes_import(
    tmp_path, monkeypatch
):
    reference = image_set(tmp_path / "reference", 2)
    generated = image_set(tmp_path / "generated", 2)
    protocol = DistributionProtocol(
        reference.id, generated.cohort_id, fake_pin(), generated.expected_ids
    )
    imports = []
    monkeypatch.setattr(
        "aster.evaluation.generative.importlib.import_module", lambda name: imports.append(name)
    )
    kwargs = dict(source_root=tmp_path / "missing-code", weights_path=tmp_path / "missing-weight")
    with pytest.raises(PermissionError):
        evaluate_media_directories(
            protocol,
            tmp_path / "reference",
            tmp_path / "generated",
            **kwargs,
            grant=EvaluationGrant(protocol.id, ("official_evaluator",), time.monotonic() + 30),
            output_directory=tmp_path / "not-authorized",
        )
    assert not (tmp_path / "not-authorized").exists()
    report = evaluate_media_directories(
        protocol,
        tmp_path / "reference",
        tmp_path / "generated",
        **kwargs,
        grant=permit(protocol),
        output_directory=tmp_path / "missing-report",
    )
    assert report["status"] == "error" and not report["metrics"] and not imports
    assert not (tmp_path / "missing-weight").exists()
    assert report["protocol"]["preprocessing"]["covariance_ddof"] == 1
    assert replace(protocol, kid_seed=1).id != protocol.id


def test_extractor_versions_source_closure_and_hash_validation(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in ("fid.py", "inception_torchscript.py", "resize.py", "utils.py"):
        (root / name).write_text("# local metadata fixture, never imported\n", encoding="utf-8")
    first = source_tree_hash(root)
    (root / "other.py").write_text("# changed closure\n", encoding="utf-8")
    assert first != source_tree_hash(root)
    pin = fake_pin()
    with pytest.raises(ValueError):
        replace(pin, revision="main")
    with pytest.raises(ValueError):
        replace(pin, dependencies=(("torch", "2"),))
    with pytest.raises(ValueError):
        replace(pin, weights_source="https://not-official.example/weights")
    weight = tmp_path / "inception-2015-12-05.pt"
    weight.write_bytes(b"not-loaded")
    with pytest.raises(ValueError, match="hash mismatch"):
        pin.verify(root, weight)


def test_real_cleanfid_api_contract_disables_download_and_preserves_manifest_order(
    tmp_path, monkeypatch
):

    reference = image_set(tmp_path / "reference", 2)
    generated = image_set(tmp_path / "generated", 2)
    protocol = DistributionProtocol(
        reference.id, generated.cohort_id, fake_pin(), generated.expected_ids
    )
    source = tmp_path / "installed-cleanfid"
    source.mkdir()
    calls = []

    class ModelCallSpy:
        def __init__(self, path, *, download, resize_inside):
            calls.append(("model", path, download, resize_inside))

        def eval(self):
            return self

        def to(self, device):
            return self

    def features(l_files, **kwargs):
        calls.append(("features", l_files, kwargs))
        raise RuntimeError("Contract test deliberately stops before producing any metric")

    fid = SimpleNamespace(__file__=str(source / "fid.py"), get_files_features=features)
    inception = SimpleNamespace(
        __file__=str(source / "inception_torchscript.py"), InceptionV3W=ModelCallSpy
    )
    monkeypatch.setattr(ExtractorPin, "verify", lambda *args: None)
    monkeypatch.setattr(
        "aster.evaluation.generative.importlib.import_module",
        lambda name: {"cleanfid.fid": fid, "cleanfid.inception_torchscript": inception}[name],
    )
    report = evaluate_media_directories(
        protocol,
        tmp_path / "reference",
        tmp_path / "generated",
        source_root=source,
        weights_path=tmp_path / "inception-2015-12-05.pt",
        grant=permit(protocol),
        output_directory=tmp_path / "contract-report",
    )
    assert calls[0] == ("model", str(tmp_path), False, False)
    assert calls[1][1] == [str(tmp_path / "reference" / f"{i}.png") for i in range(2)]
    assert (
        calls[1][2]["mode"] == "clean"
        and calls[1][2]["num_workers"] == 0
        and not calls[1][2]["verbose"]
    )
    assert report["status"] == "error" and report["metrics"] == {}


def test_video_frame_selection_population_covariance_not_image_fid(tmp_path):
    root = tmp_path / "video"
    root.mkdir()
    files = {}
    indices = tuple(range(0, 20, 2))
    for clip in range(2):
        names = []
        for index in indices:
            name = f"clip-{clip}-frame-{index}.png"
            Image.fromarray(np.full((4, 4, 3), index + clip, np.uint8)).save(root / name)
            names.append(name)
        files[str(clip)] = names
    manifest = video_directory_manifest(
        root,
        dataset_id="frames_fixture_not_benchmark",
        revision="decoded-v1",
        split="test",
        license_id="fixture",
        frames_by_id=files,
        frame_indices=indices,
        fps=30.0,
    )
    manifest.save(root)
    assert MediaManifest.load(root).verify(root).id == manifest.id
    protocol = DistributionProtocol(
        manifest.id,
        manifest.cohort_id,
        fake_pin("styleganv_i3d"),
        manifest.expected_ids,
        metrics=("fvd_styleganv_i3d",),
        frame_indices=indices,
        fps=30.0,
    )
    assert protocol.to_dict()["preprocessing"]["covariance_ddof"] == 0
    with pytest.raises(ValueError):
        replace(protocol, metrics=("fid_clean",))
    with pytest.raises(ValueError):
        replace(protocol, frame_indices=(0, 2, 4))
    real, generated = np.array([[0.0], [2.0]]), np.array([[0.0], [4.0]])

    assert population_frechet_distance(real, generated) == pytest.approx(2.0)
    assert population_frechet_distance(real, real) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        population_frechet_distance(real, np.array([[float("nan")], [1.0]]))

    class LayoutSpy(torch.nn.Module):
        def forward(self, videos, *, rescale, resize, return_features):
            assert videos.shape == (2, 3, 10, 4, 4) and videos.dtype == torch.uint8
            assert rescale and resize and return_features
            assert videos[0, 0, :, 0, 0].tolist() == list(indices)
            return videos.float().mean((1, 2, 3, 4))[:, None].expand(-1, 400)

    assert _video_features(root, manifest, LayoutSpy(), protocol, "cpu").shape == (2, 400)
    with pytest.raises(ValueError, match="selection/FPS"):
        _video_features(root, manifest, LayoutSpy(), replace(protocol, fps=24.0), "cpu")
    with pytest.raises(ValueError):
        replace(protocol, fps=True)
