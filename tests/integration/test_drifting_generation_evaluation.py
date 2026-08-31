from dataclasses import replace
from datetime import timedelta
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.core import ArtifactStore, atomic_json, read_json
from aster.evaluation.drifting_generation import (
    DriftingSamplingPlan,
    generate_drifting_shard,
    merge_drifting_shards,
    publish_drifting_generator,
)
from aster.evaluation.generation_artifacts import load_native_artifact_model
from aster.evaluation.generative import (
    GenerationCase,
    ImageSamplingPlan,
    MediaManifest,
    _generation_record,
    generate_image_shard,
    merge_image_shards,
    quantize_image,
)
from aster.methods.drifting import DriftingMethod, SpatialFeatureStatistics
from aster.models import load_model
from aster.models.drifting import DriftingConfig, DriftingGenerator
from aster.models.generative import AutoencoderConfig, AutoencoderKL
from aster.training import Trainer, ParallelConfig, ParallelContext


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def train_generator(*, discrete=True, ema=True, zero=0, precision="fp32", output_channels=3):
    torch.manual_seed(931)
    config = DriftingConfig(
        input_size=4,
        in_channels=3,
        out_channels=output_channels,
        patch_size=2,
        hidden_size=16,
        cond_dim=16,
        num_layers=1,
        num_heads=2,
        num_classes=3,
        n_cls_tokens=2,
        noise_classes=7 if discrete else 0,
        noise_coords=2,
    )
    engine = Trainer(
        DriftingGenerator(config),
        lr=0.005,
        ema_decay=0.5 if ema else None,
        zero_stage=zero,
        precision=precision,
    )
    method = DriftingMethod(
        engine,
        SpatialFeatureStatistics(patch_sizes=(2,), use_mean=False, use_std=False),
        feature_identity="local_pixel_feature_fixture_not_pretrained_mae",
        num_classes=2,
        positive_capacity=4,
        negative_capacity=8,
        positive_samples=2,
        negative_samples=2,
        generated_samples=2,
        cfg_min=1.0,
        cfg_max=4.0,
        seed=137,
    )
    batch = {"samples": torch.randn(2, output_channels, 4, 4), "labels": torch.tensor([0, 1])}
    assert method.update([batch]).updated
    assert method.update([batch]).updated
    return engine, method


def pixels(root, manifest, index=0):
    with Image.open(root / manifest.samples[index].files[0].path) as image:
        return np.asarray(image)


@pytest.mark.parametrize("discrete", [False, True])
def test_trained_drifting_single_call_noise_identity_and_sharding(tmp_path, monkeypatch, discrete):
    engine, method = train_generator(discrete=discrete)
    store = ArtifactStore(tmp_path / "store")
    rng = torch.get_rng_state().clone()
    artifact = publish_drifting_generator(method, store, tmp_path / "export")
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    config = read_json(artifact.path / "generation_contract.json")
    assert (
        config["conditioning_semantics"] == "guidance_embedding" and "generator_time" not in config
    )
    plan = DriftingSamplingPlan(
        tuple(GenerationCase(f"image-{i}", 111 + i, i % 2) for i in range(3)),
        (3, 4, 4),
        cfg_scale=2.75,
        temperature=0.85,
    )
    calls = []
    original_loader = load_native_artifact_model

    def traced_loader(value):
        model, layout = original_loader(value)
        model.register_forward_pre_hook(
            lambda module, args: calls.append((args[1].detach().cpu().tolist(), args[2]))
        )
        return model, layout

    monkeypatch.setattr(
        "aster.evaluation.drifting_generation.load_native_artifact_model", traced_loader
    )
    single = generate_drifting_shard(store, artifact.id, plan, tmp_path / "single")
    single.verify(tmp_path / "single")
    assert len(calls) == 3 and all(cfg == [2.75] for cfg, condition in calls)
    reloaded = load_model(artifact.path / "model").eval()
    assert reloaded.output.weight.abs().sum() > 0
    expected = reloaded.generate(
        torch.tensor([0]),
        cfg_scale=2.75,
        temperature=0.85,
        generator=torch.Generator().manual_seed(111),
    )
    np.testing.assert_array_equal(
        pixels(tmp_path / "single", single), quantize_image(expected[0], plan.quantization)
    )
    record = _generation_record(tmp_path / "single", single)
    source = record["sample_inputs"][0]
    rng = torch.Generator().manual_seed(111)
    noise = torch.randn((1, 3, 4, 4), generator=rng) * 0.85
    assert source["continuous_noise_sha256"] == hashlib.sha256(noise.numpy().tobytes()).hexdigest()
    assert source["noise_labels"] == (
        torch.randint(7, (1, 2), generator=rng).tolist()[0] if discrete else None
    )
    parts = [tmp_path / f"rank-{rank}" for rank in range(4)]
    for rank, part in enumerate(parts):
        generate_drifting_shard(store, artifact.id, plan, part, rank=rank, world_size=4)
    merged = merge_drifting_shards(parts[::-1], plan, tmp_path / "merged")
    assert merged.samples == single.samples
    assert (
        _generation_record(tmp_path / "merged", merged)["sample_inputs"] == record["sample_inputs"]
    )
    produced = store.publish(
        tmp_path / "merged",
        kind="generated_images",
        metadata={"manifest": merged.id},
        parents=merged.producer_artifacts,
    )
    assert MediaManifest.load(produced.path).verify(produced.path).id == merged.id
    assert (
        replace(plan, cfg_scale=3.25).id != plan.id
        and replace(plan, cfg_scale=3.25).cohort_id == plan.cohort_id
    )
    assert ImageSamplingPlan(plan.cases, plan.noise_shape, steps=2).cohort_id == plan.cohort_id


@pytest.mark.parametrize("zero,precision", [(0, "bf16"), (3, "fp32")])
def test_drifting_ema_export_uses_real_fp32_logical_weights_and_restore(tmp_path, zero, precision):
    engine, method = train_generator(zero=zero, precision=precision)
    assert all(parameter.dtype == torch.float32 for parameter in engine.model.parameters())
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    store = ArtifactStore(tmp_path / "store")
    ordinary = publish_drifting_generator(method, store, tmp_path / "ordinary")
    averaged = publish_drifting_generator(method, store, tmp_path / "averaged", ema=True)
    expected, normal = engine.export_state_dict(ema=True), engine.export_state_dict()
    loaded = load_model(averaged.path / "model")
    for key, value in expected.items():
        assert loaded.state_dict()[key].dtype == value.dtype
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    assert any(not torch.equal(normal[name], value) for name, value in expected.items())
    assert averaged.id != ordinary.id
    metadata = read_json(averaged.path / "generation_contract.json")["training"]
    assert (
        metadata["weight_selection"] == "ema"
        and metadata["ema_decay"] == 0.5
        and metadata["trainer_precision"] == precision
    )
    restored_engine, restored_method = train_generator(zero=zero, precision=precision)
    restored_engine.load_checkpoint(checkpoint, trusted=True)
    again = publish_drifting_generator(restored_method, store, tmp_path / "restored", ema=True)
    assert again.id == averaged.id
    plan = DriftingSamplingPlan((GenerationCase("ema", 938, 0),), (3, 4, 4), cfg_scale=2.0)
    generate_drifting_shard(store, averaged.id, plan, tmp_path / "images").verify(
        tmp_path / "images"
    )


def test_drifting_latent_decoding_uses_pinned_native_vae(tmp_path):
    engine, method = train_generator(output_channels=2)
    store = ArtifactStore(tmp_path / "store")
    policy = publish_drifting_generator(method, store, tmp_path / "export")
    decoder = AutoencoderKL(
        AutoencoderConfig(
            in_channels=3,
            latent_channels=2,
            base_channels=4,
            channel_mult=(1,),
            num_res_blocks=1,
            scaling_factor=0.5,
            shift_factor=0.25,
        )
    ).eval()
    decoder.save_pretrained(tmp_path / "decoder" / "model")
    vae = store.publish(tmp_path / "decoder", kind="native_decoder", metadata={})
    plan = DriftingSamplingPlan((GenerationCase("latent", 93, 1),), (3, 4, 4), cfg_scale=2.0)
    root = tmp_path / "images"
    manifest = generate_drifting_shard(store, policy.id, plan, root, decoder_artifact_id=vae.id)
    manifest.verify(root)
    assert manifest.producer_artifacts == (policy.id, vae.id)
    model = load_model(policy.path / "model").eval()
    with torch.no_grad():
        expected = decoder.decode(
            model.generate(
                torch.tensor([1]), cfg_scale=2.0, generator=torch.Generator().manual_seed(93)
            ),
            scaled=True,
        )
    np.testing.assert_array_equal(
        pixels(root, manifest), quantize_image(expected[0], plan.quantization)
    )


def test_drifting_invalid_class_retains_full_outcomes_and_rejects_wrong_consumers(tmp_path):
    engine, method = train_generator()
    store = ArtifactStore(tmp_path / "store")
    artifact = publish_drifting_generator(method, store, tmp_path / "model")
    plan = DriftingSamplingPlan(
        (GenerationCase("good", 1, 0), GenerationCase("bad", 2, 2)), (3, 4, 4)
    )
    result = generate_drifting_shard(store, artifact.id, plan, tmp_path / "images")
    assert result.expected_ids == ("good", "bad") and [
        sample.status for sample in result.samples
    ] == ["ok", "error"]
    merged = merge_drifting_shards([tmp_path / "images"], plan, tmp_path / "merged")
    with pytest.raises(ValueError, match="Failed"):
        merged.verify(tmp_path / "merged")
    assert len(_generation_record(tmp_path / "merged", merged)["sample_inputs"]) == 2
    with pytest.raises(ValueError, match="different native producer"):
        merge_image_shards([tmp_path / "images"], plan, tmp_path / "wrong-merge")
    with pytest.raises(ValueError, match="UNet2D/DiT"):
        generate_image_shard(
            store,
            artifact.id,
            ImageSamplingPlan(plan.cases, plan.noise_shape, sampler="direct_x0", steps=1),
            tmp_path / "wrong-sampler",
        )
    for case in (GenerationCase("x", 2, None), GenerationCase("x", 2, (1.0, 2.0))):
        with pytest.raises(ValueError):
            DriftingSamplingPlan((case,), (3, 4, 4))
    with pytest.raises(ValueError):
        replace(plan, cfg_scale=0.5)
    with pytest.raises(ValueError):
        replace(plan, temperature=0.0)
    with pytest.raises(ValueError, match="geometry"):
        generate_drifting_shard(
            store, artifact.id, replace(plan, noise_shape=(3, 8, 8)), tmp_path / "shape"
        )


def test_drifting_export_guards_incomplete_ema_and_stale_contract(tmp_path):
    engine, method = train_generator(ema=False)
    store = ArtifactStore(tmp_path / "store")
    with pytest.raises(RuntimeError, match="EMA"):
        publish_drifting_generator(method, store, tmp_path / "missing-ema", ema=True)
    artifact = publish_drifting_generator(method, store, tmp_path / "export")
    method._incomplete = True
    with pytest.raises(RuntimeError, match="incomplete"):
        publish_drifting_generator(method, store, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()
    loaded = load_model(artifact.path / "model")
    with torch.no_grad():
        loaded.output.weight.add_(0.01)
    loaded.save_pretrained(tmp_path / "changed" / "model")
    contract = read_json(artifact.path / "generation_contract.json")
    atomic_json(tmp_path / "changed" / "generation_contract.json", contract)
    changed = store.publish(tmp_path / "changed", kind="native_drifting_generator", metadata={})
    plan = DriftingSamplingPlan((GenerationCase("x", 1, 0),), (3, 4, 4))
    with pytest.raises(ValueError, match="weights differ"):
        generate_drifting_shard(store, changed.id, plan, tmp_path / "wrong-weights")


def _dp_export_worker(rank, rendezvous, directory, zero):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        context = ParallelContext(ParallelConfig(data_parallel=2))
        torch.manual_seed(213)
        config = DriftingConfig(
            input_size=4,
            in_channels=3,
            out_channels=3,
            hidden_size=16,
            cond_dim=16,
            num_layers=1,
            num_heads=2,
            num_classes=2,
            noise_classes=3,
            noise_coords=2,
        )
        engine = Trainer(
            DriftingGenerator(config), parallel=context, zero_stage=zero, ema_decay=0.5, lr=0.004
        )
        method = DriftingMethod(
            engine,
            SpatialFeatureStatistics(patch_sizes=(), use_mean=False, use_std=False),
            feature_identity="distributed_pixel_fixture",
            positive_capacity=3,
            negative_capacity=4,
            positive_samples=2,
            negative_samples=2,
            generated_samples=2,
        )
        torch.manual_seed(315 + rank)
        assert method.update(
            [{"samples": torch.randn(2, 3, 4, 4), "labels": torch.tensor([0, 1])}]
        ).updated
        root = Path(directory)
        store = ArtifactStore(root / "store")
        artifact = publish_drifting_generator(method, store, root / "export", ema=True)
        identities = context.world.gather_objects(artifact.id)
        assert identities[0] == identities[1]
        expected = engine.export_state_dict(ema=True)
        if rank == 0:
            loaded = load_model(artifact.path / "model")
            for name, tensor in expected.items():
                torch.testing.assert_close(loaded.state_dict()[name], tensor, atol=0, rtol=0)
            plan = DriftingSamplingPlan((GenerationCase("dp", 5, 0),), (3, 4, 4), cfg_scale=2.0)
            generate_drifting_shard(store, artifact.id, plan, root / "images").verify(
                root / "images"
            )
            atomic_json(
                root / "verified.json",
                {"artifact_id": artifact.id, "dp": 2, "zero": zero, "ema": True},
            )
        context.world.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("zero", [0, 3])
def test_drifting_dp_collective_export_publishes_one_real_artifact(tmp_path, zero):

    base = Path(tempfile.gettempdir())
    if not str(base).isascii():
        base = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    if not base.is_dir():
        pytest.skip("An existing writable ASCII temporary directory is required for Windows Gloo")
    scratch = Path(tempfile.mkdtemp(prefix="aster-drifting-export-", dir=base)).resolve()
    assert scratch.parent == base.resolve() and scratch.name.startswith("aster-drifting-export-")
    try:
        mp.spawn(
            _dp_export_worker,
            args=(str(scratch / "rendezvous"), str(tmp_path), zero),
            nprocs=2,
            join=True,
        )
    finally:
        shutil.rmtree(scratch)
    assert read_json(tmp_path / "verified.json")["zero"] == zero
