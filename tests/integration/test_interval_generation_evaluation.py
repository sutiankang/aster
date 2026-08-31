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

from aster.core import ArtifactStore, atomic_json, digest_json, read_json
from aster.evaluation.generative import (
    GenerationCase,
    ImageSamplingPlan,
    _generation_record,
    quantize_image,
)
from aster.evaluation.interval_generation import (
    MeanFlowSamplingPlan,
    ShortcutSamplingPlan,
    publish_meanflow_generator,
    publish_shortcut_generator,
    generate_interval_shard,
    merge_interval_shards,
    interval_nfe,
)
from aster.evaluation.generation_performance import (
    GenerationBenchmarkSettings,
    benchmark_image_sampler,
)
from aster.evaluation.generation_gate import _performance
from aster.methods.meanflow import MeanFlowObjective
from aster.methods.shortcut import ShortcutMethod
from aster.models import load_model
from aster.models.interval_dit import IntervalDiTConfig, IntervalDiT
from aster.models.generative import AutoencoderConfig, AutoencoderKL
from aster.training import Trainer, ParallelContext, ParallelConfig


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def train(variant, *, zero=0, precision="fp32", channels=3, ema=True):
    torch.manual_seed(375)
    config = IntervalDiTConfig(
        variant=variant,
        input_size=4,
        in_channels=channels,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_classes=2,
    )
    objective = (
        MeanFlowObjective(guidance=True, omega=1.3, kappa=0.3) if variant == "meanflow" else None
    )
    engine = Trainer(
        IntervalDiT(config),
        objective,
        zero_stage=zero,
        precision=precision,
        lr=0.003,
        ema_decay=0.5 if ema else None,
    )
    method = (
        ShortcutMethod(
            engine,
            base_steps=4,
            bootstrap_every=2,
            bootstrap_ema=True,
            ema_decay=0.9,
            bootstrap_cfg=True,
        )
        if variant == "shortcut"
        else None
    )
    batch = dict(sample=torch.randn(4, channels, 4, 4), labels=torch.tensor([0, 1, 0, 1]))
    for _ in range(2):
        assert (method.update([batch]) if method else engine.step([batch])).updated
    return engine, method, batch


def publish(variant, engine, method, store, directory, **kwargs):
    return (
        publish_meanflow_generator(engine, store, directory, **kwargs)
        if variant == "meanflow"
        else publish_shortcut_generator(method, store, directory, **kwargs)
    )


def plan_for(variant, *, channels=3, cases=None):
    cases = cases or tuple(GenerationCase(f"image-{i}", 141 + i, i % 2) for i in range(3))
    return (
        MeanFlowSamplingPlan(cases, (channels, 4, 4), timesteps=(1.0, 0.625, 0.25, 0.0))
        if variant == "meanflow"
        else ShortcutSamplingPlan(cases, (channels, 4, 4), steps=4, guidance_scale=1.75)
    )


def independent_sample(model, plan, noise, label):

    with torch.no_grad():
        value = noise.clone()
        labels = torch.tensor([label])
        if isinstance(plan, MeanFlowSamplingPlan):
            for t, r in zip(plan.timesteps, plan.timesteps[1:]):
                value -= (t - r) * model(
                    value, torch.tensor([t]), torch.tensor([t - r]), labels
                ).prediction
        else:
            log_step = torch.tensor([float(np.log2(plan.steps))])
            for index in range(plan.steps):
                time = torch.tensor([index / plan.steps])
                v = model(value, time, log_step, labels).prediction
                if plan.guidance_scale != 1:
                    u = model(value, time, log_step, None).prediction
                    v = u + plan.guidance_scale * (v - u)
                value = value + v / plan.steps
        return value


@pytest.mark.parametrize(
    "variant,guidance", [("meanflow", None), ("shortcut", 1.75), ("shortcut", 0.0)]
)
def test_interval_real_training_exact_pixels_noise_nfe_shards_and_performance(
    tmp_path, variant, guidance
):
    engine, method, _ = train(variant)
    store = ArtifactStore(tmp_path / "store")
    rng = torch.get_rng_state().clone()
    artifact = publish(variant, engine, method, store, tmp_path / "export")
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    plan = plan_for(variant)
    if guidance is not None:
        plan = replace(plan, guidance_scale=guidance)
    manifest = generate_interval_shard(store, artifact.id, plan, tmp_path / "single").verify(
        tmp_path / "single"
    )
    record = _generation_record(tmp_path / "single", manifest)
    model = load_model(artifact.path / "model").eval()
    assert model.output.weight.abs().sum() > 0
    assert record["sampling_binding"]["interval_semantics"] == (
        "duration" if variant == "meanflow" else "negative_log2_step"
    )
    for index, case in enumerate(plan.cases):
        noise = torch.randn(
            (1, *plan.noise_shape), generator=torch.Generator().manual_seed(case.seed)
        )
        expected = independent_sample(model, plan, noise, case.condition)
        with Image.open(tmp_path / "single" / manifest.samples[index].files[0].path) as image:
            np.testing.assert_array_equal(
                np.asarray(image), quantize_image(expected[0], plan.quantization)
            )
        assert (
            record["sample_inputs"][index]["noise_sha256"]
            == hashlib.sha256(noise.numpy().tobytes()).hexdigest()
        )
        assert record["sample_inputs"][index]["nfe"] == interval_nfe(plan)
    paths = [tmp_path / f"rank-{rank}" for rank in range(4)]
    for rank, path in enumerate(paths):
        generate_interval_shard(store, artifact.id, plan, path, rank=rank, world_size=4)
    merged = merge_interval_shards(paths[::-1], plan, tmp_path / "merged")
    assert merged.samples == manifest.samples
    assert (
        _generation_record(tmp_path / "merged", merged)["sample_inputs"] == record["sample_inputs"]
    )
    assert plan.cohort_id == ImageSamplingPlan(plan.cases, plan.noise_shape).cohort_id
    measured = benchmark_image_sampler(
        store,
        artifact.id,
        plan,
        GenerationBenchmarkSettings(repetitions=2, isolated_hardware_asserted=True),
        tmp_path / "performance",
    )
    assert measured["status"] == "ok"
    assert all(
        x["nfe"] == interval_nfe(plan) and x["latency_seconds"] > 0 for x in measured["records"]
    )
    assert digest_json(measured["sampling_binding"]) == digest_json(record["sampling_binding"])

    _, _, aggregates = _performance(
        tmp_path / "performance",
        {"generation": record},
        plan.cohort_id,
        artifact.id,
        {"nfe", "latency_seconds"},
    )
    assert aggregates["nfe"] == interval_nfe(plan)
    assert aggregates["latency_seconds"] > 0
    changed_generation = {
        **record,
        "native_producer_sources": {
            **record["native_producer_sources"],
            "methods/meanflow.py": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="source versions"):
        _performance(
            tmp_path / "performance",
            {"generation": changed_generation},
            plan.cohort_id,
            artifact.id,
            {"nfe"},
        )


@pytest.mark.parametrize(
    "variant,zero,precision",
    [
        ("meanflow", 0, "bf16"),
        ("meanflow", 3, "fp32"),
        ("shortcut", 0, "bf16"),
        ("shortcut", 3, "fp32"),
    ],
)
def test_interval_ema_fp32_weights_exact_resume_and_separate_shortcut_target(
    tmp_path, variant, zero, precision
):
    engine, method, batch = train(variant, zero=zero, precision=precision)
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    store = ArtifactStore(tmp_path / "store")
    ordinary = publish(variant, engine, method, store, tmp_path / "ordinary")
    averaged = publish(variant, engine, method, store, tmp_path / "ema", ema=True)
    expected = engine.export_state_dict(ema=True)
    loaded = load_model(averaged.path / "model")
    for name, value in expected.items():
        torch.testing.assert_close(value, loaded.state_dict()[name], atol=0, rtol=0)
        assert value.dtype == loaded.state_dict()[name].dtype
    assert ordinary.id != averaged.id
    if method is not None:
        assert any(
            not torch.equal(value, method.target.state_dict()[name])
            for name, value in expected.items()
        )
    training = read_json(averaged.path / "generation_contract.json")["training"]
    assert training["weight_selection"] == "ema" and training["ema_decay"] == 0.5
    if method is None:
        engine.step([batch])
    else:
        method.update([batch])
    engine.load_checkpoint(checkpoint, trusted=True)
    restored = publish(variant, engine, method, store, tmp_path / "restored", ema=True)
    assert restored.id == averaged.id
    generate_interval_shard(store, restored.id, plan_for(variant), tmp_path / "images").verify(
        tmp_path / "images"
    )


@pytest.mark.parametrize("variant", ["meanflow", "shortcut"])
def test_interval_latent_decode_and_failed_class_retains_full_cohort(tmp_path, variant):
    engine, method, _ = train(variant, channels=2)
    store = ArtifactStore(tmp_path / "store")
    artifact = publish(variant, engine, method, store, tmp_path / "export")
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
    cases = (GenerationCase("valid", 2, 0), GenerationCase("invalid", 3, 2))
    plan = plan_for(variant, channels=2, cases=cases)
    manifest = generate_interval_shard(
        store, artifact.id, plan, tmp_path / "images", decoder_artifact_id=vae.id
    )
    assert [s.status for s in manifest.samples] == ["ok", "error"]
    assert manifest.producer_artifacts == (artifact.id, vae.id)
    merged = merge_interval_shards([tmp_path / "images"], plan, tmp_path / "merged")
    assert len(_generation_record(tmp_path / "merged", merged)["sample_inputs"]) == 2
    with pytest.raises(ValueError, match="Failed"):
        merged.verify(tmp_path / "merged")
    model = load_model(artifact.path / "model").eval()
    with torch.no_grad():
        noise = torch.randn((1, *plan.noise_shape), generator=torch.Generator().manual_seed(2))
        expected = decoder.decode(independent_sample(model, plan, noise, 0), scaled=True)
    with Image.open(tmp_path / "images" / manifest.samples[0].files[0].path) as image:
        np.testing.assert_array_equal(
            np.asarray(image), quantize_image(expected[0], plan.quantization)
        )
    measured = benchmark_image_sampler(
        store,
        artifact.id,
        plan,
        GenerationBenchmarkSettings(repetitions=2),
        tmp_path / "perf",
        decoder_artifact_id=vae.id,
    )
    assert measured["status"] == "error" and len(measured["records"]) == 4


@pytest.mark.parametrize("variant", ["meanflow", "shortcut"])
def test_interval_contract_and_boundary_rejects_stale_wrong_semantics_and_no_ema(tmp_path, variant):
    engine, method, _ = train(variant, ema=False)
    store = ArtifactStore(tmp_path / "store")
    with pytest.raises(RuntimeError, match="EMA"):
        publish(variant, engine, method, store, tmp_path / "no-ema", ema=True)
    artifact = publish(variant, engine, method, store, tmp_path / "export")
    plan = plan_for(variant)
    contract = read_json(artifact.path / "generation_contract.json")
    for name, change_weights in [("weights", True), ("semantics", False)]:
        model = load_model(artifact.path / "model")
        changed = dict(contract)
        if change_weights:
            with torch.no_grad():
                model.output.weight.add_(0.01)
        else:
            changed["interval_semantics"] = (
                "negative_log2_step" if variant == "meanflow" else "duration"
            )
        model.save_pretrained(tmp_path / name / "model")
        atomic_json(tmp_path / name / "generation_contract.json", changed)
        stale = store.publish(tmp_path / name, kind="invalid_fixture", metadata={})
        with pytest.raises(ValueError):
            generate_interval_shard(store, stale.id, plan, tmp_path / ("reject-" + name))
    if method is not None:
        with pytest.raises(ValueError, match="base_steps"):
            generate_interval_shard(
                store, artifact.id, replace(plan, steps=8), tmp_path / "untrained-level"
            )
        method._incomplete = True
        with pytest.raises(RuntimeError, match="incomplete"):
            publish(variant, engine, method, store, tmp_path / "half-round")
    else:
        engine._failed = True
        with pytest.raises(RuntimeError, match="successful"):
            publish(variant, engine, method, store, tmp_path / "failed")


def test_interval_plan_validation():
    cases = (GenerationCase("x", 1, 0),)
    for times in [(0.0, 1.0), (1.0, 0.2, 0.4, 0.0), (1.0, float("nan"), 0.0), (True, 0.0)]:
        with pytest.raises(ValueError):
            MeanFlowSamplingPlan(cases, (3, 4, 4), times)
    for steps in [0, 3, True]:
        with pytest.raises(ValueError):
            ShortcutSamplingPlan(cases, (3, 4, 4), steps=steps)
    with pytest.raises(ValueError):
        ShortcutSamplingPlan(cases, (3, 4, 4), guidance_scale=float("inf"))
    with pytest.raises(ValueError):
        MeanFlowSamplingPlan((GenerationCase("x", 1, None),), (3, 4, 4))


def _dp_export_worker(rank, rendezvous, directory, variant):
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
        torch.manual_seed(837)
        config = IntervalDiTConfig(
            variant=variant,
            input_size=4,
            in_channels=3,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            num_classes=2,
        )
        objective = MeanFlowObjective() if variant == "meanflow" else None
        engine = Trainer(
            IntervalDiT(config), objective, parallel=context, zero_stage=3, ema_decay=0.5, lr=0.003
        )
        method = (
            ShortcutMethod(engine, base_steps=4, bootstrap_every=2)
            if variant == "shortcut"
            else None
        )
        torch.manual_seed(247 + rank)
        batch = {"sample": torch.randn(4, 3, 4, 4), "labels": torch.tensor([0, 1, 0, 1])}
        assert (method.update([batch]) if method else engine.step([batch])).updated
        root = Path(directory)
        store = ArtifactStore(root / "store")
        rng = torch.get_rng_state().clone()
        artifact = publish(variant, engine, method, store, root / "export", ema=True)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        identities = context.world.gather_objects(artifact.id)
        assert identities[0] == identities[1]
        expected = engine.export_state_dict(ema=True)
        if rank == 0:
            loaded = load_model(artifact.path / "model")
            for name, value in expected.items():
                torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
            plan = plan_for(variant, cases=(GenerationCase("dp", 9, 1),))
            generate_interval_shard(store, artifact.id, plan, root / "images").verify(
                root / "images"
            )
            atomic_json(
                root / "verified.json",
                {"artifact_id": artifact.id, "dp": 2, "zero": 3, "ema": True, "variant": variant},
            )
        context.world.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("variant", ["meanflow", "shortcut"])
def test_interval_dp2_zero3_collective_ema_export(tmp_path, variant):
    base = Path(tempfile.gettempdir())
    if not str(base).isascii():
        base = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    if not base.is_dir():
        pytest.skip("Existing writable ASCII temp directory is required for Windows Gloo")
    scratch = Path(tempfile.mkdtemp(prefix="aster-interval-export-", dir=base)).resolve()
    assert scratch.parent == base.resolve() and scratch.name.startswith("aster-interval-export-")
    try:
        mp.spawn(
            _dp_export_worker,
            args=(str(scratch / "rendezvous"), str(tmp_path), variant),
            nprocs=2,
            join=True,
        )
    finally:
        shutil.rmtree(scratch)
    assert read_json(tmp_path / "verified.json")["variant"] == variant
