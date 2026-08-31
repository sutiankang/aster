from dataclasses import replace
from datetime import timedelta
import math
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
from aster.evaluation.consistency_generation import (
    ConsistencySamplingPlan,
    publish_consistency_generator,
    generate_consistency_shard,
    merge_consistency_shards,
    _tensor_hash,
)
from aster.evaluation.generative import (
    GenerationCase,
    ImageSamplingPlan,
    _generation_record,
    quantize_image,
    generate_image_shard,
)
from aster.evaluation.generation_performance import (
    GenerationBenchmarkSettings,
    benchmark_image_sampler,
)
from aster.evaluation.generation_gate import _performance
from aster.methods.consistency import ConsistencyConfig, ConsistencyMethod
from aster.methods.generation import EDMObjective
from aster.models import load_model
from aster.models.generative import (
    UNet2D,
    UNetConfig,
    DiT,
    DiTConfig,
    AutoencoderKL,
    AutoencoderConfig,
)
from aster.training import Trainer, ParallelContext, ParallelConfig


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def field(
    *, kind="consistency_residual", architecture="unet", channels=3, classes=0, condition_dim=0
):
    if architecture == "dit":
        return DiT(
            DiTConfig(
                in_channels=channels,
                hidden_size=16,
                num_heads=2,
                num_layers=1,
                condition_dim=condition_dim,
                num_classes=classes,
                prediction_type=kind,
            )
        )
    return UNet2D(
        UNetConfig(
            in_channels=channels,
            model_channels=4,
            channel_mult=(1,),
            num_res_blocks=1,
            attention_levels=(),
            num_heads=1,
            num_classes=classes,
            condition_dim=condition_dim,
            prediction_type=kind,
        )
    )


def method_config(mode, *, sampling_ema=0.8):
    return ConsistencyConfig(
        mode=mode,
        total_steps=20,
        sigma_min=0.01,
        sigma_max=2.0,
        sigma_data=0.6,
        time_scale=250.0,
        teacher_time_scale=0.25,
        initial_scales=4,
        final_scales=4,
        curriculum="fixed",
        target_ema_mode="fixed",
        sampling_ema=sampling_ema,
    )


def train(
    tmp_path,
    mode,
    *,
    zero=0,
    precision="fp32",
    architecture="unet",
    channels=3,
    classes=0,
    condition_dim=0,
    ema=0.8,
):
    torch.manual_seed(257)
    store = ArtifactStore(tmp_path / "store")
    teacher_id = None
    teacher = None
    arguments = dict(
        architecture=architecture, channels=channels, classes=classes, condition_dim=condition_dim
    )
    batch = {"sample": torch.randn(3, channels, 4, 4)}
    if condition_dim:
        batch["condition"] = torch.randn(3, condition_dim)
    elif classes:
        batch["condition"] = torch.arange(3) % classes
    if mode == "cd":
        teacher_engine = Trainer(
            field(kind="edm_residual", **arguments), EDMObjective(sigma_data=0.6), lr=0.003
        )
        assert teacher_engine.step([batch]).updated
        teacher = field(kind="edm_residual", **arguments)
        teacher.load_state_dict(teacher_engine.export_state_dict())
        teacher.save_pretrained(tmp_path / "teacher" / "model")
        atomic_json(tmp_path / "teacher" / "objective.json", teacher_engine.objective.config_dict())
        teacher_id = store.publish(
            tmp_path / "teacher", kind="locally_trained_edm_fixture", metadata={}
        ).id
    engine = Trainer(field(**arguments), zero_stage=zero, precision=precision, lr=0.003)
    method = ConsistencyMethod(
        engine,
        target_factory=lambda: field(**arguments),
        config=method_config(mode, sampling_ema=ema),
        teacher=teacher,
    )
    for _ in range(3):
        assert method.update([batch]).updated
    return store, engine, method, batch, teacher_id


def formula(model, plan, config, case):

    rng = torch.Generator().manual_seed(case.seed)
    noise = torch.randn((1, *plan.noise_shape), generator=rng)
    before = _tensor_hash(rng.get_state())
    condition = (
        None
        if case.condition is None
        else torch.tensor(
            [case.condition], dtype=torch.int64 if type(case.condition) is int else torch.float32
        )
    )
    with torch.no_grad():
        x = noise * plan.sigmas[0]
        for index, sigma in enumerate(plan.sigmas):
            level = x.new_tensor([sigma])
            scale = level[:, None, None, None]
            variance = scale.square() + config.sigma_data**2
            delta = scale - config.sigma_min
            residual = model(
                x / variance.sqrt(), level.log() * config.time_scale, condition
            ).prediction
            x = (
                config.sigma_data**2 / (delta.square() + config.sigma_data**2) * x
                + config.sigma_data * delta / variance.sqrt() * residual
            )
            if plan.clip_denoised:
                x = x.clamp(-1, 1)
            if index + 1 < len(plan.sigmas):
                x += math.sqrt(plan.sigmas[index + 1] ** 2 - config.sigma_min**2) * torch.randn(
                    x.shape, generator=rng
                )
    return x, _tensor_hash(noise), before, _tensor_hash(rng.get_state())


@pytest.mark.parametrize(
    "mode,sigmas", [("ct", (2.0,)), ("cd", (2.0, 0.4)), ("ict", (2.0, 0.8, 0.1))]
)
def test_consistency_train_publish_exact_formula_nfe_rng_shards_and_performance(
    tmp_path, mode, sigmas
):
    store, engine, method, _, teacher_id = train(tmp_path, mode, classes=2)
    rng = torch.get_rng_state().clone()
    local_rng = method.generator.get_state().clone()
    artifact = publish_consistency_generator(
        method, store, tmp_path / "export", teacher_artifact_id=teacher_id
    )
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    torch.testing.assert_close(method.generator.get_state(), local_rng, atol=0, rtol=0)
    contract = read_json(artifact.path / "generation_contract.json")
    assert contract["mode"] == mode and contract["sampling_role"] == "consistency_ema"
    assert contract["preconditioning"]["time_scale"] == 250.0
    if mode == "cd":
        assert (
            contract["teacher_artifact"]["artifact_id"] == teacher_id
            and teacher_id in artifact.parents
        )
        assert contract["method_declaration"]["training"]["teacher_time_scale"] == 0.25
    else:
        assert (
            contract["teacher_artifact"] is None
            and contract["method_declaration"]["teacher_sha256"] is None
        )
    cases = tuple(GenerationCase(f"image-{i}", 167 + i, i % 2) for i in range(3))
    plan = ConsistencySamplingPlan(cases, (3, 4, 4), sigmas, clip_denoised=False)
    manifest = generate_consistency_shard(store, artifact.id, plan, tmp_path / "single").verify(
        tmp_path / "single"
    )
    record = _generation_record(tmp_path / "single", manifest)
    model = load_model(artifact.path / "model").eval()
    for index, case in enumerate(cases):
        expected, noise_hash, before, after = formula(model, plan, method.config, case)
        with Image.open(tmp_path / "single" / manifest.samples[index].files[0].path) as image:
            np.testing.assert_array_equal(
                np.asarray(image), quantize_image(expected[0], plan.quantization)
            )
        evidence = record["sample_inputs"][index]
        assert (
            evidence["initial_noise_sha256"] == noise_hash
            and evidence["rng_before_sampling_sha256"] == before
        )
        assert evidence["rng_after_sampling_sha256"] == after and evidence["nfe"] == len(sigmas)
        assert (before == after) == (len(sigmas) == 1)
    parts = [tmp_path / f"rank-{rank}" for rank in range(4)]
    for rank, path in enumerate(parts):
        generate_consistency_shard(store, artifact.id, plan, path, rank=rank, world_size=4)
    merged = merge_consistency_shards(parts[::-1], plan, tmp_path / "merged")
    assert manifest.samples == merged.samples
    assert (
        record["sample_inputs"] == _generation_record(tmp_path / "merged", merged)["sample_inputs"]
    )
    assert plan.cohort_id == ImageSamplingPlan(cases, (3, 4, 4)).cohort_id
    performance = benchmark_image_sampler(
        store,
        artifact.id,
        plan,
        GenerationBenchmarkSettings(repetitions=2, isolated_hardware_asserted=True),
        tmp_path / "perf",
    )
    assert performance["status"] == "ok"
    assert all(
        row["nfe"] == len(sigmas) and row["latency_seconds"] > 0 for row in performance["records"]
    )
    assert digest_json(performance["sampling_binding"]) == digest_json(record["sampling_binding"])
    assert performance["native_producer_sources"] == record["native_producer_sources"]
    _, _, values = _performance(
        tmp_path / "perf",
        {"generation": record},
        plan.cohort_id,
        artifact.id,
        {"nfe", "latency_seconds"},
    )
    assert values["nfe"] == len(sigmas)


@pytest.mark.parametrize("architecture,zero,precision", [("unet", 3, "bf16"), ("dit", 0, "fp32")])
def test_consistency_ordinary_vs_sampling_ema_exact_restore_and_native_architectures(
    tmp_path, architecture, zero, precision
):
    store, engine, method, batch, _ = train(
        tmp_path, "ict", zero=zero, precision=precision, architecture=architecture, condition_dim=2
    )
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    model_artifact = publish_consistency_generator(
        method, store, tmp_path / "ordinary", sampling_role="model"
    )
    ema_artifact = publish_consistency_generator(method, store, tmp_path / "ema")
    assert model_artifact.id != ema_artifact.id
    expected = engine.export_state_dict(role="consistency_ema")
    restored = load_model(ema_artifact.path / "model")
    for name, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)

    assert any(
        not torch.equal(value, method.target.state_dict()[name]) for name, value in expected.items()
    )
    method.update([batch])
    engine.load_checkpoint(checkpoint, trusted=True)
    again = publish_consistency_generator(method, store, tmp_path / "restored")
    assert again.id == ema_artifact.id
    plan = ConsistencySamplingPlan(
        (GenerationCase("vector", 91, (0.2, -0.3)),), (3, 4, 4), (2.0, 0.3)
    )
    generate_consistency_shard(store, again.id, plan, tmp_path / "image").verify(tmp_path / "image")


def test_consistency_latent_decoder_and_missing_ema(tmp_path):
    store, engine, method, _, _ = train(tmp_path, "ict", channels=2, ema=None)
    artifact = publish_consistency_generator(method, store, tmp_path / "export")
    assert read_json(artifact.path / "generation_contract.json")["sampling_role"] == "model"
    with pytest.raises(RuntimeError, match="existing sampling EMA"):
        publish_consistency_generator(
            method, store, tmp_path / "missing", sampling_role="consistency_ema"
        )
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
    vae = store.publish(tmp_path / "decoder", kind="native_decoder_fixture", metadata={})
    case = GenerationCase("latent", 199)
    plan = ConsistencySamplingPlan((case,), (2, 4, 4), (2.0, 0.4), clip_denoised=False)
    manifest = generate_consistency_shard(
        store, artifact.id, plan, tmp_path / "image", decoder_artifact_id=vae.id
    ).verify(tmp_path / "image")
    model = load_model(artifact.path / "model").eval()
    with torch.no_grad():
        expected = decoder.decode(formula(model, plan, method.config, case)[0], scaled=True)
    with Image.open(tmp_path / "image" / manifest.samples[0].files[0].path) as image:
        np.testing.assert_array_equal(
            np.asarray(image), quantize_image(expected[0], plan.quantization)
        )
    assert manifest.producer_artifacts == (artifact.id, vae.id)


def test_consistency_rejects_teacher_target_stale_weights_and_preserves_failures(
    tmp_path, monkeypatch
):
    store, engine, method, _, teacher_id = train(tmp_path, "cd", classes=2)
    for role in ("consistency_target", "consistency_teacher"):
        with pytest.raises(RuntimeError, match="never target/teacher"):
            publish_consistency_generator(method, store, tmp_path / role, sampling_role=role)
    with monkeypatch.context() as edit:
        edit.setattr(engine.roles["consistency_ema"], "model", field(classes=2))
        with pytest.raises(RuntimeError, match="owned role"):
            publish_consistency_generator(method, store, tmp_path / "replaced-owner")
    if hasattr(engine.parallel.config, "expert_parallel"):
        with monkeypatch.context() as edit:
            edit.setattr(
                engine.parallel,
                "config",
                replace(engine.parallel.config, data_parallel=2, expert_parallel=2),
            )
            with pytest.raises(RuntimeError, match="not EP"):
                publish_consistency_generator(method, store, tmp_path / "unsupported-ep")
    wrong = field(kind="edm_residual", classes=2)
    wrong.save_pretrained(tmp_path / "wrong-teacher" / "model")
    wrong_id = store.publish(
        tmp_path / "wrong-teacher", kind="wrong_teacher_fixture", metadata={}
    ).id
    with pytest.raises(RuntimeError, match="actual frozen CD teacher"):
        publish_consistency_generator(
            method, store, tmp_path / "bad-teacher", teacher_artifact_id=wrong_id
        )
    artifact = publish_consistency_generator(
        method, store, tmp_path / "export", teacher_artifact_id=teacher_id
    )
    plan = ConsistencySamplingPlan(
        (GenerationCase("good", 1, 1), GenerationCase("bad", 2, 2)), (3, 4, 4), (2.0,)
    )
    manifest = generate_consistency_shard(store, artifact.id, plan, tmp_path / "images")
    assert [s.status for s in manifest.samples] == ["ok", "error"]
    merged = merge_consistency_shards([tmp_path / "images"], plan, tmp_path / "merged")
    with pytest.raises(ValueError, match="Failed"):
        merged.verify(tmp_path / "merged")
    assert len(_generation_record(tmp_path / "merged", merged)["sample_inputs"]) == 2
    performance = benchmark_image_sampler(
        store,
        artifact.id,
        plan,
        GenerationBenchmarkSettings(repetitions=2),
        tmp_path / "failed-perf",
    )
    assert performance["status"] == "error" and len(performance["records"]) == 4
    for sigmas in [(1.0,), (2.0, 0.001)]:
        with pytest.raises(ValueError, match="sigma"):
            generate_consistency_shard(
                store, artifact.id, replace(plan, sigmas=sigmas), tmp_path / f"reject-{sigmas[-1]}"
            )
    with pytest.raises(ValueError):
        generate_image_shard(
            store,
            artifact.id,
            ImageSamplingPlan(plan.cases, plan.noise_shape, sampler="direct_x0", steps=1),
            tmp_path / "wrong-sampler",
        )
    for name in ("weights", "role", "time"):
        model = load_model(artifact.path / "model")
        contract = read_json(artifact.path / "generation_contract.json")
        if name == "weights":
            with torch.no_grad():
                next(model.parameters()).add_(0.01)
        elif name == "role":
            contract["sampling_role"] = "consistency_target"
        else:
            contract["preconditioning"]["time_scale"] = 0.25
        model.save_pretrained(tmp_path / name / "model")
        atomic_json(tmp_path / name / "generation_contract.json", contract)
        atomic_json(
            tmp_path / name / "consistency.json", read_json(artifact.path / "consistency.json")
        )
        stale = store.publish(
            tmp_path / name, kind="invalid_fixture", metadata={}, parents=artifact.parents
        )
        with pytest.raises(ValueError):
            generate_consistency_shard(store, stale.id, plan, tmp_path / ("bad-" + name))
    method._incomplete = True
    with pytest.raises(RuntimeError, match="incomplete"):
        publish_consistency_generator(method, store, tmp_path / "incomplete")


def test_consistency_plan_validation():
    cases = (GenerationCase("x", 1),)
    for sigmas in [(), (0.0,), (2.0, 3.0), (2.0, 2.0), (True,), (2.0, float("nan"))]:
        with pytest.raises(ValueError):
            ConsistencySamplingPlan(cases, (3, 4, 4), sigmas)
    with pytest.raises(ValueError):
        ConsistencySamplingPlan(cases, (3, 4, 4), (2.0,), clip_denoised=1)


def _dp_export_worker(rank, rendezvous, directory, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(777)
        context = ParallelContext(ParallelConfig(data_parallel=2))
        teacher = field(kind="edm_residual") if mode == "cd" else None
        engine = Trainer(field(), parallel=context, zero_stage=3, lr=0.003)
        method = ConsistencyMethod(
            engine, target_factory=field, config=method_config(mode), teacher=teacher
        )
        torch.manual_seed(127 + rank)
        assert method.update([{"sample": torch.randn(2 + rank, 3, 4, 4)}]).updated
        root = Path(directory)
        store = ArtifactStore(root / "store")
        rng = torch.get_rng_state().clone()
        artifact = publish_consistency_generator(method, store, root / "export")
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        identities = context.world.gather_objects(artifact.id)
        assert identities[0] == identities[1]
        expected = engine.export_state_dict(role="consistency_ema")
        if rank == 0:
            model = load_model(artifact.path / "model")
            for name, value in expected.items():
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
            plan = ConsistencySamplingPlan((GenerationCase("dp", 151),), (3, 4, 4), (2.0, 0.2))
            generate_consistency_shard(store, artifact.id, plan, root / "images").verify(
                root / "images"
            )
            atomic_json(
                root / "verified.json",
                {"mode": mode, "dp": 2, "zero": 3, "artifact_id": artifact.id},
            )
        context.world.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["ct", "cd", "ict"])
def test_consistency_dp2_zero3_collective_sampling_ema_export(tmp_path, mode):
    base = Path(tempfile.gettempdir())
    if not str(base).isascii():
        base = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    if not base.is_dir():
        pytest.skip("Existing writable ASCII temp directory required for Windows Gloo")
    scratch = Path(tempfile.mkdtemp(prefix="aster-consistency-export-", dir=base)).resolve()
    assert scratch.parent == base.resolve() and scratch.name.startswith("aster-consistency-export-")
    try:
        mp.spawn(
            _dp_export_worker,
            args=(str(scratch / "rendezvous"), str(tmp_path), mode),
            nprocs=2,
            join=True,
        )
    finally:
        shutil.rmtree(scratch)
    assert read_json(tmp_path / "verified.json")["mode"] == mode
