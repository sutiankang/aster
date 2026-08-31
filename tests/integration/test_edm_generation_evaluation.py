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

from aster.core import ArtifactStore, atomic_json, read_json
from aster.models import load_model
from aster.models.generative import (
    UNet2D,
    UNetConfig,
    DiT,
    DiTConfig,
    AutoencoderKL,
    AutoencoderConfig,
)
from aster.training import Trainer, ParallelConfig, ParallelContext
from aster.methods.generation import EDMObjective
from aster.methods.consistency import ConsistencyConfig, ConsistencyMethod
from aster.pipelines import LatentFieldObjective
from aster.evaluation.edm_generation import (
    EDMSamplingPlan,
    edm_nfe,
    edm_binding,
    publish_edm_generator,
    generate_edm_shard,
    merge_edm_shards,
    validate_consistency_teacher_baseline,
    _tensor_hash,
)
from aster.evaluation.consistency_generation import (
    ConsistencySamplingPlan,
    publish_consistency_generator,
    generate_consistency_shard,
)
from aster.evaluation.generative import GenerationCase, _generation_record, quantize_image
from aster.evaluation.generation_performance import (
    GenerationBenchmarkSettings,
    benchmark_image_sampler,
)
from aster.evaluation.generation_gate import (
    _performance,
    GenerationGateProtocol,
    evaluate_generation_gate,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def field(*, kind="edm_residual", architecture="unet", channels=3, classes=0):
    if architecture == "dit":
        return DiT(
            DiTConfig(
                in_channels=channels,
                hidden_size=16,
                num_heads=2,
                num_layers=1,
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
            prediction_type=kind,
        )
    )


def train(tmp_path, *, architecture="unet", zero=0, precision="fp32", classes=0):
    torch.manual_seed(823)
    store = ArtifactStore(tmp_path / "store")
    engine = Trainer(
        field(architecture=architecture, classes=classes),
        EDMObjective(sigma_data=0.6),
        lr=0.003,
        zero_stage=zero,
        precision=precision,
        ema_decay=0.8,
    )
    batch = {
        "sample": torch.randn(2, 3, 4, 4),
        "sigma": torch.tensor([0.2, 1.0]),
        "noise": torch.randn(2, 3, 4, 4),
    }
    if classes:
        batch["condition"] = torch.arange(2) % classes
    for _ in range(2):
        assert engine.step([batch]).updated
    return store, engine, batch


def formula(model, plan, case, sigma_data):

    rng = torch.Generator().manual_seed(case.seed)
    noise = torch.randn((1, *plan.noise_shape), generator=rng)
    before = _tensor_hash(rng.get_state())
    levels = torch.tensor(plan.sigmas, dtype=torch.float64)
    condition = (
        None if case.condition is None else torch.tensor([case.condition], dtype=torch.int64)
    )

    def denoise(x, level):
        t = level.to(x).expand(len(x))
        s = t[:, None, None, None]
        variance = s.square() + sigma_data**2
        residual = model(x / variance.sqrt(), t.log() / 4, condition).prediction
        return sigma_data**2 / variance * x + s * sigma_data / variance.sqrt() * residual

    with torch.no_grad():
        x = noise * levels[0].to(noise)
        for level, next_level in zip(levels[:-1], levels[1:]):
            gamma = (
                min(plan.churn / (len(levels) - 1), math.sqrt(2) - 1)
                if plan.churn_min
                <= level
                <= (math.inf if plan.churn_max is None else plan.churn_max)
                else 0.0
            )
            augmented = level * (1 + gamma)
            x = x + (augmented.square() - level.square()).clamp_min(0).sqrt().to(
                x
            ) * plan.noise_scale * torch.randn(x.shape, generator=rng)
            slope = (x - denoise(x, augmented)) / augmented.to(x)
            proposal = x + (next_level - augmented).to(x) * slope
            if next_level > 0:
                slope2 = (proposal - denoise(proposal, next_level)) / next_level.to(x)
                x = x + (next_level - augmented).to(x) * (slope + slope2) / 2
            else:
                x = proposal
    return x, _tensor_hash(noise), before, _tensor_hash(rng.get_state())


@pytest.mark.parametrize("architecture,churn", [("unet", 0.0), ("dit", 0.5)])
def test_edm_actual_train_formula_churn_rng_nfe_png_shards_and_performance(
    tmp_path, architecture, churn
):
    store, engine, _ = train(tmp_path, architecture=architecture, classes=2)
    rng = torch.get_rng_state().clone()
    artifact = publish_edm_generator(engine, store, tmp_path / "export", ema=True)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    cases = tuple(GenerationCase(f"image-{i}", 700 + i, i % 2) for i in range(3))
    plan = EDMSamplingPlan(
        cases,
        (3, 4, 4),
        (2.0, 0.6, 0.1, 0.0),
        churn=churn,
        churn_min=0.2,
        churn_max=2.0,
        noise_scale=1.1,
    )
    manifest = generate_edm_shard(store, artifact.id, plan, tmp_path / "single").verify(
        tmp_path / "single"
    )
    record = _generation_record(tmp_path / "single", manifest)
    model = load_model(artifact.path / "model").eval()
    assert (
        edm_nfe(plan) == 5 and record["sampling_binding"]["preconditioning"]["time_scale"] == 0.25
    )
    for index, case in enumerate(cases):
        expected, noise_hash, before, after = formula(model, plan, case, 0.6)
        with Image.open(tmp_path / "single" / manifest.samples[index].files[0].path) as image:
            np.testing.assert_array_equal(
                np.asarray(image), quantize_image(expected[0], plan.quantization)
            )
        row = record["sample_inputs"][index]
        assert (
            row["initial_noise_sha256"] == noise_hash
            and row["rng_before_sampling_sha256"] == before
        )
        assert row["rng_after_sampling_sha256"] == after and before != after and row["nfe"] == 5
    paths = [tmp_path / f"rank-{rank}" for rank in range(4)]
    for rank, path in enumerate(paths):
        generate_edm_shard(store, artifact.id, plan, path, rank=rank, world_size=4)
    merged = merge_edm_shards(paths[::-1], plan, tmp_path / "merged")
    assert (
        merged.samples == manifest.samples
        and _generation_record(tmp_path / "merged", merged)["sample_inputs"]
        == record["sample_inputs"]
    )
    perf = benchmark_image_sampler(
        store,
        artifact.id,
        plan,
        GenerationBenchmarkSettings(repetitions=2, isolated_hardware_asserted=True),
        tmp_path / "perf",
    )
    assert perf["status"] == "ok" and all(
        row["nfe"] == 5 and row["latency_seconds"] > 0 for row in perf["records"]
    )
    assert (
        perf["sampling_binding"] == record["sampling_binding"]
        and perf["native_producer_sources"] == record["native_producer_sources"]
    )
    _, _, resources = _performance(
        tmp_path / "perf",
        {"generation": record},
        plan.cohort_id,
        artifact.id,
        {"nfe", "latency_seconds"},
    )
    assert resources["nfe"] == 5


def test_edm_zero3_bf16_ordinary_ema_exact_checkpoint_restore_and_stale_contract(tmp_path):
    store, engine, batch = train(tmp_path, zero=3, precision="bf16")
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    ordinary = publish_edm_generator(engine, store, tmp_path / "ordinary")
    ema = publish_edm_generator(engine, store, tmp_path / "ema", ema=True)
    assert ordinary.id != ema.id
    for name, value in engine.export_state_dict(ema=True).items():
        torch.testing.assert_close(
            load_model(ema.path / "model").state_dict()[name], value, atol=0, rtol=0
        )
    engine.step([batch])
    engine.load_checkpoint(checkpoint, trusted=True)
    assert publish_edm_generator(engine, store, tmp_path / "restored", ema=True).id == ema.id
    model = load_model(ema.path / "model")
    contract = read_json(ema.path / "generation_contract.json")
    with torch.no_grad():
        next(model.parameters()).add_(0.1)
    model.save_pretrained(tmp_path / "stale" / "model")
    atomic_json(tmp_path / "stale" / "generation_contract.json", contract)
    atomic_json(tmp_path / "stale" / "objective.json", read_json(ema.path / "objective.json"))
    atomic_json(
        tmp_path / "stale" / "successful_update.json",
        read_json(ema.path / "successful_update.json"),
    )
    stale = store.publish(tmp_path / "stale", kind="stale_fixture", metadata={})
    with pytest.raises(ValueError, match="weights differ"):
        generate_edm_shard(
            store,
            stale.id,
            EDMSamplingPlan((GenerationCase("x", 1),), (3, 4, 4), (2.0, 0.1, 0.0)),
            tmp_path / "bad",
        )
    engine._failed = True
    with pytest.raises(RuntimeError, match="successful idle"):
        publish_edm_generator(engine, store, tmp_path / "partial")


def test_edm_latent_frozen_encoder_binding_real_decode_and_reject_wrong_decoder(tmp_path):
    torch.manual_seed(873)
    store = ArtifactStore(tmp_path / "store")
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
    )
    decoder.save_pretrained(tmp_path / "decoder")
    va = store.publish(tmp_path / "decoder", kind="native_vae_fixture", metadata={})
    objective = LatentFieldObjective(
        decoder, EDMObjective(sigma_data=0.6), encoder_identity=va.id, sample_posterior=False
    )
    engine = Trainer(field(channels=2), objective, lr=0.003)
    engine.add_role("autoencoder", decoder, trainable=False)
    assert engine.step([{"pixels": torch.randn(2, 3, 4, 4)}]).updated
    artifact = publish_edm_generator(engine, store, tmp_path / "export")
    assert va.id in artifact.parents
    case = GenerationCase("latent", 402)
    plan = EDMSamplingPlan((case,), (2, 4, 4), (2.0, 0.2, 0.0))
    with pytest.raises(ValueError, match="decoder differs"):
        generate_edm_shard(store, artifact.id, plan, tmp_path / "missing")
    manifest = generate_edm_shard(
        store, artifact.id, plan, tmp_path / "images", decoder_artifact_id=va.id
    ).verify(tmp_path / "images")
    expected = formula(load_model(artifact.path / "model").eval(), plan, case, 0.6)[0]
    with torch.no_grad():
        expected = decoder.decode(expected, scaled=True)
    with Image.open(tmp_path / "images" / manifest.samples[0].files[0].path) as image:
        np.testing.assert_array_equal(
            np.asarray(image), quantize_image(expected[0], plan.quantization)
        )
    perf = benchmark_image_sampler(
        store,
        artifact.id,
        plan,
        GenerationBenchmarkSettings(repetitions=2),
        tmp_path / "perf",
        decoder_artifact_id=va.id,
    )
    assert (
        perf["status"] == "ok"
        and perf["sampling_binding"]
        == _generation_record(tmp_path / "images", manifest)["sampling_binding"]
    )
    with torch.no_grad():
        next(decoder.parameters()).add_(0.1)
    with pytest.raises(RuntimeError, match="actual frozen training encoder"):
        publish_edm_generator(engine, store, tmp_path / "false-encoder")


def test_actual_teacher_heun_and_cd_student_share_cohort_inputs_without_fake_public_scores(
    tmp_path,
):
    store, teacher_engine, batch = train(tmp_path)
    teacher = publish_edm_generator(teacher_engine, store, tmp_path / "teacher", ema=True)
    student_engine = Trainer(field(kind="consistency_residual"), lr=0.003)
    config = ConsistencyConfig(
        mode="cd",
        total_steps=8,
        initial_scales=4,
        final_scales=4,
        curriculum="fixed",
        sigma_min=0.01,
        sigma_max=2.0,
        sigma_data=0.6,
        target_ema_mode="fixed",
        sampling_ema=0.8,
    )
    method = ConsistencyMethod(
        student_engine,
        target_factory=lambda: field(kind="consistency_residual"),
        config=config,
        teacher=load_model(teacher.path / "model"),
    )

    cd_batch = {key: value for key, value in batch.items() if key != "sigma"}
    for _ in range(2):
        assert method.update([cd_batch]).updated
    student = publish_consistency_generator(
        method, store, tmp_path / "student", teacher_artifact_id=teacher.id
    )
    cases = (GenerationCase("comparison-a", 293), GenerationCase("comparison-b", 294))
    teacher_plan = EDMSamplingPlan(cases, (3, 4, 4), (2.0, 0.5, 0.1, 0.0))
    student_plan = ConsistencySamplingPlan(cases, (3, 4, 4), (2.0,), clip_denoised=False)
    binding = validate_consistency_teacher_baseline(
        store, teacher.id, student.id, teacher_plan, student_plan
    )
    assert not binding["quality_and_performance_evaluated"]
    baseline = generate_edm_shard(store, teacher.id, teacher_plan, tmp_path / "baseline")
    candidate = generate_consistency_shard(store, student.id, student_plan, tmp_path / "candidate")
    left = _generation_record(tmp_path / "baseline", baseline)
    right = _generation_record(tmp_path / "candidate", candidate)
    assert [r["initial_noise_sha256"] for r in left["sample_inputs"]] == [
        r["initial_noise_sha256"] for r in right["sample_inputs"]
    ]
    assert all(r["nfe"] == 5 for r in left["sample_inputs"]) and all(
        r["nfe"] == 1 for r in right["sample_inputs"]
    )
    for label, artifact, plan, record in [
        ("teacher", teacher, teacher_plan, left),
        ("student", student, student_plan, right),
    ]:
        benchmark_image_sampler(
            store,
            artifact.id,
            plan,
            GenerationBenchmarkSettings(repetitions=2, isolated_hardware_asserted=True),
            tmp_path / f"perf-{label}",
        )
        _performance(
            tmp_path / f"perf-{label}",
            {"generation": record},
            plan.cohort_id,
            artifact.id,
            {"latency_seconds", "nfe"},
        )
    with pytest.raises(ValueError, match="share cohort"):
        validate_consistency_teacher_baseline(
            store,
            teacher.id,
            student.id,
            teacher_plan,
            replace(student_plan, cases=(GenerationCase("changed", 1),)),
        )

    cohorts = (teacher_plan.cohort_id, "1" * 64, "2" * 64)
    gate = GenerationGateProtocol(teacher.id, student.id, cohorts)
    result = evaluate_generation_gate(
        gate,
        {c: ("unused", "unused") for c in cohorts},
        {c: ("unused", "unused") for c in cohorts},
        output_directory=tmp_path / "gate",
    )
    assert result["status"] == "not_evaluated" and result["passed"] is False


def test_edm_failure_population_and_plan_validation(tmp_path):
    store, engine, _ = train(tmp_path, classes=2)
    artifact = publish_edm_generator(engine, store, tmp_path / "export")
    plan = EDMSamplingPlan(
        (GenerationCase("ok", 1, 0), GenerationCase("bad", 2, 2)), (3, 4, 4), (2.0, 0.1, 0.0)
    )
    manifest = generate_edm_shard(store, artifact.id, plan, tmp_path / "images")
    assert [s.status for s in manifest.samples] == ["ok", "error"]
    merged = merge_edm_shards([tmp_path / "images"], plan, tmp_path / "merged")
    with pytest.raises(ValueError, match="Failed"):
        merged.verify(tmp_path / "merged")
    assert len(_generation_record(tmp_path / "merged", merged)["sample_inputs"]) == 2
    perf = benchmark_image_sampler(
        store, artifact.id, plan, GenerationBenchmarkSettings(repetitions=2), tmp_path / "perf"
    )
    assert perf["status"] == "error" and len(perf["records"]) == 4
    for sigmas in (
        (2.0, 0.1),
        (2.0, 0.0),
        (2.0, 0.1, 0.1, 0.0),
        (math.nan, 0.1, 0.0),
        (True, 0.1, 0.0),
    ):
        with pytest.raises(ValueError):
            replace(plan, sigmas=sigmas)
    for options in (
        {"churn": math.inf},
        {"noise_scale": False},
        {"churn_max": math.inf},
        {"churn_min": 2.0, "churn_max": 1.0},
    ):
        with pytest.raises(ValueError):
            replace(plan, **options)
    raw = field(kind="consistency_residual")
    raw.save_pretrained(tmp_path / "wrong")
    atomic_json(
        tmp_path / "wrong" / "objective.json",
        {"type": "edm", "sigma_data": 0.6, "log_mean": -1.2, "log_std": 1.2},
    )
    wrong = store.publish(tmp_path / "wrong", kind="wrong_parameterization", metadata={})
    with pytest.raises(ValueError, match="native EDM residual"):
        generate_edm_shard(store, wrong.id, plan, tmp_path / "wrong-image")


def _dp_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file:///" + rendezvous.replace("\\", "/"),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        context = ParallelContext(ParallelConfig(data_parallel=2))
        torch.manual_seed(779)
        engine = Trainer(
            field(),
            EDMObjective(sigma_data=0.6),
            parallel=context,
            zero_stage=3,
            lr=0.003,
            ema_decay=0.8,
        )
        torch.manual_seed(901 + rank)
        assert engine.step([{"sample": torch.randn(2 + rank, 3, 4, 4)}]).updated
        root = Path(directory)
        store = ArtifactStore(root / "store")
        artifact = publish_edm_generator(engine, store, root / "export", ema=True)
        ids = context.world.gather_objects(artifact.id)
        assert ids[0] == ids[1]
        expected = engine.export_state_dict(ema=True)
        if rank == 0:
            actual = load_model(artifact.path / "model")
            for name, value in expected.items():
                torch.testing.assert_close(actual.state_dict()[name], value, atol=0, rtol=0)
            plan = EDMSamplingPlan((GenerationCase("dp", 26),), (3, 4, 4), (2.0, 0.1, 0.0))
            generate_edm_shard(store, artifact.id, plan, root / "images").verify(root / "images")
            atomic_json(root / "verified.json", {"artifact": artifact.id, "dp": 2, "zero": 3})
        context.world.barrier()
    finally:
        dist.destroy_process_group()


def test_edm_real_dp2_zero3_collective_ema_export(tmp_path):
    base = Path(tempfile.gettempdir())
    if not str(base).isascii():
        base = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    if not base.is_dir():
        pytest.skip("Existing writable ASCII temporary directory needed by Windows Gloo")
    scratch = Path(tempfile.mkdtemp(prefix="aster-edm-export-", dir=base)).resolve()
    assert scratch.parent == base.resolve() and scratch.name.startswith("aster-edm-export-")
    try:
        mp.spawn(_dp_worker, args=(str(scratch / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    finally:
        shutil.rmtree(scratch)
    assert read_json(tmp_path / "verified.json")["dp"] == 2
