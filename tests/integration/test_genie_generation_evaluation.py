from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from aster.core import ArtifactStore, atomic_json, digest_json, file_digest, read_json
from aster.models.genie import (
    GenieTokenizerConfig,
    GenieTokenizer,
    GenieActionConfig,
    GenieDynamicsConfig,
    GenieWorldConfig,
    GenieWorld,
)
from aster.methods.genie import GenieVQObjective, GenieWorldObjective
from aster.methods.genie_artifact_training import (
    GenieVideoSpec,
    publish_genie_videos,
    publish_genie_tokenizer,
    tokenize_genie_artifact,
    TokenizedGenieData,
    BoundGenieWorldObjective,
    publish_genie_world,
    load_trained_genie,
    tensor_identity,
)
from aster.training import Trainer
from aster.evaluation.genie_generation import (
    GenieGenerationCase,
    GenieSamplingPlan,
    generate_genie_shard,
    merge_genie_shards,
    evaluate_genie_controls,
    benchmark_genie_sampler,
    evaluate_genie_fvd,
    compare_genie_cohorts,
    publish_genie_generation,
)
from aster.evaluation.generative import MediaManifest, _generation_record
from aster.evaluation.generation_performance import GenerationBenchmarkSettings
from aster.evaluation.genie_world import evaluate_genie_controllability
from aster.planning.genie import generate_genie_video


def prepare(tmp_path, *, zero_stage=0):
    torch.set_num_threads(1)
    torch.manual_seed(831)
    common = dict(
        image_height=4,
        image_width=4,
        image_channels=3,
        hidden_size=8,
        num_heads=2,
        head_dim=4,
        encoder_layers=1,
        decoder_hidden_size=8,
        decoder_num_heads=2,
        decoder_head_dim=4,
        decoder_layers=1,
        latent_dim=3,
        max_frames=4,
        intermediate_ratio=2,
    )
    tc = GenieTokenizerConfig(**common, patch_size=2, num_codes=5)
    wc = GenieWorldConfig(
        GenieActionConfig(**common, patch_size=4, num_codes=3),
        GenieDynamicsConfig(
            spatial_tokens=4,
            vocab_size=5,
            action_dim=3,
            hidden_size=8,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            intermediate_ratio=2,
            max_frames=4,
        ),
    )
    store = ArtifactStore(tmp_path / "store")
    video = torch.rand(2, 4, 3, 4, 4)
    codec_engine = Trainer(
        GenieTokenizer(tc),
        GenieVQObjective(sequence_length=4),
        zero_stage=zero_stage,
        ema_decay=0.9,
        lr=0.001,
    )
    assert codec_engine.step([{"video": video}]).updated
    codec = publish_genie_tokenizer(codec_engine, store, tmp_path / "codec", ema=True)
    cases = {
        str(i): {"video": video[i], "valid": torch.ones(4, dtype=torch.bool)} for i in range(2)
    }
    source = publish_genie_videos(
        store,
        cases,
        tmp_path / "data",
        spec=GenieVideoSpec("synthetic_fixture", "fixture-v1", "train", "local_test_only", 8.0),
    )
    trace = tokenize_genie_artifact(store, codec.id, source.id, tmp_path / "trace")
    objective = BoundGenieWorldObjective(store, trace.id)
    engine = Trainer(
        GenieWorld(wc),
        objective,
        zero_stage=zero_stage,
        accumulation_steps=2,
        ema_decay=0.9,
        lr=0.001,
    )
    batches = [objective.batch([0]), objective.batch([1])]
    assert engine.step(batches).updated
    return store, codec_engine, codec, source, trace, engine, batches


@pytest.mark.parametrize("zero_stage", [0, 3])
def test_genie_trained_codec_trace_world_publish_and_exact_restore(tmp_path, zero_stage):
    store, codec_engine, codec, source, trace, engine, batches = prepare(
        tmp_path, zero_stage=zero_stage
    )
    rng = torch.get_rng_state().clone()
    artifact = publish_genie_world(engine, store, tmp_path / "world", ema=True)
    assert torch.equal(rng, torch.get_rng_state())
    model, contract = load_trained_genie(store, artifact.id, world=True)
    assert contract["sampling_role"] == "ema" and artifact.parents == (trace.id,)
    assert contract["objective"]["tokenizer_artifact_id"] == codec.id
    expected = engine.export_state_dict(ema=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    first = engine.step(batches)
    state = deepcopy(engine.export_state_dict())
    engine.load_checkpoint(checkpoint, trusted=True)
    second = engine.step(batches)
    assert first.loss == second.loss and first.updated and second.updated
    for name, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    engine.load_checkpoint(checkpoint, trusted=True)
    assert publish_genie_world(engine, store, tmp_path / "world_again", ema=True).id == artifact.id

    engine.objective.objective.dynamics_weight = 0.4
    with pytest.raises((ValueError, RuntimeError), match="objective|Objective"):
        publish_genie_world(engine, store, tmp_path / "wrong_objective")


def test_genie_tokenization_rejects_rehashed_wrong_codes_and_changed_pixels(tmp_path):
    store, _, codec, source, trace, engine, batches = prepare(tmp_path)
    trace_data = TokenizedGenieData(store, trace.id)
    assert torch.equal(trace_data.load("0")["tokens"], batches[0]["tokens"][0])

    target = tmp_path / "forged"
    target.mkdir()
    manifest = read_json(trace.path / "tokenization.json")
    for key, entry in manifest["entries"].items():
        saved = torch.load(trace.path / entry["path"], weights_only=True)
        if key == "0":
            saved["tokens"] = (saved["tokens"] + 1) % 5
        with (target / entry["path"]).open("xb") as stream:
            torch.save(saved, stream)
        entry["sha256"] = file_digest(target / entry["path"])
        entry["tokens"] = tensor_identity(saved["tokens"])
    atomic_json(target / "tokenization.json", manifest)
    fake = store.publish(
        target, kind=trace.kind, metadata={"trace_id": digest_json(manifest)}, parents=trace.parents
    )
    with pytest.raises(ValueError, match="actual pinned codec"):
        TokenizedGenieData(store, fake.id)
    before = deepcopy(engine.export_state_dict())
    calls = []
    hook = engine.roles["model"].model.register_forward_pre_hook(lambda *args: calls.append(1))
    try:
        broken = deepcopy(batches)
        broken[1]["video"][0, 0, 0, 0, 0] += 0.001
        with pytest.raises((ValueError, RuntimeError), match="bound video/token"):
            engine.step(broken)
        assert not calls
    finally:
        hook.remove()
    for name, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_genie_plain_world_objective_cannot_claim_artifact_training(tmp_path):
    store, _, _, _, _, engine, batches = prepare(tmp_path)
    engine.objective = GenieWorldObjective(sequence_length=4)
    with pytest.raises((ValueError, RuntimeError), match="actual objective"):
        publish_genie_world(engine, store, tmp_path / "plain")
    with pytest.raises(ValueError):
        GenieVideoSpec("data", "main", "test", "license", 8.0)


def test_genie_pure_dp_deployment_explicitly_rejects_etp(tmp_path):
    store, _, _, _, _, engine, _ = prepare(tmp_path)
    engine.parallel.config = SimpleNamespace(expert_parallel=1, expert_tensor_parallel=2)
    with pytest.raises((ValueError, RuntimeError), match="dense DP/ZeRO"):
        publish_genie_world(engine, store, tmp_path / "bad_etp")


def test_genie_true_png_two_trajectories_shards_nfe_and_float_metric(tmp_path):
    store, _, codec, source, _, engine, _ = prepare(tmp_path)
    world_artifact = publish_genie_world(engine, store, tmp_path / "world")
    plan = GenieSamplingPlan(
        (GenieGenerationCase("first", "0", 11), GenieGenerationCase("second", "1", 22)),
        source.id,
        time_index=3,
        steps=2,
    )
    original_rng = torch.get_rng_state().clone()

    for rank in range(3):
        generate_genie_shard(
            store, world_artifact.id, plan, tmp_path / f"shard{rank}", rank=rank, world_size=3
        )
    assert torch.equal(original_rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="All Genie ranks"):
        merge_genie_shards([tmp_path / "shard0", tmp_path / "shard1"], plan, tmp_path / "missing")
    merged = merge_genie_shards(
        [tmp_path / f"shard{rank}" for rank in (2, 0, 1)], plan, tmp_path / "merged"
    )
    for branch in ("inferred", "random"):
        manifest = MediaManifest.load(tmp_path / "merged" / branch).verify(
            tmp_path / "merged" / branch
        )
        assert manifest.expected_ids == ("first", "second") and all(
            len(s.files) == 4 for s in manifest.samples
        )
        assert (
            _generation_record(tmp_path / "merged" / branch, manifest)["binding"][
                "world_artifact_id"
            ]
            == world_artifact.id
        )
    run = evaluate_genie_controls(store, tmp_path / "merged")
    tokenizer, _ = load_trained_genie(store, codec.id)
    world, _ = load_trained_genie(store, world_artifact.id, world=True)
    for case, outcome in zip(plan.cases, merged["outcomes"]):
        video = engine.objective.data.corpus.load(case.source_id)["video"][None]
        direct = evaluate_genie_controllability(
            tokenizer, world, video, time_index=3, seed=case.seed, steps=2
        )
        assert run.records[case.id].metrics["delta_psnr_t3"] == direct["mean"]
        assert outcome["diagnostics"]["actual_calls"] == {
            "dynamics": 12,
            "tokenizer_encodes": 2,
            "tokenizer_decodes": 2,
            "action_encodes": 1,
        }
        values = torch.load(tmp_path / "merged" / outcome["tensor_file"], weights_only=True)
        assert torch.equal(values["inferred"][:, :1], video[:, :1])
        assert values["inferred_tokens"].dtype == torch.int64
    benchmark = benchmark_genie_sampler(
        store,
        world_artifact.id,
        plan,
        tmp_path / "benchmark",
        settings=GenerationBenchmarkSettings(repetitions=2),
    )
    assert len(benchmark["trials"]) == 4 and benchmark["status"] == "ok"
    assert all(
        row["paired_generation_seconds"] > 0 and row["cuda_peak_allocated_bytes"] is None
        for row in benchmark["trials"]
    )
    assert benchmark["hardware_isolation"] == "development_only"
    for row in benchmark["trials"]:
        expected = next(
            outcome for outcome in merged["outcomes"] if outcome["id"] == row["case_id"]
        )
        assert row["tensors"] == expected["tensors"]
    fvd = evaluate_genie_fvd(tmp_path / "merged", tmp_path / "fvd")
    assert fvd["status"] == "not_evaluated" and fvd["metrics"] == {}
    generated = publish_genie_generation(store, tmp_path / "merged")
    assert (
        generated.metadata["cohort_id"] == plan.cohort_id
        and generated.metadata["public_quality_evaluated"] is False
    )
    assert store.get(generated.id, verify=True).kind == "native_genie_generated_cohort"

    branch = tmp_path / "merged" / "inferred"
    saved = read_json(branch / "media.json")
    image_entry = saved["manifest"]["samples"][0]["files"][0]
    with Image.open(branch / image_entry["path"]) as image:
        pixels = np.array(image)
    Image.fromarray(255 - pixels).save(branch / image_entry["path"])
    image_entry["sha256"] = file_digest(branch / image_entry["path"])
    saved["manifest_id"] = digest_json(saved["manifest"])
    atomic_json(branch / "media.json", saved)
    envelope = read_json(tmp_path / "merged" / "genie_generation.json")
    envelope["record"]["manifests"]["inferred"] = saved["manifest_id"]
    envelope["id"] = digest_json(envelope["record"])
    atomic_json(tmp_path / "merged" / "genie_generation.json", envelope)
    with pytest.raises(ValueError, match="PNG pixels differ"):
        evaluate_genie_controls(store, tmp_path / "merged")


def test_genie_failed_source_frames_remain_in_every_branch_and_denominator(tmp_path):
    store, _, _, _, _, engine, _ = prepare(tmp_path)
    world = publish_genie_world(engine, store, tmp_path / "world")
    corpus = engine.objective.data.corpus
    rows = {key: corpus.load(key) for key in corpus.ids}
    rows["1"]["valid"][2:] = False
    source = publish_genie_videos(
        store,
        rows,
        tmp_path / "eval_source",
        spec=GenieVideoSpec("fixture", "v2", "validation", "local_test_only", 8.0),
    )
    plan = GenieSamplingPlan(
        (GenieGenerationCase("ok", "0", 3), GenieGenerationCase("bad", "1", 4)),
        source.id,
        time_index=3,
        steps=1,
    )
    record = generate_genie_shard(store, world.id, plan, tmp_path / "generation")
    assert [row["status"] for row in record["outcomes"]] == ["ok", "error"]
    run = evaluate_genie_controls(store, tmp_path / "generation")
    assert run.summary()["denominator"] == 2 and run.summary()["statuses"]["error"] == 1
    assert run.scores()[1] == -120.0
    for branch in ("inferred", "random"):
        manifest = MediaManifest.load(tmp_path / "generation" / branch)
        manifest.verify(tmp_path / "generation" / branch, require_complete=False)
        with pytest.raises(ValueError, match="Failed generation"):
            manifest.verify(tmp_path / "generation" / branch)


def test_genie_joint_gate_does_not_promote_without_real_official_fvd_and_isolation(tmp_path):
    store, _, _, source, _, engine, _ = prepare(tmp_path)
    world = publish_genie_world(engine, store, tmp_path / "world")
    cases = (GenieGenerationCase("a", "0", 101), GenieGenerationCase("b", "1", 102))
    baseline = GenieSamplingPlan(cases, source.id, time_index=3, steps=3)
    candidate = GenieSamplingPlan(cases, source.id, time_index=3, steps=1)
    for name, plan in (("baseline", baseline), ("candidate", candidate)):
        generate_genie_shard(store, world.id, plan, tmp_path / name)
        benchmark_genie_sampler(
            store,
            world.id,
            plan,
            tmp_path / (name + "_performance"),
            settings=GenerationBenchmarkSettings(repetitions=2),
        )
    pair = {
        "baseline": str(tmp_path / "baseline"),
        "candidate": str(tmp_path / "candidate"),
        "baseline_benchmark": str(tmp_path / "baseline_performance" / "benchmark.json"),
        "candidate_benchmark": str(tmp_path / "candidate_performance" / "benchmark.json"),
    }
    gate = compare_genie_cohorts(
        store,
        [pair],
        tmp_path / "gate",
        max_delta_psnr_regression=120.0,
        minimum_latency_improvement=0.0,
        repetitions=100,
    )
    assert gate["status"] != "promote"
    assert "official_FVD_not_successfully_evaluated" in gate["unevaluated"]
    assert "hardware_isolation_not_asserted" in gate["unevaluated"]
    assert gate["cohorts"][0]["baseline_nfe"] == [18] and gate["cohorts"][0]["candidate_nfe"] == [6]
    assert gate["comparison"]["fvd_improvement"] is None
    assert (tmp_path / "gate" / "gate.json").is_file()
    with pytest.raises(ValueError, match="cannot reuse"):
        compare_genie_cohorts(store, [pair, pair], tmp_path / "duplicate_gate", repetitions=100)

    saved = read_json(pair["candidate_benchmark"])
    saved["record"]["trials"].pop()
    saved["id"] = digest_json(saved["record"])
    atomic_json(tmp_path / "incomplete.json", saved)
    with pytest.raises(ValueError, match="omitted/reordered"):
        compare_genie_cohorts(
            store,
            [{**pair, "candidate_benchmark": str(tmp_path / "incomplete.json")}],
            tmp_path / "incomplete_gate",
            repetitions=100,
        )
