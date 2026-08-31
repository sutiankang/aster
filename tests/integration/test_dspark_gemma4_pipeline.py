import asyncio
from copy import deepcopy
import math
import pytest
import torch
from aster.core import ArtifactStore
from aster.data.dspark import DSparkTeacherFeatures, DSparkFeatureCache, publish_dspark_features
from aster.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
from aster.models.dspark_gemma4 import Gemma4DSparkConfig, Gemma4DSparkDraft
from aster.methods.dspark import DSparkMethod
from aster.methods.dspark_artifacts import publish_dspark_draft, load_dspark_draft
from aster.inference import SamplingConfig
from aster.inference.gemma4 import Gemma4SnapshotRunner
from aster.inference.dspark import DSparkDecoder
from aster.evaluation.dspark import evaluate_dspark
from aster.training import Trainer


def models(*, uniform=False):
    torch.set_num_threads(1)
    torch.manual_seed(821)
    target = Gemma4ForCausalLM(
        Gemma4TextConfig(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=24,
            head_dim=4,
            global_head_dim=8,
            hidden_size_per_layer_input=0,
            max_position_embeddings=64,
            sliding_window=3,
            final_logit_softcapping=0.15,
            tie_word_embeddings=not uniform,
        )
    ).eval()
    if uniform:
        with torch.no_grad():
            target.lm_head.weight.zero_()
    config = Gemma4DSparkConfig(
        target.config,
        num_draft_layers=2,
        target_layer_ids=(-1, 1),
        num_anchors=2,
        block_size=3,
        markov_rank=4,
        markov_head_type="vanilla",
    )
    draft = Gemma4DSparkDraft(config).initialize_from_target(target)
    if uniform:
        with torch.no_grad():
            draft.markov_head.markov_w1.weight.fill_(1)
            draft.markov_head.markov_w2.weight.zero_()
            draft.markov_head.markov_w2.weight[1].fill_(40)
    return target, draft


def setup(tmp_path):
    target, draft = models()
    teacher = DSparkTeacherFeatures(target, draft.config, vocabulary_fingerprint="gemma19_v1")
    records = {
        str(i): dict(
            input_ids=torch.randint(1, 19, (1, 8)), loss_mask=torch.ones(1, 8, dtype=torch.long)
        )
        for i in range(3)
    }
    records["1"]["loss_mask"][0, 4] = 0
    store = ArtifactStore(tmp_path / "store")
    artifact = publish_dspark_features(
        store,
        teacher,
        records,
        tmp_path / "features",
        dataset_id="synthetic_gemma821",
        revision="fixed_v1",
        license_id="synthetic_test",
    )
    cache = DSparkFeatureCache(store, artifact.id)
    assert cache.contract["extraction"] == "native_gemma4_eval_no_autocast_unpadded"
    assert cache.contract["draft_config"]["architecture"] == "dspark_gemma4"
    return target, draft, store, cache


def no_leases(decoder):
    assert decoder.pool.used_blocks == decoder.target.pool.used_blocks == 0
    assert decoder.target.pool.used_bytes == decoder.target.modality_bytes == 0
    assert decoder.target._bindings == {}


def greedy_reference(target, prompt, length):
    tokens, logprobs = [], []
    with torch.no_grad():
        for _ in range(length):
            logits = target(torch.tensor([list(prompt) + tokens])).logits[0, -1]
            token = int(logits.argmax())
            tokens.append(token)
            logprobs.append(float(logits.float().log_softmax(-1)[token]))
    return tuple(tokens), torch.tensor(logprobs)


@pytest.mark.parametrize("stage", [0, 3])
def test_gemma4_real_cache_method_fresh_resume_artifact_and_paired_evaluation(stage, tmp_path):
    target, draft, store, cache = setup(tmp_path)
    batches = [cache.batch(["0"]), cache.batch(["1", "2"])]
    settings = dict(zero_stage=stage, accumulation_steps=2, lr=0.002, ema_decay=0.9)
    engine = Trainer(draft, **settings)
    method_settings = dict(
        vocabulary_fingerprint="gemma19_v1",
        feature_cache_ids=[cache.artifact_id],
        feature_cache_store=store,
    )
    method = DSparkMethod(engine, **method_settings)
    assert method.objective.config_dict()["normalization_profile"] == "official_microbatch_mean"
    for _ in range(2):
        assert method.update(batches).updated
    checkpoint = engine.save_checkpoint(tmp_path / "training")
    expected = method.update(batches)
    expected_state = deepcopy(engine.export_state_dict(only_rank_zero=False))
    expected_runtime = deepcopy(engine.export_runtime_state())
    fresh = Gemma4DSparkDraft(draft.config).initialize_from_target(target)
    resumed = Trainer(fresh, **settings)
    resumed_method = DSparkMethod(resumed, **method_settings)
    resumed.load_checkpoint(checkpoint, trusted=True)
    actual = resumed_method.update(batches)
    assert (
        actual.updated
        and actual.loss == expected.loss
        and resumed_method.updates == method.updates == 3
    )
    torch.testing.assert_close(
        resumed.export_state_dict(only_rank_zero=False), expected_state, atol=0, rtol=0
    )
    torch.testing.assert_close(resumed.export_runtime_state(), expected_runtime, atol=0, rtol=0)
    artifact = publish_dspark_draft(resumed_method, store, tmp_path / "deployment")
    restored, contract = load_dspark_draft(store, artifact.id)
    assert type(restored) is Gemma4DSparkDraft and artifact.parents == (cache.artifact_id,)
    assert contract["receipt"]["role_updates"] == 3 and contract["quality_claim"] == "not_evaluated"
    torch.testing.assert_close(restored.state_dict(), expected_state, atol=0, rtol=0)
    runner = Gemma4SnapshotRunner(target, policy_artifact_id="synthetic_gemma_target821")
    decoder = DSparkDecoder(
        runner, restored, draft_policy_artifact_id=artifact.id, vocabulary_fingerprint="gemma19_v1"
    )
    prompt = [1, 4, 2, 3]
    sampling = SamplingConfig(max_new_tokens=8, temperature=0)
    events = []
    result = decoder.generate(prompt, sampling, on_token=events.append)
    expected_tokens, expected_logs = greedy_reference(target, prompt, 8)
    assert result.token_ids == expected_tokens and len(events) == 8
    torch.testing.assert_close(
        torch.tensor(result.raw_model_logprobs), expected_logs, atol=5e-7, rtol=2e-6
    )
    assert all(value == 0 for value in result.behavior_logprobs)
    assert result.dspark_stats["cache_profile"] == "gemma4_snapshot_full_prefix_replay"
    assert result.draft_policy_artifact_id == artifact.id
    no_leases(decoder)
    if stage == 0:
        report = asyncio.run(
            evaluate_dspark(
                decoder,
                {"short": [1, 2], "window": [3, 4, 5, 6]},
                sampling,
                protocol_id="synthetic_gemma4_greedy_v1",
                dataset_revision="fixed_v1",
                warmup=True,
            )
        )
        assert report["summary"]["succeeded"] == 2 and report["summary"]["failed"] == 0
        assert report["summary"]["exact_greedy_equivalence"] is True
        assert (
            report["summary"]["public_quality"] == "not_evaluated"
            and not report["summary"]["deployment_promoted"]
        )
        assert not report["warmup_errors"] and all(row["same_tokens"] for row in report["samples"])
        no_leases(decoder)


def test_gemma4_real_rejection_replays_lost_windows_and_releases_cancel_error_leases():
    target, draft = models(uniform=True)
    runner = Gemma4SnapshotRunner(target, policy_artifact_id="uniform_gemma_target_fixture")
    decoder = DSparkDecoder(
        runner,
        draft,
        draft_policy_artifact_id="untrained_forced_rejection_fixture",
        vocabulary_fingerprint="gemma19_v1",
    )
    sampling = SamplingConfig(max_new_tokens=8, temperature=0)
    result = decoder.generate([2, 3, 4, 5], sampling)
    assert result.token_ids == (0,) * 8 and result.dspark_stats["rejected_blocks"] == 8
    assert result.dspark_stats["target_replay_rollbacks"] == 8
    torch.testing.assert_close(
        torch.tensor(result.raw_model_logprobs),
        torch.full((8,), -math.log(19)),
        atol=3e-7,
        rtol=2e-6,
    )
    no_leases(decoder)
    cancelled = decoder.generate([2, 3], sampling, cancelled=lambda: True)
    assert cancelled.stop_reason == "cancelled" and not cancelled.token_ids
    no_leases(decoder)

    def disconnect(_):
        raise RuntimeError("client disconnected")

    with pytest.raises(RuntimeError, match="disconnected"):
        decoder.generate([2, 3], sampling, on_token=disconnect)
    no_leases(decoder)


def test_gemma4_bad_immutable_feature_row_rejected_before_training_projection(tmp_path):
    _, draft, store, cache = setup(tmp_path)
    engine = Trainer(draft, zero_stage=3)
    method = DSparkMethod(
        engine,
        vocabulary_fingerprint="gemma19_v1",
        feature_cache_ids=[cache.artifact_id],
        feature_cache_store=store,
    )
    corrupted = deepcopy(cache.batch(["0"]))
    corrupted["target_hidden_states"][0, 0, 0] += 0.01
    calls = []
    handle = draft.fc.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        with pytest.raises(ValueError, match="immutable feature cache"):
            method.update([corrupted])
    finally:
        handle.remove()
    assert not calls and not engine._failed and method.updates == 0
