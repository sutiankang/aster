import asyncio
import pytest
import torch

from aster.evaluation.dspark import evaluate_dspark
from aster.inference.dspark import DSparkDecoder
from aster.inference import ModelRunner, SamplingConfig
from aster.models import CausalLM, Qwen3Config
from aster.models.dspark import DSparkConfig, DSparkDraft


def runtime():
    torch.set_num_threads(1)
    torch.manual_seed(991)
    target = CausalLM(
        Qwen3Config(
            vocab_size=7,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
        )
    ).eval()
    with torch.no_grad():
        target.lm_head.weight.zero_()
    draft = DSparkDraft(
        DSparkConfig(target.config, block_size=2, num_anchors=1, markov_rank=2)
    ).initialize_from_target(target)
    with torch.no_grad():
        draft.markov_head.markov_w2.weight.zero_()
    return DSparkDecoder(
        ModelRunner(target, policy_artifact_id="target7"),
        draft,
        draft_policy_artifact_id="draft7",
        vocabulary_fingerprint="vocabulary7",
    )


def test_dspark_paired_evaluation_counts_real_verification_and_does_not_fake_public_quality():
    decoder = runtime()
    report = asyncio.run(
        evaluate_dspark(
            decoder,
            {"a": [1, 2], "b": [2]},
            SamplingConfig(max_new_tokens=7, temperature=0),
            protocol_id="synthetic_greedy_fixture",
            dataset_revision="seed991",
        )
    )
    summary = report["summary"]
    assert (
        summary["succeeded"] == 2 and summary["failed"] == 0 and summary["exact_greedy_equivalence"]
    )
    assert summary["acceptance_rate"] == 1 and summary["tokens_per_verification"] == 7 / 3
    assert summary["public_quality"] == "not_evaluated" and not summary["deployment_promoted"]
    assert summary["latency_ratio"] > 0
    for row in report["samples"]:
        assert (
            row["target"]["model_seconds"] > 0
            and row["dspark"]["speculation"]["target_input_tokens"] > 0
        )
    assert not decoder.pool.used_blocks and not decoder.target.pool.used_blocks


def test_dspark_evaluation_retains_failed_cases_and_never_promotes_a_success_only_average():
    decoder = runtime()
    report = asyncio.run(
        evaluate_dspark(
            decoder,
            {"ok": [1], "bad": [99]},
            SamplingConfig(max_new_tokens=2, temperature=0),
            protocol_id="invalid_fixture",
            dataset_revision="v1",
            scorer=lambda *_: 1.0,
            warmup=False,
        )
    )
    assert len(report["samples"]) == 2 and report["summary"]["failed"] == 1
    assert (
        report["summary"]["public_quality"] == "incomplete"
        and report["summary"]["latency_ratio"] is None
    )
    assert (
        not report["summary"]["exact_greedy_equivalence"]
        and not report["summary"]["deployment_promoted"]
    )


def test_dspark_single_request_lock_rejects_reentrant_generation_and_releases_after_failure():
    decoder = runtime()

    def callback(_):
        decoder.generate([1])

    with pytest.raises(RuntimeError, match="active request"):
        decoder.generate([1], on_token=callback)
    assert decoder.generate([1], SamplingConfig(max_new_tokens=1)).token_ids
