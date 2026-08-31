"""Paired DSpark quality, latency, and acceptance measurements over complete cohorts."""

from dataclasses import asdict, replace
import math

from ..core import digest_json
from ..inference import ModelRunner, InferenceEngine
from ..inference.dspark import DSparkDecoder
from ..inference.gemma4 import Gemma4SnapshotRunner


async def evaluate_dspark(
    decoder, cases, sampling, *, protocol_id, dataset_revision, scorer=None, warmup=True
):
    if (
        not isinstance(decoder, DSparkDecoder)
        or not cases
        or any(not isinstance(k, str) or not k for k in cases)
    ):
        raise ValueError("Evaluate a native DSpark decoder on a nonempty named cohort")
    if any(
        not isinstance(v, str) or not v for v in (protocol_id, dataset_revision)
    ) or dataset_revision.lower() in {"main", "master", "latest"}:
        raise ValueError("Declare a fixed evaluation protocol and dataset revision")
    if isinstance(decoder.target, Gemma4SnapshotRunner):
        baseline_runner = Gemma4SnapshotRunner(
            decoder.target.model,
            policy_artifact_id=decoder.target.policy_artifact_id,
            tokenizer=decoder.target.tokenizer,
            processor_id=decoder.target.processor_id,
            max_cache_bytes=decoder.target.pool.max_bytes,
        )
    else:
        baseline_runner = ModelRunner(
            decoder.target.model,
            policy_artifact_id=decoder.target.policy_artifact_id,
            codec=decoder.target.codec,
            tokenizer=decoder.target.tokenizer,
            block_size=decoder.target.pool.block_size,
            max_blocks=decoder.target.pool.max_blocks,
        )
    baseline = InferenceEngine(baseline_runner, max_active=1)
    rows, warmup_errors = [], []

    async def baseline_generate(prompt, settings):
        handle = await baseline.submit(prompt, settings)
        result = await handle.collect()
        if result.error_code is not None:
            raise RuntimeError(result.error_code)
        return result

    try:
        if warmup:
            first = next(iter(cases.values()))
            warm = replace(
                sampling, max_new_tokens=max(1, sampling.min_new_tokens), seed=sampling.seed
            )
            try:
                await baseline_generate(first, warm)
                decoder.generate(first, warm)
            except Exception as error:
                warmup_errors.append(f"{type(error).__name__}: {error}")
        for index, (sample_id, prompt) in enumerate(cases.items()):
            settings = replace(sampling, seed=sampling.seed + index)
            row = {
                "sample_id": sample_id,
                "prompt_identity": digest_json(list(prompt)),
                "status": "error",
            }
            outputs = {}
            for name in ("target", "dspark") if index % 2 == 0 else ("dspark", "target"):
                try:
                    before = baseline_runner.model_execution_seconds
                    result = (
                        await baseline_generate(prompt, settings)
                        if name == "target"
                        else decoder.generate(prompt, settings)
                    )
                    outputs[name] = result
                    row[name] = dict(
                        tokens=len(result.token_ids),
                        stop_reason=result.stop_reason,
                        **result.metrics(),
                    )
                    if name == "target":
                        row[name]["model_seconds"] = (
                            baseline_runner.model_execution_seconds - before
                        )
                    else:
                        row[name]["speculation"] = result.dspark_stats
                    if scorer is not None:
                        value = float(scorer(sample_id, result.text))
                        if not math.isfinite(value):
                            raise ValueError("Nonfinite official score")
                        row[name]["quality_score"] = value
                except Exception as error:
                    row[name] = {"error": f"{type(error).__name__}: {error}"}
                    outputs.pop(name, None)
            if len(outputs) == 2:
                row["status"] = "ok"
                row["same_tokens"] = outputs["target"].token_ids == outputs["dspark"].token_ids
                row["accepted_draft_tokens"] = len(outputs["dspark"].accepted_draft_tokens)
                row["proposed_draft_tokens"] = outputs["dspark"].draft_token_count
            rows.append(row)
    finally:
        await baseline.close()
    successful = [r for r in rows if r["status"] == "ok"]
    all_ok = len(successful) == len(rows)
    proposed = sum(r["proposed_draft_tokens"] for r in successful)
    accepted = sum(r["accepted_draft_tokens"] for r in successful)
    summary = dict(
        samples=len(rows),
        succeeded=len(successful),
        failed=len(rows) - len(successful),
        acceptance_rate=None if proposed == 0 else accepted / proposed,
        exact_greedy_equivalence=(all_ok and all(r["same_tokens"] for r in successful))
        if sampling.temperature == 0
        else "not_a_samplewise_stochastic_claim",
        public_quality="not_evaluated"
        if scorer is None
        else ("external_scores_protocol_not_verified" if all_ok else "incomplete"),
        latency_ratio=None,
        tokens_per_verification=None,
        deployment_promoted=False,
    )
    if all_ok:
        target_time = sum(r["target"]["end_to_end_seconds"] for r in rows)
        draft_time = sum(r["dspark"]["end_to_end_seconds"] for r in rows)
        summary["latency_ratio"] = target_time / draft_time if draft_time > 0 else None
        verifications = sum(r["dspark"]["speculation"]["target_verification_calls"] for r in rows)
        summary["tokens_per_verification"] = (
            sum(r["dspark"]["tokens"] for r in rows) / verifications if verifications else None
        )
    return dict(
        schema_version=1,
        protocol_id=protocol_id,
        dataset_revision=dataset_revision,
        target_policy_id=decoder.target.policy_artifact_id,
        draft_policy_id=decoder.draft_policy_artifact_id,
        vocabulary_fingerprint=decoder.vocabulary_fingerprint,
        sampling=asdict(sampling),
        cohort_identity=digest_json({k: list(v) for k, v in cases.items()}),
        summary=summary,
        samples=rows,
        warmup_errors=warmup_errors,
        measurement_scope="sequential_local_target_scheduler_vs_single_request_dspark_server_emit_wall_clock_not_kernel_only",
        quality_note="A quality scorer does not by itself certify its benchmark protocol or authorize deployment",
    )
