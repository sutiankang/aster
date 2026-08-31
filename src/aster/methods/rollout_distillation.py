"""Native rollouts to sequence and on-policy distillation datasets."""

import asyncio
from dataclasses import dataclass, asdict, replace
import hashlib
import torch
from ..core import digest_json, atomic_json
from ..data import causal_collate
from ..models import build_model
from ..inference import ModelRunner, InferenceEngine, SamplingConfig
from .distillation import DistillationObjective


def tensor_state_identity(state, configuration):
    digest = hashlib.sha256(digest_json(configuration).encode())
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(
            digest_json(
                {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            ).encode()
        )
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DistillationRollout:
    sample_id: str
    policy_id: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    text: str
    stop_reason: str
    raw_model_logprobs: tuple[float, ...]
    behavior_logprobs: tuple[float, ...]
    sampling: dict
    error: str | None = None


async def collect_native_rollouts(
    inference_engine, prompts, sampling, *, concurrency=4, timeout_s=60.0
):
    """Return one result per requested prompt, including timeouts, rejections, and empty outputs."""
    if not prompts or concurrency < 1:
        raise ValueError("Need nonempty prompts and bounded concurrency")
    semaphore = asyncio.Semaphore(concurrency)
    policy_id = inference_engine.runner.policy_artifact_id

    async def one(index, prompt):
        settings = replace(sampling, seed=sampling.seed + index)
        sample_id = str(index)
        async with semaphore:
            try:
                handle = await inference_engine.submit(prompt, settings, timeout_s=timeout_s)
                result = await handle.collect()
                if result.policy_artifact_id != policy_id:
                    raise ValueError("Rollout policy identity changed during collection")
                return DistillationRollout(
                    sample_id,
                    policy_id,
                    tuple(prompt),
                    result.token_ids,
                    result.text,
                    result.stop_reason,
                    result.raw_model_logprobs,
                    result.behavior_logprobs,
                    asdict(settings),
                    result.error_code,
                )
            except (ValueError, RuntimeError) as error:
                return DistillationRollout(
                    sample_id,
                    policy_id,
                    tuple(prompt),
                    (),
                    "",
                    "error",
                    (),
                    (),
                    asdict(settings),
                    str(error),
                )

    return await asyncio.gather(*(one(index, prompt) for index, prompt in enumerate(prompts)))


def sequence_distillation_examples(
    rollouts, prompt_texts, student_tokenizer, *, accept_length=False, verifier=None
):
    """Re-encode teacher text into the student's vocabulary and retain acceptance
    or rejection reasons for every sample."""
    if len(rollouts) != len(prompt_texts):
        raise ValueError("Prompt text and rollout records must align")
    examples, receipts = [], []
    for rollout, prompt in zip(rollouts, prompt_texts):
        reason = None
        if rollout.error:
            reason = rollout.error
        elif rollout.stop_reason not in ({"eos", "length"} if accept_length else {"eos"}):
            reason = "completion_not_finished"
        elif not rollout.text.strip():
            reason = "empty_completion"
        elif verifier is not None and not verifier(prompt, rollout.text):
            reason = "verification_rejected"
        if reason is None:
            prompt_ids = student_tokenizer.encode(prompt, add_special_tokens=False)
            whole = student_tokenizer.encode(prompt + rollout.text, add_special_tokens=False)
            if whole[: len(prompt_ids)] != prompt_ids:
                reason = "tokenizer_boundary_merge"
            else:
                if rollout.stop_reason == "eos":
                    whole = whole + [student_tokenizer.eos_token_id]
                labels = [-100] * len(prompt_ids) + whole[len(prompt_ids) :]
                if len(whole) < 2 or len(whole) == len(prompt_ids):
                    reason = "no_trainable_response"
                else:
                    examples.append({"input_ids": whole, "labels": labels})
        receipts.append(
            {
                "sample_id": rollout.sample_id,
                "source_policy": rollout.policy_id,
                "accepted": reason is None,
                "reason": reason,
                "stop_reason": rollout.stop_reason,
            }
        )
    return examples, receipts


def save_distillation_rollouts(path, rollouts, *, tokenizer_fingerprint, dataset_fingerprint):
    if not tokenizer_fingerprint or not dataset_fingerprint:
        raise ValueError("Rollout evidence requires token and input dataset identities")
    payload = {
        "schema_version": 1,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "rollouts": [asdict(item) for item in rollouts],
    }
    atomic_json(path, {"payload": payload, "sha256": digest_json(payload)})


class OnPolicyDistillationMethod:
    """Let the current student generate contexts, score them with a frozen teacher,
    and update through the shared trainer."""

    def __init__(
        self,
        engine,
        teacher,
        tokenizer,
        *,
        teacher_tokenizer_fingerprint,
        kind="reverse_kl",
        temperature=1.0,
        max_prompt_tokens=4096,
    ):
        fingerprint = digest_json(tokenizer.to_dict())
        if teacher_tokenizer_fingerprint != fingerprint:
            raise ValueError("On-policy token KL requires identical tokenization/templates")
        if any(
            getattr(engine.parallel.config, key, 1) > 1
            for key in (
                "tensor_parallel",
                "pipeline_parallel",
                "context_parallel",
                "gtp_remat",
                "expert_parallel",
                "expert_tensor_parallel",
            )
        ):
            raise ValueError(
                "Online dense snapshot path requires pure DP; model-parallel rollout routing is explicit"
            )
        self.engine, self.tokenizer, self.fingerprint = engine, tokenizer, fingerprint
        self.kind, self.temperature, self.max_prompt_tokens = kind, temperature, max_prompt_tokens
        self.teacher = engine.add_role("distillation_teacher", teacher, trainable=False)
        self.objective = DistillationObjective(
            self.teacher,
            kind=kind,
            temperature=temperature,
            kd_weight=1.0,
            tokenizer_fingerprints=(fingerprint, fingerprint),
        )
        self.updates, self._busy = 0, False
        self.last_rollouts = ()
        engine.register_state("on_policy_distillation", self)

    async def update(self, prompt_token_ids, *, sampling=None):
        if self._busy:
            raise RuntimeError("On-policy collection already in progress")
        sampling = sampling or SamplingConfig(eos_token_ids=(self.tokenizer.eos_token_id,))
        if len(prompt_token_ids) < self.engine.accumulation_steps:
            raise ValueError("Need a nonempty local microbatch for every accumulation slot")
        self._busy = True
        inference = None
        try:
            initial_step = self.engine.steps
            weights = self.engine.export_state_dict(only_rank_zero=False)
            identity = tensor_state_identity(weights, self.engine.model.config.to_dict())
            snapshot = build_model(self.engine.model.config)
            snapshot.load_state_dict(weights, strict=True)
            snapshot.to(self.engine.device)
            runner = ModelRunner(
                snapshot,
                policy_artifact_id=identity,
                tokenizer=self.tokenizer,
                block_size=16,
                max_blocks=max(
                    256,
                    len(prompt_token_ids) * (self.max_prompt_tokens + sampling.max_new_tokens) // 16
                    + len(prompt_token_ids),
                ),
            )
            inference = InferenceEngine(
                runner,
                max_prompt_tokens=self.max_prompt_tokens,
                max_generation_tokens=sampling.max_new_tokens,
            )
            self.last_rollouts = tuple(
                await collect_native_rollouts(inference, prompt_token_ids, sampling)
            )
            if self.engine.steps != initial_step:
                raise RuntimeError("Student updated while its rollout snapshot was active")
            examples = []
            for item in self.last_rollouts:
                if (
                    item.error
                    or item.stop_reason not in {"eos", "length"}
                    or not item.completion_ids
                ):
                    raise RuntimeError(
                        f"Rollout {item.sample_id} failed; no partial training update"
                    )
                ids = list(item.prompt_ids + item.completion_ids)
                examples.append(
                    {
                        "input_ids": ids,
                        "labels": [-100] * len(item.prompt_ids) + list(item.completion_ids),
                    }
                )
            batches = [
                causal_collate(
                    examples[index :: self.engine.accumulation_steps],
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                for index in range(self.engine.accumulation_steps)
            ]
            batches = [
                {key: value.to(self.engine.device) for key, value in batch.items()}
                for batch in batches
            ]
            result = self.engine.phase(
                "on_policy_distillation", objective=self.objective, microbatches=batches
            )
            if result.updated:
                self.updates += 1
            return result
        finally:
            if inference is not None:
                await inference.close()
            self._busy = False

    def state_dict(self):
        if self._busy:
            raise RuntimeError("Cannot checkpoint an incomplete rollout/optimization cycle")
        return {
            "tokenizer_fingerprint": self.fingerprint,
            "kind": self.kind,
            "temperature": self.temperature,
            "max_prompt_tokens": self.max_prompt_tokens,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(state[key] != value for key, value in expected.items() if key != "updates"):
            raise ValueError("On-policy distillation configuration differs")
        self.updates = state["updates"]
