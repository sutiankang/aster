"""Native on-policy text rollouts, group rewards, RLOO, and GRPO updates."""

from dataclasses import asdict, replace
import inspect
import math

import torch
from torch import nn

from ..core import LossTerm, digest_json
from ..data import causal_collate
from ..inference import ModelRunner, InferenceEngine, SamplingConfig
from ..models import build_model
from .reinforcement import GRPOObjective, group_relative_advantages
from .rollout_distillation import collect_native_rollouts, tensor_state_identity
from .supervised import sequence_logprobs


def leave_one_out_advantages(rewards, group_ids):
    """Compute A_i = r_i - sum(r_j for j != i) / (G - 1).
    Form complete prompt groups before splitting microbatches; missing rewards are
    not substituted with zero."""
    if rewards.ndim != 1 or group_ids.shape != rewards.shape or not torch.isfinite(rewards).all():
        raise ValueError("Finite rewards and aligned group IDs are required")
    advantages = torch.empty_like(rewards)
    for group in group_ids.unique():
        selected = group_ids == group
        values = rewards[selected]
        if len(values) < 2:
            raise ValueError("RLOO requires at least two scorable responses in every group")
        advantages[selected] = values - (values.sum() - values) / (len(values) - 1)
    return advantages


class RLOOObjective(nn.Module):
    def __init__(self, *, clip_low=0.2, clip_high=0.2):
        super().__init__()
        if not 0 <= clip_low < 1 or clip_high < 0:
            raise ValueError("Invalid sequence probability-ratio clipping")
        self.clip_low, self.clip_high = clip_low, clip_high

    def config_dict(self):
        return {"type": "rloo", "clip_low": self.clip_low, "clip_high": self.clip_high}

    def forward(self, policy, batch):
        logp, valid = sequence_logprobs(policy, batch)
        old = batch["old_behavior_log_probs"].detach()
        advantages = batch["advantages"].detach()
        if (
            old.shape != logp.shape
            or advantages.shape != (len(logp),)
            or (valid.sum(-1) == 0).any()
        ):
            raise ValueError("RLOO requires aligned complete action-token trajectories")

        ratio = ((logp - old) * valid).sum(-1).exp()
        if not torch.isfinite(ratio).all():
            raise ValueError("Sequence importance ratio overflow; reduce policy staleness")
        clipped = ratio.clamp(1 - self.clip_low, 1 + self.clip_high)
        values = -torch.minimum(ratio * advantages, clipped * advantages)
        return LossTerm(values.sum(), values.new_tensor(len(values)), "completion", "rloo")


def collate_policy_rollouts(rollouts, *, pad_token_id, device):
    """Align sampled action probabilities with next-token positions; prompt and
    padding tokens are excluded from the loss."""
    examples = []
    for item in rollouts:
        if (
            item.error
            or item.stop_reason not in {"eos", "length"}
            or not item.prompt_ids
            or not item.completion_ids
        ):
            raise ValueError("A failed/empty trajectory cannot silently disappear from an RL batch")
        if len(item.behavior_logprobs) != len(item.completion_ids) or len(
            item.raw_model_logprobs
        ) != len(item.completion_ids):
            raise ValueError("Every generated action token needs both probability records")
        if not all(
            math.isfinite(value) for value in (*item.raw_model_logprobs, *item.behavior_logprobs)
        ):
            raise ValueError("Rollout probabilities must be finite")
        examples.append(
            {
                "input_ids": list(item.prompt_ids + item.completion_ids),
                "labels": [-100] * len(item.prompt_ids) + list(item.completion_ids),
            }
        )
    batch = {
        key: value.to(device)
        for key, value in causal_collate(examples, pad_token_id=pad_token_id).items()
    }
    old = torch.zeros((len(examples), batch["input_ids"].shape[1] - 1), device=device)
    raw = torch.zeros_like(old)
    for row, item in enumerate(rollouts):
        start = len(item.prompt_ids) - 1
        end = start + len(item.completion_ids)
        old[row, start:end] = torch.tensor(item.behavior_logprobs, device=device)
        raw[row, start:end] = torch.tensor(item.raw_model_logprobs, device=device)
    batch["old_behavior_log_probs"], batch["old_raw_log_probs"] = old, raw
    return batch


class OnPolicyRLMethod:
    """Use one policy snapshot per complete rollout group and bind reward semantics
    and progress to the shared checkpoint."""

    def __init__(
        self,
        engine,
        reference,
        tokenizer,
        *,
        reward,
        reward_id,
        reference_tokenizer_fingerprint,
        algorithm="rloo",
        group_size=4,
        kl_weight=0.05,
        max_prompt_tokens=4096,
        clip_low=0.2,
        clip_high=0.2,
        grpo_reduction="sequence",
        max_completion_length=None,
    ):
        fingerprint = digest_json(tokenizer.to_dict())
        if reference_tokenizer_fingerprint != fingerprint or not reward_id or not callable(reward):
            raise ValueError(
                "Reward identity/callback and matching reference token semantics are required"
            )
        if algorithm not in {"rloo", "grpo"} or group_size < 2 or kl_weight < 0:
            raise ValueError("Invalid online policy-optimization configuration")
        if engine.parallel.world.size != 1:
            raise ValueError(
                "This rollout controller is single-rank; use explicit distributed rollout orchestration"
            )
        if any(isinstance(module, nn.Dropout) and module.p for module in engine.model.modules()):
            raise ValueError(
                "Online policy likelihoods require training dropout disabled in model configuration"
            )
        self.engine, self.tokenizer, self.reward = engine, tokenizer, reward
        self.fingerprint, self.reward_id = fingerprint, reward_id
        self.algorithm, self.group_size, self.kl_weight = algorithm, group_size, kl_weight
        self.max_prompt_tokens = max_prompt_tokens
        self.reference = engine.add_role("rl_reference", reference, trainable=False)
        self.objective = (
            RLOOObjective(clip_low=clip_low, clip_high=clip_high)
            if algorithm == "rloo"
            else GRPOObjective(
                clip_low=clip_low,
                clip_high=clip_high,
                kl_weight=kl_weight,
                reduction=grpo_reduction,
                max_completion_length=max_completion_length,
            )
        )
        self.settings = {
            "algorithm": algorithm,
            "group_size": group_size,
            "kl_weight": kl_weight,
            "max_prompt_tokens": max_prompt_tokens,
            "clip_low": clip_low,
            "clip_high": clip_high,
            "grpo_reduction": grpo_reduction,
            "max_completion_length": max_completion_length,
        }
        self.updates, self._busy, self.last_records = 0, False, ()
        engine.register_state("on_policy_rl", self)

    async def update(self, prompts, *, sampling=None):
        if self._busy:
            raise RuntimeError("A rollout/update cycle is already active")
        sampling = sampling or SamplingConfig(eos_token_ids=(self.tokenizer.eos_token_id,))
        if (
            sampling.temperature != 1
            or sampling.top_k
            or sampling.top_p != 1
            or sampling.repetition_penalty != 1
            or sampling.logit_bias
            or sampling.min_new_tokens
        ):
            raise ValueError("On-policy RL requires untruncated, unbiased temperature-one sampling")
        expanded = [list(prompt) for prompt in prompts for _ in range(self.group_size)]
        if len(expanded) < self.engine.accumulation_steps:
            raise ValueError("Every accumulation slot needs at least one response")
        self._busy, inference = True, None
        try:
            initial_step = self.engine.steps
            weights = self.engine.export_state_dict(only_rank_zero=False)
            identity = tensor_state_identity(weights, self.engine.model.config.to_dict())
            snapshot = build_model(self.engine.model.config)
            snapshot.load_state_dict(weights, strict=True)
            snapshot.to(self.engine.device)
            blocks = max(
                256, len(expanded) * ((self.max_prompt_tokens + sampling.max_new_tokens + 15) // 16)
            )
            runner = ModelRunner(
                snapshot,
                policy_artifact_id=identity,
                tokenizer=self.tokenizer,
                block_size=16,
                max_blocks=blocks,
            )
            inference = InferenceEngine(
                runner,
                max_prompt_tokens=self.max_prompt_tokens,
                max_generation_tokens=sampling.max_new_tokens,
            )
            settings = replace(sampling, seed=sampling.seed + self.updates * len(expanded))
            rollouts = await collect_native_rollouts(inference, expanded, settings)
            self.last_records = tuple(
                {"rollout": asdict(item), "reward": None, "reward_id": self.reward_id}
                for item in rollouts
            )
            batch = collate_policy_rollouts(
                rollouts, pad_token_id=self.tokenizer.pad_token_id, device=self.engine.device
            )
            if not torch.allclose(
                batch["old_raw_log_probs"], batch["old_behavior_log_probs"], atol=2e-5, rtol=2e-5
            ):
                raise ValueError(
                    "Recorded behavior differs from the required on-policy distribution"
                )
            rewards = []
            for index, item in enumerate(rollouts):
                value = self.reward(item, index // self.group_size)
                value = await value if inspect.isawaitable(value) else value
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        "Reward must be an explicit finite scalar, not an unscorable result"
                    )
                rewards.append(value)
                self.last_records[index]["reward"] = value
            self.reference.eval()
            with torch.no_grad():
                reference_logp, valid = sequence_logprobs(self.reference, batch)
            batch["reference_log_probs"] = reference_logp
            rewards = torch.tensor(rewards, device=self.engine.device)
            groups = torch.arange(len(prompts), device=self.engine.device).repeat_interleave(
                self.group_size
            )
            if self.algorithm == "rloo":
                penalty = ((batch["old_raw_log_probs"] - reference_logp) * valid).sum(-1)
                batch["advantages"] = leave_one_out_advantages(
                    rewards - self.kl_weight * penalty, groups
                )
            else:
                batch["advantages"] = group_relative_advantages(rewards, groups)
            if initial_step != self.engine.steps:
                raise RuntimeError("Policy changed while the rollout snapshot was being scored")
            batches = [
                {
                    key: value[index :: self.engine.accumulation_steps]
                    for key, value in batch.items()
                }
                for index in range(self.engine.accumulation_steps)
            ]
            result = self.engine.phase(
                "on_policy_" + self.algorithm, objective=self.objective, microbatches=batches
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
            raise RuntimeError("Cannot checkpoint an incomplete rollout/reward/update cycle")
        return {
            "settings": self.settings,
            "tokenizer_fingerprint": self.fingerprint,
            "reward_id": self.reward_id,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(state.get(key) != value for key, value in expected.items() if key != "updates"):
            raise ValueError("Online RL configuration, reward version or tokenizer changed")
        self.updates = state["updates"]
