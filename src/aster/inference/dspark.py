"""Single-request DSpark decoding with confidence truncation and exact target verification."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
import time
import threading
import uuid

import torch

from ..models import CausalLM, Qwen3Config
from ..models.dspark import DSparkDraft, target_state_identity
from ..models.dspark_gemma4 import Gemma4DSparkDraft
from ..models.gemma4 import Gemma4ForCausalLM
from .gemma4 import Gemma4SnapshotRunner
from .engine import GenerationResult, TokenEvent
from .sampling import SamplingConfig, distributions, speculative_accept
from .state import PagedStatePool, KVStateCodec


def confident_prefix_length(confidence_logits, threshold):

    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("DSpark confidence threshold must lie in [0,1]")
    if confidence_logits.ndim != 1 or not torch.isfinite(confidence_logits).all():
        raise ValueError("DSpark confidence logits must be a finite vector")
    failures = (confidence_logits.sigmoid() < threshold).nonzero()
    return int(failures[0]) if len(failures) else len(confidence_logits)


@dataclass(frozen=True)
class DSparkGenerationResult(GenerationResult):
    dspark_stats: dict = field(default_factory=dict)


class DSparkDecoder:
    def __init__(
        self,
        target_runner,
        draft,
        *,
        draft_policy_artifact_id,
        vocabulary_fingerprint,
        confidence_threshold=0.0,
        block_size=16,
        max_blocks=256,
    ):
        qwen = (
            type(target_runner.model) is CausalLM
            and type(target_runner.model.config) is Qwen3Config
            and isinstance(draft, DSparkDraft)
        )
        gemma = (
            type(target_runner) is Gemma4SnapshotRunner
            and type(target_runner.model) is Gemma4ForCausalLM
            and isinstance(draft, Gemma4DSparkDraft)
        )
        if not qwen and not gemma:
            raise ValueError(
                "DSpark requires a matching native Qwen3 or Gemma4 target/draft runner pair"
            )
        if not draft.config.freeze_embedding_head or not bool(draft.teacher_weights_loaded):
            raise ValueError(
                "DSpark decoder requires target-initialized native draft with frozen vocabulary/head"
            )
        if getattr(draft, "_aster_training_owned", False):
            raise ValueError("Export a dense draft snapshot before serving")
        if (
            target_runner.model.config.to_dict() != draft.config.target.to_dict()
            or target_state_identity(target_runner.model) != draft.teacher_identity
        ):
            raise ValueError(
                "DSpark draft is bound to another target weight/configuration identity"
            )
        if any(
            not isinstance(x, str) or not x
            for x in (draft_policy_artifact_id, vocabulary_fingerprint)
        ):
            raise ValueError("Explicit immutable draft and vocabulary identities are required")
        confident_prefix_length(torch.zeros(1), confidence_threshold)
        if confidence_threshold > 0 and draft.confidence_head is None:
            raise ValueError("Confidence scheduling requires the trained confidence head")
        if not (
            target_runner.codec.capabilities.truncatable
            or gemma
            and target_runner.codec.capabilities.replayable
        ):
            raise ValueError(
                "DSpark target cache must support verified truncate or replay rollback"
            )
        self.target, self.model = target_runner, deepcopy(draft).eval().requires_grad_(False)
        self.draft_policy_artifact_id, self.vocabulary_fingerprint = (
            draft_policy_artifact_id,
            vocabulary_fingerprint,
        )
        self.threshold, self.device = confidence_threshold, next(self.model.parameters()).device
        if self.model.fc.weight.dtype != target_runner.model.get_input_embeddings().weight.dtype:
            raise ValueError(
                "This DSpark serving profile requires matching target/draft stored precision"
            )
        self.pool = PagedStatePool(
            block_size=block_size, max_blocks=max_blocks, codec=KVStateCodec()
        )
        self._request_lock = threading.Lock()

    def generate(self, prompt_ids, config=None, **kwargs):

        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("DSpark decoder already has an active request")
        try:
            with (
                torch.autocast(self.target.device.type, enabled=False),
                torch.autocast(self.device.type, enabled=False),
            ):
                return self._generate(prompt_ids, config, **kwargs)
        finally:
            self._request_lock.release()

    @torch.no_grad()
    def _generate(self, prompt_ids, config=None, *, request_id=None, on_token=None, cancelled=None):
        config = config or SamplingConfig()
        prompt = tuple(prompt_ids)
        c = self.model.config
        width = len(c.target_layer_ids) * c.target.hidden_size
        if not prompt or any(
            type(t) is not int or not 0 <= t < c.target.vocab_size for t in prompt
        ):
            raise ValueError("DSpark prompt must contain target-vocabulary IDs")
        if (
            len(prompt) + config.max_new_tokens + c.block_size - 2
            > c.target.max_position_embeddings
        ):
            raise ValueError(
                "Requested DSpark generation leaves insufficient draft-block positional capacity"
            )
        request_id = request_id or uuid.uuid4().hex
        received = time.monotonic()
        generator = torch.Generator().manual_seed(config.seed)
        target_state = draft_state = None
        output, raw_logs, behavior_logs, timestamps, accepted_tokens = [], [], [], [], []
        text, reason = "", "length"
        stats = dict(
            backbone_calls=0,
            candidate_tokens=0,
            proposed_tokens=0,
            confidence_pruned_tokens=0,
            target_verification_calls=0,
            projected_context_tokens=0,
            rejected_blocks=0,
            bonus_tokens=0,
            draft_backbone_seconds=0.0,
            draft_heads_and_sampling_seconds=0.0,
            confidence_threshold=self.threshold,
            cache_profile=(
                "gemma4_snapshot_full_prefix_replay"
                if isinstance(self.target, Gemma4SnapshotRunner)
                else "native_paged_ownership_contiguous_attention"
            ),
        )
        target_seconds_before, target_tokens_before = (
            self.target.model_execution_seconds,
            self.target.input_tokens_computed,
        )
        replays_before = getattr(self.target.pool, "replay_rollbacks", 0)

        def commit(token, raw, behavior, accepted=False):
            nonlocal text
            output.append(token)
            raw_logs.append(float(raw[token]))
            behavior_logs.append(float(behavior[token]))
            if accepted:
                accepted_tokens.append(token)
            now = time.monotonic()
            timestamps.append(now)
            terminal = token in config.eos_token_ids or len(output) == config.max_new_tokens
            decoded = self.target.stream_text(output, final=terminal)
            if not decoded.startswith(text):
                raise ValueError("Tokenizer is not stream-prefix stable")
            event = TokenEvent(
                request_id,
                self.target.policy_artifact_id,
                len(output) - 1,
                token,
                raw_logs[-1],
                behavior_logs[-1],
                decoded[len(text) :],
                now,
            )
            text = decoded
            if on_token is not None:
                on_token(event)
            return terminal

        def target_forward(tokens):
            logits, features = self.target.forward_feature_batch(
                [target_state],
                [tokens],
                hidden_state_indices=tuple(i + 1 for i in c.target_layer_ids),
            )[0]
            return logits, features[None].to(self.device)

        try:
            target_state = (
                self.target.create_sequence(prompt)[0]
                if isinstance(self.target, Gemma4SnapshotRunner)
                else self.target.pool.create(self.target.policy_artifact_id)
            )
            draft_state = self.pool.create(self.draft_policy_artifact_id)
            features = torch.empty(
                1, 0, width, device=self.device, dtype=self.model.fc.weight.dtype
            )
            if len(prompt) > 1:
                _, features = target_forward(prompt[:-1])
            while len(output) < config.max_new_tokens:
                if cancelled is not None and cancelled():
                    reason = "cancelled"
                    break
                context = prompt + tuple(output)
                base = target_state.length
                if (
                    base != len(context) - 1
                    or features.shape[1] != base
                    or draft_state.length > base
                ):
                    raise RuntimeError("DSpark committed-token/cache/feature invariant failed")
                noise = torch.full(
                    (1, c.block_size), c.mask_token_id, device=self.device, dtype=torch.int64
                )
                noise[0, 0] = context[-1]
                new_context = features[:, draft_state.length : base]
                with self.pool.borrow(draft_state):
                    past = self.pool.materialize(draft_state) if draft_state.length else None
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    started = time.monotonic()
                    hidden, kv = self.model.backbone_cached(noise, new_context, state=past)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    stats["draft_backbone_seconds"] += time.monotonic() - started
                if base > draft_state.length:
                    self.pool.append(draft_state, kv)
                stats["projected_context_tokens"] += new_context.shape[1]
                stats["backbone_calls"] += 1
                previous = noise[:, 0]
                markov_state = None
                head_started = time.monotonic()
                candidates, probabilities, confidences = [], [], []

                proposal_capacity = c.target.max_position_embeddings - base - 1
                for index in range(
                    min(c.block_size, config.max_new_tokens - len(output), proposal_capacity)
                ):
                    h = hidden[:, index]
                    logits = self.model.compute_logits(h)
                    if self.model.markov_head is not None:
                        bias, markov_state = self.model.markov_head.step(h, previous, markov_state)
                        logits = logits + bias
                    confidence = self.model.confidence(h, previous)
                    if confidence is not None:
                        confidences.append(confidence[0])
                    _, q = distributions(
                        logits[0],
                        config,
                        context_ids=context + tuple(candidates),
                        generated_count=len(output) + len(candidates),
                    )
                    token = int(torch.multinomial(q.exp(), 1, generator=generator))
                    candidates.append(token)
                    probabilities.append(q.exp())
                    previous = torch.tensor([token], device=self.device)
                    if token in config.eos_token_ids:
                        break
                stats["candidate_tokens"] += len(candidates)
                keep = (
                    confident_prefix_length(torch.stack(confidences), self.threshold)
                    if confidences
                    else len(candidates)
                )
                stats["confidence_pruned_tokens"] += len(candidates) - keep
                proposals, probabilities = candidates[:keep], probabilities[:keep]
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                stats["draft_heads_and_sampling_seconds"] += time.monotonic() - head_started
                stats["proposed_tokens"] += len(proposals)
                verified, trial_features = target_forward([context[-1], *proposals])
                stats["target_verification_calls"] += 1
                features = torch.cat((features, trial_features), 1)
                terminal = rejected = False
                for index, proposal in enumerate(proposals):
                    raw, p = distributions(
                        verified[index],
                        config,
                        context_ids=prompt + tuple(output),
                        generated_count=len(output),
                    )
                    token, accepted = speculative_accept(
                        p.exp(), probabilities[index], proposal, generator
                    )
                    terminal = commit(token, raw, p, accepted)
                    if not accepted or terminal:
                        retained = base + index + 1
                        self.target.pool.truncate(target_state, retained)
                        features = features[:, :retained]
                        rejected = not accepted
                        if rejected:
                            stats["rejected_blocks"] += 1
                        break
                if not rejected and not terminal:
                    raw, p = distributions(
                        verified[len(proposals)],
                        config,
                        context_ids=prompt + tuple(output),
                        generated_count=len(output),
                    )
                    bonus = int(torch.multinomial(p.exp(), 1, generator=generator))
                    terminal = commit(bonus, raw, p)
                    stats["bonus_tokens"] += 1
                if terminal:
                    reason = "eos" if output[-1] in config.eos_token_ids else "length"
                    break
            stats["target_model_seconds"] = (
                self.target.model_execution_seconds - target_seconds_before
            )
            stats["target_input_tokens"] = self.target.input_tokens_computed - target_tokens_before
            stats["target_replay_rollbacks"] = (
                getattr(self.target.pool, "replay_rollbacks", 0) - replays_before
            )
            return DSparkGenerationResult(
                request_id,
                self.target.policy_artifact_id,
                prompt,
                tuple(output),
                tuple(raw_logs),
                tuple(behavior_logs),
                config.transform_order,
                text,
                reason,
                received,
                received,
                tuple(timestamps),
                time.monotonic(),
                accepted_draft_tokens=tuple(accepted_tokens),
                draft_token_count=stats["proposed_tokens"],
                draft_policy_artifact_id=self.draft_policy_artifact_id,
                sampling_config=asdict(config),
                tokenizer_fingerprint=self.vocabulary_fingerprint,
                dspark_stats=stats,
            )
        finally:
            if target_state is not None:
                self.target.pool.release(target_state)
            if draft_state is not None:
                self.pool.release(draft_state)
