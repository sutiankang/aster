"""Two-model speculative decoding with batched verification and transactional cache rollback."""

from __future__ import annotations
import hashlib
import json
import time
import uuid

import torch

from .engine import GenerationResult, TokenEvent
from .sampling import SamplingConfig, distributions, speculative_accept


class SpeculativeDecoder:
    def __init__(self, target, draft, *, num_draft_tokens=4, vocabulary_fingerprint=None):
        if target is draft or num_draft_tokens < 1:
            raise ValueError("Speculation needs independent runners and a positive draft length")
        if not target.codec.capabilities.truncatable or not draft.codec.capabilities.truncatable:
            raise ValueError("Speculation needs explicit truncate or replay state support")
        if target.tokenizer is not None and draft.tokenizer is not None:
            if not hasattr(target.tokenizer, "to_dict") or not hasattr(draft.tokenizer, "to_dict"):
                raise ValueError("Tokenizer identity needs an explicit auditable format")
            target_vocab = json.dumps(target.tokenizer.to_dict(), sort_keys=True)
            draft_vocab = json.dumps(draft.tokenizer.to_dict(), sort_keys=True)
            if target_vocab != draft_vocab:
                raise ValueError("Draft and target token-ID semantics differ")
        elif not vocabulary_fingerprint:
            raise ValueError(
                "Without tokenizers, an explicit shared vocabulary fingerprint is required"
            )
        self.target, self.draft, self.num_draft_tokens = target, draft, num_draft_tokens
        self.vocabulary_fingerprint = vocabulary_fingerprint

    def generate(self, prompt_ids, config=None, *, request_id=None, on_token=None, cancelled=None):
        config = config or SamplingConfig()
        prompt = tuple(prompt_ids)
        if not prompt or any(type(token) is not int or token < 0 for token in prompt):
            raise ValueError("Prompt must contain nonnegative token IDs")
        request_id = request_id or uuid.uuid4().hex
        received = time.monotonic()
        target_state = self.target.pool.create(self.target.policy_artifact_id)
        draft_state = self.draft.pool.create(self.draft.policy_artifact_id)
        generator = torch.Generator().manual_seed(config.seed)
        output, raw_logp, behavior_logp, timestamps, accepted_tokens = [], [], [], [], []
        draft_count, reason, text = 0, "length", ""

        def commit(token, raw, behavior, accepted=False):
            nonlocal text
            output.append(token)
            raw_logp.append(float(raw[token]))
            behavior_logp.append(float(behavior[token]))
            now = time.monotonic()
            timestamps.append(now)
            if accepted:
                accepted_tokens.append(token)
            final = token in config.eos_token_ids or len(output) == config.max_new_tokens
            decoded = self.target.stream_text(output, final=final)
            if not decoded.startswith(text):
                raise ValueError("Processor is not stream-prefix stable")
            event = TokenEvent(
                request_id,
                self.target.policy_artifact_id,
                len(output) - 1,
                token,
                raw_logp[-1],
                behavior_logp[-1],
                decoded[len(text) :],
                now,
            )
            text = decoded
            if on_token is not None:
                on_token(event)
            return final

        try:
            next_target = self.target.forward_batch([target_state], [prompt])[0]
            next_draft = self.draft.forward_batch([draft_state], [prompt])[0]
            while len(output) < config.max_new_tokens:
                if cancelled is not None and cancelled():
                    reason = "cancelled"
                    break
                base_length = target_state.length
                committed_context = prompt + tuple(output)
                draft_tokens, draft_probabilities = [], []
                maximum = min(self.num_draft_tokens, config.max_new_tokens - len(output))
                for _ in range(maximum):
                    _, q_log = distributions(
                        next_draft,
                        config,
                        context_ids=committed_context + tuple(draft_tokens),
                        generated_count=len(output) + len(draft_tokens),
                    )
                    proposal = int(torch.multinomial(q_log.exp(), 1, generator=generator))
                    draft_tokens.append(proposal)
                    draft_probabilities.append(q_log.exp())
                    draft_count += 1
                    next_draft = self.draft.forward_batch([draft_state], [[proposal]])[0]
                    if proposal in config.eos_token_ids:
                        break
                verified = self.target.forward_batch(
                    [target_state], [draft_tokens], return_all_logits=True
                )[0]
                rejected, terminal = False, False
                for index, proposal in enumerate(draft_tokens):
                    logits = next_target if index == 0 else verified[index - 1]
                    raw, p_log = distributions(
                        logits,
                        config,
                        context_ids=prompt + tuple(output),
                        generated_count=len(output),
                    )
                    token, accepted = speculative_accept(
                        p_log.exp().double(),
                        draft_probabilities[index].double(),
                        proposal,
                        generator,
                    )
                    terminal = commit(token, raw, p_log, accepted=accepted)
                    if not accepted:
                        self.target.pool.truncate(target_state, base_length + index)
                        self.draft.pool.truncate(draft_state, base_length + index)
                        next_target = self.target.forward_batch([target_state], [[token]])[0]
                        next_draft = self.draft.forward_batch([draft_state], [[token]])[0]
                        rejected = True
                        break
                    if terminal:
                        self.target.pool.truncate(target_state, base_length + index + 1)
                        self.draft.pool.truncate(draft_state, base_length + index + 1)
                        break
                if terminal:
                    reason = "eos" if output[-1] in config.eos_token_ids else "length"
                    break
                if not rejected:
                    next_target = verified[-1]
            return GenerationResult(
                request_id,
                self.target.policy_artifact_id,
                prompt,
                tuple(output),
                tuple(raw_logp),
                tuple(behavior_logp),
                config.transform_order,
                text,
                reason,
                received,
                received,
                tuple(timestamps),
                time.monotonic(),
                accepted_draft_tokens=tuple(accepted_tokens),
                draft_token_count=draft_count,
                draft_policy_artifact_id=self.draft.policy_artifact_id,
            )
        finally:
            self.target.pool.release(target_state)
            self.draft.pool.release(draft_state)
