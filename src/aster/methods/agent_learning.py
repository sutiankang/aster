"""Verified agent trajectories to action-only supervision without replaying tool effects."""

from dataclasses import dataclass, asdict

from ..agents.events import read_events
from ..core import atomic_json, digest_json
from ..data import causal_collate
from .supervised import CrossEntropyObjective


@dataclass(frozen=True)
class AgentTrainingCorpus:
    examples: tuple[dict, ...]
    receipts: tuple[dict, ...]
    source_log_head: str
    tokenizer_fingerprint: str
    processor_fingerprint: str

    @property
    def identity(self):
        return digest_json(asdict(self))

    def save(self, path):
        atomic_json(path, {"schema_version": 1, "identity": self.identity, "corpus": asdict(self)})


def verified_agent_corpus(
    path, *, expected_policy_id, tokenizer_fingerprint, processor_fingerprint, vocab_size
):
    """Accept completed turns with independent verification results and retain
    explicit reasons for every rejected trajectory."""
    if (
        not all((expected_policy_id, tokenizer_fingerprint, processor_fingerprint))
        or vocab_size < 1
    ):
        raise ValueError("Pinned policy, tokenizer, processor and vocabulary are required")
    events = read_events(path)
    if not events:
        raise ValueError("Cannot train on an empty Agent event log")
    turns = {}
    for event in events:
        kind, turn_id = event["kind"], event["turn_id"]
        if kind == "turn.started":
            if turn_id in turns:
                raise ValueError("Repeated turn identity in trajectory log")
            turns[turn_id] = {
                "traces": [],
                "verified": False,
                "completed": None,
                "ambiguous": False,
                "thread_id": event["thread_id"],
            }
        elif kind in {"model.trace", "verification.result", "turn.completed", "tool.ambiguous"}:
            if turn_id not in turns or turns[turn_id]["completed"] is not None:
                raise ValueError("Training event outside its active turn")
            turn = turns[turn_id]
            if turn["thread_id"] != event["thread_id"]:
                raise ValueError("Turn moved between Agent threads")
            if kind == "model.trace":
                turn["traces"].append(event)
            elif kind == "verification.result":
                turn["verified"] = event["payload"].get("passed") is True
            elif kind == "tool.ambiguous":
                turn["ambiguous"] = True
            else:
                turn["completed"] = event["payload"]
    examples, receipts = [], []
    for turn_id, turn in turns.items():
        completion = turn["completed"]
        reason = None
        if completion is None:
            reason = "incomplete_turn"
        elif completion.get("status") != "verified" or not turn["verified"]:
            reason = "not_independently_verified"
        elif turn["ambiguous"]:
            reason = "ambiguous_tool_outcome"
        elif not turn["traces"]:
            reason = "no_model_actions"
        elif completion.get("trace_sequences") != [event["sequence"] for event in turn["traces"]]:
            reason = "trace_receipt_mismatch"
        selected = []
        if reason is None:
            for event in turn["traces"]:
                trace = event["payload"]
                if (
                    trace.get("policy_artifact_id") != expected_policy_id
                    or trace.get("tokenizer_fingerprint") != tokenizer_fingerprint
                    or trace.get("processor_fingerprint") != processor_fingerprint
                    or not isinstance(trace.get("sampling_config"), dict)
                ):
                    reason = "policy_or_token_semantics_unverified"
                    break
                prompt, action = (
                    trace.get("prompt_token_ids", []),
                    trace.get("action_token_ids", []),
                )
                ids = prompt + action
                if (
                    not prompt
                    or not action
                    or trace.get("loss_mask") != [0] * len(prompt) + [1] * len(action)
                    or any(type(token) is not int or not 0 <= token < vocab_size for token in ids)
                ):
                    reason = "invalid_action_token_layout"
                    break
                if trace.get("stop_reason") not in {"eos", "length"}:
                    reason = "incomplete_model_action"
                    break
                selected.append(
                    {
                        "input_ids": ids,
                        "labels": [-100] * len(prompt) + action,
                        "source_turn_id": turn_id,
                        "source_event_hash": event["hash"],
                    }
                )
        if reason is None:
            examples.extend(selected)
        receipts.append(
            {
                "turn_id": turn_id,
                "accepted": reason is None,
                "reason": reason,
                "action_count": len(turn["traces"]),
                "accepted_action_count": len(selected) if reason is None else 0,
            }
        )
    return AgentTrainingCorpus(
        tuple(examples),
        tuple(receipts),
        events[-1]["hash"],
        tokenizer_fingerprint,
        processor_fingerprint,
    )


class AgentSFTMethod:
    """Train action-only agent supervision through the shared optimizer and checkpoint;
    corpus identity and progress are part of trainer state."""

    def __init__(self, engine, corpus, *, tokenizer, processor_fingerprint):
        if (
            digest_json(tokenizer.to_dict()) != corpus.tokenizer_fingerprint
            or processor_fingerprint != corpus.processor_fingerprint
        ):
            raise ValueError(
                "Student token/processor semantics differ from collected Agent actions"
            )
        if not corpus.examples:
            raise ValueError("No verified actions were admitted; inspect corpus rejection receipts")
        self.engine, self.corpus, self.tokenizer = engine, corpus, tokenizer
        self.objective, self.updates = CrossEntropyObjective(), 0
        engine.register_state("agent_sft", self)

    def update(self, indices):
        if len(indices) < self.engine.accumulation_steps or any(
            type(index) is not int or not 0 <= index < len(self.corpus.examples)
            for index in indices
        ):
            raise ValueError("Provide valid action indices for every accumulation slot")
        rows = [self.corpus.examples[index] for index in indices]
        batches = []
        for index in range(self.engine.accumulation_steps):
            batch = causal_collate(
                rows[index :: self.engine.accumulation_steps],
                pad_token_id=self.tokenizer.pad_token_id,
            )
            batches.append({key: value.to(self.engine.device) for key, value in batch.items()})
        result = self.engine.phase(
            "verified_agent_sft", objective=self.objective, microbatches=batches
        )
        if result.updated:
            self.updates += 1
        return result

    def state_dict(self):
        return {"corpus_identity": self.corpus.identity, "updates": self.updates}

    def load_state_dict(self, state):
        if state.get("corpus_identity") != self.corpus.identity:
            raise ValueError("Agent training evidence changed since checkpoint")
        self.updates = state["updates"]
