"""Native model computation, independent of serving, optimization, and downloads."""

from __future__ import annotations
from contextlib import ExitStack
import copy
import codecs
import time
import torch

from .state import KVStateCodec, PagedStatePool, StateError


class ModelRunner:
    kind = "token_predictor"

    def __init__(
        self,
        model,
        *,
        policy_artifact_id,
        codec=None,
        tokenizer=None,
        block_size=16,
        max_blocks=256,
        chat_template=None,
        kv_quantization=None,
    ):
        if not isinstance(policy_artifact_id, str) or not policy_artifact_id:
            raise ValueError("An immutable policy identity is mandatory")

        runtime = getattr(model, "_aster_shared_runtime_handles", ())
        self.model = copy.deepcopy(model, {id(item): item for item in runtime}).eval()
        self.model.requires_grad_(False)
        self.policy_artifact_id = policy_artifact_id
        self.codec = codec or KVStateCodec()
        self.pool = PagedStatePool(
            block_size=block_size,
            max_blocks=max_blocks,
            codec=self.codec,
            quantization=kv_quantization,
        )
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.device = next(self.model.parameters(), torch.empty(0)).device
        self.input_tokens_computed = 0
        self.forward_calls = 0
        self.model_execution_seconds = 0.0

    @classmethod
    def from_artifact(
        cls, store, artifact_id, *, loader, codec=None, tokenizer_loader=None, **kwargs
    ):

        artifact = store.get(artifact_id, verify=True)
        tokenizer = tokenizer_loader(artifact.path) if tokenizer_loader else None
        return cls(
            loader(artifact.path),
            policy_artifact_id=artifact.id,
            codec=codec,
            tokenizer=tokenizer,
            **kwargs,
        )

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def forward_batch(self, sequences, chunks, *, return_all_logits=False):

        return self._execute_batch(sequences, chunks, return_all_logits=return_all_logits)

    def forward_feature_batch(self, sequences, chunks, *, hidden_state_indices):

        if not hidden_state_indices or any(
            type(i) is not int or i < 0 for i in hidden_state_indices
        ):
            raise ValueError("Explicit nonnegative hidden-state indices are required")
        return self._execute_batch(
            sequences,
            chunks,
            return_all_logits=True,
            hidden_state_indices=tuple(hidden_state_indices),
        )

    def _execute_batch(self, sequences, chunks, *, return_all_logits, hidden_state_indices=None):
        if not sequences or len(sequences) != len(chunks):
            raise ValueError("A nonempty aligned batch is required")
        lengths, chunk_lengths = {s.length for s in sequences}, {len(chunk) for chunk in chunks}
        if len(lengths) != 1 or len(chunk_lengths) != 1 or min(chunk_lengths) < 1:
            raise ValueError("Batch must have equal cached and input lengths")
        if len({s.identity for s in sequences}) != 1:
            raise StateError("A batch cannot cross cache security domains")
        start, count = next(iter(lengths)), next(iter(chunk_lengths))
        ids = torch.tensor(chunks, dtype=torch.long, device=self.device)
        positions = torch.arange(start, start + count, device=self.device).expand(len(chunks), -1)
        mask = torch.ones((len(chunks), start + count), dtype=torch.bool, device=self.device)
        with ExitStack() as stack, torch.inference_mode():
            for sequence in sequences:
                stack.enter_context(self.pool.borrow(sequence))
            states = [self.pool.materialize(sequence) for sequence in sequences]
            state = self.codec.concatenate_batch(states) if start else None
            self._synchronize()
            started = time.monotonic()
            extra = {"output_hidden_states": True} if hidden_state_indices is not None else {}
            output = self.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                state=state,
                use_cache=True,
                **extra,
            )
            self._synchronize()
            self.model_execution_seconds += time.monotonic() - started
            self.input_tokens_computed += ids.numel()
            self.forward_calls += 1
            if (
                output.logits.ndim != 3
                or output.logits.shape[:2] != ids.shape
                or output.state is None
            ):
                raise StateError("TokenPredictor returned invalid cached output")
            features = None
            if hidden_state_indices is not None:
                states = output.hidden_states
                if (
                    states is None
                    or max(hidden_state_indices) >= len(states)
                    or any(states[i].shape[:2] != ids.shape for i in hidden_state_indices)
                ):
                    raise StateError("TokenPredictor did not return aligned target features")
                features = torch.cat([states[i] for i in hidden_state_indices], -1)
            split_states = self.codec.split_batch(output.state)

        for sequence, new_state in zip(sequences, split_states):
            _, _, _, length = self.codec.flatten(new_state)
            if length != start + count:
                raise StateError("Model cache did not append the requested token span")
            self.pool.append(sequence, new_state)
        logits = [
            (row if return_all_logits else row[-1]).detach().float().cpu() for row in output.logits
        ]
        if features is not None:
            return [(row, feature.detach()) for row, feature in zip(logits, features)]
        return logits

    def decode(self, token_ids):
        if self.tokenizer is None:
            return " ".join(str(token) for token in token_ids)
        return self.tokenizer.decode(token_ids)

    def stream_text(self, token_ids, *, final=False):

        if self.tokenizer is not None and hasattr(self.tokenizer, "to_dict"):
            kind = self.tokenizer.to_dict().get("type")
            if kind in {"byte", "byte_bpe"}:
                if kind == "byte":
                    payload = bytes(token - 3 for token in token_ids if 3 <= token < 259)
                else:
                    payload = b"".join(
                        self.tokenizer.pieces[token]
                        for token in token_ids
                        if token not in (0, 1, 2)
                    )
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                return decoder.decode(payload, final=final)
        return self.decode(token_ids)
