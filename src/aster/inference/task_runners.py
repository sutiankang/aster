"""Typed deployment runners for visual features, recurrent tokens, actions, and latent fields."""

from __future__ import annotations
from collections import OrderedDict
import copy
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import threading
import time
import uuid

import torch

from ..core import FieldOutput, TokenOutput
from ..data.actions import ActionSpec, ActionNormalizer
from .engine import GenerationResult, TokenEvent
from .sampling import SamplingConfig, sample_token
from .state import StateError


def _tree_map(value, operation):
    if isinstance(value, torch.Tensor):
        return operation(value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _tree_map(getattr(value, field.name), operation)
                for field in fields(value)
            }
        )
    if isinstance(value, tuple):
        return tuple(_tree_map(item, operation) for item in value)
    if isinstance(value, list):
        return [_tree_map(item, operation) for item in value]
    if isinstance(value, dict):
        return {key: _tree_map(item, operation) for key, item in value.items()}
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError("Only explicit tensor/dataclass/JSON-like trees are supported")


def _tensor_bytes(value):

    value = value.detach().cpu().contiguous()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def _fingerprint(value):
    def tensor(t):
        return {
            "tensor_sha256": hashlib.sha256(_tensor_bytes(t)).hexdigest(),
            "shape": list(t.shape),
            "dtype": str(t.dtype),
        }

    encoded = json.dumps(
        _tree_map(value, tensor),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class _PrivateRunner:
    def __init__(self, model, *, policy_artifact_id):
        if not isinstance(policy_artifact_id, str) or not policy_artifact_id:
            raise ValueError("Immutable policy artifact identity is required")
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.policy_artifact_id = policy_artifact_id
        self.device = next(self.model.parameters(), torch.empty(0)).device
        self.calls = self.failures = self.examples = 0
        self.model_seconds = 0.0
        self._lock = threading.RLock()

    @classmethod
    def from_artifact(cls, store, artifact_id, *, loader, **kwargs):
        artifact = store.get(artifact_id, verify=True)
        return cls(loader(artifact.path), policy_artifact_id=artifact.id, **kwargs)

    def _call(self, function, *args, examples=1, **kwargs):
        with self._lock, torch.inference_mode():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.monotonic()
            self.calls += 1
            try:
                result = function(*args, **kwargs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                self.examples += examples
                return result
            except Exception:
                self.failures += 1
                raise
            finally:
                self.model_seconds += time.monotonic() - started

    def observation(self):
        return {
            "policy_artifact_id": self.policy_artifact_id,
            "clock": "host_monotonic_synchronized_compute",
            "calls": self.calls,
            "failures": self.failures,
            "examples": self.examples,
            "model_seconds": self.model_seconds,
            "evidence_kind": "native_math_reference",
        }


class FieldRunner(_PrivateRunner):
    kind = "field_predictor"

    def __init__(self, model, *, policy_artifact_id, prediction_type):
        super().__init__(model, policy_artifact_id=policy_artifact_id)
        if prediction_type not in {
            "epsilon",
            "x0",
            "v",
            "score",
            "velocity",
            "edm_residual",
            "consistency_residual",
        }:
            raise ValueError("Explicit field parameterization is required")
        self.prediction_type = prediction_type

    def predict(self, sample, time, condition=None, **declared_inputs):
        output = self._call(
            self.model, sample, time, condition, examples=len(sample), **declared_inputs
        )
        if not isinstance(output, FieldOutput) or output.prediction_type != self.prediction_type:
            raise ValueError("Field prediction type differs from deployment/solver contract")
        if output.prediction.shape != sample.shape or not torch.isfinite(output.prediction).all():
            raise ValueError(
                "Field output must be a finite sample-shaped tensor; learned-variance output needs a separate contract"
            )
        return output


class LatentRunner(_PrivateRunner):
    kind = "latent_codec"

    def encode(self, images, *, sample=False, seed=0):

        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self._call(
            self.model.latent, images, sample=sample, generator=generator, examples=len(images)
        )

    def decode(self, scaled_latent):
        return self._call(
            self.model.decode, scaled_latent, scaled=True, examples=len(scaled_latent)
        )


@dataclass(frozen=True)
class ActionChunk:
    actions: torch.Tensor
    valid: torch.Tensor
    spec: ActionSpec
    policy_artifact_id: str
    state: object = None


class ActionRunner(_PrivateRunner):
    kind = "action_chunk_policy"

    def __init__(self, model, *, policy_artifact_id, spec, normalizer=None, pad_threshold=0.5):
        super().__init__(model, policy_artifact_id=policy_artifact_id)
        if not isinstance(spec, ActionSpec) or not 0 < pad_threshold < 1:
            raise ValueError("Action deployment requires physical ActionSpec and padding threshold")
        if normalizer is not None and (
            not isinstance(normalizer, ActionNormalizer) or normalizer.spec != spec
        ):
            raise ValueError("Normalization statistics must bind the exact ActionSpec")
        self.spec, self.normalizer, self.pad_threshold = (
            spec,
            copy.deepcopy(normalizer),
            pad_threshold,
        )

    def predict_chunk(self, observation, *, state=None):
        output = self._call(self.model.predict_chunk, observation, state=state)
        actions, pad = output.actions, output.pad_logits
        if (
            actions.ndim != 3
            or actions.shape[-1] != len(self.spec.names)
            or pad.shape != actions.shape[:2]
            or actions.shape[1] < self.spec.execution_horizon
            or not torch.isfinite(actions).all()
            or torch.isnan(pad).any()
        ):
            raise ValueError("Action model output does not satisfy the explicit execution contract")

        valid = pad.sigmoid() < self.pad_threshold
        if not valid[:, : self.spec.execution_horizon].all():
            raise ValueError(
                "Requested execution horizon contains predicted padding; no stale-action fallback"
            )
        physical = self.normalizer.denormalize(actions) if self.normalizer else actions
        if not torch.isfinite(physical).all():
            raise ValueError("Action denormalization overflowed physical coordinates")
        return ActionChunk(physical, valid, self.spec, self.policy_artifact_id, output.state)


@dataclass(frozen=True)
class DynamicsStateHandle:
    policy_artifact_id: str
    native_state: object


class DynamicsRunner(_PrivateRunner):
    kind = "latent_dynamics"

    def observe(self, observations, actions, is_first, *, state=None, sample=False, seed=0):
        if state is not None:
            state = self._unwrap(state)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        sequence, prior, final = self._call(
            self.model.observe,
            observations,
            actions,
            is_first,
            state=state,
            sample=sample,
            generator=generator,
            examples=len(observations),
        )
        return {
            "state": DynamicsStateHandle(self.policy_artifact_id, sequence),
            "prior_logits": prior,
            "final_state": DynamicsStateHandle(self.policy_artifact_id, final),
            **self._call(self.model.predictions, sequence, examples=0),
        }

    def imagine(self, state, actions, *, sample=False, seed=0):

        generator = torch.Generator(device=self.device).manual_seed(seed)
        sequence = self._call(
            self.model.imagine,
            self._unwrap(state),
            actions,
            sample=sample,
            generator=generator,
            examples=len(actions),
        )
        return {
            "state": DynamicsStateHandle(self.policy_artifact_id, sequence),
            **self._call(self.model.predictions, sequence, examples=0),
        }

    def _unwrap(self, state):
        if (
            not isinstance(state, DynamicsStateHandle)
            or state.policy_artifact_id != self.policy_artifact_id
        ):
            raise StateError("Dynamics snapshot belongs to another policy version")
        return state.native_state.fork()


class EncoderRunner(_PrivateRunner):
    kind = "modality_encoder"

    def __init__(self, model, *, policy_artifact_id, processor_id, max_cache_bytes=64 * 1024**2):
        super().__init__(model, policy_artifact_id=policy_artifact_id)
        if not processor_id or type(max_cache_bytes) is not int or max_cache_bytes < 1:
            raise ValueError("Encoder cache requires processor identity and a byte bound")
        self.processor_id, self.max_cache_bytes = processor_id, max_cache_bytes
        self._cache = OrderedDict()
        self.cache_bytes = self.cache_hits = 0

    def encode(self, inputs, *, tenant="local", use_cache=True):
        if not tenant or not isinstance(inputs, dict):
            raise ValueError("Explicit tenant and named modality inputs are required")
        key = _fingerprint(
            {
                "policy": self.policy_artifact_id,
                "processor": self.processor_id,
                "tenant": tenant,
                "inputs": inputs,
            }
        )
        with self._lock:
            if use_cache and key in self._cache:
                self.cache_hits += 1
                self._cache.move_to_end(key)
                return _tree_map(self._cache[key][0], lambda t: t.to(self.device).clone())
            output = self._call(self.model, **inputs)
            size = 0

            def store_tensor(tensor):
                nonlocal size
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError("Cannot cache non-finite encoder features")
                size += tensor.numel() * tensor.element_size()
                return tensor.detach().cpu().clone()

            stored = _tree_map(output, store_tensor)
            if use_cache and size <= self.max_cache_bytes:
                while self._cache and self.cache_bytes + size > self.max_cache_bytes:
                    _, (_, removed) = self._cache.popitem(last=False)
                    self.cache_bytes -= removed
                self._cache[key] = stored, size
                self.cache_bytes += size
            return _tree_map(output, lambda t: t.detach().clone())


@dataclass(frozen=True)
class TokenStateHandle:
    policy_artifact_id: str
    processor_id: str
    native_state: object


class StatefulTokenRunner(_PrivateRunner):
    kind = "token_predictor_snapshot"
    supported_states = {
        "dense_kv",
        "window_kv",
        "mla_latent",
        "indexed_mla",
        "hybrid_delta",
        "mamba_ssm",
        "qwen3_vl_kv",
        "compressed_window_mqa",
        "gemma4_shared_kv",
    }

    def __init__(self, model, *, policy_artifact_id, processor_id="none", tokenizer=None):
        super().__init__(model, policy_artifact_id=policy_artifact_id)
        self.processor_id, self.tokenizer = processor_id, tokenizer

    def forward(self, input_ids, *, state=None, modality_inputs=None):
        native = None
        if state is not None:
            if not isinstance(state, TokenStateHandle) or (
                state.policy_artifact_id,
                state.processor_id,
            ) != (self.policy_artifact_id, self.processor_id):
                raise StateError("Token snapshot belongs to another policy/processor")
            native = state.native_state
            if getattr(native, "kind", None) not in self.supported_states:
                raise StateError("Unknown state kind needs an explicit runner")
            native = native.fork()
        inputs = dict(modality_inputs or {})
        if set(inputs) & {"state", "use_cache", "input_ids", "inputs_embeds"}:
            raise ValueError("Modality inputs cannot override runner cache controls")
        output = self._call(
            self.model,
            input_ids=input_ids,
            state=native,
            use_cache=True,
            examples=len(input_ids),
            **inputs,
        )
        if (
            not isinstance(output, TokenOutput)
            or getattr(output.state, "kind", None) not in self.supported_states
        ):
            raise StateError("Model returned an unsupported native state")
        return TokenOutput(
            output.logits,
            TokenStateHandle(self.policy_artifact_id, self.processor_id, output.state),
            output.hidden_states,
            output.auxiliary,
        )

    def fork(self, handle):
        if not isinstance(handle, TokenStateHandle) or (
            handle.policy_artifact_id,
            handle.processor_id,
        ) != (self.policy_artifact_id, self.processor_id):
            raise StateError("Foreign snapshot")
        return TokenStateHandle(
            self.policy_artifact_id, self.processor_id, handle.native_state.fork()
        )

    def replay(self, input_ids, *, modality_inputs=None):
        """Recompute state from complete explicit inputs. Visual spans must remain
        complete; recurrent state does not support arbitrary prefix truncation."""
        return self.forward(input_ids, modality_inputs=modality_inputs)

    def _generation_inputs(self, prompt, modality_inputs):

        from ..models.decoder import CausalLM
        from ..models.moe import MixtralForCausalLM, DeepSeekV3ForCausalLM
        from ..models.hybrid import Qwen3NextForCausalLM
        from ..models.qwen35 import Qwen35ForCausalLM
        from ..models.sparse import DeepSeekV32ForCausalLM
        from ..models.deepseek_v4 import DeepSeekV4ForCausalLM
        from ..models.mamba import MambaForCausalLM
        from ..models.gpt import GPT2ForCausalLM
        from ..models.qwen_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextForCausalLM
        from ..models.gemma4 import Gemma4ForCausalLM
        from ..models.gemma4_vl import Gemma4ForConditionalGeneration

        if modality_inputs is not None and not isinstance(modality_inputs, dict):
            raise ValueError("Generation modality_inputs must be an explicit dictionary")
        inputs = dict(modality_inputs or {})
        if inputs.get("position_ids") is not None:
            raise ValueError(
                "Custom position_ids need explicit per-step forward; generate cannot infer their continuation"
            )
        inputs.pop("position_ids", None)
        padding = inputs.pop("attention_mask", None)
        media_fields = {
            Qwen3VLForConditionalGeneration: {
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
                "mm_token_type_ids",
            },
            Gemma4ForConditionalGeneration: {
                "pixel_values",
                "image_position_ids",
                "image_batch_indices",
                "pixel_values_videos",
                "video_position_ids",
                "video_batch_indices",
                "mm_token_type_ids",
            },
        }
        if set(inputs) - media_fields.get(type(self.model), set()):
            raise ValueError(
                "Unclassified generation input fields require explicit per-step forward"
            )
        if padding is None:
            return inputs, None

        physical_token_models = {
            CausalLM,
            MixtralForCausalLM,
            DeepSeekV3ForCausalLM,
            Qwen3NextForCausalLM,
            Qwen35ForCausalLM,
            DeepSeekV32ForCausalLM,
            DeepSeekV4ForCausalLM,
            MambaForCausalLM,
            GPT2ForCausalLM,
            Qwen3VLForConditionalGeneration,
            Qwen3VLTextForCausalLM,
            Gemma4ForCausalLM,
            Gemma4ForConditionalGeneration,
        }
        if type(self.model) not in physical_token_models:
            raise ValueError("This model needs an explicit generation mask/physical-state protocol")
        if (
            not isinstance(padding, torch.Tensor)
            or padding.shape != (1, len(prompt))
            or padding.device != self.device
            or padding.requires_grad
            or padding.is_complex()
            or not ((padding == 0) | (padding == 1)).all()
        ):
            raise ValueError(
                "Generation attention_mask must be fixed binary [1,prompt_length] on the model device"
            )
        if not bool(padding[0, -1]):
            raise ValueError("Generation must end its prompt at a valid token, not a padding query")
        if type(self.model) is DeepSeekV4ForCausalLM and not bool(padding.all()):
            raise ValueError("DeepSeekV4 currently supports only contiguous unpadded generation")

        return inputs, padding.detach().bool().clone()

    def generate(self, prompt, config=None, *, modality_inputs=None, cancelled=None, on_token=None):
        config = config or SamplingConfig()
        if not prompt or any(type(i) is not int or i < 0 for i in prompt):
            raise ValueError("A nonempty token prompt is required")
        prompt = list(prompt)
        prefill_inputs, history_padding = self._generation_inputs(prompt, modality_inputs)
        received, request_id = time.monotonic(), uuid.uuid4().hex
        generator, state = torch.Generator().manual_seed(config.seed), None
        tokens, raw, behavior, times = [], [], [], []
        transforms = config.transform_order
        reason = "length"
        for index in range(config.max_new_tokens):
            if cancelled and cancelled():
                reason = "cancelled"
                break
            chunk = prompt if state is None else [tokens[-1]]
            inputs = dict(prefill_inputs) if state is None else {}
            if history_padding is not None:
                if state is not None:
                    history_padding = torch.cat(
                        (history_padding, history_padding.new_ones((1, 1))), -1
                    )
                inputs["attention_mask"] = history_padding
            output = self.forward(
                torch.tensor([chunk], device=self.device), state=state, modality_inputs=inputs
            )
            state = output.state
            if (
                history_padding is not None
                and getattr(state.native_state, "seen_tokens", None) != history_padding.shape[1]
            ):
                raise StateError(
                    "Generated cache physical length differs from the declared mask protocol"
                )
            sampling_context = prompt + tokens
            if getattr(state.native_state, "kind", None) == "gemma4_shared_kv":
                from .gemma4 import Gemma4SnapshotRunner

                sampling_context = Gemma4SnapshotRunner.sampling_context_ids(self, sampling_context)
                transforms = (
                    Gemma4SnapshotRunner.sampling_prefix_transforms + config.transform_order
                )
            selected = sample_token(
                output.logits[0, -1],
                config,
                generator,
                context_ids=sampling_context,
                generated_count=index,
            )
            tokens.append(selected.token_id)
            raw.append(selected.raw_model_logprob)
            behavior.append(selected.behavior_logprob)
            times.append(time.monotonic())
            text = self.tokenizer.decode(tokens) if self.tokenizer else " ".join(map(str, tokens))
            if on_token:
                on_token(
                    TokenEvent(
                        request_id,
                        self.policy_artifact_id,
                        index,
                        tokens[-1],
                        raw[-1],
                        behavior[-1],
                        text,
                        times[-1],
                    )
                )
            if tokens[-1] in config.eos_token_ids and len(tokens) >= config.min_new_tokens:
                reason = "eos"
                break
        return GenerationResult(
            request_id,
            self.policy_artifact_id,
            tuple(prompt),
            tuple(tokens),
            tuple(raw),
            tuple(behavior),
            transforms,
            self.tokenizer.decode(tokens) if self.tokenizer else " ".join(map(str, tokens)),
            reason,
            received,
            received,
            tuple(times),
            time.monotonic(),
        )
