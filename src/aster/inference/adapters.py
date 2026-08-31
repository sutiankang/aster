"""Shared-base online LoRA with content identities and request-scoped residency."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import threading
import torch
from torch import nn
from torch.nn import functional as F
from .runner import ModelRunner
from .paged_attention import PagedAttentionRunner
from .state import StateError, PrefixIdentity
from aster.models.config import LlamaConfig, Qwen2Config, Qwen3Config


@dataclass(frozen=True)
class LoRAWeights:
    a: torch.Tensor  # [rank, in_features]
    b: torch.Tensor  # [out_features, rank]
    alpha: float


class _AdapterLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.active = None

    def forward(self, value):
        result = self.base(value)
        if self.active is not None:
            a, b, scale = self.active
            result = result + F.linear(F.linear(value, a), b) * scale
        return result


class MultiLoRARunner:
    """Serve host-registered adapters over one immutable base model.

    Adapter identities bind their contents to the base artifact. Request pins cover
    queueing, execution, and preemption; a pinned adapter cannot be unloaded. Prefix
    reuse is separated by adapter identity. Selection never merges into base weights."""

    def __init__(self, runner, *, max_adapters=16, max_adapter_bytes=256 * 1024**2):
        if type(runner) not in {ModelRunner, PagedAttentionRunner} or type(
            runner.model.config
        ) not in {LlamaConfig, Qwen2Config, Qwen3Config}:
            raise ValueError("Online LoRA requires the native dense or paged single-worker runner")
        if any(type(x) is not int or x < 1 for x in (max_adapters, max_adapter_bytes)):
            raise ValueError("LoRA residency budgets must be positive integers")
        if type(runner) is ModelRunner and any(
            getattr(layer.self_attn, "attention_backend", None) is not None
            for layer in runner.model.model.layers
        ):
            raise ValueError(
                "Dense fused-provider plus online LoRA requires a separately admitted projection provider"
            )
        self._runner = runner
        self._lock = threading.RLock()
        self._adapters = {}
        self._pins = {}
        self._domains = {}
        self._layers = {}
        self.max_adapters, self.max_adapter_bytes = max_adapters, max_adapter_bytes
        self.resident_bytes = 0
        self._targets = frozenset(
            name for name, module in runner.model.named_modules() if type(module) is nn.Linear
        )

    def __getattr__(self, name):
        return getattr(self._runner, name)

    def register_adapter(self, weights, *, base_artifact_id):
        if (
            base_artifact_id != self.policy_artifact_id
            or not isinstance(weights, dict)
            or not weights
        ):
            raise ValueError("Adapter requires explicit matching base artifact and named targets")
        if any(not isinstance(name, str) or not name for name in weights):
            raise ValueError("Invalid LoRA target")
        if not set(weights) <= self._targets:
            raise ValueError("Adapter targets must be original model Linear paths")
        with self._lock:
            prepared = {}
            modules = {}
            hasher = hashlib.sha256()
            hasher.update(base_artifact_id.encode())
            size = 0
            for name in sorted(weights):
                module = self.model.get_submodule(name)
                linear = module.base if type(module) is _AdapterLinear else module
                value = weights[name]
                if type(linear) is not nn.Linear or not isinstance(value, LoRAWeights):
                    raise ValueError("Only explicit native Linear A/B adapters are supported")
                a, b = value.a, value.b
                if (
                    not isinstance(a, torch.Tensor)
                    or not isinstance(b, torch.Tensor)
                    or a.ndim != 2
                    or b.ndim != 2
                    or a.shape[0] < 1
                    or a.shape[1] != linear.in_features
                    or b.shape != (linear.out_features, a.shape[0])
                    or not a.is_floating_point()
                    or not b.is_floating_point()
                    or not torch.isfinite(a).all()
                    or not torch.isfinite(b).all()
                    or type(value.alpha) not in {int, float}
                    or not math.isfinite(value.alpha)
                    or value.alpha <= 0
                ):
                    raise ValueError("Invalid LoRA matrix layout, finite values or alpha")
                size += (a.numel() + b.numel()) * linear.weight.element_size()
                if size > self.max_adapter_bytes:
                    raise ValueError("Single adapter exceeds residency budget")
                a = (
                    a.detach()
                    .to(device=linear.weight.device, dtype=linear.weight.dtype)
                    .clone()
                    .contiguous()
                )
                b = (
                    b.detach()
                    .to(device=linear.weight.device, dtype=linear.weight.dtype)
                    .clone()
                    .contiguous()
                )
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError("LoRA conversion overflow")
                hasher.update(
                    json.dumps(
                        [name, list(a.shape), list(b.shape), str(a.dtype), float(value.alpha)]
                    ).encode()
                )
                for tensor in (a, b):
                    hasher.update(tensor.cpu().view(torch.uint8).numpy().tobytes())
                prepared[name] = (a, b, float(value.alpha) / a.shape[0])
                modules[name] = module
            identity = "lora:" + hasher.hexdigest()
            if identity in self._adapters:
                return identity
            if (
                len(self._adapters) >= self.max_adapters
                or self.resident_bytes + size > self.max_adapter_bytes
            ):
                raise ValueError(
                    "Adapter residency budget exceeded; explicitly remove an unpinned adapter"
                )
            for name, module in modules.items():
                if name not in self._layers:
                    wrapper = _AdapterLinear(module)
                    parent, _, child = name.rpartition(".")
                    setattr(self.model.get_submodule(parent), child, wrapper)
                    self._layers[name] = wrapper
            self._adapters[identity] = (prepared, size)
            self._pins[identity] = 0
            self.resident_bytes += size
            return identity

    def remove_adapter(self, identity):
        with self._lock:
            if identity not in self._adapters:
                raise ValueError("Unknown adapter")
            if self._pins[identity]:
                raise StateError("Adapter still belongs to queued/running requests")
            _, size = self._adapters.pop(identity)
            self._pins.pop(identity)
            self.resident_bytes -= size

    def register_trained_adapter(self, trained_model, *, base_artifact_id):
        """Import native LoRALinear parameters only after matching every frozen base
        parameter and the model configuration. Do not discard non-adapter training updates."""
        from aster.methods.distillation import LoRALinear

        targets = {
            name: module
            for name, module in trained_model.named_modules()
            if type(module) is LoRALinear
        }
        if not targets:
            raise ValueError("No native trained LoRA modules found")
        with self._lock:
            expected = {}
            for name, value in self.model.state_dict().items():
                for target in self._layers:
                    if name.startswith(target + ".base."):
                        name = target + "." + name[len(target + ".base.") :]
                        break
                expected[name] = value
            actual = {}
            for name, value in trained_model.state_dict().items():
                if any(name in {target + ".a", target + ".b"} for target in targets):
                    continue
                for target in targets:
                    if name.startswith(target + ".base."):
                        name = target + "." + name[len(target + ".base.") :]
                        break
                actual[name] = value
            if expected.keys() != actual.keys() or any(
                expected[name].dtype != actual[name].dtype
                or not torch.equal(expected[name].detach().cpu(), actual[name].detach().cpu())
                for name in expected
            ):
                raise ValueError("Trained adapter base weights differ from this deployment")
            if trained_model.config.to_dict() != self.model.config.to_dict():
                raise ValueError("Adapter model configuration changed")
            return self.register_adapter(
                {
                    name: LoRAWeights(module.a, module.b, module.scale * module.rank)
                    for name, module in targets.items()
                },
                base_artifact_id=base_artifact_id,
            )

    def resolve_model_identity(self, name):
        with self._lock:
            if not isinstance(name, str):
                raise ValueError("Model identity must be a string")
            if name == self.policy_artifact_id:
                return PrefixIdentity(self.policy_artifact_id)
            if name not in self._adapters:
                raise ValueError("Unknown registered model/adapter identity")
            return PrefixIdentity(self.policy_artifact_id, adapter=name)

    def prepare_request(self, prompt, identity, modality_inputs, *, max_prefill_tokens):
        if modality_inputs is not None:
            raise ValueError("This LoRA runner only supports token requests")
        with self._lock:
            if identity.policy_artifact_id != self.policy_artifact_id:
                raise StateError("Base policy mismatch")
            if identity.adapter != "none" and identity.adapter not in self._adapters:
                raise StateError("Unknown adapter")
            key = identity.fingerprint()
            previous, count = self._domains.get(key, (identity.adapter, 0))
            self._domains[key] = (previous, count + 1)
            if identity.adapter != "none":
                self._pins[identity.adapter] += 1
            return identity

    def release_request(self, identity):
        with self._lock:
            key = identity.fingerprint()
            adapter, count = self._domains[key]
            if count == 1:
                self._domains.pop(key)
            else:
                self._domains[key] = adapter, count - 1
            if adapter != "none":
                self._pins[adapter] -= 1

    @contextmanager
    def _selected(self, sequences):
        with self._lock:
            if (
                not sequences
                or len({s.identity for s in sequences}) != 1
                or sequences[0].identity not in self._domains
            ):
                raise StateError("LoRA execution requires one admitted request identity")
            adapter, _ = self._domains[sequences[0].identity]
            weights = self._adapters[adapter][0] if adapter != "none" else {}
            try:
                for name, layer in self._layers.items():
                    layer.active = weights.get(name)
                yield
            finally:
                for layer in self._layers.values():
                    layer.active = None

    def forward_batch(self, sequences, chunks, **kwargs):
        with self._selected(sequences):
            return self._runner.forward_batch(sequences, chunks, **kwargs)

    def forward_feature_batch(self, sequences, chunks, **kwargs):
        with self._selected(sequences):
            return self._runner.forward_feature_batch(sequences, chunks, **kwargs)

    def adapter_metrics(self):
        with self._lock:
            return {
                "adapters": len(self._adapters),
                "resident_bytes": self.resident_bytes,
                "pinned_requests": sum(self._pins.values()),
                "request_domains": len(self._domains),
            }
