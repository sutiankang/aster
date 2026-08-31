"""Strict local configuration and weight loading without remote-code execution."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch
from aster.core import atomic_json, read_json


def configuration_key(config):
    return hashlib.sha256(
        json.dumps(
            config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def semantic_buffers(model):

    result = {}
    for prefix, module in model.named_modules(remove_duplicate=False):
        names = getattr(module, "_aster_semantic_buffers", ())
        if (
            not isinstance(names, tuple)
            or any(not isinstance(n, str) or not n or "." in n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Semantic buffer declarations must be unique local names in a tuple")
        for name in names:
            value = module._buffers.get(name)
            if name not in module._non_persistent_buffers_set or not isinstance(
                value, torch.Tensor
            ):
                raise ValueError("Semantic buffer must name an existing nonpersistent tensor")
            result[(prefix + "." if prefix else "") + name] = value
    return result


class LocalModelMixin:
    @staticmethod
    def _runtime_buffers(model):

        return semantic_buffers(model)

    @staticmethod
    def _atomic_tensors(target, tensors):
        descriptor, temporary = tempfile.mkstemp(prefix=".weights.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(tensors, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save_pretrained(self, path):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        weights = target / "model.pt"
        tensors = self.state_dict()
        self._atomic_tensors(weights, tensors)
        manifest = {
            "schema_version": 1,
            "config": self.config.to_dict(),
            "weight_format": "torch_weights_only",
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "config_sha256": configuration_key(self.config),
            "tensor_dtypes": {name: str(value.dtype) for name, value in tensors.items()},
        }
        runtime = self._runtime_buffers(self)
        if runtime:
            auxiliary = target / "runtime_buffers.pt"
            self._atomic_tensors(auxiliary, runtime)
            manifest["runtime_buffers"] = {
                "sha256": hashlib.sha256(auxiliary.read_bytes()).hexdigest(),
                "tensor_dtypes": {name: str(value.dtype) for name, value in runtime.items()},
            }

        atomic_json(target / "config.json", manifest)

    @classmethod
    def from_pretrained(cls, path):
        from . import build_model
        from .config import config_from_dict

        target = Path(path)
        manifest = read_json(target / "config.json")
        if manifest["schema_version"] != 1 or manifest["weight_format"] != "torch_weights_only":
            raise ValueError("Unsupported local model format")
        weights = target / "model.pt"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
            raise ValueError("Model weight checksum mismatch")
        config = config_from_dict(manifest["config"])
        if configuration_key(config) != manifest["config_sha256"]:
            raise ValueError("Model configuration checksum mismatch")
        model = build_model(config)
        if cls is not LocalModelMixin and not isinstance(model, cls):
            raise ValueError("Saved model architecture differs from requested class")
        tensors = torch.load(weights, map_location="cpu", weights_only=True)
        if not isinstance(tensors, dict) or any(
            not isinstance(v, torch.Tensor) for v in tensors.values()
        ):
            raise ValueError("Model weights must be a tensor state dictionary")
        if manifest.get("tensor_dtypes") != {
            name: str(value.dtype) for name, value in tensors.items()
        }:
            raise ValueError("Weight dtype manifest mismatch")

        aliases = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        for names in aliases.values():
            if len(names) > 1 and any(
                tensors[n].dtype != tensors[names[0]].dtype
                or not torch.equal(tensors[n], tensors[names[0]])
                for n in names[1:]
            ):
                raise ValueError("A tied parameter has inconsistent saved tensor values")
        model.load_state_dict(tensors, strict=True, assign=True)
        for names in aliases.values():
            first = model.get_parameter(names[0])
            for name in names[1:]:
                owner, _, attribute = name.rpartition(".")
                setattr(model.get_submodule(owner), attribute, first)
        runtime_manifest = manifest.get("runtime_buffers")
        if runtime_manifest is not None:
            auxiliary = target / "runtime_buffers.pt"
            if hashlib.sha256(auxiliary.read_bytes()).hexdigest() != runtime_manifest["sha256"]:
                raise ValueError("Runtime buffer checksum mismatch")
            runtime = torch.load(auxiliary, map_location="cpu", weights_only=True)
            expected = cls._runtime_buffers(model)
            if (
                not isinstance(runtime, dict)
                or set(runtime) != set(expected)
                or any(not isinstance(v, torch.Tensor) for v in runtime.values())
            ):
                raise ValueError("Runtime buffer names differ from the configured model")
            if runtime_manifest["tensor_dtypes"] != {
                name: str(value.dtype) for name, value in runtime.items()
            }:
                raise ValueError("Runtime buffer dtype manifest mismatch")
            for name, value in runtime.items():
                if value.shape != expected[name].shape or value.layout != expected[name].layout:
                    raise ValueError("Runtime buffer shape/layout mismatch")
                owner, _, attribute = name.rpartition(".")
                model.get_submodule(owner)._buffers[attribute] = value

        return model
