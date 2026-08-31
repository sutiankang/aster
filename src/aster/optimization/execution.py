"""Static-shape torch.compile and CUDA Graph preparation, execution, and cleanup."""

from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field
import threading
import time
import torch


def _signature(inputs):
    if not inputs or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor) or value.requires_grad
        for key, value in inputs.items()
    ):
        raise ValueError("Execution buckets require nonempty named, inference-only tensor inputs")
    if any(not value.is_contiguous() for value in inputs.values()):
        raise ValueError("Bucket inputs must be contiguous; layout conversion must be explicit")
    return tuple(
        (key, tuple(value.shape), str(value.dtype), str(value.device), tuple(value.stride()))
        for key, value in sorted(inputs.items())
    )


def _clone_output(output):
    if not isinstance(output, torch.Tensor):
        raise ValueError(
            "Static execution provider currently returns one Tensor, not arbitrary model/cache objects"
        )
    if not torch.isfinite(output).all():
        raise ValueError("Execution provider produced a non-finite output")
    return output.detach().clone()


@dataclass
class ExecutionBucket:
    signature: tuple
    status: str = "preparing"
    prepare_seconds: float = 0.0
    forward_seconds: float = 0.0
    calls: int = 0
    failure_type: str | None = None
    callable: object = field(default=None, repr=False)
    static_inputs: dict | None = field(default=None, repr=False)
    static_output: torch.Tensor | None = field(default=None, repr=False)


class _ExecutionProvider:
    def __init__(self, model, *, policy_artifact_id, max_buckets=4, atol=1e-5, rtol=1e-4):
        if (
            not policy_artifact_id
            or type(max_buckets) is not int
            or max_buckets < 1
            or not all(math.isfinite(x) and x >= 0 for x in (atol, rtol))
        ):
            raise ValueError("Provider requires policy identity and positive bucket bounds")
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.policy_artifact_id, self.max_buckets = policy_artifact_id, max_buckets
        self.atol, self.rtol = atol, rtol
        self._versions = self._tensor_versions()
        self.buckets, self._closed, self._lock = {}, False, threading.RLock()

    def _tensor_versions(self):

        return tuple(
            (name, id(value), value._version)
            for name, value in (*self.model.named_parameters(), *self.model.named_buffers())
        )

    def _policy_check(self):
        if self._closed:
            raise RuntimeError("Execution provider is closed")
        current = self._tensor_versions()
        if current != self._versions:
            raise RuntimeError(
                "Policy tensor version changed; all compiled/captured buckets are invalid"
            )

    def _new_bucket(self, name, inputs):
        self._policy_check()
        if (
            not isinstance(name, str)
            or not name
            or name in self.buckets
            or len(self.buckets) >= self.max_buckets
        ):
            raise ValueError("Bucket identity already exists or provider capacity is exhausted")
        bucket = ExecutionBucket(_signature(inputs))
        self.buckets[name] = bucket
        return bucket

    @staticmethod
    def _synchronize(inputs):
        for device in {value.device for value in inputs.values() if value.device.type == "cuda"}:
            torch.cuda.synchronize(device)

    def _get(self, name, inputs):
        self._policy_check()
        bucket = self.buckets.get(name)
        if bucket is None or bucket.status != "ready":
            raise RuntimeError("Execution bucket is absent/not ready; no eager fallback")
        if _signature(inputs) != bucket.signature:
            raise ValueError("Input shape/dtype/device/layout differs from the prepared bucket")
        return bucket

    def observation(self):
        return {
            "policy_artifact_id": self.policy_artifact_id,
            "provider": self.provider,
            "evidence_kind": self.evidence_kind,
            "torch_version": torch.__version__,
            "clock": "host_monotonic_synchronized_forward",
            "closed": self._closed,
            "buckets": {
                name: {
                    "status": b.status,
                    "signature": b.signature,
                    "prepare_seconds": b.prepare_seconds,
                    "forward_seconds": b.forward_seconds,
                    "calls": b.calls,
                    "failure_type": b.failure_type,
                }
                for name, b in self.buckets.items()
            },
        }

    def close(self):
        with self._lock:
            if any(parameter.device.type == "cuda" for parameter in self.model.parameters()):
                torch.cuda.synchronize()
            self.buckets.clear()
            self._closed = True


class CompileProvider(_ExecutionProvider):
    def __init__(self, model, *, policy_artifact_id, backend="inductor", **kwargs):
        if backend not in {"inductor", "aot_eager"}:
            raise ValueError("Explicit native torch.compile backend required")
        super().__init__(model, policy_artifact_id=policy_artifact_id, **kwargs)
        self.backend, self.provider = backend, "torch_compile_" + backend
        self.evidence_kind = (
            "accelerated_kernel" if backend == "inductor" else "native_math_reference"
        )

    def prepare(self, name, example_inputs):
        with self._lock, torch.inference_mode():
            bucket = self._new_bucket(name, example_inputs)
            started = time.monotonic()
            try:
                expected = _clone_output(self.model(**example_inputs))
                compiled = torch.compile(
                    self.model, backend=self.backend, fullgraph=True, dynamic=False
                )

                actual = _clone_output(compiled(**example_inputs))
                self._synchronize(example_inputs)
                torch.testing.assert_close(actual, expected, atol=self.atol, rtol=self.rtol)
                self._policy_check()
                bucket.callable, bucket.status = compiled, "ready"
            except Exception as error:
                bucket.status, bucket.failure_type = "failed", type(error).__name__
                raise
            finally:
                bucket.prepare_seconds = time.monotonic() - started
        return bucket

    def __call__(self, name, **inputs):
        with self._lock, torch.inference_mode():
            bucket = self._get(name, inputs)
            self._synchronize(inputs)
            started = time.monotonic()
            try:
                result = _clone_output(bucket.callable(**inputs))
                self._synchronize(inputs)
                bucket.calls += 1
                return result
            except Exception as error:
                bucket.status, bucket.failure_type = "failed", type(error).__name__
                raise
            finally:
                bucket.forward_seconds += time.monotonic() - started


class CUDAGraphProvider(_ExecutionProvider):
    provider = "torch_cuda_graph"
    evidence_kind = "accelerated_kernel"

    def __init__(self, model, *, policy_artifact_id, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA Graph capture requires a provisioned CUDA runtime/device; no fake CPU fallback"
            )
        super().__init__(model, policy_artifact_id=policy_artifact_id, **kwargs)
        devices = {tensor.device for tensor in (*self.model.parameters(), *self.model.buffers())}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("CUDA Graph provider requires a single-device CUDA model")
        self.device = next(iter(devices))

    def prepare(self, name, example_inputs, *, warmup_steps=3):
        if type(warmup_steps) is not int or warmup_steps < 1:
            raise ValueError("Graph warmup must have a positive explicit step count")
        with self._lock, torch.cuda.device(self.device), torch.inference_mode():
            if any(value.device != self.device for value in example_inputs.values()):
                raise ValueError("Graph inputs must reside on the model device")
            bucket = self._new_bucket(name, example_inputs)
            started = time.monotonic()
            try:
                static = {key: value.detach().clone() for key, value in example_inputs.items()}
                expected = _clone_output(self.model(**static))
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(warmup_steps):
                        self.model(**static)
                torch.cuda.current_stream(self.device).wait_stream(stream)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = self.model(**static)
                graph.replay()
                torch.cuda.synchronize(self.device)
                torch.testing.assert_close(output, expected, atol=self.atol, rtol=self.rtol)
                self._policy_check()
                bucket.callable, bucket.static_inputs, bucket.static_output = graph, static, output
                bucket.status = "ready"
            except Exception as error:
                bucket.status, bucket.failure_type = "failed", type(error).__name__
                raise
            finally:
                bucket.prepare_seconds = time.monotonic() - started
        return bucket

    def __call__(self, name, **inputs):
        with self._lock, torch.cuda.device(self.device), torch.inference_mode():
            bucket = self._get(name, inputs)
            torch.cuda.synchronize(self.device)
            started = time.monotonic()
            try:
                for key, value in inputs.items():
                    bucket.static_inputs[key].copy_(value)
                bucket.callable.replay()
                torch.cuda.synchronize(self.device)

                output = _clone_output(bucket.static_output)
                torch.cuda.synchronize(self.device)
                bucket.calls += 1
                return output
            except Exception as error:
                bucket.status, bucket.failure_type = "failed", type(error).__name__
                raise
            finally:
                bucket.forward_seconds += time.monotonic() - started
