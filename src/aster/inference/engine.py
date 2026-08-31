"""Transport-independent continuous batching, backpressure, and cancellation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import math
import time
import uuid
from typing import Any

import torch
from aster.core.async_work import settle_thread

from .sampling import SamplingConfig, sample_token
from .state import CacheCapacityError, PrefixCache, PrefixIdentity


class OverloadedError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    policy_artifact_id: str
    index: int
    token_id: int
    raw_model_logprob: float
    behavior_logprob: float
    text: str
    timestamp: float


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    policy_artifact_id: str
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    raw_model_logprobs: tuple[float, ...]
    behavior_logprobs: tuple[float, ...]
    sampling_transform_order: tuple[str, ...]
    text: str
    stop_reason: str
    received_at: float
    started_at: float | None
    token_timestamps: tuple[float, ...]
    finished_at: float
    prefix_hit_tokens: int = 0
    error_code: str | None = None
    accepted_draft_tokens: tuple[int, ...] = ()
    draft_token_count: int = 0
    draft_policy_artifact_id: str | None = None
    preemption_count: int = 0

    sampling_config: dict | None = None
    tokenizer_fingerprint: str | None = None
    processor_fingerprint: str | None = None
    adapter_id: str = "none"

    def metrics(self):

        times = self.token_timestamps
        return {
            "clock": "server_monotonic_emit",
            "queue_seconds": None
            if self.started_at is None
            else self.started_at - self.received_at,
            "ttft_seconds": None if not times else times[0] - self.received_at,
            "itl_seconds": [b - a for a, b in zip(times, times[1:])],
            "tpot_seconds": None if len(times) < 2 else (times[-1] - times[0]) / (len(times) - 1),
            "end_to_end_seconds": self.finished_at - self.received_at,
            "output_tokens": len(self.token_ids),
        }


@dataclass
class _Request:
    request_id: str
    prompt: tuple[int, ...]
    config: SamplingConfig
    identity: PrefixIdentity
    received_at: float
    deadline: float
    queue: asyncio.Queue
    completion: asyncio.Future
    generator: torch.Generator
    state: Any = None
    started_at: float | None = None
    token_ids: list[int] = field(default_factory=list)
    raw_logp: list[float] = field(default_factory=list)
    behavior_logp: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    text: str = ""
    cancel_requested: bool = False
    prefix_hit_tokens: int = 0
    preemptions: int = 0
    grammar: Any = None
    archive_handle: str | None = None


class RequestHandle:
    def __init__(self, engine, request):
        self._engine, self._request = engine, request

    @property
    def request_id(self):
        return self._request.request_id

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._request.queue.empty() and self._request.completion.done():
            raise StopAsyncIteration
        event = await self._request.queue.get()
        if event is None:
            raise StopAsyncIteration
        return event

    async def result(self):

        return await asyncio.shield(self._request.completion)

    async def cancel(self):
        if not self._request.completion.done():
            self._engine.cancel(self.request_id)
        return await self.result()

    async def collect(self):

        async for _ in self:
            pass
        return await self.result()


class InferenceEngine:
    def __init__(
        self,
        runner,
        *,
        max_active=8,
        max_queued=64,
        max_batch_tokens=128,
        prefill_chunk_size=64,
        max_output_events=256,
        max_prompt_tokens=32768,
        max_generation_tokens=4096,
        prefix_cache_entries=128,
        offload_archive=None,
    ):
        if (
            min(
                max_active,
                max_queued,
                max_batch_tokens,
                prefill_chunk_size,
                max_output_events,
                max_prompt_tokens,
                max_generation_tokens,
            )
            < 1
        ):
            raise ValueError("All scheduler limits must be positive")
        self.runner = runner
        if offload_archive is not None:
            from .offload import PagedStateArchive

            if (
                not isinstance(offload_archive, PagedStateArchive)
                or offload_archive.pool is not runner.pool
            ):
                raise ValueError("Online offload archive must own this exact paged pool")
        self.offload_archive = offload_archive
        self.offload_capacity_fallbacks = 0
        self.max_active, self.max_queued = max_active, max_queued
        self.max_batch_tokens, self.prefill_chunk_size = max_batch_tokens, prefill_chunk_size
        self.max_output_events = max_output_events
        self.max_prompt_tokens, self.max_generation_tokens = (
            max_prompt_tokens,
            max_generation_tokens,
        )
        self.prefix_cache = (
            runner.create_prefix_cache(max_entries=prefix_cache_entries)
            if hasattr(runner, "create_prefix_cache")
            else PrefixCache(runner.pool, max_entries=prefix_cache_entries)
        )
        self._pending = OrderedDict()
        self._active = OrderedDict()
        self._all = {}
        self._worker = None
        self._wake = None
        self._closing = False
        self._fatal = False
        self._admission_paused = False
        self.completed_count = 0
        self.failed_count = 0
        self.emitted_tokens = 0
        self.observation_started_at = time.monotonic()

    @property
    def ready(self):
        return not self._closing and not self._fatal

    @property
    def active_count(self):
        return len(self._active)

    @property
    def queued_count(self):
        return len(self._pending)

    async def start(self):
        if not self.ready:
            raise RuntimeError("Inference engine is not accepting requests")
        if self._worker is None:
            self._wake = asyncio.Event()
            self._worker = asyncio.create_task(self._run(), name="aster-native-inference")

    async def submit(
        self,
        prompt_ids,
        config=None,
        *,
        request_id=None,
        timeout_s=60.0,
        identity=None,
        grammar=None,
        modality_inputs=None,
    ):
        await self.start()
        config = config or SamplingConfig()
        prompt = tuple(prompt_ids)
        if (
            not prompt
            or len(prompt) > self.max_prompt_tokens
            or any(type(token) is not int or token < 0 for token in prompt)
        ):
            raise ValueError("Prompt must contain bounded nonnegative integer token IDs")
        if config.max_new_tokens > self.max_generation_tokens:
            raise ValueError("Requested generation exceeds deployment limit")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Timeout must be positive and finite")
        if len(self._all) >= self.max_active + self.max_queued:
            raise OverloadedError("Admission queue is full")
        identity = identity or PrefixIdentity(self.runner.policy_artifact_id)
        if identity.policy_artifact_id != self.runner.policy_artifact_id:
            raise ValueError("Request policy does not match this immutable deployment")
        identity.fingerprint()
        request_id = request_id or uuid.uuid4().hex
        if (
            not isinstance(request_id, str)
            or not request_id
            or len(request_id) > 128
            or request_id in self._all
        ):
            raise ValueError("Invalid or duplicate active request id")
        now = time.monotonic()
        request = _Request(
            request_id,
            prompt,
            config,
            identity,
            now,
            now + timeout_s,
            asyncio.Queue(self.max_output_events + 1),
            asyncio.get_running_loop().create_future(),
            torch.Generator(device="cpu").manual_seed(config.seed),
        )
        if grammar is not None:
            from .structured import FiniteJSONGrammar

            if not isinstance(grammar, FiniteJSONGrammar):
                raise ValueError("Only a compiled native grammar can enter the scheduler")
        request.grammar = grammar
        if hasattr(self.runner, "prepare_request"):
            request.identity = self.runner.prepare_request(
                prompt, identity, modality_inputs, max_prefill_tokens=self.max_batch_tokens
            )
        elif modality_inputs is not None:
            raise ValueError("This runner has no declared multimodal request binding")
        self._pending[request_id] = self._all[request_id] = request
        self._wake.set()
        return RequestHandle(self, request)

    def cancel(self, request_id):
        request = self._all.get(request_id)
        if request is None:
            return False
        request.cancel_requested = True
        self._wake.set()
        return True

    def _finish(self, request, reason, error_code=None):
        if request.completion.done():
            return
        self._pending.pop(request.request_id, None)
        self._active.pop(request.request_id, None)
        self._all.pop(request.request_id, None)
        if request.state is not None:
            self.runner.pool.release(request.state)
        if request.archive_handle is not None:
            self.offload_archive.release(
                request.archive_handle, identity=request.identity.fingerprint()
            )
            request.archive_handle = None
        if hasattr(self.runner, "release_request"):
            self.runner.release_request(request.identity)
        result = GenerationResult(
            request.request_id,
            request.identity.policy_artifact_id,
            request.prompt,
            tuple(request.token_ids),
            tuple(request.raw_logp),
            tuple(request.behavior_logp),
            getattr(self.runner, "sampling_prefix_transforms", ()) + request.config.transform_order,
            request.text,
            reason,
            request.received_at,
            request.started_at,
            tuple(request.timestamps),
            time.monotonic(),
            request.prefix_hit_tokens,
            error_code,
            preemption_count=request.preemptions,
            adapter_id=request.identity.adapter,
        )
        request.completion.set_result(result)

        request.queue.put_nowait(None)
        self.completed_count += 1
        self.failed_count += int(reason not in {"length", "eos", "grammar_complete"})

    def _expired(self, request):
        if request.cancel_requested or self._closing:
            self._finish(request, "cancelled")
            return True
        if time.monotonic() >= request.deadline:
            self._finish(request, "timeout")
            return True
        return False

    async def _activate(self):
        if not self._active:
            self._admission_paused = False
        for request in list(self._pending.values()):
            if self._expired(request):
                continue
            if len(self._active) >= self.max_active or self._admission_paused:
                break
            if request.archive_handle is not None:
                try:
                    request.state = await self.offload_archive.restore_async(
                        request.archive_handle, identity=request.identity.fingerprint()
                    )
                except CacheCapacityError:
                    self.prefix_cache.clear()
                    if self._active:
                        self._admission_paused = True
                        break

                    try:
                        request.state = await self.offload_archive.restore_async(
                            request.archive_handle, identity=request.identity.fingerprint()
                        )
                    except CacheCapacityError:
                        self._finish(request, "error", "restore_capacity")
                        continue
                self.offload_archive.release(
                    request.archive_handle, identity=request.identity.fingerprint()
                )
                request.archive_handle = None
                if self._expired(request):
                    continue
            else:
                request.state = self.prefix_cache.lookup(request.identity, request.prompt)
                request.prefix_hit_tokens = max(request.prefix_hit_tokens, request.state.length)
            self._pending.pop(request.request_id)
            request.started_at = request.started_at or time.monotonic()
            self._active[request.request_id] = request

    def _plan(self):
        budget = self.max_batch_tokens
        groups = OrderedDict()

        requests = list(self._active.values())
        requests.sort(
            key=lambda request: (
                len(request.prompt) + len(request.token_ids) - request.state.length > 1
            )
        )
        for request in requests:
            if self._expired(request):
                continue
            if budget == 0:
                break
            if request.queue.qsize() >= self.max_output_events:
                self._finish(request, "backpressure", "slow_consumer")
                continue
            start = request.state.length
            context = request.prompt + tuple(request.token_ids)
            count = min(len(context) - start, self.prefill_chunk_size, budget)
            if hasattr(self.runner, "plan_chunk_length"):
                count = self.runner.plan_chunk_length(request.state, context, count)
                if count > budget:
                    continue
            if count < 1:
                raise RuntimeError("Scheduler state is ahead of committed token history")
            chunk = context[start : start + count]
            key = (request.identity.fingerprint(), start, count)
            groups.setdefault(key, []).append((request, chunk))
            budget -= count
            self._active.move_to_end(request.request_id)
        return list(groups.values())

    def _emit(self, request, logits):
        if request.state.length < len(request.prompt) + len(request.token_ids):
            return
        sampling_context = request.prompt + tuple(request.token_ids)
        if hasattr(self.runner, "sampling_context_ids"):
            sampling_context = self.runner.sampling_context_ids(sampling_context)
        sample = sample_token(
            logits,
            request.config,
            request.generator,
            context_ids=sampling_context,
            generated_count=len(request.token_ids),
            allowed_token_ids=request.grammar.allowed_tokens(request.token_ids)
            if request.grammar
            else None,
        )
        request.token_ids.append(sample.token_id)
        request.raw_logp.append(sample.raw_model_logprob)
        request.behavior_logp.append(sample.behavior_logprob)
        now = time.monotonic()
        request.timestamps.append(now)
        terminal = (
            sample.token_id in request.config.eos_token_ids
            or len(request.token_ids) >= request.config.max_new_tokens
        )
        decoded = self.runner.stream_text(request.token_ids, final=terminal)

        if not decoded.startswith(request.text):
            raise ValueError("Tokenizer decode is not prefix-stable; supply a streaming decoder")
        delta = decoded[len(request.text) :]
        request.text = decoded
        request.queue.put_nowait(
            TokenEvent(
                request.request_id,
                request.identity.policy_artifact_id,
                len(request.token_ids) - 1,
                sample.token_id,
                sample.raw_model_logprob,
                sample.behavior_logprob,
                delta,
                now,
            )
        )
        self.emitted_tokens += 1
        if sample.token_id in request.config.eos_token_ids:
            self._finish(request, "eos")
        elif request.grammar is not None and request.grammar.accepting(request.token_ids):
            self._finish(request, "grammar_complete")
        elif len(request.token_ids) >= request.config.max_new_tokens:
            self._finish(request, "length")

    async def _preempt(self, request):
        if self.offload_archive is not None and request.state.length:
            try:
                request.archive_handle = await self.offload_archive.put_async(request.state)
            except CacheCapacityError:
                self.offload_capacity_fallbacks += 1
        self.runner.pool.release(request.state)
        request.state = None
        request.preemptions += 1
        self._active.pop(request.request_id, None)
        self._pending[request.request_id] = request
        self._admission_paused = True

    async def _execute_group(self, group, *, capacity_retry=False):
        group = [
            pair
            for pair in group
            if not pair[0].completion.done()
            and self._active.get(pair[0].request_id) is pair[0]
            and pair[0].state is not None
        ]
        requests = [pair[0] for pair in group]
        chunks = [pair[1] for pair in group]
        if not requests:
            return
        before = [request.state.length for request in requests]
        try:
            work = await settle_thread(
                self.runner.forward_batch, [request.state for request in requests], chunks
            )
            if work.cancelled:
                raise asyncio.CancelledError from work.error
            logits = work.unwrap()
            for request, row in zip(requests, logits):
                if self._expired(request):
                    continue
                self.prefix_cache.publish(request.identity, request.prompt, request.state)
                try:
                    self._emit(request, row)
                except Exception:
                    self._finish(request, "error", "invalid_model_output")
        except CacheCapacityError:
            for request, length in zip(requests, before):
                self.runner.pool.truncate(request.state, length)
            self.prefix_cache.clear()
            if not capacity_retry:
                await self._execute_group(group, capacity_retry=True)
                return
            candidates = [
                request
                for request in self._active.values()
                if request not in requests and request.state.length > 0
            ]
            if candidates:
                victim = max(candidates, key=lambda request: request.state.length)
                await self._preempt(victim)
                await self._execute_group(group, capacity_retry=True)
            elif len(requests) > 1:
                for request in requests[1:]:
                    await self._preempt(request)
                await self._execute_group([group[0]], capacity_retry=True)
            else:
                self._finish(requests[0], "error", "cache_capacity")
        except asyncio.CancelledError:
            for request in requests:
                self._finish(request, "cancelled")
            raise
        except Exception:
            for request in requests:
                self._finish(request, "error", "model_execution_failed")

    async def _run(self):
        try:
            while True:
                await self._activate()
                if self._closing and not self._all:
                    return
                if not self._active:
                    self._wake.clear()
                    if not self._all and not self._closing:
                        await self._wake.wait()
                    continue
                for group in self._plan():
                    await self._execute_group(group)
                await asyncio.sleep(0)
        except BaseException:
            self._fatal = True
            for request in list(self._all.values()):
                self._finish(request, "error", "worker_stopped")
            raise
        finally:
            self.prefix_cache.clear()

    async def close(self):
        self._closing = True
        if self._wake is not None:
            self._wake.set()
        if self._worker is not None:
            await asyncio.shield(self._worker)
        else:
            self.prefix_cache.clear()

    def observation(self):
        elapsed = time.monotonic() - self.observation_started_at
        return {
            "window_seconds": elapsed,
            "emitted_tokens": self.emitted_tokens,
            "tokens_per_second": self.emitted_tokens / elapsed if elapsed else 0.0,
            "completed_requests": self.completed_count,
            "failed_requests": self.failed_count,
            "active_requests": self.active_count,
            "queued_requests": self.queued_count,
            "offload": None if self.offload_archive is None else self.offload_archive.metrics(),
            "offload_capacity_fallbacks": self.offload_capacity_fallbacks,
            "evidence_kind": "local_engine_observation_not_public_benchmark",
        }
