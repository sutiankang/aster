"""Warm new immutable model versions before atomically routing new requests."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass
import time

from .runner import ModelRunner
from .engine import InferenceEngine
from .sampling import SamplingConfig


@dataclass(frozen=True)
class DeploymentRecord:
    artifact_id: str
    load_seconds: float
    warmup_seconds: float
    warmup_input_tokens: int
    activated_at: float
    previous_artifact_id: str | None


class DeploymentRouter:
    def __init__(
        self,
        store,
        *,
        loader,
        tokenizer_loader=None,
        chat_template_loader=None,
        max_versions=2,
        runner_options=None,
        engine_options=None,
    ):
        if max_versions < 2:
            raise ValueError("Atomic rollout/rollback requires at least two version slots")
        self.store, self.loader = store, loader
        self.tokenizer_loader, self.chat_template_loader = tokenizer_loader, chat_template_loader
        self.max_versions = max_versions
        self.runner_options, self.engine_options = runner_options or {}, engine_options or {}
        self._versions, self._requests = {}, {}
        self._current = None
        self._deployment_lock = asyncio.Lock()
        self._closed = False
        self.records = []

    @property
    def runner(self):
        if self._current is None:
            raise RuntimeError("No deployed artifact")
        return self._versions[self._current].runner

    @property
    def ready(self):
        return (
            not self._closed and self._current is not None and self._versions[self._current].ready
        )

    async def start(self):
        if not self.ready:
            raise RuntimeError("Router has no ready deployment")

    async def deploy(self, artifact_id, *, warmup_prompt_ids):
        async with self._deployment_lock:
            if self._closed:
                raise RuntimeError("Deployment router is closed")
            if artifact_id in self._versions:
                self.store.get(artifact_id, verify=True)
                self._current = artifact_id
                return self.records[-1] if self.records else None
            if len(self._versions) >= self.max_versions:
                evictable = [
                    key
                    for key, engine in self._versions.items()
                    if key != self._current and not engine.active_count and not engine.queued_count
                ]
                if not evictable:
                    raise RuntimeError("Deployment version capacity is occupied by live requests")
                removed = evictable[0]
                await self._versions[removed].close()
                del self._versions[removed]
            before = time.monotonic()
            artifact = self.store.get(artifact_id, verify=True)

            def load():
                tokenizer = self.tokenizer_loader(artifact.path) if self.tokenizer_loader else None
                template = (
                    self.chat_template_loader(artifact.path) if self.chat_template_loader else None
                )
                return ModelRunner(
                    self.loader(artifact.path),
                    policy_artifact_id=artifact.id,
                    tokenizer=tokenizer,
                    chat_template=template,
                    **self.runner_options,
                )

            runner = await asyncio.to_thread(load)
            loaded = time.monotonic()
            engine = InferenceEngine(runner, **self.engine_options)
            try:
                handle = await engine.submit(
                    warmup_prompt_ids, SamplingConfig(max_new_tokens=1, temperature=0)
                )
                result = await handle.collect()
                if result.stop_reason != "length":
                    raise RuntimeError("Deployment warmup failed")
            except BaseException:
                await engine.close()
                raise
            warmed = time.monotonic()
            record = DeploymentRecord(
                artifact.id,
                loaded - before,
                warmed - loaded,
                runner.input_tokens_computed,
                warmed,
                self._current,
            )

            engine.prefix_cache.clear()
            engine.completed_count = engine.failed_count = engine.emitted_tokens = 0
            engine.observation_started_at = time.monotonic()
            self._versions[artifact.id] = engine
            self._current = artifact.id
            self.records.append(record)
            return record

    async def rollback(self, artifact_id):
        async with self._deployment_lock:
            if artifact_id not in self._versions or not self._versions[artifact_id].ready:
                raise ValueError("Rollback target is not a retained ready snapshot")
            self.store.get(artifact_id, verify=True)
            self._current = artifact_id

    async def submit(self, prompt_ids, config=None, **options):
        await self.start()
        identity = options.get("identity")
        artifact_id = identity.policy_artifact_id if identity is not None else self._current
        if artifact_id not in self._versions:
            raise ValueError("Requested deployment is not retained")
        requested = options.get("request_id")
        if requested is not None and requested in self._requests:
            raise ValueError("Request id is already active in another model version")
        engine = self._versions[artifact_id]
        handle = await engine.submit(prompt_ids, config, **options)
        self._requests[handle.request_id] = engine
        handle._request.completion.add_done_callback(
            lambda _: self._requests.pop(handle.request_id, None)
        )
        return handle

    def cancel(self, request_id):
        engine = self._requests.get(request_id)
        return engine.cancel(request_id) if engine else False

    def observation(self):
        return {
            "active_artifact_id": self._current,
            "versions": {key: engine.observation() for key, engine in self._versions.items()},
            "warmup_excluded": True,
        }

    async def close(self):
        self._closed = True
        await asyncio.gather(*(engine.close() for engine in self._versions.values()))
