"""Synchronous TP-by-PP inference with sharded weights and stage-local KV state."""

from __future__ import annotations
import copy
from dataclasses import asdict
import json
import time
import uuid

import torch
from torch import nn
import torch.distributed as dist

from ..core import TokenOutput
from ..models.config import LlamaConfig, Qwen2Config, Qwen3Config
from ..models.decoder import CausalLM, DecoderLayer
from ..nn import KVState
from ..training.parallel import ColumnParallelLinear, RowParallelLinear
from .engine import GenerationResult, TokenEvent
from .runner import ModelRunner
from .sampling import SamplingConfig, sample_token


def _column(source, group, *, gather=False):
    target = ColumnParallelLinear(
        source.in_features,
        source.out_features,
        group,
        bias=source.bias is not None,
        gather_output=gather,
    ).to(source.weight)
    with torch.no_grad():
        target.weight.copy_(source.weight.chunk(group.size, 0)[group.rank])
        if source.bias is not None:
            target.bias.copy_(source.bias.chunk(group.size, 0)[group.rank])
    return target


def _row(source, group):
    target = RowParallelLinear(
        source.in_features,
        source.out_features,
        group,
        bias=source.bias is not None,
        input_is_parallel=True,
    ).to(source.weight)
    with torch.no_grad():
        target.weight.copy_(source.weight.chunk(group.size, 1)[group.rank])
        if source.bias is not None:
            target.bias.copy_(source.bias)
    return target


class ParallelCausalPredictor(nn.Module):
    def __init__(self, model, context):
        super().__init__()
        c = model.config
        if not isinstance(model, CausalLM) or type(c) not in {
            LlamaConfig,
            Qwen2Config,
            Qwen3Config,
        }:
            raise ValueError("Parallel inference currently requires native dense Llama/Qwen2/Qwen3")
        if (
            context.cp.size != 1
            or getattr(context.config, "gtp_remat", 1) != 1
            or any(c.window_for_layer(i) for i in range(c.num_hidden_layers))
        ):
            raise ValueError("Context/window parallel inference needs its own state implementation")
        if c.num_hidden_layers < context.pp.size or any(
            value % context.tp.size
            for value in (
                c.num_attention_heads,
                c.num_key_value_heads,
                c.intermediate_size,
                c.vocab_size,
            )
        ):
            raise ValueError(
                "TP dimensions must divide and every PP stage must own at least one layer"
            )
        if any(type(layer) is not DecoderLayer for layer in model.model.layers):
            raise ValueError("Unknown decoder layer cannot be automatically tensor-sharded")
        self.config, self.context = c, context
        self.stage_start = c.num_hidden_layers * context.pp.rank // context.pp.size
        self.stage_end = c.num_hidden_layers * (context.pp.rank + 1) // context.pp.size
        self.model_key = (
            model.model_key
            + f":tp{context.tp.rank}/{context.tp.size}:pp{context.pp.rank}/{context.pp.size}"
        )
        self.embed_tokens = (
            copy.deepcopy(model.model.embed_tokens) if context.pp.rank == 0 else None
        )
        self.layers = nn.ModuleList(
            copy.deepcopy(model.model.layers[self.stage_start : self.stage_end])
        )
        for layer in self.layers:
            a = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj"):
                setattr(a, name, _column(getattr(a, name), context.tp))
            a.o_proj = _row(a.o_proj, context.tp)
            a.num_heads //= context.tp.size
            a.num_kv_heads //= context.tp.size
            layer.mlp.gate_proj = _column(layer.mlp.gate_proj, context.tp)
            layer.mlp.up_proj = _column(layer.mlp.up_proj, context.tp)
            layer.mlp.down_proj = _row(layer.mlp.down_proj, context.tp)
        last = context.pp.rank == context.pp.size - 1
        self.norm = copy.deepcopy(model.model.norm) if last else None
        self.lm_head = _column(model.lm_head, context.tp, gather=True) if last else None

        self._aster_shared_runtime_handles = (context, context.tp, context.pp, context.tp_pp)
        self.eval().requires_grad_(False)

    def forward(
        self, input_ids, *, attention_mask=None, position_ids=None, state=None, use_cache=False
    ):
        if input_ids.ndim != 2 or not input_ids.numel() or input_ids.dtype != torch.long:
            raise ValueError("Parallel predictor expects nonempty int64 BT token IDs")
        b, t = input_ids.shape
        seen = 0 if state is None else state.seen_tokens
        if state is not None and (
            not isinstance(state, KVState)
            or state.kind != "dense_kv"
            or state.model_key != self.model_key
            or len(state.layers) != len(self.layers)
        ):
            raise ValueError("PP stage/cache identity mismatch")
        if seen + t > self.config.max_position_embeddings:
            raise ValueError("Position support exceeded")
        position_ids = (
            torch.arange(seen, seen + t, device=input_ids.device)[None].expand(b, -1)
            if position_ids is None
            else position_ids
        )
        if position_ids.shape != input_ids.shape or (position_ids < 0).any():
            raise ValueError("Invalid parallel positions")
        pp = self.context.pp
        parameter = next(self.parameters())
        if pp.rank == 0:
            hidden = self.embed_tokens(input_ids)
        else:
            hidden = torch.empty(
                b, t, self.config.hidden_size, device=parameter.device, dtype=parameter.dtype
            )
            dist.recv(hidden, src=pp.ranks[pp.rank - 1], group=pp.handle)
        layers = []
        for index, layer in enumerate(self.layers):
            hidden, present, _ = layer(
                hidden,
                position_ids,
                attention_mask,
                state.layers[index] if state is not None else None,
                seen,
                use_cache,
            )
            if use_cache:
                layers.append(present)
        if pp.rank < pp.size - 1:
            dist.send(hidden.contiguous(), dst=pp.ranks[pp.rank + 1], group=pp.handle)
            logits = torch.empty(
                b, t, self.config.vocab_size, device=parameter.device, dtype=parameter.dtype
            )
        else:
            logits = self.lm_head(self.norm(hidden))
        if pp.size > 1:
            dist.broadcast(logits, src=pp.ranks[-1], group=pp.handle)
        cache = KVState(tuple(layers), seen + t, self.model_key) if use_cache else None
        return TokenOutput(logits, cache)


def _json_broadcast(value, group, device, *, limit=1024 * 1024):

    encoded = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if group.rank == 0
        else b""
    )
    if group.size == 1:
        if len(encoded) > limit:
            raise ValueError("Collective request exceeds bound")
        return value
    count = torch.tensor([len(encoded)], dtype=torch.long, device=device)
    dist.broadcast(count, src=group.ranks[0], group=group.handle)
    size = int(count.item())
    if not 0 < size <= limit:
        raise ValueError("Collective request exceeds bound")
    payload = (
        torch.tensor(list(encoded), dtype=torch.uint8, device=device)
        if group.rank == 0
        else torch.empty(size, dtype=torch.uint8, device=device)
    )
    dist.broadcast(payload, src=group.ranks[0], group=group.handle)
    return json.loads(bytes(payload.cpu().tolist()).decode("utf-8"))


class CollectiveGenerator:
    def __init__(self, model, context, *, policy_artifact_id, tokenizer=None, **runner_options):
        if not isinstance(model, ParallelCausalPredictor) or model.context is not context:
            raise ValueError("Use the same explicitly constructed ParallelContext")
        self.context = context
        self.runner = ModelRunner(
            model, policy_artifact_id=policy_artifact_id, tokenizer=tokenizer, **runner_options
        )
        self.group = context.tp_pp
        self._busy = False

    def generate(self, prompt=None, config=None, *, request_id=None, cancelled=None, on_token=None):
        if self._busy:
            raise RuntimeError("CollectiveGenerator is not a concurrent scheduler")
        leader = self.group.rank == 0
        packet = None
        if leader:
            try:
                config = config or SamplingConfig()
                if (
                    not isinstance(config, SamplingConfig)
                    or not prompt
                    or any(type(i) is not int or i < 0 for i in prompt)
                ):
                    raise ValueError("Invalid prompt/config")
                if (
                    max(prompt) >= self.runner.model.config.vocab_size
                    or len(prompt) + config.max_new_tokens
                    > self.runner.model.config.max_position_embeddings
                ):
                    raise ValueError("Request exceeds vocabulary or position support")
                packet = {
                    "prompt": list(prompt),
                    "config": asdict(config),
                    "id": request_id or uuid.uuid4().hex,
                }
            except (TypeError, ValueError):
                packet = {"error": "Invalid collective request"}
        packet = _json_broadcast(packet, self.group, self.runner.device)
        if "error" in packet:
            raise ValueError(packet["error"])
        prompt, config, request_id = (
            packet["prompt"],
            SamplingConfig(**packet["config"]),
            packet["id"],
        )
        sequence = self.runner.pool.create(self.runner.policy_artifact_id)
        generator = torch.Generator().manual_seed(config.seed)
        tokens, raw, behavior, timestamps = [], [], [], []
        started = time.monotonic()
        reason = "length"
        chunk = prompt
        self._busy = True
        try:
            for index in range(config.max_new_tokens):
                signal = None
                if leader:
                    try:
                        signal = {"cancel": bool(cancelled and cancelled())}
                    except Exception:
                        signal = {"cancel": True}
                if _json_broadcast(signal, self.group, self.runner.device)["cancel"]:
                    reason = "cancelled"
                    break
                logits = self.runner.forward_batch([sequence], [chunk])[0]
                record = None
                if leader:
                    try:
                        record = asdict(
                            sample_token(
                                logits,
                                config,
                                generator=generator,
                                context_ids=prompt + tokens,
                                generated_count=index,
                            )
                        )
                    except (TypeError, ValueError, RuntimeError):
                        record = {"error": "Collective sampling failed"}
                record = _json_broadcast(record, self.group, self.runner.device)
                if "error" in record:
                    raise ValueError(record["error"])
                token = record["token_id"]
                tokens.append(token)
                raw.append(record["raw_model_logprob"])
                behavior.append(record["behavior_logprob"])
                timestamps.append(time.monotonic())
                if leader and on_token is not None:
                    try:
                        on_token(
                            TokenEvent(
                                request_id,
                                self.runner.policy_artifact_id,
                                index,
                                token,
                                raw[-1],
                                behavior[-1],
                                self.runner.decode(tokens),
                                timestamps[-1],
                            )
                        )
                    except Exception:
                        cancelled = lambda: True
                if token in config.eos_token_ids and index + 1 >= config.min_new_tokens:
                    reason = "eos"
                    break
                chunk = [token]
            return GenerationResult(
                request_id,
                self.runner.policy_artifact_id,
                tuple(prompt),
                tuple(tokens),
                tuple(raw),
                tuple(behavior),
                config.transform_order,
                self.runner.decode(tokens),
                reason,
                started,
                started,
                tuple(timestamps),
                time.monotonic(),
            )
        finally:
            self.runner.pool.release(sequence)
            self._busy = False
