"""Decoder execution directly over KV page tables without concatenating full history."""

from contextlib import ExitStack
import time

import torch

from aster.models.config import LlamaConfig, Qwen2Config, Qwen3Config
from aster.models.decoder import CausalLM, DecoderLayer
from aster.nn.attention import GroupedQueryAttention
from aster.optimization.online_attention import AttentionBlock, online_attention
from aster.optimization.fused_attention import (
    KernelWork,
    paged_fused_attention,
    assert_dense_attention_layout,
    _select_backend,
)
from .runner import ModelRunner
from .state import KVStateCodec, StateError
from aster.optimization.kv_quantization import KVQuantization, quantize_kv


class PagedAttentionRunner(ModelRunner):
    backend = "torch_online_paged"

    def __init__(
        self,
        model,
        *,
        policy_artifact_id,
        backend,
        tokenizer=None,
        block_size=16,
        max_blocks=256,
        query_block_size=32,
        key_block_size=None,
        chat_template=None,
        attention_fallback=None,
        kv_quantization=None,
    ):
        if backend not in {"torch_online_paged", "triton_fused_paged"}:
            raise ValueError("Select explicit backend='torch_online_paged' or 'triton_fused_paged'")
        if backend == "torch_online_paged" and attention_fallback is not None:
            raise ValueError("The Torch reference does not require a fallback")
        if kv_quantization is not None and (
            not isinstance(kv_quantization, KVQuantization) or backend != "torch_online_paged"
        ):
            raise ValueError(
                "Quantized paged KV currently requires the native tile-dequantization backend"
            )
        if key_block_size is None:
            key_block_size = 64 if backend == "torch_online_paged" else 32
        if type(model) is not CausalLM or type(model.config) not in {
            LlamaConfig,
            Qwen2Config,
            Qwen3Config,
        }:
            raise ValueError("Paged online runner supports native dense Llama/Qwen2/Qwen3 only")
        if (
            any(
                type(layer) is not DecoderLayer
                or type(layer.self_attn) is not GroupedQueryAttention
                for layer in model.model.layers
            )
            or model.config.attention_dropout != 0
        ):
            raise ValueError("Unexpected attention/layer semantics or nonzero dropout")
        if any(type(value) is not int or value < 1 for value in (query_block_size, key_block_size)):
            raise ValueError("Attention tile sizes must be positive integers")
        assert_dense_attention_layout(model)

        super().__init__(
            model,
            policy_artifact_id=policy_artifact_id,
            codec=KVStateCodec(kind="gqa_kv"),
            tokenizer=tokenizer,
            block_size=block_size,
            max_blocks=max_blocks,
            chat_template=chat_template,
            kv_quantization=kv_quantization,
        )
        self.query_block_size, self.key_block_size = query_block_size, key_block_size
        self.backend, self.attention_fallback = backend, attention_fallback
        self.attention_work = KernelWork()
        if backend == "triton_fused_paged":
            parameter = next(model.parameters())
            shape = (
                1,
                model.config.num_attention_heads,
                1,
                model.model.layers[0].self_attn.head_dim,
            )
            probe = parameter.new_empty(shape)
            actual, reason = _select_backend(
                "triton_fused", attention_fallback, probe, probe, query_block_size, key_block_size
            )
            self.attention_work.backend, self.attention_work.fallback_reason = actual, reason
        else:
            self.attention_work.backend = "torch_online_paged"

    @classmethod
    def from_artifact(cls, store, artifact_id, *, loader, tokenizer_loader=None, **kwargs):

        artifact = store.get(artifact_id, verify=True)
        tokenizer = tokenizer_loader(artifact.path) if tokenizer_loader else None
        return cls(
            loader(artifact.path), policy_artifact_id=artifact.id, tokenizer=tokenizer, **kwargs
        )

    def forward_batch(self, sequences, chunks, *, return_all_logits=False, padding_masks=None):

        if (
            not sequences
            or len(sequences) != len(chunks)
            or len({id(value) for value in sequences}) != len(sequences)
        ):
            raise ValueError("Need a nonempty batch of distinct sequences and aligned chunks")
        starts, counts = (
            {sequence.length for sequence in sequences},
            {len(chunk) for chunk in chunks},
        )
        if len(starts) != 1 or len(counts) != 1 or min(counts) < 1:
            raise ValueError("Scheduler must group matching cached/chunk lengths")
        if len({sequence.identity for sequence in sequences}) != 1:
            raise StateError("A projection batch cannot cross cache security domains")
        start, count, batch = next(iter(starts)), next(iter(counts)), len(sequences)
        if start + count > self.model.config.max_position_embeddings:
            raise ValueError("Sequence exceeds declared position support")
        if any(
            type(token) is not int or not 0 <= token < self.model.config.vocab_size
            for chunk in chunks
            for token in chunk
        ):
            raise ValueError("Input token outside this immutable model vocabulary")
        ids = torch.tensor(chunks, device=self.device, dtype=torch.long)
        positions = torch.arange(start, start + count, device=self.device).expand(batch, -1)
        if padding_masks is None:
            padding = torch.ones((batch, count), dtype=torch.bool, device=self.device)
        else:
            padding = torch.as_tensor(padding_masks, device=self.device)
            if padding.shape != ids.shape or not ((padding == 0) | (padding == 1)).all():
                raise ValueError("Chunk padding must be a binary batch-by-token matrix")
            padding = padding.bool()
        with ExitStack() as stack, torch.inference_mode():
            pages = [stack.enter_context(self.pool.read_pages(sequence)) for sequence in sequences]

            stack.callback(self._synchronize)
            self._synchronize()
            started = time.monotonic()
            hidden = self.model.model.embed_tokens(ids)
            new_layers = []
            for index, layer in enumerate(self.model.model.layers):
                normalized = layer.input_layernorm(hidden)
                attention = layer.self_attn

                def split(value, heads):
                    return value.reshape(batch, count, heads, attention.head_dim).transpose(1, 2)

                query = attention.rope(
                    attention.q_norm(split(attention.q_proj(normalized), attention.num_heads)),
                    positions,
                )
                key = attention.rope(
                    attention.k_norm(split(attention.k_proj(normalized), attention.num_kv_heads)),
                    positions,
                )
                value = split(attention.v_proj(normalized), attention.num_kv_heads)
                attended = []
                for row in range(batch):
                    old_blocks = []
                    for page in pages[row]:
                        if len(page.payload) != 2 * len(self.model.model.layers) + 1:
                            raise StateError(
                                "Paged decoder cache layout differs from its layer/padding plan"
                            )
                        old_blocks.append(
                            AttentionBlock(
                                page.payload[2 * index],
                                page.payload[2 * index + 1],
                                page.offset,
                                page.payload[-1][:, 0, :, 0],
                            )
                        )

                    blocks = old_blocks + [
                        AttentionBlock(
                            quantize_kv(key[row : row + 1], self.pool.quantization),
                            quantize_kv(value[row : row + 1], self.pool.quantization),
                            start,
                            padding[row : row + 1],
                        )
                    ]
                    kernel = (
                        online_attention
                        if self.backend == "torch_online_paged"
                        else paged_fused_attention
                    )
                    options = (
                        {}
                        if self.backend == "torch_online_paged"
                        else {"backend": "triton_fused", "fallback": self.attention_fallback}
                    )
                    attended.append(
                        kernel(
                            query[row : row + 1],
                            blocks,
                            query_positions=positions[row : row + 1],
                            window=attention.window,
                            scale=attention.scale,
                            query_block_size=self.query_block_size,
                            key_block_size=self.key_block_size,
                            work=self.attention_work,
                            **options,
                        )
                    )
                result = torch.cat(attended, 0).transpose(1, 2).reshape(batch, count, -1)
                hidden = hidden + attention.o_proj(result)
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
                new_layers.append((key, value))
            logits = self.model.lm_head(self.model.model.norm(hidden))
            self._synchronize()
            self.model_execution_seconds += time.monotonic() - started
            self.forward_calls += 1
            self.input_tokens_computed += ids.numel()

        for row, sequence in enumerate(sequences):
            suffix = tuple((key[row : row + 1], value[row : row + 1]) for key, value in new_layers)
            suffix += ((padding[row : row + 1, None, :, None],),)
            self.pool.append_delta(sequence, suffix)
        return [(row if return_all_logits else row[-1]).detach().float().cpu() for row in logits]
