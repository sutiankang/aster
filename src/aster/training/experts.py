"""Variable-size all-to-all expert dispatch, top-k combination, and gradient return."""

from __future__ import annotations

import torch
from torch import nn
import torch.distributed as dist

from .parallel import Group


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, send_counts, receive_counts, group):
        ctx.send, ctx.receive, ctx.group = send_counts, receive_counts, group
        if group.size == 1:
            return value
        result = torch.empty(
            (sum(receive_counts), *value.shape[1:]), dtype=value.dtype, device=value.device
        )
        dist.all_to_all_single(
            result,
            value.contiguous(),
            output_split_sizes=receive_counts,
            input_split_sizes=send_counts,
            group=group.handle,
        )
        return result

    @staticmethod
    def backward(ctx, gradient):
        if ctx.group.size == 1:
            return gradient, None, None, None
        result = torch.empty(
            (sum(ctx.send), *gradient.shape[1:]), dtype=gradient.dtype, device=gradient.device
        )
        dist.all_to_all_single(
            result,
            gradient.contiguous(),
            output_split_sizes=ctx.send,
            input_split_sizes=ctx.receive,
            group=ctx.group.handle,
        )
        return result, None, None, None


class ExpertParallelMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        group: Group,
        *,
        top_k: int = 2,
    ):
        super().__init__()
        if type(num_experts) is not int or num_experts < 1 or num_experts % group.size:
            raise ValueError("num_experts 必须整除 EP")
        if type(top_k) is not int or not 1 <= top_k <= num_experts:
            raise ValueError("非法 top_k")
        self.group, self.num_experts, self.top_k = group, num_experts, top_k
        self.experts_per_rank = num_experts // group.size
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleDict(
            {
                str(group.rank * self.experts_per_rank + i): nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.GELU(),
                    nn.Linear(intermediate_size, hidden_size),
                )
                for i in range(self.experts_per_rank)
            }
        )
        local_group = Group((group.ranks[group.rank],), None, 0)
        for parameter in self.experts.parameters():
            parameter._aster_gradient_group = local_group

    def forward(self, inputs):
        original_shape = inputs.shape
        tokens = inputs.reshape(-1, original_shape[-1])
        probabilities = self.router(tokens).float().softmax(-1)
        weights, selected = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        destinations = selected.flatten() // self.experts_per_rank
        permutation = destinations.argsort(stable=True)
        counts = torch.bincount(destinations, minlength=self.group.size).to(torch.int64)
        receive = torch.empty_like(counts)
        if self.group.size > 1:
            dist.all_to_all_single(receive, counts, group=self.group.handle)
        else:
            receive.copy_(counts)
        send_counts, receive_counts = counts.tolist(), receive.tolist()
        repeated = (
            tokens[:, None, :].expand(-1, self.top_k, -1).reshape(-1, tokens.shape[-1])[permutation]
        )
        dispatched = _AllToAll.apply(repeated, send_counts, receive_counts, self.group)
        expert_ids = selected.flatten()[permutation]
        if self.group.size > 1:
            received_ids = torch.empty(
                sum(receive_counts), dtype=expert_ids.dtype, device=expert_ids.device
            )
            dist.all_to_all_single(
                received_ids,
                expert_ids.contiguous(),
                output_split_sizes=receive_counts,
                input_split_sizes=send_counts,
                group=self.group.handle,
            )
        else:
            received_ids = expert_ids

        outputs = torch.zeros_like(dispatched)
        for expert_id, expert in self.experts.items():
            positions = (received_ids == int(expert_id)).nonzero(as_tuple=True)[0]
            outputs = outputs.index_copy(0, positions, expert(dispatched[positions]))
        returned = _AllToAll.apply(outputs, receive_counts, send_counts, self.group)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(len(permutation), device=permutation.device)
        combined = returned[inverse].reshape(tokens.shape[0], self.top_k, tokens.shape[-1])
        return (combined * weights.to(combined.dtype).unsqueeze(-1)).sum(1).reshape(original_shape)
