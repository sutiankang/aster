"""Independent weight-rematerialization sharding axis with explicit supported combinations."""

from __future__ import annotations

import torch.distributed as dist
from .sharding import shard_module, zero3_units


def rematerialize_weights(
    model, context, *, initializer=None, device="cpu", offload_parameters=False
):
    if context.gtp_remat.size == 1:
        return model
    if context.pp.size != 1 or context.cp.size != 1:
        raise ValueError("当前 GTP reference 验收域为 dense TP×GTP×DP，PP/CP/EGTP 未开放")
    for parameter in model.parameters():
        domain = getattr(parameter, "_aster_gradient_group", context.dp_cp_gtp)
        if domain.ranks != context.dp_cp_gtp.ranks:
            raise ValueError("EGTP 需要独立专家网格，不能复用 dense GTP 域")
        if not parameter.is_meta and context.dp_cp_gtp.size > 1:
            dist.broadcast(
                parameter.data, src=context.dp_cp_gtp.ranks[0], group=context.dp_cp_gtp.handle
            )
    model = shard_module(
        model,
        context.gtp_remat,
        initializer=initializer,
        device=device,
        offload_parameters=offload_parameters,
    )
    for unit in zero3_units(model):
        unit._aster_gtp = True
        for shard in unit.shards:
            shard._aster_gradient_group = context.dp_cp
            shard._aster_weight_remat_group = context.gtp_remat
    return model
