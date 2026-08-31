"""Gradient buckets with phase-owned asynchronous data-parallel reduction."""

from __future__ import annotations

import torch
import torch.distributed as dist


class GradientBucketReducer:
    """Submit ready microbatch gradients so subsequent computation may overlap reduction."""

    def __init__(self, parameters, default_group, *, bucket_bytes):
        if type(bucket_bytes) is not int or bucket_bytes < 1:
            raise ValueError("bucket_bytes 必须为正整数")
        self.parameters, self.default_group, self.bucket_bytes = (
            parameters,
            default_group,
            bucket_bytes,
        )
        self.pending = []
        self.launched_buckets = 0

    def submit(self, term, gradients):
        bucket = []
        size = 0
        domain = None

        def launch():
            if not bucket:
                return
            flat = torch.cat([gradient.reshape(-1) for _, gradient, _ in bucket])
            group = bucket[0][2]
            work = (
                dist.all_reduce(flat, group=group.handle, async_op=True) if group.size > 1 else None
            )
            self.pending.append(
                (
                    term,
                    [(index, gradient.shape, gradient.numel()) for index, gradient, _ in bucket],
                    flat,
                    work,
                )
            )
            self.launched_buckets += 1

        for index, (parameter, gradient) in enumerate(zip(self.parameters, gradients)):
            if gradient is None:
                continue
            group = getattr(parameter, "_aster_gradient_group", self.default_group)
            key = (group.ranks, gradient.device, gradient.dtype)
            nbytes = gradient.numel() * gradient.element_size()
            if bucket and (domain != key or size + nbytes > self.bucket_bytes):
                launch()
                bucket = []
                size = 0
            bucket.append((index, gradient, group))
            domain = key
            size += nbytes
        launch()

    def finish(self, buffers):
        for term, descriptors, flat, work in self.pending:
            if work is not None:
                work.wait()
            offset = 0
            for index, shape, count in descriptors:
                value = flat.narrow(0, offset, count).reshape(shape)
                if buffers[term][index] is None:
                    buffers[term][index] = value.clone()
                else:
                    buffers[term][index].add_(value)
                offset += count
        self.pending.clear()
