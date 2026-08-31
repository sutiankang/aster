"""Packed visual attention with explicitly declared image/video segment boundaries."""

import torch
import torch.nn.functional as F
from .attention import scaled_attention


def packed_vision_attention(
    query, key, value, angles, lengths, *, interleaved=False, implementation="reference"
):
    """Bidirectional attention on [N,H,D] segments with explicit image/video lengths."""
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 3:
        raise ValueError("Packed vision query/key/value must share [tokens,heads,head_dim]")
    n, heads, dim = query.shape
    if implementation not in {"reference", "sdpa"}:
        raise ValueError("Unknown packed attention implementation")
    if angles.shape != (n, dim // 2) or dim % 2 or sum(lengths) != n or any(x < 1 for x in lengths):
        raise ValueError("Invalid rotary shape or packed vision segment boundaries")
    phases = angles.repeat_interleave(2, -1) if interleaved else torch.cat((angles, angles), -1)
    cosine, sine = phases.cos()[:, None], phases.sin()[:, None]

    def rotate(tensor):
        if interleaved:
            pairs = torch.view_as_complex(tensor.float().reshape(*tensor.shape[:-1], -1, 2))
            frequency = torch.polar(torch.ones_like(angles), angles)[:, None]
            return torch.view_as_real(pairs * frequency).flatten(-2).to(tensor.dtype)
        a, b = tensor.float().chunk(2, -1)
        return (tensor.float() * cosine + torch.cat((-b, a), -1) * sine).to(tensor.dtype)

    query, key = rotate(query), rotate(key)
    outputs, offset = [], 0
    for length in lengths:
        part = slice(offset, offset + length)
        q, k, v = (x[part].transpose(0, 1)[None] for x in (query, key, value))
        mask = torch.ones(1, 1, length, length, dtype=torch.bool, device=query.device)

        result = (
            F.scaled_dot_product_attention(q, k, v)
            if implementation == "sdpa"
            else scaled_attention(q, k, v, mask)
        )
        outputs.append(result[0].transpose(0, 1).reshape(length, heads * dim))
        offset += length
    return torch.cat(outputs)
