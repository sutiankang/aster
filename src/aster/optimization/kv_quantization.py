"""Per-token/head low-bit KV codes and scales without a floating-point shadow history."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class KVQuantization:
    format: str = "int8"

    def __post_init__(self):
        if self.format not in {"int8", "fp8_e4m3fn", "fp8_e5m2"}:
            raise ValueError("KV format must be explicit INT8 / FP8 E4M3FN / FP8 E5M2")

    @property
    def dtype(self):
        return {
            "int8": torch.int8,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }[self.format]

    @property
    def maximum(self):
        return 127.0 if self.format == "int8" else torch.finfo(self.dtype).max


@dataclass(frozen=True)
class QuantizedKV:
    values: torch.Tensor
    scales: torch.Tensor
    source_dtype: torch.dtype

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def dtype(self):
        return self.source_dtype

    @property
    def device(self):
        return self.values.device

    @property
    def nbytes(self):
        return (
            self.values.numel() * self.values.element_size()
            + self.scales.numel() * self.scales.element_size()
        )

    def narrow(self, dim, start, count):
        if dim != self.ndim - 2:
            raise ValueError("Quantized KV token axis must immediately precede feature width")
        return QuantizedKV(
            self.values.narrow(dim, start, count),
            self.scales.narrow(dim, start, count),
            self.source_dtype,
        )

    def dequantize(self, *, dtype=None):

        return (self.values.to(self.scales.dtype) * self.scales).to(dtype or self.source_dtype)


@torch.no_grad()
def quantize_kv(value, config):
    if config is None or not value.is_floating_point():
        return value
    if not isinstance(config, KVQuantization) or value.ndim < 3 or value.shape[-1] < 1:
        raise ValueError("Quantized KV needs explicit layout [...,tokens,width] and format")
    if (
        value.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        or not torch.isfinite(value).all()
    ):
        raise ValueError("Quantized KV requires finite floating input")
    accumulation = torch.float64 if value.dtype == torch.float64 else torch.float32
    source = value.detach().to(accumulation)
    maximum = source.abs().amax(-1, keepdim=True)

    scales = torch.where(
        maximum == 0,
        torch.ones_like(maximum),
        (maximum / config.maximum).clamp_min(torch.finfo(accumulation).tiny),
    )
    normalized = (source / scales).clamp(-config.maximum, config.maximum)
    if config.format == "int8":
        normalized = normalized.round()
    return QuantizedKV(normalized.to(config.dtype), scales, value.dtype)


def allocate_kv_like(value, dim, count):
    shape = list(value.shape)
    shape[dim] = count
    if isinstance(value, QuantizedKV):
        if dim != value.ndim - 2:
            raise ValueError("Invalid quantized KV sequence axis")
        scale_shape = list(shape)
        scale_shape[-1] = 1
        return QuantizedKV(
            torch.empty(shape, dtype=value.values.dtype, device=value.device),
            torch.empty(scale_shape, dtype=value.scales.dtype, device=value.device),
            value.dtype,
        )
    return torch.empty(shape, dtype=value.dtype, device=value.device)


def copy_kv(target, source, dim, target_start, source_start, count):
    """Copy raw codes and scales for append/COW to avoid cumulative requantization drift."""
    if isinstance(target, QuantizedKV) != isinstance(source, QuantizedKV):
        raise ValueError("KV storage formats differ")
    if isinstance(target, QuantizedKV):
        target.values.narrow(dim, target_start, count).copy_(
            source.values.narrow(dim, source_start, count)
        )
        target.scales.narrow(dim, target_start, count).copy_(
            source.scales.narrow(dim, source_start, count)
        )
    else:
        target.narrow(dim, target_start, count).copy_(
            source.narrow(dim, source_start, count).detach()
        )


def kv_tile(value, start, count, dtype):
    tile = value.narrow(value.ndim - 2, start, count)
    return tile.dequantize(dtype=dtype) if isinstance(tile, QuantizedKV) else tile.to(dtype)


def finite_kv(value):
    if isinstance(value, QuantizedKV):
        return bool(
            torch.isfinite(value.values.float()).all()
            and torch.isfinite(value.scales).all()
            and (value.scales > 0).all()
        )
    return bool(torch.isfinite(value).all())


def clone_kv(value, *, device, pin_memory=False):
    """Copy codes/scales across memory tiers; the caller must await device completion
    before releasing the source lease."""

    def copy(tensor):
        target = torch.empty_like(tensor, device=device, pin_memory=pin_memory)
        target.copy_(tensor, non_blocking=True)
        return target

    if isinstance(value, QuantizedKV):
        return QuantizedKV(copy(value.values), copy(value.scales), value.dtype)
    return copy(value)
