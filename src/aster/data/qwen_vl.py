"""Native Qwen3-VL RGB preprocessing and dynamic visual-token packing."""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from ..core import atomic_json, read_json, digest_json


@dataclass(frozen=True)
class QwenMediaConfig:
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    image_min_pixels: int = 65536
    image_max_pixels: int = 16777216
    video_min_pixels: int = 131072
    video_max_pixels: int = 786432
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    rescale_factor: float = 1 / 255
    image_backend: str = "torch"
    video_backend: str = "torch"
    video_cap_pixels_per_frame: bool = False
    max_video_tokens: int = 768
    sample_fps: float = 2.0
    min_frames: int = 4
    max_frames: int = 768
    max_input_pixels: int = 64 * 1024 * 1024
    max_sequence_length: int = 32768

    def __post_init__(self):
        for name in ("image_mean", "image_std"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        dimensions = (
            self.patch_size,
            self.temporal_patch_size,
            self.merge_size,
            self.image_min_pixels,
            self.image_max_pixels,
            self.video_min_pixels,
            self.video_max_pixels,
            self.max_video_tokens,
            self.min_frames,
            self.max_frames,
            self.max_input_pixels,
            self.max_sequence_length,
        )
        if any(type(x) is not int or x < 1 for x in dimensions):
            raise ValueError("Qwen media sizes and budgets must be positive integers")
        if (
            self.image_min_pixels > self.image_max_pixels
            or self.video_min_pixels > self.video_max_pixels
            or self.min_frames > self.max_frames
        ):
            raise ValueError("Qwen media minimum exceeds maximum")
        if self.image_backend not in {"pil", "torch"} or self.video_backend != "torch":
            raise ValueError(
                "Qwen images support explicit PIL/Torch; audited video backend is Torch only"
            )
        if (
            len(self.image_mean) != 3
            or len(self.image_std) != 3
            or any(not math.isfinite(x) for x in self.image_mean)
            or any(
                not math.isfinite(x) or x <= 0
                for x in (*self.image_std, self.rescale_factor, self.sample_fps)
            )
        ):
            raise ValueError("Qwen normalization and frame rate must be finite and valid")
        if type(self.video_cap_pixels_per_frame) is not bool:
            raise ValueError("The version-dependent per-frame cap must be explicit bool")

    @property
    def factor(self):
        return self.patch_size * self.merge_size

    def to_dict(self):
        return dict(
            schema_version=1,
            type="qwen3_vl_media",
            source_revision="42ca97014c85d71a88ad60d55f08cb9fb4d26e2c",
            **asdict(self),
        )

    @classmethod
    def from_dict(cls, value):

        value = dict(value)
        declaration = {
            key: value.pop(key, None) for key in ("schema_version", "type", "source_revision")
        }
        if declaration != dict(
            schema_version=1,
            type="qwen3_vl_media",
            source_revision="42ca97014c85d71a88ad60d55f08cb9fb4d26e2c",
        ):
            raise ValueError("Unsupported Qwen media schema or source revision")
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Qwen media artifact requires every explicit preprocessing option")
        return cls(**value)

    @property
    def fingerprint(self):
        return digest_json(self.to_dict())


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    total_num_frames: int
    frame_indices: tuple[int, ...] | None = None

    def __post_init__(self):
        if (
            not isinstance(self.fps, (float, int))
            or isinstance(self.fps, bool)
            or not math.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("Video FPS must be supplied explicitly as a finite positive value")
        if type(self.total_num_frames) is not int or self.total_num_frames < 1:
            raise ValueError("Video total frame count must be positive")
        if self.frame_indices is not None:
            values = tuple(self.frame_indices)
            object.__setattr__(self, "frame_indices", values)
            if (
                not values
                or any(type(x) is not int or not 0 <= x < self.total_num_frames for x in values)
                or any(a > b for a, b in zip(values, values[1:]))
            ):
                raise ValueError(
                    "Video frame indices must be bounded, nondecreasing original-frame indices"
                )


@dataclass(frozen=True)
class RawImage:
    pixels: object


@dataclass(frozen=True)
class RawVideo:
    frames: torch.Tensor
    metadata: VideoMetadata
    do_sample_frames: bool = True
    sample_fps: float | None = None
    num_frames: int | None = None


def image_size(height, width, config):
    factor = config.factor
    if min(height, width) < 1 or max(height, width) / min(height, width) > 200:
        raise ValueError("Qwen image aspect ratio must not exceed 200")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > config.image_max_pixels:
        beta = math.sqrt(height * width / config.image_max_pixels)
        h, w = (
            max(factor, math.floor(height / beta / factor) * factor),
            max(factor, math.floor(width / beta / factor) * factor),
        )
    elif h * w < config.image_min_pixels:
        beta = math.sqrt(config.image_min_pixels / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w


def video_size(frames, height, width, config):
    c = config
    factor = c.factor
    if frames < c.temporal_patch_size:
        raise ValueError(
            "Qwen video must contain at least temporal_patch_size sampled frames; use an image for one frame"
        )
    if min(height, width) < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Qwen video aspect ratio must not exceed 200")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    t = round(frames / c.temporal_patch_size) * c.temporal_patch_size
    maximum = c.video_max_pixels
    if c.video_cap_pixels_per_frame:
        cap = max(
            min(c.max_video_tokens * factor * factor, maximum // frames),
            int(c.video_min_pixels * 1.05),
        )
        maximum = cap * frames
    if t * h * w > maximum:
        beta = math.sqrt(frames * height * width / maximum)
        h, w = (
            max(factor, math.floor(height / beta / factor) * factor),
            max(factor, math.floor(width / beta / factor) * factor),
        )
    elif t * h * w < c.video_min_pixels:
        beta = math.sqrt(c.video_min_pixels / (frames * height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w


def _image_tensor(image):
    if isinstance(image, torch.Tensor):
        return image
    if not isinstance(image, np.ndarray):
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise ValueError("Expected decoded RGB tensor/array/PIL image, never a path or URL")
        image = np.array(image.convert("RGB"), copy=True)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("NumPy RGB input must be uint8 HWC")
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).contiguous()


def _validate_pixels(frames, config):
    if (
        not isinstance(frames, torch.Tensor)
        or frames.ndim != 4
        or frames.shape[1] != 3
        or min(frames.shape) < 1
        or frames.dtype != torch.uint8
    ):
        raise ValueError("Raw RGB must be uint8 TCHW, not already-normalized floating pixels")
    if frames.numel() // 3 > config.max_input_pixels:
        raise ValueError("Raw media exceeds the explicit input pixel budget")


def _resize(frames, size, backend):
    if backend == "pil" and frames.device.type != "cpu":
        raise ValueError("PIL preprocessing is an explicit CPU backend")
    if frames.shape[-2:] == size:
        return frames
    if backend == "pil":
        from PIL import Image

        arrays = [
            np.array(
                Image.fromarray(frame.permute(1, 2, 0).numpy()).resize(
                    size[::-1], Image.Resampling.BICUBIC
                ),
                copy=True,
            )
            for frame in frames
        ]
        return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()

    if frames.device.type == "cpu":
        return F.interpolate(
            frames.contiguous(), size=size, mode="bicubic", align_corners=False, antialias=True
        )
    return (
        F.interpolate(
            frames.float(), size=size, mode="bicubic", align_corners=False, antialias=True
        )
        .clamp(0, 255)
        .round()
        .to(torch.uint8)
    )


def _normalize(frames, config, backend):
    mean = torch.tensor(config.image_mean, device=frames.device, dtype=torch.float32)[
        None, :, None, None
    ]
    std = torch.tensor(config.image_std, device=frames.device, dtype=torch.float32)[
        None, :, None, None
    ]
    if backend == "pil":
        return ((frames.double() * config.rescale_factor).float() - mean) / std
    return (frames.float() - mean * (1 / config.rescale_factor)) / (
        std * (1 / config.rescale_factor)
    )


@dataclass(frozen=True)
class PackedMedia:
    kind: str
    pixels: torch.Tensor
    grid: torch.Tensor
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    fps: float | None
    backend: str
    merge_size: int
    processor_id: str

    @property
    def placeholder_count(self):
        return int(self.grid.prod()) // self.merge_size**2

    @property
    def frame_placeholder_count(self):
        return int(self.grid[0, 1:].prod()) // self.merge_size**2

    def metadata(self):
        return dict(
            kind=self.kind,
            grid=self.grid.tolist(),
            original_size=self.original_size,
            resized_size=self.resized_size,
            frame_indices=self.frame_indices,
            timestamps=self.timestamps,
            fps=self.fps,
            backend=self.backend,
            processor_id=self.processor_id,
            pixels_sha256=hashlib.sha256(
                self.pixels.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        )


def _pack(frames, config):
    from types import SimpleNamespace
    from ..models.qwen_vl import pack_qwen_pixels

    c = SimpleNamespace(
        in_channels=3,
        patch_size=config.patch_size,
        spatial_merge_size=config.merge_size,
        temporal_patch_size=config.temporal_patch_size,
    )
    count = (
        math.ceil(len(frames) / config.temporal_patch_size)
        * (frames.shape[-2] // config.factor)
        * (frames.shape[-1] // config.factor)
    )
    if count > config.max_sequence_length:
        raise ValueError("Processed visual tokens exceed the explicit sequence budget")
    return pack_qwen_pixels(frames, c)


def prepare_image(image, config):
    frames = _image_tensor(image)[None]
    _validate_pixels(frames, config)
    size = image_size(*frames.shape[-2:], config)
    if math.prod(size) > config.max_input_pixels:
        raise ValueError("Resized image exceeds the pixel allocation budget")
    normalized = _normalize(
        _resize(frames, size, config.image_backend), config, config.image_backend
    )
    pixels, grid = _pack(normalized, config)
    return PackedMedia(
        "image",
        pixels,
        grid,
        tuple(frames.shape[-2:]),
        size,
        (0,),
        (),
        None,
        config.image_backend,
        config.merge_size,
        config.fingerprint,
    )


def prepare_video(video, config):
    if not isinstance(video, RawVideo) or not isinstance(video.metadata, VideoMetadata):
        raise ValueError("Video input requires explicit validated VideoMetadata")
    frames, meta, c = video.frames, video.metadata, config
    _validate_pixels(frames, c)
    if type(video.do_sample_frames) is not bool:
        raise ValueError("Frame sampling must be explicitly enabled or disabled")
    indices = meta.frame_indices
    if indices is None:
        if len(frames) != meta.total_num_frames:
            raise ValueError("Pre-sampled video frames need their original frame indices")
        indices = tuple(range(len(frames)))
    if len(indices) != len(frames):
        raise ValueError("Frame indices must align exactly with decoded frames")
    if video.sample_fps is not None and video.num_frames is not None:
        raise ValueError("sample_fps and num_frames are mutually exclusive")
    if video.do_sample_frames:
        if indices != tuple(range(meta.total_num_frames)):
            raise ValueError("Do not silently resample already-sampled frames")
        if video.num_frames is not None:
            count = video.num_frames
            if type(count) is not int or not 1 <= count <= c.max_frames:
                raise ValueError("Requested frame count exceeds the explicit frame budget")
        else:
            fps = c.sample_fps if video.sample_fps is None else video.sample_fps
            if (
                not isinstance(fps, (int, float))
                or isinstance(fps, bool)
                or not math.isfinite(fps)
                or fps <= 0
            ):
                raise ValueError("Sample FPS must be finite positive")
            count = min(
                max(int(meta.total_num_frames / meta.fps * fps), c.min_frames),
                c.max_frames,
                meta.total_num_frames,
            )
        selected = np.linspace(0, meta.total_num_frames - 1, count).round().astype(np.int64)
        indices = tuple(int(x) for x in selected)
        frames = frames[torch.from_numpy(selected).to(frames.device)]
    elif video.sample_fps is not None or video.num_frames is not None:
        raise ValueError("Sampling overrides conflict with do_sample_frames=False")
    if len(frames) > c.max_frames:
        raise ValueError("Pre-sampled clip exceeds the explicit frame budget")
    size = video_size(len(frames), *frames.shape[-2:], c)
    if len(frames) * math.prod(size) > c.max_input_pixels:
        raise ValueError("Resized clip exceeds the pixel allocation budget")
    normalized = _normalize(_resize(frames, size, c.video_backend), c, c.video_backend)
    pixels, grid = _pack(normalized, c)
    extended = indices + (indices[-1],) * ((-len(indices)) % c.temporal_patch_size)
    timestamps = tuple(
        (extended[i] / meta.fps + extended[i + c.temporal_patch_size - 1] / meta.fps) / 2
        for i in range(0, len(extended), c.temporal_patch_size)
    )
    return PackedMedia(
        "video",
        pixels,
        grid,
        tuple(video.frames.shape[-2:]),
        size,
        indices,
        timestamps,
        meta.fps,
        c.video_backend,
        c.merge_size,
        c.fingerprint,
    )


@dataclass(frozen=True)
class PreparedQwenBatch:
    model_inputs: dict
    labels: torch.Tensor
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    media: tuple[PackedMedia, ...]
    processor_id: str

    @property
    def media_fingerprint(self):
        return digest_json([item.metadata() for item in self.media])

    def training_batch(self):
        return dict(model_inputs=dict(self.model_inputs), labels=self.labels)


class Qwen3VLProcessor:
    def __init__(
        self, config, *, encode_text: Callable[[str], list[int]], tokenizer_id: str, pad_token_id=0
    ):
        if (
            not isinstance(config, QwenMediaConfig)
            or not callable(encode_text)
            or not isinstance(tokenizer_id, str)
            or not tokenizer_id
        ):
            raise ValueError(
                "Processor requires explicit media config, local tokenizer callable and immutable tokenizer ID"
            )
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("Padding token ID must be explicit")
        self.config, self.encode_text, self.tokenizer_id, self.pad_token_id = (
            config,
            encode_text,
            tokenizer_id,
            pad_token_id,
        )

    @property
    def fingerprint(self):
        return digest_json(
            dict(
                media=self.config.to_dict(),
                tokenizer_id=self.tokenizer_id,
                pad_token_id=self.pad_token_id,
                template="caller_explicit",
            )
        )

    def save_pretrained(self, directory):

        atomic_json(
            Path(directory) / "qwen_processor.json",
            dict(
                schema_version=1,
                media=self.config.to_dict(),
                tokenizer_id=self.tokenizer_id,
                pad_token_id=self.pad_token_id,
                fingerprint=self.fingerprint,
                template="caller_explicit",
            ),
        )

    @classmethod
    def from_pretrained(cls, directory, *, encode_text, tokenizer_id):
        value = read_json(Path(directory) / "qwen_processor.json")
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "schema_version",
                "media",
                "tokenizer_id",
                "pad_token_id",
                "fingerprint",
                "template",
            }
            or value["schema_version"] != 1
            or value["template"] != "caller_explicit"
        ):
            raise ValueError("Unsupported Qwen processor artifact")
        if value["tokenizer_id"] != tokenizer_id:
            raise ValueError("Local tokenizer identity differs from the saved processor")
        result = cls(
            QwenMediaConfig.from_dict(value["media"]),
            encode_text=encode_text,
            tokenizer_id=tokenizer_id,
            pad_token_id=value["pad_token_id"],
        )
        if result.fingerprint != value["fingerprint"]:
            raise ValueError("Qwen processor artifact fingerprint mismatch")
        return result

    def prepare(self, examples, model_config):
        from ..models.qwen_vl import Qwen3VLConfig, multimodal_positions
        from ..models.cosmos3_vlm import Cosmos3VLMConfig

        if not isinstance(model_config, (Qwen3VLConfig, Cosmos3VLMConfig)):
            raise ValueError(
                "Processor supports audited native Qwen3VL and Cosmos3 Qwen understanding"
            )
        c = model_config
        vision = c.vision_config
        text = c.mot if isinstance(c, Cosmos3VLMConfig) else c.text_config
        if (
            vision.patch_size,
            vision.temporal_patch_size,
            vision.spatial_merge_size,
            vision.in_channels,
        ) != (self.config.patch_size, self.config.temporal_patch_size, self.config.merge_size, 3):
            raise ValueError("Processor patch/temporal/merge contract differs from the model")
        if not isinstance(examples, (list, tuple)) or not examples:
            raise ValueError("Qwen media batch must be an explicit nonempty sequence of examples")
        rows, masks, all_media = [], [], []
        reserved = {
            c.image_token_id,
            c.video_token_id,
            c.vision_start_token_id,
            c.vision_end_token_id,
        }
        for pieces in examples:
            if not isinstance(pieces, (list, tuple)) or not pieces:
                raise ValueError("Each Qwen example must be an explicit nonempty content sequence")
            ids, supervised, pending = [], [], ""

            def append(values, valid):
                if any(type(x) is not int or not 0 <= x < text.vocab_size for x in values):
                    raise ValueError("Tokenizer IDs exceed the locked vocabulary")
                ids.extend(values)
                supervised.extend([valid] * len(values))

            def flush():
                nonlocal pending
                if pending:
                    values = list(self.encode_text(pending))
                    if reserved.intersection(values):
                        raise ValueError("Ordinary tokenizer text contains a reserved visual token")
                    append(values, True)
                    pending = ""

            for piece in pieces:
                if isinstance(piece, str):
                    pending += piece
                    continue
                if isinstance(piece, tuple):
                    flush()
                    if reserved.intersection(piece):
                        raise ValueError(
                            "Visual markers must be created from real media, not injected token tuples"
                        )
                    append(piece, False)
                    continue
                if isinstance(piece, RawImage):
                    media = prepare_image(piece.pixels, self.config)
                elif isinstance(piece, RawVideo):
                    media = prepare_video(piece, self.config)
                else:
                    raise ValueError(
                        "Unknown Qwen content piece; use explicit text/tokens/RawImage/RawVideo"
                    )
                all_media.append(media)
                if media.kind == "image":
                    flush()
                    append(
                        [c.vision_start_token_id]
                        + [c.image_token_id] * media.placeholder_count
                        + [c.vision_end_token_id],
                        False,
                    )
                else:
                    for timestamp in media.timestamps:
                        pending += f"<{timestamp:.1f} seconds>"
                        flush()
                        append(
                            [c.vision_start_token_id]
                            + [c.video_token_id] * media.frame_placeholder_count
                            + [c.vision_end_token_id],
                            False,
                        )
            flush()
            if not ids or len(ids) > self.config.max_sequence_length:
                raise ValueError(
                    "Qwen prepared sequence is empty or exceeds the explicit token budget"
                )
            rows.append(ids)
            masks.append(supervised)
        if not 0 <= self.pad_token_id < text.vocab_size or self.pad_token_id in reserved:
            raise ValueError("Padding ID conflicts with model vocabulary/visual markers")
        device = all_media[0].pixels.device if all_media else torch.device("cpu")
        if any(media.pixels.device != device for media in all_media):
            raise ValueError("Media in one batch must share an explicit device")
        length = max(map(len, rows))
        ids = torch.full((len(rows), length), self.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        labels = torch.full_like(ids, -100)
        for i, (row, targets) in enumerate(zip(rows, masks)):
            values = torch.tensor(row, dtype=torch.long, device=device)
            ids[i, : len(row)] = values
            mask[i, : len(row)] = True
            labels[i, : len(row)] = values.masked_fill(
                ~torch.tensor(targets, dtype=torch.bool, device=device), -100
            )
        inputs = dict(input_ids=ids, attention_mask=mask)
        for kind, pixel_key, grid_key in (
            ("image", "pixel_values", "image_grid_thw"),
            ("video", "pixel_values_videos", "video_grid_thw"),
        ):
            items = [x for x in all_media if x.kind == kind]
            if items:
                inputs[pixel_key] = torch.cat([x.pixels for x in items])
                inputs[grid_key] = torch.cat([x.grid for x in items])
        types = torch.where(ids == c.image_token_id, 1, torch.where(ids == c.video_token_id, 2, 0))
        positions, deltas = multimodal_positions(
            ids,
            types,
            self.config.merge_size,
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            mask,
        )
        if isinstance(c, Cosmos3VLMConfig):
            if "pixel_values" in inputs and "pixel_values_videos" in inputs:
                raise ValueError(
                    "Cosmos3 Qwen understanding accepts image or video per prefill, not both"
                )
        else:
            inputs["mm_token_type_ids"] = types
        return PreparedQwenBatch(
            inputs, labels, positions, deltas, tuple(all_media), self.fingerprint
        )
