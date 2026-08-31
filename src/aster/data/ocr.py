"""Explicit OCR tiling, placeholder alignment, and bounded grounding parsing."""

import ast
from dataclasses import dataclass
import math
import re
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DocumentViews:
    global_pixels: torch.Tensor
    local_pixels: torch.Tensor | None
    crop_grid: tuple[int, int]
    original_size: tuple[int, int]
    visual_tokens: int


def _resize(image, size):

    return (
        F.interpolate(
            image[None].float(), size=size, mode="bicubic", align_corners=False, antialias=True
        )[0]
        .round()
        .clamp(0, 255)
    )


def prepare_document_views(image, vision_config, *, crop=True, max_crops=6):
    if (
        image.ndim != 3
        or image.shape[0] != 3
        or image.dtype != torch.uint8
        or min(image.shape[1:]) < 1
    ):
        raise ValueError("Document image must be nonempty uint8 RGB CHW")
    if type(max_crops) is not int or not 2 <= max_crops <= 64:
        raise ValueError("max_crops must be a bounded integer >=2")
    c = vision_config
    height, width = image.shape[1:]
    global_side, local_side = c.sam_config.image_size, c.local_image_size

    ratio = global_side / max(height, width)
    resized_h, resized_w = max(1, round(height * ratio)), max(1, round(width * ratio))
    resized = _resize(image, (resized_h, resized_w))
    global_image = image.new_full((3, global_side, global_side), 127).float()
    top, left = round((global_side - resized_h) / 2), round((global_side - resized_w) / 2)
    global_image[:, top : top + resized_h, left : left + resized_w] = resized
    local, grid = None, (1, 1)
    if crop and max(height, width) > local_side:
        candidates = {
            (cols, rows)
            for count in range(2, max_crops + 1)
            for cols in range(1, count + 1)
            for rows in range(1, count + 1)
            if 2 <= cols * rows <= max_crops
        }
        candidates = sorted(candidates, key=lambda x: x[0] * x[1])
        closest, difference = (1, 1), float("inf")
        for cols, rows in candidates:
            error = abs(width / height - cols / rows)
            if (
                error < difference
                or error == difference
                and width * height > 0.5 * local_side**2 * cols * rows
            ):
                closest, difference = (cols, rows), error
        cols, rows = grid = closest
        tiled = _resize(image, (rows * local_side, cols * local_side))

        local = (
            tiled.reshape(3, rows, local_side, cols, local_side)
            .permute(1, 3, 0, 2, 4)
            .reshape(rows * cols, 3, local_side, local_side)
        )
        local = (local / 255 - 0.5) / 0.5
    global_image = ((global_image / 255 - 0.5) / 0.5)[None]
    tokens = c.global_queries + 1 + (0 if local is None else len(local) * c.local_queries)
    return DocumentViews(global_image, local, grid, (width, height), tokens)


def prepare_ocr_inputs(
    image, prompt, encode_text, config, *, bos_token_id=0, crop=True, max_crops=6
):

    if not isinstance(prompt, str) or prompt.count("<image>") != 1:
        raise ValueError("Document prompt must contain exactly one <image>")
    views = prepare_document_views(image, config.vision_config, crop=crop, max_crops=max_crops)
    before, after = prompt.split("<image>")
    prefix, suffix = list(encode_text(before)), list(encode_text(after))
    values = [bos_token_id] + prefix + [config.image_token_id] * views.visual_tokens + suffix
    if any(type(x) is not int or not 0 <= x < config.text_config.vocab_size for x in values):
        raise ValueError("Tokenizer IDs do not fit the locked OCR vocabulary")
    if config.image_token_id in prefix + suffix:
        raise ValueError("Tokenizer text unexpectedly contains reserved image placeholder ID")
    ids = torch.tensor([values], dtype=torch.long, device=image.device)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids, dtype=torch.bool),
        "pixel_values": views.global_pixels,
        "pixel_values_local": (views.local_pixels,),
        "images_spatial_crop": torch.tensor(
            [views.crop_grid], device=image.device, dtype=torch.long
        ),
        "images_seq_mask": ids.eq(config.image_token_id),
    }, views


@dataclass(frozen=True)
class GroundedRegion:
    label: str
    normalized_box: tuple[float, float, float, float]
    pixel_box: tuple[float, float, float, float]


def parse_grounding(text, image_size, *, max_regions=10000):

    if (
        not isinstance(text, str)
        or len(text) > 2_000_000
        or len(image_size) != 2
        or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in image_size)
        or type(max_regions) is not int
        or not 1 <= max_regions <= 10000
    ):
        raise ValueError("Invalid document text/image dimensions")
    result = []
    pattern = r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>"
    for match in re.finditer(pattern, text, flags=re.DOTALL):
        label, raw = match.groups()
        if len(label) > 1024 or len(raw) > 65536:
            raise ValueError("Oversized grounding record")
        try:
            boxes = ast.literal_eval(raw)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as error:
            raise ValueError("Grounding coordinates must be literal numeric boxes") from error
        if not isinstance(boxes, (list, tuple)):
            raise ValueError("Grounding coordinates must be a list")
        for box in boxes:
            if (
                not isinstance(box, (tuple, list))
                or len(box) != 4
                or any(
                    type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 1000
                    for x in box
                )
            ):
                raise ValueError("Grounding box must contain four finite coordinates in [0,1000]")
            x1, y1, x2, y2 = map(float, box)
            if x1 > x2 or y1 > y2:
                raise ValueError("Grounding box corners are reversed")
            width, height = image_size
            result.append(
                GroundedRegion(
                    label,
                    (x1, y1, x2, y2),
                    (x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000),
                )
            )
            if len(result) > max_regions:
                raise ValueError("Too many grounded regions")
    return tuple(result)


@dataclass(frozen=True)
class DocumentResult:
    text: str
    token_ids: tuple[int, ...]
    regions: tuple[GroundedRegion, ...]
    stopped_on_eos: bool


@torch.no_grad()
def generate_document(
    model, inputs, decode_tokens, *, image_size, max_new_tokens=256, eos_token_id=1
):

    from aster.inference.sampling import SamplingConfig, sample_token

    if type(max_new_tokens) is not int or max_new_tokens < 1 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("Document generation needs one row and a positive token budget")
    vocab_size = model.config.text_config.vocab_size
    if eos_token_id is not None and (
        type(eos_token_id) is not int
        or not 0 <= eos_token_id < vocab_size
        or eos_token_id == model.config.image_token_id
    ):
        raise ValueError("EOS must be a valid non-image token ID or None")
    allowed = tuple(x for x in range(vocab_size) if x != model.config.image_token_id)
    config = SamplingConfig(temperature=0.0)
    generator = torch.Generator(device="cpu").manual_seed(0)
    modes = {module: module.training for module in model.modules()}
    ids = inputs["input_ids"]
    padding = inputs.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
    produced, stopped = [], False
    try:
        model.eval()
        output = model(**inputs, use_cache=True)
        for _ in range(max_new_tokens):
            token = sample_token(
                output.logits[0, -1], config, generator, allowed_token_ids=allowed
            ).token_id
            if token == eos_token_id:
                stopped = True
                break
            produced.append(token)
            if len(produced) == max_new_tokens:
                break
            padding = torch.cat(
                (padding, torch.ones(1, 1, dtype=torch.bool, device=padding.device)), 1
            )
            output = model(
                torch.tensor([[token]], dtype=torch.long, device=ids.device),
                attention_mask=padding,
                state=output.state,
                use_cache=True,
            )
    finally:
        for module, mode in modes.items():
            module.training = mode
    text = decode_tokens(produced)
    return DocumentResult(text, tuple(produced), parse_grounding(text, image_size), stopped)
