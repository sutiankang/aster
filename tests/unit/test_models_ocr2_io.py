import pytest
import torch
from aster.models import OCR2Config, build_model
from aster.data.ocr import (
    prepare_document_views,
    prepare_ocr_inputs,
    parse_grounding,
    generate_document,
)


def test_models_ocr2_preprocessing_tile_order_count_and_mask():
    c = OCR2Config()

    image = torch.zeros(3, 24, 48, dtype=torch.uint8)
    image[:, :, 24:] = 255
    inputs, views = prepare_ocr_inputs(
        image, "<image>Read", lambda text: [3] * len(text), c, max_crops=2
    )
    assert views.crop_grid == (2, 1) and views.local_pixels.shape == (2, 3, 24, 24)
    assert views.local_pixels[0].mean() == -1 and views.local_pixels[1].mean() == 1
    assert (
        views.visual_tokens
        == c.vision_config.global_queries + 1 + 2 * c.vision_config.local_queries
    )
    assert inputs["images_seq_mask"].sum() == views.visual_tokens
    assert float(views.global_pixels[0, 0, 0, 0]) == pytest.approx(127 / 255 * 2 - 1)
    small = prepare_document_views(torch.full((3, 12, 12), 255, dtype=torch.uint8), c.vision_config)
    assert small.local_pixels is None and small.crop_grid == (1, 1)
    assert torch.equal(small.global_pixels, torch.ones_like(small.global_pixels))


def test_models_ocr2_safe_grounding_boxes_reject_code():
    text = "## Report\n<|ref|>table<|/ref|><|det|>[[100,200,800,900]]<|/det|>"
    regions = parse_grounding(text, (640, 480))
    assert regions[0].label == "table" and regions[0].pixel_box == (64.0, 96.0, 512.0, 432.0)
    for raw in (
        "__import__('os').system('echo bad')",
        "[[True,0,1,2]]",
        "[[5,0,1,2]]",
        "[[0,0,2000,2]]",
    ):
        with pytest.raises(ValueError):
            parse_grounding(f"<|ref|>bad<|/ref|><|det|>{raw}<|/det|>", (10, 10))


def test_models_ocr2_native_document_generation_uses_cache_and_public_sampler():
    torch.set_num_threads(1)
    torch.manual_seed(524)
    c = OCR2Config()
    model = build_model(c)
    inputs, views = prepare_ocr_inputs(
        torch.zeros(3, 12, 12, dtype=torch.uint8),
        "<image>read",
        lambda text: [3] if text else [],
        c,
    )
    lengths = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: lengths.append(
            kwargs.get("input_ids", args[0] if args else None).shape[1]
        ),
        with_kwargs=True,
    )
    result = generate_document(
        model,
        inputs,
        lambda ids: ",".join(map(str, ids)),
        image_size=views.original_size,
        max_new_tokens=3,
        eos_token_id=None,
    )
    hook.remove()
    assert len(result.token_ids) == 3 and lengths == [inputs["input_ids"].shape[1], 1, 1]
    assert c.image_token_id not in result.token_ids and model.training
    with pytest.raises(ValueError, match="EOS"):
        generate_document(
            model, inputs, lambda ids: "", image_size=views.original_size, eos_token_id=-1
        )
