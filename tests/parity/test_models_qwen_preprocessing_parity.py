from dataclasses import asdict
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from aster.data.qwen_vl import (
    QwenMediaConfig,
    RawImage,
    RawVideo,
    VideoMetadata,
    Qwen3VLProcessor,
    prepare_image,
    prepare_video,
    image_size,
    video_size,
)
from aster.data.tokenization import ByteTokenizer
from aster.models import Qwen3VLConfig, Qwen3VLTextConfig, build_model


def tiny_media(**kwargs):
    return QwenMediaConfig(
        patch_size=2,
        image_min_pixels=64,
        image_max_pixels=256,
        video_min_pixels=64,
        video_max_pixels=768,
        max_sequence_length=512,
        **kwargs,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(7, 11), (35, 23), (3, 5)])
def test_models_qwen_raw_image_actual_transformers_pil_pixels_geometry(shape):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(703)
    c = tiny_media(image_backend="pil", image_mean=(0.48, 0.45, 0.4), image_std=(0.27, 0.26, 0.28))
    oracle = tf.Qwen2VLImageProcessorPil(
        min_pixels=c.image_min_pixels,
        max_pixels=c.image_max_pixels,
        patch_size=c.patch_size,
        temporal_patch_size=c.temporal_patch_size,
        merge_size=c.merge_size,
        image_mean=c.image_mean,
        image_std=c.image_std,
    )
    raw = torch.randint(0, 256, (3, *shape), dtype=torch.uint8)
    expected = oracle(images=raw, input_data_format="channels_first", return_tensors="pt")
    native = prepare_image(raw, c)
    torch.testing.assert_close(native.pixels, expected["pixel_values"], atol=0, rtol=0)
    torch.testing.assert_close(native.grid, expected["image_grid_thw"], atol=0, rtol=0)
    assert native.placeholder_count == int(expected["image_grid_thw"].prod()) // c.merge_size**2


@pytest.mark.oracle
@pytest.mark.parametrize("cap", [False, True])
def test_models_qwen_raw_video_actual_sample_timestamp_patch_and_resize_formula(cap):
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessor as OfficialVideo,
        smart_resize,
    )
    from transformers.models.qwen3_vl.processing_qwen3_vl import (
        Qwen3VLProcessor as OfficialProcessor,
    )

    torch.set_num_threads(1)
    torch.manual_seed(704)
    c = tiny_media(video_cap_pixels_per_frame=cap, max_video_tokens=8)
    raw = torch.randint(0, 256, (9, 3, 19, 27), dtype=torch.uint8)
    metadata = VideoMetadata(12.0, 9)
    native = prepare_video(RawVideo(raw, metadata, num_frames=5), c)
    source_indices = OfficialVideo.sample_frames(
        SimpleNamespace(fps=c.sample_fps, min_frames=c.min_frames, max_frames=c.max_frames),
        SimpleNamespace(fps=12.0, total_num_frames=9),
        num_frames=5,
    )
    assert native.frame_indices == tuple(source_indices.tolist())
    timestamps = OfficialProcessor._calculate_timestamps(None, source_indices.copy(), 12.0, 2)
    assert native.timestamps == tuple(timestamps) and native.timestamps[-1] == 8 / 12
    maximum = c.video_max_pixels
    if cap:
        maximum = (
            max(min(c.max_video_tokens * c.factor**2, maximum // 5), int(c.video_min_pixels * 1.05))
            * 5
        )
    h, w = smart_resize(
        5, 19, 27, temporal_factor=2, factor=4, min_pixels=c.video_min_pixels, max_pixels=maximum
    )
    assert native.resized_size == (h, w)
    resized = F.interpolate(
        raw[torch.from_numpy(source_indices)],
        size=(h, w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(c.image_mean)[None, :, None, None] * (1 / c.rescale_factor)
    std = torch.tensor(c.image_std)[None, :, None, None] * (1 / c.rescale_factor)
    normalized = (resized.float() - mean) / std

    patches, t, gh, gw = OfficialVideo.patchify(None, normalized[None], 2, 2, 2)
    torch.testing.assert_close(native.pixels, patches[0], atol=0, rtol=0)
    assert native.grid.tolist() == [[t, gh, gw]] == [[3, h // 2, w // 2]]


@pytest.mark.oracle
@pytest.mark.parametrize("video", [False, True])
def test_models_qwen_raw_to_actual_transformers_positions_gradients_cached_generation(video):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(705)
    config = Qwen3VLConfig(
        text_config=Qwen3VLTextConfig(vocab_size=300),
        image_token_id=280,
        video_token_id=281,
        vision_start_token_id=282,
        vision_end_token_id=283,
    )
    tokenizer = ByteTokenizer()
    processor = Qwen3VLProcessor(
        tiny_media(image_backend="pil"),
        encode_text=lambda text: tokenizer.encode(text, add_special_tokens=False),
        tokenizer_id=tokenizer.fingerprint,
    )
    medium = (
        RawVideo(
            torch.randint(256, (5, 3, 7, 11), dtype=torch.uint8), VideoMetadata(10.0, 5), False
        )
        if video
        else RawImage(torch.randint(256, (3, 7, 11), dtype=torch.uint8))
    )
    prepared = processor.prepare([[(1,), "Inspect ", medium, " Answer"]], config)
    native = build_model(config).eval()
    values = asdict(config.text_config)
    for key in ("rope", "mrope_section", "layer_types", "sliding_window"):
        values.pop(key)
    values["rope_parameters"] = dict(
        rope_type="default",
        rope_theta=config.text_config.rope.theta,
        mrope_section=list(config.text_config.mrope_section),
    )
    tc = tf.Qwen3VLTextConfig(**values)
    vc = tf.Qwen3VLVisionConfig(**asdict(config.vision_config))
    oc = tf.Qwen3VLConfig(
        text_config=tc,
        vision_config=vc,
        image_token_id=280,
        video_token_id=281,
        vision_start_token_id=282,
        vision_end_token_id=283,
        tie_word_embeddings=False,
    )
    oc._attn_implementation = "eager"
    official = tf.Qwen3VLForConditionalGeneration(oc).eval()
    official.load_state_dict(native.state_dict(), strict=True)
    inputs = prepared.model_inputs
    positions, delta = official.model.get_rope_index(
        inputs["input_ids"],
        inputs["mm_token_type_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        attention_mask=inputs["attention_mask"],
    )
    torch.testing.assert_close(prepared.position_ids, positions, atol=0, rtol=0)
    torch.testing.assert_close(prepared.rope_deltas, delta, atol=0, rtol=0)
    left, right = native(**inputs, use_cache=True), official(**inputs, use_cache=True)
    torch.testing.assert_close(left.logits, right.logits, atol=3e-6, rtol=4e-5)
    coefficient = torch.randn_like(left.logits) / left.logits.numel()
    (left.logits * coefficient).sum().backward()
    (right.logits * coefficient).sum().backward()
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(official.named_parameters())[name].grad,
            atol=3e-6,
            rtol=6e-4,
            msg=name,
        )
    for _ in range(3):
        token = left.logits[:, -1:].argmax(-1)
        assert torch.equal(token, right.logits[:, -1:].argmax(-1))
        left = native(token, state=left.state, use_cache=True)
        right = official(token, past_key_values=right.past_key_values, use_cache=True)
        torch.testing.assert_close(left.logits, right.logits, atol=3e-6, rtol=4e-5)
