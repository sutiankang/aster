from dataclasses import asdict
import pytest
import torch
from aster.models import SigLIPConfig, SigLIPVisionConfig, SigLIPTextConfig, build_model
from aster.models.siglip import normalize_siglip_pixels


def _gradients(native, oracle, left, right):
    torch.testing.assert_close(left, right, atol=5e-6, rtol=4e-5)
    scale = torch.randn_like(left)
    (left * scale).sum().backward()
    (right * scale).sum().backward()
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=8e-5, rtol=7e-4, msg=name
        )


@pytest.mark.oracle
@pytest.mark.parametrize("use_head", [True, False])
def test_models_siglip_vision_official_weights_gradients(use_head):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(32)
    config = SigLIPVisionConfig(vision_use_head=use_head)
    reference_config = tf.SiglipVisionConfig(**asdict(config))
    reference_config._attn_implementation = "eager"
    native, oracle = build_model(config), tf.SiglipVisionModel(reference_config)
    oracle.load_state_dict(native.state_dict(), strict=True)
    pixels = torch.randn(2, 3, 16, 24)
    left, right = (
        native(pixels, interpolate_pos_encoding=True),
        oracle(pixels, interpolate_pos_encoding=True),
    )
    _gradients(
        native,
        oracle,
        left.pooler_output if use_head else left.last_hidden_state,
        right.pooler_output if use_head else right.last_hidden_state,
    )


@pytest.mark.oracle
def test_models_siglip_text_official_mask_and_last_padding_token():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(33)
    config = SigLIPTextConfig()
    reference_config = tf.SiglipTextConfig(**asdict(config))
    reference_config._attn_implementation = "eager"
    native, oracle = build_model(config), tf.SiglipTextModel(reference_config)
    oracle.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 3, 4, 0], [2, 4, 5, 7]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    _gradients(
        native,
        oracle,
        native(ids, attention_mask=mask).pooler_output,
        oracle(ids, attention_mask=mask).pooler_output,
    )


@pytest.mark.oracle
def test_models_siglip_full_contrastive_official_gradients():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(34)
    config = SigLIPConfig()
    reference_config = tf.SiglipConfig(
        text_config=asdict(config.text_config), vision_config=asdict(config.vision_config)
    )
    reference_config._attn_implementation = "eager"
    native, oracle = build_model(config), tf.SiglipModel(reference_config)
    oracle.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 4, 2, 0], [1, 7, 3, 0]])
    pixels = torch.randn(2, 3, 16, 16)
    _gradients(
        native, oracle, native(ids, pixels).logits_per_text, oracle(ids, pixels).logits_per_text
    )


@pytest.mark.oracle
def test_models_siglip_processor_normalization():
    tf = pytest.importorskip("transformers")
    pixels = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
    processor = tf.SiglipImageProcessorPil(do_resize=False)
    reference = processor(
        images=[x.permute(1, 2, 0).numpy() for x in pixels], return_tensors="pt"
    ).pixel_values
    torch.testing.assert_close(normalize_siglip_pixels(pixels), reference, atol=1e-7, rtol=1e-7)
