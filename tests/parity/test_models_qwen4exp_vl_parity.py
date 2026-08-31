from dataclasses import asdict
import pytest
import torch
import torch.nn.functional as F
from aster.models import Qwen4ExpConfig, build_model
from aster.models.qwen_vl import pack_qwen_pixels
from aster.nn.delta import delta_public_parameter_name
from test_models_qwen4exp_parity import source_forward


@pytest.mark.oracle
def test_models_qwen4exp_full_image_source_and_installed_vision_gradient_parity():
    tf = pytest.importorskip("transformers")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    torch.set_num_threads(1)
    torch.manual_seed(431)
    c = Qwen4ExpConfig()
    native = build_model(c)
    config = asdict(c.vision_config)
    config.pop("deepstack_visual_indexes")
    vc = tf.Qwen3_5VisionConfig(**config)
    vc._attn_implementation = "eager"
    vision = Qwen3_5VisionModel(vc)
    vision.load_state_dict(native.vision_tower.state_dict(), strict=True)
    weights = {
        k: v.detach().clone().requires_grad_(v.is_floating_point())
        for k, v in native.language_model.state_dict().items()
    }
    ids = torch.tensor([[1, 26, 28, 28, 28, 28, 27, 3, 2]])
    pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 8), c.vision_config)
    left_pixels = pixels.clone().requires_grad_()
    right_pixels = pixels.clone().requires_grad_()
    actual = native(
        ids, pixel_values=left_pixels, image_grid_thw=grid, mm_token_type_ids=(ids == 28).long()
    )
    features = vision(right_pixels, grid_thw=grid).pooler_output
    embedded = F.embedding(ids, weights["model.embed_tokens.weight"]).masked_scatter(
        (ids == 28)[..., None], features
    )
    positions = torch.tensor(
        [
            [[0, 1, 2, 2, 2, 2, 4, 5, 6]],
            [[0, 1, 2, 2, 3, 3, 4, 5, 6]],
            [[0, 1, 2, 3, 2, 3, 4, 5, 6]],
        ]
    )
    expected, _ = source_forward(ids, positions, weights, c.text_config, inputs_embeds=embedded)
    torch.testing.assert_close(actual.logits, expected, atol=5e-6, rtol=6e-5)
    assert actual.auxiliary["rope_delta"].item() == -2
    factor = torch.randn_like(expected)
    (actual.logits * factor).sum().backward()
    (expected * factor).sum().backward()
    torch.testing.assert_close(left_pixels.grad, right_pixels.grad, atol=8e-5, rtol=1e-3)
    for name, parameter in native.language_model.named_parameters():
        other = weights[delta_public_parameter_name(name)]
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is None and other.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-4, rtol=3e-3, msg=name)
    for name, parameter in native.vision_tower.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(vision.named_parameters())[name].grad,
            atol=2e-4,
            rtol=3e-3,
            msg=name,
        )
    public = [native.official_weight_name(key) for key in native.state_dict()]
    assert len(public) == len(set(public))
    assert "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight" in public
    assert "model.visual.patch_embed.proj.weight" in public
