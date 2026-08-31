from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from aster.models import SigLIPConfig, SigLIPTextConfig, SigLIPVisionConfig, build_model, load_model
from aster.models.siglip import normalize_siglip_pixels


def test_models_siglip_real_pair_training_and_storage(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    model = build_model(SigLIPConfig())
    ids = torch.tensor([[1, 4, 7, 0], [1, 8, 2, 0]])
    pixels = torch.randn(2, 3, 16, 16)
    sign = 2 * torch.eye(2) - 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def objective():
        return -F.logsigmoid(sign * model(ids, pixels).logits_per_text).sum() / 2

    before = objective().item()
    for _ in range(8):
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        optimizer.step()
    assert objective().item() < before
    model.save_pretrained(tmp_path / "siglip")
    restored = load_model(tmp_path / "siglip")
    torch.testing.assert_close(
        model(ids, pixels).logits_per_image, restored(ids, pixels).logits_per_image
    )
    assert model.vision_model(pixels).last_hidden_state.shape == (2, 16, 32)


def test_models_siglip_patch_only_text_pooling_and_validation(tmp_path):
    vision = build_model(SigLIPVisionConfig(vision_use_head=False))
    out = vision(torch.randn(1, 3, 16, 24), interpolate_pos_encoding=True)
    assert out.last_hidden_state.shape == (1, 24, 32) and out.pooler_output is None
    vision.save_pretrained(tmp_path / "patches")
    assert load_model(tmp_path / "patches").config.vision_use_head is False
    text = build_model(SigLIPTextConfig())
    ids = torch.tensor([[1, 3, 4, 0]])
    result = text(ids, attention_mask=torch.tensor([[1, 1, 1, 0]]))
    torch.testing.assert_close(result.pooler_output, text.head(result.last_hidden_state[:, -1]))
    with pytest.raises(ValueError):
        build_model(SigLIPConfig(vision_config=SigLIPVisionConfig(vision_use_head=False)))
    with pytest.raises(ValueError):
        vision(torch.zeros(1, 3, 17, 16))
    with pytest.raises(ValueError):
        normalize_siglip_pixels(torch.ones(1, 3, 4, 4) * 2)
