import pytest
import torch
from aster.models import OCR2SAMConfig, OCR2VisualConfig, build_model, load_model
from aster.models.ocr2_vision import visual_causal_mask
from aster.core import LossTerm
from aster.training import Trainer


def test_models_ocr2_visual_real_query_visibility_and_export(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(501)
    c = OCR2VisualConfig()
    model = build_model(c).eval()
    for side, count in (
        (c.local_image_size, c.local_queries),
        (c.sam_config.image_size, c.global_queries),
    ):
        images = torch.randn(2, 3, side, side)
        output = model(images)
        assert output.shape == (2, count, c.decoder_config.hidden_size)
        assert torch.isfinite(output).all()
    mask = visual_causal_mask(3)[0, 0]
    assert mask[:3, :3].all() and not mask[:3, 3:].any()
    assert mask[3:, :3].all() and torch.equal(
        mask[3:, 3:], torch.ones(3, 3, dtype=torch.bool).tril()
    )
    features = torch.randn(1, c.sam_config.output_channels, 3, 3)
    original = model.qwen2_model(features).detach()
    with torch.no_grad():
        model.qwen2_model.query_768.weight[-1].add_(10 * torch.randn(c.decoder_config.hidden_size))
    changed = model.qwen2_model(features).detach()
    torch.testing.assert_close(original[:, :-1], changed[:, :-1], atol=0, rtol=0)
    assert not torch.equal(original[:, -1], changed[:, -1])
    model.save_pretrained(tmp_path / "model")
    torch.testing.assert_close(
        load_model(tmp_path / "model").eval()(images), model(images), atol=0, rtol=0
    )
    assert "sam_model.pos_embed" in model.state_dict()
    assert "sam_model.blocks.0.attn.rel_pos_h" in model.state_dict()


class VisualRegression(torch.nn.Module):
    def config_dict(self):
        return {"type": "ocr_visual_regression_test"}

    def forward(self, model, batch):
        errors = (model(batch["pixels"]) - batch["target"]).square().mean((1, 2))
        return LossTerm(errors.sum(), errors.new_tensor(len(errors), dtype=torch.int64), "sample")


@pytest.mark.parametrize("stage", [0, 3])
def test_models_ocr2_visual_train_and_parameter_codec_resume(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(502)
    c = OCR2VisualConfig()
    model = build_model(c)
    batch = {
        "pixels": torch.randn(1, 3, c.local_image_size, c.local_image_size),
        "target": torch.randn(1, c.local_queries, c.decoder_config.hidden_size),
    }
    trainer = Trainer(model, VisualRegression(), zero_stage=stage, lr=0.003)
    initial = trainer.step([batch]).loss
    for _ in range(14):
        final = trainer.step([batch]).loss
    assert final < initial * 0.8
    trainer.save_checkpoint(tmp_path / "checkpoint")
    expected = trainer.step([batch])
    weights = trainer.export_state_dict()
    trainer.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    actual = trainer.step([batch])
    assert expected.loss == actual.loss
    for name, x in trainer.export_state_dict().items():
        torch.testing.assert_close(x, weights[name], atol=0, rtol=0)


def test_models_ocr2_sam_local_interpolation_relative_window_and_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(503)
    c = OCR2SAMConfig()
    model = build_model(c)

    with torch.no_grad():
        model.position.pos_embed.normal_(0, 0.05)
        for block in model.blocks:
            block.attn.relative.rel_pos_h.normal_(0, 0.05)
            block.attn.relative.rel_pos_w.normal_(0, 0.05)
    images = torch.randn(1, 3, 24, 24, requires_grad=True)
    output = model(images)
    assert output.shape == (1, c.output_channels, 3, 3)
    output.square().mean().backward()
    assert images.grad.abs().sum() > 0 and model.position.pos_embed.grad.abs().sum() > 0
    assert model.blocks[1].attn.relative.rel_pos_h.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="square"):
        model(torch.randn(1, 3, 24, 16))
