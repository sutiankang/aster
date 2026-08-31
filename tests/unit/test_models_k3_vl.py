import pytest
import torch
from aster.models import KimiK3Config, build_model, load_model
from aster.models.kimi import pack_kimi_patches
from aster.methods import CrossEntropyObjective
from aster.training import Trainer


def _batch(config):
    pixels, grid = pack_kimi_patches(torch.randn(1, 3, 8, 8), config.vision_config)
    ids = torch.tensor([[1, 31, 31, 31, 31, 3, 7, 2]])
    labels = ids.clone()
    labels[:, :5] = -100
    return {
        "model_inputs": {"input_ids": ids, "pixel_values": pixels, "grid_thw": grid},
        "labels": labels,
    }


@pytest.mark.parametrize("stage", [0, 3])
def test_models_k3_vlm_joint_training_cache_and_export(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(322)
    config = KimiK3Config()
    model = build_model(config)
    batch = _batch(config)
    trainer = Trainer(model, CrossEntropyObjective(), lr=0.003, zero_stage=stage)
    initial = trainer.step([batch]).loss
    for _ in range(14):
        final = trainer.step([batch]).loss
    assert final < initial * 0.55
    trainer.save_checkpoint(tmp_path / "resume")
    expected = trainer.step([batch])
    weights = trainer.export_state_dict()
    trainer.load_checkpoint(tmp_path / "resume", trusted=True)
    actual = trainer.step([batch])
    assert expected.loss == actual.loss
    for key, value in trainer.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    native = build_model(config)
    native.load_state_dict(weights)
    native.eval()
    inputs = batch["model_inputs"]
    ids = inputs["input_ids"]
    all_logits = native(**inputs).logits
    first = native(
        ids[:, :5], pixel_values=inputs["pixel_values"], grid_thw=inputs["grid_thw"], use_cache=True
    )
    tail = native(ids[:, 5:], state=first.state, use_cache=True)
    torch.testing.assert_close(tail.logits, all_logits[:, 5:], atol=3e-6, rtol=4e-5)
    assert first.state.seen_tokens == 5 and tail.state.seen_tokens == ids.shape[1]
    native.save_pretrained(tmp_path / "vlm")
    torch.testing.assert_close(
        load_model(tmp_path / "vlm").eval()(**inputs).logits, all_logits, atol=0, rtol=0
    )


def test_models_k3_vlm_frozen_language_visual_gradient_and_ownership():
    torch.set_num_threads(1)
    torch.manual_seed(323)
    c = KimiK3Config()
    model = build_model(c)
    batch = _batch(c)
    model.language_model.requires_grad_(False)
    CrossEntropyObjective()(model, batch).mean.backward()
    assert model.vision_tower.patch_embed.proj.weight.grad.abs().sum() > 0
    assert model.mm_projector.proj[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.language_model.parameters())
    inputs = batch["model_inputs"]
    with pytest.raises(ValueError, match="placeholders"):
        model(
            torch.tensor([[1, 31, 31, 2]]),
            pixel_values=inputs["pixel_values"],
            grid_thw=inputs["grid_thw"],
        )
    with pytest.raises(ValueError, match="actual visual"):
        model(inputs["input_ids"])
    first = model(**inputs, use_cache=True)
    with pytest.raises(ValueError, match="uncached"):
        model(**inputs, state=first.state)
    with pytest.raises(ValueError, match="snapshot"):
        first.state.truncate(2)
