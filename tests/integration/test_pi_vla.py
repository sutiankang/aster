import torch
import pytest
from aster.models.pi_vla import PiVLA, PiVLAConfig
from aster.models.siglip import SigLIPVisionConfig
from aster.models.policies import PiConfig
from aster.methods.actions import PiActionObjective
from aster.training import Trainer


def small(pi05=False):
    return PiVLAConfig(
        vision=SigLIPVisionConfig(
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
            vision_use_head=False,
        ),
        expert=PiConfig(
            action_dim=2,
            action_horizon=3,
            prefix_width=16,
            action_width=8,
            prefix_mlp=24,
            action_mlp=16,
            num_layers=1,
            num_heads=2,
            kv_heads=1,
            head_dim=4,
            pi05=pi05,
        ),
        vocab_size=16,
        max_prompt_length=8,
        prompt_contains_state=pi05,
    )


def observation():
    return {
        "images": {
            "front": torch.rand(2, 3, 8, 8) * 2 - 1,
            "wrist": torch.rand(2, 3, 8, 8) * 2 - 1,
        },
        "image_masks": {"front": torch.tensor([True, True]), "wrist": torch.tensor([False, False])},
        "input_ids": torch.tensor([[1, 3, 2], [1, 4, 2]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.bool),
        "proprio": torch.randn(2, 2),
    }


@pytest.mark.parametrize("pi05", [False, True])
def test_pixels_language_actions_training_cache_and_export(tmp_path, pi05):
    torch.manual_seed(31)
    torch.set_num_threads(1)
    model = PiVLA(small(pi05))
    data = observation()
    actions = torch.randn(2, 3, 2)
    engine = Trainer(model, PiActionObjective(), lr=0.001)
    for _ in range(2):
        assert engine.step([{"actions": actions, "observation": data}]).updated
    assert model.vision_projection.weight.abs().sum() > 0
    model.eval()
    noise = torch.randn_like(actions)
    cached = model.sample_actions(data, noise=noise, steps=3)
    uncached = model.sample_actions(data, noise=noise, steps=3, cache_prefix=False)
    torch.testing.assert_close(cached, uncached, atol=1e-6, rtol=1e-5)
    changed = {
        **data,
        "images": {**data["images"], "wrist": torch.zeros_like(data["images"]["wrist"])},
    }
    torch.testing.assert_close(
        model.sample_actions(changed, noise=noise, steps=3), cached, atol=1e-6, rtol=1e-5
    )
    model.save_pretrained(tmp_path / "model")
    restored = PiVLA.from_pretrained(tmp_path / "model")
    torch.testing.assert_close(
        restored.sample_actions(data, noise=noise, steps=3), cached, atol=1e-6, rtol=1e-5
    )
