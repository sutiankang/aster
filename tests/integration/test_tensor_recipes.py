import torch
import pytest
from aster.core import ArtifactStore
from aster.tensor_recipes import fit_tensors
from aster.models import load_model
from aster.models.config import config_from_dict


@pytest.mark.parametrize("kind", ["flow", "rssm", "act"])
def test_shared_tensor_training_domain_artifact_roundtrip(tmp_path, kind):
    torch.set_num_threads(1)
    torch.manual_seed(29)
    if kind == "flow":
        model = {
            "architecture": "unet2d",
            "in_channels": 1,
            "model_channels": 8,
            "channel_mult": [1],
            "attention_levels": [],
            "num_heads": 2,
            "num_res_blocks": 1,
            "prediction_type": "velocity",
        }
        data = {"sample": torch.randn(4, 1, 4, 4)}
    elif kind == "rssm":
        model = {
            "architecture": "rssm",
            "observation_dim": 3,
            "action_dim": 2,
            "deter_dim": 8,
            "stochastic_variables": 2,
            "classes": 2,
            "hidden_size": 8,
            "blocks": 2,
            "reward_bins": 11,
        }
        data = {
            "observations": torch.randn(4, 3, 3),
            "actions": torch.randn(4, 3, 2),
            "is_first": torch.tensor([[True, False, False]] * 4),
            "terminated": torch.zeros(4, 3, dtype=torch.bool),
            "rewards": torch.randn(4, 3),
        }
    else:
        from aster.models.actions import ACTConfig

        config = ACTConfig(
            action_dim=2,
            proprio_dim=3,
            vision_dim=4,
            hidden_size=16,
            num_heads=2,
            encoder_layers=1,
            decoder_layers=1,
            posterior_layers=1,
            latent_dim=4,
            horizon=3,
            feedforward_size=32,
        )
        model = config.to_dict()
        data = {
            "proprio": torch.randn(4, 3),
            "vision_tokens": torch.randn(4, 2, 4),
            "actions": torch.randn(4, 3, 2),
        }
    torch.save(data, tmp_path / "data.pt")
    config = {
        "model": model,
        "objective": {"name": kind},
        "data": str(tmp_path / "data.pt"),
        "preprocessing": {"type": "synthetic_tensor_fixture", "version": "1"},
        "training": {"steps": 2, "batch_size": 2},
    }
    store = ArtifactStore(tmp_path / "artifacts")
    result = fit_tensors(config, {}, tmp_path / "run", store)
    artifact = store.get(result.artifacts["model"])
    loaded = load_model(artifact.path / "model")
    assert loaded.config.to_dict() == config_from_dict(model).to_dict()
    assert result.details["steps"] == 2 and result.metrics["final_loss"] >= 0
