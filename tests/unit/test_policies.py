import numpy as np
import pytest
import torch
from aster.models.policies import DiffusionPolicyConfig, DiffusionPolicy1D, PiConfig, PiActionExpert
from aster.methods.generation import DiffusionObjective, DiffusionSchedule
from aster.methods.actions import PiActionObjective
from aster.data.actions import UniformActionTokenizer
from aster.training import Trainer


def test_diffusion_action_policy_uses_common_denoising_engine():
    torch.set_num_threads(1)
    model = DiffusionPolicy1D(
        DiffusionPolicyConfig(action_dim=2, condition_dim=4, down_dims=(8, 16), time_dim=8)
    )
    batch = {"sample": torch.randn(2, 8, 2), "condition": torch.randn(2, 4)}
    assert Trainer(model, DiffusionObjective(DiffusionSchedule.create(10))).step([batch]).updated


@pytest.mark.parametrize("pi05", [False, True])
def test_pi_double_expert_training_and_prefix_cache(pi05):
    torch.manual_seed(2)
    config = PiConfig(
        action_dim=2,
        action_horizon=4,
        prefix_width=16,
        action_width=8,
        prefix_mlp=32,
        action_mlp=16,
        num_layers=2,
        num_heads=2,
        kv_heads=1,
        head_dim=4,
        pi05=pi05,
    )
    model = PiActionExpert(config)
    observation = {
        "prefix_embeds": torch.randn(2, 5, 16),
        "prefix_mask": torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        ),
        "proprio": torch.randn(2, 2),
    }
    batch = {"actions": torch.randn(2, 4, 2), "observation": observation}
    assert Trainer(model, PiActionObjective()).step([batch]).updated
    model.eval()
    noise = torch.randn(2, 4, 2)
    time = torch.tensor([0.5, 0.7])
    full = model(noise, time, observation).prediction
    state = model.encode_prefix(observation)
    cached = model(noise, time, observation, prefix_state=state).prediction
    torch.testing.assert_close(full, cached, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        model.sample_actions(observation, noise=noise, steps=3, cache_prefix=True),
        model.sample_actions(observation, noise=noise, steps=3, cache_prefix=False),
        atol=2e-6,
        rtol=2e-5,
    )

    before = state.layers[0][0].clone()
    model(noise * 2, time, observation, prefix_state=state)
    torch.testing.assert_close(before, state.layers[0][0])


def test_openvla_token_boundary_matches_numpy_digitize():
    tokenizer = UniformActionTokenizer(32000)
    actions = torch.tensor([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    expected = 32000 - np.digitize(actions.clamp(-1, 1).double().numpy(), np.linspace(-1, 1, 256))
    assert tokenizer.encode(actions).tolist() == expected.tolist()
    reconstructed = tokenizer.decode(tokenizer.encode(actions))
    assert (reconstructed - actions.clamp(-1, 1)).abs().max() <= 2 / 255
