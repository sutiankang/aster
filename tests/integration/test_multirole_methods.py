import copy
from dataclasses import replace
import torch
from aster.models.generative import UNetConfig, UNet2D
from aster.models.world import RSSMConfig, RSSMWorldModel
from aster.methods.generative_distillation import DMDMethod
from aster.methods.world_model import ImaginedActorCritic
from aster.methods.reinforcement import GaussianActor, mlp
from aster.training import Trainer


def test_dmd_online_fake_score_and_generator_are_separate_updates():
    torch.set_num_threads(1)
    torch.manual_seed(18)
    config = UNetConfig(
        in_channels=1,
        model_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        attention_levels=(),
        num_heads=2,
        prediction_type="x0",
    )
    generator = UNet2D(config)
    real = UNet2D(replace(config, prediction_type="edm_residual"))
    fake = UNet2D(replace(config, prediction_type="edm_residual"))
    original = copy.deepcopy(real.state_dict())
    engine = Trainer(generator, lr=0.001)
    method = DMDMethod(engine, real, fake)
    result = method.update([{"noise": torch.randn(2, 1, 4, 4), "sigma": torch.tensor([0.5, 1.0])}])
    assert result["fake_score"][0].updated and result["generator"].updated
    assert generator.output[-1].weight.abs().sum() > 0
    for key, value in original.items():
        torch.testing.assert_close(value, real.state_dict()[key])


def test_imagined_world_actor_value_updates(tmp_path):
    torch.manual_seed(19)
    config = RSSMConfig(
        observation_dim=3,
        action_dim=2,
        deter_dim=8,
        stochastic_variables=2,
        classes=2,
        hidden_size=8,
        blocks=2,
        reward_bins=11,
    )
    world = RSSMWorldModel(config)
    actor = GaussianActor(config.feature_dim, 2, hidden=8)
    value = mlp(config.feature_dim, 1, hidden=8)
    engine = Trainer(world, lr=0.001)
    method = ImaginedActorCritic(engine, actor, value, horizon=3)
    batch = {
        "observations": torch.randn(2, 3, 3),
        "actions": torch.randn(2, 3, 2),
        "rewards": torch.randn(2, 3),
        "is_first": torch.tensor([[True, False, False], [True, False, False]]),
        "terminated": torch.zeros(2, 3, dtype=torch.bool),
    }
    result = method.update([batch])
    assert all(step.updated for step in result.values())
    engine.save_checkpoint(tmp_path / "imagined")
    method.update([batch])
    expected = copy.deepcopy(actor.state_dict())
    engine.load_checkpoint(tmp_path / "imagined", trusted=True)
    method.update([batch])
    for key, value in expected.items():
        torch.testing.assert_close(value, actor.state_dict()[key])
