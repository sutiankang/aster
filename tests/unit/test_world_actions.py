import torch
from aster.models.world import RSSMConfig, RSSMWorldModel, symlog, symexp, two_hot
from aster.models.actions import ACTConfig, ACTPolicy
from aster.methods.world_model import WorldModelObjective, cem_plan
from aster.methods.actions import ACTObjective
from aster.data.actions import ActionSpec, ActionNormalizer, TemporalEnsembler
from aster.training import Trainer


def test_rssm_observe_reset_training_imagination():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = RSSMWorldModel(
        RSSMConfig(
            observation_dim=4,
            action_dim=2,
            deter_dim=16,
            stochastic_variables=2,
            classes=4,
            hidden_size=16,
            blocks=2,
            reward_bins=21,
        )
    )
    batch = {
        "observations": torch.randn(2, 4, 4),
        "actions": torch.randn(2, 4, 2),
        "is_first": torch.tensor([[1, 0, 1, 0], [1, 0, 0, 0]], dtype=torch.bool),
        "rewards": torch.randn(2, 4),
        "terminated": torch.zeros(2, 4, dtype=torch.bool),
    }
    assert Trainer(model, WorldModelObjective()).step([batch]).updated
    sequence, prior, final = model.observe(
        batch["observations"], batch["actions"], batch["is_first"], sample=False
    )
    restart, _ = model.step(
        model.initial(1),
        batch["actions"][0:1, 2],
        batch["observations"][0:1, 2],
        reset=torch.ones(1, dtype=torch.bool),
        sample=False,
    )
    torch.testing.assert_close(sequence.deter[0, 2], restart.deter[0])
    torch.testing.assert_close(sequence.stochastic[0, 2], restart.stochastic[0])
    assert not final.capabilities.truncatable
    action, info = cem_plan(
        model, final.reorder(torch.tensor([0])), horizon=3, population=8, elites=2, iterations=2
    )
    assert action.shape == (2,) and (action.abs() <= 1).all()


def test_world_distribution_utilities():
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    torch.testing.assert_close(symexp(symlog(x)), x)
    support = torch.linspace(-2, 2, 5)
    target = two_hot(torch.tensor([-3.0, -0.5, 2.0]), support)
    torch.testing.assert_close(target.sum(-1), torch.ones(3))
    torch.testing.assert_close((target * support).sum(-1), torch.tensor([-2.0, -0.5, 2.0]))


def test_act_cvae_training_and_deterministic_inference():
    config = ACTConfig(
        proprio_dim=3,
        action_dim=2,
        vision_dim=4,
        hidden_size=16,
        latent_dim=4,
        horizon=3,
        num_heads=2,
        posterior_layers=1,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_size=32,
    )
    model = ACTPolicy(config)
    batch = {
        "proprio": torch.randn(2, 3),
        "vision_tokens": torch.randn(2, 4, 4),
        "actions": torch.randn(2, 3, 2),
        "action_padding": torch.tensor([[False, False, True], [False, False, False]]),
    }
    assert Trainer(model, ACTObjective()).step([batch]).updated
    model.eval()
    first = model.predict_chunk(batch).actions
    torch.testing.assert_close(first, model.predict_chunk(batch).actions)
    assert first.shape == (2, 3, 2)
    assert model(**{k: batch[k] for k in ("proprio", "vision_tokens")}).mean is None


def test_action_units_normalization_and_zero_valid_ensemble():
    spec = ActionSpec(("x", "grip"), ("m", "ratio"), "base", "delta", 20.0, 2)
    data = torch.randn(4, 3, 2)
    normalizer = ActionNormalizer.fit(data, spec=spec)
    torch.testing.assert_close(normalizer.denormalize(normalizer.normalize(data)), data)
    ensemble = TemporalEnsembler(3, decay=0.0)
    ensemble.add(0, torch.zeros(3, 2))
    torch.testing.assert_close(ensemble.action(0), torch.zeros(2))
    ensemble.add(1, torch.ones(3, 2))
    torch.testing.assert_close(ensemble.action(1), torch.full((2,), 0.5))
