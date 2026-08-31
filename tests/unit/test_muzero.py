from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from aster.models.muzero import (
    MuZeroConfig,
    MuZeroModel,
    scalar_transform,
    inverse_scalar_transform,
    support_targets,
)
from aster.methods.muzero import MuZeroObjective, MuZeroMethod, nstep_value_targets
from aster.training import Trainer


def _batch():
    return dict(
        observations=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        actions=torch.tensor([[0, 1], [1, 0], [1, 1]]),
        policy_targets=torch.tensor([[[0.0, 1.0]] * 3] * 3),
        value_targets=torch.tensor([[0.7, 0.8, 1.0]] * 3),
        reward_targets=torch.tensor([[0.0, 1.0]] * 3),
        valid=torch.tensor([[True, True, True], [True, True, False], [True, True, True]]),
    )


def _config():
    return MuZeroConfig(
        observation_dim=3, num_actions=2, latent_dim=8, hidden_size=16, support_size=5
    )


@pytest.mark.parametrize("epsilon", [0.0, 0.001, 0.1])
def test_muzero_scalar_support_and_stable_inverse(epsilon):
    values = torch.tensor([-1e5, -3.7, -0.01, 0.0, 0.01, 3.7, 1e5], dtype=torch.float64)
    transformed = scalar_transform(values, epsilon)
    torch.testing.assert_close(
        inverse_scalar_transform(transformed, epsilon), values, atol=3e-10, rtol=1e-10
    )
    safe = torch.tensor([-10.0, -2.0, 0.0, 2.0, 10.0])
    target = support_targets(safe, 30, epsilon)
    torch.testing.assert_close(target.sum(-1), torch.ones_like(safe))
    assert (target.gt(0).sum(-1) <= 2).all()
    torch.testing.assert_close(
        (target * torch.arange(-30, 31)).sum(-1), scalar_transform(safe, epsilon)
    )
    clipped = support_targets(torch.tensor([-1e9, 1e9]), 5)
    assert clipped[0, 0] == 1 and clipped[1, -1] == 1


def test_muzero_gradient_scaling_does_not_change_forward_or_current_dynamics_gradients():
    torch.manual_seed(98)
    model = MuZeroModel(_config())
    unscaled = MuZeroModel(replace(_config(), dynamics_gradient_scale=1.0))
    unscaled.load_state_dict(model.state_dict())
    latent = torch.randn(3, 8, requires_grad=True)
    other = latent.detach().clone().requires_grad_()
    left, right = (
        model.recurrent(torch.tensor([0, 1, 0]), latent),
        unscaled.recurrent(torch.tensor([0, 1, 0]), other),
    )
    for key in ("prior_logits", "value_logits", "reward_logits", "embedding"):
        torch.testing.assert_close(getattr(left, key), getattr(right, key), atol=0, rtol=0)
    for prediction in (left, right):
        (
            prediction.prior_logits.sum()
            + prediction.value_logits.sum()
            + prediction.reward_logits.sum()
        ).backward()
    torch.testing.assert_close(latent.grad, 0.5 * other.grad)
    for p, q in zip(model.dynamics.parameters(), unscaled.dynamics.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_muzero_nstep_final_observation_and_termination_semantics():
    reward = torch.tensor([[1.0, 2.0, 3.0]] * 2)
    values = torch.tensor([[10.0, 20.0, 30.0, 40.0]] * 2)
    terminal = torch.tensor([[False, True, False], [False, False, False]])
    truncated = torch.tensor([[False, False, False], [False, True, False]])
    actual = nstep_value_targets(
        reward, values, terminal, truncated=truncated, n_steps=3, discount=0.5
    )
    torch.testing.assert_close(
        actual, torch.tensor([[2.0, 2.0, 23.0, 40.0], [9.5, 17.0, 23.0, 40.0]])
    )


def test_muzero_objective_coordinate_mask_per_k_and_importance_formula():
    torch.manual_seed(51)
    model, objective = MuZeroModel(_config()), MuZeroObjective(value_weight=0.25)
    data = _batch()
    data["importance_weights"] = torch.tensor([1.0, 0.7, 0.4])
    bundle = objective(model, data)
    prediction = model(data["observations"], data["actions"])
    expected = {name: torch.zeros(()) for name in ("policy", "value", "reward")}
    for b in range(3):
        for step in range(3):
            if data["valid"][b, step]:
                expected["policy"] += (
                    -F.log_softmax(prediction[step].prior_logits[b], -1)[1]
                    * data["importance_weights"][b]
                    / 2
                )
                target = support_targets(data["value_targets"][b, step], 5)
                expected["value"] += (
                    -(target * F.log_softmax(prediction[step].value_logits[b], -1)).sum()
                    * data["importance_weights"][b]
                    / 2
                )
            if step and data["valid"][b, step - 1]:
                target = support_targets(data["reward_targets"][b, step - 1], 5)
                expected["reward"] += (
                    -(target * F.log_softmax(prediction[step].reward_logits[b], -1)).sum()
                    * data["importance_weights"][b]
                    / 2
                )
    for term in bundle.terms:
        torch.testing.assert_close(term.numerator, expected[term.name])
        assert term.denominator.dtype == torch.int64 and term.denominator == 3
        assert term.weight == (0.25 if term.name == "value" else 1.0)


@pytest.mark.parametrize("stage", [0, 3])
def test_muzero_train_unroll_shared_engine_exact_resume_and_export(stage, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(52)
    model, objective = MuZeroModel(_config()), MuZeroObjective()
    engine = Trainer(model, objective, lr=0.005, zero_stage=stage, max_grad_norm=None)
    method, data = MuZeroMethod(engine, objective), _batch()
    first = method.update([data]).loss
    for _ in range(80):
        last = method.update([data]).loss
    assert last < first * 0.45
    engine.save_checkpoint(tmp_path / "checkpoint")
    expected = method.update([data])
    weights = engine.export_state_dict()
    engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    replayed = method.update([data])
    assert replayed.loss == expected.loss and method.updates == 82
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    restored = MuZeroModel(_config())
    restored.load_state_dict(weights)
    restored.save_pretrained(tmp_path / "model")
    from aster.models import load_model

    loaded = load_model(tmp_path / "model")
    torch.testing.assert_close(
        loaded(data["observations"]).prior_logits,
        restored(data["observations"]).prior_logits,
        atol=0,
        rtol=0,
    )


def test_muzero_invalid_trajectory_rejected_before_training_mutation():
    engine = Trainer(MuZeroModel(_config()))
    method, data = MuZeroMethod(engine), _batch()
    before = deepcopy(engine.export_state_dict())
    data["valid"][0, 1] = False
    with pytest.raises(ValueError, match="preflight"):
        method.update([data])
    assert engine.steps == 0
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
