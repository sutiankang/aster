from copy import deepcopy
import math

import pytest
import torch

from aster.optimization.evolution import CMAES


def test_cmaes_native_first_generation_matches_author_formula():
    mean = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    es = CMAES(mean, sigma=0.4, population=6, seed=8)
    samples = es.ask()
    values = samples.square().sum(-1)
    ordered = samples[values.argsort(stable=True)]
    weights = torch.tensor([math.log(3.5) - math.log(i) for i in (1, 2, 3)], dtype=torch.float64)
    weights /= weights.sum()
    new_mean = (ordered[:3] * weights[:, None]).sum(0)
    effective = 1 / weights.square().sum().item()
    cc, cs = (4 + effective / 3) / (7 + 2 * effective / 3), (effective + 2) / (8 + effective)
    c1 = 2 / (4.3**2 + effective)
    cmu = min(1 - c1, 2 * (effective - 2 + 1 / effective) / (25 + effective))
    damp = 2 * effective / 6 + 0.3 + cs
    ps = math.sqrt(cs * (2 - cs) * effective) / 0.4 * (new_mean - mean)
    hsig = ps.square().sum().item() / 3 / (1 - (1 - cs) ** 2) < 3
    pc = math.sqrt(cc * (2 - cc) * effective) / 0.4 * hsig * (new_mean - mean)
    expected = (1 - c1 * (1 - (1 - float(hsig) ** 2) * cc * (2 - cc)) - cmu) * torch.eye(
        3, dtype=torch.float64
    ) + c1 * torch.outer(pc, pc)
    for weight, candidate in zip(weights, ordered):
        expected += cmu * weight / 0.4**2 * torch.outer(candidate - mean, candidate - mean)
    sigma = 0.4 * math.exp(min(1.0, cs / damp * (ps.square().sum().item() / 3 - 1) / 2))
    es.tell(values)
    torch.testing.assert_close(es.mean, new_mean, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(es.covariance, expected, atol=1e-14, rtol=1e-14)
    assert abs(es.sigma - sigma) < 1e-14


def test_cmaes_optimizes_and_pending_round_restores_exactly():
    torch.set_num_threads(1)
    es = CMAES(torch.ones(4), sigma=0.5, population=12, seed=6)
    for _ in range(30):
        es.tell(es.ask().square().sum(-1))
    pending = es.ask()
    checkpoint = deepcopy(es.state_dict())
    with pytest.raises(RuntimeError, match="unevaluated"):
        es.ask()
    with pytest.raises(ValueError, match="finite scores"):
        es.tell(torch.full((12,), math.nan))
    assert torch.equal(es.pending, pending)
    es.tell(pending.square().sum(-1))
    following = es.ask()
    restored = CMAES(torch.ones(4), sigma=0.5, population=12, seed=122)
    restored.load_state_dict(checkpoint)
    restored.tell(restored.pending.square().sum(-1))
    torch.testing.assert_close(restored.ask(), following, atol=0, rtol=0)
    es.tell(following.square().sum(-1))
    for _ in range(50):
        es.tell(es.ask().square().sum(-1))
    assert es.best_value < 1e-8


def test_cmaes_budget_and_checkpoint_validation():
    with pytest.raises(ValueError, match="memory budget"):
        CMAES(torch.ones(100), max_covariance_bytes=100)
    es = CMAES(torch.zeros(3))
    state = es.state_dict()
    state["tensors"]["eigenvalues"][0] = -1
    with pytest.raises(ValueError, match="eigensystem"):
        es.load_state_dict(state)
