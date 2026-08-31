from types import SimpleNamespace
import torch
import pytest
from aster.methods import SigmoidContrastiveObjective


def test_siglip_binary_pairs_exact_scaling_and_gradient():
    logits = torch.tensor([[2.0, -1.0], [0.0, 3.0]], requires_grad=True)
    term = SigmoidContrastiveObjective()(
        lambda: SimpleNamespace(logits_per_text=logits), {"model_inputs": {}}
    )
    labels = torch.eye(2)
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum") / 2
    )
    torch.testing.assert_close(term.mean, expected)
    (actual_grad,) = torch.autograd.grad(term.mean, logits, retain_graph=True)
    (ref_grad,) = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_grad, ref_grad)
    rectangle = torch.zeros(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match="explicit pair_labels"):
        SigmoidContrastiveObjective()(
            lambda: SimpleNamespace(logits_per_text=rectangle), {"model_inputs": {}}
        )
    term = SigmoidContrastiveObjective()(
        lambda: SimpleNamespace(logits_per_text=rectangle),
        {"model_inputs": {}, "pair_labels": torch.zeros(2, 3, dtype=torch.bool)},
    )
    torch.testing.assert_close(term.mean, 3 * torch.log(torch.tensor(2.0)))
