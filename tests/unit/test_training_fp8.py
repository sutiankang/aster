from copy import deepcopy
import pytest
import torch

from aster.core.contracts import LossTerm
from aster.training import Trainer, FP8Recipe, FP8Quantizer, FP8Linear


def objective(model, batch):
    x, y = batch
    error = (model(x) - y).square()
    return LossTerm(error.sum(), torch.tensor(error.numel()), "elements")


def test_fp8_true_byte_storage_history_and_zero():
    quantizer = FP8Quantizer(torch.float8_e4m3fn, FP8Recipe(history_length=3))
    quantized, scale = quantizer(torch.tensor([0.0, 1.0, -2.0]))
    assert quantized.dtype == torch.float8_e4m3fn and quantized.element_size() == 1
    torch.testing.assert_close(quantized.float() * scale, torch.tensor([0.0, 1.0, -2.0]))
    _, second = quantizer(torch.tensor([16.0]))
    torch.testing.assert_close(second, scale)
    _, third = quantizer(torch.tensor([4.0]))
    assert third > second and quantizer.updates == 3
    zero, inverse = FP8Quantizer(torch.float8_e5m2)(torch.zeros(3))
    assert torch.isfinite(zero.float()).all() and inverse == 1


def test_fp8_hybrid_forward_and_backward_match_declared_quantized_formula():
    torch.manual_seed(789)
    layer = FP8Linear(5, 3, recipe=FP8Recipe(scaling="current"))
    inputs = torch.randn(4, 5, requires_grad=True)
    outputs = layer(inputs)
    qx = (inputs.detach() / layer.inputs.last_inverse_scale).to(
        torch.float8_e4m3fn
    ).float() * layer.inputs.last_inverse_scale
    qw = (layer.weight.detach() / layer.weights.last_inverse_scale).to(
        torch.float8_e4m3fn
    ).float() * layer.weights.last_inverse_scale
    torch.testing.assert_close(outputs, qx @ qw.T + layer.bias)
    gradient = torch.randn_like(outputs)
    dx, dw, db = torch.autograd.grad(outputs, (inputs, layer.weight, layer.bias), gradient)
    qg = (gradient / layer.gradients.last_inverse_scale).to(
        torch.float8_e5m2
    ).float() * layer.gradients.last_inverse_scale
    torch.testing.assert_close(dx, qg @ qw)
    torch.testing.assert_close(dw, qg.T @ qx)
    torch.testing.assert_close(db, gradient.sum(0))


def test_fp8_checkpoint_next_update_and_eval_does_not_mutate_history(tmp_path):
    torch.manual_seed(177)
    trainer = Trainer(FP8Linear(3, 2), objective, lr=0.01)
    batch = (torch.ones(4, 3), torch.zeros(4, 2))
    trainer.step([batch])
    path = trainer.save_checkpoint(tmp_path / "fp8.json")
    trainer.step([batch])
    expected = deepcopy(trainer.model.state_dict())
    trainer.load_checkpoint(path)
    trainer.step([batch])
    for key, value in expected.items():
        torch.testing.assert_close(value, trainer.model.state_dict()[key], rtol=0, atol=0)
    state = deepcopy(trainer.model.state_dict())
    trainer.evaluate([batch])
    for key, value in state.items():
        torch.testing.assert_close(value, trainer.model.state_dict()[key], rtol=0, atol=0)
    different = Trainer(FP8Linear(3, 2, recipe=FP8Recipe(scaling="current")), objective, lr=0.01)
    with pytest.raises(ValueError, match="配置"):
        different.load_checkpoint(path)


def test_fp8_hardware_path_never_silently_falls_back():
    with pytest.raises(RuntimeError, match="CUDA"):
        FP8Linear(16, 16, implementation="scaled_mm")(torch.ones(16, 16))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="真实 FP8 TensorCore 验收需要 Hopper/更新 CUDA 硬件",
)
def test_fp8_cuda_scaled_mm_matches_reference():
    torch.manual_seed(644)
    reference = FP8Linear(32, 32, recipe=FP8Recipe(scaling="current")).cuda()
    hardware = FP8Linear(
        32, 32, recipe=FP8Recipe(scaling="current"), implementation="scaled_mm"
    ).cuda()
    hardware.load_state_dict(reference.state_dict())
    x = torch.randn(32, 32, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = reference(x), hardware(y)
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    gradient = torch.randn_like(a)
    ga = torch.autograd.grad(a, (x, reference.weight), gradient)
    gb = torch.autograd.grad(b, (y, hardware.weight), gradient)
    for actual, expected in zip(gb, ga):
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
