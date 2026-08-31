import math
import pytest
import torch
import torch.nn.functional as F

from aster.models.genie import STStack


@pytest.mark.parametrize("qk_norm", [False, True])
def test_genie_st_attention_separability_formula_and_all_gradients(qk_norm):
    torch.set_num_threads(1)
    torch.manual_seed(896)

    model = STStack(8, 2, 3, 2, 4, 4, 2, qk_norm, 1e-5).double()
    x = torch.randn(2, 3, 4, 8, dtype=torch.float64, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    weights = {
        name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()
    }

    def linear(path, value):
        return value @ weights[path + ".weight"].T + weights[path + ".bias"]

    def norm(path, value):
        return F.layer_norm(
            value, (value.shape[-1],), weights[path + ".weight"], weights[path + ".bias"], 1e-5
        )

    def attention(path, value, causal):
        b, s = value.shape[:2]
        q, k, v = linear(path + ".qkv", value).reshape(b, s, 3, 2, 3).unbind(2)
        if qk_norm:
            q, k = norm(path + ".query_norm", q), norm(path + ".key_norm", k)
        score = torch.einsum("bshd,bthd->bhst", q, k) / math.sqrt(3)
        if causal:
            score = score.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), -torch.inf)
        combined = torch.einsum("bhst,bthd->bshd", score.softmax(-1), v).reshape(b, s, 6)
        return linear(path + ".output", combined)

    value = (
        expected_x
        + weights["spatial_position.weight"][None, None]
        + weights["temporal_position.weight"][None, :3, None]
    )
    for index in range(2):
        prefix = f"blocks.{index}"
        value = value + attention(
            prefix + ".spatial", norm(prefix + ".spatial_norm", value).reshape(6, 4, 8), False
        ).reshape(2, 3, 4, 8)
        temporal = norm(prefix + ".temporal_norm", value).transpose(1, 2).reshape(8, 3, 8)
        value = value + attention(prefix + ".temporal", temporal, True).reshape(
            2, 4, 3, 8
        ).transpose(1, 2)
        value = value + linear(
            prefix + ".ffn.2", F.gelu(linear(prefix + ".ffn.0", norm(prefix + ".ffn_norm", value)))
        )
    expected = norm("norm", value)
    actual = model(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    probe = torch.randn_like(expected)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad, atol=1e-11, rtol=1e-9)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, weights[name].grad, atol=1e-11, rtol=1e-9)
