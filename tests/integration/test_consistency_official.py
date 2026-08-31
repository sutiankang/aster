import ast
from copy import deepcopy
import hashlib
import os
import urllib.request

import numpy as np
import pytest
import torch
from torch import nn

from aster.core import FieldOutput
from aster.methods.consistency import ConsistencyConfig, _ConsistencyObjective


@pytest.mark.skipif(
    os.environ.get("ASTER_RUN_REMOTE_CONSISTENCY_ORACLE") != "1",
    reason="Explicit network source-oracle execution is not enabled",
)
@pytest.mark.parametrize("distillation", [False, True])
def test_pinned_actual_openai_consistency_loss_and_gradient(distillation):
    url = "https://raw.githubusercontent.com/openai/consistency_models/e32b69ee436d518377db86fb2127a3972d0d8716/cm/karras_diffusion.py"
    with urllib.request.urlopen(url, timeout=30) as response:
        source = response.read()
    assert (
        hashlib.sha256(source).hexdigest()
        == "b8fcf9f53e63cff19db676814545ee7644259364236de5c10ac3d69007ee5177"
    )

    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in {"KarrasDenoiser", "get_weightings"}
    ]
    namespace = {
        "th": torch,
        "np": np,
        "F": torch.nn.functional,
        "mean_flat": lambda value: value.mean(tuple(range(1, value.ndim))),
        "append_dims": lambda value, dimensions: value.reshape(
            *value.shape, *([1] * (dimensions - value.ndim))
        ),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), url, "exec"), namespace)
    denoiser = namespace["KarrasDenoiser"](
        distillation=True, loss_norm="l2", weight_schedule="karras"
    )
    teacher_denoiser = namespace["KarrasDenoiser"](
        distillation=False, loss_norm="l2", weight_schedule="karras"
    )

    class Field(nn.Module):
        def __init__(self, wrapped, kind):
            super().__init__()
            self.value = nn.Linear(3, 3)
            self.wrapped, self.kind = wrapped, kind

        def forward(self, sample, time, condition=None):
            value = self.value(sample) + time[:, None] * 0.0001
            return FieldOutput(value, self.kind) if self.wrapped else value

    torch.manual_seed(361)
    original = Field(False, "consistency_residual")
    native = deepcopy(original)
    native.wrapped = True
    original_target = deepcopy(original).requires_grad_(False)
    native_target = deepcopy(native).requires_grad_(False)
    original_teacher = Field(False, "edm_residual") if distillation else None
    native_teacher = deepcopy(original_teacher)
    if native_teacher is not None:
        native_teacher.wrapped = True
    sample, noise = torch.randn(3, 3), torch.randn(3, 3)

    torch.manual_seed(4)
    indices = torch.randint(0, 3, (3,))
    high = (80 ** (1 / 7) + indices / 3 * (0.002 ** (1 / 7) - 80 ** (1 / 7))).pow(7)
    low = (80 ** (1 / 7) + (indices + 1) / 3 * (0.002 ** (1 / 7) - 80 ** (1 / 7))).pow(7)
    torch.manual_seed(4)
    official = denoiser.consistency_losses(
        original,
        sample,
        num_scales=4,
        target_model=original_target,
        teacher_model=original_teacher,
        teacher_diffusion=teacher_denoiser,
        noise=noise,
    )["loss"].mean()
    config = ConsistencyConfig(
        mode="cd" if distillation else "ct",
        total_steps=10,
        initial_scales=4,
        final_scales=4,
        curriculum="fixed",
        target_ema_mode="fixed",
        weighting="karras",
        teacher_time_scale=250.0,
        metric="mse",
    )
    actual = _ConsistencyObjective(config, native_target, native_teacher)(
        native, dict(sample=sample, noise=noise, sigma_high=high, sigma_low=low)
    ).mean
    torch.testing.assert_close(actual, official, rtol=4e-6, atol=1e-7)
    official.backward()
    actual.backward()
    for first, second in zip(original.parameters(), native.parameters()):
        torch.testing.assert_close(first.grad, second.grad, rtol=1e-5, atol=3e-7)
