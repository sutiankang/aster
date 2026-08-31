import ast
import functools
from functools import lru_cache
import hashlib
import os
import urllib.request

import pytest
import torch
from torch import nn

from aster.models.adversarial import ActNorm2d, PatchDiscriminator, PatchDiscriminatorConfig


COMMIT = "3ba01b241669f5ade541ce990f7650a3b8f65318"
HASHES = {
    "taming/modules/discriminator/model.py": "df6c2f8d360c4f4a918e2af7b3feff4ddb5293c0199d047a72963717958d61b0",
    "taming/modules/util.py": "2c5e57e149579e130be70fcc6f45004fcdd1e98f1429ea13cdc13e69ebf767c2",
}
pytestmark = pytest.mark.skipif(
    os.environ.get("ASTER_RUN_REMOTE_PATCHGAN_ORACLE") != "1",
    reason="Pinned PatchGAN source oracle requires explicit network opt-in",
)


@lru_cache(None)
def definitions(path, names):
    url = f"https://raw.githubusercontent.com/CompVis/taming-transformers/{COMMIT}/{path}"
    source = urllib.request.urlopen(url, timeout=20).read()
    assert hashlib.sha256(source).hexdigest() == HASHES[path]
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(nodes) == len(names)
    return compile(ast.Module(body=nodes, type_ignores=[]), url, "exec")


def source_classes():
    scope = dict(torch=torch, nn=nn, functools=functools)
    exec(definitions("taming/modules/util.py", ("ActNorm",)), scope)
    exec(
        definitions(
            "taming/modules/discriminator/model.py", ("NLayerDiscriminator", "weights_init")
        ),
        scope,
    )
    return scope


def test_actual_source_actnorm_initialization_forward_inverse_logdet_all_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(971)
    scope = source_classes()
    official = scope["ActNorm"](4, logdet=True).double()
    native = ActNorm2d(4, logdet=True).double()
    x = torch.randn(3, 4, 5, 6, dtype=torch.float64, requires_grad=True)
    native.initialize(x.detach())
    expected, expected_logdet = official(x)
    actual, actual_logdet = native(x)
    for a, b in (
        (native.affine.loc, official.loc),
        (native.affine.scale, official.scale),
        (actual, expected),
        (actual_logdet, expected_logdet),
    ):
        torch.testing.assert_close(a, b, atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(native(actual, reverse=True), official(expected, reverse=True))
    native_grads = torch.autograd.grad(
        actual.square().sum() + actual_logdet.sum(), (x, *native.parameters()), retain_graph=True
    )
    source_grads = torch.autograd.grad(
        expected.square().sum() + expected_logdet.sum(), (x, *official.parameters())
    )
    for a, b in zip(native_grads, source_grads):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("normalization", ["batchnorm", "actnorm"])
def test_actual_source_patchgan_all_parameter_and_input_gradients(normalization):
    torch.set_num_threads(1)
    torch.manual_seed(821)
    scope = source_classes()
    official = scope["NLayerDiscriminator"](
        input_nc=3, ndf=4, n_layers=2, use_actnorm=normalization == "actnorm"
    )
    official.apply(scope["weights_init"])
    calibration = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        official(calibration)
    official.eval()
    native = PatchDiscriminator(
        PatchDiscriminatorConfig(base_channels=4, num_layers=2, normalization=normalization)
    )
    native.load_reference_state(official.state_dict()).eval()
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    actual, expected = native(x), official(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    native_grads = torch.autograd.grad(
        actual.square().mean(), (x, *native.parameters()), retain_graph=True
    )
    source_grads = torch.autograd.grad(expected.square().mean(), (x, *official.parameters()))
    for a, b in zip(native_grads, source_grads):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
