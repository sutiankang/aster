import ast
from collections import namedtuple
from functools import lru_cache
import hashlib
import os
from types import SimpleNamespace
import urllib.request

import pytest
import torch
from torch import nn

from aster.models.perceptual import LPIPS, LPIPSConfig


COMMIT = "082bb24f84c091ea94de2867d34c4544f68e0963"
HASHES = {
    "lpips/lpips.py": "3eaeb96c0029b2d849745e6691ccb0d85361bed9cf9953b7ceaed8dc948a6d83",
    "lpips/pretrained_networks.py": "6a27f714c51796db466e86bebba6a617c1bfc4d566f3a1756497629c1248686e",
    "lpips/__init__.py": "eccfb23848beefe67bed3d0698e40c7e2f4ed43f7e92320e411681523a7a1776",
}


@lru_cache(None)
def definitions(path, names):
    url = f"https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/{COMMIT}/{path}"
    source = urllib.request.urlopen(url, timeout=20).read()
    assert hashlib.sha256(source).hexdigest() == HASHES[path]
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(nodes) == len(names)
    return compile(ast.Module(body=nodes, type_ignores=[]), url, "exec")


def feature_factory(backbone):

    layers = []
    if backbone == "vgg":
        layout = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ]
        width = 3
        for item in layout:
            if item == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.extend((nn.Conv2d(width, item, 3, padding=1), nn.ReLU(inplace=False)))
                width = item
    else:
        layers = [
            nn.Conv2d(3, 64, 11, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        ]
    return SimpleNamespace(features=nn.Sequential(*layers))


@pytest.mark.skipif(
    os.environ.get("ASTER_RUN_REMOTE_LPIPS_ORACLE") != "1",
    reason="Pinned source oracle requires explicit network opt-in",
)
@pytest.mark.parametrize("backbone", ["vgg", "alex"])
def test_actual_official_full_width_lpips_forward_and_input_gradients(backbone):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    feature_scope = dict(
        torch=torch,
        namedtuple=namedtuple,
        tv=SimpleNamespace(
            vgg16=lambda **_: feature_factory("vgg"), alexnet=lambda **_: feature_factory("alex")
        ),
    )
    exec(definitions("lpips/pretrained_networks.py", ("vgg16", "alexnet")), feature_scope)
    norm_scope = dict(torch=torch)
    exec(definitions("lpips/__init__.py", ("normalize_tensor",)), norm_scope)
    scope = dict(
        torch=torch,
        nn=nn,
        lpips=SimpleNamespace(normalize_tensor=norm_scope["normalize_tensor"]),
        pn=SimpleNamespace(vgg16=feature_scope["vgg16"], alexnet=feature_scope["alexnet"]),
    )
    exec(
        definitions(
            "lpips/lpips.py",
            ("LPIPS", "ScalingLayer", "NetLinLayer", "spatial_average", "upsample"),
        ),
        scope,
    )
    native = LPIPS(LPIPSConfig(backbone=backbone, allow_untrained=True))
    official = scope["LPIPS"](pretrained=False, net=backbone, pnet_rand=True, verbose=False)
    reference_weights = {}
    for name in official.net.state_dict():
        _, index, kind = name.split(".")
        reference_weights[name] = native.features[int(index)].state_dict()[kind]
    official.net.load_state_dict(reference_weights, strict=True)
    with torch.no_grad():
        for index, layer in enumerate(native.linear):
            getattr(official, f"lin{index}").model[1].weight.copy_(layer.weight)
    left = torch.rand(1, 3, 32, 32, requires_grad=True) * 2 - 1
    right = torch.rand_like(left, requires_grad=True) * 2 - 1
    actual = native(left, right)
    expected = official(left, right)
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-5)
    actual_grads = torch.autograd.grad(actual.sum(), (left, right), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (left, right))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=3e-7, rtol=3e-5)
