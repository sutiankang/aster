from dataclasses import asdict
import ast
import functools
import hashlib
import os
import urllib.request
import pytest
import torch
from aster.models.vit import ViTConfig, ViTModel
from aster.nn.normalization import BatchNorm1d
from aster.nn.parameter_codec import public_parameter_names


@pytest.mark.oracle
@pytest.mark.parametrize("shape,amp", [((16, 16), False), ((20, 12), False), ((16, 16), True)])
def test_models_lewm_vit_actual_transformers_same_weights_all_gradients(shape, amp):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(720)
    c = ViTConfig()
    native = ViTModel(c)
    tc = tf.ViTConfig(**asdict(c))
    tc._attn_implementation = "eager"
    official = tf.ViTModel(tc, add_pooling_layer=False, use_mask_token=False)
    official.load_state_dict(native.state_dict(), strict=True)
    x = torch.randn(2, 3, *shape, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        actual = native(x, interpolate_pos_encoding=True).last_hidden_state
        expected = official(x, interpolate_pos_encoding=True).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=4e-5)
    coefficient = torch.randn_like(actual) / actual.numel()
    ng = torch.autograd.grad(
        (actual * coefficient).sum(), (x, *native.parameters()), retain_graph=True
    )
    og = torch.autograd.grad((expected * coefficient).sum(), (x, *official.parameters()))

    torch.testing.assert_close(ng[0], og[0], atol=2e-6, rtol=3e-4)
    expected_grads = dict(zip(dict(official.named_parameters()), og[1:]))
    names = public_parameter_names(native)
    for (name, _), value in zip(native.named_parameters(), ng[1:]):
        torch.testing.assert_close(
            value, expected_grads[names[name]], atol=3e-6, rtol=6e-4, msg=name
        )


def test_models_lewm_batchnorm_actual_torch_formula_stats_and_all_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(721)
    native, official = BatchNorm1d(5).double(), torch.nn.BatchNorm1d(5).double()
    with torch.no_grad():
        official.weight.normal_()
        official.bias.normal_()
    native.load_state_dict(official.state_dict(), strict=True)
    for _ in range(3):
        x = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
        a, b = native(x), official(x)
        torch.testing.assert_close(a, b, atol=2e-15, rtol=2e-14)
        ga = torch.autograd.grad(a.square().sum(), (x, *native.parameters()), retain_graph=True)
        gb = torch.autograd.grad(b.square().sum(), (x, *official.parameters()))
        for left, right in zip(ga, gb):
            torch.testing.assert_close(left, right, atol=3e-13, rtol=3e-11)
        for name, value in native.state_dict().items():
            torch.testing.assert_close(value, official.state_dict()[name], atol=0, rtol=0)


@functools.lru_cache(None)
def _lewm_source():

    def rearrange(x, pattern, **axes):
        if pattern == "b t (h d) -> b h t d":
            return x.reshape(x.shape[0], x.shape[1], axes["h"], -1).transpose(1, 2)
        if pattern == "b h t d -> b t (h d)":
            return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        if pattern in ("b t ... -> (b t) ...", "b t d -> (b t) d", "b s ... -> (b s) ..."):
            return x.flatten(0, 1)
        if pattern == "(b t) d -> b t d":
            return x.reshape(axes["b"], -1, x.shape[-1])
        if pattern == "(b s) ... -> b s ...":
            return x.reshape(axes["b"], axes["s"], *x.shape[1:])
        raise AssertionError("Unexpected source layout: " + pattern)

    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, rearrange=rearrange)
    hashes = {
        "module.py": "0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620",
        "jepa.py": "41bad7fd21e0f14aea4c9c3d39a9c87037e787746d953ab62cdc0677e938ce96",
    }
    for path, digest in hashes.items():
        url = (
            "https://raw.githubusercontent.com/lucas-maes/le-wm/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/"
            + path
        )
        raw = urllib.request.urlopen(url, timeout=25).read()
        assert hashlib.sha256(raw).hexdigest() == digest
        nodes = [
            node
            for node in ast.parse(raw).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), url, "exec"), scope)
    return scope


remote = pytest.mark.skipif(
    os.environ.get("ASTER_RUN_REMOTE_LEWM_ORACLE") != "1",
    reason="Pinned author source execution requires explicit opt-in",
)


@remote
def test_models_lewm_actual_author_whole_model_all_parameter_input_and_target_gradients():
    from aster.models.lewm import LeWMConfig, LeWorldModel

    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(722)
    c = LeWMConfig()
    native = LeWorldModel(c)

    with torch.no_grad():
        for name, parameter in native.named_parameters():
            if "adaLN_modulation.1" in name:
                parameter.normal_(0, 0.1)
    scope = _lewm_source()
    tc = tf.ViTConfig(**asdict(c.encoder))
    tc._attn_implementation = "eager"
    official = scope["JEPA"](
        tf.ViTModel(tc, add_pooling_layer=False, use_mask_token=False),
        scope["ARPredictor"](
            num_frames=c.history_size,
            depth=c.predictor_depth,
            heads=c.predictor_heads,
            mlp_dim=c.predictor_mlp_dim,
            input_dim=c.embed_dim,
            hidden_dim=c.predictor_hidden_dim,
            dim_head=c.predictor_head_dim,
        ),
        scope["Embedder"](
            input_dim=c.action_dim,
            smoothed_dim=c.action_smoothed_dim,
            emb_dim=c.embed_dim,
            mlp_scale=c.action_mlp_scale,
        ),
        scope["MLP"](
            c.encoder.hidden_size, c.projector_hidden_dim, c.embed_dim, norm_fn=torch.nn.BatchNorm1d
        ),
        scope["MLP"](
            c.embed_dim, c.projector_hidden_dim, c.embed_dim, norm_fn=torch.nn.BatchNorm1d
        ),
    )
    official.load_state_dict(native.state_dict(), strict=True)
    pixels = torch.randn(3, 4, 3, 16, 16, requires_grad=True)
    actions = torch.randn(3, 3, 2, requires_grad=True)
    output = native(pixels, actions)
    source = official.encode(dict(pixels=pixels, action=actions))
    predictions = official.predict(source["emb"][:, :-1], source["act_emb"])
    torch.testing.assert_close(output.embeddings, source["emb"], atol=2e-6, rtol=4e-5)
    torch.testing.assert_close(output.predictions, predictions, atol=3e-6, rtol=5e-5)
    native_loss = (output.predictions - output.embeddings[:, 1:]).square().mean()
    source_loss = (predictions - source["emb"][:, 1:]).square().mean()
    target_gradient = torch.autograd.grad(native_loss, output.embeddings, retain_graph=True)[0]
    assert target_gradient[:, -1].abs().sum() > 0
    ng = torch.autograd.grad(
        native_loss, (pixels, actions, *native.parameters()), retain_graph=True
    )
    og = torch.autograd.grad(source_loss, (pixels, actions, *official.parameters()))
    for a, b in zip(ng[:2], og[:2]):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=7e-4)
    names = public_parameter_names(native)
    official_grad = dict(zip(dict(official.named_parameters()), og[2:]))
    for (name, _), value in zip(native.named_parameters(), ng[2:]):
        torch.testing.assert_close(
            value, official_grad[names[name]], atol=3e-6, rtol=7e-4, msg=name
        )
    for name, value in native.state_dict().items():
        torch.testing.assert_close(
            value, official.state_dict()[name], atol=3e-7, rtol=3e-5, msg=name
        )
    native.eval()
    official.eval()
    candidates = torch.randn(2, 4, 6, c.action_dim)
    initial = torch.randn(2, 3, 3, 16, 16)
    actual = native.rollout_latents(native.encode(initial), candidates)
    expected = official.rollout(
        {"pixels": initial[:, None].expand(2, 4, 3, 3, 16, 16)}, candidates
    )["predicted_emb"]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=5e-5)


@remote
def test_models_lewm_sigreg_actual_author_statistic_rng_and_gradient():
    from aster.methods.lewm import SIGReg

    torch.set_num_threads(1)
    torch.manual_seed(723)
    value = torch.randn(4, 9, 8, requires_grad=True)
    native = SIGReg(knots=17, num_proj=32)
    official = _lewm_source()["SIGReg"](knots=17, num_proj=32)
    torch.manual_seed(725)
    actual = native(value)
    torch.manual_seed(725)
    expected = official(value)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    ga = torch.autograd.grad(actual, value, retain_graph=True)[0]
    gb = torch.autograd.grad(expected, value)[0]
    torch.testing.assert_close(ga, gb, atol=0, rtol=0)


@remote
def test_models_lewm_cem_actual_stable_worldmodel_solver_every_candidate_and_warmstart():
    import time
    import types
    import numpy as np
    from typing import Any
    from aster.models.lewm import LeWorldModel
    from aster.planning.lewm import LeWMCEM, LeWMCEMConfig

    torch.set_num_threads(1)
    torch.manual_seed(738)
    native = LeWorldModel().eval()
    url = "https://raw.githubusercontent.com/galilai-group/stable-worldmodel/3a85ac6888c39db90af648993fd0b23ac4c0a51d/stable_worldmodel/solver/cem.py"
    raw = urllib.request.urlopen(url, timeout=25).read()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "d88c86dcd1bd1e6d89221ac22079a3efe296cc7566532de0d36605e2f1536050"
    )
    nodes = [node for node in ast.parse(raw).body if isinstance(node, ast.ClassDef)]
    scope = dict(
        torch=torch,
        np=np,
        Any=Any,
        Costable=object,
        gym=types.SimpleNamespace(Space=object),
        time=time,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), url, "exec"), scope)
    calls = []

    class Cost:
        def get_cost(self, info, candidates):
            calls.append(candidates.clone())
            history = native.encode(info["pixels"][:, 0])
            goal = native.encode(info["goal"][:, 0])[:, -1]
            return (
                (native.rollout_latents(history, candidates)[:, :, -1] - goal[:, None])
                .square()
                .sum(-1)
            )

    config = LeWMCEMConfig(
        horizon=4, num_samples=12, topk=4, n_steps=3, batch_size=2, initial_std=0.7
    )
    traced = []
    original_rollout = native.rollout_latents

    def trace(history, candidates):
        traced.append(candidates.clone())
        return original_rollout(history, candidates)

    native.rollout_latents = trace
    ours = LeWMCEM(native, config, seed=739)
    source = scope["CEMSolver"](
        Cost(), batch_size=2, num_samples=12, topk=4, n_steps=3, var_scale=0.7, seed=739
    )
    source._n_envs, source._action_dim = 3, 2
    source._config = types.SimpleNamespace(horizon=4, action_block=1)
    pixels, goals = torch.randn(3, 2, 3, 16, 16), torch.randn(3, 1, 3, 16, 16)
    prefix = torch.randn(3, 2, 2) * 0.2
    for initial in (None, prefix):
        traced.clear()
        actual = ours.solve(pixels, goals, init_action=initial)
        ours_candidates = list(traced)
        traced.clear()
        expected = source.solve(dict(pixels=pixels, goal=goals), init_action=initial)
        assert len(ours_candidates) == len(traced) == 6
        for a, b in zip(ours_candidates, traced):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(actual.actions, expected["actions"], atol=0, rtol=0)
        torch.testing.assert_close(actual.std, expected["var"][0], atol=0, rtol=0)
        torch.testing.assert_close(
            actual.elite_cost, torch.tensor(expected["costs"]), atol=0, rtol=0
        )

    assert any((item.abs() > 1).any() for item in calls)
