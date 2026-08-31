"""

Wan TeaCache source comparisons.
Reference source Copyright 2024-2025 Alibaba Wan Team.
TeaCache repository: Apache-2.0."""

import ast
import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aster.models.video_world import WanVideoConfig, WanVideoDiT
from aster.optimization.wan_teacache import (
    TEACACHE_SOURCE,
    WanCacheSampler,
    WanTeaCacheSettings,
    WanTeaCacheSession,
    calibrate_wan_teacache,
)


@pytest.mark.skipif(
    os.environ.get("ASTER_RUN_TEACACHE_SOURCE_ORACLE") != "1",
    reason="explicit pinned official-source execution not enabled",
)
@pytest.mark.parametrize("mode", ["default", "retention"])
def test_exact_upstream_cache_blocks_against_native_session(mode):
    import requests

    client = requests.Session()
    client.trust_env = False
    response = client.get(
        "https://raw.githubusercontent.com/ali-vilab/TeaCache/"
        + TEACACHE_SOURCE["commit"]
        + "/"
        + TEACACHE_SOURCE["path"],
        timeout=30,
    )
    response.raise_for_status()
    assert hashlib.sha256(response.content).hexdigest() == TEACACHE_SOURCE["sha256"]
    tree = ast.parse(response.text)
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "teacache_forward"
    )
    cache_blocks = [
        n
        for n in function.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Attribute)
        and n.test.attr == "enable_teacache"
    ]
    assert len(cache_blocks) == 2
    program = compile(
        ast.fix_missing_locations(ast.Module(body=cache_blocks, type_ignores=[])),
        "<pinned-official-TeaCache-cache-blocks>",
        "exec",
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(732)
        model = WanVideoDiT(
            WanVideoConfig(
                latent_channels=2,
                hidden_size=12,
                intermediate_size=24,
                num_heads=2,
                num_layers=2,
                text_dim=4,
                text_length=3,
                frequency_dim=4,
            )
        ).eval()
        with torch.no_grad():
            model.head.head.weight.normal_(std=0.1)
        noise = torch.randn(1, 2, 3, 2, 2)
        positive, negative = {"text": torch.randn(1, 2, 4)}, {"text": torch.randn(1, 2, 4)}
        sampler = WanCacheSampler(steps=8, guidance_scale=2.0)
        fitted = calibrate_wan_teacache(
            model,
            [
                {
                    "id": "actual-native-trajectory",
                    "noise": noise,
                    "positive": positive,
                    "negative": negative,
                }
            ],
            policy_artifact_id="a" * 64,
            dataset_fingerprint="b" * 64,
            sampler=sampler,
            mode=mode,
        )
        native = WanTeaCacheSession(
            model,
            policy_artifact_id="a" * 64,
            sampler=sampler,
            condition=positive,
            negative_condition=negative,
            calibration=fitted,
            settings=WanTeaCacheSettings(
                threshold=1e8, mode=mode, maximum_relative_output_error=1e8
            ),
        )
        official = SimpleNamespace(
            enable_teacache=True,
            use_ref_steps=mode == "retention",
            cnt=0,
            ret_steps=2 if mode == "default" else 10,
            cutoff_steps=14 if mode == "default" else 16,
            coefficients=fitted.coefficients,
            teacache_thresh=1e8,
            accumulated_rel_l1_distance_even=0.0,
            accumulated_rel_l1_distance_odd=0.0,
            previous_e0_even=None,
            previous_e0_odd=None,
            previous_residual_even=None,
            previous_residual_odd=None,
        )
        with torch.inference_mode():
            for index, sigma in enumerate(sampler.evaluation_times("cpu")):
                value = noise + index * 0.01
                for branch, condition in (("positive", positive), ("negative", negative)):
                    prepared = model.prepare(value, sigma.expand(1), condition)
                    official.blocks = [
                        lambda x, _block=block, **kw: _block(
                            x, kw["e"], prepared.grid, prepared.text, prepared.image
                        )
                        for block in model.blocks
                    ]
                    namespace = {
                        "self": official,
                        "np": np,
                        "e": prepared.embedding,
                        "e0": prepared.modulation,
                        "x": prepared.hidden.clone(),
                        "kwargs": {"e": prepared.modulation},
                    }
                    exec(program, namespace)
                    expected = model.finish(namespace["x"], prepared).prediction
                    actual = native.predict(value, round_index=index, branch=branch).prediction
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    state = native._states[branch]
                    suffix = "even" if branch == "positive" else "odd"
                    assert state["accumulated"] == getattr(
                        official, "accumulated_rel_l1_distance_" + suffix
                    )
                    torch.testing.assert_close(
                        state["residual"],
                        getattr(official, "previous_residual_" + suffix),
                        atol=0,
                        rtol=0,
                    )
                    official.cnt += 1
        assert native.reused_backbone_calls == (12 if mode == "default" else 6)
    finally:
        torch.set_num_threads(previous_threads)
