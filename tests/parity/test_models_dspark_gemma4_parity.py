import ast
from dataclasses import replace
from functools import lru_cache, wraps
import hashlib
import os
from types import SimpleNamespace
from typing import Callable, Optional
import urllib.request
import pytest
import torch
from aster.models.gemma4 import Gemma4TextConfig
from aster.models.dspark_gemma4 import Gemma4DSparkConfig, Gemma4DSparkDraft


pytestmark = pytest.mark.skipif(
    os.environ.get("ASTER_RUN_REMOTE_DSPARK_ORACLE") != "1",
    reason="Pinned DSpark source requires explicit network opt-in",
)
COMMIT = "005e03b81cec38b7da6399833d609ee89a2587f2"
HASHES = {
    "config.py": "c198c38470bf2021beaead92148ec3fd5bbdefa98f6bd5c4667abd16f1be651a",
    "modeling.py": "173d7e81cda3f7a48bb9b633584a53e189f1c68008b6e083df862f7f402d641e",
}
TF_HASHES = {
    "models/gemma4/modeling_gemma4.py": "7b84c5d7dbe57a37b81fa6f28b0b4600c5a304cbfeccb2daf9db842d5a456536",
    "modeling_rope_utils.py": "f370d8588169c07fc5245ddb83d9a29282f670f98d99a4d7fa814a777649f3d5",
}


@lru_cache(None)
def definitions(path):
    url = f"https://raw.githubusercontent.com/deepseek-ai/DeepSpec/{COMMIT}/deepspec/modeling/dspark/gemma4/{path}"
    raw = urllib.request.urlopen(url, timeout=25).read()
    assert hashlib.sha256(raw).hexdigest() == HASHES[path]
    nodes = [
        node for node in ast.parse(raw).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    return compile(ast.Module(body=nodes, type_ignores=[]), url, "exec")


@lru_cache(None)
def tf_definitions(path, names):
    url = f"https://raw.githubusercontent.com/huggingface/transformers/v5.10.2/src/transformers/{path}"
    raw = urllib.request.urlopen(url, timeout=25).read()
    assert hashlib.sha256(raw).hexdigest() == TF_HASHES[path]
    nodes = [
        node
        for node in ast.parse(raw).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(nodes) == len(names)
    return compile(ast.Module(body=nodes, type_ignores=[]), url, "exec")


def actual_tf510_math(scope, tf):
    from transformers.models.gemma4 import modeling_gemma4 as current
    from transformers.activations import ACT2FN
    import math

    scope.update(
        {
            name: getattr(current, name)
            for name in (
                "Gemma4VisionPatchEmbedder",
                "Gemma4AudioRelPositionalEncoding",
                "Gemma4AudioAttention",
                "Gemma4VisionRotaryEmbedding",
                "Gemma4TextRouter",
                "Gemma4TextExperts",
                "Gemma4TextDecoderLayer",
                "Gemma4ClippableLinear",
                "Gemma4VisionModel",
            )
        }
    )
    scope.update(
        Gemma4TextConfig=tf.Gemma4TextConfig,
        Gemma4Config=tf.Gemma4Config,
        PreTrainedModel=tf.PreTrainedModel,
        PreTrainedConfig=tf.PreTrainedConfig,
        Callable=Callable,
        Optional=Optional,
        wraps=wraps,
        ACT2FN=ACT2FN,
        maybe_autocast=current.maybe_autocast,
        init=current.init,
        math=math,
        auto_docstring=lambda cls: cls,
    )
    exec(
        tf_definitions(
            "modeling_rope_utils.py",
            ("_compute_proportional_rope_parameters", "dynamic_rope_update"),
        ),
        scope,
    )
    scope["ROPE_INIT_FUNCTIONS"] = {"proportional": scope["_compute_proportional_rope_parameters"]}
    exec(
        tf_definitions(
            "models/gemma4/modeling_gemma4.py",
            (
                "rotate_half",
                "apply_rotary_pos_emb",
                "Gemma4PreTrainedModel",
                "Gemma4RMSNorm",
                "Gemma4TextMLP",
                "Gemma4TextRotaryEmbedding",
                "Gemma4TextScaledWordEmbedding",
            ),
        ),
        scope,
    )


@pytest.mark.parametrize(
    "variant",
    [
        "shared_source_mlp",
        "positive_shared_threshold",
        "independent_kv",
        "softcap_rnn",
        "vanilla_bias",
        "trainable_head",
    ],
)
def test_models_dspark_gemma4_actual_config_model_logits_and_all_gradients(variant):
    import copy
    import test_dspark_official as shared
    from test_models_gemma4_parity import oracle_config

    tf = pytest.importorskip("transformers")
    from transformers.modeling_layers import GradientCheckpointingLayer

    torch.set_num_threads(1)
    torch.manual_seed(751)
    target = Gemma4TextConfig(
        hidden_size_per_layer_input=0,
        hidden_size=16,
        intermediate_size=32,
        head_dim=4,
        global_head_dim=8,
        global_rotary_fraction=0.5,
    )
    if variant == "independent_kv":
        target = replace(
            target, attention_k_eq_v=False, num_kv_shared_layers=0, global_rope_factor=2.0
        )
    if variant == "softcap_rnn":
        target = replace(target, final_logit_softcapping=0.07)
    if variant == "vanilla_bias":
        target = replace(target, attention_bias=True)
    if variant == "positive_shared_threshold":
        target = replace(target, num_kv_shared_layers=1)
    config = Gemma4DSparkConfig(
        target,
        num_draft_layers=2,
        target_layer_ids=(-1, 1),
        block_size=3,
        num_anchors=3,
        markov_rank=4,
        markov_head_type="rnn"
        if variant == "softcap_rnn"
        else ("vanilla" if variant == "vanilla_bias" else "gated"),
        freeze_embedding_head=variant != "trainable_head",
    )
    scope = shared.common_scope()
    exec(
        shared.definitions("deepspec/modeling/dspark/common.py", ("validate_target_layer_ids",)),
        scope,
    )
    scope.update(copy=copy, TRAIN_ATTN_IMPLEMENTATION="flex_attention")
    exec(definitions("config.py"), scope)

    class Args(dict):
        __getattr__ = dict.__getitem__

    args = Args(config.to_dict())
    args["confidence_head_alpha"] = 1.0
    source_config = oracle_config(tf, target)

    source_config.global_head_dim = target.global_head_dim
    source_config.num_global_key_value_heads = target.num_global_key_value_heads
    source_config.allow_global_per_layer_attribute_access = True
    official_config = scope["build_draft_config"](
        SimpleNamespace(model_type="gemma4", text_config=source_config), args
    )
    assert official_config.layer_types == ["full_attention"] * 2
    assert official_config.num_kv_shared_layers == target.num_kv_shared_layers

    official_config._attn_implementation = "eager"
    actual_tf510_math(scope, tf)
    scope.update(
        Gemma4TextConfig=tf.Gemma4TextConfig,
        GradientCheckpointingLayer=GradientCheckpointingLayer,
        apply_gemma4_rotary_pos_emb=scope["apply_rotary_pos_emb"],
        F=torch.nn.functional,
        Cache=object,
        Optional=Optional,
        log_sampler_stats=lambda **_: None,
    )

    def dense_mask(mask_mod, *, B, H, Q_LEN, KV_LEN, device):
        return mask_mod(
            torch.arange(B, device=device)[:, None, None],
            0,
            torch.arange(Q_LEN, device=device)[None, :, None],
            torch.arange(KV_LEN, device=device)[None, None, :],
        )[:, None]

    scope["create_block_mask"] = dense_mask
    exec(definitions("modeling.py"), scope)
    official = scope["Gemma4DSparkModel"](official_config)
    official.set_embedding_head_trainable(not config.freeze_embedding_head)
    native = Gemma4DSparkDraft(config)
    state = {
        name.replace("confidence_head.proj.", "confidence_head."): value
        for name, value in official.state_dict().items()
    }
    state.update(
        teacher_weights_loaded=torch.tensor(True),
        teacher_fingerprint=native.teacher_fingerprint.clone(),
    )
    native.load_state_dict(state, strict=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 7, 6, 5, 4, 3, 2]])
    loss_mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1]])
    anchors = torch.tensor([[0, 3, 0], [1, 3, 4]])
    keep = torch.tensor([[True, True, False], [True, True, True]])
    scope["sample_anchor_positions"] = lambda **_: (anchors, keep)
    features, last = torch.randn(2, 7, 32), torch.randn(2, 7, 16)
    if not config.freeze_embedding_head:
        last = None
    actual = native(ids, features, loss_mask, last, anchor_positions=anchors, block_keep_mask=keep)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected = official(ids, features, loss_mask, last)
    for field in (
        "draft_logits",
        "confidence_pred",
        "aligned_target_logits",
        "target_ids",
        "eval_mask",
        "block_keep_mask",
    ):
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), atol=7e-7, rtol=5e-5, msg=field
        )
    ng = {name: p for name, p in native.named_parameters() if p.requires_grad}
    og = {
        name.replace("confidence_head.proj.", "confidence_head."): p
        for name, p in official.named_parameters()
        if p.requires_grad
    }
    assert set(ng) == set(og)
    names = sorted(ng)
    a = torch.autograd.grad(
        actual.draft_logits.square().sum() + actual.confidence_pred.square().sum(),
        tuple(ng[name] for name in names),
    )
    b = torch.autograd.grad(
        expected.draft_logits.square().sum() + expected.confidence_pred.square().sum(),
        tuple(og[name] for name in names),
    )
    for name, left, right in zip(names, a, b):
        torch.testing.assert_close(
            left, right, atol=6e-6, rtol=9e-5, msg=lambda detail: f"{name}: {detail}"
        )
    assert (
        native.layers[0].mlp.gate_proj.weight.shape == official.layers[0].mlp.gate_proj.weight.shape
    )

    native.eval()
    official.eval()
    noise = ids[:, : config.block_size]
    with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        _, past = native.backbone_cached(noise, features[:, :2])
        cached, retained = native.backbone_cached(noise, features[:, 2:5], state=past)
        position_ids = torch.arange(5 + config.block_size)[None].expand(2, -1)
        author_hidden = official._forward_backbone(
            position_ids=position_ids,
            noise_embedding=official.embed_tokens(noise),
            target_hidden_states=features[:, :5],
            attention_mask=torch.ones(
                2, 1, config.block_size, 5 + config.block_size, dtype=torch.bool
            ),
        )
    torch.testing.assert_close(cached, author_hidden, atol=3e-6, rtol=5e-5)
    assert all(value.shape[2] == 5 for pair in retained for value in pair)
