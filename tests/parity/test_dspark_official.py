import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from typing import Callable, Optional, Tuple
import urllib.request

import pytest
import torch
from torch import nn

from aster.models import Qwen3Config
from aster.models.dspark import DSparkConfig, DSparkDraft, sample_anchors, block_attention_mask
from aster.nn.markov import MarkovHead


COMMIT = "005e03b81cec38b7da6399833d609ee89a2587f2"
HASHES = {
    "deepspec/modeling/dspark/common.py": "45273cf6c806cdb1ad249df3e3b4d4dd34be47e74ae342608f60ede39febd088",
    "deepspec/modeling/dspark/markov_head.py": "6659bcdc12d923d4fc16cc2280c03078c2110a4792305e1f3f42b5468f75ef46",
    "deepspec/modeling/dspark/qwen3/modeling.py": "9ad90bcb926b32b5144e8a27816ef18b51e9e5ee083108c5fbd8f63bc92b3953",
    "deepspec/modeling/dspark/loss.py": "2e91efcaff780eec0748ef3f6f0a31374f119f609c664cc79289fdd922335328",
    "deepspec/eval/dspark/draft_ops.py": "9d07a301fa643ee3b558a093647ff3db2918a47c96267d3546081eb9df44b799",
}
pytestmark = pytest.mark.skipif(
    os.environ.get("ASTER_RUN_REMOTE_DSPARK_ORACLE") != "1",
    reason="Pinned DSpark source execution requires explicit network opt-in",
)


@lru_cache(None)
def definitions(path, names):
    url = f"https://raw.githubusercontent.com/deepseek-ai/DeepSpec/{COMMIT}/{path}"
    source = urllib.request.urlopen(url, timeout=20).read()
    assert hashlib.sha256(source).hexdigest() == HASHES[path]
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(nodes) == len(names)
    return compile(ast.Module(body=nodes, type_ignores=[]), url, "exec")


def common_scope():
    scope = dict(torch=torch, nn=nn, Optional=Optional, dataclass=dataclass)
    names = (
        "DSparkForwardOutput",
        "AcceptRatePredictor",
        "build_anchor_candidate_mask",
        "sample_anchor_positions",
        "create_position_ids",
        "create_noise_embed",
        "build_eval_mask",
        "create_dspark_attention_mask",
    )
    exec(definitions("deepspec/modeling/dspark/common.py", names), scope)
    exec(
        definitions(
            "deepspec/modeling/dspark/markov_head.py",
            ("VanillaMarkov", "GatedMarkovHead", "RNNHead", "build_markov_head"),
        ),
        scope,
    )
    return scope


@pytest.mark.parametrize("kind", ["vanilla", "gated", "rnn"])
def test_actual_dspark_markov_head_teacher_forced_full_gradient(kind):
    torch.set_num_threads(1)
    torch.manual_seed(197)
    scope = common_scope()
    native = MarkovHead(23, 5, 8, kind)
    cls = scope[{"vanilla": "VanillaMarkov", "gated": "GatedMarkovHead", "rnn": "RNNHead"}[kind]]
    settings = dict(vocab_size=23, markov_rank=5)
    if kind != "vanilla":
        settings["hidden_size"] = 8
    official = cls(**settings)
    official.load_state_dict(native.state_dict())
    hidden = torch.randn(2, 3, 4, 8, requires_grad=True)
    tokens = torch.randint(23, (2, 3, 4))
    base = torch.randn(2, 3, 4, 23)
    actual = native(hidden, tokens) + base
    expected = official.apply_block_logits(base, token_ids=tokens, hidden_states=hidden)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    inputs_a = (hidden, *native.parameters()) if kind != "vanilla" else tuple(native.parameters())
    inputs_b = (
        (hidden, *official.parameters()) if kind != "vanilla" else tuple(official.parameters())
    )
    a = torch.autograd.grad(actual.square().sum(), inputs_a, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), inputs_b)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)


def test_actual_dspark_sampler_and_mask_source():
    scope = common_scope()
    mask = torch.tensor([[1, 1, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
    torch.manual_seed(9)
    actual = sample_anchors(mask, 9)
    torch.manual_seed(9)
    expected = scope["sample_anchor_positions"](
        seq_len=7, loss_mask=mask, num_anchors=9, device=mask.device
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def dense_mask(mask_mod, *, B, H, Q_LEN, KV_LEN, device):
        return mask_mod(
            torch.arange(B)[:, None, None],
            0,
            torch.arange(Q_LEN)[None, :, None],
            torch.arange(KV_LEN)[None, None, :],
        )[:, None]

    scope["create_block_mask"] = dense_mask
    actual_mask = block_attention_mask(*actual, 7, 3)
    expected_mask = scope["create_dspark_attention_mask"](
        anchor_positions=expected[0],
        block_keep_mask=expected[1],
        seq_len=7,
        block_size=3,
        device=mask.device,
    )
    assert torch.equal(actual_mask, expected_mask)


@pytest.mark.parametrize("kind", ["gated", "rnn"])
def test_actual_complete_qwen3_dspark_logits_confidence_and_all_trainable_gradients(kind):
    pytest.importorskip("transformers")
    from transformers import Qwen3Config as OfficialConfig
    from transformers.models.qwen3 import modeling_qwen3 as qwen
    from typing_extensions import Unpack

    torch.set_num_threads(1)
    torch.manual_seed(91)
    scope = common_scope()
    scope.update(
        {
            name: getattr(qwen, name)
            for name in (
                "Qwen3MLP",
                "Qwen3PreTrainedModel",
                "Qwen3RMSNorm",
                "Qwen3RotaryEmbedding",
                "eager_attention_forward",
                "rotate_half",
                "GradientCheckpointingLayer",
                "ALL_ATTENTION_FUNCTIONS",
                "FlashAttentionKwargs",
            )
        }
    )
    scope.update(
        Callable=Callable,
        Tuple=Tuple,
        Unpack=Unpack,
        Cache=qwen.Cache,
        log_sampler_stats=lambda **_: None,
    )

    def additive_mask(mask_mod, *, B, H, Q_LEN, KV_LEN, device):
        visible = mask_mod(
            torch.arange(B, device=device)[:, None, None],
            0,
            torch.arange(Q_LEN, device=device)[None, :, None],
            torch.arange(KV_LEN, device=device)[None, None, :],
        )[:, None]
        return torch.zeros(visible.shape, device=device).masked_fill(~visible, float("-inf"))

    scope["create_block_mask"] = additive_mask
    names = (
        "apply_rotary_pos_emb",
        "Qwen3DSparkAttention",
        "Qwen3DSparkDecoderLayer",
        "Qwen3DSparkModel",
    )
    exec(definitions("deepspec/modeling/dspark/qwen3/modeling.py", names), scope)
    target = Qwen3Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config = DSparkConfig(
        target,
        num_draft_layers=2,
        target_layer_ids=(-1, 1),
        block_size=3,
        num_anchors=2,
        markov_rank=4,
        markov_head_type=kind,
    )
    official_config = OfficialConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_theta=target.rope.theta,
        rms_norm_eps=target.rms_norm_eps,
        attention_bias=False,
        tie_word_embeddings=False,
        layer_types=["full_attention"] * 2,
    )
    for key in (
        "target_layer_ids",
        "block_size",
        "mask_token_id",
        "num_anchors",
        "markov_rank",
        "markov_head_type",
        "enable_confidence_head",
        "confidence_head_with_markov",
    ):
        setattr(official_config, key, getattr(config, key))
    official_config._attn_implementation = "eager"
    official = scope["Qwen3DSparkModel"](official_config)
    official.set_embedding_head_trainable(False)

    from aster.models import CausalLM
    from aster.models.dspark_import import import_dspark_state

    target_model = CausalLM(target).eval()
    with torch.no_grad():
        target_model.get_input_embeddings().weight.copy_(official.embed_tokens.weight)
        target_model.lm_head.weight.copy_(official.lm_head.weight)
    native = import_dspark_state(config, official.state_dict(), target=target_model)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 7, 6, 5, 4, 3, 2]])
    loss_mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1]])
    anchors, keep = torch.tensor([[0, 3], [1, 3]]), torch.ones(2, 2, dtype=torch.bool)
    scope["sample_anchor_positions"] = lambda **_: (anchors, keep)
    context, last = torch.randn(2, 7, 32), torch.randn(2, 7, 16)
    actual = native(ids, context, loss_mask, last, anchor_positions=anchors, block_keep_mask=keep)
    expected = official(ids, context, loss_mask, last)
    for field in (
        "draft_logits",
        "confidence_pred",
        "aligned_target_logits",
        "target_ids",
        "eval_mask",
        "block_keep_mask",
    ):
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), atol=3e-7, rtol=2e-5
        )
    native_parameters = {name: p for name, p in native.named_parameters() if p.requires_grad}
    official_parameters = {
        name.replace("confidence_head.proj.", "confidence_head."): p
        for name, p in official.named_parameters()
        if p.requires_grad
    }
    assert set(native_parameters) == set(official_parameters)
    names = sorted(native_parameters)
    a = torch.autograd.grad(
        actual.draft_logits.square().sum() + actual.confidence_pred.square().sum(),
        tuple(native_parameters[name] for name in names),
    )
    b = torch.autograd.grad(
        expected.draft_logits.square().sum() + expected.confidence_pred.square().sum(),
        tuple(official_parameters[name] for name in names),
    )
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=5e-6, rtol=5e-5)


@pytest.mark.parametrize("gamma", [None, 0.0, 4.0])
def test_actual_dspark_weighted_ce_l1_and_detached_confidence_loss(gamma):
    from aster.models.dspark import DSparkOutput
    from aster.methods.dspark import dspark_loss_terms
    import torch.nn.functional as F

    torch.set_num_threads(1)
    torch.manual_seed(411)
    scope = common_scope()
    scope.update(F=F, add_metric=lambda *args, **kwargs: None)
    names = (
        "_build_loss_weight_mask",
        "_compute_local_probabilistic_stats",
        "_compute_accept_rate_3d",
        "_compute_local_l1_term",
        "_collect_local_terms",
        "_build_loss",
    )
    exec(definitions("deepspec/modeling/dspark/loss.py", names), scope)
    logits = torch.randn(2, 3, 4, 19, requires_grad=True)
    confidence = torch.randn(2, 3, 4, requires_grad=True)
    keep = torch.tensor([[True, True, False], [True, False, False]])
    mask = torch.rand(2, 3, 4) > 0.2
    mask = mask.int().cumprod(-1).bool() & keep[:, :, None]
    output = DSparkOutput(
        logits,
        torch.randint(19, (2, 3, 4)),
        mask,
        keep,
        confidence,
        torch.randn_like(logits),
        torch.zeros(2, 3, dtype=torch.long),
    )
    native = dspark_loss_terms(output, decay_gamma=gamma)
    terms, has_confidence = scope["_collect_local_terms"](
        outputs=output, loss_decay_gamma=gamma, l1_loss_alpha=0.9
    )
    expected = scope["_build_loss"](
        loss_terms=terms,
        global_denominators={
            key: terms[key] for key in ("ce_loss_den", "l1_loss_den", "confidence_loss_den")
        },
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
        has_confidence=has_confidence,
        world_size=1,
    )
    actual = sum(t.numerator / t.denominator * t.weight for t in native.terms)
    torch.testing.assert_close(actual.float(), expected, atol=3e-7, rtol=3e-7)
    for a, b in zip(
        torch.autograd.grad(actual, (logits, confidence), retain_graph=True),
        torch.autograd.grad(expected, (logits, confidence)),
    ):
        torch.testing.assert_close(a, b, atol=2e-8, rtol=3e-6)


def test_actual_dspark_confidence_prefix_threshold_source():
    from aster.inference.dspark import confident_prefix_length

    scope = {"torch": torch}
    exec(definitions("deepspec/eval/dspark/draft_ops.py", ("_confident_prefix_length",)), scope)
    for values in ([0.8, 0.8, 0.8], [0.8, 0.6, 0.9], [0.2, 0.8, 0.9]):
        logits = torch.logit(torch.tensor(values))
        for threshold in (0.0, 0.3, 0.7, 1.0):
            assert confident_prefix_length(logits, threshold) == scope["_confident_prefix_length"](
                logits[None], block_size=3, threshold=threshold
            )
