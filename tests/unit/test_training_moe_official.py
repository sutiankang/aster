import ast
import hashlib
import os
from typing import Optional
import urllib.request

import pytest
import torch
from torch.nn import functional as F

from aster.core import TokenOutput
from aster.models import MixtralConfig, build_model
from aster.training import (
    Trainer,
    ParallelContext,
    parallelize_mixtral,
    ExpertParallelCrossEntropyObjective,
)
from aster.training.portable import logical_tensors, optimizer_mapping, gather_tensor


@pytest.mark.oracle
def test_complete_zero3_mixtral_matches_actual_transformers_gradients():
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(910)
    config = MixtralConfig(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        sliding_window=3,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    native = build_model(config)
    values = config.to_dict()
    values.pop("architecture")
    values.pop("rope")
    official_config = transformers.MixtralConfig(
        **values, rope_theta=config.rope.theta, pad_token_id=None
    )
    official_config._attn_implementation = "eager"
    official = transformers.MixtralForCausalLM(official_config)
    official.load_state_dict(native.state_dict(), strict=True)
    context = ParallelContext()
    model = parallelize_mixtral(native, context)
    engine = Trainer(
        model,
        ExpertParallelCrossEntropyObjective(context),
        zero_stage=3,
        max_grad_norm=None,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
    )
    ids = torch.tensor([[1, 3, 5, 7, 2], [1, 8, 9, 2, 0]])
    valid = ids.ne(0)
    logits = official(ids, attention_mask=valid, use_cache=False).logits[:, :-1]
    losses = F.cross_entropy(
        logits.flatten(0, 1), ids[:, 1:].flatten(), reduction="none"
    ).reshape_as(ids[:, 1:])
    loss = losses.masked_select(valid[:, 1:]).mean()
    loss.backward()
    result = engine.step([{"input_ids": ids, "attention_mask": valid}])
    assert result.loss == pytest.approx(float(loss.detach()), abs=2e-6)
    _, owners, sharded = optimizer_mapping(engine.roles["model"])
    for entry in logical_tensors(engine.model, context):
        if entry.parameter:
            gradient = gather_tensor(
                owners[id(entry.tensor)].grad, entry, context, optimizer_sharded=sharded
            )
            torch.testing.assert_close(
                gradient,
                dict(official.named_parameters())[entry.name].grad,
                atol=2e-6,
                rtol=3e-5,
                msg=entry.name,
            )


@pytest.mark.skipif(
    os.environ.get("ASTER_RUN_REMOTE_MOE_ORACLE") != "1",
    reason="Explicit opt-in required for pinned official source download",
)
def test_seq_aux_matches_unmodified_official_switch_function():
    commit = "f2f0f7bfd88fcb1243df55275988d6af52daea35"
    source = urllib.request.urlopen(
        f"https://raw.githubusercontent.com/NVIDIA/Megatron-LM/{commit}/megatron/core/transformer/moe/moe_utils.py",
        timeout=30,
    ).read()
    assert (
        hashlib.sha256(source).hexdigest()
        == "1b13f06e7bf0a08e9361f7c337b9c3de3be57153d372dd7f676f33aecd0a83dd"
    )
    tree = ast.parse(source.decode())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "switch_load_balancing_loss_func"
    )

    namespace = {"torch": torch, "Optional": Optional}
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), f"megatron-{commit}/moe_utils.py", "exec"
        ),
        namespace,
    )
    official = namespace["switch_load_balancing_loss_func"]
    torch.set_num_threads(1)
    torch.manual_seed(77)
    gate = torch.randn(3, 5, 4, requires_grad=True)
    reference = gate.detach().clone().requires_grad_(True)
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)

    class FixedOutput:
        def __call__(self, **kwargs):
            return TokenOutput(
                torch.zeros(3, 5, 7, requires_grad=True),
                auxiliary={
                    "router": (
                        {
                            "logits": gate.reshape(15, 4),
                            "indices": gate.topk(2, -1).indices.reshape(15, 2),
                        },
                    )
                },
            )

    objective = ExpertParallelCrossEntropyObjective(ParallelContext(), router_aux_coefficient=0.1)
    terms = objective(
        FixedOutput(), {"input_ids": torch.ones(3, 5, dtype=torch.long), "attention_mask": valid}
    )
    actual = terms.terms[1].numerator
    expected = 0.0
    for row in range(3):
        probabilities = reference[row, valid[row]].softmax(-1)
        if not len(probabilities):
            continue
        selected = probabilities.topk(2, -1).indices
        counts = torch.bincount(selected.flatten(), minlength=4)
        expected = expected + official(probabilities, counts, len(probabilities), 2, 4, 1.0)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    (a,) = torch.autograd.grad(actual, gate)
    (b,) = torch.autograd.grad(expected, reference)
    torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
