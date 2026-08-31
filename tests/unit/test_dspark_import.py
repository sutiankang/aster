from copy import deepcopy
import pytest
import torch
from aster.models import CausalLM, Qwen3Config
from aster.models.dspark import DSparkConfig, DSparkDraft, target_state_identity
from aster.models.dspark_gemma4 import Gemma4DSparkConfig, Gemma4DSparkDraft
from aster.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
from aster.models.dspark_import import import_dspark_state


def objects(family):
    torch.set_num_threads(1)
    torch.manual_seed(783)
    if family == "qwen":
        target = CausalLM(
            Qwen3Config(
                vocab_size=19,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
            )
        ).eval()
        draft = DSparkDraft(
            DSparkConfig(
                target.config, target_layer_ids=(-1, 0), block_size=3, num_anchors=1, markov_rank=3
            )
        )
    else:
        target = Gemma4ForCausalLM(
            Gemma4TextConfig(
                vocab_size=19,
                hidden_size=16,
                intermediate_size=24,
                head_dim=4,
                global_head_dim=8,
                hidden_size_per_layer_input=0,
            )
        ).eval()
        draft = Gemma4DSparkDraft(
            Gemma4DSparkConfig(
                target.config, target_layer_ids=(-1, 0), block_size=3, num_anchors=1, markov_rank=3
            )
        )
    draft.initialize_from_target(target).eval()
    source = {
        key.replace("confidence_head.", "confidence_head.proj."): value.clone()
        for key, value in draft.state_dict().items()
        if key not in {"teacher_weights_loaded", "teacher_fingerprint"}
    }
    return target, draft, source


@pytest.mark.parametrize("family", ["qwen", "gemma"])
@pytest.mark.parametrize("head", ["required", "from_target"])
def test_dspark_official_schema_roundtrip_and_input_storage_independence(family, head):
    target, draft, source = objects(family)
    if head == "from_target":
        source.pop("embed_tokens.weight")
        source.pop("lm_head.weight")
    before = torch.get_rng_state()
    loaded = import_dspark_state(draft.config, source, target=target, embedding_head=head)
    assert torch.equal(
        before, torch.get_rng_state()
    ) and loaded.teacher_identity == target_state_identity(target)
    for key, value in draft.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)
    source["fc.weight"].zero_()
    assert loaded.fc.weight.any() and torch.equal(loaded.fc.weight, draft.fc.weight)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        output = target(ids, output_hidden_states=True)
    batch = dict(
        input_ids=ids,
        loss_mask=torch.ones_like(ids),
        target_hidden_states=torch.cat(output.hidden_states[:2], -1),
        target_last_hidden_states=output.hidden_states[-1],
        anchor_positions=torch.tensor([[1]]),
        block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(
        loaded(**batch).draft_logits, draft(**batch).draft_logits, atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "kind", ["missing", "shape", "nan", "wrong_target", "native_receipt", "duplicate", "half_head"]
)
def test_dspark_import_rejects_bad_weights_without_mutation_or_rng_change(kind):
    target, draft, source = objects("qwen")
    policy = "required"
    if kind == "missing":
        source.pop("fc.weight")
    elif kind == "shape":
        source["fc.weight"] = torch.zeros(1)
    elif kind == "nan":
        source["fc.weight"][0, 0] = float("nan")
    elif kind == "wrong_target":
        source["lm_head.weight"].add_(0.1)
    elif kind == "native_receipt":
        source["teacher_weights_loaded"] = torch.tensor(True)
    elif kind == "duplicate":
        source["confidence_head.weight"] = source["confidence_head.proj.weight"].clone()
    else:
        source.pop("embed_tokens.weight")
        policy = "from_target"
    expected = deepcopy(target.state_dict())
    rng = torch.get_rng_state()
    with pytest.raises(ValueError):
        import_dspark_state(draft.config, source, target=target, embedding_head=policy)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in expected.items():
        torch.testing.assert_close(value, target.state_dict()[key], atol=0, rtol=0)
