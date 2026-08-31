from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from aster.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
from aster.models.dspark_gemma4 import Gemma4DSparkConfig, Gemma4DSparkDraft
from aster.data.dspark import DSparkTeacherFeatures


def objects(*, dtype=torch.float32, k_eq_v=True):
    torch.set_num_threads(1)
    torch.manual_seed(787)
    c = Gemma4TextConfig(
        vocab_size=31,
        hidden_size=24,
        intermediate_size=32,
        hidden_size_per_layer_input=0,
        head_dim=4,
        global_head_dim=8,
        attention_k_eq_v=k_eq_v,
        global_rope_factor=2.0,
        global_rotary_fraction=0.5,
        final_logit_softcapping=0.7,
    )
    teacher = Gemma4ForCausalLM(c).to(dtype).eval()
    config = Gemma4DSparkConfig(
        c, num_draft_layers=2, target_layer_ids=(-1, 1), block_size=3, num_anchors=2, markov_rank=4
    )
    draft = Gemma4DSparkDraft(config).to(dtype).initialize_from_target(teacher).eval()
    extractor = DSparkTeacherFeatures(teacher, config, vocabulary_fingerprint="tiny_gemma31")
    ids = torch.randint(1, 31, (2, 8))
    mask = torch.ones_like(ids)
    return draft, teacher, extractor, ids, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_models_gemma4_teacher_features_are_real_scaled_history_and_autocast_stable(dtype):
    draft, teacher, extractor, ids, mask = objects(dtype=dtype)
    original_mode = teacher.training
    with torch.no_grad():
        expected = teacher(ids, output_hidden_states=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = extractor.extract(ids, mask)
    torch.testing.assert_close(
        actual["target_hidden_states"],
        torch.cat((expected.hidden_states[0], expected.hidden_states[2]), -1),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        actual["target_last_hidden_states"], expected.hidden_states[-1], atol=0, rtol=0
    )
    assert actual["teacher_identity"] == draft.teacher_identity
    assert (
        actual["target_hidden_states"].dtype == dtype
        and not actual["target_hidden_states"].requires_grad
    )
    assert extractor.extraction_profile == "native_gemma4_eval_no_autocast_unpadded"
    assert teacher.training == original_mode and all(p.requires_grad for p in teacher.parameters())

    with torch.no_grad():
        teacher.lm_head.weight.add_(1)
    repeated = extractor.extract(ids, mask)
    torch.testing.assert_close(
        actual["target_hidden_states"], repeated["target_hidden_states"], atol=0, rtol=0
    )


@pytest.mark.parametrize("k_eq_v", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_models_gemma4_context_cache_incremental_empty_requery_rollback(dtype, k_eq_v):
    draft, _, extractor, ids, mask = objects(dtype=dtype, k_eq_v=k_eq_v)
    features = extractor.extract(ids, mask)["target_hidden_states"]
    state = None
    old = 0
    projected = []
    tolerance = (
        dict(atol=2e-6, rtol=5e-5) if dtype == torch.float32 else dict(atol=0.008, rtol=0.016)
    )

    def full(noise, end):
        positions = torch.arange(end + draft.config.block_size)[None].expand(len(ids), -1)
        visible = torch.ones(
            len(ids), 1, draft.config.block_size, end + draft.config.block_size, dtype=torch.bool
        )
        with torch.no_grad():
            return draft.backbone(noise, features[:, :end], positions, visible)

    for end in (0, 2, 5, 7):
        noise = torch.zeros(2, draft.config.block_size, dtype=torch.int64)
        noise[:, 0] = ids[:, end]
        before = None if state is None else deepcopy(state)
        handle = draft.fc.register_forward_pre_hook(
            lambda module, args: projected.append(args[0].shape[1])
        )
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                hidden, updated = draft.backbone_cached(noise, features[:, old:end], state=state)
        finally:
            handle.remove()
        torch.testing.assert_close(hidden, full(noise, end), **tolerance)
        heads = (
            draft.config.target.num_global_key_value_heads
            if k_eq_v
            else draft.config.target.num_key_value_heads
        )
        for pair in updated:
            for value in pair:
                assert value.shape == (2, heads, end, draft.config.target.global_head_dim)
                assert value.dtype == dtype and not value.requires_grad

                assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        if state is not None:
            for pair, saved in zip(state, before):
                for value, expected in zip(pair, saved):
                    torch.testing.assert_close(value, expected, atol=0, rtol=0)
        state, old = updated, end
    assert projected == [0, 2, 3, 2]
    alternate = torch.tensor([[9, 3, 5], [8, 7, 2]])
    changed, same_cache = draft.backbone_cached(alternate, features[:, 7:7], state=state)
    torch.testing.assert_close(changed, full(alternate, 7), **tolerance)
    for pair, original in zip(same_cache, state):
        for value, expected in zip(pair, original):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)

    shortened = tuple(tuple(value[:, :, :4].clone() for value in pair) for pair in state)
    restored, restored_cache = draft.backbone_cached(alternate, features[:, 4:6], state=shortened)
    torch.testing.assert_close(restored, full(alternate, 6), **tolerance)
    assert all(value.shape[2] == 6 for pair in restored_cache for value in pair)


def test_models_gemma4_cache_and_teacher_reject_wrong_identity_schema_before_projection():
    from aster.models import CausalLM, Qwen3Config
    from aster.models.dspark import DSparkConfig

    draft, teacher, extractor, ids, mask = objects()
    with pytest.raises(ValueError, match="matched native"):
        DSparkTeacherFeatures(teacher, DSparkConfig(Qwen3Config()), vocabulary_fingerprint="x")
    with pytest.raises(ValueError, match="matched native"):
        DSparkTeacherFeatures(CausalLM(Qwen3Config()), draft.config, vocabulary_fingerprint="x")
    features = extractor.extract(ids, mask)["target_hidden_states"]
    noise = ids[:, :3]
    hidden, state = draft.backbone_cached(noise, features[:, :2])
    calls = []
    handle = draft.fc.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        bad = features[:, :1].clone()
        bad[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            draft.backbone_cached(noise, bad, state=state)
        wrong_dtype = tuple(tuple(value.double() for value in pair) for pair in state)
        with pytest.raises(ValueError, match="precision"):
            draft.backbone_cached(noise, features[:, :0], state=wrong_dtype)
        with pytest.raises(ValueError, match="tuples"):
            draft.backbone_cached(noise, features[:, :0], state=(None,))
        draft.train()
        with pytest.raises(ValueError, match="eval"):
            draft.backbone_cached(noise, features[:, :0], state=state)
    finally:
        handle.remove()
    assert not calls
