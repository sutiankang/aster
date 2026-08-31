from copy import deepcopy

import pytest
import torch

from aster.core import ArtifactStore
from aster.data.dspark import DSparkTeacherFeatures, DSparkFeatureCache, publish_dspark_features
from aster.models import CausalLM, Qwen3Config
from aster.models.dspark import DSparkConfig, DSparkDraft
from aster.methods.dspark import DSparkMethod
from aster.methods.dspark_artifacts import publish_dspark_draft, load_dspark_draft
from aster.inference import ModelRunner, SamplingConfig
from aster.inference.dspark import DSparkDecoder
from aster.training import Trainer


def setup(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(782)
    target = CausalLM(
        Qwen3Config(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()
    config = DSparkConfig(
        target.config, target_layer_ids=(-1, 1), num_anchors=2, block_size=3, markov_rank=4
    )
    teacher = DSparkTeacherFeatures(target, config, vocabulary_fingerprint="unit_vocab19")
    records = {
        str(i): dict(
            input_ids=torch.randint(19, (1, 8)), loss_mask=torch.ones(1, 8, dtype=torch.long)
        )
        for i in range(3)
    }
    store = ArtifactStore(tmp_path / "store")
    artifact = publish_dspark_features(
        store,
        teacher,
        records,
        tmp_path / "features",
        dataset_id="unit_seed782",
        revision="fixed_v1",
        license_id="synthetic_test",
    )
    cache = DSparkFeatureCache(store, artifact.id)
    model = DSparkDraft(config).initialize_from_target(target)
    return target, model, store, cache


@pytest.mark.parametrize("stage", [0, 3])
def test_native_target_cache_training_checkpoint_artifact_load_and_speculation_closed_loop(
    stage, tmp_path
):
    target, model, store, cache = setup(tmp_path)
    batches = [cache.batch(["0"]), cache.batch(["1", "2"])]
    engine = Trainer(model, zero_stage=stage, accumulation_steps=2, lr=0.002, ema_decay=0.9)
    method = DSparkMethod(
        engine,
        vocabulary_fingerprint="unit_vocab19",
        feature_cache_ids=[cache.artifact_id],
        feature_cache_store=store,
    )
    for _ in range(3):
        assert method.update(batches).updated
    checkpoint = engine.save_checkpoint(tmp_path / "training")
    artifact = publish_dspark_draft(method, store, tmp_path / "deployment")
    restored, contract = load_dspark_draft(store, artifact.id)
    assert artifact.parents == (cache.artifact_id,) and contract["receipt"]["role_updates"] == 3
    assert contract["quality_claim"] == "not_evaluated"
    for name, value in engine.export_state_dict(only_rank_zero=False).items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    runner = ModelRunner(target, policy_artifact_id="unit_target19")
    decoder = DSparkDecoder(
        runner,
        restored,
        draft_policy_artifact_id=artifact.id,
        vocabulary_fingerprint="unit_vocab19",
    )
    result = decoder.generate([1, 2], SamplingConfig(max_new_tokens=6, temperature=0))
    expected = []
    with torch.no_grad():
        for _ in range(6):
            expected.append(int(target(torch.tensor([[1, 2] + expected])).logits[0, -1].argmax()))
    assert result.token_ids == tuple(expected) and result.draft_policy_artifact_id == artifact.id
    engine.load_checkpoint(checkpoint, trusted=True)
    assert method.updates == 3 and method.update(batches).updated


def test_dspark_cache_checks_real_tensors_not_only_declared_artifact_id(tmp_path):
    _, model, store, cache = setup(tmp_path)
    engine = Trainer(model, zero_stage=3)
    method = DSparkMethod(
        engine,
        vocabulary_fingerprint="unit_vocab19",
        feature_cache_ids=[cache.artifact_id],
        feature_cache_store=store,
    )
    batch = cache.batch(["0"])
    corrupted = deepcopy(batch)
    corrupted["target_hidden_states"][0, 0, 0] += 0.01
    calls = []
    handle = model.fc.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        with pytest.raises(ValueError, match="immutable feature cache"):
            method.update([corrupted])
    finally:
        handle.remove()
    assert not calls and not engine._failed
    assert method.update([batch]).updated


def test_dspark_untrained_or_unbound_training_cannot_claim_audited_deployment(tmp_path):
    _, model, store, cache = setup(tmp_path)
    engine = Trainer(model)
    method = DSparkMethod(engine, vocabulary_fingerprint="unit_vocab19")
    with pytest.raises(RuntimeError, match="provenance"):
        publish_dspark_draft(method, store, tmp_path / "untrained")
    assert method.update([cache.batch(["0"])]).updated
    with pytest.raises(RuntimeError, match="bound immutable"):
        publish_dspark_draft(method, store, tmp_path / "unbound")
    assert not (tmp_path / "untrained").exists() and not (tmp_path / "unbound").exists()


def test_dspark_feature_cache_precision_contract_ignores_ambient_autocast(tmp_path):
    target, model, _, _ = setup(tmp_path)
    teacher = DSparkTeacherFeatures(target, model.config, vocabulary_fingerprint="unit_vocab19")
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = teacher.extract(ids, torch.ones_like(ids))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = teacher.extract(ids, torch.ones_like(ids))
    assert actual["teacher_identity"] == expected["teacher_identity"]
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, actual[name], atol=0, rtol=0)
