import pytest

from aster.methods.factory import build_objective
from aster.methods.cosmos3 import Cosmos3AudioAutoencoderObjective, Cosmos3VisualFlowObjective


def test_audio_codec_factory_preserves_source_kl_and_reconstruction():
    objective = Cosmos3AudioAutoencoderObjective(kl_weight=0.03, sample_posterior=False)
    config = objective.config_dict()
    rebuilt = build_objective({"name": "cosmos3_avae2", **config})
    assert rebuilt.config_dict() == config
    for field in ("type", "kl_definition", "reconstruction"):
        with pytest.raises(ValueError, match="metadata differs"):
            build_objective({"name": "cosmos3_avae2", **config, field: "not_the_actual_formula"})
    with pytest.raises(TypeError):
        build_objective({"name": "cosmos3_avae2", "imaginary_feature": True})


def test_visual_flow_factory_has_explicit_prefill_semantics():
    for kind in ("image", "video", "none"):
        objective = build_objective({"name": "cosmos3_visual_flow", "visual_prefill": kind})
        assert isinstance(objective, Cosmos3VisualFlowObjective)
        assert objective.config_dict()["visual_prefill"] == kind
