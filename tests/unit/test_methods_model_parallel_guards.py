from types import SimpleNamespace

import pytest

from aster.core import digest_json
from aster.models.drifting import DriftingConfig
from aster.models.interval_dit import IntervalDiTConfig
from aster.models.planet import PlaNetConfig
from aster.methods.drifting import DriftingMethod
from aster.methods.shortcut import ShortcutMethod
from aster.methods.muzero import MuZeroMethod
from aster.methods.planet_loop import PlaNetLoop, PlaNetReplay
from aster.methods.rollout_distillation import OnPolicyDistillationMethod


@pytest.mark.parametrize("axis", ["expert_parallel", "expert_tensor_parallel"])
@pytest.mark.parametrize(
    "method", ["drifting", "shortcut", "muzero", "planet", "onpolicy_distillation"]
)
def test_unadapted_expert_layout_is_rejected_before_mutation(axis, method):
    parallel = SimpleNamespace(
        config=SimpleNamespace(**{axis: 2}),
        world=SimpleNamespace(gather_objects=lambda value: [value]),
    )
    engine = SimpleNamespace(parallel=parallel, accumulation_steps=1, roles={}, states={})
    if method == "drifting":
        engine.model = SimpleNamespace(config=DriftingConfig())
        invoke = lambda: DriftingMethod(engine, None, feature_identity="guard-test")
    elif method == "shortcut":
        engine.model = SimpleNamespace(config=IntervalDiTConfig(variant="shortcut"))
        invoke = lambda: ShortcutMethod(engine)
    elif method == "muzero":
        invoke = lambda: MuZeroMethod(engine)
    elif method == "planet":
        c = PlaNetConfig(observation_dim=2, action_dim=1)
        engine.model = SimpleNamespace(config=c)
        invoke = lambda: PlaNetLoop(engine, None, PlaNetReplay(c))
    else:
        tokenizer = SimpleNamespace(to_dict=lambda: {"name": "guard-tokenizer"})
        invoke = lambda: OnPolicyDistillationMethod(
            engine, None, tokenizer, teacher_tokenizer_fingerprint=digest_json(tokenizer.to_dict())
        )
    with pytest.raises(ValueError, match="DP|sharding"):
        invoke()
    assert not engine.roles and not engine.states
