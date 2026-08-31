from copy import deepcopy

import pytest
import torch
from torch import nn

from aster.core import LossTerm, atomic_json, read_json
from aster.training import Trainer
from aster.training.state import read_payload, write_payload


class Objective:
    def __init__(self, scale=1.0):
        self.scale = scale

    def config_dict(self):
        return {"scale": self.scale, "reduction": "global_example_mean"}

    def __call__(self, model, batch):
        value = (model(batch) - 1).square().sum() * self.scale
        return LossTerm(value, torch.tensor(len(batch), dtype=torch.int64), "example")


def build(zero=0):
    torch.set_num_threads(1)
    torch.manual_seed(379)
    return Trainer(
        nn.Linear(2, 1),
        Objective(),
        zero_stage=zero,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.02, momentum=0.7),
    )


def batch():
    return torch.tensor([[1.0, 3.0], [2.0, -1.0]])


def rewrite(path, mutate, target):
    manifest = read_json(path)
    key = "entry" if "entry" in manifest else "entries"
    entry = manifest[key] if key == "entry" else manifest[key][0]
    payload = read_payload(path.parent, entry, trusted=False)
    mutate(payload)
    changed = write_payload(target.parent, target.name, payload)
    manifest[key] = changed if key == "entry" else [changed]
    atomic_json(target, manifest)
    return target


@pytest.mark.parametrize("zero", [0, 1, 2, 3])
def test_actual_override_role_receipt_native_portable_resume(zero, tmp_path):
    engine = build(zero)
    assert engine.last_successful_update() is None
    override = Objective(3.0)
    assert engine.phase("custom_update", objective=override, microbatches=[batch()]).updated
    record = engine.last_successful_update()
    assert record == {
        "role": "model",
        "role_updates": 1,
        "phase": "custom_update",
        "objective_configuration": {
            "class": f"{Objective.__module__}.Objective",
            "codec": "config_dict",
            "configuration": override.config_dict(),
        },
    }
    detached = engine.last_successful_update()
    detached["objective_configuration"]["configuration"]["scale"] = 123
    assert engine.last_successful_update() == record
    checkpoint = engine.save_checkpoint(tmp_path / "native")
    portable = engine.save_portable_checkpoint(tmp_path / "portable")
    assert engine.step([batch()]).updated
    expected = engine.export_state_dict()
    next_record = engine.last_successful_update()
    assert (
        next_record["role_updates"] == 2
        and next_record["objective_configuration"]["configuration"]["scale"] == 1.0
    )
    engine.load_checkpoint(checkpoint)
    assert engine.last_successful_update() == record
    engine.step([batch()])
    assert engine.last_successful_update() == next_record
    for name, value in engine.export_state_dict().items():
        assert torch.equal(value, expected[name])
    dense = build()
    dense.load_portable_checkpoint(portable, seed=19)
    assert dense.last_successful_update() == record
    dense.step([batch()])
    for name, value in dense.export_state_dict().items():
        assert torch.equal(value, expected[name])
    engine.add_role("other", nn.Linear(2, 1))
    engine.phase("other_update", role="other", objective=Objective(5.0), microbatches=[batch()])
    assert engine.last_successful_update(role="other")["role_updates"] == 1
    assert engine.last_successful_update() == next_record


def test_descriptor_frozen_at_entry_and_unknown_callable_is_explicit():
    engine = build()

    class MutatingObjective(Objective):
        def __call__(self, model, batch):
            result = super().__call__(model, batch)
            self.scale = 17.0
            return result

    objective = MutatingObjective(4.0)
    engine.phase("entry", objective=objective, microbatches=[batch()])
    assert (
        engine.last_successful_update()["objective_configuration"]["configuration"]["scale"] == 4.0
    )

    def no_codec(model, value):
        return LossTerm(
            model(value).square().sum(), torch.tensor(len(value), dtype=torch.int64), "example"
        )

    engine.phase("unknown", objective=no_codec, microbatches=[batch()])
    assert engine.last_successful_update()["objective_configuration"] is None


def test_skipped_or_failed_update_does_not_replace_receipt(tmp_path):
    engine = build()
    engine.step([batch()])
    prior = engine.last_successful_update()
    checkpoint = engine.save_checkpoint(tmp_path / "good")

    class Overflow(Objective):
        def __call__(self, model, value):
            return LossTerm(
                model(value).sum() * float("inf"),
                torch.tensor(len(value), dtype=torch.int64),
                "example",
            )

    skipped = engine.phase("overflow", objective=Overflow(), microbatches=[batch()])
    assert not skipped.updated and skipped.overflow and engine.last_successful_update() == prior
    optimizer = engine.roles["model"].optimizer
    old = optimizer.step

    def partial():
        old()
        raise RuntimeError("optimizer storage failure after write")

    optimizer.step = partial
    with pytest.raises(RuntimeError, match="storage failure"):
        engine.phase("failed", objective=Objective(2.0), microbatches=[batch()])
    assert engine.roles["model"].successful_update == prior
    with pytest.raises(RuntimeError, match="idle valid"):
        engine.last_successful_update()
    optimizer.step = old
    engine.load_checkpoint(checkpoint)
    assert engine.last_successful_update() == prior


@pytest.mark.parametrize("portable", [False, True])
def test_old_checkpoint_has_no_invented_receipt_and_bad_receipt_is_prewrite_rejected(
    portable, tmp_path
):
    engine = build()
    engine.step([batch()])
    record = engine.last_successful_update()
    save = engine.save_portable_checkpoint if portable else engine.save_checkpoint
    load = (
        (lambda path: engine.load_portable_checkpoint(path, seed=3))
        if portable
        else engine.load_checkpoint
    )
    original = save(tmp_path / "original")
    legacy = rewrite(
        original,
        lambda payload: payload["roles"]["model"].pop("successful_update"),
        tmp_path / "legacy",
    )
    engine.step([batch()])
    load(legacy)
    assert engine.last_successful_update() is None and engine.roles["model"].updates == 1
    load(original)
    assert engine.last_successful_update() == record
    before = engine.export_state_dict()
    for field, changed in [
        ("role_updates", 7),
        ("role", "wrong"),
        ("phase", ""),
        (
            "objective_configuration",
            {"class": "x", "codec": "config_dict", "configuration": {"x": float("nan")}},
        ),
    ]:

        def mutate(payload):
            payload["roles"]["model"]["successful_update"][field] = changed
            for value in payload["roles"]["model"].get("model", {}).values():
                if value.is_floating_point():
                    value.fill_(123.0)

        bad = rewrite(original, mutate, tmp_path / f"bad-{field}")
        with pytest.raises(ValueError, match="provenance|finite JSON"):
            load(bad)
        assert engine.last_successful_update() == record and not engine._failed
        for name, value in engine.export_state_dict().items():
            assert torch.equal(value, before[name])
