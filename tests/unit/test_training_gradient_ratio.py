from copy import deepcopy

import pytest
import torch
from torch import nn

from aster.core import LossBundle, LossTerm, atomic_json, read_json
from aster.training import Trainer
from aster.training.state import read_payload, write_payload


class RatioObjective:
    def __init__(self, *, active=True, reference_weight=0.3, detached=False, overflow=False):
        self.active, self.reference_weight, self.detached, self.overflow = (
            active,
            reference_weight,
            detached,
            overflow,
        )

    def config_dict(self):
        return dict(
            active=self.active,
            reference_weight=self.reference_weight,
            detached=self.detached,
            overflow=self.overflow,
        )

    def __call__(self, model, batch):
        x, target = batch
        y = model(x).float()
        nll = (y - target).square().sum()
        adversarial = (y * torch.tensor([1.0, -2.0], device=y.device)).sum()
        if self.detached:
            adversarial = adversarial.detach()
        if self.overflow:
            adversarial = adversarial * float("inf")
        return LossBundle(
            (
                LossTerm(
                    nll,
                    torch.tensor(y.numel(), dtype=torch.int64),
                    "pixel",
                    "nll",
                    self.reference_weight,
                ),
                LossTerm(
                    adversarial,
                    torch.tensor(len(y), dtype=torch.int64),
                    "example",
                    "gan",
                    2.0 if self.active else 0.0,
                ),
                LossTerm(
                    y.pow(4).sum(),
                    torch.tensor(y.numel(), dtype=torch.int64),
                    "pixel",
                    "other",
                    0.11,
                ),
            )
        )


def model():
    torch.set_num_threads(1)
    torch.manual_seed(141)
    return nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))


def batches():
    generator = torch.Generator().manual_seed(75)
    x = torch.randn(5, 3, generator=generator)
    y = torch.randn(5, 2, generator=generator)
    return [(x[:1], y[:1]), (x[1:], y[1:])]


def register(engine, **options):
    engine.register_gradient_ratio(
        "adaptive",
        reference_term="nll",
        target_term="gan",
        parameter="2.weight",
        multiplier=0.7,
        **options,
    )


def oracle_step(reference, optimizer, objective, window):
    terms = [objective(reference, batch).terms for batch in window]
    means = [
        sum(group[i].numerator for group in terms) / sum(group[i].denominator for group in terms)
        for i in range(3)
    ]
    first, second = [
        torch.autograd.grad(value, reference[2].weight, retain_graph=True)[0] for value in means[:2]
    ]
    ratio = (first.double().norm() / (second.double().norm() + 1e-4)).clamp(0, 1e4).detach()
    loss = objective.reference_weight * means[0] + 2 * 0.7 * ratio * means[1] + 0.11 * means[2]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.7)
    optimizer.step()
    return (
        float(ratio),
        float(first.double().norm()),
        float(second.double().norm()),
        float(loss.detach()),
        float(norm),
    )


@pytest.mark.parametrize("zero", [0, 1, 2, 3])
@pytest.mark.parametrize("offload", ["none", "cpu", "nvme"])
def test_ratio_matches_unweighted_global_gradients_and_exact_resume(zero, offload, tmp_path):
    source = model()
    reference = deepcopy(source)
    objective = RatioObjective()
    factory = lambda p: torch.optim.SGD(p, lr=0.025, momentum=0.6, weight_decay=0.01)
    optimizer = factory(reference.parameters())
    engine = Trainer(
        source,
        objective,
        zero_stage=zero,
        optimizer_factory=factory,
        accumulation_steps=2,
        max_grad_norm=0.7,
        ema_decay=0.8,
        offload_optimizer=offload,
        offload_directory=tmp_path / "disk" if offload == "nvme" else None,
    )
    register(engine)
    assert engine.last_gradient_ratio("adaptive") is None
    for index in range(2):
        expected = oracle_step(reference, optimizer, objective, batches())
        result = engine.step(batches())
        receipt = engine.last_gradient_ratio("adaptive")
        assert receipt["role_updates"] == index + 1 and receipt["active"]
        assert receipt["ratio"] == pytest.approx(expected[0], rel=2e-6)
        assert receipt["reference_norm"] == pytest.approx(expected[1], rel=2e-6)
        assert receipt["target_norm"] == pytest.approx(expected[2], rel=2e-6)
        assert receipt["effective_weight"] == pytest.approx(2 * 0.7 * expected[0], rel=2e-6)
        assert result.loss == pytest.approx(expected[3], rel=3e-6)
        assert result.grad_norm == pytest.approx(expected[4], rel=3e-6)
        for name, value in engine.export_state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], atol=2e-7, rtol=3e-6)
        if index == 0:
            native = engine.save_checkpoint(tmp_path / "native")
            portable = engine.save_portable_checkpoint(tmp_path / "portable")
    expected = engine.export_state_dict()
    saved_ratio = engine.last_gradient_ratio("adaptive")
    engine.load_checkpoint(native)
    engine.step(batches())
    assert engine.last_gradient_ratio("adaptive") == saved_ratio
    for name, value in engine.export_state_dict().items():
        assert torch.equal(value, expected[name])
    dense = Trainer(
        model(),
        objective,
        optimizer_factory=factory,
        accumulation_steps=2,
        max_grad_norm=0.7,
        ema_decay=0.8,
    )
    register(dense)
    dense.load_portable_checkpoint(portable, seed=77)
    dense.step(batches())
    for name, value in dense.export_state_dict().items():
        assert torch.equal(value, expected[name])


def test_zero_reference_weight_still_computes_probe_and_warmup_has_no_fake_norms(tmp_path):
    objective = RatioObjective(reference_weight=0.0)
    engine = Trainer(model(), objective, accumulation_steps=2, max_grad_norm=None)
    register(engine)
    assert (
        engine.step(batches()).updated
        and engine.last_gradient_ratio("adaptive")["reference_norm"] > 0
    )
    warmup = RatioObjective(active=False, detached=True)
    engine.phase("warmup", objective=warmup, microbatches=batches())
    record = engine.last_gradient_ratio("adaptive")
    assert not record["active"] and record["ratio"] is None and record["effective_weight"] == 0
    checkpoint = engine.save_checkpoint(tmp_path / "good")
    result = engine.phase(
        "bad_number", objective=RatioObjective(overflow=True), microbatches=batches()
    )
    assert (
        result.overflow and not result.updated and engine.last_gradient_ratio("adaptive") == record
    )
    with pytest.raises(ValueError, match="no global gradient"):
        engine.phase(
            "missing_gradient", objective=RatioObjective(detached=True), microbatches=batches()
        )
    with pytest.raises(RuntimeError, match="idle valid"):
        engine.last_gradient_ratio("adaptive")
    engine.load_checkpoint(checkpoint)
    assert engine.last_gradient_ratio("adaptive") == record


def test_policy_validation_and_record_corruption_precede_weight_writes(tmp_path):
    engine = Trainer(model(), RatioObjective(), accumulation_steps=2)
    register(engine)
    with pytest.raises(ValueError, match="exactly one"):
        engine.register_gradient_ratio(
            "duplicate", reference_term="nll", target_term="gan", parameter="0.weight"
        )
    with pytest.raises(ValueError, match="FQN"):
        engine.register_gradient_ratio(
            "bad_name", reference_term="nll", target_term="other", parameter="missing"
        )
    engine.step(batches())
    path = engine.save_checkpoint(tmp_path / "original")
    before = engine.export_state_dict()
    manifest = read_json(path)
    payload = read_payload(path.parent, manifest["entries"][0], trusted=False)
    payload["gradient_ratio_records"]["adaptive"]["ratio"] += 1
    for value in payload["roles"]["model"]["model"].values():
        value.fill_(123)
    entry = write_payload(path.parent, "bad", payload)
    atomic_json(tmp_path / "bad", {**manifest, "entries": [entry]})
    with pytest.raises(ValueError, match="registered formula"):
        engine.load_checkpoint(tmp_path / "bad")
    assert not engine._failed
    for name, value in engine.export_state_dict().items():
        assert torch.equal(value, before[name])


def test_unmodified_official_adaptive_weight_function_oracle():
    import ast
    import hashlib
    import os
    import types
    import urllib.request

    if os.environ.get("ASTER_RUN_REMOTE_GRADIENT_RATIO_ORACLE") != "1":
        pytest.skip("Explicit opt-in for pinned official source download")
    url = "https://raw.githubusercontent.com/CompVis/taming-transformers/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/losses/vqperceptual.py"
    source = urllib.request.urlopen(url, timeout=30).read()
    assert (
        hashlib.sha256(source).hexdigest()
        == "b46889cabb89785dd82c9b1fcf07ad8f1d4a9daacf6b4f74d88992b7007e8b1a"
    )
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "VQLPIPSWithDiscriminator"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "calculate_adaptive_weight"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), url, "exec"),
        namespace,
    )
    reference = model()
    objective = RatioObjective()
    terms = [objective(reference, value).terms for value in batches()]
    means = [
        sum(group[i].numerator for group in terms) / sum(group[i].denominator for group in terms)
        for i in (0, 1)
    ]
    expected = namespace["calculate_adaptive_weight"](
        types.SimpleNamespace(discriminator_weight=0.7), *means, last_layer=reference[2].weight
    )
    engine = Trainer(model(), objective, accumulation_steps=2)
    register(engine)
    engine.step(batches())
    assert engine.last_gradient_ratio("adaptive")["effective_weight"] == pytest.approx(
        2 * float(expected), rel=2e-6
    )
