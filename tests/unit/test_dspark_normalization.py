from copy import deepcopy
from contextlib import nullcontext
from datetime import timedelta
import ast
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional
from types import SimpleNamespace
import urllib.request

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from aster.models import CausalLM, Qwen3Config
from aster.models.dspark import DSparkConfig, DSparkDraft, DSparkOutput
from aster.methods.dspark import DSparkMethod, DSparkObjective, dspark_loss_terms
from aster.training import Group, ParallelContext, Trainer


def _objects():
    torch.set_num_threads(1)
    torch.manual_seed(703)
    target = CausalLM(
        Qwen3Config(
            vocab_size=13,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        )
    ).eval()
    model = DSparkDraft(
        DSparkConfig(
            target.config,
            target_layer_ids=(-1, 0),
            num_anchors=2,
            block_size=3,
            markov_rank=3,
            markov_head_type="rnn",
        )
    ).initialize_from_target(target)
    batches = []
    for size in (1, 2):
        ids = torch.randint(13, (size, 7))
        with torch.no_grad():
            output = target(ids, output_hidden_states=True)
        mask = torch.ones_like(ids)
        if size == 1:
            mask[:, 5:] = 0
        batches.append(
            dict(
                input_ids=ids,
                loss_mask=mask,
                target_hidden_states=torch.cat(output.hidden_states[:2], -1),
                target_last_hidden_states=output.hidden_states[-1],
                anchor_positions=torch.tensor([[0, 2]]).expand(size, -1),
                block_keep_mask=torch.ones(size, 2, dtype=torch.bool),
            )
        )
    return model, batches


def _bound(model, batches):
    return [
        dict(
            batch,
            teacher_identity=model.teacher_identity,
            vocabulary_fingerprint="normalization_vocab13",
        )
        for batch in batches
    ]


def _empty(batches):
    batches = deepcopy(batches)
    for batch in batches:
        batch["loss_mask"].zero_()
        batch["block_keep_mask"].zero_()
    return batches


def _reference(model, rank_batches, profile):

    windows = []
    for micro in range(len(rank_batches[0])):
        rows = [batches[micro] for batches in rank_batches]
        merged = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
        windows.append(dspark_loss_terms(model(**merged), denominator_offset=0.0).terms)
    if profile == "official_microbatch_mean":
        return sum(
            sum(t.weight * t.numerator / (t.denominator.float() + 1e-6) for t in terms)
            for terms in windows
        ) / len(windows)
    return sum(
        windows[0][i].weight
        * sum(terms[i].numerator for terms in windows)
        / (sum(terms[i].denominator for terms in windows) + 1e-6)
        for i in range(len(windows[0]))
    )


def _factory(parameters):
    return torch.optim.SGD(parameters, lr=0.025, momentum=0.7, weight_decay=0.01)


def _close_model(engine, expected, *, exact=False):
    for key, value in engine.export_state_dict(only_rank_zero=False).items():
        torch.testing.assert_close(
            value, expected[key], atol=0 if exact else 8e-8, rtol=0 if exact else 2e-5
        )


@pytest.mark.parametrize("profile", ["official_microbatch_mean", "global_window"])
@pytest.mark.parametrize("stage", range(4))
def test_dspark_profiles_actual_full_model_all_zero_layouts(profile, stage):
    model, batches = _objects()
    reference = deepcopy(model)
    expected = _reference(reference, [batches], profile)
    expected.backward()
    _factory(reference.parameters()).step()
    engine = Trainer(
        model,
        accumulation_steps=2,
        zero_stage=stage,
        max_grad_norm=None,
        optimizer_factory=_factory,
    )
    method = DSparkMethod(
        engine, vocabulary_fingerprint="normalization_vocab13", normalization_profile=profile
    )
    result = method.update(_bound(model, batches))
    assert result.updated and abs(result.loss - float(expected.detach())) < 5e-7
    _close_model(engine, reference.state_dict())


def test_dspark_profiles_are_distinct_and_preflight_does_not_sample_rng():
    model, batches = _objects()
    official = _reference(model, [batches], "official_microbatch_mean")
    weighted = _reference(model, [batches], "global_window")
    assert abs(float((official - weighted).detach())) > 1e-4
    engine = Trainer(model, accumulation_steps=2)
    method = DSparkMethod(
        engine, vocabulary_fingerprint="normalization_vocab13", empty_window_policy="skip"
    )
    stochastic = [
        {
            key: value
            for key, value in batch.items()
            if key not in {"anchor_positions", "block_keep_mask"}
        }
        for batch in _bound(model, batches)
    ]
    rng = torch.get_rng_state().clone()
    method.objective.preflight_microbatches(model, stochastic)
    assert torch.equal(rng, torch.get_rng_state()) and method.objective._window_has_targets
    assert method.objective.config_dict()["normalization_profile"] == "official_microbatch_mean"


@pytest.mark.parametrize("profile", ["official_microbatch_mean", "global_window"])
def test_dspark_gemma4_explicit_method_admission_actual_zero3_formula(profile):
    from aster.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
    from aster.models.dspark_gemma4 import Gemma4DSparkConfig, Gemma4DSparkDraft

    _, batches = _objects()
    target = Gemma4ForCausalLM(
        Gemma4TextConfig(
            vocab_size=13,
            hidden_size=24,
            intermediate_size=32,
            head_dim=4,
            global_head_dim=8,
            hidden_size_per_layer_input=0,
            final_logit_softcapping=0.7,
        )
    ).eval()
    model = Gemma4DSparkDraft(
        Gemma4DSparkConfig(
            target.config,
            target_layer_ids=(-1, 1),
            num_anchors=2,
            block_size=3,
            markov_rank=3,
            markov_head_type="rnn",
        )
    ).initialize_from_target(target)
    for batch in batches:
        with torch.no_grad():
            output = target(batch["input_ids"], output_hidden_states=True)
        batch["target_hidden_states"] = torch.cat(
            (output.hidden_states[0], output.hidden_states[2]), -1
        )
        batch["target_last_hidden_states"] = output.hidden_states[-1]
    dense = deepcopy(model)
    expected = _reference(dense, [batches], profile)
    expected.backward()
    _factory(dense.parameters()).step()
    engine = Trainer(
        model, accumulation_steps=2, zero_stage=3, max_grad_norm=None, optimizer_factory=_factory
    )
    method = DSparkMethod(
        engine, vocabulary_fingerprint="normalization_vocab13", normalization_profile=profile
    )
    actual = method.update(_bound(model, batches))
    assert actual.updated and abs(actual.loss - float(expected.detach())) < 5e-7
    _close_model(engine, dense.state_dict())


@pytest.mark.parametrize("profile", ["official_microbatch_mean", "global_window"])
@pytest.mark.parametrize("policy", ["official_step", "skip"])
@pytest.mark.parametrize("stage", [0, 3])
def test_dspark_entire_empty_window_explicit_optimizer_clock_receipt_policy(profile, policy, stage):
    model, batches = _objects()
    engine = Trainer(
        model,
        accumulation_steps=2,
        zero_stage=stage,
        max_grad_norm=None,
        optimizer_factory=_factory,
        ema_decay=0.9,
    )
    method = DSparkMethod(
        engine,
        vocabulary_fingerprint="normalization_vocab13",
        normalization_profile=profile,
        empty_window_policy=policy,
    )
    method.update(_bound(model, batches))
    before = deepcopy(engine.export_state_dict(only_rank_zero=False))
    receipt = engine.last_successful_update()
    optimizer = deepcopy(engine.roles["model"].optimizer.state_dict())
    result = method.update(_bound(model, _empty(batches)))
    assert result.loss == 0 and result.updated == (policy == "official_step")
    if policy == "skip":
        _close_model(engine, before, exact=True)
        assert method.updates == 1 and engine.last_successful_update() == receipt
        _assert_nested_equal(engine.roles["model"].optimizer.state_dict(), optimizer)
    else:
        assert method.updates == 2 and engine.last_successful_update()["role_updates"] == 2
        assert any(
            not torch.equal(value, before[key])
            for key, value in engine.export_state_dict(only_rank_zero=False).items()
        )


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_dspark_normalization_fresh_resume_and_profile_identity(precision, tmp_path):
    def build(profile="official_microbatch_mean", policy="skip"):
        model, batches = _objects()
        engine = Trainer(
            model,
            accumulation_steps=2,
            zero_stage=3,
            max_grad_norm=None,
            precision=precision,
            ema_decay=0.9,
        )
        return (
            engine,
            DSparkMethod(
                engine,
                vocabulary_fingerprint="normalization_vocab13",
                normalization_profile=profile,
                empty_window_policy=policy,
            ),
            _bound(model, batches),
        )

    engine, method, batches = build()
    method.update(batches)
    checkpoint = engine.save_checkpoint(tmp_path / precision)
    expected = method.update(batches)
    state = deepcopy(engine.export_state_dict(only_rank_zero=False))
    other, restored, rows = build()
    other.load_checkpoint(checkpoint, trusted=True)
    actual = restored.update(rows)
    assert actual.loss == expected.loss and restored.updates == 2
    _close_model(other, state, exact=True)
    for profile, policy in [
        ("global_window", "skip"),
        ("official_microbatch_mean", "official_step"),
    ]:
        wrong, _, _ = build(profile, policy)
        with pytest.raises(ValueError):
            wrong.load_checkpoint(checkpoint, trusted=True)


def test_dspark_invalid_profiles_and_implicit_distributed_group_rejected():
    model, _ = _objects()
    for settings in (
        {"normalization_profile": "auto"},
        {"empty_window_policy": "auto"},
        {"normalization_world_size": 2},
    ):
        with pytest.raises(ValueError):
            DSparkObjective(
                teacher_identity=model.teacher_identity, vocabulary_fingerprint="v13", **settings
            )


def _distributed_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        context = ParallelContext()
        for scenario, stage in [
            (scenario, stage) for scenario in ("unequal", "empty_micro") for stage in range(4)
        ]:
            for profile in ("official_microbatch_mean", "global_window"):
                model, batches = _objects()
                dense = deepcopy(model)
                if scenario == "empty_micro":
                    batches[0] = _empty([batches[0]])[0]
                    if rank == 1:
                        batches = _empty(batches)
                elif rank == 1:
                    batches[0] = _empty([batches[0]])[0]
                all_batches = context.dp.gather_objects(batches)
                expected = _reference(dense, all_batches, profile)
                expected.backward()
                _factory(dense.parameters()).step()
                engine = Trainer(
                    model,
                    parallel=context,
                    accumulation_steps=2,
                    zero_stage=stage,
                    max_grad_norm=None,
                    optimizer_factory=_factory,
                    ema_decay=0.9,
                )
                method = DSparkMethod(
                    engine,
                    vocabulary_fingerprint="normalization_vocab13",
                    normalization_profile=profile,
                    empty_window_policy="skip",
                )
                rows = _bound(model, batches)
                result = method.update(rows)
                assert result.updated and abs(result.loss - float(expected.detach())) < 6e-7
                _close_model(engine, dense.state_dict())
                checkpoint = engine.save_checkpoint(
                    Path(directory) / f"{scenario}-{profile}-{stage}"
                )
                expected = method.update(rows)
                weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
                fresh, _ = _objects()
                other = Trainer(
                    fresh,
                    parallel=context,
                    accumulation_steps=2,
                    zero_stage=stage,
                    max_grad_norm=None,
                    optimizer_factory=_factory,
                    ema_decay=0.9,
                )
                restored = DSparkMethod(
                    other,
                    vocabulary_fingerprint="normalization_vocab13",
                    normalization_profile=profile,
                    empty_window_policy="skip",
                )
                other.load_checkpoint(checkpoint, trusted=True)
                actual = restored.update(rows)
                assert actual.loss == expected.loss and restored.updates == 2
                _close_model(other, weights, exact=True)
                receipt = other.last_successful_update()
                assert not restored.update(_bound(fresh, _empty(batches))).updated
                _close_model(other, weights, exact=True)
                assert other.last_successful_update() == receipt
                official_empty = DSparkMethod(
                    other,
                    vocabulary_fingerprint="normalization_vocab13",
                    state_name="official_empty",
                    normalization_profile=profile,
                    empty_window_policy="official_step",
                )
                assert official_empty.update(_bound(fresh, _empty(batches))).updated
                assert (
                    other.last_successful_update()["role_updates"] == 3
                    and official_empty.updates == 1
                )
                bad = deepcopy(rows)
                if rank == 1:
                    bad[-1]["teacher_identity"] = "0" * 64
                calls = []
                handle = fresh.fc.register_forward_pre_hook(lambda *_: calls.append(1))
                try:
                    with pytest.raises(ValueError, match="another teacher"):
                        restored.update(bad)
                finally:
                    handle.remove()
                assert not calls and not other._failed

                if rank == 1:
                    other.loss_groups["dspark_ce"] = Group((rank,))
                with pytest.raises(ValueError, match="normalization domain"):
                    restored.update(rows)
    finally:
        dist.destroy_process_group()


def test_dspark_true_dp2_profiles_all_zero_layouts_empty_microbatch_and_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_dspark_normalization_", dir=root)).resolve()
    try:
        mp.spawn(
            _distributed_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True
        )
    finally:
        if directory.parent == root.resolve() and directory.name.startswith(
            "aster_dspark_normalization_"
        ):
            shutil.rmtree(directory)


@pytest.mark.skipif(
    os.environ.get("ASTER_RUN_REMOTE_DSPARK_ORACLE") != "1",
    reason="Explicit pinned-source network opt-in",
)
def test_dspark_actual_official_microbatch_builder_and_unconditional_optimizer_boundary():
    commit = "005e03b81cec38b7da6399833d609ee89a2587f2"
    base = f"https://raw.githubusercontent.com/deepseek-ai/DeepSpec/{commit}/"
    source = urllib.request.urlopen(base + "deepspec/modeling/dspark/loss.py", timeout=20).read()
    assert (
        hashlib.sha256(source).hexdigest()
        == "2e91efcaff780eec0748ef3f6f0a31374f119f609c664cc79289fdd922335328"
    )
    names = {
        "_build_loss_weight_mask",
        "_compute_local_probabilistic_stats",
        "_compute_accept_rate_3d",
        "_compute_local_l1_term",
        "_collect_local_terms",
        "_build_loss",
    }
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == len(names)
    scope = dict(
        torch=torch,
        F=F,
        Optional=Optional,
        DSparkForwardOutput=DSparkOutput,
        add_metric=lambda *a, **k: None,
    )
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            base + "deepspec/modeling/dspark/loss.py",
            "exec",
        ),
        scope,
    )
    model, batches = _objects()
    official = deepcopy(model)
    reference = []
    for batch in batches:
        terms, confidence = scope["_collect_local_terms"](
            outputs=official(**batch), loss_decay_gamma=4.0, l1_loss_alpha=0.9
        )
        reference.append(
            scope["_build_loss"](
                loss_terms=terms,
                global_denominators={
                    key: terms[key] for key in ("ce_loss_den", "l1_loss_den", "confidence_loss_den")
                },
                ce_loss_alpha=0.1,
                l1_loss_alpha=0.9,
                confidence_head_alpha=1.0,
                has_confidence=confidence,
                world_size=1,
            )
        )
    expected = sum(reference) / len(reference)
    expected.backward()
    _factory(official.parameters()).step()
    engine = Trainer(model, accumulation_steps=2, max_grad_norm=None, optimizer_factory=_factory)
    actual = DSparkMethod(engine, vocabulary_fingerprint="normalization_vocab13").update(
        _bound(model, batches)
    )
    assert abs(actual.loss - float(expected.detach())) < 3e-7
    _close_model(engine, official.state_dict())

    raw = urllib.request.urlopen(base + "deepspec/trainer/base_trainer.py", timeout=20).read()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "61d8dc7f0f6ea34befbf65a8e9132fcca95d5afbed47484a536c476c1048f9b0"
    )
    node = next(
        node
        for node in ast.parse(raw).body
        if isinstance(node, ast.ClassDef) and node.name == "BaseTrainer"
    )
    scope = dict(
        torch=torch,
        nullcontext=nullcontext,
        CUDAPrefetcher=lambda data, device: data,
        FSDP=SimpleNamespace(
            clip_grad_norm_=lambda model, limit: torch.nn.utils.clip_grad_norm_(
                model.parameters(), limit
            )
        ),
        training_logger=SimpleNamespace(
            start_session=lambda **kwargs: None, on_optimizer_step=lambda **kwargs: None
        ),
    )
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            base + "deepspec/trainer/base_trainer.py",
            "exec",
        ),
        scope,
    )
    trainer = scope["BaseTrainer"].__new__(scope["BaseTrainer"])
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(local_batch_size=1, max_grad_norm=1.0),
        logging=SimpleNamespace(checkpointing_steps=9),
    )
    trainer.max_train_steps = 1
    trainer.next_micro_step = 0
    trainer.gradient_accumulation_steps = 2
    trainer.micro_batches_per_epoch = 2
    trainer.device = "cpu"
    trainer.suspend_controller = SimpleNamespace(monitoring=nullcontext, requested=lambda: False)
    trainer._build_train_dataloader = lambda **kwargs: [torch.ones(1, 1), torch.ones(1, 1)]
    trainer.run_batch = lambda batch: trainer.model(batch).sum() * 0.0
    trainer.save_and_eval_checkpoint = lambda: None
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1, weight_decay=0.2)
    calls = []

    def step():
        calls.append(True)
        optimizer.step()

    trainer.optimizer = SimpleNamespace(step=step, get_learning_rate=lambda: 0.1)
    trainer.model.no_sync = nullcontext
    weight = trainer.model.weight.detach().clone()
    trainer.train()
    assert calls == [True] and trainer.next_micro_step == 2
    torch.testing.assert_close(trainer.model.weight, weight * 0.98, atol=2e-8, rtol=1e-7)
