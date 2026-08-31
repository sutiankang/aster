from copy import deepcopy
import ast
import hashlib
import os
import random
import urllib.request

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from aster.models.vmc import MDNRNN, MDNRNNConfig
from aster.methods.vmc import MDNRNNObjective
from aster.methods.vmc_stream import VMCSequenceStream, MDNStreamMethod
from aster.training import Trainer


def episodes(*, distribution=False, width=3, frames=(13, 17, 22)):
    generator = torch.Generator().manual_seed(854)
    data = []
    for length in frames:
        latent = torch.randn(length, width, generator=generator) * 0.2
        row = dict(actions=torch.randn(length, 1, generator=generator) * 0.2)
        if distribution:
            row.update(mean=latent, logvar=torch.full_like(latent, -2.0))
        else:
            row["latents"] = latent
        data.append(row)
    return data


def config(**kwargs):
    return MDNRNNConfig(latent_size=3, hidden_size=8, mixtures=2, **kwargs)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested_equal(left, right)
    else:
        assert a == b


def test_legacy_disjoint_layout_episode_restart_and_no_preview_mutation():
    data = [
        dict(latents=torch.arange(7.0).reshape(-1, 1), actions=torch.arange(7.0).reshape(-1, 1)),
        dict(
            latents=torch.arange(7.0, 17.0).reshape(-1, 1),
            actions=torch.arange(7.0, 17.0).reshape(-1, 1),
        ),
    ]
    stream = VMCSequenceStream(data, batch_size=2, sequence_length=4, shuffle=False)
    before = deepcopy(stream.state_dict())
    batches, _ = stream.preview(2)
    assert stream.config_dict()["dropped_frames"] == 1
    assert torch.equal(
        batches[0]["latents"][..., 0], torch.tensor([[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]])
    )
    assert torch.equal(
        batches[1]["latents"][..., 0],
        torch.tensor([[4.0, 5.0, 6.0, 7.0], [12.0, 13.0, 14.0, 15.0]]),
    )
    assert batches[0]["restart"][0, 0] and batches[1]["restart"][0, -1]
    assert batches[0]["restart"].sum() + batches[1]["restart"].sum() == 2

    pairs = [
        (int(row[index]), int(row[index + 1]))
        for batch in batches
        for row in batch["latents"][..., 0]
        for index in range(3)
    ]
    assert (3, 4) not in pairs and (11, 12) not in pairs
    assert_nested_equal(stream.state_dict(), before)
    data[0]["latents"].zero_()
    assert torch.equal(stream.preview(1)[0][0]["latents"], batches[0]["latents"])


def test_stream_rank_lanes_preview_rng_and_identity_restore():
    data = episodes(distribution=True)
    streams = [
        VMCSequenceStream(data, batch_size=3, sequence_length=4, seed=13, rank=rank, world_size=2)
        for rank in range(2)
    ]
    assert [stream.local_batch_size for stream in streams] == [1, 2]
    assert streams[0].order == streams[1].order
    state = streams[1].state_dict()
    first, next_rng = streams[1].preview(1)
    again, _ = streams[1].preview(1)
    assert_nested_equal(first, again)
    streams[1]._commit(1, next_rng)
    assert not torch.equal(streams[1].state_dict()["latent_rng"], state["latent_rng"])
    streams[1].load_state_dict(state)
    assert_nested_equal(streams[1].preview(1)[0], first)
    altered = episodes(distribution=True)
    altered[0]["mean"][0, 0] += 0.01
    other = VMCSequenceStream(
        altered, batch_size=3, sequence_length=4, seed=13, rank=1, world_size=2
    )
    with pytest.raises(ValueError, match="identity"):
        other.load_state_dict(state)
    with pytest.raises(ValueError, match="rank identity"):
        streams[0].load_state_dict(state)


def manual_window(model, batches, state):
    losses, states = [], []
    for batch in batches:
        output = model(
            batch["latents"][:, :-1],
            batch["actions"][:, :-1],
            batch["restart"][:, :-1],
            state=state,
        )
        distribution = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=output.logmix),
            torch.distributions.Normal(output.mean, output.logstd.exp()),
        )
        nll = -distribution.log_prob(batch["latents"][:, 1:]).sum()
        target = batch["restart"][:, 1:].float()
        restart = (
            F.binary_cross_entropy_with_logits(output.restart_logits, target, reduction="none")
            * (1 + target * 9)
        ).sum()
        count = target.numel()
        losses.append((nll, restart, count))
        state = output.state.detach()
        states.append(state)
    loss = sum(row[0] for row in losses) / (
        sum(row[2] for row in losses) * model.config.latent_size
    ) + sum(row[1] for row in losses) / sum(row[2] for row in losses)
    return loss, state, states


@pytest.mark.parametrize("zero", [0, 1, 2, 3])
def test_tbptt_real_update_matches_detached_torch_oracle_and_global_counts(zero):
    torch.manual_seed(33)
    model = MDNRNN(config())
    dense = deepcopy(model)
    stream = VMCSequenceStream(episodes(), batch_size=3, sequence_length=4, shuffle=False)
    factory = lambda p: torch.optim.SGD(p, lr=0.003, momentum=0.6)
    engine = Trainer(
        model,
        MDNRNNObjective(sequence_length=4),
        zero_stage=zero,
        accumulation_steps=2,
        max_grad_norm=None,
        max_grad_value=1.0,
        optimizer_factory=factory,
    )
    method = MDNStreamMethod(engine, stream)
    optimizer = factory(dense.parameters())
    state = None
    for _ in range(2):
        batches, _ = stream.preview(2)
        optimizer.zero_grad(set_to_none=True)
        loss, state, _ = manual_window(dense, batches, state)
        loss.backward()
        torch.nn.utils.clip_grad_value_(dense.parameters(), 1.0)
        optimizer.step()
        result = method.step()
        assert result.loss == pytest.approx(float(loss.detach()), abs=8e-7)
        assert (
            result.terms["mixture_nll"]["denominator"] == 54
            and result.terms["restart"]["denominator"] == 18
        )
        assert not method.state.cell.requires_grad and method.state.cell.grad_fn is None
        torch.testing.assert_close(method.state.cell, state.cell, atol=2e-7, rtol=2e-5)
        for name, value in engine.export_state_dict().items():
            torch.testing.assert_close(value, dense.state_dict()[name], atol=2e-7, rtol=3e-5)
    assert stream.cursor == 4
    with pytest.raises(ValueError, match="insufficient"):
        method.step()


@pytest.mark.parametrize(
    "zero,precision", [(0, "fp32"), (1, "fp32"), (2, "fp32"), (3, "fp32"), (0, "bf16"), (3, "bf16")]
)
def test_tbptt_dropout_latent_adam_offload_exact_next_update_and_epoch(zero, precision, tmp_path):
    torch.manual_seed(81)
    c = config(input_dropout=0.1, recurrent_dropout=0.1, output_dropout=0.1)
    engine = Trainer(
        MDNRNN(c),
        MDNRNNObjective(sequence_length=4),
        zero_stage=zero,
        precision=precision,
        offload_optimizer="cpu",
        offload_parameters="cpu" if zero == 3 else "none",
        max_grad_norm=None,
        max_grad_value=1.0,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.001, eps=1e-4),
    )
    method = MDNStreamMethod(
        engine,
        VMCSequenceStream(episodes(distribution=True), batch_size=3, sequence_length=4, seed=44),
    )
    assert method.step().updated
    path = engine.save_checkpoint(tmp_path / "checkpoint")
    expected = method.step()
    weights = engine.export_state_dict()
    expected_state = deepcopy(method.state_dict())
    engine.load_checkpoint(path)
    assert method.step() == expected
    assert_nested_equal(method.state_dict(), expected_state)
    assert_nested_equal(engine.export_state_dict(), weights)
    with pytest.raises(ValueError, match="Unconsumed"):
        method.advance_epoch()
    method.advance_epoch(drop_remaining=True)
    assert method.state is None and method.stream.epoch == 1 and method.dropped_chunks == 2
    next_epoch = engine.save_checkpoint(tmp_path / "epoch")
    expected = method.step()
    weights = engine.export_state_dict()
    engine.load_checkpoint(next_epoch)
    assert method.step() == expected
    assert_nested_equal(engine.export_state_dict(), weights)
    with pytest.raises(ValueError, match="portable"):
        engine.save_portable_checkpoint(tmp_path / "unsafe")


def test_failed_optimizer_does_not_commit_carry_or_cursor_and_requires_full_restore(tmp_path):
    torch.manual_seed(773)
    engine = Trainer(
        MDNRNN(config()), MDNRNNObjective(sequence_length=4), max_grad_norm=None, max_grad_value=1.0
    )
    method = MDNStreamMethod(
        engine, VMCSequenceStream(episodes(distribution=True), batch_size=3, sequence_length=4)
    )
    method.step()
    checkpoint = engine.save_checkpoint(tmp_path / "good")
    committed = deepcopy(method.state_dict())
    expected = method.step()
    weights = engine.export_state_dict()
    engine.load_checkpoint(checkpoint)
    optimizer = engine.roles["model"].optimizer
    original_step = optimizer.step

    def broken_step():
        original_step()
        raise RuntimeError("real optimizer changed weights before storage failure")

    optimizer.step = broken_step
    with pytest.raises(RuntimeError, match="storage failure"):
        method.step()
    assert method.stream.cursor == committed["stream"]["cursor"]
    assert torch.equal(method.state.cell.cpu(), committed["carry"]["cell"])
    assert_nested_equal(method.stream.state_dict(), committed["stream"])
    with pytest.raises(RuntimeError, match="incomplete"):
        method.state_dict()
    with pytest.raises(ValueError, match="phase"):
        engine.save_checkpoint(tmp_path / "bad")
    with pytest.raises(RuntimeError):
        engine.export_state_dict()
    optimizer.step = original_step
    engine.load_checkpoint(checkpoint)
    assert method.step() == expected
    assert_nested_equal(engine.export_state_dict(), weights)


def test_numerical_overflow_no_false_state_commit(tmp_path):
    torch.manual_seed(514)
    engine = Trainer(MDNRNN(config()), MDNRNNObjective(sequence_length=4))
    method = MDNStreamMethod(engine, VMCSequenceStream(episodes(), batch_size=3, sequence_length=4))
    method.step()
    path = engine.save_checkpoint(tmp_path / "good")
    previous = deepcopy(method.state_dict())
    with torch.no_grad():
        engine.model.output.bias[5:7].fill_(-1000)
    with pytest.raises(RuntimeError, match="skipped"):
        method.step()
    assert_nested_equal(method.stream.state_dict(), previous["stream"])
    assert torch.equal(method.state.cell.cpu(), previous["carry"]["cell"]) and engine._failed
    engine.load_checkpoint(path)
    assert method.step().updated


@pytest.mark.skipif(
    os.environ.get("ASTER_RUN_REMOTE_VMC_ORACLE") != "1",
    reason="Explicit opt-in to pinned official source oracle",
)
def test_author_create_batches_unmodified_function_oracle():
    url = "https://raw.githubusercontent.com/hardmaru/WorldModelsExperiments/fd982b9691a941b52c6addbde29bc801ca6202c8/doomrnn/rnn_train.py"
    source = urllib.request.urlopen(url, timeout=30).read()
    assert (
        hashlib.sha256(source).hexdigest()
        == "3d3d6a976157d7ef0a6482659901373d3596a1b41e9e83f5b49d27419a159cf7"
    )
    parsed = ast.parse(source)
    functions = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in {"get_frame_count", "create_batches"}
    ]
    namespace = dict(np=np, random=random.Random(73), N_z=64)
    exec(compile(ast.Module(body=functions, type_ignores=[]), url, "exec"), namespace)
    data = episodes(distribution=True, width=64)

    data = [{key: value.half().float() for key, value in row.items()} for row in data]
    oracle_data = [
        [row["mean"].numpy(), row["logvar"].numpy(), row["actions"].numpy()[:, 0]] for row in data
    ]
    stream = VMCSequenceStream(data, batch_size=3, sequence_length=4, seed=73)
    for epoch in range(2):
        result = namespace["create_batches"](oracle_data, batch_size=3, seq_length=4)
        for index in range(stream.num_chunks):
            for name, chunks in zip(("mean", "logvar", "actions", "restart"), result):
                actual = stream._rows[name][:, index * 4 : (index + 1) * 4]
                if name == "actions":
                    actual = actual[..., 0]
                np.testing.assert_array_equal(actual.numpy(), chunks[index])
        stream._advance_epoch()
