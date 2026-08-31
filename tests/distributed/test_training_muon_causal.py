from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import LlamaConfig, Qwen2Config, Qwen3Config, build_model
from aster.methods import CrossEntropyObjective
from aster.training import (
    Trainer,
    ParallelConfig,
    ParallelContext,
    MuonFactory,
    MuonWithAuxAdam,
    parallelize_causal_lm,
    TensorParallelCrossEntropyObjective,
)
from aster.training.portable import logical_tensors, gather_tensor, optimizer_mapping
from aster.training.sharding import Zero3Unit


def configuration(kind):
    return kind(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        max_position_embeddings=32,
    )


def data(replica):
    ids = torch.tensor([[1, 4, 6, 8, 9], [1, 5, 3, 0, 0], [1, 9, 11, 13, 2]])
    ids = (ids + replica).remainder(17)
    padding = torch.ones_like(ids, dtype=torch.bool)
    padding[1, -2:] = False
    labels = ids.clone()
    labels[:, 0] = -100
    mask = padding.clone()
    mask[:, 2] = False
    batch = dict(
        input_ids=ids,
        labels=labels,
        attention_mask=padding,
        loss_mask=mask,
        position_ids=torch.arange(2, 7)[None].expand(3, -1),
    )
    split = 1 if replica == 0 else 2
    return [
        {key: value[:split] for key, value in batch.items()},
        {key: value[split:] for key, value in batch.items()},
    ]


def factory(model, profile):
    return MuonFactory.from_model(
        model,
        auxiliary_modules=("lm_head",),
        profile=profile,
        muon_options={"lr": 0.001},
        auxiliary_options={"lr": 0.0005, "eps": 1e-5},
    )


def independent_optimizer(model, specification):
    parameters = dict(model.named_parameters())
    return MuonWithAuxAdam(
        [
            {
                **{k: v for k, v in group.items() if k != "names"},
                "params": [parameters[name] for name in group["names"]],
            }
            for group in specification.groups
        ]
    )


def capture(engine):
    optimizer, owners, sharded = optimizer_mapping(engine.roles["model"])
    entries = logical_tensors(engine.model, engine.parallel)
    actual = {}
    original = optimizer.step

    def step(*args, **kwargs):
        auxiliary_expected = []
        for group in optimizer.param_groups:
            if group["use_muon"]:
                continue
            for owner in group["params"]:
                if owner.grad is None:
                    continue
                parameter = torch.nn.Parameter(owner.detach().clone())
                parameter.grad = owner.grad.clone()
                reference = MuonWithAuxAdam(
                    [{**{k: v for k, v in group.items() if k != "params"}, "params": [parameter]}]
                )
                reference.state[parameter] = deepcopy(optimizer.state.get(owner, {}))
                reference.step()
                auxiliary_expected.append((owner, parameter))
        for entry in entries:
            if not entry.parameter or id(entry.tensor) not in owners:
                continue
            owner = owners[id(entry.tensor)]
            if owner.grad is not None:
                actual[entry.name] = gather_tensor(
                    owner.grad, entry, engine.parallel, optimizer_sharded=sharded
                )
        result = original(*args, **kwargs)
        for owner, expected in auxiliary_expected:
            assert torch.equal(owner, expected)
        return result

    optimizer.step = step
    return actual


def worker(rank, rendezvous, directory, data_parallel):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2 * data_parallel,
        timeout=timedelta(seconds=180),
    )
    try:
        context = ParallelContext(ParallelConfig(tensor_parallel=2, data_parallel=data_parallel))
        cases = [
            (kind, profile, zero, "fp32", "none")
            for kind in (LlamaConfig, Qwen2Config, Qwen3Config)
            for profile in ("keller", "moonlight")
            for zero in range(4)
        ]
        cases += [
            (Qwen3Config, p, 3, "bf16", o) for p in ("keller", "moonlight") for o in ("cpu", "nvme")
        ]
        for kind, profile, zero, precision, offload in cases:
            tag = f"{kind.architecture}-{profile}-{zero}-{precision}-{offload}"
            torch.manual_seed(429)
            source = build_model(configuration(kind))
            oracle = deepcopy(source)
            specification = factory(source, profile)
            reference_optimizer = independent_optimizer(oracle, specification)

            def make():
                torch.manual_seed(429)
                dense = build_model(configuration(kind))
                model = parallelize_causal_lm(dense, context)
                return Trainer(
                    model,
                    TensorParallelCrossEntropyObjective(context),
                    parallel=context,
                    zero_stage=zero,
                    optimizer_factory=factory(dense, profile),
                    accumulation_steps=2,
                    precision=precision,
                    offload_optimizer=offload,
                    max_grad_norm=0.7,
                    ema_decay=0.9,
                    offload_directory=Path(directory) / f"offload-{rank}"
                    if offload == "nvme"
                    else None,
                )

            engine = make()
            actual_gradients = capture(engine)
            for update in range(2):
                if precision == "fp32":
                    terms = [
                        CrossEntropyObjective()(oracle, batch)
                        for replica in range(data_parallel)
                        for batch in data(replica)
                    ]
                    reference_optimizer.zero_grad(set_to_none=True)
                    (sum(t.numerator for t in terms) / sum(t.denominator for t in terms)).backward()
                    norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), 0.7)
                    reference_gradients = {
                        name: p.grad.clone() for name, p in oracle.named_parameters()
                    }
                result = engine.step(data(context.dp.rank))
                assert result.updated
                actual = engine.export_state_dict(only_rank_zero=False)
                if precision == "fp32":
                    assert result.grad_norm == pytest.approx(float(norm), rel=3e-5, abs=5e-7)
                    for name, parameter in oracle.named_parameters():
                        torch.testing.assert_close(
                            actual_gradients[name],
                            reference_gradients[name],
                            atol=2e-7,
                            rtol=8e-5,
                            msg=lambda message: f"{tag}/{name}: " + message,
                        )
                        parameter.grad = actual_gradients[name].clone()
                    reference_optimizer.step()

                    for name, value in oracle.state_dict().items():
                        torch.testing.assert_close(
                            actual[name],
                            value,
                            atol=1e-8,
                            rtol=0,
                            msg=lambda message: f"{tag}/{name}: " + message,
                        )

                    oracle.load_state_dict(actual)
                    opt, own, shr = optimizer_mapping(engine.roles["model"])
                    parameters = dict(oracle.named_parameters())
                    for entry in logical_tensors(engine.model, context):
                        if entry.name not in parameters:
                            continue
                        state = opt.state[own[id(entry.tensor)]]
                        expected_state = reference_optimizer.state[parameters[entry.name]]
                        for key, value in state.items():
                            complete = (
                                gather_tensor(value, entry, context, optimizer_sharded=shr)
                                if isinstance(value, torch.Tensor)
                                else value
                            )
                            if isinstance(complete, torch.Tensor):
                                torch.testing.assert_close(
                                    complete, expected_state[key], atol=1e-8, rtol=2e-6
                                )
                            else:
                                assert complete == expected_state[key]
                            expected_state[key] = deepcopy(complete)
                if update == 0:
                    checkpoint = engine.save_checkpoint(Path(directory) / tag)
                    if zero == 3 and precision == "fp32":
                        engine.save_portable_checkpoint(
                            Path(directory) / f"portable-{kind.architecture}-{profile}"
                        )

            optimizer, owners, _ = optimizer_mapping(engine.roles["model"])
            query = next(
                e
                for e in logical_tensors(engine.model, context)
                if e.name.endswith("q_proj.weight")
            )
            owner = owners[id(query.tensor)]
            assert owner._aster_muon_layout.shape == (8, 8)
            assert owner.numel() == (32 if zero == 0 else 32 // data_parallel)
            state = (
                optimizer._aster_state_loader(owner)
                if hasattr(optimizer, "_aster_state_loader")
                else optimizer.state[owner]
            )
            assert state["momentum_buffer"].numel() == owner.numel()
            if zero == 3:
                units = [
                    module for module in engine.model.modules() if isinstance(module, Zero3Unit)
                ]
                assert all(p.numel() == 0 for unit in units for p in unit.module.parameters())
            expected_ema = engine.export_state_dict(ema=True, only_rank_zero=False)
            fresh = make()
            fresh.load_checkpoint(checkpoint)
            fresh.step(data(context.dp.rank))
            for name, value in fresh.export_state_dict(only_rank_zero=False).items():
                assert torch.equal(value, actual[name])
            for name, value in fresh.export_state_dict(ema=True, only_rank_zero=False).items():
                assert torch.equal(value, expected_ema[name])
            exported = build_model(configuration(kind))
            exported.load_state_dict(actual, strict=True)
            assert exported.lm_head.weight is exported.model.embed_tokens.weight
            assert exported.model.embed_tokens.weight.shape[0] == 17
            if zero == 3 and precision == "fp32" and rank == 0:
                torch.save(
                    {"weights": actual, "gradients": actual_gradients},
                    Path(directory) / f"next-{kind.architecture}-{profile}.pt",
                )
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("data_parallel", [1, 2], ids=["tp2-dp1", "tp2-dp2"])
def test_complete_muon_models_tp2_dp2_zero_profiles_offload_resume_export(tmp_path, data_parallel):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-muon-tp-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-muon-tp-")
    try:
        mp.spawn(
            worker,
            args=(str(directory / "store"), str(tmp_path), data_parallel),
            nprocs=2 * data_parallel,
            join=True,
        )
    finally:
        shutil.rmtree(directory)

    torch.set_num_threads(1)
    for kind in (LlamaConfig, Qwen2Config, Qwen3Config):
        for profile in ("keller", "moonlight"):
            model = build_model(configuration(kind))
            engine = Trainer(
                model,
                CrossEntropyObjective(),
                optimizer_factory=factory(model, profile),
                accumulation_steps=2 * data_parallel,
                max_grad_norm=0.7,
                ema_decay=0.9,
            )
            engine.load_portable_checkpoint(
                tmp_path / f"portable-{kind.architecture}-{profile}", seed=943
            )
            expected = torch.load(
                tmp_path / f"next-{kind.architecture}-{profile}.pt", weights_only=True
            )
            optimizer = engine.roles["model"].optimizer
            original = optimizer.step

            def step(*args, **kwargs):
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad, expected["gradients"][name], atol=2e-7, rtol=8e-5
                    )
                    parameter.grad.copy_(expected["gradients"][name])
                return original(*args, **kwargs)

            optimizer.step = step
            engine.step([batch for replica in range(data_parallel) for batch in data(replica)])
            for name, value in engine.export_state_dict().items():
                torch.testing.assert_close(value, expected["weights"][name], atol=1e-8, rtol=0)
