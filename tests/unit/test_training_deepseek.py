from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import faulthandler
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import DeepSeekV3Config, DeepSeekV32Config, build_model
from aster.methods.supervised import CrossEntropyObjective
from aster.nn.experts import TopKRouter
from aster.nn.parameter_codec import public_parameter_names
from aster.training import ParallelConfig, ParallelContext, Trainer
from aster.training.portable import gather_tensor, logical_tensors, optimizer_mapping
from aster.training.runtime_state import apply_runtime_state
from aster.training.sharding import Zero3Unit, zero3_units


def configuration(family, dense=False, *, dropout=0.0):
    cls = DeepSeekV3Config if family == "v3" else DeepSeekV32Config
    return cls(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        kv_lora_rank=4,
        q_lora_rank=6,
        qk_nope_head_dim=2,
        qk_rope_head_dim=4,
        v_head_dim=4,
        moe_intermediate_size=6,
        attention_dropout=dropout,
        first_k_dense_replace=2 if dense else 1,
    )


def batches():
    generator = torch.Generator().manual_seed(703)
    result = []
    for count in (3, 2):
        ids = torch.randint(1, 19, (count, 6), generator=generator)
        mask = torch.ones_like(ids)
        mask[0, -2:] = 0
        labels = ids.clone()
        labels[-1, 2] = -100
        loss_mask = torch.ones_like(ids)
        loss_mask[0, 1] = 0
        positions = torch.arange(6)[None].expand(count, -1).clone()
        result.append(
            dict(
                input_ids=ids,
                labels=labels,
                attention_mask=mask,
                loss_mask=loss_mask,
                position_ids=positions,
            )
        )
    return result


def initialized(config):
    torch.manual_seed(501)
    model = build_model(config)
    for module in model.modules():
        if isinstance(module, TopKRouter) and module.sigmoid:
            module.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 0.4, 0.3]))
    return model


def sgd(parameters):
    return torch.optim.SGD(parameters, lr=0.02, momentum=0.6, weight_decay=0.03)


def gradients(engine):
    _, owners, sharded = optimizer_mapping(engine.roles["model"])
    result = {}
    for entry in logical_tensors(engine.model, engine.parallel):
        if not entry.parameter:
            continue
        gradient = owners[id(entry.tensor)].grad
        result[entry.name] = (
            None
            if gradient is None
            else gather_tensor(gradient, entry, engine.parallel, optimizer_sharded=sharded)
        )
    return result


def compare_step(engine, reference, optimizer, local, full):
    objective = CrossEntropyObjective()
    optimizer.zero_grad(set_to_none=True)
    terms = [objective(reference, batch) for batch in full]
    expected_loss = sum(term.numerator for term in terms) / sum(term.denominator for term in terms)
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.4)
    optimizer.step()
    actual = engine.step(local)
    assert actual.updated and not actual.overflow
    assert actual.loss == pytest.approx(float(expected_loss.detach()), abs=2e-6, rel=2e-5)
    assert actual.grad_norm == pytest.approx(float(expected_norm), abs=3e-6, rel=3e-5)
    gathered = gradients(engine)
    names = public_parameter_names(reference)
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(
            gathered[names[name]], parameter.grad, atol=3e-6, rtol=5e-5, msg=f"{name}/gradient"
        )
    for name, value in engine.export_state_dict(only_rank_zero=False).items():
        torch.testing.assert_close(
            value,
            reference.state_dict()[name],
            atol=3e-6,
            rtol=3e-5,
            msg=f"{name}/weight-or-buffer",
        )


@pytest.mark.parametrize("family", ["v3", "v32"])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("zero", [0, 1, 2, 3])
def test_deepseek_full_model_outputs_global_gradients_and_sgd(family, dense, zero):
    torch.set_num_threads(1)
    model = initialized(configuration(family, dense))
    reference = deepcopy(model)
    engine = Trainer(
        model,
        CrossEntropyObjective(),
        zero_stage=zero,
        accumulation_steps=2,
        optimizer_factory=sgd,
        max_grad_norm=0.4,
    )
    window = batches()
    with torch.no_grad():
        torch.testing.assert_close(
            engine.model(window[0]["input_ids"]).logits,
            reference(window[0]["input_ids"]).logits,
            atol=0,
            rtol=0,
        )
    optimizer = sgd(reference.parameters())
    for _ in range(2):
        compare_step(engine, reference, optimizer, window, window)
    if zero == 3:
        units = zero3_units(engine.model)
        assert all(unit.gathers > 0 and unit.gathers == unit.releases for unit in units)
        assert all(
            parameter.numel() == 0 for unit in units for parameter in unit.module.parameters()
        )
        for layer in engine.model.model.layers:
            assert isinstance(layer.self_attn.kv_b_proj, Zero3Unit)
            assert len(layer.self_attn.kv_b_proj.shards) == 1
        with pytest.raises(RuntimeError, match="unsupported access"):
            engine.model.model.layers[0].self_attn.kv_b_proj.weight.view(1)
        if not dense:
            gate = engine.model.model.layers[1].mlp.gate
            assert isinstance(gate.projection, Zero3Unit)
            assert torch.equal(gate.e_score_correction_bias, torch.tensor([0.0, 0.0, 0.4, 0.3]))


@pytest.mark.parametrize("family", ["v3", "v32"])
@pytest.mark.parametrize("zero", [0, 1, 2, 3])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_deepseek_adam_fresh_resume_dropout_ema_and_standard_export(
    family, zero, precision, tmp_path
):
    torch.set_num_threads(1)
    config = configuration(family, dropout=0.15)

    def make():
        return Trainer(
            initialized(config),
            CrossEntropyObjective(),
            zero_stage=zero,
            accumulation_steps=2,
            precision=precision,
            ema_decay=0.8,
            offload_optimizer="cpu",
            offload_parameters="cpu" if zero == 3 else "none",
        )

    engine = make()
    window = batches()
    assert engine.step(window).updated
    path = engine.save_checkpoint(tmp_path / "native")
    first = engine.step(window)
    expected = engine.export_state_dict()
    expected_ema = engine.export_state_dict(ema=True)
    next_rng = torch.rand(4)
    fresh = make()
    fresh.load_checkpoint(path)
    assert fresh.step(window) == first
    for name, value in fresh.export_state_dict().items():
        assert torch.equal(value, expected[name])
    for name, value in fresh.export_state_dict(ema=True).items():
        assert torch.equal(value, expected_ema[name])
    assert torch.equal(torch.rand(4), next_rng)

    assert any(name.endswith("gate.weight") for name in expected)
    assert not any(".projection.weight" in name or ".shards." in name for name in expected)
    independent = build_model(config)
    independent.load_state_dict(expected, strict=True)
    apply_runtime_state(independent, fresh.export_runtime_state())
    independent.eval()
    fresh.model.eval()
    ids = window[0]["input_ids"]
    with torch.no_grad():
        torch.testing.assert_close(independent(ids).logits, fresh.model(ids).logits, atol=0, rtol=0)


def test_router_codec_is_explicit_and_keeps_initialization_and_buffer():
    torch.manual_seed(32)
    router = TopKRouter(4, 4, 2, sigmoid=True, std=0.03)
    torch.manual_seed(32)
    expected = torch.empty(4, 4)
    torch.nn.init.normal_(expected, std=0.03)
    assert torch.equal(router.weight, expected)
    state = router.state_dict()
    assert set(state) == {"weight", "e_score_correction_bias"}
    assert public_parameter_names(router) == {"projection.weight": "weight"}
    router.load_state_dict(state, strict=True)
    with pytest.raises(ValueError, match="both internal and public"):
        router.load_state_dict({**state, "projection.weight": state["weight"]}, strict=True)


@pytest.mark.oracle
@pytest.mark.parametrize("family", ["v3", "v32"])
def test_deepseek_zero3_direct_actual_transformers_full_gradients(family):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    config = configuration(family)
    model = initialized(config)
    values = asdict(config)
    values.pop("rope")
    values.update(head_dim=config.qk_rope_head_dim, pad_token_id=None)
    if family == "v32":
        values["rope_parameters"] = {"rope_type": "default", "rope_theta": config.rope.theta}
    else:
        values.update(rope_theta=config.rope.theta, rope_interleave=config.rope.interleaved)
    name = "DeepseekV3" if family == "v3" else "DeepseekV32"
    oracle_config = getattr(tf, name + "Config")(**values)
    oracle_config._attn_implementation = "eager"

    oracle_config._experts_implementation = "eager"
    oracle = getattr(tf, name + "ForCausalLM")(oracle_config)
    oracle.load_state_dict(model.state_dict(), strict=True)
    engine = Trainer(
        model,
        CrossEntropyObjective(),
        zero_stage=3,
        accumulation_steps=2,
        optimizer_factory=sgd,
        max_grad_norm=0.4,
    )
    window = batches()
    objective = CrossEntropyObjective()
    with torch.no_grad():
        torch.testing.assert_close(
            engine.model(window[0]["input_ids"]).logits,
            oracle(window[0]["input_ids"], use_cache=False).logits,
            atol=4e-6,
            rtol=4e-5,
        )
    terms = [objective(oracle, batch) for batch in window]
    loss = sum(term.numerator for term in terms) / sum(term.denominator for term in terms)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), 0.4)
    sgd(oracle.parameters()).step()
    result = engine.step(window)
    assert result.updated and result.loss == pytest.approx(float(loss.detach()), abs=2e-6)
    assert result.grad_norm == pytest.approx(float(norm), abs=3e-6, rel=4e-5)
    actual = gradients(engine)
    for name, parameter in oracle.named_parameters():
        torch.testing.assert_close(actual[name], parameter.grad, atol=4e-6, rtol=6e-5, msg=name)
    for name, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, oracle.state_dict()[name], atol=3e-6, rtol=5e-5, msg=name)


def _worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    faulthandler.dump_traceback_later(130, repeat=False)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=150),
    )
    try:
        context = ParallelContext(ParallelConfig(data_parallel=2))
        full = batches()
        local = []
        for index, batch in enumerate(full):
            if index == 0:
                selection = slice(0, 1) if rank == 0 else slice(1, None)
                local.append({key: value[selection] for key, value in batch.items()})
            elif rank == 0:
                empty = {key: value[:1].clone() for key, value in batch.items()}
                empty["labels"].fill_(-100)
                empty["loss_mask"].zero_()
                local.append(empty)
            else:
                local.append(batch)
        for family in ("v3", "v32"):
            for dense in (False, True):
                for zero in (0, 1, 2, 3):
                    config = configuration(family, dense)
                    model = initialized(config)
                    reference = deepcopy(model)
                    engine = Trainer(
                        model,
                        CrossEntropyObjective(),
                        parallel=context,
                        zero_stage=zero,
                        optimizer_factory=sgd,
                        accumulation_steps=2,
                        max_grad_norm=0.4,
                        ema_decay=0.8,
                    )
                    compare_step(engine, reference, sgd(reference.parameters()), local, full)
                    path = engine.save_checkpoint(Path(directory) / f"{family}-{dense}-{zero}")
                    first = engine.step(local)
                    expected = engine.export_state_dict(only_rank_zero=False)

                    fresh = Trainer(
                        initialized(config),
                        CrossEntropyObjective(),
                        parallel=context,
                        zero_stage=zero,
                        optimizer_factory=sgd,
                        accumulation_steps=2,
                        max_grad_norm=0.4,
                        ema_decay=0.8,
                    )
                    fresh.load_checkpoint(path)
                    assert fresh.step(local) == first
                    for name, value in fresh.export_state_dict(only_rank_zero=False).items():
                        assert torch.equal(value, expected[name])
                    invalid = deepcopy(local)
                    if rank == 1:
                        invalid[1]["labels"] = invalid[1]["labels"].float()
                    before_gathers = [unit.gathers for unit in zero3_units(fresh.model)]
                    before_clock = fresh.steps
                    forward_calls = []
                    hook = fresh.model.register_forward_pre_hook(
                        lambda *args: forward_calls.append(1)
                    )
                    try:
                        with pytest.raises(ValueError, match="Labels"):
                            fresh.step(invalid)
                    finally:
                        hook.remove()
                    assert forward_calls == []
                    assert fresh.steps == before_clock and not fresh._failed
                    assert [unit.gathers for unit in zero3_units(fresh.model)] == before_gathers
                    for name, value in fresh.export_state_dict(only_rank_zero=False).items():
                        assert torch.equal(value, expected[name])
                    reloaded = build_model(config)
                    reloaded.load_state_dict(expected, strict=True)
                    if zero == 3:
                        units = zero3_units(fresh.model)
                        assert all(unit.gathers == unit.releases and unit.gathers for unit in units)
                        assert all(
                            p.numel() == 0 for unit in units for p in unit.module.parameters()
                        )
                        assert sum(p.numel() for unit in units for p in unit.shards) < sum(
                            sum(unit.sizes) for unit in units
                        )
                        with torch.no_grad():
                            torch.testing.assert_close(
                                fresh.model(full[0]["input_ids"]).logits,
                                reloaded(full[0]["input_ids"]).logits,
                                atol=0,
                                rtol=0,
                            )
                    assert fresh.step(local) == engine.step(local)
                    corrected = engine.export_state_dict(only_rank_zero=False)
                    for name, value in fresh.export_state_dict(only_rank_zero=False).items():
                        assert torch.equal(value, corrected[name])
                    dist.barrier()

            for precision in ("fp32", "bf16"):
                config = configuration(family, dropout=0.15)

                def make():
                    return Trainer(
                        initialized(config),
                        CrossEntropyObjective(),
                        parallel=context,
                        zero_stage=3,
                        accumulation_steps=2,
                        precision=precision,
                        ema_decay=0.8,
                        offload_optimizer="cpu",
                        offload_parameters="cpu",
                    )

                engine = make()
                assert engine.step(local).updated
                path = engine.save_checkpoint(Path(directory) / f"{family}-adam-{precision}")
                expected_result = engine.step(local)
                expected = engine.export_state_dict(only_rank_zero=False)
                expected_ema = engine.export_state_dict(ema=True, only_rank_zero=False)
                rng = torch.rand(5)
                fresh = make()
                fresh.load_checkpoint(path)
                assert fresh.step(local) == expected_result
                for name, value in fresh.export_state_dict(only_rank_zero=False).items():
                    assert torch.equal(value, expected[name])
                for name, value in fresh.export_state_dict(ema=True, only_rank_zero=False).items():
                    assert torch.equal(value, expected_ema[name])
                assert torch.equal(torch.rand(5), rng)
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


def test_deepseek_dp2_full_model_zero_all_stages(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-deepseek-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-deepseek-")
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        shutil.rmtree(directory)
