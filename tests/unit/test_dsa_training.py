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

from aster.models import DeepSeekV32Config, build_model, load_model
from aster.methods.sparse_indexer import DSAIndexerObjective, prepare_dsa_stage
from aster.nn.latent_attention import MultiheadLatentAttention
from aster.nn.attention import attention_mask
from aster.training import Trainer, ParallelContext, ParallelConfig
from aster.training.sharding import zero3_units, shard_module
from aster.training.parallel import Group


def config():
    return DeepSeekV32Config(
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
        first_k_dense_replace=1,
        index_topk=2,
    )


def model(stage):
    torch.manual_seed(431)
    return prepare_dsa_stage(build_model(config()), stage)


def batches():
    ids = torch.tensor([[1, 3, 5, 2, 7, 4], [2, 4, 8, 3, 1, 5], [3, 2, 8, 5, 4, 1]])
    mask = torch.ones_like(ids)
    mask[0, :2] = 0
    mask[1, -1] = 0
    labels = ids.clone()
    labels[2, 3] = -100
    query = torch.ones_like(ids)
    query[2, -2:] = 0
    return [dict(input_ids=ids, labels=labels, attention_mask=mask, indexer_query_mask=query)]


def loss(bundle):
    return sum(term.mean * term.weight for term in bundle.terms)


def sgd(parameters):
    return torch.optim.SGD(parameters, lr=0.02, momentum=0.6, weight_decay=0.03)


def test_dsa_teacher_is_actual_mla_attention_dense_warmup_and_topk():
    torch.set_num_threads(1)
    sample = batches()[0]
    for stage in ("dense_warmup", "sparse_training"):
        native = model(stage)
        attn = native.model.layers[0].self_attn
        hidden = torch.randn(3, 6, 8, requires_grad=True)
        positions = torch.arange(6)[None].expand(3, -1)
        output, _, info = attn(hidden, positions, sample["attention_mask"])

        c = config()
        q = (
            attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden)))
            .reshape(3, 6, 2, 6)
            .transpose(1, 2)
        )
        qn, qr = q.split((2, 4), -1)
        latent, kr = attn.kv_a_proj_with_mqa(hidden).split((4, 4), -1)
        decoded = attn.kv_b_proj(attn.kv_a_layernorm(latent)).reshape(3, 6, 2, 6).transpose(1, 2)
        keys = torch.cat(
            (decoded[..., :2], attn.rope(kr[:, None], positions).expand(-1, 2, -1, -1)), -1
        )
        query = torch.cat((qn, attn.rope(qr, positions)), -1)
        visible = attention_mask(3, 6, 6, padding=sample["attention_mask"], device=hidden.device)
        if stage == "sparse_training":
            selected = torch.zeros_like(visible).scatter(-1, info["indices"][:, None], True)
            visible &= selected
        scores = (query.float() @ keys.float().transpose(-1, -2)) * attn.scale
        scores = scores.masked_fill(~visible, -torch.inf)
        expected = (
            torch.where(visible.any(-1, keepdim=True), scores, 0)
            .softmax(-1)
            .masked_fill(~visible, 0)
        )
        torch.testing.assert_close(info["teacher_probabilities"], expected, atol=2e-7, rtol=2e-6)
        assert not info["teacher_probabilities"].requires_grad
        assert torch.equal(info["training_visible"], visible[:, 0])
        if stage == "dense_warmup":
            dense = MultiheadLatentAttention(c)
            dense.load_state_dict(
                {k: v for k, v in attn.state_dict().items() if not k.startswith("indexer.")}
            )
            torch.testing.assert_close(
                output, dense(hidden, positions, sample["attention_mask"])[0], atol=0, rtol=0
            )


def test_dsa_main_ce_and_indexer_kl_have_disjoint_gradient_owners():
    torch.set_num_threads(1)
    native = model("sparse_training")
    objective = DSAIndexerObjective("sparse_training")
    terms = objective(native, batches()[0]).terms
    terms[0].mean.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in native.named_parameters()
        if ".indexer." not in n
    )
    assert all(p.grad is None for n, p in native.named_parameters() if ".indexer." in n)
    native.zero_grad(set_to_none=True)
    terms = objective(native, batches()[0]).terms
    sum(term.mean for term in terms[1:]).backward()
    assert all(p.grad is None for n, p in native.named_parameters() if ".indexer." not in n)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in native.named_parameters()
        if ".indexer." in n
    )


@pytest.mark.parametrize("stage", ["dense_warmup", "sparse_training"])
@pytest.mark.parametrize("zero", [0, 1, 2, 3])
def test_dsa_zero_updates_frozen_ownership_fresh_resume_and_export(stage, zero, tmp_path):
    torch.set_num_threads(1)
    objective = DSAIndexerObjective(stage)
    reference = model(stage)
    before = deepcopy(reference.state_dict())
    engine = Trainer(
        model(stage), objective, zero_stage=zero, optimizer_factory=sgd, max_grad_norm=None
    )
    expected_loss = loss(objective(reference, batches()[0]))
    expected_loss.backward()
    sgd([p for p in reference.parameters() if p.requires_grad]).step()
    result = engine.step(batches())
    assert result.updated and result.loss == pytest.approx(float(expected_loss.detach()), abs=1e-6)
    for name, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=2e-7, rtol=2e-5)
        if stage == "dense_warmup" and ".indexer." not in name:
            assert torch.equal(value, before[name])
    checkpoint = engine.save_checkpoint(tmp_path / "train.json")
    next_result = engine.step(batches())
    expected = engine.export_state_dict()
    fresh = Trainer(
        model(stage),
        DSAIndexerObjective(stage),
        zero_stage=zero,
        optimizer_factory=sgd,
        max_grad_norm=None,
    )
    fresh.load_checkpoint(checkpoint)
    assert fresh.step(batches()) == next_result
    for name, value in fresh.export_state_dict().items():
        assert torch.equal(value, expected[name])
    if zero == 3:
        units = zero3_units(fresh.model)
        assert all(unit.gathers == unit.releases and unit.gathers > 0 for unit in units)
        assert all(p.numel() == 0 for unit in units for p in unit.module.parameters())
        if stage == "dense_warmup":
            assert all(
                not p.requires_grad and p.grad is None
                for unit in units
                if ".indexer." not in unit.logical_name
                for p in unit.shards
            )
    deploy = build_model(config())
    deploy.load_state_dict(expected, strict=True)
    deploy.save_pretrained(tmp_path / "deploy")
    deploy = load_model(tmp_path / "deploy")
    ids = batches()[0]["input_ids"]
    prefix = deploy(ids[:, :3], use_cache=True)
    suffix = deploy(ids[:, 3:], state=prefix.state, use_cache=True)
    torch.testing.assert_close(suffix.logits, deploy(ids).logits[:, 3:], atol=2e-6, rtol=2e-5)


def test_dsa_invalid_window_no_forward_stage_mutation_and_empty_queries(tmp_path):
    torch.set_num_threads(1)
    engine = Trainer(
        model("dense_warmup"),
        DSAIndexerObjective("dense_warmup"),
        zero_stage=3,
        accumulation_steps=2,
    )
    valid = batches()[0]
    invalid = deepcopy(valid)
    invalid["indexer_query_mask"] = torch.ones(2, 6)
    units = zero3_units(engine.model)
    before = [u.gathers for u in units]
    with pytest.raises(ValueError, match="indexer_query_mask"):
        engine.step([valid, invalid])
    assert before == [u.gathers for u in units] and engine.steps == 0 and not engine._failed
    with pytest.raises(ValueError, match="ownership"):
        prepare_dsa_stage(engine.model, "sparse_training")
    with pytest.raises(ValueError, match="cache"):
        engine.model(valid["input_ids"], use_cache=True)
    native = model("dense_warmup")
    empty = deepcopy(valid)
    empty["indexer_query_mask"].zero_()
    terms = DSAIndexerObjective("dense_warmup")(native, empty).terms
    assert all(term.denominator.dtype == torch.int64 and term.denominator == 0 for term in terms)
    loss(type("Bundle", (), {"terms": terms})()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in native.parameters())
    with pytest.raises(ValueError, match="caller-supplied"):
        DSAIndexerObjective("dense_warmup")(
            native, {**valid, "teacher_attention": torch.ones(3, 6, 6)}
        )
    other = Trainer(
        model("sparse_training"),
        DSAIndexerObjective("sparse_training"),
        zero_stage=3,
        accumulation_steps=2,
    )
    engine.step([valid, valid])
    path = engine.save_checkpoint(tmp_path / "warmup.json")
    with pytest.raises(ValueError):
        other.load_checkpoint(path)


def test_zero3_frozen_leaf_transmits_input_gradient_without_parameter_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    dense = torch.nn.Linear(3, 4).requires_grad_(False)
    sharded = shard_module(deepcopy(dense), Group())
    left = torch.randn(2, 3, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    dense(left).square().sum().backward()
    sharded(right).square().sum().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
    assert all(not p.requires_grad and p.grad is None for p in sharded.shards)
    assert sharded.gathers == sharded.releases == 2


def test_indexer_query_count_does_not_overflow_half_precision():
    from aster.methods.sparse_indexer import indexer_distillation

    scores = torch.zeros(1, 70000, 1, dtype=torch.float16, requires_grad=True)
    term = indexer_distillation(
        scores, torch.ones_like(scores), torch.ones_like(scores, dtype=torch.bool)
    )
    assert term.denominator.dtype == torch.int64 and term.denominator == 70000
    term.mean.backward()
    assert torch.isfinite(scores.grad).all()


def _distributed_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    faulthandler.dump_traceback_later(140)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=160),
    )
    try:
        context = ParallelContext(ParallelConfig(data_parallel=2))
        full = batches()
        local = [{key: value[:1] if rank == 0 else value[1:] for key, value in full[0].items()}]
        for stage in ("dense_warmup", "sparse_training"):
            for zero in (0, 1, 2, 3):
                native = model(stage)
                reference = deepcopy(native)

                if rank == 1:
                    with torch.no_grad():
                        for p in native.parameters():
                            p.add_(0.01)
                objective = DSAIndexerObjective(stage)
                engine = Trainer(
                    native,
                    objective,
                    parallel=context,
                    zero_stage=zero,
                    optimizer_factory=sgd,
                    max_grad_norm=None,
                )
                expected_loss = loss(objective(reference, full[0]))
                expected_loss.backward()
                sgd([p for p in reference.parameters() if p.requires_grad]).step()
                result = engine.step(local)
                assert result.updated and result.loss == pytest.approx(
                    float(expected_loss.detach()), abs=1e-6
                )
                for name, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(
                        value, reference.state_dict()[name], atol=3e-7, rtol=3e-5, msg=name
                    )
                path = engine.save_checkpoint(Path(directory) / f"{stage}-{zero}.json")
                following = engine.step(local)
                expected = engine.export_state_dict(only_rank_zero=False)
                fresh = Trainer(
                    model(stage),
                    DSAIndexerObjective(stage),
                    parallel=context,
                    zero_stage=zero,
                    optimizer_factory=sgd,
                    max_grad_norm=None,
                )
                fresh.load_checkpoint(path)
                assert fresh.step(local) == following
                for name, value in fresh.export_state_dict(only_rank_zero=False).items():
                    assert torch.equal(value, expected[name])
                invalid = deepcopy(local)
                if rank == 1:
                    invalid[0]["indexer_query_mask"] = torch.ones(1)
                gathers = [unit.gathers for unit in zero3_units(fresh.model)]
                with pytest.raises(ValueError, match="indexer_query_mask"):
                    fresh.step(invalid)
                assert (
                    gathers == [unit.gathers for unit in zero3_units(fresh.model)]
                    and not fresh._failed
                )
                dist.barrier()
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


def test_dsa_two_stage_dp2_zero0_through_zero3(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-dsa-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-dsa-")
    try:
        mp.spawn(
            _distributed_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True
        )
    finally:
        shutil.rmtree(directory)
