from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import LlamaConfig, Qwen3Config, build_model
from aster.methods import CrossEntropyObjective
from aster.training import (
    Trainer,
    ParallelContext,
    ParallelConfig,
    parallelize_causal_lm,
    CausalPipelineCrossEntropyObjective,
)
from aster.training.sharding import Zero3Unit
from test_training_causal_parallel import _config, _data, _gradients


def _batches(replica):
    return [_data(replica), {key: value[:, :-1] for key, value in _data(replica).items()}]


def _dense_step(model, optimizer, replicas):
    terms = [
        CrossEntropyObjective()(model, batch)
        for replica in range(replicas)
        for batch in _batches(replica)
    ]
    numerator, denominator = (
        sum(item.numerator for item in terms),
        sum(item.denominator for item in terms),
    )
    optimizer.zero_grad(set_to_none=True)
    (numerator / denominator).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
    optimizer.step()
    return float(numerator.detach() / denominator), float(norm), float(denominator)


def _worker(rank, rendezvous, directory, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    try:
        grid = (
            ParallelConfig(pipeline_parallel=2, data_parallel=2)
            if mode == "pp_dp"
            else ParallelConfig(tensor_parallel=2, pipeline_parallel=2)
        )
        context = ParallelContext(grid)
        for kind in (LlamaConfig, Qwen3Config):
            for stage, schedule, tied in (
                (0, "gpipe", False),
                (3, "1f1b", False),
                (0, "gpipe", True),
                (3, "1f1b", True),
            ):
                torch.manual_seed(42)
                config = replace(_config(kind, 1, tied), num_hidden_layers=2)
                source = build_model(config)
                reference = deepcopy(source)
                factory = lambda parameters: torch.optim.SGD(
                    parameters, lr=0.02, momentum=0.8, weight_decay=0.01
                )
                reference_optimizer = factory(reference.parameters())
                model = parallelize_causal_lm(source, context, pipeline_schedule=schedule)
                local_count = sum(parameter.numel() for parameter in model.parameters())
                assert local_count < sum(parameter.numel() for parameter in source.parameters())
                objective = CausalPipelineCrossEntropyObjective(context)
                engine = Trainer(
                    model,
                    objective,
                    parallel=context,
                    zero_stage=stage,
                    optimizer_factory=factory,
                    lr=0.02,
                    max_grad_norm=0.7,
                    accumulation_steps=2,
                )
                for _ in range(2):
                    loss, norm, count = _dense_step(reference, reference_optimizer, context.dp.size)
                    result = engine.step(_batches(context.dp.rank))
                    assert result.updated and result.loss == pytest.approx(loss, rel=2e-6, abs=2e-7)
                    assert result.grad_norm == pytest.approx(norm, rel=3e-5, abs=8e-7)
                    assert result.terms["ce"]["denominator"] == count
                    parameters = dict(reference.named_parameters(remove_duplicate=False))
                    for name, gradient in _gradients(engine).items():
                        torch.testing.assert_close(
                            gradient,
                            parameters[name].grad,
                            rtol=1e-4,
                            atol=3e-7,
                            msg=lambda message: (
                                f"{mode}/{kind.architecture}/{stage}/{name}: " + message
                            ),
                        )
                    weights = engine.export_state_dict(only_rank_zero=False)
                    assert set(weights) == set(reference.state_dict())
                    for name, value in reference.state_dict().items():
                        torch.testing.assert_close(weights[name], value, rtol=8e-5, atol=5e-7)
                report = engine.evaluate(_batches(context.dp.rank))
                terms = [
                    CrossEntropyObjective()(reference, batch)
                    for replica in range(context.dp.size)
                    for batch in _batches(replica)
                ]
                expected_loss = float(
                    sum(term.numerator for term in terms).detach()
                    / sum(term.denominator for term in terms)
                )
                assert report["ce"]["mean"] == pytest.approx(expected_loss, rel=2e-6, abs=3e-7)
                if stage == 3:
                    units = [unit for unit in model.modules() if isinstance(unit, Zero3Unit)]
                    assert units and all(
                        parameter.numel() == 0
                        for unit in units
                        for parameter in unit.module.parameters()
                    )
                    if context.dp.size > 1:
                        assert (
                            sum(parameter.numel() for parameter in model.parameters()) < local_count
                        )
                checkpoint = Path(directory) / f"{mode}-{kind.architecture}-{stage}-tied{tied}"
                engine.save_checkpoint(checkpoint)
                engine.step(_batches(context.dp.rank))
                expected = engine.export_state_dict(only_rank_zero=False)
                engine.load_checkpoint(checkpoint, trusted=True)
                engine.step(_batches(context.dp.rank))
                for key, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                deployed = build_model(config)
                deployed.load_state_dict(expected, strict=True)

                assert deployed(_data(0)["input_ids"]).logits.shape[-1] == 17
                assert (deployed.lm_head.weight is deployed.model.embed_tokens.weight) is tied
        bad = _batches(context.dp.rank)
        if rank == 1:
            bad[1]["position_ids"][0, 0] = -1
        before = deepcopy(model.state_dict())
        with pytest.raises(ValueError, match="preflight"):
            engine.step(bad)
        assert not engine._failed
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

        assert engine.step(_batches(context.dp.rank)).updated
        native = engine.roles["model"].optimizer
        while hasattr(native, "optimizer"):
            native = native.optimizer
        if context.pp.rank == 1:
            native.param_groups[0]["lr"] *= 2
        before = deepcopy(model.state_dict())
        with pytest.raises(ValueError, match="PP tied optimizer"):
            engine.step(_batches(context.dp.rank))
        assert not engine._failed
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        unsupported = ParallelContext(ParallelConfig(pipeline_parallel=4))
        with pytest.raises(ValueError, match="Cross-stage tied"):
            parallelize_causal_lm(
                build_model(replace(config, num_hidden_layers=4, tie_word_embeddings=True)),
                unsupported,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["pp_dp", "tp_pp"])
def test_full_causal_pipeline_training_resume_and_export(mode, tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    rendezvous = Path(tempfile.mkdtemp(prefix="aster_causal_pp_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(rendezvous / "rdzv"), str(tmp_path), mode), nprocs=4, join=True)
    finally:
        assert rendezvous.parent == root.resolve() and rendezvous.name.startswith(
            "aster_causal_pp_"
        )
        shutil.rmtree(rendezvous)
