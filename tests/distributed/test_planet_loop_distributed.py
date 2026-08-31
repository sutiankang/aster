from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models.planet import PlaNetConfig, PlaNetWorldModel
from aster.methods.planet import PlaNetObjective
from aster.methods.planet_loop import PlaNetLoop, PlaNetReplay
from aster.training import ParallelContext, Trainer


class Simulator:
    def __init__(self):
        self.value, self.steps = 0.0, 0

    def config_dict(self):
        return dict(name="native_toy_dp_collection", horizon=6)

    def reset(self):
        self.value = 0.0
        self.steps = 0
        return torch.tensor([self.value, 1.0])

    def step(self, action):
        self.value = 0.7 * self.value + float(action[0])
        self.steps += 1
        return (
            torch.tensor([self.value, 1.0]),
            -((self.value - 0.5) ** 2),
            False,
            self.steps == 6,
            {},
        )

    def state_dict(self):
        return dict(value=self.value, steps=self.steps)

    def load_state_dict(self, state):
        self.value, self.steps = state["value"], state["steps"]


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        parallel = ParallelContext()
        c = PlaNetConfig(
            observation_dim=2,
            action_dim=1,
            state_size=3,
            belief_size=6,
            hidden_size=8,
            reward_layers=1,
            reward_hidden_size=8,
            mean_only=True,
        )
        for stage in range(4):
            torch.manual_seed(246)
            model = PlaNetWorldModel(c)
            initial = deepcopy(model.state_dict())
            objective = PlaNetObjective(sequence_length=4, free_nats=0)
            engine = Trainer(
                model,
                objective,
                parallel=parallel,
                zero_stage=stage,
                max_grad_norm=None,
                optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.001, momentum=0.9),
            )
            replay = PlaNetReplay(c, sequence_length=4, seed=14 + rank)
            loop = PlaNetLoop(
                engine,
                Simulator(),
                replay,
                batch_size=1 + rank,
                seed=21 + rank,
                planner=dict(horizon=2, population=4, elites=2, iterations=1),
            )
            loop.collect_steps(12, random=True)
            cursor = deepcopy(replay.state_dict())
            sample = replay.sample(1 + rank)
            combined = parallel.world.gather_objects(sample)
            replay.load_state_dict(cursor)
            full = {key: torch.cat([row[key] for row in combined]) for key in sample}
            dense = PlaNetWorldModel(c)
            dense.load_state_dict(initial)
            optimizer = torch.optim.SGD(dense.parameters(), lr=0.001, momentum=0.9)
            loss = sum(term.mean * term.weight for term in objective(dense, full).terms)
            loss.backward()
            optimizer.step()
            result = loop.train_step()
            assert result.updated and abs(result.loss - loss.item()) < 2e-5
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, dense.state_dict()[name], atol=2e-6, rtol=5e-5)
            loop.refresh_world()
            loop.collect_steps(2)
            checkpoint = engine.save_checkpoint(Path(output) / f"zero{stage}")
            expected = loop.collect_steps(4)
            expected_update = loop.train_step()
            expected_weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(checkpoint, trusted=True)
            actual = loop.collect_steps(4)
            actual_update = loop.train_step()
            assert actual == expected and actual_update.loss == expected_update.loss
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, expected_weights[name], atol=0, rtol=0)

            before = deepcopy(replay.state_dict())
            if rank == 0:
                replay.episodes.clear()
                replay.order.clear()
                replay.pending.clear()
            failed = False
            try:
                loop.train_step()
            except ValueError as error:
                failed = "Cannot sample" in str(error)
            assert failed and not engine._failed
            replay.load_state_dict(before)
    finally:
        dist.destroy_process_group()


def test_planet_collection_replay_true_dp2_all_zero_unequal_batches(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_planet_loop_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_planet_loop_"):
            shutil.rmtree(directory)
