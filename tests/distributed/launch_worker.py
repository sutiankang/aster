import torch
from aster.training import ParallelConfig
from aster.training.launch import distributed_session

torch.set_num_threads(1)
with distributed_session(ParallelConfig(data_parallel=2)) as context:
    value = context.dp.all_reduce(torch.tensor(context.rank + 1))
    assert int(value) == 3
    print(f"ASTER_LAUNCH_OK rank={context.rank}", flush=True)
