# 原生 MuZero：模型、搜索、回放与训练

这条实现已经连通 learned representation/dynamics/reward/value/policy、原生
PUCT/Gumbel 搜索、轨迹重新分析、n-step 目标、带版本 PER、统一 Trainer 与整体恢复。
模型是显式命名的 `muzero_vector` 最小向量 MLP 实例；不是 Atari ResNet 或棋盘编码
的重命名，不宣称实现整个 MuZero/AlphaZero 产品栈或达到论文成绩。

## 分层与坐标

- `models.muzero`：初始表示 h、动作条件 dynamics g、policy/value/reward heads；
  隐状态 min-max 归一化、跨 dynamics 的半梯度、MuZero scalar transform 与 two-hot support。
- `planning.mcts`：每次模拟只做一次完整 batch 的模型回调；PUCT 和 Gumbel Sequential
  Halving 分别实现。树中累计的是带初始估值的一致 mean backup，详见 `MCTS.md`。
- `methods.muzero_replay`：冻结搜索快照、完整 episode、重新搜索、固定 K 窗口和 PER。
- `methods.muzero`：三头目标按 K 缩放，实际轨迹数由 Trainer 做全局计数；多角色生命周期
  中没有环境调用或偷偷访问 reset 后观测的路径。

一个 episode 有 T 次动作与奖励、T+1 个观测。`rewards[t]` 对应动作 `actions[t]`
进入下一状态；初始状态没有 reward 监督。terminal 的最后状态不再搜索和 bootstrap；
truncation 的最后状态必须是 reset 前的 `final_observation`，需要价值 bootstrap。
`valid` 标出有监督的状态，`reward_valid` 单独标出实际观察到的边。两者不能共用：
截断后的末观测有效，但从它出发的虚构动作和 reward 不存在。

## 统一入口

```python
from aster.models.muzero import MuZeroConfig, MuZeroModel
from aster.methods.muzero import MuZeroMethod
from aster.methods.muzero_replay import MuZeroReplay, MuZeroSearch
from aster.training import Trainer

config = MuZeroConfig(observation_dim=16, num_actions=4)
trainer = Trainer(MuZeroModel(config), lr=3e-4, zero_stage=3)
method = MuZeroMethod(trainer)
planner = MuZeroSearch.from_trainer(trainer, num_simulations=32, seed=7)
replay = MuZeroReplay(config, unroll_steps=5, td_steps=10, seed=11)
trainer.register_state('planner', planner)
trainer.register_state('replay', replay)

# episode 由外部已获授权的采集器提供；这里不会自行启动环境或执行动作。
analysis = planner.reanalyze(episode)
replay.add_episode(episode, analysis)
batch = replay.sample(32, device=trainer.device)
result = method.update([batch])
planner.refresh(trainer)       # 明确同步权重，搜索不会偷读正在训练的分片。
trainer.save_checkpoint(checkpoint_directory)
```

所有 rank 都须进入 `from_trainer/refresh` 的权重收集。当前搜索是独立复制快照的
参考实现，不是分布式搜索服务；可在 CPU 或 Torch 支持的设备运行，CUDA 尚未实测。
搜索参数、模型权重摘要、RNG、回放抽样 RNG/槽版本跟随训练 checkpoint 恢复。
内容摘要用于本地血缘一致性，不是外部不可信轨迹的签名认证。

## 已验证与边界

`test_muzero_replay.py` 验证了 terminal/truncation 窗口、ZeRO0/3 整体精确恢复，
以及一个离线单步 bandit：没有向 actor 直接提供正确动作标签；训练 learned reward
后反复重新搜索，最终搜索选中高奖励动作。它证明闭环可学习，不是公开基准成绩。
`test_muzero.py` 另验证独立损失、梯度缩放、81 步学习、导出/加载和前置拒绝。

尚未包含官方 Atari/棋类观测预处理、多玩家 self-play 调度、异步多机 reanalysis
服务、跨任务泛化或真实环境成绩；这些不能由向量实例的通过推断出来。

来源：[MuZero 论文附录](https://arxiv.org/html/1911.08265v2)、
[DeepMind mctx](https://github.com/google-deepmind/mctx)。运行时不导入 JAX/mctx；
可选官方执行器对照在依赖缺失时明确跳过，不用自测冒充它。
