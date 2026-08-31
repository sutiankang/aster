# TD3、TD3+BC 与 IQL：原生多角色训练

实现位于 `src/aster/methods/offline.py`，不导入作者仓库的 learner。训练角色、
优化器、梯度归约、目标网络及检查点都归同一个 `Trainer` 管理。此页仅描述
已实现的向量 MLP 支持域，不表示 D4RL 得分、GPU 性能或所有模型组合已复现。

## 公式与角色边界

TD3 的 Q 目标为 `r + discount * min(Q1_target, Q2_target)`。目标动作来自冻结
actor，并加裁剪高斯噪声后裁剪到动作范围；两份 Q 均使用均方误差。只有每
`policy_delay` 次成功的 critic 更新才更新 actor，再进行 actor/critic Polyak
平均。TD3+BC 增加行为克隆项，并用 `alpha / mean(abs(Q1(s, pi(s))))` 缩放策略
Q 目标。这个分母在**全部累积 microbatch 和 DP 域**上先求总和/计数；不能分别
求各小批次的 lambda，也不能对不等样本数的各 rank 均值做简单平均。

IQL 严格按 `expectile V -> advantage-weighted BC actor -> bootstrap Q -> target Q`
顺序执行。actor 使用刚更新的 V 和旧的 target Q；Q 使用刚更新的 V。默认策略
是 `Normal(tanh(mean_network(s)), exp(log_std))`，不是 SAC 的 tanh 变换分布，
因此动作似然没有 SAC 的 Jacobian 项；部署抽样再 clip 到 `[-1,1]`。

std 是唯一可训练的状态无关向量，放在可拦截 `forward` 的独立纯叶子中，避免
ZeRO-3 读取父模块裸参数而绕过 gather/release。所有训练损失用 FP32 汇总、
int64 计数，避免 BF16 将 257 个样本错误舍入为 256。额外的 TD3+BC 尺度统计也
遵守与 Trainer 相同的 autocast。目标网络由 `clone_target/update_target` 复制
逻辑权重，不会对空的 ZeRO-3 占位参数做 `deepcopy`。

## 构造与数据

```python
import torch
from aster.training import Trainer, ParallelContext
from aster.methods.offline import IQLActor, ContinuousTwinQ, StateValue, IQLMethod

# 先按部署环境初始化 torch.distributed；ParallelContext读取现有进程组。
# 每个rank都运行这段代码；不是仅主rank调用。
context = ParallelContext()
def adam(parameters):
    return torch.optim.Adam(parameters, lr=3e-4)

engine = Trainer(IQLActor(17, 6), parallel=context, zero_stage=3,
                 optimizer_factory=adam, accumulation_steps=2, max_grad_norm=None)
method = IQLMethod(engine, ContinuousTwinQ(17, 6), StateValue(17),
                   critic_optimizer_factory=adam, value_optimizer_factory=adam)
result = method.update(local_microbatches)
engine.save_checkpoint(checkpoint_path)  # 所有rank共同保存一个提交清单
```

TD3Method 接受 `critic_optimizer` 或 `critic_optimizer_factory`；IQLMethod 另接受
`value_optimizer` 或 `value_optimizer_factory`。每对参数只能选一个。没有指定时，
critic/value 使用 `Adam(lr=engine.lr)`，不是带默认衰减的 AdamW。actor 优化器仍
由 Trainer 的调用者明确选择；要复现作者 Adam 配方，不能使用 Trainer 默认
AdamW 然后声称算法更新相同。工厂在 ZeRO-3 替换参数后收到真实分片参数。

每个 microbatch 含 `observations[B,O]`、`next_observations[B,O]`、`actions[B,A]`、
`rewards[B]`、布尔 `terminated[B]`。前四项必须为有限浮点张量且在 Trainer 设备上；
动作范围和输入维度按模型配置校验。未知字段、错误设备、非有限值在任何模型
forward 之前全 rank 对称拒绝。每个 rank 的 microbatch 个数必须相同且等于
`accumulation_steps`；局部 B 可以不同或为零，但全局至少有一个 transition。

可选 `truncated[B]` 必须为布尔值；时间限制**不关闭 bootstrap**。调用者必须
提供截断时的 final observation，而不是 reset 后的 observation。可选 `discounts[B]`
表示已包含 gamma/gamma**n 和 terminal mask 的完整系数，必须在 `[0,1]`；真
terminal 对应系数必须为零。未指定时使用 `gamma * (~terminated)`。

## 恢复不是“重建优化器再加载权重”

原生 checkpoint 保存所有训练/冻结角色、优化器 moment、随机数、更新计数和方法
参数。模型顶层配置同时记录动作支持域、std 边界等同 shape 也会改变语义的字段。
DDP/ZeRO 的 rank、精度、累积和所有权布局必须与保存时一致；方法注册的状态不会
被 portable 迁移静默丢掉，目前需要用原生同拓扑检查点精确续跑。

多个 phase 不是数据库式原子事务。后半轮失败时，前半轮权重可能已更新，随机数
也可能已推进。因此方法从第一次 forward 前标记 `incomplete`，只有全部必要
phase 和目标更新成功后才清除。梯度 overflow 导致跳过任一 phase 同样失败；
继续 update 和保存 checkpoint 都会被拒绝，必须恢复最后一个完整检查点。不得
把此时内存中的半轮策略导出/发布。输入预检失败尚未进入计算，不污染此前状态。

## 验证与剩余边界

- `tests/unit/test_offline.py`：三种方法 × ZeRO0–3 的实际更新/随机 next-step
  精确恢复；BF16 × ZeRO0/3；std 梯度、计数、终止/截断、同 shape 配置拒绝、
  实际半轮非有限梯度失败和完整恢复。
- `tests/distributed/test_offline_distributed.py`：真实 Gloo 两进程，三种方法 ×
  ZeRO0–3，与独立完整批次 Torch/Adam 公式 oracle 比较所有角色权重；2/4 不等
  rank 样本量、1+3 不等 microbatch、ZeRO-3 空 rank、真实参数释放、保存续跑、
  单 rank 数据/overflow 错误，以及不支持拓扑的提前拒绝。
- 这里只支持 native 向量 MLP 的 DP/ZeRO0–3。TP、PP、SP、CP、EP、GTP 没有
  为这些 actor/critic 定义分片策略，不因为底层存在相关算子就宣称能任意组合。
- 目标网络目前完整驻留；没有伪称 target 同样 ZeRO 分片。CPU/disk offload 是
  Trainer 的独立能力，本次方法测试没有覆盖每个 offload/精度/拓扑的笛卡尔积。
- 未提供环境采集、D4RL 数据许可/归一化、奖励变换、离线策略评价或公开控制任务
  得分的整套协议。本测试是数值与工程完整性验证，不是公开 benchmark。
- 本机虽有物理 RTX2060，此次 torch2.11.0+cpu 没有使用 GPU；没有 CUDA/NCCL/
  多机吞吐、显存峰值或公开任务质量实测，不能据此宣称硬件性能等价。

## 公开来源

本次于 2026-08-30 实际读取以下作者源码；链接为分支视图，不冒称不可变 commit
pin 或已经执行作者环境。核心目标由独立原生代码实现，没有复制完整仓库。

- [sfujim/TD3 — TD3.py](https://github.com/sfujim/TD3/blob/master/TD3.py)
- [sfujim/TD3_BC — TD3_BC.py](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py)
- [IQL learner 更新顺序](https://github.com/ikostrikov/implicit_q_learning/blob/master/learner.py)、
  [critic 目标](https://github.com/ikostrikov/implicit_q_learning/blob/master/critic.py)、
  [actor 优势权重](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py)、
  [策略分布](https://github.com/ikostrikov/implicit_q_learning/blob/master/policy.py)。
