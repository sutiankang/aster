# PlaNet：连续状态、训练与规划

此实现与离散 DreamerV3 RSSM 是不同模型。生产代码只调用本仓库和 PyTorch 原语，
不在运行时导入、下载或执行 PlaNet/TensorFlow 仓库。

## 来源与公式

主要来源为 [google-research/planet](https://github.com/google-research/planet/tree/c04226b6db136f5269625378cd6a0aa875a92842)，
固定提交 `c04226b6db136f5269625378cd6a0aa875a92842`：

- `models/rssm.py`：连续高斯 prior/posterior、softplus(std)+0.1、确定性 belief、可选 mean-only/future-rnn。
- `networks/conv_ha.py`：64×64 像素编码器与反卷积解码器，单位方差 Gaussian likelihood。
- `networks/basic.py`、`scripts/configs.py`：奖励 MLP、权重、free-nats 和网络默认参数。
- `tools/preprocess.py`：5-bit 量化和 uniform dequantization。
- `tools/overshooting.py`、`training/utility.py`：多步 latent 预测。
- `control/planning.py`：CEM 候选采样、截断、精英重拟合与规划。
- `control/mpc_agent.py`、`random_episodes.py`、`wrappers.py`：真实观测更新信念、探索和episode字段对齐。
- `tools/numpy_episodes.py` 的 reload_loader 与 `tools/chunk_sequence.py`：随机episode轮换和连续分块。

GRU 另核对 TensorFlow `v1.13.1` 的
[`gru_ops.py`](https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/contrib/rnn/python/ops/gru_ops.py)：
reset 先乘 hidden 再进入 candidate 矩阵。不能以 PyTorch 默认 GRUCell 直接替代。
以上源码 Apache-2.0；PlaNet Authors 2019、TensorFlow Authors 2016 的来源声明保留于此。
正式对外发布仍受仓库总许可审计约束。

## 共享生命周期

`PlaNetConfig` / `PlaNetWorldModel` 已接入 `build_model`、配置解析、`save_pretrained`
和安全本地加载。`PlaNetObjective` 已接入目标工厂 `name='planet'`，并使用统一
`Trainer` 的累积、混合精度、ZeRO、EMA、检查点与精确恢复。

```python
import torch
from aster.models.planet import PlaNetConfig, PlaNetWorldModel
from aster.methods.planet import PlaNetObjective, preprocess_planet_images
from aster.training import Trainer
from aster.planning.planet import planet_cem_plan

model = PlaNetWorldModel(PlaNetConfig(action_dim=6))
engine = Trainer(model, PlaNetObjective(sequence_length=50),
    optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-3, eps=1e-4),
    max_grad_norm=1000.)
# data 必须来自明确的本地训练集/环境轨迹；不自动下载数据或启动真实设备。
# data = {observations: [B,50,3,64,64], previous_actions: [B,50,6],
#         rewards: [B,50], is_first: bool[B,50], valid: bool[B,50] (可选)}
# uint8 图像先通过 preprocess_planet_images 转为 [-.5,.5) float。
# engine.step([data])
# 在环境观测后更新 posterior，再调用 planet_cem_plan，只执行返回的首动作。
```

`previous_actions[:,t]` 导致 `observations[:,t]`，与许多 replay 中“当前观测后的动作”
约定不同，不能直接重命名字段而不移位。首帧 reset 同时清空 state 和前动作。
`observation_dim>0` 是明确标记的向量观测适配器，不能据此宣称像素基准成绩。

图像负对数似然按像素/通道求和；奖励使用原生连续 Gaussian，不用 symlog 或 two-hot。
损失分母是 int64 有效 transition 数。固定 `sequence_length` 是训练协议的一部分，
完整累积窗口在分布式参数 gather 前检查，避免不同长度触发不同通信顺序。

## Overshooting 与上游旧代码的差异

`overshooting_distance=D` 比较距离 2..D 的 prior 与 posterior；目标 posterior 可停止
梯度，但起点 posterior 保持梯度。默认关闭（同官方默认配置）。

公开旧 `training/utility.py` 取得了 overshooting mask，却未用于最终 mean。这里做了
显式工程修正：排除越界、padding、跨 episode 的 pair，并按有效 pair 独立归一化。
因此不宣称与旧源码有 padding 时的数值逐位相同；有效 pair 上的高斯 KL 与展开公式相同。
这一区别保存在目标配置中，不能在恢复时无声切换。

CEM 使用无折扣预测奖励和，不复用带 continuation/discount 的 Dreamer 规划器。多个
环境各自维护起点与候选序列；只运行奖励头，不为所有候选解码图像。给定 generator
可复现计划，调用结束后恢复原有 train/eval 模式。

## 已验与仍待完成

`methods/planet_loop.py` 现把模拟器→episode回放→训练→刷新冻结MPC模型连成真实流程。
调用者显式提供可恢复的模拟器；本仓不会连接真实设备。环境返回原始 uint8 像素或
声明的 float32 向量，reset行的前动作/reward置零，最后观测保留，terminated与truncated
分别保存。随机seed采集不调用世界模型；MPC用最新真实观测及上一执行动作更新信念，
然后采样CEM计划、加探索噪声并裁剪至归一化动作范围。

`PlaNetReplay` 保留官方reload式episode随机轮换和连续chunk规则：默认取
`max(1,L//T-1)`个chunk，随机offset可落在完整合法范围，不是逐窗口独立乱采。
长度不足明确拒绝，不用复制帧冒充原始分布。像素在采样时重新dequantize；episode、
pending chunks、顺序、RNG、模拟器状态、正在收集的历史和信念都进入共享checkpoint。
训练未提交时不误推进replay游标；模拟器已执行后失败必须恢复完整checkpoint。
`refresh_world()` 只允许所有rank都在episode边界进行，避免中途换模型导致信念失效。

最小循环：建立`PlaNetLoop(engine, env, replay)`，先`collect_steps(..., random=True)`，
再`train_step()`；episode边界`refresh_world()`后`collect_steps(...)`使用MPC。
可用固定预算自行交替循环，保存仍走`engine.save_checkpoint`。默认batch50/sequence50
适配公开配方，测试使用小尺寸；这不是公开DMControl效果已经达标的声明。

- 64×64 真像素前向、全参数梯度、本地保存/加载。
- 独立官方公式转写：连续 RSSM 全图/参数/输入梯度、像素 NHWC flatten 与反卷积图。
- FP32/BF16、ZeRO0/3、累积、EMA 与随机下一步完全恢复。
- 真实双进程 ZeRO0–3，不等 batch/valid 数、跨 episode mask，与完整 batch SGD 更新对照。
- 小型条件动力学学习实验，仅用于检测训练链路；不作为公开任务性能成绩。
- 像素/向量真实采集→回放→学习→MPC及episode中途恢复，独立采集RNG不污染训练RNG。
- 真实DP2全ZeRO0–3，不等本地batch更新对照、完整采集恢复、单rank空replay对称拒绝。

尚未完成官方 TensorFlow1 运行时对照、预训练权重映射、真实 DMControl 全配方/公开
episode 分数、多机 CUDA 性能、TP/PP 模型提供器、NVMe/异步采集和跨拓扑环境迁移。
当前不是“完整 PlaNet agent 已达到论文效果”的声明，相关要求继续保留在能力清单。
