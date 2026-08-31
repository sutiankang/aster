# 一致性模型的训练、蒸馏和少步采样

`aster.methods.consistency`提供原生CT、CD和iCT生命周期，复用统一Trainer的参数、优化器、DP/ZeRO、混合精度、RNG和checkpoint。它不是LCM、CTM、DMD或Drifting的别名，也不因为一次小模型测试通过就声称复现论文FID。

## 来源和精确算法边界

- CD/旧CT基于[OpenAI锁定源码](https://github.com/openai/consistency_models/tree/e32b69ee436d518377db86fb2127a3972d0d8716)，MIT许可；阅读`cm/karras_diffusion.py`、`cm/script_util.py`和`cm/train_util.py`。固定源码文件SHA256为`b8fcf9f53e63cff19db676814545ee7644259364236de5c10ac3d69007ee5177`。
- iCT依据作者[论文v1的式8–10](https://arxiv.org/html/2310.14189v1)：当前权重stop-gradient目标、向量Pseudo-Huber、离散对数正态区间概率、倍增课程。未找到可核验的作者iCT训练源码；社区实现不作为“官方代码”证据。没有把旧CT的EMA滞后目标继续称为iCT。
- 网络接收`consistency_residual`；CD教师接收`edm_residual`。边界预条件保证`sigma_min`时恒等映射。时间单位显式分开：student默认`250*log(sigma)`，与OpenAI ADM一致；teacher默认`.25*log(sigma)`，与本仓库现有EDMObjective一致。如果teacher按其他时间单位训练，必须设置`teacher_time_scale`，不能靠猜测checkpoint名称转换。

这里复用本地UNet/DiT等条件场；没有把本地简化网络的权重名称说成能直接加载全部OpenAI ADM/NCSN++预训练权重。论文中的低敏感Fourier嵌入、按分辨率dropout和ImageNet去AdaGN的完整架构配方尚未在本模块补齐，不能把目标函数通过对照说成已复现整套论文训练配方。

| 模式 | 相邻样本 | target | 默认课程/指标 |
|---|---|---|---|
| CT | 同一clean与同一noise的相邻sigma | 自适应EMA冻结副本 | 原平方根课程、均匀区间、MSE |
| CD | 真实EDM teacher的Heun一步 | 固定衰减EMA冻结副本，默认衰减0 | 固定40个level、均匀区间、MSE |
| iCT | 同一clean与同一noise的相邻sigma | 每步复制当前student，衰减必须0 | 区间数倍增、lognormal、inverse-delta、vector Pseudo-Huber |

MSE/L1沿用源码逐样本像素均值；Pseudo-Huber先对整个样本的平方差求和再开根，不能用逐像素Huber均值替代。课程参数、指标、权重、时间单位、种子均进入checkpoint身份。`scales_and_ema(k)`中的k是已成功提交的更新数。

## 最小训练入口

```python
import torch
from aster.models.generative import UNet2D, UNetConfig
from aster.training import Trainer
from aster.methods.consistency import ConsistencyConfig, ConsistencyMethod

model_config = UNetConfig(prediction_type='consistency_residual')
model = UNet2D(model_config)
engine = Trainer(model, zero_stage=3, accumulation_steps=2,
                 optimizer_factory=lambda p: torch.optim.RAdam(p, lr=1e-4))
method = ConsistencyMethod(engine, target_factory=lambda: UNet2D(model_config),
                           config=ConsistencyConfig(mode='ict', total_steps=800000))
result = method.update([{'sample': images_a}, {'sample': images_b}])
engine.save_checkpoint('checkpoint.json')
```

DP场景在每rank构造同一ParallelContext并传入Trainer。各rank的B可不同，各microbatch B也可不同；loss分子按真实样本数归一化，计数是int64。整个窗口的shape、condition、有限值和课程游标在任何ZeRO物化前全rank检查。支持可选`noise`、`interval_indices`以建立严格数值对照；不传则从可恢复的rank-local CPU generator采样，之后移动到计算设备。相同rank、相同输入序列和相同拓扑才能声明随机序列精确恢复。

CD必须另传真实`teacher`且其配置声明`edm_residual`。`target_factory`返回同class、同配置的独立网络，运行时通过逻辑参数复制建立冻结target，不对ZeRO3空参数做deepcopy。采样EMA是独立`consistency_ema`角色，不能把它和iCT的零衰减训练target混为一谈；可显式设`sampling_ema=None`关闭。

target仍以train模式使用相同dropout RNG，但无梯度；EDM teacher固定eval。两次调用后恢复student前向之后的RNG状态，避免重复消耗随机序列。拒绝BatchNorm这类有统计写回的归一化，不能将副作用重算假装纯函数。

## 恢复、发布、评价

更新顺序为student optimizer→target→采样EMA；只有全部成功才增加method游标。异常或overflow后不假装原子回滚，而是同时封锁method checkpoint和Trainer直接导出，要求加载最后完整checkpoint。student角色更新时钟与method时钟绑定，外部另开phase更新student后不能继续使用过时iCT target。教师和冻结目标的实际张量摘要随状态记录，外部偷偷修改它们会失败。此安全参考实现逐轮检查完整冻结模型摘要，有显式CPU传输/扫描成本。

`engine.load_checkpoint(...)`同时恢复student/RAdam、所有冻结角色、模型配置、课程游标及两个RNG来源。跨DP或课程迁移不是精确续跑，当前注册method状态后不允许portable静默丢弃它们。

部署使用所有rank共同调用`engine.export_state_dict(role='consistency_ema')`；rank0用独立模型`load_state_dict(strict=True)`后`save_pretrained`，并把`method.export_config()`保存为`consistency.json`，一起交给`core.ArtifactStore.publish`。关闭采样EMA时导出`model`。这与其他生成模型使用同一种制品存储，不创建独立私有权重格式。

```python
from aster.methods.consistency import sample_consistency
one_step = sample_consistency(deployed, noise, [80.])
two_step = sample_consistency(deployed, noise, [80., .821], generator=generator)
```

`noise`是单位高斯；列表给出实际求值sigma，NFE恰为列表长度；多步按下个sigma重新注入噪声。`.821`是论文特定CIFAR配置的示例，不是所有模型的万能最优值。默认每次预测clip到[-1,1]，其他数据范围需显式关闭并固定到评测协议。低精度部署由调用方提供对应autocast上下文。

评价应复用`evaluation.generative`的同数据split/预处理/特征器权重固定协议，联合报告FID/KID、适用的IS/precision/recall、NFE和实测端到端时延。teacher和student用同一输入/种子预算比较。这里未下载公开数据或官方权重，未生成公共benchmark分数；单步能运行不代表质量与teacher相同。

## 验证范围

- `tests/unit/test_consistency.py`：公式、边界、采样NFE、真实UNet的CT/CD/iCT×ZeRO0–3、RAdam过rectification阶段、dropout、BF16、冻结角色身份、恢复、异常边界、标准制品与独立采样。
- `tests/distributed/test_consistency_distributed.py`：真实DP2/UNet/三模式×ZeRO0–3，非均匀样本计数、逐参数梯度/更新与dense对照、冻结目标、RNG续跑、失败前集体拒绝；另测RAdam DP2/ZeRO3→dense portable下一步。
- `tests/unit/test_training_radam.py`：RAdam ZeRO0–3×none/CPU/disk offload，完整param-group语义、8次更新、native恢复和portable，不是用Adam替代。
- `tests/integration/test_consistency_official.py`：只有显式`ASTER_RUN_REMOTE_CONSISTENCY_ORACLE=1`时读取完整hash锁定的OpenAI源码；实际执行原KarrasDenoiser的CT/CD L2分支，比较loss和每参数梯度。默认不联网，不安装piq/LPIPS，不能据此宣称LPIPS分支也验证。

跨布局的GroupNorm前bias存在约1e-11的浮点零梯度；默认RAdam eps在第6步rectification可把它放大成约9.36e-7参数差。此完整模型对照显式用eps=1e-4且保留严格梯度检查；默认eps仍独立覆盖恢复与迁移，不以放宽容差掩盖归约错误。

当前完整生命周期只开放DP×ZeRO，不开放TP/PP/CP/EP/GTP条件场组合。冻结teacher/target/采样EMA是完整副本，有真实内存成本；没有宣传为全部模型也已分片。LPIPS、连续时间CT、progressive-distillation、LCM的CFG/潜空间流程、官方权重转换和专用GPU融合尚未在本模块实现。本机有物理RTX2060，但当前torch为CPU构建；没有GPU、多机带宽或加速比实测证据。
