# MeanFlow / Shortcut 的训练制品、采样与评价

本模块把已有原生训练接入不可变制品、完整PNG样本清单和联合质量/性能门禁，
不加载外部生成pipeline。模型与训练数学见 `INTERVAL_GENERATION.md`；这里的
制品桥接不等于作者大规模预训练权重映射，也不证明已经达到公开ImageNet FID。

2026-08-30核对作者实现：[MeanFlow solver](https://github.com/Gsunshine/meanflow/blob/main/meanflow.py)
按给定递减时间表用区间平均速度更新；[Shortcut工程](https://github.com/kvfrans/shortcut-models/blob/main/train.py)
及 [推理入口](https://github.com/kvfrans/shortcut-models/blob/main/helper_inference.py)区分步数和步长条件。
生成清单固定实际本地源文件hash；链接用于说明出处，不把可变main分支当运行身份。

| 项目 | MeanFlow | Shortcut |
|---|---|---|
| 模型输出 | average_velocity | average_velocity |
| 采样方向 | noise=1 → data=0 | noise=0 → data=1 |
| 第三个输入 | duration = t-r | log2(1/dt) = log2(steps) |
| 采样计划 | 显式严格递减时间表 | 2的幂次步数，不超过训练base_steps |
| guidance | 训练目标内嵌；不增加推理CFG字段 | 0只null、1只conditional、其它scale两次预测后插值 |

## 真实训练到制品

普通MeanFlow发布会核对Trainer最后一次成功phase的真实目标class/config与当前
role update clock；若用临时override训练后导出另一份default，明确拒绝。记录随
contract持久化并在消费时再检查，checkpoint恢复同一记录。它只证明最后成功
更新，不能推导所有早期训练目标相同；Shortcut继续使用其专属Method生命周期。

```python
from aster.core import ArtifactStore
from aster.models.interval_dit import IntervalDiT, IntervalDiTConfig
from aster.methods.meanflow import MeanFlowObjective
from aster.training import Trainer
from aster.evaluation.interval_generation import publish_meanflow_generator

engine = Trainer(
    IntervalDiT(IntervalDiTConfig(variant='meanflow', in_channels=3)),
    MeanFlowObjective(), lr=1e-4, ema_decay=.999,
)
# batch来自实际数据管线：sample为FP32 BCHW，labels为对应真实int64类别。
engine.step([batch])
checkpoint = engine.save_checkpoint('runs/meanflow/checkpoint')
store = ArtifactStore('artifacts')
artifact = publish_meanflow_generator(engine, store, 'runs/meanflow/export', ema=True)
```

Shortcut用 `publish_shortcut_generator(method, store, directory, ema=False)`。
二者均要求至少一次成功更新，所有DP ranks共同调用并复用Trainer通信组，只有leader
发布目录，最后所有rank验证同一制品。支持DP/ZeRO，明确拒绝未实现的TP/PP/CP/GTP。

`ema=True`选择Trainer主模型维护的EMA，绝不偷偷选择Shortcut用于自举的
`shortcut_target`角色；未启用主模型EMA时拒绝。导出逻辑权重保持真实FP32存储，
BF16 autocast训练可用；不是将BF16文件悄悄转成FP32后宣称原始精度一样。
权重聚合需要临时完整CPU字典，未宣称流式低峰值导出。

contract固定模型配置、真实目标配置、完整更新计数、普通/EMA选择、权重指纹、
方向和第三输入语义。加载时重新计算权重指纹，旧contract不能挂到新权重上。
部署目录不含optimizer、队列、训练RNG；恢复须用完整Trainer checkpoint。
内容hash保证字节一致，不证明调用者没有主动替换过Python对象或伪造训练历史，
工程发布仍须来自受控宿主、保存对应运行/checkpoint来源。

## 采样与固定完整样本集

```python
from aster.evaluation.generative import GenerationCase
from aster.evaluation.interval_generation import (
    MeanFlowSamplingPlan, ShortcutSamplingPlan,
    generate_interval_shard, merge_interval_shards,
)

cases = tuple(GenerationCase(f'case-{i}', 10000+i, i % 1000) for i in range(1000))
plan = MeanFlowSamplingPlan(cases, noise_shape=(3, 32, 32), timesteps=(1., .5, 0.))
# ShortcutSamplingPlan(cases, (3,32,32), steps=4, guidance_scale=1.5)
for rank in range(2):
    generate_interval_shard(store, artifact.id, plan, f'run/part-{rank}', rank=rank, world_size=2)
manifest = merge_interval_shards(['run/part-0', 'run/part-1'], plan, 'run/images')
```

上例是执行接口，不是推荐的公开评价样本数/类别数；实际应先登记数据协议所需的
完整集合。每例独立seed、FP32标准高斯、B=1；记录实际noise SHA256及hook观察的
模型调用次数NFE，而不是仅把steps填进报告。采样失败仍留在全集中，不能重抽后
冒充原计划。每个rank包括空rank必须合并，跨权重/代码/环境混合分片拒绝。

可传 `decoder_artifact_id` 使用独立固定的原生KL-VAE，执行 `decode(..., scaled=True)`，
模型与decoder两个身份进入血缘。需预先声明latent形状与真实训练预处理；没有
指定decoder时要求输出能形成RGB PNG，不自动猜VAE来源、latent scale或值域。

同cases、seed与quantization共享cohort ID；MeanFlow时间表/Shortcut步数/CFG属于
不同plan身份，不能混淆。FP32 CPU/GPU高斯结果不保证逐字节相同，因此另存noise hash
和实际执行环境。PNG量化明确选择，不在评价时偷偷变更resize/rounding。

## 性能与联合门禁

`benchmark_image_sampler`现接受这两类typed plan，实际调用相同native sampler。
每次预热和测量都保留全case×repeat，记录hook NFE与同步sampler+可选VAE延迟。
CPU显存字段为None；CUDA峰值需要实际CUDA运行。测量不包含加载、输入RNG、PNG编码。
普通开发测量默认没有晋级资格，宿主隔离必须真实由调用者提供。

`evaluate_media_directories`识别 `native_interval_generated` 并绑定完整生成记录；
`evaluate_generation_gate`识别两类plan。门禁只能使用真正官方clean-fid特征的完整
FID/KID集合统计，至少三组独立cohort，再联合检查质量非劣和实际资源改善。
参见 `GENERATION_GATE.md`。缺公开源/权重/数据/授权一律 `not_evaluated`，不是PASS。

## 测试边界

`tests/integration/test_interval_generation_evaluation.py`真实微型训练后对照独立采样公式
及PNG，测试真实NFE、噪声身份、分片/空rank、完整失败集、普通/EMA、BF16 autocast、
ZeRO3、checkpoint恢复后相同制品、DP2 collective EMA发布及性能/gate资源读取。
错误方向、时间表、未训练的Shortcut步长层、旧权重contract都必须拒绝。

这些测试不是公开FID/KID成绩，也不是GPU性能结论。尚未覆盖作者生产checkpoint
映射、生产规模数据训练、公开质量验证，以及分布式TP/PP采样和GPU专用优化。
