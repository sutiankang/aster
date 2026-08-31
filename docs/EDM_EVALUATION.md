# EDM教师基线与Consistency学生的同协议比较

`evaluation.edm_generation`使用本仓库原生 `EDMObjective`、UNet2D/DiT和
`sample_edm`；不调用外部pipeline，不下载权重。新增内容是实际训练制品发布、
明确的Heun采样计划、逐样本PNG/NFE/RNG证据，以及既有公共质量/性能门禁接入。

## 算法依据与边界

参考[EDM作者官方采样器](https://github.com/NVlabs/edm/blob/main/generate.py)的
Heun预测/校正、最后一步Euler和churn区间约定，并核对
[OpenAI锁定版本Heun实现](https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/karras_diffusion.py)。
2026-08-30核验；NVIDIA页面链接是上游移动分支，不冒充固定源码版本。
实际执行的是仓库现有原生数学实现，PNG和性能记录锁定所有实际消费本地源文件的SHA256。

NVIDIA上游[许可证](https://github.com/NVlabs/edm/blob/main/LICENSE.txt)为
CC BY-NC-SA 4.0，而不是MIT。本包没有复制或vendor上游源码；没有授予官方权重、
数据或衍生代码新的商业许可。需要使用上游材料时必须另外核对对应许可证。

原生此路径为FP32网络/采样状态，sigma表为FP64；上游默认FP64采样累积，且有
网络sigma范围裁剪和 `round_sigma`。此路径的连续native模型不做未知权重网络的
sigma rounding，因此不宣称上游任意权重逐位兼容，也不宣称完整工程/图像质量等价。
它不是Cosmos-Predict1专属Euler：后者初始幅度和guidance约定不同，另有独立接口。

## 从真实更新发布

```python
from aster.evaluation.edm_generation import publish_edm_generator

# engine已用原生EDMObjective完成更新；所有DP ranks一起调用。
teacher = publish_edm_generator(engine, store, 'run/teacher-export', ema=True)
```

默认发布普通model，EMA需显式选择且必须真实存在。发布拒绝未更新、失败/忙碌状态、
错误参数化、非FP32存储和未接通的TP/PP/CP/EP/GTP部署。DP/ZeRO使用现成通信域，
逻辑权重导出后leader写入；不是另造训练器。BF16 autocast可用，最终存储仍为FP32。
部署只保存模型/训练目标/角色身份；继续训练必须加载完整Trainer checkpoint。

`objective.json`保留真正EDM目标的 `sigma_data/log_mean/log_std`，
`generation_contract.json`记录实际选择的权重指纹和更新计数。消费时重新核对。
发布还核对 `Trainer.last_successful_update()`：记录必须是该角色当前update clock，
并与最后实际phase执行前冻结的目标class/codec/config一致。训练使用临时override、
导出却读取另一份default，或更新后修改sigma_data，都会拒绝。记录同时保存为
`successful_update.json`并进入contract；原生checkpoint恢复同一记录，旧checkpoint
缺证据时不补猜。它只证明最后成功更新，不保证早期步骤目标不变或完整历史合规。
已经由 `tensor_fit` 或本地流程发布、带真实 `objective.json` 的普通native模型也可
采样；它被明确标记为“制品中的目标声明，不是完整训练历史证明”，不会补造更新计数。
新tensor_fit同时保存最后实际成功目标记录；旧手工制品没有时，采样绑定明确
`actual_successful_objective_bound=False`，不会把旧声明升级为已验证phase来源。

若目标是 `LatentFieldObjective`，必须带内容寻址的VAE identity。发布加载此制品并
逐权重核对实际冻结训练编码器；消费和性能测量必须指定同一个decoder。解码使用
native KL-VAE的 `decode(scaled=True)`，不猜scale/shift。普通像素EDM后另配VAE
只属于显式调用者组合，不自动构成潜空间训练证明。

## 实际sigma、churn和NFE

```python
from aster.evaluation.generative import GenerationCase
from aster.evaluation.edm_generation import EDMSamplingPlan, generate_edm_shard, merge_edm_shards

cases = tuple(GenerationCase(f'image-{i}', 10000+i) for i in range(1000))
plan = EDMSamplingPlan(cases, (3, 32, 32), sigmas=(80., 10., 1., .002, 0.))
generate_edm_shard(store, teacher.id, plan, 'run/teacher-rank0')
images = merge_edm_shards(['run/teacher-rank0'], plan, 'run/teacher-images')
```

数字仅演示接口，不是质量推荐配方。计划包含至少两个正sigma和最终0，严格递减。
EDM训练连续log-normal分布没有离散训练betas，不能从所谓训练steps推导出采样表。
实际输入是每例独立seed的单位FP32高斯，sampler乘第一个sigma。时间输入固定
`log(sigma)/4`，不是Consistency student默认的 `250*log(sigma)`。

每个正sigma做一次预测；除最后到0外，还做一次校正，所以正sigma数N对应
`2*N-1`次网络调用。实际NFE由forward hook测量，不直接把公式填进性能表。
本native sampler每个区间均消耗一次churn随机张量，churn=0时乘零但仍推进RNG；
记录进入/离开sampler状态hash。每例独立generator保证分片不改变其它样本随机序列。
`churn_max=None`表达无上界，JSON中不写Infinity；采样时才转为内存中的无穷界。

条件只接受模型实际支持的类别/固定宽度向量或显式None，不猜文字embedding。
本typed plan不提供虚假的CFG标志；尚未接通的EDM guidance组合不能忽略参数后运行。
失败样本不重抽、不丢弃，所有分片包含空rank一同合并，完整计划/源码/制品/环境必须相同。

## 同一教师的真实CD学生比较

```python
from aster.evaluation.edm_generation import validate_consistency_teacher_baseline

comparison = validate_consistency_teacher_baseline(
    store, teacher.id, student.id, teacher_plan, student_plan,
)
```

student必须由 `publish_consistency_generator(..., teacher_artifact_id=teacher.id)`
绑定过真实CD教师。检查实际教师权重、CD teacher时间单位、sigma_data、同cohort、
初始sigma和噪声几何；CT/iCT或另一份同架构权重不能冒充此教师。返回值明确尚未评价。
随后两种原生producer分别生成同cases/seed/量化的PNG，并由
`benchmark_image_sampler`测量各自真实NFE、同步延迟和可用时的CUDA分配峰值。

两条路径已进入 `evaluate_media_directories` 与 `evaluate_generation_gate`。
门禁仍需至少三个独立完整cohort、固定官方特征器/权重/数据版本、对应完整性能矩阵
和显式执行授权。NFE减少不等于质量保持；官方资源缺失为 `not_evaluated`，绝不
用微型模型像素距离填FID，或把每图分数伪装成FID bootstrap。CPU不产生虚构显存。

本包测试真实训练/普通EMA/checkpoint恢复、UNet/DiT、churn公式与PNG、RNG/NFE、
分片含空rank、失败全集、BF16 autocast/ZeRO3、实际DP2导出及CD师生同噪声比较。
未执行任何公开FID/KID实验，未验证CUDA/多机性能，也未实现全部上游架构与权重映射。
