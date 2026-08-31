# Consistency：制品、真实采样与联合评价

`evaluation.consistency_generation`消费本仓库 `ConsistencyMethod` 的CT/CD/iCT训练结果，
连接标准ArtifactStore、PNG完整样本清单、原生性能测量与公开FID/KID联合门禁。
算法由 `methods.consistency` 实现，不调用OpenAI、Diffusers或其他外部生成运行时。

2026-08-30核验的采样依据是[锁定的OpenAI源码](https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/karras_diffusion.py)：
边界预条件、`250*log(sigma)`网络时间与去噪后重新注入噪声。原入口以课程索引映射
实际sigma；本接口要求显式实际sigma，避免“采样steps”和“训练课程level数”混淆。
这里的iCT训练依据作者论文，不存在已验证的iCT官方代码oracle；不能借旧CT源码
名义声明iCT全部官方工程一致。训练公式和已验证范围见 `CONSISTENCY.md`。

## 三个冻结角色不能混用

| 角色 | 用途 | 能否作为部署生成器 |
|---|---|---|
| `model` | 优化器更新的student | 可显式选择 |
| `consistency_ema` | 独立采样EMA | 配置启用时默认使用 |
| `consistency_target` | 相邻sigma训练目标 | 明确拒绝 |
| `consistency_teacher` | CD的EDM教师 | 明确拒绝 |

iCT的训练target每轮跟随当前权重，并不意味着采样EMA也必须零衰减。
CT/CD/iCT模式、训练配置、teacher实际权重摘要和所有更新时钟由Method核验，
方法未注册、零次训练、半轮、过期target/EMA或外部改过的teacher均不能发布。

```python
from aster.core import ArtifactStore
from aster.evaluation.consistency_generation import publish_consistency_generator

# method是已经真实训练成功的ConsistencyMethod；所有DP ranks共同调用。
store = ArtifactStore('artifacts')
artifact = publish_consistency_generator(method, store, 'runs/consistency/export')
# 普通权重可显式 sampling_role='model'。
# 未启用采样EMA时默认model；强行选择不存在的consistency_ema会拒绝。
```

发布复用Trainer的DP/ZeRO通信域，leader单次写入，所有rank核验同一artifact ID。
FP32存储权重逐张量导出，不通过悄悄换dtype伪造同权重；BF16 autocast训练可用。
完整CPU权重聚合仍需要内存，未宣传为低峰值流式导出。临时模型初始化用隔离RNG，
不会改变下一次训练随机序列。部署不含优化器/训练RNG，精确恢复使用完整checkpoint。

根目录保存原 `method.export_config()` 的 `consistency.json`，同时保存更严格的
`generation_contract.json`：所选角色、真实权重指纹、完整模型配置、时间单位及训练
身份。加载重新计算指纹，不能给另一份权重挂旧contract。只有手工model文件和旧
`consistency.json`、缺少权重绑定时，不自动推断其实际角色或训练来源。

CD可提供 `teacher_artifact_id`，发布前加载该本地原生制品并逐权重核对实际冻结教师；
不匹配拒绝，匹配后进入parent血缘。省略时仍保存Method真实教师摘要，但不宣称
自动绑定了某个teacher制品。CT/iCT不能凭空添加教师。teacher预训练质量/数据合规
不是hash能够证明的，测试中的本地微型教师不叫生产预训练教师。

## 精确sigma与随机数语义

```python
from aster.evaluation.generative import GenerationCase
from aster.evaluation.consistency_generation import (
    ConsistencySamplingPlan, generate_consistency_shard, merge_consistency_shards,
)

cases = tuple(GenerationCase(f'case-{i}', 10000+i) for i in range(1000))
plan = ConsistencySamplingPlan(
    cases, noise_shape=(3, 32, 32), sigmas=(80., .821), clip_denoised=True,
)
generate_consistency_shard(store, artifact.id, plan, 'run/rank0')
manifest = merge_consistency_shards(['run/rank0'], plan, 'run/images')
```

上例的形状、80和.821不是通用推荐值，须匹配实际模型与已声明协议。
首个sigma必须等于训练配置 `sigma_max`，其余严格递减且不低于 `sigma_min`。
不会根据len(sigmas)重建Karras课程，也不会自动添加0、DDIM端点或隐藏模型调用。
实际NFE为 `len(sigmas)`；每张图用forward hook实测核验。

输入是单位FP32高斯，sampler自己乘首个sigma。随后每次去噪后按下一个sigma注入
独立噪声。初始噪声和后续噪声共用每例独立generator，记录初始噪声hash、进入和
离开sampler的RNG状态hash。只有一步时前后状态相同，多步消费量由真实路径决定。
上游最后会进行一次乘零的随机生成，本接口不做这次无效消耗；每例独立RNG保证不
影响其它样本，但不声称随机状态与上游逐字节一致。

student使用自己的 `time_scale`，默认250；CD teacher可能按本仓库EDM `.25`单位
训练，二者分别保存，推理不能拿teacher单位替换student单位。参数化必须明确为
`consistency_residual`，不能拿 `edm_residual`、`x0`、epsilon或velocity偷换。

condition可为显式None、模型支持的真实类别或固定宽度向量，不接受越界类别后
静默转null，也不猜文本embedding来源。可另传 `decoder_artifact_id`，使用固定原生
KL-VAE的 `decode(scaled=True)`；latent需显式匹配，VAE缩放/偏移不从模型名称猜测。
clip在去噪后执行；latent不应无依据地套像素clip范围。PNG量化规则单独固定。

## 分片、性能和评价

支持固定 `rank/world_size` 分片；每例独立seed，因此单片/多片相同输入产生同样图。
失败不重抽、不丢弃，所有rank包括空rank必须合并。模型、decoder、源码、环境和
完整计划随清单记录，损坏/重排/缺失样本不能进入公共指标。

`benchmark_image_sampler`执行同一native sampler，保留全case×repeat，记录真实NFE、
预热后的同步sampler+可选VAE延迟，CUDA存在时记录实际allocator峰值；CPU不会填写
虚构显存。计时不含模型加载、初始噪声构造或PNG写盘。

`evaluate_media_directories`已识别 `native_consistency_generated`；
`evaluate_generation_gate`已识别 `ConsistencySamplingPlan`。同cases/seed/量化的不同
步数、EMA选择或训练模式能共享cohort，但各自完整plan和模型身份不同。联合门禁
要求质量与性能来自同一源码/权重/采样配置和环境。官方源、特征权重、参考集或
授权缺失时返回 `not_evaluated`，不以toy距离替代公共FID/KID，更不自动部署。

## 本地验收与未完成部分

新增集成测试运行真实CT/CD/iCT微型训练，对照独立预条件和噪声公式、真实PNG、
NFE/RNG、普通/EMA、checkpoint恢复、UNet与DiT、BF16 autocast、DP2 ZeRO3发布。
另外测试角色误用、旧权重contract、错误时间单位、错误教师、越界sigma、失败全集
与性能/gate资源读取。本包不下载公开数据/权重，不生成任何公共FID成绩。

仍不等于生产效果验收：官方大模型权重映射、完整作者架构配方、LCM/CTM流程、
TP/PP/EP生成、GPU性能、多机通信和公开质量/资源联合实验尚需各自实施与硬件证据。
CD教师的摘要/制品血缘不等于已经完成教师Heun与学生的公开质量/速度对比。教师的
EDM typed PNG/性能桥接已由独立 `evaluation.edm_generation` 接通，见
`EDM_EVALUATION.md`；它严格使用教师预条件和真实Heun NFE，不借用Consistency
student的时间尺度。公开质量/资源联合实验仍须真实数据、官方特征权重和硬件授权。
