# 生成质量和真实资源改善的联合门禁

入口分为 `aster.evaluation.generation_performance.benchmark_image_sampler` 和 `aster.evaluation.generation_gate.evaluate_generation_gate`。现支持原生UNet/DiT的VP diffusion、Flow、DMD直接生成及独立Drifting；不是公共模型成绩，也未实现视频FVD晋级。

## 实际测量什么

性能运行加载固定模型/可选VAE，每个完整cohort先预热至少1轮，再测至少2轮。batch固定1，case×repetition完整矩阵保留，失败不能从均值中删掉。模型forward hook实际计数NFE；Euler/Heun/RK4和两分支CFG的公式仅用于检查计数，不拿配置步数冒充观测值。

延迟覆盖求解器或单次生成及可选VAE解码，不含权重加载、初始随机输入构造、输出搬到CPU/校验、图片编码或排队。CUDA显式同步。CUDA memory指标是重置后的 `max_memory_allocated` 绝对峰值，包含当前模型存储，不是reserved显存或NVML整卡占用。CPU返回 `None` 并标记 allocator峰值不可用，绝不拿Python内存代替。报告还保存真实输出shape/hash、硬件/软件/线程数、模型及采样绑定。

默认 `isolated_hardware_asserted=False` 只生成开发证据，不能晋级。宿主确认独占硬件、无同进程其他模型和稳定负载后才能显式声明True；模块锁只能阻止本模块并发调用，不是跨进程硬件沙箱。测试中的小模型耗时不是生产吞吐、TTFT或公开速度比较。

```python
from aster.evaluation.generation_performance import GenerationBenchmarkSettings, benchmark_image_sampler

def benchmark_controlled(store, model_id, sampling_plan, destination, *, device, decoder_id=None):
    # 仅在宿主真正安排了独占环境后调用此配置。
    settings = GenerationBenchmarkSettings(warmup_repetitions=2, repetitions=5,
                                            isolated_hardware_asserted=True)
    return benchmark_image_sampler(store, model_id, sampling_plan, settings, destination,
                                   device=device, decoder_artifact_id=decoder_id)
```

## 质量统计的单位是完整采样集合

事先固定至少3个独立seed cohort，每个cohort都分别运行baseline和candidate，保留同一完整样本/条件/seed集合。不同cohort的随机seed不重叠，参考集及其特征字节、生成样本数量、官方特征器/预处理/KID控制相同。当前默认至少3次是可执行下限，并不代表统计功效充分；重要发布应设计更多重复、预先确定容忍阈值和功效分析。

每次质量报告必须来自 `evaluate_media_directories` 并保留完整2048维Inception feature矩阵。门禁核验文件hash和完整分母，然后使用本地已审核的官方 `clean-fid.fid_from_feats` / `kernel_distance` 重新算每个cohort的FID/KID，核对报告值。不能直接提交一个 `passed=True` 或随便填的FID数字。

随后对**完整cohort之间的成对差值**进行bootstrap：质量使用baseline−candidate绝对差，资源使用 `(baseline−candidate)/baseline`。它不是把一个FID复制到每张图再做配对bootstrap，也不把图片当作有独立FID分数的样本。latency/NFE先在各完整cohort内汇总，memory使用该cohort观测峰值；置信区间的样本数始终是独立cohort数，不是trial或图片数。

多项质量/资源比较使用Bonferroni分配置信预算。有限样本bootstrap仅描述固定reference和已训练模型下的生成/运行重复性，不涵盖FID有限样本偏差、reference抽样、数据污染或训练seed变化，不是官方leaderboard统计保证。

## 执行与结果

```python
import time
from aster.evaluation.generation_gate import GenerationGateProtocol, evaluate_generation_gate
from aster.evaluation.suites import EvaluationGrant

def check_release(baseline_id, candidate_id, cohort_ids, quality_pairs, performance_pairs,
                  source_root, weights_path, destination):
    protocol = GenerationGateProtocol(
        baseline_id, candidate_id, tuple(cohort_ids),
        quality_max_regression=(('fid_clean', 1.0), ('kid_clean', 0.0005)),
        resource_max_relative_regression=(('latency_seconds', 0.05), ('nfe', 0.0)),
        required_relative_improvements=(('latency_seconds', 0.10),),
    )
    # quality_pairs/performance_pairs均为 {cohort_id: (baseline报告目录,candidate报告目录)}。
    # 宿主必须先审核本地官方源和资源；此许可仅批准数学API import，不执行archive。
    grant = EvaluationGrant(protocol.id, ('official_metric_recompute',), time.monotonic()+3600)
    return evaluate_generation_gate(protocol, quality_pairs, performance_pairs,
        source_root=source_root, weights_path=weights_path, grant=grant, output_directory=destination)
```

以上容忍值只是API示例，不是可普遍采用的发布标准。质量所有指标须非劣、受限资源不能超出回归容忍值，而且至少一项事先声明的资源改善必须严格大于零。仅减少NFE不能自动宣称延迟变快；若要求真实latency改善，必须有足够的实测改善证据。若memory是必需指标而只有CPU报告，则无法晋级。

- `promote`：全部质量/资源证据与阈值通过，`passed=True`。
- `reject`：已知失败、缺失样本/试次、身份/环境/数值不一致或阈值未通过。
- `not_evaluated`：官方源/权重/依赖/授权缺失，独立cohort不足，缺声明的报告或资源指标不可测。此状态始终 `passed=False`，绝不退回合成特征估算PASS。

报告固定每份原始质量/性能报告hash及每项区间；性能和PNG生产者必须匹配同一原生sampler源码hash，不能拼接旧实现的质量与新实现的速度。内容hash证明字节一致，不独立证明来源诚实或执行过程真实；受信宿主仍须管理ArtifactStore发布权限和执行环境。该包没有自动部署权限。

当前测试覆盖真实原生CPU forward计数/延迟和失败矩阵，以及统计/门禁拒绝逻辑。真正官方多cohort联合运行由 `ASTER_APPROVED_GENERATION_GATE` 指向已审核配置后执行；默认无资源时明确跳过，不制造公开FID/通过案例。CUDA峰值测试同样必须有CUDA设备，CPU不能替代验收。
