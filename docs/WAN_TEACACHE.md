# 原生 Wan2.1 TeaCache：近似推理优化闭环

这里接入的是训练完成后可选的**有误差残差复用**，不是知识蒸馏，不改变训练
权重，不减少求解器的时间步。标准无缓存入口仍保留。当前验收范围为本仓库
`WanVideoDiT` + `WanVideoVAE`、单进程、batch=1、FP32、原生 Euler/Heun。
没有声称已运行1.3B/14B权重、CUDA速度测试或公开FVD成绩。

## 官方来源与准确的适配边界

- [TeaCache固定版本源文件](https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4Wan2.1/teacache_generate.py)：
  commit `7c10efc4702c6b619f47805f7abe4a7a08085aa0`；实际1024行；SHA256
  `97af76136337869152f3d6fe9e049cadc2c480740c492749fcd5efa80d9bf7ee`。
- 519–568行是本包对照的决策与两分支残差复用块。源码版权为Alibaba Wan
  Team 2024–2025；TeaCache仓库为Apache-2.0，发布分发仍须保留相应许可声明。
- 默认探针是时间嵌入 **e**，retention模式使用时间投影 **e0**；不是首层
  AdaLN调制输入。默认首、末求值轮完整执行；retention前5轮完整执行，
  **不额外强制末轮刷新**。
- 条件/负条件各有previous probe、累计相对L1、previous residual。原式使用
  全tensor mean；多项式可为负，其负值会抵消累计量。没有擅自 `max(0, y)`。
- 官方预拟合系数区分T2V-1.3B、T2V-14B、I2V-480P、I2V-720P及两种模式；
  原官方生成使用UniPC/DPM++。本包**不提供可套到tiny/Euler/Heun的官方
系数捷径**，也不接受把目录名写成“1.3B”作为型号证明。必须对实际制品和
  当前求解器生成完整轨迹，再拟合自己的残差变化曲线。

`optimization/step_cache.py`继续是单独的原生2D DiT近似方法。它的首AdaLN
探针、最大batch相对误差、非负截断、max_skip等不是这里的Wan官方决策；
二者没有别名关系，也不能互换校准文件。

## 实际计算路径

原 `WanVideoDiT.forward`现在只组合三个原生方法，权重键没有变化：

1. `prepare`执行原输入校验、patch、文本/图像处理及e/e0。
2. `run_blocks`执行全部Transformer块；缓存命中时不调用它，而是当前
   patch hidden 加上该CFG分支上一次保存的主干残差。
3. `finish`始终执行**当前时间**的输出头和unpatchify；不缓存最终velocity。

没有给类全局挂`cnt`，没有调用/改写官方模型。`WanTeaCacheSession`用显式
`round_index`和`branch`校验顺序，模型/条件更改、维度更改或失败会封闭会话。
`reset(condition=..., negative_condition=...)`显式开始新请求并清空两支状态。
缓存对象不是可恢复的训练checkpoint，也不能接着失败的半轮运行。

Heun每个ODE步有两轮场求值，包括末端sigma=0；计数按真实求值轮，而不是
误用官方一轮一次预测的步数。CFG=1只有正分支，其他值包括0保留原生Wan
采样器的两次调用语义。此求解器身份被校准记录绑定。

`audit_every>0`对每若干候选命中执行完整主干；默认模式真实刷新时也会
检查复用输出的相对L1。超预算则本次两分支都停用缓存并记录`guard_failed`。
审计不是免费的：完整主干调用和额外head都计数。未审计的点没有逐点误差
保证，最终视频仍必须通过完整质量评价。

无缓存baseline不保存无用途的probe/residual，也不做缓存距离扫描；避免
人为放大基线内存和时间。计数分开记录：

- `field_calls`：求解器实际请求场输出的次数，cache hit也算一次；
- `full_backbone_calls`、`reused_backbone_calls`：完整主干/复用次数；
- `audit_backbone_calls`、`head_calls`：审计与输出头的真实额外成本。

## 制品、校准与完整评价

主接口位于`aster.evaluation.wan_teacache`：

```python
calibration = publish_wan_cache_calibration(
    store, field_id, calibration_plan, calibration_dir, mode="default"
)
settings = WanTeaCacheSettings(
    threshold=0.1, audit_every=3, maximum_relative_output_error=0.05
)
baseline = generate_wan_cache_cohort(store, field_id, vae_id, holdout_plan, baseline_dir)
candidate = generate_wan_cache_cohort(
    store, field_id, vae_id, holdout_plan, candidate_dir,
    calibration_artifact_id=calibration.id, cache_settings=settings
)
```

上述`VideoSamplingPlan`持有固定cases、seed、condition artifact、latent/output
几何、FPS、solver、shift、CFG、量化。条件仍是实际本地tensor制品；允许
声明编码器来源，但不会由声明推断已经运行官方T5/CLIP。校准保存所有
case×branch×round的实际测量、拟合参数与源码哈希；加载时重新验证多项式
确由记录的测量拟合，模型全部参数和非持久buffer也参与前向身份。
本包显式拒绝autocast；校准与生成的device、宿主、线程、TF32等数值环境
必须相同，换目标环境须重新校准。不能因为参数存储是FP32，就把autocast
产生的另一套计算精度误报为本包已经验证的FP32路径。

生成记录保存原始float32视频、连续PNG、实际计数与每次缓存决策。消费者
不只检查文件hash，还检查PNG每个像素确为该float视频按约定量化后的结果。
失败clip保留ID和全集分母，没有丢弃失败后重新抽seed。

`benchmark_wan_cache_cohort`真正运行相同计划，先预热再跑完整重复测量矩阵。
计时边界为**同步后的原生采样器（含缓存及审计）+VAE解码**，不含模型加载、
制品hash扫描、初始noise生成、PNG编码/文件IO；不是HTTP排队、TTFT或ITL。
CUDA时同步并记录allocator绝对峰值，包含常驻权重与两支缓存；CPU峰值为
`None`，不拿Python内存追踪伪装Torch显存。硬件独占是宿主显式声明，不是
本程序自动证明。每次性能测量的输出fingerprint/决策必须与质量产物相同。

`compare_wan_cache_cohorts`只允许相同模型、VAE、条件、solver、steps和CFG
下“无缓存→有缓存”的比较，不能混入少步或小模型的额外收益。它要求：

- 无失败样本/预热/计时轮；runtime guard未失败；完整主干NFE不回退；
- 校准与评价case ID/seed不重叠；不同评价cohort也不复用seed；
- 配对原始像素RMSE在预算内（只是回归诊断，不是公开感知分数）；
- 固定官方I3D源码、特征权重、参考全集、FPS、帧选择和执行授权后真实FVD；
- 至少3个独立完整cohort，按**整个cohort**聚合FVD和latency做区间；
  没有逐视频“FID/FVD”，没有对不存在的逐样本分数做bootstrap；
- 感知质量非劣与真实latency改善同时成立，可选实际CUDA内存回退限制。

`evaluate_wan_cache_fvd(..., resources=None)`明确`not_evaluated`、空指标。
没有获批I3D资源/参考数据不下载，门禁不晋级；功能测试未产生公开FVD成绩。
即使门禁晋级也不会自动部署。校准/评价同prompt新seed只说明该固定prompt
分布内的效果，不自动证明未见prompt泛化、版权合规或整个训练集无污染。

## 验收与当前缺口

新增单元覆盖：同原生采样器逐bit、实际block/head hook计数、CFG两支残差、
默认/retention边界、负多项式累计、审计禁用、reset、模型/条件变化拒绝。
既有Wan独立公式/全参数梯度与真实训练/ZeRO3用例也复跑。

`tests/parity/test_wan_teacache_source.py`是显式批准公开源访问后的额外oracle：
固定源码SHA256后只执行其两段cache if块，默认/retention都逐步比较输出、
累计量和两个残差。默认离线测试跳过；它证明cache块语义，不证明整个
官方Wan FlashAttention/pipeline或官方大型权重的生成质量。

集成测试执行真实tiny模型训练更新→发布→校准→完整视频→性能→门禁，并
测试改写系数/PNG、缺CFG、缺重复测量的fail-closed行为。tiny VAE只是测试
fixture，没有把可运行性当作优质视频模型。GPU、BF16、分布式缓存、官方
大权重系数适配、UniPC/DPM++、公开FVD正向晋级仍未认证。
