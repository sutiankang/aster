# 原生视频世界的训练来源、生成产物与联合门禁

本包接通的是仓库内 **Genie 2024公开机制实例**，不是未公开的Genie 3系统。
编码、潜在动作推断、MaskGIT动力学和像素解码均由本仓库模型执行；没有第三方
Genie训练器、远端生成服务或运行用户任意callback的入口。公开I3D只用于可选评价。

## 从真实训练到可核验的离散数据

1. 用原生 `GenieVQObjective` 和共享Trainer训练 `GenieTokenizer`。
   `publish_genie_tokenizer(engine, store, directory, ema=False)` 由所有DP/ZeRO rank
   共同导出完整普通/EMA权重，核验最后一次成功phase的实际目标及角色update clock。
   随机初始化、裸模型目录和把当前default倒填成过去目标都不能进入此训练制品入口。
2. `publish_genie_videos(..., spec=GenieVideoSpec(...))` 固定已解码像素、有效帧、完整
   ID全集、数据revision/split/license/FPS。当前只接FP32 `[T,C,H,W]`、范围[0,1]、
   同一长度/尺寸bucket；不悄悄resize、截短、补抽样或更改归一化。许可是调用者声明，
   不是自动法律合规证明，也不宣称自己下载/清洗过公开视频。
3. `tokenize_genie_artifact(store, codec_id, video_id, directory)` 真正运行冻结原生
   编码器，记录模型权重身份、源像素/valid/token数值hash和实现源码hash。当前编码
   协议固定CPU FP32，避免不同设备临界VQ距离改变离散码却被当成同一个trace。
4. `BoundGenieWorldObjective(store, trace_id, parallel=context)` 启动时逐样本重跑相同编码器，验证
   真实token。给错误token重算文件hash再发布一个新目录仍会被拒绝。训练batch从
   `objective.batch(indices, device=...)` 取得；额外携带 `source_indices`，在整个
   多rank传共享parallel context，对某节点缺失trace先对称汇总错误。整个
   accumulation窗口任何训练forward/参数gather之前核验像素、token和valid逐行
   数值身份，再调用既有联合目标。mask只控制有效未来位置，不能篡改监督target。
5. `publish_genie_world` 要求实际目标就是绑定trace的联合目标，标准完整checkpoint
   另由Trainer保存/恢复。部署制品绑定trace、实际最后目标、角色权重和采样EMA选择。

这里的证据只证明**最后一次成功更新声明与当前部署权重**，不声称所有历史步骤都
使用同一编码器。内容hash也不是面对恶意宿主Python的远程证明或数字签名。可信
宿主仍能改代码/制造回执；本模块防止普通工程链路中的错配和被重新发布的错误数据。
完整trace复核是显式启动成本，未实现大规模零扫描索引服务。

## 真正的两条生成轨迹

`GenieSamplingPlan(cases, video_artifact_id, time_index, steps, ...)` 固定验证视频、
样本ID、独立seed、时间索引、PSNR floor和PNG量化；采样优化步数等另外进入plan ID。
可控性协议每个模型分别从真值前缀推断自己的潜在动作，不假定不同动作码本中的
数字ID代表相同物理动作。动作随机流用seed+1，两条视频采样共用seed；生成像素
条件都只有第0帧。未来真值只用于动作推断和计算指标，不交给生成decoder。

`generate_genie_shard` 保存inferred/random两套连续RGB PNG与原始浮点视频、离散
token、动作码。实际forward hooks分别统计Dynamics NFE、tokenizer编解码和LAM编码。
T个未来帧、K次MaskGIT的两轨迹NFE为2TK，公式只校验真实计数，不代替实际调用。
`merge_genie_shards` 要求所有rank（包括空rank）齐全、模型/源码/环境一致。失败
样本留在两分支的原位置；不平均幸存样本，也不自动补seed。
`publish_genie_generation` 可将已核验的完整双轨迹目录整体发布到ArtifactStore，
同时固定原始数值文件、PNG、计划、模型/码本/训练trace/评价数据血缘。

`evaluate_genie_controls` 重新读取原始浮点产物算paired Δ_t PSNR，而不是PNG量化
后的分数。结果接共享 `ComparisonProtocol/EvaluationRun`；失败仍进入完整分母，
默认1e-12 MSE floor对应有限失败分-120dB，且联合门禁还会拒绝任何失败。
这不是FID/FVD，不是CoinRun或公开视频榜单成绩。

## 质量、NFE和真实资源同一身份

`benchmark_genie_sampler` 先warmup，随后测完整case×repetition矩阵。范围是一次LAM
动作推断及两条原生视频生成，包含采样随机数；不含模型加载、trace重编码、数据I/O、
PSNR计算和PNG写入。因此它不应被命名为TTFT或单条视频延迟。CUDA明确同步并读
真实Torch allocator绝对峰值；CPU显存字段为None，不能拿Python内存伪装显存。
所有trial保留原始输出数值hash，门禁要求它们与质量评价产物逐case完全相同。

`evaluate_genie_fvd(..., resources=GenieFVDResources(...))` 复用固定StyleGAN-V I3D
协议：至少10帧、明确FPS、原始RGB8、固定源树和权重hash、显式执行授权，完全离线。
无真实官方资源时写 `not_evaluated`，不下载、不填fixture分数。inferred分支FVD是
**由参考视频推断动作条件的生成分布**，不是无条件视频生成指标；条件第0帧也在固定
clip中。切换分支/删掉条件帧需新协议，不能与原协议直接比较。

`compare_genie_cohorts` 同时核验不同训练权重或少步采样candidate与baseline的固定数据、
cohort、源码、NFE、产物和性能身份。真正晋级要求至少3个不重用seed的完整独立
cohort、ΔPSNR及实际FVD非劣、延迟改善、NFE无回退，并可选真实显存上限。
FVD置信区间在**完整cohort总体指标**之间聚合，不伪造“每视频一个FVD”。硬件独占
只能由受信宿主明确声明，这不是OS隔离；缺官方FVD/独立cohort/硬件声明/所需显存
一律不能晋级。输出不自动部署模型。
当前publisher要求最后实际目标是BoundGenieWorldObjective；尚无独立Genie蒸馏
Method的发布契约，不能把降低MaskGIT采样步数本身叫作已经完成蒸馏训练。
评价集必须声明validation/test/evaluation，且与绑定训练数据不存在完整像素张量的
精确重复；这里只查精确重复，不宣称解决近重复、外部预训练污染或许可审计。

## 当前范围与证据限制

本包仅有原生CPU小实例、真实本机DP/ZeRO及错误恢复证据，不能据此宣称学得了稳定
真实动作语义、重现Genie 11B效果或取得公开FVD/多机GPU吞吐。TP/PP/CP/EP/ETP/GTP
训练制品发布明确拒绝。源代码变化后旧trace不能静默沿用，须在新固定环境重新验证
编码并生成新trace；绝不为让测试通过而关闭源码身份检查。

来源：[Genie 2024论文](https://arxiv.org/html/2402.15391v1)定义先训练视频tokenizer再
联合LAM/dynamics的两阶段流程；[Google MaskGIT固定源码](https://github.com/google-research/maskgit/blob/1db23594e1bd328ee78eadcd148a19281cd0f5b8/maskgit/libml/parallel_decode.py)
提供迭代采样/置信度/Gumbel重遮盖机制。未得到Google内部Genie训练源码，不能用
MaskGIT的Apache-2.0许可推断未公开权重的许可。FVD具体预处理/资源授权另见
[GENERATIVE_EVALUATION.md](GENERATIVE_EVALUATION.md)。
