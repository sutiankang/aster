# 原生 Wan 视频产物与分布评价

`aster.evaluation.video_generation` 将本仓库 `WanVideoDiT`、因果 `WanVideoVAE` 和 `VideoGenerationPipeline` 接到不可变帧制品及 FVD 协议。不是外部 Wan/Diffusers 命令包装，不调用任意生成回调。

## 真实支持边界

- T2V 的实际数值文本特征；I2V 的数值文本、图像特征、mask+VAE 条件潜变量。
- 原生 sigma 从 1→0 的 Euler/Heun 采样、shift、显式 positive/negative CFG；不把它们叫作 UniPC/DPM++。
- 原生因果 VAE 解码，输出帧数必须为 `1+(latent_T-1)*temporal_stride`，空间尺寸必须对应真实 stride。
- 每样本固定 seed，生成连续全帧 RGB8 PNG，完整 FPS 声明；单 rank 与完整多 rank 合并。
- 首包只接受 FP32 field/VAE 权重。没有自动 autocast，也不声称半精度组合已经验证。GPU FP32 是允许的实际设备路径，但本机只验证 CPU。
- 当前不生成压缩视频文件、不自动改 FPS/补帧/裁剪，不自动运行未绑定的文本 encoder。FVD 本身仍需本地已审核官方 I3D 权重。

模型与训练方法的实现/数学对照属于各自测试；此包重点验证生成到产物和评测的工程连通性。小随机初始化模型的几帧图像不证明视频质量或官方预训练效果。

## 1. 条件也是制品，不是回调

每个条件 key 存一组 `positive` 和可选 `negative` 张量：

```python
conditions = {
    "prompt-001": {
        "positive": {"text": text_features, "text_lengths": original_lengths},
        "negative": {"text": negative_features, "text_lengths": negative_lengths},
    },
}
```

`text` 为 `[1,L,D]` 浮点特征，`text_lengths` 为 `[1]` 整数。I2V 另显式提供 `[1,L,D_image]` 的 `image_features`、`[1,C_cond,T_latent,H_latent,W_latent]` 的 `video_condition`；后者可以由本仓库 `methods.video_generation.image_video_condition` 从首帧构造，不能从目标未来视频偷取条件。

```python
from aster.evaluation.video_generation import publish_video_conditions

condition_artifact = publish_video_conditions(
    store, conditions, condition_staging_directory,
    source_artifact_ids=(input_dataset_artifact_id,),
    declared_encoder_artifact_id=encoder_artifact_id,
    declared_processor_artifact_id=processor_artifact_id,
)
```

这些输入制品 ID 必须实际存在并校验通过。编码器和 processor 的 ID 可以省略，但会明确保留未知来源边界；不会把任意向量叫成官方 UMT5/CLIP 特征。

此发布函数仅保存**调用者已计算**的数值条件，因此统一记录 `origin=caller_provided_numeric_tensors`、`encoder_execution_verified=false`。即使提供声明的 encoder ID，也不伪造实际编码执行证明。需要公开文本条件 benchmark 时，宿主仍要提供完整 prompt/processor/encoder 执行血缘与数据污染审计。

每个 key 有独立 `weights_only` 张量文件及 SHA256/dtype/shape 清单，使用时逐 case 加载，不一次占满全集 embedding 内存。没有 pickle processor、可执行回调或远程模型代码。正、负条件包含在同一制品身份中；CFG 没有 negative 时该样本明确失败。

## 2. 采样、分片与合并

```python
from aster.evaluation.video_generation import (
    VideoGenerationCase, VideoSamplingPlan, generate_video_shard, merge_video_shards,
)

plan = VideoSamplingPlan(
    cases=tuple(VideoGenerationCase(f"clip-{i}", 5000+i, "prompt-001") for i in range(100)),
    condition_artifact_id=condition_artifact.id,
    latent_shape=(16, 5, 32, 32),  # C,T,H,W；须匹配已发布原生模型。
    output_shape=(17, 256, 256),  # T,H,W；示例对应时间stride4、空间stride8。
    fps=16., steps=30, solver="heun", shift=5., guidance_scale=5.,
)
for rank in range(2):
    generate_video_shard(store, field_id, vae_id, plan, output_root/f"rank-{rank}",
        rank=rank, world_size=2, device="cpu")
media = merge_video_shards([output_root/"rank-0", output_root/"rank-1"], plan, output_root/"complete")
media.verify(output_root/"complete")
video_artifact = store.publish(output_root/"complete", kind="generated_video_frames",
    metadata={"manifest_id": media.id, "sampling_plan_id": plan.id}, parents=media.producer_artifacts)
```

示例参数只描述形状关系，实际生成成本可能很高；应根据已训练配置和计算预算设置。输出目录必须未存在。外部调度可以把不同 rank 放到独立进程/机器，但必须固定相同模型、条件制品、源版本和环境。

每个产物父节点固定为 field、VAE、condition 三个制品。记录实际条件转换精度、源文件 hash、完整采样参数、环境和端到端耗时。帧清单记录每一帧文件/尺寸/hash、原帧索引、FPS、样本 seed。

分片按全局样本 index 分配，尾部不复制样本；合并必须有全部 rank，拒绝重复/缺失 rank、参数或版本混合。失败样本保留其预期帧索引/FPS/seed，不能通过丢掉失败视频提高成绩。写盘中断留下未登记文件也会被全集核验拒绝。

## 3. 同条件蒸馏比较与 FVD

`plan.id` 绑定所有实际配置。`plan.cohort_id` 固定案例/seed、条件制品、输出几何、FPS、量化，故 30 步 CFG baseline 与少步无 CFG student 可以共享比较集合，但其真实方法参数与模型权重仍分别保留。

```python
from aster.evaluation.generative import DistributionProtocol, evaluate_media_directories

protocol = DistributionProtocol(
    reference_manifest_id=reference_media.id,
    generated_cohort_id=plan.cohort_id,
    extractor=reviewed_local_i3d_pin,
    expected_generated_ids=media.expected_ids,
    metrics=("fvd_styleganv_i3d",),
    frame_indices=plan.frame_indices, fps=plan.fps,
)
# 经宿主审核后的 EvaluationGrant 与本地 source_root/weights_path 同图片评价接口。
report = evaluate_media_directories(protocol, reference_root, video_artifact.path,
    source_root=i3d_source_root, weights_path=i3d_weights_path, grant=grant,
    output_directory=report_directory, device="cpu")
```

协议要求参考和生成的帧选择/FPS 一致，加载 I3D 前即检查。完整 native 视频采样计划也复核一次，单个 rank 不能冒充全集。FVD 需至少 10 帧；本地短视频功能测试不能自动成为同一公共 FVD 协议。

固定 I3D 权重/源、预处理、总体协方差与报告解释见 [GENERATIVE_EVALUATION.md](GENERATIVE_EVALUATION.md)。权重缺失时写错误报告，不下载，也不用随机特征替代。FPS 只是明确的数据/播放时间约定，不意味着模型理解了任意 FPS 的训练条件。

## 验证证据

`tests/integration/test_video_generation_evaluation.py` 实际运行小原生 Wan 和因果 VAE，覆盖：11 帧逐帧与直接 pipeline 对照、T2V/I2V/CFG、单 rank 与分片哈希一致、真实条件制品与独立加载、模型/VAE/条件父节点、缺 negative/错形状/缺分片/改 FPS/不完整分片的拒绝路径，以及基线–蒸馏 cohort 身份。

官方 I3D 未在本机提供，因此**生成链路已运行、FVD接口已连接，公开FVD分数仍未验证**。耗时包含采样、VAE、PNG 编码/IO，不冒称纯模型 latency；质量非劣和实际资源改善仍须共同满足上层 gate。
