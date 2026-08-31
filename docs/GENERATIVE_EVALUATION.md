# 生成产物与公开分布指标

实现位于 `aster.evaluation.generative`。本模块把**实际原生模型生成的图片**、完整样本清单和公开特征器接起来；不会把随机特征上的 Fréchet 公式测试写成公开 FID 成绩。

## 支持范围与证据

| 环节 | 实际实现 | 当前验证边界 |
|---|---|---|
| 生成 | 本仓库 UNet2D/DiT，绑定训练日程的 DDPM/DDIM、Flow Euler/Heun/RK4、DMD direct-x0；可接独立原生 KL-VAE 制品解码 | CPU 实际tensor_fit/DMD更新→制品→PNG，分片与单 rank 一致；不是公开预训练模型效果 |
| 样本分片 | `index % world_size`，独立样本 seed、完整 rank 集合、失败记录、内容 hash、制品血缘 | 实际分片生成/合并测试；没有把尾部重复采样当补齐 |
| 图片 FID/KID | 可选官方 clean-fid，显式本地 Inception 权重，不下载；原始特征保存 | API 调用契约与拒绝路径已测试；当前环境缺官方依赖/权重，真实特征器测试明确跳过 |
| 视频生成/FVD | 原生 Wan field+causal VAE→连续帧制品；本地 StyleGAN-V I3D TorchScript 的真实调用，固定帧/FPS、总体协方差高斯距离 | native T2V/I2V 实际生成、逐帧对照/分片测试；未运行公开 I3D 权重，无公开 FVD 成绩 |
| 自动晋级 | 不在此模块自动晋级 | 需质量非劣 + 实测性能 + 可靠性证据；一个 FID 数字不能自动获得“更快且等效”资格 |

官方评价依赖仅在评价时 lazy import。模型/采样器不是 Diffusers、Transformers 或外部仓库进程包装。

Drifting已经通过独立的 `DriftingSamplingPlan` 与 guidance-embedding contract 接入相同PNG/公开特征器评价底座，见 [DRIFTING_GENERATION.md](DRIFTING_GENERATION.md)。它不进入DMD的 `direct_x0` 时间语义。

## 1. 原生生成与不可变产物

`ImageSamplingPlan` 固定完整样本 ID、seed、条件向量/类别、噪声 CHW、步数、solver、guidance、时间方向、训练链 respacing 索引和输出量化。VAE ID 在调用参数和实际 `sampling_binding` 中单独固定。条件目前是模型已支持的数值向量/类别，**不是声称已自动完成任意文本 tokenizer/encoder 管线**。

### 训练制品与真实日程

共享 `tensor_fit` 在制品 `model/` 保存模型，在根目录保存 `objective.json`，内容来自实际目标的 `config_dict()`。Diffusion 的原始 `betas`、`timestep_map`、learned-variance 设置随权重一起进入制品 hash。生产器同时支持手工原生模型的根目录布局，但两种布局同时出现会拒绝，不以目录探测顺序决定使用哪份权重。

DDPM/DDIM **不再接受采样端的 `schedule='cosine'` 等重建选项**，也不从推理步数创建 beta；缺少已固定 `objective.json` 的旧模型必须由了解原训练配置的调用者补齐并重新发布为新制品，不能自动猜测。`steps=50` 从1000步原链选择50个包含首尾的整数均匀索引 `i*(999)//49`；也可用 `respacing_indices=(...)` 固定自己的有序子集，其长度必须等于 `steps`。这不是冒称 OpenAI `ddim50` 字符串使用的选点策略。

子链重新推导 beta，使选中位置的累计 alpha 与训练边缘相同；模型收到的时间仍是 `original.timestep_map[indices]`。因此已经重映射过的训练时间也不会被错误改成 `0..steps-1`。当前离散采样至少两步，完整链保留原 beta 的逐位表示。`sampling_binding` 保存实际模型相对路径、原目标文件 hash、原日程身份、选中索引、有效 beta/alpha/time；合并分片要求绑定完全一致。

Flow 制品若有训练目标，采样方向必须匹配目标声明。无目标文件的手工 velocity fixture 仍可用于算法测试，但明确记录 `training_semantics_bound=false`，不能冒充已验证训练配方。

### DMD单步部署，不是DMD2全工程

```python
from aster.evaluation.generative import publish_dmd_generator, ImageSamplingPlan, GenerationCase, generate_image_shard

def export_and_sample_dmd(method, store, export_directory, images_directory):
    # method必须已经完成原生DMDMethod.update；本有界导出器目前只验证单rank。
    artifact = publish_dmd_generator(method, store, export_directory)
    plan = ImageSamplingPlan((GenerationCase('image-0', 1234),), (3, 32, 32),
                             sampler='direct_x0', steps=1)
    return generate_image_shard(store, artifact.id, plan, images_directory)
```

导出 helper 从 Trainer 合并真正的逻辑权重，绑定固定 `generator_time`、`sigma_data`、真实更新次数及 generator/real-score/fake-score 权重指纹。每张图只执行一次 `generator(noise, generator_time, condition)`，要求 `prediction_type='x0'`；不会把它重复塞进 DDIM 或 Flow 求解器，也不加入未经对应训练的两次前向 CFG 插值。采样时重新核验 generator 权重指纹。

部署制品不包含 optimizer/RNG/score 训练状态；完整续跑使用 `Trainer.save_checkpoint/load_checkpoint`，恢复后可再次导出。调用者传入的 `parents` 只声明初始来源，不被伪称为自动验证过的教师制品对应关系；使用过的角色权重由单独指纹记录。

DMD的完整边界跨越所有fake-score更新和最后generator更新；任一异常或overflow跳步都会保留未完成标记，拒绝保存/导出/继续下一轮，必须恢复上一个完整checkpoint。并非自动撤销已经发生的fake更新。新checkpoint固定sigma_data、generator_time、fake更新次数及EDM噪声分布；旧版缺这些语义字段的checkpoint不能被无声认证为精确恢复。实际异常与真实autograd非有限梯度故障注入均覆盖了恢复后下一轮权重、随机扰动与损失逐位一致性。

`direct_x0` 当前严格只接 DMD contract。Drifting 的 guidance-embedding 输入、Consistency 边界预处理、DMD2 GAN/回放以及多步学生不是该模式的别名。DMD 教师所需 EDM 目录采样和生成分布专用质量＋性能联合门禁仍是后续独立工作；本包不宣称少步模型已达到教师质量。

两个身份不能混淆：

- `plan.id`：完整实际采样配置。50 步和 4 步不同，写入每次产物的 revision/generation.json。
- `plan.cohort_id`：相同样本 ID、条件、seed 和输出量化。允许用同一评价协议比较不同步数/solver/权重；优化参数没有从实际运行记录中消失。

下面函数消费已经训练并发布的本仓库模型。没有为示例下载或随机冒充公开预训练模型：

```python
from pathlib import Path
from aster.evaluation.generative import (
    GenerationCase, ImageSamplingPlan, generate_image_shard, merge_image_shards,
)

def generate_cohort(store, policy_id, output_root, *, decoder_id=None):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    plan = ImageSamplingPlan(
        cases=tuple(GenerationCase(f"image-{i:05d}", 10000+i) for i in range(1000)),
        noise_shape=(3, 32, 32),  # 必须匹配当前已训练模型；latent 模型另配原生 decoder。
        sampler="flow_heun", steps=20,
    )
    for rank in range(2):
        generate_image_shard(store, policy_id, plan, root/f"rank-{rank}",
            rank=rank, world_size=2, decoder_artifact_id=decoder_id, device="cpu")
    media = merge_image_shards([root/"rank-0", root/"rank-1"], plan, root/"complete")
    media.verify(root/"complete")
    artifact = store.publish(root/"complete", kind="generated_images",
        metadata={"media_manifest_id": media.id, "sampling_plan_id": plan.id},
        parents=media.producer_artifacts)
    return plan, artifact
```

示例按顺序执行两个 rank 是最简单的可复现调用；相同函数可由调用者分配到独立进程/机器，输出目录不能共享写入。同 rank 内目前固定 batch=1，代价是吞吐较低；不能用这条路径宣传 vLLM 式高吞吐。源文件 hash、环境、模型/decoder 制品和实际图片 hash 都保留；要求使用新启动的固定软件环境，磁盘热更新不是可复现运行。

原始场输出先完成以下明确量化，再保存 PNG：

- `minus_one_one_stylegan`：`clip(x*127.5+128, 0, 255)` 后向 uint8 截断。
- `zero_one_round`：`floor(clip(x,0,1)*255+0.5)`。

不悄悄把 RGB、alpha、灰度、16-bit 图混在一个协议中；当前目录清单要求 RGB8、静态 PNG/JPEG/WebP。`sample.status="error"` 保留预期 ID/seed，不重抽样。任一错误/缺图/重复 rank/额外图片/改图/修改计划都阻止有效评价。单独一个不完整 rank 也不能冒充全部样本。

## 2. 固定参考目录

参考数据必须由调用者提供合法、本地、固定 revision 的完整 split，明确许可声明。`files_by_id` 是整个预期集合，不是先 glob 再挑好看的图片。权重、参考数据都不会下载：

```python
from aster.evaluation.generative import image_directory_manifest

def freeze_reference(root, full_id_to_filename, *, dataset_id, revision, split, license_id):
    manifest = image_directory_manifest(root, dataset_id=dataset_id, revision=revision,
        split=split, license_id=license_id, files_by_id=full_id_to_filename)
    manifest.save(root)
    return manifest
```

清单保存每张文件字节 SHA256、尺寸、ID 和集合顺序，指标前后均重新核验。不存在“缺少几张就降低分母继续”的路径。许可字段是声明和审计入口，不构成自动法律合规判定。

## 3. 真正 clean-fid 目录评价

已按 [clean-fid 的真实文件特征/FID/KID API](https://github.com/GaParmar/clean-fid/blob/main/cleanfid/fid.py)、[Inception 本地加载器](https://github.com/GaParmar/clean-fid/blob/main/cleanfid/inception_torchscript.py)、[clean resize 实现](https://github.com/GaParmar/clean-fid/blob/main/cleanfid/resize.py) 核验接口。

执行时使用：

1. `InceptionV3W(local_directory, download=False, resize_inside=False)`，本地文件必须叫 `inception-2015-12-05.pt`。
2. `fid.get_files_features(explicit_files, model=model, mode="clean", num_workers=0, ...)`，显式列表顺序，不调用会筛选/截短文件集合的便利入口。
3. `fid.fid_from_feats(real, generated)` 和 `fid.kernel_distance(..., num_subsets, max_subset_size)`。

协议固定 Pillow RGB8 解码（不自动 EXIF 旋转/ICC 变换）、逐通道 float32 bicubic resize 到 299、缩放后不再量化、网络 `(x-128)/128`、2048 维特征及 `ddof=1` 协方差。KID 是有限样本无偏估计，可为负值，不能强行裁成零。KID 实际 subset 数/大小/seed 全部入指纹。

官方 KID 使用全局 NumPy RNG；本模块在锁内保存/恢复状态，**评价仍要求独立进程**，不能与其他业务线程同时修改 NumPy RNG。原始特征 `.npy` 保存完整顺序与 hash，允许后续审计，不当作逐图片“正确/错误”。

以下函数只记录已经安装且审核过的源/权重，不是自动安装器。`revision` 应是经宿主核验的完整 commit。源码 hash、依赖版本、权重 hash、官方权重 URL 和许可声明共同留档；hash 不等于独立认证“此文件确实出自官方”。TorchScript 是可执行代码，必须可信、隔离运行时由部署方提供：

```python
from dataclasses import asdict
import time
from aster.core import atomic_json
from aster.evaluation.generative import (
    record_local_extractor, DistributionProtocol, evaluate_media_directories,
)
from aster.evaluation.suites import EvaluationGrant

def evaluate_images(reference, generated, reference_root, generated_root, *,
                    source_root, weights_path, reviewed_commit, reviewed_license, output_root):
    pin = record_local_extractor("cleanfid_inception", revision=reviewed_commit,
        source_root=source_root, weights_path=weights_path, license_id=reviewed_license)
    protocol = DistributionProtocol(reference.id, generated.cohort_id, pin,
        expected_generated_ids=generated.expected_ids, kid_seed=1729)
    # 只有已审核这些本地文件的宿主可以构造许可；模型回答“允许”不是执行授权。
    grant = EvaluationGrant(protocol.id, ("official_evaluator", "torchscript_execution"),
        time.monotonic()+3600)
    return evaluate_media_directories(protocol, reference_root, generated_root,
        source_root=source_root, weights_path=weights_path, grant=grant,
        output_directory=output_root, device="cpu")
```

`source_root` 是**当前解释器实际导入的 cleanfid 包目录**，不是旁边另一份同版本源码。固定整个 Python 源树、Torch/NumPy/Pillow/SciPy/torchvision/clean-fid 版本；不加载已有缓存 FID stats，以免混入其他 split/resize 协议。缺任一依赖、hash 不一致或非有限特征/分数时写 `status=error, metrics={}`，不返回兼容名字的替代分数。

研究过 [torch-fidelity 官方 Inception 实现](https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/feature_extractor_inceptionv3.py)：其输入是 uint8 BCHW，TensorFlow 兼容 bilinear resize，不是 clean-fid 的 float32 PIL bicubic；当前源码本地权重路径用 `torch.load(..., weights_only=False)`。本包暂只实现 clean-fid 这一图片协议，没有为了扩大名单把另一个协议冒充相同 FID。

## 4. 视频 FVD 的独立协议

[StyleGAN-V 官方 FVD](https://github.com/universome/stylegan-v/blob/master/src/metrics/frechet_video_distance.py) 公开了固定 I3D TorchScript URL 与 `rescale=True, resize=True, return_features=True` 调用；其说明明确与 [原始 TensorFlow FVD](https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py) 存在 upsampling 差异。不能把任意 I3D/CLIP/VideoMAE 特征距离统称为可横向比较的 FVD。

当前实现：

- `video_directory_manifest(...)` 接每段视频明确排序的帧文件、原视频 FPS、实际 `frame_indices`，例如 `(0,2,...,30)` 与 30 FPS；短视频不能静默丢弃或补帧。可变帧率视频须在数据 revision 中先固定解码/时间戳策略。
- 原始像素打包为 uint8 BCTHW，固定本地 archive 执行上述 kwargs，核验 `[clips,400]` Kinetics pre-softmax 特征。
- 同一 batch 帧尺寸一致；不悄悄 padding/crop。resize 的实际数学由已经固定 hash 的 archive 拥有。
- `DistributionProtocol(..., metrics=("fvd_styleganv_i3d",), frame_indices=..., fps=...)`；独立于图片协议。
- 协方差使用 [StyleGAN-V FeatureStats](https://github.com/universome/stylegan-v/blob/master/src/metrics/metric_utils.py) 的总体分母 `N`；本仓库自写对称 PSD 特征值公式，与原 SciPy sqrtm 的数值路径分开记录，不能声称位级一致。

不下载或再分发 I3D 权重；调用者必须审核 archive 来源、模型权重/数据许可。StyleGAN-V/NVIDIA 文件许可与本仓库许可不能自动互换。原生视频采样现已由 `evaluation.video_generation` 接入，见 [VIDEO_GENERATIVE_EVALUATION.md](VIDEO_GENERATIVE_EVALUATION.md)；公开 I3D extractor 尚未在本机运行，因此没有已验证公共 FVD 成绩。

## 5. 如何解释报告与成本

报告只有总体 `fid_clean`、`kid_clean` 或带协议名的 FVD，方向均 lower-is-better；模型、采样步骤、样本全集、失败项、原始特征、环境、源与权重身份均保留。未通过核验时没有可用于晋级的有效数值。

报告不自动生成 CI：FID 的小样本偏差、跨 seed 重复采样与 KID 子集方差是不同问题。需要事先固定多个完整生成 cohort、匹配 seed 的重复试验和明确重采样单位；不可把一个 FID 复制 N 份计算“极窄置信区间”。参考集、候选样本数、数据污染审计和训练集关系必须公开。

目前固定 batch=1 的生成耗时字段叫 `end_to_end_seconds_including_io`，包含 PNG 编码/IO，不是假称纯模型 latency；评价 wall time 同样不代表推理 throughput。要证明 KD、少步或缓存优化达到质量–性能目标，须另测同硬件/分辨率/样本数的生成 p50/p95、内存、真实模型调用数和失败率，并将两种证据共同交给上层 gate。

FID2048 协方差求解和大样本 Inception 特征都可能耗费较多 CPU/内存；本最小适配保留全量原始特征，不声称是无限规模流式统计。公开常用大样本协议不得因本地资源不足悄悄改为几张图。

## 6. 测试与可选官方验证

`tests/unit/test_evaluation_generative.py` 覆盖实际 native 生成、rank 合并/缺失/篡改、失败全集、图像量化、分布比较指纹、官方调用的无下载参数、视频布局与总体协方差。API spy 在产生特征/分数前主动失败，因此不存在“mock 公共成绩”。

`tests/integration/test_generative_official_optional.py` 默认跳过。只有 `ASTER_APPROVED_GENERATIVE_EVAL` 指向已审核本地配置 JSON 才运行真实官方特征器。配置含 `approved_local_execution=true`、`protocol`（`asdict(DistributionProtocol)`）、`reference_root`、`generated_root`、`source_root`、`weights_path`、`deadline_seconds` 与可选 `device`。它不安装依赖、不下载数据或模型。

本机当前未提供这些官方依赖/权重：**目录接口可执行、工程与契约测试通过，不等于已取得公共 FID/KID/FVD 基准成绩。**
