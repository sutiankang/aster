# Drifting训练制品与一次前向采样

`aster.evaluation.drifting_generation` 消费本仓库 `DriftingMethod` 和 `DriftingGenerator`，生成真实PNG/完整评价清单。它不启动作者仓库的JAX训练器，也不把普通扩散模型换个名字。

## 三个不能混淆的语义

- `cfg_scale` 是输入到网络的 guidance embedding，一张图只前向一次；不是扩散时间，也不是正/负提示两次预测的CFG线性插值。
- 模型含离散噪声embedding时，先抽连续FP32标准正态噪声，再用同一个独立seed generator抽离散类别。这与原生 `model.generate()` 顺序一致；每个样本记录实际连续噪声字节hash和离散索引，不承诺CPU/CUDA不同随机实现位级相同。
- 导出选择当前权重或Trainer真正维护的EMA。BF16训练可以保留FP32参数存储；发布检查实际dtype，并不会把不支持的BF16参数强转FP32后声称等值。

```python
from aster.evaluation.generative import GenerationCase
from aster.evaluation.drifting_generation import (
    DriftingSamplingPlan, publish_drifting_generator,
    generate_drifting_shard, merge_drifting_shards,
)

def sample_trained(method, store, export_directory, shard_directories, merged_directory, *, decoder_id=None):
    # method须已完成真实update；DP时所有rank共同调用导出，只有leader写制品。
    artifact = publish_drifting_generator(method, store, export_directory, ema=True)
    if method.engine.parallel.rank != 0:
        return None  # 后面的本地分片调度只由leader执行，避免多rank重复写目录。
    c = method.engine.model.config
    plan = DriftingSamplingPlan(
        cases=tuple(GenerationCase(f'image-{i}', 1000+i, i % method.settings['num_classes']) for i in range(100)),
        noise_shape=(c.in_channels, c.input_size, c.input_size), cfg_scale=2., temperature=1.,
    )
    # 以下图片分片任务由调用者调度；不是让每个训练rank重复执行这整个循环。
    for rank, path in enumerate(shard_directories):
        generate_drifting_shard(store, artifact.id, plan, path,
                               rank=rank, world_size=len(shard_directories), decoder_artifact_id=decoder_id)
    return merge_drifting_shards(shard_directories, plan, merged_directory)
```

部署制品固定 `method='drifting'`、`conditioning_semantics='guidance_embedding'`、连续/离散噪声协议、完整模型配置、实际权重指纹、训练特征身份/权重身份、训练CFG分布、成功更新次数及EMA选择。DMD的 `direct_x0` 消费者明确拒绝这个家族，不会把 guidance embedding 当成 `generator_time`。

输入类别限于Method声明的训练类别。`cfg_scale>=1` 可显式超过声明训练范围，但绑定中会记录该事实，不能因此保证外推质量。温度、CFG、噪声shape和实际模型均进入运行身份；同样本ID/seed/类别/量化形成共享cohort，允许对比不同优化候选。

## 产物、恢复与评价

输出通道是latent时可传 `decoder_artifact_id`，只接受另一个固定的本仓库FP32 KL-VAE，使用其真实scale/shift解码。无匹配decoder的非RGB输出会保留为失败样本，不重采样或降低分母。分片合并要求全部rank（包括空rank）与完全相同的绑定；原始噪声证据随完整清单合并。

部署模型不包含队列、训练RNG或optimizer。完整恢复仍使用Trainer checkpoint；Method处于队列已修改但更新未完成状态时，发布器调用其状态检查并拒绝发布。导出需要短暂复制队列以核验完整状态，再合并逻辑权重，不宣称已实现超大模型低峰值流式导出。

已覆盖CPU真实训练后的单次forward、连续/离散噪声逐项和PNG对照、分片/空rank、错误全集、原生VAE解码、EMA与checkpoint恢复、BF16训练的FP32存储及ZeRO3导出。另有独立真实Gloo DP2/ZeRO0和ZeRO3集体导出测试；不是单进程假分片。CUDA、多机网络和生产吞吐尚须相应硬件验收。

评价入口仍为 [GENERATIVE_EVALUATION.md](GENERATIVE_EVALUATION.md) 的固定特征器/数据版本协议；`_generation_record` 对此家族显式路由并校验真实输入证据。当前小模型用的是明确的像素特征fixture，不是预训练MAE，不代表公开FID成绩；公开Inception/I3D资源未提供时不下载，也不制造成绩。专用分布质量＋性能晋级门禁仍未完成。
