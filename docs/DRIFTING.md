# 原生 Drifting：从特征预训练到一次生成

这一链路自己实现生成器、特征器、优化目标、类别队列、训练CFG和恢复协议；运行时
不调用作者的JAX/Flax训练器。源码依据是作者的
[生成器](https://github.com/lambertae/drifting/blob/main/models/generator.py)、
[力场](https://github.com/lambertae/drifting/blob/main/drift_loss.py)、
[训练](https://github.com/lambertae/drifting/blob/main/train.py)与
[MAE特征器](https://github.com/lambertae/drifting/blob/main/models/mae_model.py)。
本轮2026-08-30实读main；尚未取得可核验的commit与完整许可，不能捏造版本锁或商用许可。

## 组成与数据流

```
真实图像/离线VAE潜变量 → MAEResNet + MaskedAutoencodingObjective → 特征权重
                                                                    ↓冻结
真实样本 → 每类FIFO正队列 + 无条件真实负队列 → 多尺度特征 ────────┐
噪声 + 类别 + 训练CFG → DriftingGenerator → 多尺度特征（保留输入梯度） → Drifting力
                                      ↑                              │
                             Trainer / DP / ZeRO / EMA ←─────────────┘
                                      ↓完整边界导出
                         原生生成制品 → 一次前向 → 图像/固定评价清单
```

`DriftingGenerator`实现LightningDiT/DitGen：patch顺序为(p_h,p_w,C)，两维固定初始化的
可学习位置编码、可选类别前缀、QK norm/RoPE、RMSNorm或LayerNorm、SwiGLU或GELU、
AdaLN-zero与直接x0输出。类别和多坐标离散噪声各有自己的embedding。
CFG输入是`class + discrete_noise + 0.02*RMS(CFG_embedding)`，不是扩散时间，也不是
推理时对有条件/无条件的两次输出作插值。

`MAEResNet`实现四阶段ResNet、每级额外GroupNorm、UNet重建decoder、可选分类头。
mask是每张图独立Bernoulli patch采样；重建损失通道求和、按遮挡空间位置数归一化。
不能把它换成固定数量mask或按所有像素/通道平均而仍声称同一目标。配置保留作者
decoder的32组归一化，所以`base_channels`需为32倍数；缩小层数/输入可低成本验证。

`SpatialFeatureStatistics`支持完整图像向量、每通道RMS、空间特征、全局/局部均值和
标准差，MAE中间block可按间隔输出。标准差使用带1e-6的总体方差。`encoder=None`
明确表示像素特征，不能称为已预训练MAE或高质量感知特征。

## 训练契约

```python
from aster.models.drifting_features import MAEResNet, MAEResNetConfig
from aster.methods.masked_autoencoding import MaskedAutoencodingObjective
from aster.models.drifting import DriftingGenerator, DriftingConfig
from aster.methods.drifting import DriftingMethod, SpatialFeatureStatistics
from aster.training import Trainer

# samples是自己提供的、预处理身份固定的真实BCHW FP32数据；不隐式下载训练集。
feature_model = MAEResNet(MAEResNetConfig(in_channels=3, num_classes=1000))
pretrain = Trainer(feature_model, MaskedAutoencodingObjective(), lr=1e-4)
# 在真实数据上循环：pretrain.step([{'samples': samples}])
# 进入生成训练前从完整checkpoint恢复/导出特征模型，不能把随机特征当作已训练。

generator = DriftingGenerator(DriftingConfig(input_size=32, in_channels=3,
    out_channels=3, num_classes=1000))
engine = Trainer(generator, lr=1e-4, ema_decay=.999, zero_stage=0)
features = SpatialFeatureStatistics(feature_model.encoder, input_patch_size=1)
method = DriftingMethod(engine, features, feature_identity='my-mae-and-preprocess-v1',
    positive_capacity=64, negative_capacity=512, positive_samples=32,
    negative_samples=16, generated_samples=8, cfg_power=5.)
# method.update([{'samples': samples.cpu(), 'labels': labels.cpu()}])
# engine.save_checkpoint('run/checkpoint')
```

实际高类别数/高分辨率队列可能很大，必须显式设置`max_bank_bytes`；默认512MiB。
预算不足在分配之前失败。类别队列无样本时拒绝采样；已有样本不足时有放回抽样。
无条件负队列保存真实数据，不是缓存生成器输出。负权重为`(cfg-1)*(G-1)/N`。

训练CFG用截断幂律逆CDF，支持no-CFG比例。正负特征和全部力场目标停止梯度，生成
特征保留冻结编码器的输入Jacobian。力场依次使用双向softmax几何均值、正负质量耦合、
每个温度的力归一化；每个特征键是独立LossTerm，不能先拼接后换成普通MSE。

DP对全局坐标/力尺度只归约充分统计量，类别队列仍各rank本地。支持原生ZeRO0–3；
当前要求`accumulation_steps=1`，因为跨microbatch非线性归一化不能冒充同一个完整batch。
尚未接模型TP/PP/CP，相关组合明确拒绝。FP32存储+BF16 autocast已测；不把纯BF16
参数存储或CUDA特定kernel性能列为已验证。

checkpoint包括所有角色、优化器、EMA、全局RNG、队列、队列RNG、更新数、特征权重及
预处理配置身份。队列已经修改而优化器失败时，状态标记不完整，不允许保存或发布；
需要恢复最后完整checkpoint。单rank非法输入/特征会对称失败，避免其他rank死等。

## 已验证与未验证

- 生成器独立BHWC函数：FP32/BF16前向、输入和全部参数梯度；非零AdaLN/output避免零初始化掩盖错误。
- MAE独立函数：完整encoder/decoder、损失分母和全部参数梯度；这是公式对照，不是实际JAX oracle。
- 真双进程：不等组数全局力场对照、ZeRO0–3更新、队列/RNG/模型精确恢复。
- 100次小型两类分布学习回归，固定噪声MSE由0.160降至约0.00077；只证明可学习闭环，不是公开图像质量成绩。
- 特征器预训练、生成器训练、EMA/普通权重选择和一次生成制品接口均不需要上游运行库。

完整生产权重映射、作者大规模配置的公开FID复现、ConvNeXt特征分支、JAX跨框架实跑、
GPU吞吐/显存以及多机NCCL尚未验收。它们仍是未完成事项，不由本页的局部测试替代。
生成制品接口及其具体限制见`evaluation/drifting_generation.py`，DMD的固定生成时间
契约不能用于Drifting。
