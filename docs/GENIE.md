# 潜在动作视频世界：公开机制与工程实例

本模块不是 Genie 3 的官方实现，也不宣称重现未公开的训练系统、数据或权重。
截至本次核对，Google 的产品页和 Genie 2024 论文没有提供可用于整包对照的训练源码。
因此这里明确实现 **Genie 2024 公开机制的一个可训练原生实例**；未知细节单独列出。
运行时不调用 Google 服务、第三方 Genie 复现仓库、JAX 或 TensorFlow。

## 一个流程，而非三个孤立模型

`GenieTokenizer` 先用像素重构及 VQ 损失训练。`encode_genie_video` 把真实 tokenizer
产生的离散索引与原像素一起交给 `GenieWorld`。后者联合训练 `GenieLatentAction`
和 `GenieDynamics`：动作模型从视频推断离散控制，动力学的 CE 明确停止对动作编码
的反向传播。两条目标经同一 Trainer 的全局分子/分母处理，一次提交优化器更新。
独立模型保存/加载复用标准配置与安全 tensor 格式，不自动下载或运行远程代码。

核心结构是空间注意力、因果时间注意力，再接一个 FFN。图像的 patch 排列和逆排列
写在模型中；视频 tokenizer 编解码都是时间因果的。动作 t 只看前缀至下一帧，
其重构 decoder 只能读前一帧和量化动作，不能从未来像素旁路取得答案。
VQ 使用 DeepMind Sonnet 的非 EMA 欧氏距离/straight-through 公式；commitment
和 codebook 的停止梯度方向不同，分别提供有效 latent 坐标计数。

MaskGIT 输入与目标处于同一帧位置，动作 t-1 加到目标帧 t。默认遮盖率逐序列
从 [.5,1] 均匀采样，再逐 token Bernoulli 遮盖；只监督有效被遮盖位置。
第 0 帧固定作为条件，padding 不参与任何损失。不同 rank 可以有效数量不同、
甚至其中一个 rank 没有未来帧监督，但不能跳过其必要通信。

## 最小使用

```python
from aster.models.genie import GenieTokenizer, GenieTokenizerConfig
from aster.methods.genie import GenieVQObjective, GenieWorldObjective, encode_genie_video
from aster.training import Trainer
from aster.planning.genie import generate_genie_video

# video: float[B,T,C,H,W]，范围[0,1]；world使用显式GenieWorldConfig构造。
tokenizer = GenieTokenizer(GenieTokenizerConfig())
tokenizer_trainer = Trainer(tokenizer, GenieVQObjective(sequence_length=T), lr=3e-4)
tokenizer_trainer.step([{'video': video}])
batch = encode_genie_video(tokenizer, video)
world_trainer = Trainer(world, GenieWorldObjective(sequence_length=T), lr=3e-5)
world_trainer.step([batch])
generated, diagnostics = generate_genie_video(tokenizer, world, video[:, :1], action_ids)
```

这只是接口示意，一步预训练不会得到好模型。ZeRO3/并行训练后部署前要由所有 rank
共同导出，再构建完整冻结实例；不要直接将部分参数当成部署模型。正式训练还需固定
tokenizer 权重和数据处理版本、验证集、预算、调度器及训练制品，而非拿任意索引冒充
某个已训练码本的输出。当前纯 tensor 桥接本身不是不可伪造的制品来源证明。

逐帧推理只读取动作码本，不调用潜在动作 encoder/decoder。原生 MaskGIT 复用已知
token、cosine 重遮盖日程与带 Gumbel 的置信度筛选，记录实际 Dynamics 调用次数。
`token_temperature` 和 `choice_temperature` 分开，`mask_order='random'` 是显式扩展。
PyTorch 和 JAX 的随机位流不同，不宣称相同 seed 逐 bit 一致。有限 context 超界明确
报错；未实现精确长视频缓存时，不悄悄裁掉历史后声称条件完全相同。

## 评价和证据

`evaluation/genie_world.py` 实际运行“视频推断动作”和“随机动作”两条生成轨迹，
只以第 0 帧为像素条件，计算逐样本 paired Δ_t PSNR；默认 t=4。MSE floor、seed、
时间索引、采样参数和实际 NFE 均进入结果。它不把随机小模型结果报告成公开质量。
PSNR 不是 FVD；真正 FVD 需要批准的官方特征提取器和完整公共视频集合。

测试覆盖独立 FP64 ST 前向/全参数梯度、VQ 反向方向、未来信息隔离、MaskGIT 固定
token、FP32/BF16、ZeRO0/3 精确恢复与像素到视频生成。真实 DP2 对比所有 ZeRO0–3
的 tokenizer/world 更新、全局范数和 checkpoint，包含无未来监督的 rank。
合成学习测试只证明小实例可优化，不等于无标签动作语义或真实环境控制已经成功。

## 明确未复原的部分

learned separable 位置编码、空间平均池化、pre-LN/GELU、head LayerNorm 形式、初始化、
同帧 masked-only CE 是本实例显式选择；论文并未给出足够源码来证明这些全部与内部
版本相同。默认小尺寸配置用于实际跑通，不冒充论文 11B 超参数。未完成官方权重映射、
连续交互长时缓存、BC 动作映射实验、公开 CoinRun 多种子控制成绩、公开视频 FVD、
大规模 TP/CP 和 GPU 吞吐认证。Genie 3 的产品表现不能自动作为这些实现的证据。

来源：[Genie 2024 论文](https://arxiv.org/html/2402.15391v1)、
[官方 Genie 产品页](https://deepmind.google/models/genie/)、
[Google MaskGIT 固定源码](https://github.com/google-research/maskgit/tree/1db23594e1bd328ee78eadcd148a19281cd0f5b8)、
[DeepMind Sonnet VQ-VAE](https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py)。
后两者源码 Apache-2.0；不能据此推断 Google 未发布模型权重的许可。
