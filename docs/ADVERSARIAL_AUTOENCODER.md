# 感知重建与对抗训练：共享训练框架中的完整方法

本模块把原生KL-VAE、原生LPIPS、原生PatchGAN和全局自适应梯度权重组成一个方法。
不是调用CompVis训练器，也不是把特征MSE更名为LPIPS或把两个独立训练脚本拼起来。
它补充[感知自编码器](PERCEPTUAL_AUTOENCODER.md)，后者仍可独立使用。

## 来源与数学

- VAE训练损失/优化器归属参考[CompVis latent-diffusion](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/losses/contperceptual.py)。
- PatchGAN与ActNorm参考[taming-transformers](https://github.com/CompVis/taming-transformers/tree/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules)。本地`ActNorm2d`拆分状态容器与纯参数叶，以满足ZeRO3参数所有权；仿射仍为`scale * (x + loc)`。
- LPIPS参考[PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity/tree/082bb24f84c091ea94de2867d34c4544f68e0963)。标准完整VGG/Alex结构与随机小宽度测试模型区别明确，不附带或自动下载感知权重。

生成器的NLL是`(|x-reconstruction| + perceptual_weight*LPIPS)*exp(-logvar)+logvar`。
默认每张图所有坐标求和，再除全局样本数；`pixel_reduction='mean'`是显式可选变体。
KL逐潜变量坐标求和。每次生成器更新只编码、采样、解码一次；同一假图同时用于NLL、KL和GAN。

GAN项为`-mean(D(fake))`。自适应系数是末层decoder卷积weight上的
`clip(||grad(NLL)|| / (||grad(GAN)|| + 1e-4), 0, 1e4) * disc_weight`，再乘`disc_factor`。
取的是整个累积窗口、所有DP样本归一化后的两组梯度，而不是平均每个microbatch的比值。
系数不参与二阶求导。引擎复用已存在的分目标梯度，不克隆完整模型或额外跑一遍forward。
ZeRO0/1为选定参数的两份probe额外做DP sum；ZeRO2/3使用已reduce-scatter的有效shard范数。

判别器支持hinge和vanilla softplus。它在生成器更新后重新重建fake；无显式noise时，G/D各自
采样后验。G阶段只冻结D参数，仍保留输入梯度；D阶段冻结G，fake不反传到G。

## 使用与初始化边界

```python
import torch
from aster.models import AutoencoderKL, AutoencoderConfig
from aster.models.perceptual import LPIPS
from aster.models.adversarial import PatchDiscriminator, PatchDiscriminatorConfig
from aster.methods.adversarial_autoencoder import AdversarialAutoencoderMethod
from aster.training import Trainer, ParallelContext

# images是显式预处理到[-1,1]的RGB NCHW；各rank已按本地数据分片获得校准集。
context = ParallelContext()
generator = AutoencoderKL(AutoencoderConfig())
perceptual = LPIPS.from_pretrained(local_trained_lpips_directory)
discriminator = PatchDiscriminator(PatchDiscriminatorConfig(normalization='actnorm'))
# 此例用posterior mode的重建校准，是明确的数据初始化选择；不是隐式更改训练后验。
# 若需逐步复现某个官方实验，应使用同一校准样本/采样噪声/初始化权重。
with torch.no_grad():
    calibration = generator.decode(generator.encode(images).mode())
discriminator.initialize(calibration, group=context.dp)

adam = lambda parameters: torch.optim.Adam(parameters, lr=4.5e-6, betas=(.5, .9))
engine = Trainer(generator, parallel=context, zero_stage=3, accumulation_steps=2,
                 optimizer_factory=adam, precision='bf16', ema_decay=.999)
method = AdversarialAutoencoderMethod(engine, perceptual, discriminator,
    discriminator_optimizer_factory=adam, disc_start=50001,
    disc_factor=1., disc_weight=.5, kl_weight=1e-6)
result = method.update([{'sample': images_a}, {'sample': images_b}])
engine.save_checkpoint(checkpoint_directory)
```

`images`、两个训练batch、路径、设备和多进程初始化由调用程序提供，不猜数据范围或硬件。
校准必须在判别器优化器/分片创建之前；不同rank可以有不同B/H/W，但必须相同初始权重。
ActNorm统计使用FP64两遍中心化估计，std按N-1校正后写回参数；不是逐rank校准参数再平均。
数据初始化是一次性事件，不能重置已训练参数。零方差通道遵循上游`std+1e-6`。

本方法支持纯DP和ZeRO0–3，不默默接纳未验证的TP/PP/EP组合。原生PatchGAN也提供普通
BatchNorm结构用于公式对照，但多角色GAN方法明确要求已初始化ActNorm，避免将有状态
BN更新藏在ZeRO3重算中。默认D优化器为Adam `(0.5,0.9)`；已经存在的G优化器不被更换。

## 时钟、异常和恢复

`disc_start`按本方法成功完成的G+D迭代数计数，不猜不同Lightning版本的global_step含义。
warmup时D仍以零分子构建零梯度并推进Adam step，不能完全跳过D优化器导致随后状态不同。
G未更新时不运行D；G已更新而D失败/overflow则为半完成事务，必须恢复完整checkpoint，
方法拒绝把它保存为有效状态。不存在隐式原子回滚承诺。

checkpoint同时保存G/D/冻结LPIPS、两个优化器、EMA、RNG、方法计数和自适应策略记录。
重建方法时先加载同一内容身份的感知权重；不接受同shape不同特征网络替换。
全窗口输入检查在首个模型collective之前发生，避免一个rank第二个batch损坏引起不对称执行。

## 实际验证与限制

- PatchGAN/ActNorm官方类原文在精确提交和文件SHA校验下执行：输出、所有参数/输入梯度、
  初始化、逆变换、logdet，3项实际通过；只在测试时联网，训练无上游运行时依赖。
- 原生PatchGAN真实DP2全局校准、全部ZeRO更新/恢复；完整GAN方法真实DP2、非均匀batch、
  warmup到active、全batch独立公式的全局系数/SGD更新、双角色checkpoint精确恢复均通过。
- BF16/ZeRO3无显式噪声的两个后验采样，fresh-model checkpoint恢复逐bit一致。
- FP32不同batch形状与归约顺序在GroupNorm平移不变方向产生约1e-8的理论零梯度噪声；
  默认Adam eps可把它放大为约1e-4参数差。这不是把参数容差放宽就能声称逐bit等价。
  测试分别检查全batch梯度与SGD更新，以及将实际合成梯度交给独立Adam的逐bit更新。
- 尚无公开训练LPIPS/VAE权重与真实数据上的重建/FID质量结果，也无GPU速度结论。
  对抗项接通不意味着训练一定收敛或效果自动达到某个公开模型；真实实验需固定数据、
  权重、采样器与评测协议后验证。
