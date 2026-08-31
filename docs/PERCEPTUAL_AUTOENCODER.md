# 原生感知编码器训练

`models/perceptual.py`自主实现LPIPS所用的VGG16和AlexNet特征层、版本化输入
scaling、五级归一化距离、校准卷积及可选空间图。运行时只依赖PyTorch，不调用
torchvision、lpips、Diffusers或外部训练器。VGG默认完整64/128/256/512/512宽度；
小宽度是明确声明的实验结构，不是预训练VGG的等价权重替代。

## 权重和指标边界

默认`LPIPS(LPIPSConfig())`没有下载行为，也不能以随机参数计算正式感知指标。
必须提供本地backbone与配对的LPIPS校准权重；`load_reference_weights`先完整
验证，再导入torchvision `features.N.*`和官方`linK.model.1.weight`格式。
原始检查点应以`torch.load(..., weights_only=True)`读取。分类头不参与感知计算。

`weight_identity()`绑定完整配置和实际张量字节。这个指纹能检查权重是否相同，
不能证明调用者声称的ImageNet/BAPPS训练来源。来源、许可、预处理与校准版本
仍应进入正式评测制品审核。此包没有下载官方大权重，没有产出公开LPIPS成绩。

`allow_untrained=True`仅供结构/公式测试和显式自定义实验。`learned=False`
计算无校准的归一化特征基线，也不能称为经过人类感知标注校准的LPIPS。
冻结模型不能训练校准器；LPIPS的BAPPS排序校准训练不属于此模块。

## 与训练框架组合

```python
from aster.models import AutoencoderConfig, AutoencoderKL, LPIPS
from aster.methods.perceptual_autoencoder import PerceptualAutoencoderMethod
from aster.training import Trainer

# 来自已审核、完整导入本地参考权重后save_pretrained生成的制品。
metric = LPIPS.from_pretrained('/data/artifacts/lpips-vgg-0.1')
engine = Trainer(AutoencoderKL(AutoencoderConfig()), zero_stage=3,
                 precision='bf16', accumulation_steps=4, ema_decay=0.999)
method = PerceptualAutoencoderMethod(engine, metric, kl_weight=1e-6,
                                    perceptual_weight=1., pixel_reduction='sum')
# 每个microbatch: {'sample': RGB[B,3,H,W]}，明确归一化至[-1,1]。
# 可提供posterior_noise[B,latent_channels,H/downsample,W/downsample]，
# 用于可复核的同噪声对照；不提供时使用checkpoint管理的训练随机流。
result = method.update(microbatches)
engine.save_checkpoint('/data/runs/vae/checkpoint')
```

固定感知网络作为`perceptual_autoencoder_metric`冻结角色，权重也保存在统一
checkpoint中。恢复新进程时须以相同感知制品重建方法，错误来源会被配置指纹
拒绝；随后checkpoint恢复该角色、生成器、优化器、EMA、随机流和方法进度。
未经此Method管理的游离感知对象不能冒称已经进入checkpoint。

损失对齐CompVis的非对抗项：每图LPIPS广播至该图所有像素，与L1相加，构造
`NLL=(L1+w*LPIPS)*exp(-logvar)+logvar`，对坐标求和、对全局样本数取平均。
KL对潜坐标求和、对全局样本数取平均。`pixel_reduction='mean'`是显式工程变体，
不能与官方sum模式共享同一损失权重并声称比例不变。logvar固定，与原KL编码器
优化器未包含loss.logvar的有效行为一致；本方法未加入GAN判别器/自适应GAN权重。

LPIPS保持FP32，生成器仍可BF16；冻结参数不意味着切断重建分支的输入梯度。
输入规范、完整累积窗口、噪声维度、权重内容都在首个参数gather前核验。
目前支持纯DP以及ZeRO0/1/2/3；不自动推断TP/PP/EP/ETP/CP的感知网络分片。

## 已有证据和数值修正

- 13个单元/工作流测试：VGG/Alex各版本和空间分支的FP64独立公式、双输入梯度、
  原始权重映射、完整导出重载、FP32/BF16和ZeRO0/3恢复、整窗口坏数据拒绝。
- 真实DP2测试覆盖全部ZeRO阶段、每rank不等microbatch大小、同噪声全批次梯度
  范数/权重更新对照、精确恢复，以及单rank坏输入的对称拒绝。
- 2个固定源码官方oracle已经实际执行：完整标准宽度VGG/Alex，原LPIPS前向及
  特征切片类不改body，比对同权重输出和双输入梯度。只将torchvision的普通
  卷积骨架用本地同结构构造器提供；不调用整个外部训练/下载模块。

归一化仍是`x/(sqrt(sum(x²))+1e-10)`，epsilon在平方根外。零向量处采用
`vector_norm`的有限零次梯度，避免旧sqrt链在退化输入上产生`0*inf`的NaN。
共享VAE的KL统计也已提升FP32并用`expm1(logvar)`消除近零相消，FP64保持FP64。
两者是明确的数值稳定性改进，不假称与上游退化输入的NaN行为逐bit一致。

来源：

- [PerceptualSimilarity固定版本](https://github.com/richzhang/PerceptualSimilarity/tree/082bb24f84c091ea94de2867d34c4544f68e0963)
- [CompVis感知/NLL/KL与对抗损失](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/losses/contperceptual.py)
- [CompVis编码器优化器的实际参数归属](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/autoencoder.py)

标准感知权重和公开数据质量结果、GAN联合优化、更广泛视频/音频感知指标及GPU
性能仍需后续工程验证。图像LPIPS不能直接替代音频质量指标或视频FVD。
