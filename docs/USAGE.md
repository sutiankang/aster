# 使用与组合

## 一个训练器，不同目标

`Trainer.step`接收**恰好**`accumulation_steps`个microbatch。每个LossTerm的全局有效计数独立归一化；token、样本、动作元素的分母不会混用。

```python
from aster.models import UNetConfig, UNet2D
from aster.methods.generation import DiffusionSchedule, DiffusionObjective
from aster.training import Trainer

model = UNet2D(UNetConfig(in_channels=3))
objective = DiffusionObjective(DiffusionSchedule.create(1000), min_snr_gamma=5.)
trainer = Trainer(model, objective, device="cpu", lr=3e-4, ema_decay=.999)
result = trainer.step([{"sample": images}])  # images: BCHW，像素或已定义缩放的latent
trainer.save_checkpoint("runs/image-001/checkpoint")
```

真实训练需设置数据归一化、EMA、足够步数、独立验证集及公开评价协议。示例不保证未经训练的输出质量。

## 生成链

模型与采样器解耦：

- `UNet2D`：多尺度残差、时间FiLM、空间注意力、对称skip。
- `DiT`：patch、二维位置编码、adaLN-zero、条件与反patch。
- `AutoencoderKL`：Gaussian posterior、重参数化、KL、显式latent scale/shift。
- `DiffusionObjective`：epsilon/x0/v/score、Min-SNR、可学习variance的独立VLB梯度。
- `sample_diffusion`：DDPM或DDIM；少步DDPM须先用`DiffusionSchedule.respaced`重新计算保持边缘分布的日程。
- `EDMObjective/sample_edm`：预条件、log-normal噪声、Karras sigma、Heun与显式churn。
- `FlowObjective/sample_flow`：线性/余弦路径，Euler/Heun/RK4；noise→data与OpenPI式data→noise时间约定不混用。
- `reflow_pairs`保留噪声和teacher终点耦合；`ConsistencyDistillationObjective`用teacher ODE和target映射。
- `DMDMethod`把fake-score在线训练和生成器更新放进两个明确phase。
- `DriftingObjective`在原生特征空间构造停止梯度的多温度力；冻结特征encoder时仍保留输入梯度。

这些都是不同的训练配方，不应把单一Flow MSE改名为所有加速方法。选择哪条链需要匹配数据、teacher权重和质量/成本目标。

## 强化学习与世界模型

`CategoricalActorCritic + PPOObjective`消费固定old log-prob、old value、return和advantage。
`generalized_advantage`严格区分terminal和time-limit truncation；后一种保留bootstrap，但停止跨episode递推。
`ReplayBuffer/NStepAccumulator`保存抽样RNG、覆盖版本、pending n-step片段，挂到`Trainer.register_state`后进入同一checkpoint。

`SACMethod(Trainer(actor), critic)`声明actor、双Q、target、entropy temperature四个角色。
target更新使用训练器的逻辑参数布局接口，不能把ZeRO shard和完整tensor用zip相加。
`GRPOObjective`需要与生成轨迹完全对齐的behavior log-prob、reference log-prob和group advantage；sequence/token/constant分母分别对应明确的目标变体。

`RSSMWorldModel`提供observe/step/imagine，离散latent的straight-through与unimix、分组循环动力学、reward/continue头。
`WorldModelObjective`分别训练重建、reward、continue、动力学KL和表征KL。
`ImaginedActorCritic`额外更新想象轨迹上的策略与价值；`cem_plan`是显式reward-MPC。
向量观测RSSM与完整图像Dreamer配置不是同一效果声明。

## 机器人动作

`ACTPolicy`是条件VAE+DETR动作查询块，训练时posterior读取动作、推理时latent=0。
视觉输入是同仓库encoder的`vision_tokens`，视觉骨干、位置编码和camera顺序必须纳入组合制品。
`ACTObjective`默认与公开ACT一样masked-L1后按padded slot计数；`normalization="valid"`是明确变体。

`ActionSpec`固定名字、单位、坐标、absolute/delta/velocity、控制频率、执行horizon。
`ActionNormalizer`只在训练split拟合；`TemporalEnsembler`按绝对tick对齐，合法零动作不是padding，过期动作不会重用。
仓库的这些类不会直接执行真实硬件动作。

## 官方评价与证据

`ComparisonProtocol`固定数据、任务、预处理、prompt、seed和计分器版本；候选制品ID放在`EvaluationRun`。
`EvaluationRecord`保留成功、错误、超时、跳过；未交付样本在finalize时明确记作missing_result。
`quality_gate`读取并校验真实报告，以逐样本配对bootstrap检验非劣性，不比较两个独立置信区间。

`LanguageEvaluator`用原生模型打分/生成；`lm_eval_adapter`只将官方task请求转交原生实现。
`evaluate_official_language`允许使用官方lm-evaluation-harness的任务与计分；数据下载、使用许可和外部服务授权仍由调用者负责。
`evaluate_language_artifact(store, artifact_id, task_name=..., dataset_revision=..., output_directory=..., max_length=...)`
从校验后的制品加载原生模型和同一tokenizer，提前固定实际样本全集及任务配置，调用
真实官方评测器，再把逐样本准确率/EM与原始请求结果发布为带父制品的评价证据。
可选依赖为`.[language-eval]`，固定lm-eval 0.4.12；`task_manager`可显式给出官方
TaskManager读取本地许可任务。当前桥接一次一个leaf任务，支持`acc/acc_norm/exact_match`；
不是任意聚合指标或可执行代码benchmark的通用授权器。

`limit`始终标记subset；缺结果保留分母计失败，重复doc ID或样本内容变化拒绝发布。
Python/NumPy/Torch/few-shot四组seed都固定，NumPy标量显式转为JSON数值，未知对象和
NaN/Inf不会被str()掩盖。Windows的HF datasets缓存宜使用短独立目录，以避开其
filelock层长路径限制。质量门禁仍需相同任务/数据/预处理协议，不能拿fixture成绩上线。
本地FID/FVD函数是指定特征下的Fréchet数学核；只有同时固定公开特征提取、resize/quantization和样本协议才能报告可比较的FID/FVD。

## 性能优化的边界

量化制品保存真正打包的权重、尺度和算法/校准指纹。当前可用实现见`aster.inference.optimization`；浮点dequant参考forward不等于INT4 GPU算子加速。
部署时先验证制品、warmup，再切换服务版本；已有请求继续绑定旧版本，不跨请求混权重。
多机、多卡和加速kernel是否可用，由实际环境、能力检查与测试证据决定，不通过静默fallback伪装成功。
