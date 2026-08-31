# 方法与端到端组合

所有训练阶段都交给`Trainer.phase`，不隐藏第二套优化器或调用其他仓库的Trainer。
“原生公式/存储/恢复已测”与“预训练模型质量/硬件吞吐已达到官方水平”分别记录。
本页给出真正连接的API；输入张量来自调用者有许可的数据，不自动下载权重。

## 从张量数据进入同一训练框架

`TensorTreeDataset`读取`torch.load(weights_only=True, mmap=True)`可安全解码的张量树，
所有叶子第一维必须是样本数。文件指纹、camera顺序、归一化和split作为显式配置。
`fit_tensors`通过`build_model`和`build_objective`训练，并发布与语言配方相同的制品。
flow、RSSM、ACT已有真实读文件→训练→恢复→制品测试。

```python
from aster.core import ArtifactStore
from aster.tensor_recipes import fit_tensors

# config包含model、objective、data、preprocessing、training；不自动猜pixel/action尺度。
result = fit_tensors(config, {}, output_directory, ArtifactStore(artifact_directory))
```

DP/ZeRO使用显式`parallel=context`或`distributed-train --kind tensor`，参见TRAINING.md。
低层TP/PP等能力不等于任意模型已具备自动分片配方。

## 生成模型的组合优化

`LatentGenerationPipeline`把VAE缩放约定、field模型和采样器组合为保存/重载单位。
`LatentFieldObjective`可训练像素编码后的场，也能读取带encoder身份的预计算latent。
冻结VAE不表示可以忽略其scale/shift。EDM网络返回`edm_residual`，一致性网络返回
`consistency_residual`；和epsilon/x0/velocity的输出不能互换。

可连接的不同加速路径包括：

- DDPM/DDIM→respaced日程或DPM++2M；蒸馏使用真实两步teacher到一步student目标。
- Flow→reflow耦合数据→新field训练；不是把新独立噪声与teacher终点错误配对。
- 多步teacher→consistency目标，或DMD的fake-score/生成器两角色迭代。
- 结构剪枝→蒸馏恢复→QAT/打包量化→同协议评价→质量gate。
- DiT残差缓存真正跳过主干；误差审计可禁用复用，最终仍必须通过端到端质量评价。

各配方有自己的数学假设。少步、量化、剪枝和近似缓存属于有损变换，不能根据
“能运行”宣称质量不下降；比较时固定数据、特征提取器、prompt、种子和计分方法。

## 在线文本强化学习

`OnPolicyRLMethod`实际执行当前策略快照→原生队列生成→奖励→冻结参考→更新。
RLOO计算完整组的leave-one-out baseline，KL作为序列奖励惩罚，并裁剪整段响应概率比。
GRPO则使用对应的逐token surrogate和KL项。组优势在microbatch划分前计算。

```python
import asyncio
from aster.core import digest_json
from aster.methods.policy_gradient import OnPolicyRLMethod
from aster.inference import SamplingConfig

method = OnPolicyRLMethod(engine, reference_model, tokenizer,
    reward=score_completion, reward_id="my-verifier-v1",
    reference_tokenizer_fingerprint=digest_json(tokenizer.to_dict()),
    algorithm="rloo", group_size=4)
result = asyncio.run(method.update(prompt_token_ids,
    sampling=SamplingConfig(max_new_tokens=128, seed=17,
                            eos_token_ids=(tokenizer.eos_token_id,))))
```

奖励回调签名为`score_completion(rollout, prompt_index)`，返回有限标量。
模型概率、行为概率、终止原因和奖励版本均保留；失败样本不会静默从训练组消失。
当前在线控制器为单rank、temperature=1完整softmax，不接受greedy/top-p当作论文
on-policy分布。分布式loss本身可用，不冒称已经自动完成异步多机rollout集群。
RLOO和GRPO均已测真实更新、冻结参考和恢复后的下一轮轨迹/参数精确一致。

来源：[TRL RLOO](https://github.com/huggingface/trl/blob/main/trl/trainer/rloo_trainer.py)。

## 蒸馏不是一个泛用KL的别名

`DistillationObjective`提供同词表forward/reverse/混合KL/JS与特征关系目标。
`OnPolicyDistillationMethod`用学生真实生成的上下文让冻结教师重新打分；
`sequence_distillation_examples`可以将教师文本重新分词到不同词表，精确检查prompt边界。
截断响应不自动加EOS，验证失败有独立收据。

`TinyBERTStudent`把共享`fit_dense`放入学生角色；`TinyBERTObjective`使用**softmax前**
缩放QK分数和隐藏层投影MSE。`official_slots`保留原代码padding slot分母，
`valid`是显式变体，不能混为相同目标。层映射由调用者声明。

`MiniLMObjective(version=1)`迁移QK/VV；v2重新分relation heads迁移QQ/KK/VV。
后者允许不同hidden大小和attention head数，但要求token位置语义一致。
`EncoderDistillationMethod.export_student`去除训练投影，导出真正可部署的原生学生。
TinyBERT/MiniLM均已验证ZeRO-3更新、导出和精确续跑。

```python
from aster.methods.encoder_distillation import TinyBERTStudent, EncoderDistillationMethod
from aster.training import Trainer

student = TinyBERTStudent(native_bert_student, teacher.config.hidden_size)
engine = Trainer(student, lr=3e-4, zero_stage=3)
method = EncoderDistillationMethod(engine, teacher,
    tokenizer_fingerprints=(tokenizer_id, tokenizer_id), kind="tinybert",
    attention_pairs=((0, 1),), hidden_pairs=((0, 0), (1, 2)))
method.update([batch])
deployable_student = method.export_student()  # 分布式时所有rank调用，仅rank0返回模型
```

来源：[TinyBERT训练](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/general_distill.py)、
[MiniLM官方工程](https://github.com/microsoft/unilm/tree/master/minilm)、
[MiniLMv2论文公式](https://aclanthology.org/2021.findings-acl.188/)。

## Agent数据回流

`NativeAgentPolicy`记录实际tokenizer、renderer/processor指纹及真实采样配置。
`verified_agent_corpus`验证事件哈希链、turn完成状态、验证结果和动作收据；
每个决策保存当时真实prompt，只有新action标签有效，工具观察和历史动作不重复监督。
所有未验证/失败turn保留拒绝原因。`AgentSFTMethod`接统一Trainer并绑定语料身份。

这是验证筛选后的SFT/序列蒸馏，不是将greedy轨迹改名为on-policy Agent RL。
读取日志不会执行工具、恢复授权或重试有歧义的副作用。
已有原生模型→读文件工具→验证→学生训练→精确恢复的集成测试。

## 机器人动作与世界模型

`PiVLA`组合SigLIP多相机图像、Gemma式文本embedding与独立动作专家。
π0和π0.5的连续state/文本state语义不同，配置显式检查。训练与缓存采样共用权重。
`OpenVLAForActionPrediction`则是DINOv2+SigLIP视觉融合、BOS后视觉插入及离散动作token，
训练必须调用`align_labels`给插入视觉位置加-100，不能假设输出长度等于原文本长度。
输出动作还需ActionSpec/反归一化，框架不自动驱动真实硬件。

`JEPAModel`使用图像/视频tubelet encoder、masked predictor和只更新encoder的EMA目标；
mask在attention前应用，防止目标像素泄漏。它不是RSSM，也不把特征预测冒称像素视频生成。

`TDMPC2WorldModel`是另一条完整隐式世界模型路径：SimNorm潜变量、动作动力学、
symlog two-hot reward/Q、Q ensemble；`TDMPC2Policy`提供独立Gaussian先验。
`TDMPC2Method`按world→policy→target-Q顺序更新，所有优化器与目标由共享Trainer拥有。
`TDMPC2Planner`实现策略轨迹注入、精英softmax加权MPPI、warm start与首动作探索。
state/RGB64、多任务动作mask、episodic单任务均有原生路径；方法目前使用完整episode窗口。
多任务embedding不再在forward隐式修改参数。方法在目标计算之前和world优化之后
调用统一Trainer的持久行投影：全DP访问行取并集，更新真实ZeRO owner与CPU/磁盘
master，不重置moment。该投影支持DP、ZeRO 0–3和offload；TP/PP布局显式拒绝。
部署模型仍需要对实际访问task行执行相同投影，不能假定优化器更新后的所有行都未超限。

```python
import torch
from aster.training import Trainer
from aster.methods.tdmpc2 import TDMPC2Method, TDMPC2Planner

# 原作者配方使用Adam；明确传入，勿依赖其他配方的AdamW默认值。
engine = Trainer(world, optimizer=torch.optim.Adam(world.parameters(), lr=3e-4))
method = TDMPC2Method(engine, prior,
    policy_optimizer=torch.optim.Adam(prior.parameters(), lr=3e-4, eps=1e-5))
method.update([episode_window])
planner = TDMPC2Planner(world, prior)
normalized_action, predictions = planner.plan(observation, first=True, eval_mode=True)
```

TD-MPC2已测核心公式、两阶段真实训练/target、精确续跑、状态/RGB/多任务及MPPI。
这是公式/工作流测试，未执行原仓库整包或公开控制任务benchmark。
来源：[TD-MPC2作者工程](https://github.com/nicklashansen/tdmpc2)，实际读取了world_model、
layers、math、scale、init和tdmpc2训练/规划源码；本轮未成功解析main的不可变commit，
因此不把动态分支引用冒称已完成SourceLock。

## 视频生成不是逐帧2D扩散

`WanVideoDiT`是原生Wan2.1计算图：三维patch、按T/H/W独立频率的RoPE、完整投影宽度
的QK RMSNorm、六路时间调制，以及**分别归一化后相加**的图像/文本cross-attention。
支持T2V、I2V与首尾帧条件。原始T5 embedding先补0再经过有bias的投影，不能把这些
补位误改成attention mask。`time`是噪声比例sigma，内部乘`time_scale=1000`；
速度目标是noise-data，采样从1走到0。

`WanVideoVAE`首帧独立，随后按时间压缩步长成组；解码输出1、stride、stride帧。
缓存局部持有且训练不detach，因此不会跨视频泄漏，跨块梯度也不截断。
默认小模型的latent统计是恒等变换；仅`WanVAEConfig.public_wan21()`声明公开16通道
checkpoint的mean/std。`decode_chunks`允许逐块输出；模型内不clamp训练重建梯度。

```python
from aster.methods.video_generation import VideoGenerationPipeline, WanVideoObjective
from aster.training import Trainer

pipeline = VideoGenerationPipeline(native_wan, native_video_vae.eval())
batch = pipeline.training_batch(video, text_features, image_features=image_features)
engine = Trainer(native_wan, WanVideoObjective(), lr=1e-4)
engine.step([batch])
pipeline.eval()
frames = pipeline.generate(noise, batch['condition'], steps=30, solver='heun', shift=5.)
```

T2V不传`image_features`。I2V条件使用首帧+全0未来经VAE编码，以及按时间压缩分组的
mask，不从目标未来帧泄漏信息。输入文字/图像特征需绑定实际编码器/processor制品，
不能将随机特征或其他编码器冒称官方UMT5/CLIP特征。Euler/Heun是明确的solver选择，
尚不冒称官方UniPC/DPM++、Cosmos动作控制、Genie或全部视频生成模型。

针对性测试覆盖完整场公式/全参数与输入梯度、3D位置/padding、VAE时间卷积、
因果前缀、原生AE和video flow训练、模型保存/加载、精确续跑。没有公开预训练成绩。
来源：[Wan视频骨干](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py)、
[因果视频VAE](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py)、
[I2V条件构造](https://github.com/Wan-Video/Wan2.1/blob/main/wan/image2video.py)。
已解析仓库HEAD `ae487cc653b4a1791fec8201af20d2102a2514f3`，但固定SHA raw内容获取未成功；
本轮实际读取的是main内容，不能把解析HEAD等同于逐文件SHA核验。

## OT、反演与跨模态蒸馏的组合

`transport_pairing`提供等权精确Hungarian与log域Sinkhorn耦合，返回真实配对索引以同步
重排标签/条件。Sinkhorn不收敛会报错，不回退成不同目标的随机配对。它直接给
`FlowObjective`准备data/noise；不依赖POT/SciPy。
`integrate_flow`支持正/反向Euler、Heun、RK4；`flow_log_likelihood`以连续变量变换
公式积分密度，可选精确散度或固定Hutchinson探针。后者假设batch样本互不耦合，
不适用于跨样本归一化/注意力或随机训练态网络。
来源：[TorchCFM OT](https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py)、
[Meta Flow Matching ODE](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py)。

`PredictionAlignment`固定输出空间、处理器和教师制品身份；
`MultimodalDistillationMethod`将ACT动作、DiT/视频场或明确选定的feature目标接入共享
Trainer。field蒸馏要求师生使用完全相同的噪声样本与时间参数化，action蒸馏不让教师
偷看训练目标动作posterior。有效元素mask按实际单位归一化。已有ACT、DiT和真实LLaVA
logits/hidden KD的更新与恢复测试；不是把所有模态强行套入同词表KL。

`GrootVLA`/`GrootFlowObjective`则接原生GR00T N1.7视觉语言骨干、embodiment映射、
交替cross/self-attention动作头、flow训练与RTC重叠冻结。该实现还测试了cross-KV
缓存与重算一致性、完整小型训练和随机目标精确续跑；机器人动作仍需评测协议与环境
许可，框架不会自动驱动真实硬件。

## CQL：真实离线策略、双 Q 与多角色训练

`models.conservative.CQLPolicy` / `CQLTwinQ` 和 `methods.conservative.CQLMethod`
是本仓库原生模块，不导入 rlkit。离线样本约定为 FP32 `observations`、
`next_observations`、`actions`、`rewards` 与独立 bool `terminated`/`truncated`；
动作先按数据协议归一化到 [-1,1]。本方法不猜测额外 n-step discount。

```python
import torch
from aster.models import CQLPolicy, CQLPolicyConfig
from aster.models.conservative import CQLTwinQ
from aster.methods.conservative import CQLMethod
from aster.training import Trainer

policy = CQLPolicy(CQLPolicyConfig(observation_dim=17, action_dim=6))
trainer = Trainer(policy, optimizer_factory=lambda p: torch.optim.Adam(p, lr=3e-4),
                  max_grad_norm=None, zero_stage=3)
method = CQLMethod(trainer, CQLTwinQ(17, 6), lagrange=True,
                   target_action_gap=5., critic_lr=3e-4, coefficient_lr=3e-4)
result = method.update([replay_batch])  # 外部加载并核验的数据，不在方法里隐式采环境。
trainer.save_checkpoint(checkpoint_path)
# 任意 ZeRO 布局先按名字收集真正完整权重，再构建独立部署策略。
weights = trainer.export_state_dict()  # 所有 rank 一起参与收集。
if trainer.parallel.world.rank == 0:
    deployed = CQLPolicy(policy.config)
    deployed.load_state_dict(weights)
    deployed.save_pretrained(policy_directory)
```

支持重要性采样/非重要性采样保守目标、温度自动学习、Lagrange action-gap、
BC warmup、熵/无熵 TD backup、max-Q backup。保留官方未归一化 logsumexp 和
`Q(s, a~π(.|s'))` 的状态配对。多角色更新不会跨参数更新继续反向旧图：固定旧策略
proposal 后先做旧 Q 下的 actor 更新，再做 Q 更新；独立完整梯度测试核验这一交换。
中途失败拒绝保存半轮状态。真实 DP2、不等长批次、ZeRO 0–3、BF16 0/3、下一轮
随机更新精确恢复均有测试；未在多机 CUDA 环境测量吞吐。

这里的“重要性采样”指保守目标中的动作 proposal 密度修正，不等于 PER 的状态
回放权重。当前 CQL 要求均匀回放，非全一 `importance_weights` 在任何更新/抽噪前
显式拒绝；避免把带 PER 权重的共享 replay 接入后悄悄丢掉权重。

固定离线连续 bandit 测试中，actor 只收到奖励/Q 梯度而非目标动作监督；200 轮
确定性动作误差从约 0.205 降到 0.034。这是算法连接性测试，不是 D4RL 成绩。
网络、概率密度和初始化的来源为官方
[CQL trainer](https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py)、
[policy](https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/policies.py)、
[distribution](https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/distributions.py)。
本次读取分支源码，未取得逐文件固定 commit/hash 的完整锁，且不推断其未核实的许可。

## MTP 与主模型联合训练

`MultiTokenPredictionObjective(depth=2, base_weight=1., mtp_weight=.1)`
与 `QwenMTPForCausalLM` 共同由一个 Trainer 拥有；共享 embedding/head 不重复放入
优化器。主 next-token loss 和每个 draft 深度按各自有效目标 token 数归一化，再按
深度平均。`loss_mask`、`-100` 和右侧 padding 都在正确的未来目标位置处理。
`detach_base` 切断隐藏状态梯度，不等于冻结共享参数；真正仅训 draft 应先冻结主干。
当前训练目标明确拒绝没有完成状态重置实现的 packed 文档和隐式多模态输入。

结构依据 [vLLM Qwen MTP](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)，
辅助 CE/深度平均依据 [DeepSeek-V3 报告第 2.2 节](https://arxiv.org/html/2412.19437v1#S2.SS2)。
这是已公开结构和目标的明确组合，不冒称 Qwen 未公开的完整预训练配方。
配置工厂可直接选 `name=mtp`；视频与机器人 flow 目标分别选 `wan_video`/`groot_flow`。

## 高斯条件路径和 SB-CFM

`GaussianFlowPath` 区分固定方差双端条件路径、target-conditional 路径和 Brownian
bridge 路径；`GaussianFlowObjective` 将它们接入原生 Hungarian/Sinkhorn、条件同步
重排和统一 Trainer。配置入口为 `name=gaussian_flow`。SB 默认 exact，可显式选择
`coupling=sinkhorn`，默认熵正则系数为 `2*sigma**2`。最优耦合发生在每个 rank 的每个
microbatch 中，拆批会改变 OT 目标，不承诺它等于单次全局大 batch 配对。

target 路径中的 `noise` 是标准高斯 epsilon；另两类路径里它是起点 x0，独立桥噪声
放在 `perturbation`。target 的终点是围绕数据、标准差 sigma 的高斯，而不是当
sigma>0 时仍然声称精确干净数据。SB 的条件速度有端点奇点，因此训练使用显式记录的
开区间时间；数值分母 epsilon 与作者实现一致，也可显式取 0 验证解析公式。

生成直接使用现有 `integrate_flow` 或 `sample_flow`，无需另一套采样后端。
这实现的是边缘概率流训练，不冒称已同时训练 score 或实现随机 SDE 似然。
独立公式/输入梯度、解析高斯终点、OT 条件索引、真实 DiT + ZeRO3 和随机恢复有测试。
来源：[作者 TorchCFM 路径源码](https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/conditional_flow_matching.py)，
运行时不导入该库或它使用的 POT/SciPy。

MuZero 的原生模型、搜索、重新分析、PER 和统一训练闭环见 [MUZERO.md](MUZERO.md)。
