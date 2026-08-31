# World Models：视觉、动力学、控制器共享训练生命周期

这里实现作者 Doom 分支的 VAE → MDN-RNN → CMA-ES 控制器，不包含自动驾驶。
运行时只依赖本仓原生代码与 PyTorch，不启动游戏、不调用 TensorFlow 或 pycma。
训练小尺寸实例能检验公式和流程，但不构成真实 Doom、通用机器人或公开基准成绩。

## 结构和不能混用的公式

`models/vmc.py` 包含三个独立、可本地保存/加载的配置与模型。VAE 复用原生
64×64 卷积编解码原语，保持 NHWC 展平顺序，输出 sigmoid 像素。重构损失是
每张图像像素平方误差之和；KL 是每个样本各 latent 维求和后施加
`max(KL, kl_tolerance * latent_size)`。这不同于 PlaNet 的 likelihood/free-nats 约定。

MDN 使用 TF LayerNormBasicLSTMCell 的 i,j,f,o 门顺序、forget bias=1。
输入是当前 `[z_t, action_t, restart_t]`；restart 先清空 cell/hidden，但保留当前动作。
可选 recurrent dropout 只作用于候选 cell；output dropout 不改变下一步的 hidden。
每个 latent 坐标有自己的 K 分量混合高斯，不能替换成共享 K 分量的多元分布。
目标是下一帧 latent 和 restart；跨 episode 的下一帧仍然是监督数据，不能直接删除。

混合 NLL 按有效 transition×latent 坐标归一化，restart BCE 单独按有效 transition
归一化，并对正样本加权。梯度裁剪使用 `max_grad_norm=None, max_grad_value=1.`：
先完成梯度累积和全局有效数量归一化，再逐元素裁剪，不能在各 rank 上先裁剪局部和。
显式初始化与作者 TF 默认随机初始化不宣称逐 bit 一致；同权重公式由独立测试检查。

VAE 编码保存 `mean/logvar`，MDN 每次训练重新采样，避免固定一次随机 z。
采样时温度同时作用于 mixture logits/T 和高斯 std×sqrt(T)，restart 使用 logit>0。
控制器输入顺序为 `[z, cell, hidden]`，可显式选择 `[z, hidden]`，线性层无 bias 后接 tanh。

## 最小真实调用

```python
import torch
from aster.models.vmc import VMCVAE, VMCVAEConfig, MDNRNN, MDNRNNConfig
from aster.methods.vmc import VMCVAEObjective, MDNRNNObjective, encode_vmc_episodes
from aster.planning.vmc import VMCControllerSearch
from aster.training import Trainer

# images: float [B,T,3,64,64]、范围[0,1]；actions: [B,T,A]；restart: bool[B,T]
vae = VMCVAE(VMCVAEConfig(latent_size=64))
vision = Trainer(vae, VMCVAEObjective(), lr=1e-4)
vision.step([{'images': images.flatten(0, 1)}])
encoded = encode_vmc_episodes(vae, images)
mdn = MDNRNN(MDNRNNConfig(latent_size=64, action_dim=actions.shape[-1]))
memory = Trainer(mdn, MDNRNNObjective(sequence_length=images.shape[1]),
    max_grad_norm=None, max_grad_value=1.,
    optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-3, eps=1e-4))
memory.step([dict(encoded, actions=actions, restart=restart)])
search = VMCControllerSearch(memory, encoded['mean'][:, 0], encoded['logvar'][:, 0])
memory.register_state('controller_search', search)
search.step()
memory.save_checkpoint('run/checkpoint')
search.controller().save_pretrained('run/controller')
```

上述一步训练只展示接口，实际必须训练到验证集收敛后才冻结 dynamics 和搜索控制器。
ZeRO3/分布式训练后，独立编码需要先由所有 rank 参与 `export_state_dict`，再在评估
进程建立完整冻结 VAE；不能把局部分片当成完整模型直接编码。搜索构造本身会集体导出
MDN 快照。后续搜索应在明确协调进程运行；当前不宣称已实现 MPI 分布式候选评估。
训练 checkpoint 和独立模型制品不同，恢复 checkpoint 须使用对应可信来源协议。

## 搜索、恢复与评价

`optimization/evolution.py` 原生实现作者 purecma 正权重 CMA-ES：完整协方差、进化路径、
步长控制和延迟特征分解。候选 ask/tell 为事务式协议，非有限结果拒绝提交；保存 pending
候选、随机数状态、协方差、路径和计数，可在 ask 后恢复。大协方差分配有显式内存上限。
这不是 pycma 的全部变体，尚不包含 active negative weights、重启或噪声处理策略。

搜索身份绑定真实 MDN 权重、配置、更新计数、初始分布和 rollout 协议。梦境每步实际
调用 MDN，统计 episode return，并区分 predicted termination 与 horizon truncation。
默认 reward=1 包含死亡那一步，符合作者 Doom 梦境。不同模型/数据不能复用旧搜索状态。
L2 penalty 默认零；开启时采用最小化 `-return + penalty`，明确修正旧 wrapper 的相反符号。

测试包括 VAE 真反传与编码、独立 LSTM/混合密度公式、FP32/BF16、ZeRO0/3、EMA、
随机 dropout/latent 精确续跑、真实 DP2 全部 ZeRO0–3 的不等有效计数更新对照、CMA
公式和 pending 恢复。合成存活任务实际训练 MDN 后用 CMA 提高回报；它不是 Doom 分数。

## 顺序分块TBPTT与完整续跑

`methods/vmc_stream.py`提供`VMCSequenceStream`和`MDNStreamMethod`，并非每batch
重新清零的目标包装。数据先按episode打乱、拼接、截取全局`B*T`的整数倍，再排列
为B条连续stream；DP只划分这些行，允许各rank局部B不同，但时间分块数相同。
每个episode首帧生成restart。VAE的mean/logvar保存不变，每次读取chunk重新采样
latent；sampler使用独立CPU随机生成器，不污染模型dropout随机数。

```python
from aster.methods.vmc_stream import VMCSequenceStream, MDNStreamMethod

# episodes是独立episode列表；每项mean/logvar[T,Z]、actions[T,A]。
stream = VMCSequenceStream(episodes, batch_size=100, sequence_length=500,
    seed=7, rank=memory.parallel.dp.rank, world_size=memory.parallel.dp.size)
method = MDNStreamMethod(memory, stream)
method.step()                        # 读取候选chunk，训练成功后提交cursor/carry
memory.save_checkpoint('run/mdn')   # sampler与LSTM状态自动注册在同一个checkpoint
method.step()
memory.load_checkpoint('run/mdn')
method.step()                        # 固定拓扑下一次更新/隐藏状态/随机流完全一致
```

这里明确采用作者`legacy_disjoint`规则：每个chunk有T帧，模型只展开前T−1帧，
末帧仅作target；下一个不重叠chunk继承final_state。因此边界的末帧→下一chunk
首帧这对transition不参与监督，末帧也未作为RNN输入。这是锁定源码的实际行为，
不能默默改成重叠窗口后宣称复现原配方。每个chunk之间detach隐藏状态，限制反向
时间跨度；`accumulation_steps>1`则连续多个chunk共用一次optimizer更新，保持
每chunk截断，但更新频率不同于作者默认的每chunk一步，属显式配置。

默认不重复或padding尾部不足一个累积窗口的数据；此时`step()`前置拒绝。
完成epoch后`method.advance_epoch()`重新shuffle并清空carry；若确需丢弃剩余
chunk，必须显式`advance_epoch(drop_remaining=True)`，数量进入checkpoint。
原始不足`B*T`的帧尾部也在stream配置的`dropped_frames`中记录。

注册状态包含数据内容SHA256、全局/局部行布局、epoch、permutation、cursor、
shuffle RNG、latent RNG、已detach的cell/hidden与optimizer更新时钟。数据或
配置变化拒绝恢复；外部绕过Method推进同一角色也会被时钟检查拒绝。实际forward
后异常、optimizer部分写入或overflow都不提交sampler/carry，并封锁导出/保存，
须恢复最后完整checkpoint；这不是对optimizer/RNG内存事务回滚的虚假承诺。
尚未开始forward的错误在各rank对称拒绝，可修正后继续。

当前支持DP×ZeRO0–3及已有CPU optimizer/parameter offload，FP32/BF16；
TP/PP/CP/GTP/EP明确拒绝。全episode数据规范成FP32保存在host，epoch布局另有
拼接副本，不宣称磁盘流式加载或低峰值存储。随机分布与作者相同，但torch RNG
不是NumPy RNG；也不默认复刻作者读取过程中额外float16量化的舍入。
rank-local stream/carry尚无跨拓扑转换，portable checkpoint拒绝丢弃这些状态；
标准模型权重仍可集体导出成独立冻结MDN。

新增证据：`tests/unit/test_vmc_stream.py`的独立分布损失/截断梯度/更新对照，
FP32/BF16、全ZeRO/Adam/offload、epoch与真实失败续跑；
`tests/distributed/test_vmc_stream_distributed.py`真实DP2、全局3行按1/2分配、
ZeRO0–3逐参数梯度/全局norm/状态对照和BF16随机精确恢复。设置
`ASTER_RUN_REMOTE_VMC_ORACLE=1`才执行从锁定源提取的未改动`create_batches`
两epoch对照；源SHA256为`3d3d6a976157d7ef0a6482659901373d3596a1b41e9e83f5b49d27419a159cf7`。
这些是CPU数学与存储/通信证据，未提供CUDA或公开Doom模型效果结论。

仍待补齐：原始数据文件读取及TF checkpoint映射、完整游戏采集、MPI候选评估、
跨拓扑stream重划分、真实公开多种子环境成绩和CUDA吞吐。已有测试不覆盖这些项。

## 官方来源

- [WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments/tree/fd982b9691a941b52c6addbde29bc801ca6202c8)：`doomrnn/doomrnn.py`、`model.py`、`rnn_train.py`、`vae_train.py`、`es.py`；README 标注 MIT，但本次未找到完整独立 LICENSE 文件，发布审计不作已完成声明。
- [TensorFlow v1.8 LSTM](https://github.com/tensorflow/tensorflow/blob/v1.8.0/tensorflow/contrib/rnn/python/ops/rnn_cell.py)：门、归一化和 dropout 语义；不调用其运行时。
- [作者 purecma](https://github.com/CMA-ES/pycma/blob/master/cma/purecma.py)：正权重 ask/tell 公式，源码声明 public domain；不等同于完整 pycma 功能或所有商业许可均已审核。
