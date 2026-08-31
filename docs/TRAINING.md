# 自主训练运行时与验收边界

本模块不调用 Megatron、DeepSpeed 或上游 Trainer。PyTorch 提供张量、自动微分、基础通信和 Adam/AdamW/RAdam/SGD；Aster 拥有目标归一化、调度、分片、角色更新、恢复及制品导出。以下是已落地行为，不表示任意组合或 GPU 性能都已验收。

## 统一目标与角色

```python
import torch
from aster.core import LossTerm
from aster.training import Trainer

def objective(model, batch):
    x, target = batch
    error = (model(x) - target).square()
    return LossTerm(error.sum(), torch.tensor(error.numel(), device=error.device), "elements")

trainer = Trainer(model, objective, lr=3e-4, accumulation_steps=2)
result = trainer.step([microbatch_a, microbatch_b])
trainer.save_checkpoint("run/checkpoint.json")
```

`LossTerm.numerator` 是可微分子，`denominator` 是不求导的实际计数。每个 loss term 独立累积梯度，得到该目标全部 microbatch/rank 的计数后再组合 `weight * sum(grad_numerator) / sum(count)`。tokens、samples、actions 等不同单位不能先混合梯度再统一除数。所有 rank 和 microbatch 的 name/unit/weight 顺序必须一致；空 rank 仍返回同名零分子/零分母 term 并参与通信，不能跳过整个 collective。

目标计数默认归约 `DP×CP`，因为常规 TP rank 计算同一批样本。SP 分散 token 的目标必须显式 `register_loss_group("loss", context.dp_tp)`；不是看到 `unit="tokens"` 就推测通信域。注册时核验所有成员的域声明，域写入 checkpoint。不同 term 可有不同域。

`add_role(name, module, optimizer=..., trainable=...)` 建立唯一参数所有者；同一 tensor 不允许跨 role/optimizer 共享。`phase(name, role=..., objective=..., microbatches=..., freeze_roles=...)` 只更新指定 owner。冻结 critic 不切断对 actor 输入的梯度，退出时恢复原始 `requires_grad`。未参与目标的参数保持 `grad=None`，不凭空进行 AdamW decay。一个 phase 发生异常后运行时锁定，必须从完整 checkpoint 恢复。

`Trainer(..., optimizer_factory=fn)` 与 `add_role(..., optimizer_factory=fn)` 在模型移动/ZeRO3分片完成后调用 `fn(list_of_actual_parameters)`。列表为此role按模块声明顺序去重的真实trainable参数，ZeRO3时不是已释放的旧参数。工厂与optimizer实例互斥，返回新建原生optimizer，必须恰好拥有全部参数且无重复；可建立多组不同lr/eps/betas/weight_decay。工厂本身不序列化、不猜 `__dict__`，checkpoint记录实际优化器完整类名、各组配置及对应参数名字；同shape调换组归属也拒绝精确恢复。ZeRO3仍拒绝提前绑定旧参数的optimizer实例，工厂失败会在所有rank同步报错；若模型已分片，重试须重新构建该角色模块。

```python
trainer = Trainer(model, objective, zero_stage=3,
                  optimizer_factory=lambda parameters: torch.optim.Adam(
                      parameters, lr=1e-3, eps=1e-8, weight_decay=0.))
trainer.add_role("policy", policy, optimizer_factory=lambda parameters:
                 torch.optim.Adam(parameters, lr=3e-4, eps=1e-5))
```

Adam的L2先进入梯度及moment，AdamW才是解耦衰减，不能把二者替换；具体语义依据 [PyTorch 2.11 Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html)。ZeRO1/2、CPU/disk offload和portable保留原始class与param_group选项，支持Adam/AdamW/SGD，含Adam AMSGrad状态。已有CPU/DP2非均匀batch、12种ZeRO/卸载组合、coupled-vs-decoupled和跨DP2/ZeRO3→dense下一步测试；GPU fused/capturable/differentiable optimizer模式未做生产验收，不因普通Adam通过即宣称全部内核路径验证。

RAdam同样作为明确的逐元素optimizer实现ZeRO0–3、CPU/disk offload和portable；保留rectification、betas、eps、coupled/decoupled选项，不替换成Adam。`test_training_radam.py`跨8步验证rectification前后及恢复；完整一致性模型生命周期见[CONSISTENCY.md](CONSISTENCY.md)，其中DP2/ZeRO3→dense也检查真实RAdam状态迁移。

`clone_target(source, target, factory=..., source_path="encoder")` 由显式 provider 重建无 optimizer 的冻结 target；可只克隆源 role 的子模块，encoder+predictor 仍属于同一 optimizer。`update_target(source, target, decay, buffers="copy"|"ema", source_path=None)` 默认沿用 clone 的源路径，按逻辑名称逐张量 gather，而非直接 zip ZeRO shard 与完整权重。路径/链接进入 checkpoint 身份；子树遍历安全剥离源前缀，不读空占位参数。整数 buffer 始终复制；浮点 buffer 默认复制。同布局 tied 参数只更新一次。目标保持相同 PP/EP 逻辑分区，可跨 ZeRO/TP 存储布局；PP 子树路径当前需额外全局映射，显式拒绝。它是独立状态变更；actor→critic→temperature→target 的多 phase 方法必须保存自己的 phase cursor，不能将某阶段 overflow 说成具有全方法原子回滚。

## 拓扑和分片

`ParallelConfig(tensor_parallel, pipeline_parallel, context_parallel, data_parallel, gtp_remat=1)` 的 rank 网格为 `[DP, PP, CP, GTP_remat, TP]`；R=1 保持原四轴顺序。调用者初始化/销毁默认进程组；所有 rank 同顺序构造 `ParallelContext`。除基本轴，还有 `dp_cp`、`dp_tp`、`tp_pp`（固定 DP/CP/R 的模型域）、`stage`（固定 PP 的全部副本/张量域）、`dp_gtp`、`dp_cp_gtp`、`dp_tp_gtp`。默认 loss 域为 `dp_cp_gtp`；R=1 与原来 DP×CP 相同。GTP 数据应按 `dp_gtp.rank/size` 分发，不能忽略额外 remat 轴。`Group` 明确持有 ranks、handle、本组 rank；单进程组不会把 `None` 错当 WORLD。

| 能力 | 自主实现 | 已有 CPU 数值/存储证据 | 当前边界 |
|---|---|---|---|
| DP | 分子梯度求和、全局分母、全局唯一参数范数/裁剪、统一 overflow 跳步 | 真实 2 rank，非均匀 batch 与一个全空 mask rank | 无容错 elastic 成员变化 |
| TP | Column/Row linear，互为伴随的通信算子，稳定 vocab CE | TP2×DP2，前向/梯度/AdamW 更新 | provider 显式构造分片层，不自动改写任意模型 |
| ZeRO1 | 每参数 optimizer owner shard，完整梯度规约后取 owner 更新 | optimizer 状态分片、更新/恢复 | Adam/AdamW/RAdam/SGD，未融合 bucket |
| ZeRO2 | 每 term/microbatch 真 reduce-scatter，持久梯度为 shard | 真通信、更新与裁剪对照 | 不假装为异步 RS overlap |
| ZeRO3 | 叶子单元真 shard，原参数 numel=0，前向临时 gather、反向重算后 RS；同布局 tying 共用一个 shard；meta shard init | 常驻元素数、gather/release、2/4 rank 更新；DP2 meta/tied/CPU参数存储/恢复 | 纯函数单 Tensor 叶子；叶子 buffer、含直接参数容器仍拒绝；裸 ParameterList/Dict 索引不调用 forward，整树预检拒绝而非悄悄返回空参数 |
| SP | sequence all-gather/RS 伴随，分片 MLP，replicated bias 额外规约 | SP2 与 dense 更新一致，ZeRO0–3 | 需显式 SP 算子与 loss 域 |
| CP | gather attention 数学 oracle；原生 P2P ring 在线 softmax 与显式 backward | CP2 causal 更新，单 rank MHA/GQA float64 梯度 | 等长局部序列、dropout=0；无 fused FlashAttention kernel |
| EP/ETP | variable-split EP all-to-all与ETP AG/RS；完整Mixtral独立attention TP及expert ETP/EDP布局 | EP2×EDP2旧闭环；新增TP1/2×EP2×ETP2×EDP1及TP2×EP2×ETP2×EDP2，ZeRO0–3完整模型梯度/更新/恢复/导出 | PP/CP/GTP折叠、attention TP非1且不等ETP、capacity dropping、shared expert、DeepSeek与融合kernel未接此provider |
| PP | 单 Tensor stage 边界、动态 shape header、异步 P2P、GPipe/serial/1F1B | PP2×DP2，ZeRO0–3，1F1B graph 峰值 | 未测 GPU/多机；不是 zero-bubble |
| VPP | 本地多个 chunks 的 interleaved1F1B、每 edge 通信标签、一次窗口更新 | PP2×DP2×2 chunks，ZeRO0–3，多目标/非均匀 batch/evaluate 与 dense 一致 | `M % PP == 0`；当前 Gloo，NCCL 不支持 tag，显式拒绝直到单独 channel/顺序化 transport 验收 |
| GTP remat reference | 独立权重重建轴，保留 TP slice；remat AG/RS 与 DP 真副本规约分开 | TP2×R2、DP2×R2 真4进程，存储小于完整权重、更新/clip/恢复/导出一致 | 仅 dense TP×R×DP，`zero_stage=0`；flatten shard ABI，不是官方 dim0 对齐；没有 per-weight ticket/prefetch/量化 AG/CUDAgraph |

SP 不额外乘一个物理 world 轴。旧独立EP算子由provider传组、专家为local owner；EP-only Mixtral把DP划分成EP×EDP，新ETP provider把同一WORLD另排为ETP×EP×EDP，expert只在EDP的同一全局专家副本间归约。未测试的拓扑组合不因单轴测试通过就自动变为“完全支持”。

## 流水线接口

### 完整 Llama / Qwen2 / Qwen3 训练 provider

`parallelize_causal_lm(dense_model, context, pipeline_schedule="1f1b")` 从本地完整模型构造独立的原生训练布局，不改变输入模型。只匹配确切的三个dense配置/DecoderLayer，不把MoE、MLA或视觉模型猜测成同一种结构。TP-only返回`TensorParallelCausalLM`；PP>1返回`CausalPipelineStage`。词表embedding/head、QKV、注意力output、gate/up/down真实分片；模型标准配置及全局FQN保留。

- TP-only使用`TensorParallelCrossEntropyObjective(context)`；公开`forward`仍返回完整`TokenOutput`，专用`forward_sharded`+分布式CE不物化完整词表logits。不同TP看到相同目标，因此分母只跨DP求和。
- 词表以ceil行数补齐存储，padding logits排除在softmax外。显式`_aster_tp_global_shape`进入逻辑layout/恢复身份，导出/portable/target更新恢复真实词表维度。259词表和tied embedding/head保留一个参数所有者；不会把260词表保存成原模型。
- GQA的KV头可整除TP时按头分片；否则保留完整KV参数，只计算本rank query对应的KV头，并汇总复制参数的TP梯度。这是真实的复制路径，不宣称所有KV也已减少存储。Qwen2 QKV bias、Qwen3共享Q/K RMSNorm的额外TP梯度、显式RoPE position、padding和窗口公式均保留。
- PP使用`CausalPipelineCrossEntropyObjective(context)`。`Trainer`在任何F/B调度之前调用目标显式`preflight_microbatches(model,batches)`，检查完整窗口及TP×PP输入一致；hook不得做模型forward或更新。stage边界只传hidden，位置/掩码在各阶段显式持有。不得把WORLD预检放进1F1B的forward，因其他stage此时可能在backward。
- PP按全局层号分区，非首stage不保存embedding，非尾stage不保存norm/head；`parameter_names`完整映射公开参数与buffer。当前支持GPipe/serial/1F1B。PP2跨首尾stage的tied权重作为同一逻辑权重的两个计算副本：使用点梯度在PP域求和、各自使用一致的optimizer状态/配置更新，全局范数只计首stage一次。每次phase前校验两端当前学习率及所有param-group选项；不一致在更新前拒绝。PP>2的跨阶段tied权重仍明确拒绝，尚无仅首尾参与的子域。

真实CPU测试包括三模型TP2×DP2×ZeRO0–3的逐参数梯度/SGD momentum更新/global norm、完全无有效监督的DP rank、奇数词表、共享权重、标准导出与恢复；另测默认AdamW与非零attention dropout的固定拓扑逐bit续跑。Llama极小梯度的诊断保留在测试输出：某gate权重dense梯度`1.496057e-9`、TP梯度`1.541724e-9`，默认Adam可把此舍入差放大为`6.88e-6`参数差；以实际TP梯度驱动dense AdamW，更新误差为0。不能仅放宽参数容差来掩盖梯度/归约错误。

对应测试：`tests/unit/test_training_causal_provider.py`、`tests/distributed/test_training_causal_parallel.py`、`test_training_causal_pipeline.py`、`test_training_causal_recipe.py`。PP矩阵覆盖完整Llama/Qwen3的PP2×DP2及TP2×PP2，ZeRO0/3、tied/untied、不同microbatch长度、逐参数梯度、evaluate、checkpoint及标准导出；不是仅线性层的通信测试。TP2×PP2时DP=1，不把此测试称为同时验证非平凡DP分片。

本provider仍从完整权重构造，存在初始化峰值；导出仍是全字典host路径。CP/SP/EP/GTP/FP8、TP语言KD、VPP模型分区和任意跨模态模型组合未开放，错误显式报告。0样本batch先拒绝，0有效目标用全mask样本表示；空batch需模型reshape/attention另行实现。没有GPU或真正多机性能证据。

模型公式参考[Transformers decoder锁定源码](https://github.com/huggingface/transformers/tree/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models)，训练布局复用本文件已列的Megatron/PyTorch通信依据。没有调用官方训练engine替代本地实现。

```python
from aster.training import PipelineStage, PipelineObjective, PipelineLossSpec

stage = PipelineStage(local_module, context.pp, schedule="1f1b",
                      parameter_names=local_to_global_parameter_names)
loss = PipelineObjective(criterion, specs=(PipelineLossSpec("loss", "elements"),))
trainer = Trainer(stage, loss, parallel=context, accumulation_steps=4)
```

criterion 只在逻辑尾 stage 执行。预声明 specs 避免每个 microbatch 都全流水线同步元数据；整个窗口末尾同步真实统计。常量指标必须声明 `differentiable=False`，不能让某 stage 不执行反向而其他 stage 等待梯度。无 specs 的 legacy GPipe 保留逐批元数据同步，只是可用对照，不是高吞吐路径。

`VirtualPipelineStage([chunk0, chunk1, ...], context.pp, parameter_names=...)` 将逻辑 stage `v` 放在 `v % PP` 的物理 rank。当前每 rank 相同 chunk 数，按标准 warmup/steady/cooldown 的依赖序执行；所有 chunk 共享本 role 的一次 optimizer 更新。PP 导出必须明确全局 FQN 映射，包括 buffer，不猜名称。

失败节点退出/通信错误依赖底层进程组 timeout，未实现跨 rank 自动重建。模型内部 collective 发生异常时不能保证所有 peer 都立即收到 Python 异常；launcher 应统一终止剩余进程并从最后快照恢复。

## 内存与优化

- `precision="fp32"|"bf16"|"fp16"`：FP16 要求 CUDA，动态 loss scale；overflow 时所有 rank 不更新 optimizer/scheduler/EMA。FP8 不是全模型 autocast 字符串：provider 显式使用 `FP8Linear`，其他算子保持高精度，不能仅改 flag 声称全架构 FP8 化。
- `FP8Linear(recipe=FP8Recipe(...), implementation="reference"|"scaled_mm")`：E4M3FN 输入/权重、E5M2 反向，真实 1-byte 张量和 current/delayed amax history，可选 power-of-two scale。reference 的 GEMM 是 FP32 解量化数学路径，不是加速。`scaled_mm` 真正调用 torch CUDA FP8 primitive，要求维度整除16，无可用 CUDA 时失败、不回退。CPU 验证前反向量化公式、历史、checkpoint；GPU测试显式 skipped。history 时钟是每次 quantizer 调用，不是官方所有聚合 recipe；eval 不改变历史，stateful FP8 模块不能放入当前纯函数 ZeRO3/重复重算边界。
- `zero_stage=3, sharded_initializer=callback` 可从全 meta 模型直接生成本 rank shard。回调签名 `(logical_name, tp_local_shape, dtype, flat_offset, valid_count, storage_device)`，只返回局部有效元素；padding 由引擎做，meta buffer 必须由 provider 初始化。与先生成每 rank 完整大模型再切分不同。分布式 seed/官方初始化统计由 provider 确定，不臆造通用初始化规则。
- ZeRO3可读取叶子参数的 `dtype/device/shape` 只读元数据（例如UNet/视觉patch projection的dtype对齐），但元数据不是Tensor；外部functional调用直接使用该参数仍明确拒绝，属性读取不会隐式通信。BERT式纯 `nn.Module` 容器的own参数若完全是后代叶子同对象别名，可保留逻辑双FQN并共享单一shard/optimizer；自定义有forward的own+children模块、裸ParameterList/Dict读取仍不受支持。标准导出恢复完整别名，native checkpoint校验别名布局。
- `Embedding/EmbeddingBag(max_norm=...)` 的forward会原地投影权重行，不能当纯函数重算。所有分布式角色，以及ZeRO1/2/3或optimizer offload训练角色均在任何参数变更前对称拒绝这种隐式forward写回；单rank ZeRO0无offload保留原生语义。分布式持久投影使用下节显式事件，不会自动修改用户模块的max_norm配置。
- `zero_stage=3, offload_parameters="cpu"`：参数 shard 常驻 CPU，在所需单元 staging 到计算设备 gather，反向 RS 后回 CPU；参数 owner 的优化器也随其在 CPU 运行。CUDA PCIe/显存收益未测。与 meta 初始化一起才避免开始时先把完整模型移到 GPU 的峰值；不接受 NVMe parameter paging 标志。
- `offload_optimizer="cpu"`：真实 CPU master weights 和动量，CUDA 时 pinned staging；顺序传输，不声称已有 H2D/D2H compute overlap。
- `offload_optimizer="nvme", offload_directory=...`：磁盘保存 optimizer states，逐参数加载、更新、原子写回、从 optimizer 内存驱逐。CPU master weights 仍常驻，峰值活动状态为最大参数状态大小。目录可在 NVMe 上，但不会把普通磁盘测成 NVMe 带宽。没有异步 tile prefetch/writeback，也不是参数 offload。
- `activation_offload="cpu"`：autograd saved-tensor pack/unpack 到 CPU；其中可包括保存的参数，不伪称仅 activation。GPU 显存收益须在 GPU 测量。
- `checkpoint_activation(function, *args, **kwargs)`：provider 显式重算边界，non-reentrant 支持多目标 `autograd.grad`，重放 RNG；函数不得带环境/数据游标/BatchNorm 更新等不可重放副作用。
- `communication_overlap=True, bucket_bytes=...`：当前 ZeRO0，按 group/dtype/device 构建异步 DP buckets，可与下一 microbatch 计算重叠；末尾 wait 后再裁剪/更新。不是 backward-hook 级 overlap，不接受 ZeRO1–3 静默回退。
- 自主 `Muon`：BF16 Newton–Schulz、动量/Nesterov、解耦衰减、两种形状 LR 校正；与已安装 PyTorch 2.11 Muon 参数更新逐步对照。当前完整二维矩阵和 DP；TP/ZeRO flatten shard 不等价于全矩阵正交化，明确拒绝。普通向量参数不可偷偷换另一个算法。

梯度缓冲当前使用 FP32（即使模型张量是 FP64），因此 FP64 optimizer 更新不是本 Trainer 的精确双精度承诺；ring attention 自身的 FP64 数学测试是独立范围。所有目标各有一组梯度 buffers，这是支持不同分母的显式成本。

### 显式持久embedding投影

```python
# 构建时明确使用nn.Embedding(..., max_norm=None)，不在ZeRO重算里隐藏写回。
trainer.register_embedding_projection("model", "task_embedding", max_norm=1., norm_type=2.)
# 所有rank在phase边界调用；本rank没有访问行时传空int64 Tensor。
trainer.project_embedding("model", "task_embedding", task_indices)
result = trainer.step(microbatches)
```

注册和每次调用都会对称验证role/path/policy/event counter。先合并所有rank访问行的union，只对这些行中范数超限者执行 `w *= max_norm / (norm + 1e-7)`；该公式对应 [PyTorch 2.11 embedding_renorm_cpu_](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/Embedding.cpp)。参数写回不经过optimizer.step，AdamW动量/方差/步数不变。分片、CPU/disk master和计算副本保持一致；低精度compute未访问行不会覆盖FP32 master尾数。范数使用FP32累积（FP64权重用FP64），不承诺与所有低精度原生kernel逐bit一致。

已验证DP2 × ZeRO0–3 × optimizer none/cpu/disk，含ZeRO3 CPU参数storage、tied权重、非均匀batch、空rank、只访问行变化、下一次AdamW与dense对照、精确checkpoint恢复。当前实现会gather整张逻辑embedding表到host并检查DP访问行字节一致，常驻参数仍可为shard；这是持久存储/数值契约，不是稀疏行通信加速。TP词表/列切分、PP/CP/GTP和自定义参数通信域明确拒绝。

policy属于native checkpoint身份，调用计数通过同一 `register_state` 机制保存。预检失败不改值/计数；若写回阶段部分失败，所有rank被标记failed，必须恢复完整checkpoint，不宣称跨设备事务回滚。投影不自动更新EMA/target：方法决定何时投影、何时target更新，并保存自己的阶段状态。一个累积窗口应在任何相关forward之前投影该窗口全部会访问的行；含tied输出head时尤其不能在保存计算图之后再改权重。部署/planner也须保持同一显式policy，不能假设optimizer更新后所有未访问行始终满足范数限制。带projection状态时portable拓扑迁移仍拒绝静默丢弃事件记录。

`add_role` 成功后会给该role根模型和子模块附加普通Python标记 `_aster_training_owned=True`。规划器等组件据此拒绝直接原地修改训练对象，并要求调用者显式传入由engine执行的投影回调（所有rank同序进入，不可仅leader调用）。标记不进入state_dict/权重或配置hash；独立build+load导出制品不会携带它，deepcopy训练对象则保留标记，避免误绕过所有权。标记不是权限沙箱，也不会自动触发collective。

## 检查点、迁移与制品

`save_checkpoint` 是所有 rank 必须进入的 collective；每 rank 先写唯一 payload、fsync、哈希，再由 rank0 原子提交 JSON manifest。旧 payload 保留，避免覆盖中途破坏上一快照，清理由单独显式策略负责。恢复默认 `weights_only=True`，核验路径、链接、大小、SHA256 和身份配置，不在失败后静默启用不安全 pickle。

保存内容含各 role 的参数/optimizer/EMA/scheduler/update 数、全局 step、loss scale、Python/NumPy/torch CPU/CUDA RNG，以及 `register_state(name, object)` 挂入的数据/replay/环境/方法阶段。对象必须提供 `state_dict/load_state_dict`，其 codec 完整性由方法或数据实现负责。固定拓扑下通过下一次更新精确一致测试；源代码公式变化仍需应用的代码版本/制品身份约束，不能仅凭Python callable自动检测。

`last_successful_update(role='model')`返回最后成功更新的独立JSON副本：`role`、
`role_updates`、`phase`、`objective_configuration={class,codec,configuration}`。
记录取该次phase实际执行前冻结的声明，因此`phase(objective=override)`不会被
导出时的默认目标冒替。只有真实updated且optimizer/scheduler/EMA与模式恢复完成、
WORLD末端错误检查通过后才提交；overflow/无有效梯度不覆盖旧记录，部分失败封锁
读取/导出并要求完整恢复。末端集体检查不能救活已经退出的进程，仍须通信超时。

native与portable都保存此记录；恢复前核验结构、有限JSON、role更新时钟和WORLD
一致性，再写模型/optimizer。旧快照缺失时保留None，不按当前配置补造历史；
无codec函数更新记录的目标字段也是None。此接口仅证明最后一次声明的成功更新，
不证明所有历史步骤、训练数据、外部teacher/encoder被任意代码修改的情况或宿主
可信性。公开制品若要求可核验训练目标，应使用core的严格消费验证；缺失记录须
拒绝该证明声明，而非禁止使用旧权重本身。

具有顶层 `config.to_dict()` 的模型还会把该有限JSON配置纳入每个role的native恢复身份；同shape的dropout、分布范围、architecture标签变化也拒绝加载。默认objective主动提供 `config_dict()`（优先）或 `to_dict()` 时，同样保存其class/codec/有限JSON树；拒绝NaN/Inf、Tensor、任意对象和非字符串键，不用str或 `__dict__` 猜语义。初始化前对称检查全rank相同配置，保存/恢复重新读取当前声明；GR00T Beta浓度、KD温度变化即使不影响shape也拒载。没有config的普通 `nn.Module`、无codec函数继续可用，但无参数超参、闭包捕获状态和phase临时objective须Method显式 `register_state`；teacher权重必须另登记冻结role，不能以配置字典冒充权重快照。此检查是native同拓扑续跑身份，不把portable拓扑迁移泛化成任意模型公式互换。

`save_portable_checkpoint` / `load_portable_checkpoint(..., seed=...)` 合并逻辑参数和 Adam/AdamW/RAdam/SGD 状态，再重新按 TP/ZeRO 切分，含 EMA/scheduler。已验证 TP2×DP2 ZeRO3 → 单进程 dense 下一次更新一致。它是显式 optimizer 拓扑迁移，**不是**原 rank RNG/游标的精确续跑。已经注册 data/replay/env 状态时拒绝 portable 保存/恢复，避免静默丢失。完整字典 gather 会使用 host RAM，磁盘 offload 的完整 checkpoint 同样有全状态 host 峰值，尚未实现大模型流式低内存迁移。

`export_state_dict(role=..., ema=False, only_rank_zero=True)` 所有 rank 进入，返回标准全局逻辑名字的 CPU tensors，供主线 ArtifactStore 发布。遵循 `state_dict` 的 persistent 规则：RoPE等可重建临时buffer不会混入模型制品；portable运行时迁移可以单独携带它们。不会把 runtime 原生 `shards.0` 名称冒充官方完整模型权重。GTP 导出还区分 remat shard 与 DP 真副本 owner，避免同一 FQN 被重复发布。

## 启动与多机部署

`python -m aster.training.launch` 默认只打印命令；显式 `--execute` 才启动。`--launcher torchrun` 保留官方进程启动器，`--launcher native` 是固定 rank 的自主进程启动器；二者使用相同 `RANK/WORLD_SIZE/LOCAL_RANK/LOCAL_WORLD_SIZE/MASTER_ADDR/MASTER_PORT` 契约。无 shell 拼接、自动安装、SSH 或隐式失败重启。多机在各节点分别指定相同 nnodes/master 和不同 node_rank；同构本地进程数，不允许多机 master 为 loopback。

```text
python -m aster.training.launch --launcher native --nproc-per-node 2 --execute tests/distributed/launch_worker.py
```

这是随仓库的通信 smoke，不是模型质量评测。实际训练入口使用 `with distributed_session(ParallelConfig(...)) as context:`，再创建模型/Trainer。初始化前检查 rank 数值、LOCAL_RANK 映射、并行网格、NCCL/CUDA 和卡数；timeout 同时进入 rendezvous/进程组配置，错误保留 rank/master 线索。生命周期只销毁自己创建的组。native launcher 并发收集日志，任一 worker 失败终止其余进程，不继续部分更新。

当前 Windows `torch==2.11.0+cpu` 的官方 torchrun agent 和 env:// agent-client 都直接构造默认 libuv TCPStore，单设 `USE_LIBUV=0` 不足。现可显式选择 `--launcher torchrun --store-backend legacy_tcp`：专用入口注册固定成员 rendezvous，agent 与 worker 都显式传 `use_libuv=False`，真实官方 torchrun 仍负责进程管理。不 monkey-patch 全局 Torch，不暗中换成 native launcher，失败不自动重启。本机真实双进程已通过，原 torchrun 跳过项已改为执行此兼容路径；Linux 默认后端与真实多机 NCCL 仍需对应环境验收。

```text
python -m aster.training.launch --launcher torchrun --store-backend legacy_tcp --nproc-per-node 2 --execute tests/distributed/launch_worker.py
```

冻结计算叶子的 ZeRO3 gather/输入梯度/无参数梯度现已实现，可用于 [DSA 两阶段训练](DSA_TRAINING.md)。叶子内混合冻结仍拒绝；阶段切换必须重建 optimizer 所有权，而非运行中随意修改 `requires_grad`。

## 用户可执行的 collective 训练配方

`aster distributed-train train.json --kind language|tensor --output ... --store ...` 接入自主Trainer；它不在每个rank执行一份Workflow。训练配置与单进程 `language_fit`/`tensor_fit` 相同，`training`可增加以下设置：

```json
{
  "steps": 1000,
  "batch_size": 4,
  "accumulation_steps": 4,
  "device": "cuda",
  "precision": "bf16",
  "zero_stage": 2,
  "checkpoint_every": 100,
  "replica_tail": "drop"
}
```

这是嵌入完整模型/数据配方的 `training` 对象，不是单独完整配置。CPU验证使用 `device="cpu", precision="fp32"`。`batch_size` 是每DP副本的microbatch大小；全局batch为 `batch_size × accumulation_steps × DP`。总步数只计成功optimizer更新；连续overflow/零有效目标超过 `max_consecutive_skips`（默认16）显式失败，不无限等待。

在安装支持torchrun的环境，可以直接用官方启动器：

```text
python -m torch.distributed.run --standalone --nproc-per-node=2 -m aster distributed-train train.json --kind language --output runs/train --store artifacts --backend nccl
```

本机已验证的Windows/native入口（从仓库根目录运行，CPU配置）：

```text
python -m aster.training.launch --launcher native --nproc-per-node 2 --backend gloo --execute src/aster/training/recipe_worker.py distributed-train train.json --kind language --output runs/train --store artifacts
```

多机在每个节点加 `--nnodes N --node-rank I --master-addr HOST --master-port PORT`，必须相同代码/配置与共享可见的输出/checkpoint目录；入口在训练前进行共享文件可见性核验。CUDA配置用 `device="cuda"` 绑定当前 `LOCAL_RANK`，不允许每个worker硬编码到同一 `cuda:0`。不自动登录远端机器、不启动SSH、不下载数据/模型。

语言配方可显式设置顶层`"training_provider": "native_tp"`，CLI增加`--tensor-parallel 2`，剩余进程作为DP。完整PP配方使用`"training_provider": "native_pipeline"`及`--pipeline-parallel 2`，可选顶层`"pipeline_schedule": "1f1b"`；必须至少每stage一层，PP2允许`tie_word_embeddings=true`，更多stage目前要求false。例：4进程加`--tensor-parallel 2 --pipeline-parallel 2`得到TP2×PP2×DP1。TP×PP不是额外数据副本，全局batch公式始终只乘DP。tensor配方仍只接受DP/ZeRO；语言provider的成功不替其他模态证明模型并行。

工程约束与恢复语义：

- 同一全局seed初始化所有模型；随后Python/NumPy/torch训练随机流使用seed+rank。checkpoint恢复覆盖各rank RNG，加载后不再重新seed。
- 采样先全局shuffle再等长stride切分。默认每epoch明确丢弃至多DP−1条记录，尾部随epoch重新shuffle；`replica_tail="error"` 可要求整除，绝不用重复样本填充，也不让短rank抢先进入新epoch。数据少于DP明确失败。
- 每rank保存自己的sampler、RNG、训练状态；checkpoint collective写payload，leader最后提交manifest。方法目标、预处理、数据fingerprint、batch、precision等进入恢复身份；可以延长steps/改变checkpoint频率，不可悄悄换损失或数据。
- 只有leader持目录锁、写history与发布制品；全部rank参加逻辑权重合并，再由leader创建普通CPU模型导出，避免保存ZeRO空参数壳。所有rank收到同一个StageResult；CLI只由leader打印一份JSON。
- 模型制品与单进程接口相同，继续接评估/蒸馏/量化/质量gate。`ema_decay` 的EMA保存在checkpoint；当前配方默认发布在线权重，不假装自动选择最佳EMA模型。
- 失败运行不隐式重试。用新output目录，并在配置中显式 `resume` 指向最后完整 `checkpoint-N`/`checkpoint-final` manifest。同一配置/代码的已完成输出可验证制品后只读重入。拓扑变化不是精确续跑，需单独数据状态迁移，当前配方拒绝。
- API可直接 `fit_language(..., parallel=context)` / `fit_tensors(..., parallel=context)` 参与collective。需要锁和stage manifest时使用 `run_distributed_recipe`。普通 `train/run/evaluate` 在多rank环境明确拒绝，防止多份Workflow争写目录。
- 默认dense及tensor配方只自动编排DP/ZeRO；上文三个确切语言模型可显式选择原生TP/PP provider。其他架构或CP/SP/EP仍需调用方提供实际模型分片、batch布局与对应objective，使用下层 `Trainer`/provider接口；不会把开启一个flag说成任意架构已自动模型并行。底层数学/通信验证与某架构生产配方验收是不同证据。

## 内部参数所有权与公开权重名称

为满足ZeRO3纯叶子物化，模型可将原来父模块上的参数移入独立计算单元。
例如GatedDeltaNet在内部使用`decay_gate.A_log/dt_bias`，公开权重仍为`A_log/dt_bias`。
provider必须显式提供局部`_aster_parameter_key_map`以及匹配的state_dict导入/导出
codec；运行时不猜名称、不宽泛移除层级。递归导出校验源key确实存在且为parameter、
目标名称无碰撞；共享参数可保留多个不同公开别名，但只拥有一个optimizer storage。

逻辑名称与物理EMA名称分开：dense EMA按公开state_dict名称保存，ZeRO3 EMA仍按
内部`decay_gate.shards.N`保存。原生checkpoint身份同时记录这些显式映射，避免
内部shard形状相同但公开参数语义改变后错误续跑。独立测试覆盖嵌套前缀、target
Polyak、EMA、全ZeRO、dense↔ZeRO3 portable/optimizer迁移、非法映射与别名所有权。
这不是自动支持任意裸ParameterList/动态容器；其计算仍必须经过明确的叶子forward。

## 测试矩阵与硬件验收

源目录中的独立验收文件：

- `tests/unit/test_training.py`：多分母、无效/空目标、未用参数、角色/target/buffer、精确恢复、EMA、各 ZeRO 单 rank、CPU/disk offload、Muon 对官方、activation 重算 RNG。
- `tests/distributed/test_training_distributed.py`：2/4 真 Gloo 进程，DP、TP×DP、ZeRO0–3、空 rank、多目标、overflow、clip、恢复、target 与 async buckets。
- `tests/distributed/test_training_layouts.py`：SP/CP ring/EP/GPipe/1F1B 各自组合 ZeRO0–3，dense 一步和多步更新对照。
- `tests/distributed/test_training_virtual_pipeline.py`：4 进程 interleaved PP/DP，多目标、非均匀计数、导出、独立 evaluate。
- `tests/distributed/test_training_reshard.py`：TP2×DP2 ZeRO3 → dense 的 optimizer/EMA 迁移。
- `tests/unit/test_training_fp8.py`、`test_training_sharing.py`：量化公式/低精度状态/恢复，tied/meta 与 JEPA 子树目标。
- `tests/distributed/test_training_meta.py`、`test_training_gtp.py`：真2/4进程 meta/tied/CPU shard、独立 GTP remat 域。
- `tests/unit/test_training_launch.py`、`tests/distributed/test_training_torchrun.py`：启动协议、shell=False/显式执行、真实 TCPStore 与 native/官方 torchrun 真双进程；Windows 的 torchrun 显式选择 legacy_tcp，不再跳过。
- `tests/unit/test_training_recipes.py`、`tests/distributed/test_training_recipe_distributed.py`：等长无重复DP采样、persistent导出规则、双进程语言/flow训练、ZeRO0/3、全局有效token归一化、rank独立RNG、精确next-update恢复、单leader发布和native launcher→真实CLI→可加载制品。
- `tests/distributed/test_training_container_alias.py`：DP2+ZeRO3纯容器叶子别名、真实半量参数storage、单次optimizer更新、dense对照和精确恢复。
- `tests/unit/test_training_projection.py`、`tests/distributed/test_training_projection_distributed.py`：显式投影DP2全ZeRO/optimizer卸载矩阵、row union、真实owner/moments、低精度master尾数、异常写回/恢复；分布式objective配置不一致在模型修改前拒绝。
- `tests/unit/test_training_configuration.py`：原生同shape模型配置、真实GR00T/KD objective参数变化拒载，旧无config调用路径和有限JSON校验。
- `tests/unit/test_training_provenance.py`、`tests/distributed/test_training_provenance_distributed.py`：actual phase override、独立role时钟、执行前冻结声明、ZeRO0–3 native/portable恢复、legacy无记录、坏记录写入前拒绝、DP2单rank真实optimizer写入后异常与全体不提交。
- `tests/unit/test_training_adam.py`、`tests/distributed/test_training_adam_distributed.py`：真实optimizer工厂重绑、多参数组及coupled Adam、全ZeRO/卸载矩阵、原生恢复/portable迁移、工厂单rank异常传播。
- `tests/unit/test_training_parameter_codec.py`：显式公开参数key映射、dense/ZeRO EMA storage分离、target/迁移/恢复、映射拒绝和共享参数单一owner。
- `tests/unit/test_offline.py`、`tests/distributed/test_offline_distributed.py`：TD3/TD3+BC/IQL多角色DP2×ZeRO0–3、独立全批公式oracle、空rank、半轮失败、BF16与精确续跑；见OFFLINE_RL.md的支持边界。

Windows 测试为 FileStore 建独立 ASCII 临时目录；pytest 关闭 cache 并使用仓库外唯一 basetemp。CPU 小张量测试是数学、存储与通信契约证据，不是公开模型质量 benchmark。

GPU/多机验收仍必须记录：GPU 型号/显存、CUDA/NCCL/torch、拓扑网络、模型/数据 revision、precision、batch/序列、warmup、tokens/s、step P50/P95、MFU、peak allocated/reserved、host RSS、I/O 带宽、通信与计算 trace；对 dense baseline 做更新误差/overflow/重启曲线验证，再单独测质量指标。本机物理 GPU 为 RTX2060，但当前使用 CPU 版 torch，无可用 CUDA runtime；RTX2060 也不是 Hopper FP8 TensorCore 验收硬件。当前不声称 NCCL overlap、FP8 kernel、Flash fused ring 或官方吞吐等价。

## 官方依据与 single-owner 原则

官方代码是算法和对照依据，不作为默认训练 engine：

- [Megatron-Core 进程域与专家网格](https://github.com/NVIDIA/Megatron-LM/blob/f2f0f7bfd88fcb1243df55275988d6af52daea35/megatron/core/parallel_state.py)，审查锁定 commit `f2f0f7bfd88fcb1243df55275988d6af52daea35`。
- [Megatron CP 算法与通信方式](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)，动态文档只作说明，版本审计以锁定源码为准。
- [DeepSpeed ZeRO/offload 配置](https://github.com/deepspeedai/DeepSpeed/blob/87d9ecd8e0a4fd7778a58ac0f69cc85951f78ea0/deepspeed/runtime/zero/offload_config.py)，锁定 `87d9ecd8e0a4fd7778a58ac0f69cc85951f78ea0`，0.19.6 审查基线。
- [DeepSpeed Pipeline Engine 限制](https://github.com/deepspeedai/DeepSpeed/blob/87d9ecd8e0a4fd7778a58ac0f69cc85951f78ea0/deepspeed/runtime/pipe/engine.py)：该 engine 对 ZeRO2/3 的限制属于其实现边界，不能误认为 PP+ZeRO2/3 数学不成立。Aster 自主路径另有真实进程测试，未调用该 engine。
- [PyTorch pipeline schedules](https://github.com/pytorch/pytorch/blob/e4d9b6187e6ef2635cc2b648fbb409d25d6a9d9a/torch/distributed/pipelining/schedules.py)、[分布式 checkpoint](https://github.com/pytorch/pytorch/blob/e4d9b6187e6ef2635cc2b648fbb409d25d6a9d9a/torch/distributed/checkpoint/state_dict.py)，设计审查 pin `e4d9b6187e6ef2635cc2b648fbb409d25d6a9d9a`。本机实际数值对照环境是 `torch==2.11.0+cpu`；interleaved 与 Muon 还直接审查了该安装版官方 Python 源码，不能把设计 pin 和安装版混成同一构建。
- [Muon 作者实现](https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py)，该 commit 也由 PyTorch 官方 `_muon.py` 引用；本地 LR/NS 口径与安装版逐步对照。
- [Transformer Engine 2.18 FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)（2026-08-30 实际打开）：HYBRID/current/delayed recipe依据。Aster 不调用 TE 训练模块，也没有宣称其所有聚合/微缩放模式等价。
- [官方 GTP guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/generalized_tensor_parallel.html) 与 [实际源码](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/generalized_tensor_parallelism.py)（2026-08-30 检查动态 main）：官方仍标 Experimental，要求 TE≥2.19；这不是稳定API。Aster 的 `rematerialize_weights` 只覆盖独立 remat 轴的同步数学/存储契约，不宣称具有官方预取链、量化通信、CUDA ticket cache 或相同权重 ABI。

许可来源已逐一打开核验：Megatron 原生文件默认 [BSD-3-Clause 风格许可及仓库内多来源许可](https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE)，不能把整个库都写成 Apache；[PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE)、[DeepSpeed Apache-2.0](https://github.com/deepspeedai/DeepSpeed/blob/master/LICENSE)、[TransformerEngine LICENSE](https://github.com/NVIDIA/TransformerEngine/blob/main/LICENSE)、[Muon MIT](https://github.com/KellerJordan/Muon/blob/master/LICENSE)。这里为自主算法/协议实现，没有整包复制上游；未来引入具体 kernel 源文件仍须逐文件保留其 copyright/license/NOTICE，不以仓库总标签替代文件许可。

Megatron 模型并行与 DeepSpeed ZeRO 的思想可组合，但同一个参数不能同时由两个 engine 所有。Aster 中更新 owner 只有自己的 Trainer/ShardOptimizer；不让 DS engine 和 Megatron optimizer 同时缩放、裁剪、step 或保存同一份状态。若未来加官方 runtime adapter，应是互斥 backend，独立声明能力矩阵，而非叠套两个 engine。

未完成项保留为明确 backlog：NCCL VPP transport、zero-bubble、任意比例EP/ETP/attention-TP折叠及EGTP、GTP async ticket/prefetch/量化AG、NVMe parameter paging、有状态/裸参数容器 ZeRO3 单元、FP8 CUDA 硬件验收及更多 fused kernels、ZeRO bucket overlap、异步多 tile I/O、流式 checkpoint 与跨拓扑 rank-data/replay/RNG 转换；不作为本轮已有功能。

## 完整Mixtral EP×EDP训练包

`parallelize_mixtral(model, context)`只接受本仓库的完整`MixtralForCausalLM`，不把DeepSeek/混合注意力模型名称映射成同一种MoE。`ExpertParallelCrossEntropyObjective`和普通Trainer共用optimizer、checkpoint、全局计数、导出与制品发布链路。

```python
context = ParallelContext(ParallelConfig(data_parallel=4, expert_parallel=2))
model = parallelize_mixtral(build_model(config), context)
objective = ExpertParallelCrossEntropyObjective(context, router_aux_coefficient=0.02)
engine = Trainer(model, objective, parallel=context, zero_stage=3)
```

此例只有4个进程，EP不会再乘一次WORLD。具体所有权为：

| 内容 | 通信/副本域 | 实际存储 |
|---|---|---|
| attention、norm、embedding、head、router | DP `(0,1,2,3)` | ZeRO0复制；ZeRO1/2分optimizer/梯度；ZeRO3常驻参数shard |
| 不同expert之间token分发/返回 | EP `(0,1)`、`(2,3)` | 每rank仅持`E/EP`个专家，不先计算全部专家 |
| 同一批global expert的副本 | EDP `(0,2)`、`(1,3)` | 专家optimizer/梯度/ZeRO3仅在对应EDP中分片 |
| CE样本与有效token计数 | DP `(0,1,2,3)` | 分母仍是全部数据，不误除EDP大小 |

provider保留Mixtral原生GQA、滑动窗口、RoPE、SwiGLU和可选tied embedding/head。Router以真实logits-only Linear叶子拦截ZeRO3物化，top-k在其外执行；公开key仍为`gate.weight`。PackedExperts使用`[local_E,2I,H]`和`[local_E,H,I]`，运行时保存显式EP切片轴，导出/portable先还原EDP shard再按EP拼回标准`[E,...]`。不得直接将本地分片`save_pretrained`为完整模型。

ZeRO轨迹现在按每个真实DP/EDP group分别核对，不能要求不同专家分区有相同全局FQN。轨迹检查发生在forward之后，仍不能代替provider事前检查；错误的collective顺序或进程硬退出最终依赖进程组超时。

路由策略与边界：

- dropless softmax top-k，已选权重重新归一化；变长all-to-all及其反向均为真实通信。
- 没收到token的专家仍执行0-row GEMM路径，保持EDP物化/反向次序；不能因本地无token跳过整个expert参数owner。
- `router_aux_coefficient=0`为纯CE；非零时为明确的Megatron **sequence-level** Switch平衡目标：每序列`E/(K*T²)`，各层求和，然后用独立有效sequence分母。不是HF跨层/整batch Mixtral aux，也不是global-batch频率目标；此逐序列目标在不同microbatch切分下保持可加。
- attention padding不进入router统计；`loss_mask`只控制CE监督。当前padding token仍会计算/分发，尚未做token compaction。
- 完整dense GQA的`B=0`会有歧义reshape，provider在任何模型通信前对称拒绝；空监督rank必须显式给一行全mask占位，其CE/aux分母为0。数据配方默认等长DP切分并显式丢弃尾部，不私自构造重复数据。
- 此旧`parallelize_mixtral`入口保持attention TP=ETP=PP=CP=GTP=1，`num_experts % EP == 0`且`DP % EP == 0`；新ETP入口与后续独立闭环见下节，不悄悄改变旧provider的布局。
- 初始化仍从完整dense模型复制；导出聚合完整CPU字典。不是流式低峰值大模型加载，也没有声明GroupedGEMM、DeepEP、FP8或通信重叠速度等价。

实际用户入口使用语言配方`training_provider: native_moe`，可选`router_aux_coefficient`，再执行`distributed-train ... --expert-parallel 2`。WORLD由原生launcher或torchrun提供；完整4进程测试包含两次独立启动和resume，不是在同一Python对象上调用两次step。

验证文件：

- `tests/unit/test_training_moe_provider.py`：完整模型ZeRO0–3随机jitter恢复，另测ZeRO0/3 BF16；标准tied导出、seq_aux切分等价、非法B=0前置拒绝。
- `tests/distributed/test_training_moe_parallel.py`：真实4进程EP2×EDP2，逐参数梯度/global norm/SGD momentum更新与完整dense对照；不等长batch、全mask rank、空expert；ZeRO3原权重numel=0与gather/release；native exact resume；portable迁回独立dense优化器的下一次更新；EP局部源克隆成完整dense冻结target；另有FP32/BF16真实AdamW、CPU optimizer/parameter offload和router RNG恢复。BF16 GEMM结果显式转回残差通信dtype后写入，避免`index_copy`源/目标dtype不一致。
- `tests/distributed/test_training_moe_recipe.py`：真实CLI的JSONL→4进程训练→分片checkpoint→新进程恢复→标准制品与history一致，每rank sampler为DP4。
- `tests/unit/test_training_moe_official.py`：完整ZeRO3模型梯度对照实际Transformers 5.16.1；显式`ASTER_RUN_REMOTE_MOE_ORACLE=1`才下载并执行未改动的官方Switch函数。固定MCore commit `f2f0f7bfd88fcb1243df55275988d6af52daea35`，`moe_utils.py` SHA256 `1b13f06e7bf0a08e9361f7c337b9c3de3be57153d372dd7f676f33aecd0a83dd`。

本包已有CPU/Gloo数学、通信、真实存储与工程生命周期证据。机器有物理RTX2060，但当前`torch==2.11.0+cpu`未提供CUDA；因此没有NCCL、多机网络、GPU显存/吞吐或公开数据集质量成绩。GPU验收仍须以实际模型测训练loss/梯度、峰值显存、tokens/s、dispatch imbalance、A2A与GEMM时长，并保存硬件/版本/全局batch。

官方设计依据：[独立EP/ETP/EDP网格](https://github.com/NVIDIA/Megatron-LM/blob/f2f0f7bfd88fcb1243df55275988d6af52daea35/megatron/core/parallel_state.py)、[A2A→ETP AG→GEMM→ETP RS→A2A顺序](https://github.com/NVIDIA/Megatron-LM/blob/f2f0f7bfd88fcb1243df55275988d6af52daea35/megatron/core/transformer/moe/token_dispatcher.py)、[专家与dense的梯度归一化区别](https://github.com/NVIDIA/Megatron-LM/blob/f2f0f7bfd88fcb1243df55275988d6af52daea35/megatron/core/distributed/distributed_data_parallel.py)、[Switch公式](https://github.com/NVIDIA/Megatron-LM/blob/f2f0f7bfd88fcb1243df55275988d6af52daea35/megatron/core/transformer/moe/moe_utils.py)。Megatron原生源按文件BSD-3-Clause等许可归属记录，本包自主重写，oracle只在显式测试时取固定源函数。

## 每phase声明与逐元素梯度裁剪

`phase()`在batch读取、preflight、参数冻结与任何模型gather之前，WORLD核对实际phase name、role、freeze_roles、显式objective配置和裁剪参数。目标构造后被修改或phase使用另一个目标，不能绕过构造时的检查。codec异常/NaN和单rank非法冻结要求均对称失败，尚未进入phase时不污染RNG/权重或把engine标为失败。旧式无codec函数仍可用，但闭包内部算法状态必须用Method的`register_state`保存，不能靠运行时猜`__dict__`。

`Trainer(..., max_grad_norm=None, max_grad_value=1.)`提供逐元素`[-1,1]`裁剪，适用于VMC MDN-RNN等明确配方。它发生在每项loss独立全局归一化、DP/EDP归约与finite检查之后，对实际optimizer参数shard执行；若两种裁剪同时开启，顺序固定为global norm→value。没有梯度的参数仍为None，不凭空引入weight decay。CPU/disk optimizer复制的正是裁剪后梯度，不用自造不兼容Adam子类。

`max_grad_value`在native/portable checkpoint与phase声明中校验。布局/裁剪身份改变会拒绝精确恢复，不自动把旧快照解释为新配置。`tests/unit/test_training_gradient_value.py`覆盖ZeRO0–3×none/CPU/disk真实Adam、恢复与portable下一步；`tests/distributed/test_training_gradient_value_distributed.py`用DP2不等样本、正负抵消梯度验证“先全局mean再clamp”，并拒绝不同rank阈值。`tests/distributed/test_training_phase_declaration.py`断言不同目标/codec/role/freeze在ZeRO3新增gather为0时拒绝，修正后仍能更新并精确恢复。

## 完整Mixtral EP×ETP×EDP与attention TP（2026-08-31增量）

`parallelize_mixtral_tensor(model, context)`和`ExpertTensorParallelCrossEntropyObjective`
是完整原生Mixtral训练provider，复用共享Trainer，不调用官方训练器。
`ParallelConfig.expert_tensor_parallel`新增独立ETP轴，**不再乘一次WORLD**：
同一组物理rank分别排成attention `[DP,TP]`与expert `[EDP,EP,ETP]`。
`context.tp`、`context.etp`、`context.ep`、`context.edp`始终是显式域，不互作别名。

| 已验证完整模型矩阵 | WORLD | attention DP | expert EDP | ZeRO |
|---|---:|---:|---:|---|
| TP1、EP2、ETP2 | 4 | 4 | 1 | 0/1/2/3 |
| TP2、EP2、ETP2 | 4 | 2 | 1 | 0/1/2/3 |
| TP2、EP2、ETP2 | 8 | 4 | 2 | 0/1/2/3 |

执行顺序为完整hidden→attention-TP token scatter→本地router→变长EP A2A→
变长ETP all-gather→局部SwiGLU专家→ETP reduce-scatter→反向EP A2A→
按top-k权重合并→attention-TP token gather。输入token每个只路由一次，不能把
TP复制的完整序列重复发送。ETP通信是基础torch真实collective；变长AG/RS使用
显式等宽padding，元数据通过object collective交换。空专家/空ETP组仍保留叶子
调用和伴随路径，不因本地token数为0跳过EDP物化。

专家实际持有`gate_up=[E/EP,2I/ETP,H]`及`down=[E/EP,H,I/ETP]`。
gate/up分别分片后保存在同一个局部权重中；`_aster_tp_stripes=2`使规范codec
重组为`[gate0,gate1,...,up0,up1,...]`，不能直接cat各rank的交错段。
显式EP轴、ETP轴/组、stripe数、EDP owner均纳入native恢复身份；portable逐轴
还原全局参数与Adam/SGD状态，再按目标布局切分。tied词表共享一个optimizer owner，
ceil-padded词表通过分布式softmax排除padding logits，导出裁去实现用的尾行。
共享分布式CE将FP16/BF16 logits提升到FP32做max/exp/sum/log，和标准autocast
CE一致；不改变FP32/FP64。单组及真实TP2均对照torch CE输出与局部梯度。

梯度规则有三类：dense参数按attention DP归约；expert局部权重按EDP归约；
只处理局部token的router额外在attention TP求和。CE分母仍是全局有效监督token，
不是EDP大小；逐sequence Switch aux另有有效sequence分母。post-attention RMSNorm
由scatter反向gather得到完整输入梯度，不能再额外乘TP。全局norm对独立EP/ETP
片只计一次，与完整dense模型逐参数梯度和裁剪后SGD更新对照。

用户入口仍为`training_provider: native_moe`，ETP非1时显式选择新provider，例如
4进程`distributed-train ... --tensor-parallel 2 --expert-parallel 2 --expert-tensor-parallel 2`。
sampler按attention DP2划分，因此batch_size=1、accumulation=2的真实global batch为4，
不能错误地按WORLD4再翻倍。随机router jitter作用于TP复制的完整hidden；配方
使用`seed+attention_TP_leader_rank`，DP独立而TP共享，恢复覆盖其原随机状态。
直接调用Trainer时，目标在forward前核对同TP输入和jitter RNG，不一致对称拒绝。

真实证据在`tests/distributed/test_training_moe_tensor.py`：4/8进程完整模型，
所有ZeRO stage的梯度/global norm/更新对照、实际局部numel、gather/release、
不等batch及全mask rank、empty expert/ETP、双stripe标准导出、dense target clone、
native exact resume、portable迁回独立dense后下一次SGD更新；另有真实AdamW、
FP32/BF16、CPU参数/master与磁盘optimizer-state offload、EMA精确恢复。
`tests/unit/test_training_moe_tensor_provider.py`单列codec与拒绝契约。
CLI测试文件为`tests/distributed/test_training_moe_tensor_recipe.py`，验收结果在本包
冻结记录中单列：新ETP+旧EP/TP/PP真实CLI专项5 passed /152.82s；其中新ETP含
3次独立启动，完整2步与1步checkpoint→恢复第2步的权重/history精确一致。

边界：当前PP=CP=GTP=1，attention TP必须为1或等于ETP，expert数整除EP、
intermediate宽度整除ETP、query heads整除TP。TP>1时`attention_dropout`必须为0：
独立head随机流与共享router随机流的双流tracker尚未实现，不能让两者共享同一
不正确的mask分布。没有任意attention-TP/ETP比例折叠、PP×MoE、DeepSeek/shared
expert、capacity dropping、token compaction、GroupedGEMM/DeepEP/FP8融合或通信
重叠性能证据。初始化仍有完整模型复制峰值，导出/portable仍需完整host字典。

来源沿用上述Megatron commit；`token_dispatcher.py`的固定SHA256为
`18f9b0ef848ab36006d2cee2ac94da1be16426bbcc1200a327bcaf0ddb97d7de`，
官方ETP AG/RS次序与本地伴随通信对照；完整dense公式另由实际Transformers
oracle验证，来源许可按文件保留。当前仅CPU/Gloo数学、真实存储和生命周期
验收，物理RTX2060未被CPU版torch启用，没有NCCL/多机链路/GPU吞吐或质量成绩。

最终交叉回归：独立`aster_etp_validation_20260831_04`副本，冻结时443个输入文件
逐文件源/副本SHA256一致；`test_training*.py`及offline/conservative/VMCstream/
consistency单元与真实多进程集合 **333 passed, 2 skipped /753.34s**。两个skip分别
是本Windows wheel缺libuv的torchrun agent路径（native launcher已真进程验证），
以及需要Hopper/更新CUDA硬件的FP8 scaled_mm验收。已启用固定MCore Switch与
VMC官方源oracle；测试结束后再次核对25个训练模块及6个入口/方法源码与该副本
相同。此计数不包含后续semantic runtime-buffer补丁，后者另做独立验收。

## 显式语义buffer与完整恢复/导出（2026-08-31）

`state_dict()`仍遵守PyTorch的persistent语义，不把RoPE频率或临时KV塞进官方
权重键。模型通过`_aster_semantic_buffers`逐owner声明真正影响前向的非持久
buffer；`training/runtime_state.py`消费该协议。比如频率经历BF16→FP32后已经
舍入，仅重新计算config不能得到原状态。native checkpoint现在另存每role的
`runtime_state`，identity绑定FQN/shape/dtype/layout/alias；所有role预检通过后
才写权重。portable也标识semantic条目，严格拒绝缺失/类型变化/坏值/别名冲突，
不依靠copy_默默转换频率精度。有效checkpoint仍可修复当前内存的NaN状态。

部署请全rank同序调用`engine.export_state_dict(role=...)`与
`engine.export_runtime_state(role=..., only_rank_zero=True)`；后者只gather显式
buffer并使用同一TP/PP全局FQN codec，副本值不一致时拒绝。leader独立构建模型、
加载标准权重，再`apply_runtime_state(model, state)`，最后`save_pretrained`。
该安装函数拒绝Trainer-owned/分片对象，保持buffer别名和保存的dtype。共享
recipe已经接入；EMA权重使用同role固定语义buffer，不将RoPE伪造为EMA shadow。
没有语义payload的旧checkpoint不能声称这一精确保证。动态推理cache不在此协议。

专项证据：`test_training_runtime_state.py`20项；fresh Llama的native/portable
next update、EMA、独立部署FP32/BF16舍入保存，以及坏payload写入前拒绝。
`test_training_runtime_state_distributed.py`3项分别为真实DP2×ZeRO0–3、TP2×ZeRO3、
PP2×ZeRO3，验证重新创建实例的恢复、全局标准导出、单rank坏状态对称失败。
加旧provenance/config/reshard交叉集合分别28 passed/10.03s与17 passed/59.19s。
既有ZeRO3叶子参数+buffer预检仍拒绝带running statistics的BatchNorm，避免反向
重算二次更新统计；这不代表任意有状态模块都已支持ZeRO3。

## 自适应全局梯度比率（2026-08-31）

`register_gradient_ratio(name, role='model', reference_term=..., target_term=...,
parameter='decoder.last.weight', eps=1e-4, min_ratio=0, max_ratio=1e4, multiplier=1)`
绑定明确参数FQN和两个独立目标。先在完整累积窗口求各自全局mean梯度，再计算
`clip(norm(reference)/(norm(target)+eps))*multiplier`，乘到target的外部weight；
不能平均microbatch/rank比率，也不能把不同分母的梯度先混起来。reference的外部
weight为0仍可被active策略用作probe；target weight为0是显式warmup，不伪造范数。
active目标缺少该FQN梯度会失败，不默默退成常数。真正零梯度则按eps公式处理。

本包支持pure DP×ZeRO0–3、CPU/disk optimizer offload及ZeRO0 bucket overlap。
DP/ZeRO1仅复制指定FQN的两组梯度做额外SUM，不污染待最终归约的term buffers；
ZeRO2/3已RS，只对排除padding的局部平方和做标量SUM。无额外forward或整模型
probe副本。额外通信/内存成本是该FQN两个梯度，而非免费。TP/PP/CP/EP/ETP/GTP
及非全DP loss domain明确拒绝，不将矩阵/目标分片组合假装已经验证。

策略配置进入native/portable身份，`last_gradient_ratio(name)`只返回成功更新
提交的动态记录（两范数、clipped ratio、外部/有效系数、role update clock）。
overflow、失败或跳步不覆盖记录；恢复先核公式/有限值/时钟与WORLD一致性再写入。
多phase方法仍由方法层封锁半轮checkpoint；引擎不把部分优化器写入冒充可回滚事务。

来源为CompVis/taming-transformers MIT，commit
`3ba01b241669f5ade541ce990f7650a3b8f65318`的
[vqperceptual.py](https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/losses/vqperceptual.py)，
文件SHA256 `b46889cabb89785dd82c9b1fcf07ad8f1d4a9daacf6b4f74d88992b7007e8b1a`。
opt-in测试抽取其未改写`calculate_adaptive_weight`函数直接执行对照。实现范数
用FP64聚合降低归约误差，允许与上游FP32舍入有小数值差，公式与detach语义相同。

新测试`test_training_gradient_ratio.py`15项及
`test_training_gradient_ratio_distributed.py`1项：16 passed/27.79s，已启用官方
AST oracle。双进程测试内实际跑ZeRO0–3全局不等计数/空有效rank/padding范数、
overlap、BF16+disk、fresh native下一步恢复；本地覆盖三个offload模式的全梯度/
裁剪/SGD更新及portable dense下一步。加runtime/provenance首轮42 passed/13.68s。
这是CPU/Gloo数学、真实通信/存储和恢复证据，无CUDA吞吐或生成质量晋级声明。

## Muon与辅助Adam的统一owner（2026-08-31）

新入口为`training.MuonWithAuxAdam`和`MuonFactory`；旧`optim.Muon`的既有profile
未被悄悄改变。一个角色只有一个optimizer，它的显式param_group分别使用Muon
或辅助Adam。嵌入/输出头不是第二个私有训练器，不会漏掉scheduler、溢出跳步、
EMA或checkpoint。`add_role(..., optimizer_factory=...)`可为其他角色独立配置。

| 显式profile | Muon动量与矩阵尺度 | 辅助Adam |
| --- | --- | --- |
| `keller` | EMA动量；Nesterov插值；更新乘`sqrt(max(1,rows/cols))` | bias-corrected二阶矩开方后加eps |
| `moonlight` | SGD式动量累积；`g+beta*m`；LR乘`0.2*sqrt(max(rows,cols))` | 原二阶矩开方后加eps，再整体bias correction |

二者不能混成一个“Muon默认”。每组保存profile、源commit/文件hash、lr、weight
decay、momentum/betas、epsilon、NS轮数及`missing_grad`。默认`skip`不给未使用
参数施加衰减/动量；复现Keller当前源码的补零行为须明确选择`missing_grad='zero'`。
磁盘逐参数eviction模式拒绝zero策略，防止重复更新当轮其他参数。权重衰减使用
原始LR，不使用Moonlight矩阵尺度调整后的LR。closure仅调用一次。

```python
from aster.training import Trainer, MuonFactory, parallelize_causal_lm

# 先在完整模型上明确选组，再建立TP/ZeRO存储。Embedding按真实模块类型排除，
# 输出头必须由调用者列出；tied embedding/head按同一个Parameter去重。
factory = MuonFactory.from_model(
    model, auxiliary_modules=('lm_head',), profile='moonlight',
    muon_options={'lr': 0.001, 'momentum': 0.95},
    auxiliary_options={'lr': 0.0003, 'betas': (0.9, 0.95), 'eps': 1e-8},
)
# 使用TP时：model = parallelize_causal_lm(model, context)
# 同时使用TensorParallelCrossEntropyObjective(context)，不能把局部词表当完整logits。
engine = Trainer(model, objective, parallel=context, zero_stage=3,
                 optimizer_factory=factory, precision='bf16')
engine.set_scheduler(lambda optimizer: scheduler_factory(optimizer))
```

手工精确选组可使用`MuonFactory([{'names': [...], 'use_muon': True,
'profile': 'keller', ...}, ...])`；所有trainable逻辑FQN必须恰好覆盖一次，未知键、
缺失参数、重复别名、错误矩阵几何均拒绝。每rank必须具有相同参数collective
遍历顺序和相同逐参数选项，不能仅比较无序dict后让Q/K矩阵通信互相错配。
`from_model`适合本包已验证的Llama/Qwen2/Qwen3；额外模型公开key codec或特殊
二维参数语义需显式命名组，不按名字包含`embed/head`猜测。

实际分片路径是：local momentum更新 → DP/ZeRO gather回TP局部矩阵 → TP gather
完整逻辑矩阵（去除词表padding）→ BF16五阶Newton–Schulz → 同一codec重新TP/DP
切分 → 本地参数更新。**不是flatten局部分片正交化**。常驻momentum仍是原来的
本地shard；完整矩阵是临时量。CPU/disk master保留显式布局，checkpoint不序列化
进程组对象，fresh factory按新布局重建。portable可将matrix momentum和辅助
moments迁回独立dense模型。标准导出保留官方参数键、裁去词表padding及tied权重。

完整NS在每个相关rank上重复执行，临时内存/通信不免费：每矩阵需要完整`M×N`
及`min(M,N)²` Gram和迭代临时量，复杂度`O(MN+min(M,N)²)`；当前不是分布式Gram、
whole-parameter轮转owner、重叠all-gather或Megatron GTP的高性能实现。初始化和
portable仍有完整模型/字典峰值。支持范围为TP×DP×ZeRO0–3；PP/CP/EP/ETP/GTP
明确拒绝。本地Keller可显式选择conv2d展平或batched3D，Moonlight锁定版本仅2D；
这不等于完整MoE训练已获得Muon支持。

来源均为MIT，实际未改写函数AST作为数值oracle执行，不调用官方训练引擎：

- [KellerJordan/Muon](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py)，
  commit `f98f1cacc0263b04290753e32be8d498c1efc806`，文件SHA256
  `2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`。
- [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight/blob/c2ad5b20c605086526a179d36901bfc41b52b44b/examples/toy_train.py)，
  commit `c2ad5b20c605086526a179d36901bfc41b52b44b`，文件SHA256
  `8df3ec6e2f2cd5af8aee59ffb48b2219f394da6f049d95d881c66a6d13d00874`。

Megatron固定版本`f2f0f7bfd88fcb1243df55275988d6af52daea35`的
`optimizer/emerging_optimizers.py`目前接入NVIDIA-NeMo/Emerging-Optimizers，具有
自己的layer-wise/TP/GTP算法和限制，不等于本包上述profile；Moonlight旧PR1428
已关闭但未合并，不能拿该PR宣称Megatron已原样实现那个toy公式。

证据文件：`test_training_muon.py`含官方AST五步逐位对照、完整tied Llama全部ZeRO、
CPU/disk、native/portable、明确几何与负例；`test_training_muon_distributed.py`
是真实DP2下24个stage/profile/precision/offload组合；
`test_training_muon_causal.py`是真实TP2DP1和TP2DP2各28个完整模型组合（Llama、
Qwen2、Qwen3，GQA复制KV/QKNorm/position/padding/不等microbatch/tied词表）。
验证全局梯度、常驻状态实际numel、ZeRO3释放、独立完整矩阵公式、fresh native/EMA
逐位恢复，以及六个portable→独立dense下一步。首次两张量拓扑分别1 passed/40.56s
和1 passed/57.10s；最终源码锁定后的交叉结果另记。

数值边界透明记录：TP补齐词表改变CPU `lerp_` SIMD尾部形状。实测同梯度第2步
辅助moment差1.86e-9、最终单元素差4.66e-10；测试另做同物理shape辅助更新逐位
oracle，跨shape完整权重严格绝对误差1e-8，并逐项核moment。ZeRO3 tied参数反传
累加还会产生约3e-8梯度差，不能把跨布局结果宣称全部逐位。相同拓扑/配置的native
恢复则仍逐位相等。CPU/Gloo证据不证明GPU/NCCL、多机性能或公开质量；FP16 CUDA
专用测试在当前CPU wheel下skip，BF16真实训练/溢出不提交/恢复为CPU实测。

闭包交叉记录：训练19个unit文件加Muon真实DP/TP、旧Adam、portable reshard、
semantic runtime buffer与gradient ratio多进程集合，**279 passed, 7 skipped /
230.65s**；53个训练源码和对应测试文件前后SHA256无变化。7个skip分别为4个
CUDA FP16用例、1个Hopper FP8用例，以及本次未启用的MCore/梯度比率远程源码
oracle（它们在此前各自包中已单独实测，本包不重记为通过）。本包两个Muon
官方源码oracle明确启用并通过。

末端通信安全小修单列验证：所有rank在role构造前先核对是否使用命名Muon工厂；
每个phase复用已有optimizer-owner WORLD预检，重新核对实际Muon参数顺序、
当前lr/profile和完整字段，避免动态修改后挂在NS矩阵gather。单rank工厂协议/
参数重排/学习率/必需字段不一致，真实DP2 ZeRO3在任何forward前统一拒绝，
权重和clock不动、不标记半轮失败。该小修+Muon单测/旧共享权重/receipt/完整
PP4rank回归 **54 passed, 4 skipped /77.47s**；最终TP2DP2三模型28组合与独立
dense迁移再次 **1 passed /51.95s**，5个相关runtime源文件前后SHA256无变化。
最后将optimizer映射/字段读取也纳入同一对称错误边界，单rank删除`params`
容器不能在WORLD预检外抛出异常；真实DP2全部ZeRO/坏字段与旧共享权重专项
**12 passed /25.25s**。这项仅扩错误捕获范围，不改变更新公式或通信布局。

## DeepSeek V3/V3.2纯DP训练的参数所有权修复

完整模型的MLA吸收路径原先在父attention直接读取`kv_b_proj.weight.view`，
与ZeRO3“仅叶子forward中物化完整权重”的契约冲突。现在`LatentKVProjection`
仍继承`nn.Linear`、仍保留同一`kv_b_proj.weight`，但通过明确的`query`和`value`
模式在拥有参数的叶子内消费完整矩阵；两次调用各自gather/release，反向对同一
分片累加梯度。没有解开guard、常驻完整权重，或把K/V拆成不同优化器参数。

`TopKRouter`的correction bias保留在无参数父模块，父模块负责离散top-k与
tuple输出；唯一可训练矩阵位于单Tensor输出的`RouterProjection`叶子。
显式`projection.weight -> weight`双向codec保持官方公开`gate.weight`及严格
加载语义，重复公开/内部名字仍拒绝。普通参数名遍历是内部所有权名，公式oracle
用`public_parameter_names`明确映射，不以宽松字符串替换忽略未知权重。共享
Mixtral/QwenNext/Qwen3.5/MTP的状态和梯度也须一起回归。旧物理结构的native
checkpoint不是此新布局的精确续跑文件；标准模型权重的公开键保持不变。

`tests/unit/test_training_deepseek.py`验证两个完整家族、各自dense与默认混合
dense/MoE层、ZeRO0–3：不等长microbatch的全局有效token CE、所有参数梯度和
global norm、SGD动量更新；另以真实AdamW验证FP32/BF16、随机attention dropout、
CPU optimizer/parameter offload、EMA、fresh native恢复后的下一步与RNG逐位一致。
独立模型通过strict标准权重加`export_runtime_state`恢复语义RoPE，而非把非持久
buffer混入官方state_dict。真实DP2包含不等样本数和空监督rank、全部上述模型/
ZeRO布局、ZeRO3真实参数释放/常驻分片numel、坏后微批零forward对称拒绝。
V3.2被明确加入已审计纯token前置校验集合，未放宽任意模型鸭子类型。

数学参考为[DeepSeek原作者MLA吸收实现](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)
以及[Transformers V3路由](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)
/[V3.2稀疏模型](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py)。
真实执行oracle是已安装Transformers5.16.1，绝不把动态main当作同一快照；两个
modeling文件SHA256分别为`da66249787ddac6ba2dd603d3d39d791011d2385751da68248ac9d03fad07fd2`
和`bbc83144985bb3669b102d611dda9261a97dfad29cc73228a36d2da1d09ddddc`；分布版
V3与V3.2应分别固定来源版本；本段列出V3.2文件摘要
（均未从版本号推断Git commit）。两个新增直接ZeRO3官方oracle明确配置官方
`experts_implementation='eager'`：tiny专家宽6不满足默认grouped_mm的16byte
stride，不修改官方计算源码或把基础公式对照称作融合grouped-GEMM验收。

边界：本包不是DeepSeek TP/EP/ETP/PP provider；独立indexer KL训练与router动态
load-correction更新也不是本包CE对照已验内容。专家选择仍离散，CE对indexer
保持`grad=None`，不凭空施加weight decay。没有GPU/NCCL、多机性能、FP8/kernel
或公开预训练质量证据；本机物理GPU存在，但当前PyTorch为CPU构建。

本包最后固定owner文件组合结果：**75 passed, 1 skipped /66.82s**，覆盖新增
36项DeepSeek训练测试（其中两项直接执行官方模型、1项启动真实DP2并遍历20个
family/布局/精度组合），旧decoder/sparse测试、实际Transformers同权重/全梯度/
缓存oracle、QwenNext/Qwen3.5/MTP、旧完整ZeRO3 Mixtral oracle。11个本包源码/
测试文件运行前后SHA256一致。唯一skip为未显式启用的旧远程Megatron辅助损失
源码oracle，不是把缺GPU测试计为通过。另行真实EP/ETP交叉 **4 passed /120.00s**：
EP2×EDP2、attention TP1/2×EP2×ETP2×EDP1、TP2×EP2×ETP2×EDP2（4/8进程），
全部ZeRO0–3完整Mixtral梯度/更新/恢复/导出。这验证共享router重构未破坏已有
provider，不扩大DeepSeek自身的并行矩阵，也不将前几次重跑结果重复累计。
