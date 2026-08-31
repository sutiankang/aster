# DeepSeek DSpark：训练、部署与评价闭环

这里的Spark是[DeepSeek DeepSpec / DSpark](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2)，不是Apache Spark，也不是给已有MTP或通用投机解码换名。核对提交固定为`005e03b81cec38b7da6399833d609ee89a2587f2`。原生运行不依赖DeepSpec/Transformers；对应版本的作者源码仅在显式开启的独立oracle测试中执行。

## 一个贯通的实现

```text
已绑定目标模型与token-ID语义
  → 目标回复/训练mask → 选中层和最终层特征 → 不可变缓存
  → DSparkMethod → 同一个Trainer、DP/ZeRO、EMA、checkpoint与数据游标
  → 校验实际成功更新和缓存父制品 → 独立部署草稿
  → 并行草稿backbone + 小型Markov头 + 置信截断
  → 目标验证/拒绝重采样/缓存回滚 → 成对质量和完整延迟报告
```

Qwen3和Gemma4是两个真实实现，不互用错误的归一化或head公式。Gemma4保留embedding缩放、global head维度、K=V选项、无scale的V归一化、四处RMSNorm、layer scalar、全局RoPE与基础logit softcap；Markov残差在softcap之后加入。该Gemma4草稿明确拒绝MoE和per-layer输入，不能推广为任意Gemma配置。

模块分别在`models/dspark*.py`、`nn/markov.py`、`data/dspark.py`、`methods/dspark*.py`、`inference/dspark.py`、`evaluation/dspark.py`。`examples/dspark_pipeline.py`是可直接运行的端到端例子。

## 训练语义必须明确

每个anchor构成一个并行块：它可看anchor之前的目标特征和自己块内所有query，不能看其他anchor块或未来目标上下文。保留随机anchor抽样、无效anchor遮蔽、块内连续有效label及距离衰减。Markov头提供vanilla、gated、rnn三种真实结构；confidence使用分布重叠量的停止梯度监督，不是“模型自己预测的置信度即正确率”。

默认权重为CE 0.1、概率L1 0.9、confidence BCE 1、距离衰减gamma 4。精确张量公式与作者loss源码单独对照；缩小的模型结构不等于作者公开大模型训练配置。

| 设置 | 实际目标与边界 |
|---|---|
| `normalization_profile='official_microbatch_mean'`（默认） | 每微批汇总所有DP副本的分母，再平均G个微批的梯度；有效token数不同的微批仍各占1/G |
| `normalization_profile='global_window'` | 全窗口各副本numerator之和除以全窗口denominator之和；这是可选的另一目标，不称为官方默认 |
| `empty_window_policy='official_step'`（默认） | 整窗无有效label仍执行官方optimizer step；AdamW衰减/已有moment可能更新权重 |
| `empty_window_policy='skip'` | 仅整窗所有rank都无有效label时跳过更新与更新计数；单个空rank/微批不改变其余数据的定义 |

这些设置、权重、teacher/vocabulary身份、缓存集合和DP计数域进入checkpoint及更新receipt。不能续跑时换目标却保留旧优化器状态。整个窗口在首次forward/参数gather前先验证；坏的后一微批或单rank错误不会让其他rank独自进入backward。

当前正式Method支持纯DP×ZeRO0–3。冻结目标embedding/head作为持久buffer而非伪装可训练参数；训练不更新它们，但draft hidden仍能经过冻结head反传。它们在每个rank完整复制，这不是节省词表显存的分片head实现。TP/PP/EP草稿训练、分布式目标特征抽取仍需专门provider。

## 数据与模型身份

`DSparkTeacherFeatures`复制一个只读目标快照。两家族的选中层都按HF hidden history索引读取，最后层包含最终norm。显式关闭环境autocast，避免同一个teacher hash生成另一种数值语义的缓存。只保存选中层及最终hidden，不保存巨大的逐token词表logits。

`publish_dspark_features`要求固定数据revision、明确许可声明和非空样本ID；每条是一个无padding序列。batch目前要求等长，不能无声补padding改变anchor语义。缓存绑定模型完整权重/配置/语义buffer、token-ID语义及提取家族。训练绑定缓存时会读取并比较实际tensor，而非只相信调用者写的cache ID；这是有I/O成本的严格profile，不称TB级高吞吐loader。

`publish_dspark_draft`需要实际成功的目标更新receipt及已绑定缓存。导出共享Trainer管理的当前完整权重/可选EMA和语义buffer，重新创建独立模型；不发布ZeRO3参数空壳。证明范围是“最近成功更新的目标、批准缓存集合和当前角色权重”，不是对所有历史数据的外部审计。

`models.dspark_import.import_dspark_state`可导入已安全读取的作者state_dict：检查完整名称/形状/有限值，唯一显式名称映射是confidence的`proj`层，冻结embedding/head必须和显式target逐值同精度相等。`embedding_head='from_target'`只接收两个词表tensor均被省略的源，禁止缺键后偷偷随机初始化。导入返回新模型，不修改target、不扰动全局RNG，也不生成虚假的训练成功receipt。调用者负责核对checkpoint配置和tokenizer；不是任意Hub配置自动转换器，不恢复上游FSDP optimizer或未序列化buffer的舍入历史。

## 推理：保持目标分布，不保证一定更快

目标cache始终比已提交文本落后一个anchor。每轮一次目标调用验证`anchor + proposals`；全接受时使用最后一个分布发出bonus token。逐位置confidence sigmoid阈值只截短提议长度，不豁免`min(1,p/q)`接受检验；拒绝时按正部`(p-q)+`重采样，并回滚目标状态及特征。双方都使用统一temperature/top-k/top-p/repetition规则。

草稿只增量投影新增context，并缓存context K/V，不把本轮query K/V永久留在历史。Qwen目标使用物理页所有权；当前草稿attention仍物化连续历史，不冒充融合PagedAttention。Gemma4目标的局部窗口/shared-owner状态不能简单切tensor：使用已验证的完整前缀replay恢复；实际replay调用、token量和时间全部计入报告。

当前decoder显式单请求，阻止并发重入；不等于连续批调度或多GPU服务。要求目标/草稿的存储精度一致，禁用环境autocast。EOS、取消、回调异常都释放两边状态；Gemma4同时释放绑定lease。缓存容量和位置容量是明确的失败边界，不自动修改采样配置。

`evaluate_dspark`使用同一固定cohort、目标、采样配置，交替运行顺序，保留所有失败样本。报告接受率、tokens/verification、TTFT、ITL、总延迟、backbone/head/目标时间与回滚量。随机采样的RNG消耗不同，不能要求同seed逐token相同；greedy则逐token及原始logprob对照。外接官方评分器不自动认证其数据/protocol，默认公开质量为`not_evaluated`，不自动晋级部署。

## 运行与验收

在项目根目录、已安装本项目的环境中：

```text
python examples/dspark_pipeline.py runs/dspark-qwen-001 --steps 12 --zero-stage 3
python examples/dspark_pipeline.py runs/dspark-gemma-001 --family gemma4 --steps 12 --zero-stage 3
```

输出目录必须不存在。例子包括真正的目标回复、缓存、可恢复sampler、fresh engine精确下一步恢复、部署导出/加载和留出提示的成对推理。目标是随机初始化的小模型，仅验证工程链，不能将小规模loss下降解释为语言能力。实际8步演示两家族已跑通；本机Qwen短例的接受率为0、总延迟也比目标单独运行更慢，照实保留，没有用“调用次数减少”伪造加速。

专项测试覆盖官方三种Markov全梯度、Qwen/Gemma完整草稿输出/梯度、随机anchor/mask、三loss、confidence调度；两种归一化和空窗口策略的真实DP2×ZeRO0–3、fresh恢复、缓存篡改拒绝；Gemma软截断logit和真实拒绝replay；严格权重导入。远程oracle需显式设置`ASTER_RUN_REMOTE_DSPARK_ORACLE=1`，逐文件SHA锁定源码。对应Gemma测试还锁定Transformers 5.10.2数学定义；不能拿本机安装的5.16.1冒充作者依赖版本。

测试报告与固定源码摘要见`docs/scope/validation-*.json`。GPU编译/吞吐、正式大模型权重、公开数学/代码/对话基准、不同thinking模式再训练均未由这些CPU测试证明。作者默认Qwen3-4B缓存约38TB；本项目不会自动下载它或假装已完成同规模训练。
