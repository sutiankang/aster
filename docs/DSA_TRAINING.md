# DeepSeek DSA 两阶段训练

原生 `LightningIndexer`、真实 MLA teacher、统一 Trainer、ZeRO 分片、checkpoint
和部署制品已经接成一条链。不是用随机 attention 目标反传一次就称训练完成。
算法依据 [DeepSeek-V3.2 报告 §2.1.1](https://arxiv.org/html/2512.02556v1)；
推理结构另参照 [作者 V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)。
报告公开的训练算法可复现，但这里没有其预训练语料、全部超参数和模型质量证明。

## 生命周期

```python
from aster.models import load_model
from aster.methods.sparse_indexer import prepare_dsa_stage, DSAIndexerObjective
from aster.training import Trainer

model = load_model(local_native_model_directory)
prepare_dsa_stage(model, 'dense_warmup')
trainer = Trainer(model, DSAIndexerObjective('dense_warmup'), zero_stage=3)
result = trainer.step([batch])
trainer.save_checkpoint(checkpoint_path)
```

| 阶段 | 主注意力集合 | 可训练参数 | 损失 |
|---|---|---|---|
| `dense_warmup` | 完整 causal/padding 可见集合 | 仅 indexer | 主注意力 teacher 到 indexer 的 KL |
| `sparse_training` | indexer 选出的 top-k 可见集合 | 主模型与 indexer | 主 CE + 选集上的 indexer KL，梯度隔离 |

主注意力的 Q/K、缩放、RoPE 和 mask 来自同一次真实前向；teacher 是 dropout 前
的概率，对所有主 attention head 求和，再沿 key 归一化。teacher 与 indexer 输入
停止梯度，top-k 离散选集也不可微，所以 CE 不更新 indexer，KL 不更新主模型。
不用第二个随机 teacher，也不把训练 label 直接当索引器检索标签。

每层 KL 按全局有效 query 数归一化，再做层平均 × `indexer_weight`；CE 独立按
有效预测 token 计数。padding query 和显式 `indexer_query_mask` 均被排除，分母
使用 int64，不能在 FP16 中将大于 65504 的计数变成无穷大。`loss_mask` 只约束
语言监督，不暗中改变索引器的检索训练域。

阶段必须在 Trainer 创建前设置。改变冻结策略会改变 optimizer 所有权，不能在
同一个 Trainer 上随意解冻后沿用旧状态。跨阶段从标准权重与语义 buffer 重建模型，
创建新 Trainer；同阶段 checkpoint 则恢复完整 optimizer/RNG/采样器下一步状态。
阶段与 KL 权重属于 checkpoint 身份，不一致时拒绝恢复。

## ZeRO 与完整窗口检查

冻结主模型仍使用真实 ZeRO-3 分片，不常驻完整权重：前向按叶子临时 gather；
需要向输入传播梯度时重算，但不给冻结参数创建梯度、reduce-scatter 或 optimizer
状态。当前叶子只能全部冻结或全部可训练；叶子内混合冻结策略仍明确拒绝。
ZeRO0/1/2 也同步训练模型内冻结参数的初态，避免各 DP rank 使用不同 teacher。

目标的 `validate_training_context` 在参数移动、分片和 optimizer 构造前检查本地
拓扑声明，再由 Trainer 的 WORLD 握手汇总错误。整个累积窗口的 token/label/mask
检查在前向 collective 之前完成；单 rank 的坏尾批不能让其他 rank 卡在参数 gather。

已测 pure DP × ZeRO0–3，包含不等长 rank 样本、梯度/更新对照、冻结权重不变、
真实双进程新实例恢复、错误窗口零前向与缓存部署。当前训练 profile 要求 attention
dropout=0；DeepSeek 专属 TP/PP/CP/EP/ETP/GTP provider 不在这个验收范围内。

## 可执行闭环

```text
python examples/dsa_pipeline.py runs/dsa-001 --steps 2 --zero-stage 3
```

默认是随机小模型与仓库短文本，明确不是语言能力成绩。可用 `--source-directory`
传入包含原生 `model/` 与 `tokenizer/` 的目录，用 `--data` 提供本地许可 JSONL。
示例建立不可变父子制品、执行两阶段训练、分别验证下一步精确恢复，再从最终制品
重载验证缓存/无缓存 logits。不得拿示例训练步数代替生产预训练规模。

## 性能边界

主 attention/indexer 的参考路径仍有 dense score 与 boolean top-k mask。显式收集
teacher 会分配 `[B,H,Q,K]` 概率；这是可核验的算法实现，**不是稀疏 GPU 加速核**。
FlashMLA/DeepGEMM/FP8 indexer、稀疏 backward 融合、超长上下文吞吐与公开质量
仍需单独实现/验收，不能由 CPU 公式和训练恢复测试推导。
