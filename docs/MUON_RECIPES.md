# 语言训练的 Muon 配方

`examples/recipes/language_muon.json` 是可直接运行的本地小配置，不下载模型或数据；
默认词表是随制品保存的 ByteTokenizer。这是工程流程示例，不表示公开语言模型质量。
在仓库根目录、已安装 Aster 或设置 `PYTHONPATH=src` 后运行：

```powershell
python -m aster train examples/recipes/language_muon.json --output runs/muon --store artifacts
```

## 配置与真实参数归属

省略 `training.optimizer`、设置 `null` 或 `{ "type": "adamw" }`，都继续执行原生
`Trainer` 的原默认 AdamW 路径；不会改变其 betas、eps 或 weight decay。
默认配方的 checkpoint 稳定身份也省略未启用的新字段，不因多出 `optimizer:null`
而发生无意义的不匹配。已有 Workflow 的完整源码身份仍正常校验，不能绕过源码变更检查。

Muon 必须明确 `profile` 和 `matrix_learning_rate`：

- `training.learning_rate`：辅助 Adam 的学习率；选择 Muon 后只作用于辅助参数。
- `optimizer.matrix_learning_rate`：Muon 矩阵的学习率；不与前者相乘，不存在第二个
  `auxiliary_lr` 别名，未知字段直接拒绝。
- `auxiliary_modules`：从完整、未分片模型解析的精确模块全名，必须包含 `lm_head`。
  允许显式增加模块，例如某一层投影；不接受通配符或按名字猜参数用途。
- 所有 `nn.Embedding` 参数、显式辅助模块参数和所有非二维参数走辅助 Adam。
  其余二维参数走 Muon。共享 embedding/head 只有一个更新 owner。

分组在 TP/ZeRO 改写存储前完成，固定的是原逻辑参数名，绝不是本地 `shards.N`。
已有 `MuonFactory` 在分片完成后绑定实际 owner；Newton–Schulz 作用于完整逻辑矩阵，
更新后再切回本地分片。这里复用同一个优化器，没有第二套配方专用实现。

## 两个官方 profile 不可混同

| profile | 来源 | 未指定时的 weight decay | 辅助 Adam epsilon |
|---|---|---|---|
| `keller` | [KellerJordan/Muon 的固定 muon.py](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py) | 矩阵/辅助均 0 | 1e-10 |
| `moonlight` | [MoonshotAI/Moonlight 的固定 toy_train.py](https://github.com/MoonshotAI/Moonlight/blob/c2ad5b20c605086526a179d36901bfc41b52b44b/examples/toy_train.py) | 矩阵/辅助均 0.1 | 1e-8 |

二者都采用 BF16 Newton–Schulz，但动量更新、形状缩放和辅助 Adam epsilon 的放置
并不相同。`profile` 选择现有后端的真实公式，不能只改元数据标签。可显式设置
`matrix_weight_decay`、`auxiliary_weight_decay`、`momentum`、`nesterov`、`ns_steps`、
`normalization_epsilon`、`auxiliary_betas`、`auxiliary_epsilon`、`missing_grad`。
配方默认 `missing_grad="skip"` 是显式的工程缺梯度策略，**不等同于 Keller 作者对
unused parameter 的逐步语义**；选择 `"zero"` 才把缺梯度视为零梯度继续对应状态更新。
本页 Keller 示例明确采用 `"zero"`。当前完整语言模型通常各参数均有梯度，这不能
替代对缺梯度分支的区别说明。
这些控制、精确来源 commit/SHA256 和有序 FQN 分组都进入 checkpoint 配方身份与制品
`recipe.json` 的 `execution.optimizer`，也进入制品 metadata 与阶段回执。

## 支持边界与续训

此 JSON 入口只认证原生 **Llama / Qwen2 / Qwen3**，支持 `dense` 和 `native_tp`，
TP×DP×ZeRO 0–3。PP、CP、EP、ETP、GTP 和其它模型/张量领域配方在进入 provider 前
明确拒绝；不能因为优化器能构造就声称支持所有架构组合。TP 配方当前为监督 CE，
teacher/KD 仍遵守原有并行目标限制。默认 AdamW 的其它已有配方不受该限制影响。

两进程 TP 示例：把示例的 `training_provider` 改为 `native_tp`，可设置
`training.zero_stage=3`，然后使用现有分布式入口：

```powershell
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m aster distributed-train path/to/muon_tp.json --output runs/muon-tp --store artifacts --tensor-parallel 2 --backend gloo
```

使用训练生成的 `checkpoint-final` 续训时，在 JSON 顶层添加 `resume` 指向该文件，
增加 `training.steps` 为期望的**总更新数**，并使用新的 output 目录。更改 profile、
任一侧学习率、参数组、动量或 NS 步数不是精确续训，应新建训练；加载会拒绝这些变化。
恢复包括优化器矩、数据游标和 RNG，不只加载模型权重。

本地验收涵盖两 profile×单进程 ZeRO0–3、真实 TP2×ZeRO3 JSON 训练/逐位续训/导出，
以及实际双进程 CLI。CPU 小配置不代表多机 GPU 吞吐或公开质量分数；完整矩阵 gather
是带明确内存成本的正确性 reference，不宣称其优化器通信已达到生产 GPU 最优性能。
