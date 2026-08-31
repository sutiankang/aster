# 原生分块注意力与可选融合核

本包补的是注意力真实数据流，不调用 `flash_attn`、外部模型或 SDPA 当作本仓库实现。现有默认模型公式保持不变，必须显式选择新 provider。

## 实现与验收边界

| 入口 | 实际实现 | 验收口径 |
|---|---|---|
| `torch_tiled` | 自定义 autograd；逐 Q/K tile 在线 softmax；反向重算概率、归约 GQA 的共享 KV 梯度 | CPU 公式、梯度、保存量、原生模型和 Trainer 可执行；不是融合 GPU kernel |
| `triton_fused` | 本仓库 Triton F/B；每 Q tile 的前向/dQ、每 KV tile 的 dK/dV；FP32 累积 | 已编写并接真实模型入口；当前无 CUDA，不能声称 GPU 已编译、通过或更快 |
| `torch_online_paged` | 原有 Torch 页表扫描 | 默认行为保留；不拼接历史 KV |
| `triton_fused_paged` | 本仓库核逐实际页执行，FP32 输出/logsumexp 合并 | per-page launch，不是 vLLM 单核读取 GPU page table；GPU 验收尚缺 |

Triton 首版固定 NVIDIA compute capability ≥8.0、FP16/BF16、Q/K/V 等宽 head_dim 32/64/128、32×32 tile。支持 GQA、causal/noncausal、左滑窗、非2次幂 Q/K 长度、明确绝对位置和 batch 行各自的 binary padding。它不是 packed `cu_seqlens` varlen 实现：padding 槽仍占物理存储。

不支持 attention dropout、任意 additive bias、FP8、二阶梯度、TP/PP/CP/EP kernel provider、量化投影、CUDA Graph capture、异构或递归状态族。任何这些能力不能由同名配置开关算作已完成。未测试的 GPU 后端保持可显式试验，不自动成为默认选择。

## 数学与内存

每个 tile 维护最大值 `m`、质量 `l=Σexp(score-m)`、未归一化输出 `a=Σexp(score-m)V`。合并时更新最大值并同时重标定 l 和 a。全遮蔽行规定输出0，logsumexp=-inf，不能进行 `-inf-(-inf)`。

反向只保存 Q/K/V、输出 O、LSE 及线性长度的固定位置/mask；重算当前概率块 P，使用 `dS=P*(dO·V - sum(dO*O))`。dK/dV 要对所有共享 KV 的 query head 求和。测试不仅检查最终梯度，还利用 saved-tensor hook 与算子分派检测无完整 Q×K 中间结果；没有让 autograd 隐式保存所有 tile 的二次图。

Torch profile 在 FP32（double输入为FP64）累积，禁用外层 autocast 对内部 einsum 的再次降精度。Triton dot 使用半精度概率输入和 FP32 累积，故声明数值容差，不声称逐bit相同。输入/输出有限检查有同步成本，现阶段不发布吞吐提升数字。

## 使用

```python
from aster.models import build_model, LlamaConfig
from aster.optimization.fused_attention import set_attention_backend
from aster.methods.supervised import CrossEntropyObjective
from aster.training import Trainer

model = build_model(LlamaConfig())
set_attention_backend(model, backend="torch_tiled", query_block_size=16, key_block_size=32)
trainer = Trainer(model, CrossEntropyObjective())
```

必须在 Trainer 构造前选择。provider 无参数，不增加模型权重键；`precision_contract()` 将 backend、显式 fallback、tile、实现源码哈希、官方来源和可选编译器版本纳入共享 checkpoint，防止无声更换恢复语义。普通 `save_pretrained` 是逻辑模型权重制品，重新载入后需再次显式选运行 provider；它不冒称自动保存部署策略。

setter会拒绝已经由Trainer拥有的模型；安装后再进行TP/PP转换也会被其真实provider前置检查拒绝。运行时只允许本仓库ZeRO3对原始dense Linear的物化wrapper，不能把这种生命周期和TP分片混为一谈。

CUDA模型可明确选择 `backend="triton_fused"`。未满足硬件/profile时默认抛错。如果调用者允许 `fallback="torch_tiled"`，`provider.work.backend/fallback_reason` 记录真正执行的路径，绝不把回退结果称作 Triton 测试。编译或数值失败不会被吞掉回退。

```python
from aster.inference import PagedAttentionRunner, InferenceEngine

runner = PagedAttentionRunner(model.eval(), policy_artifact_id=artifact.id,
    backend="triton_fused_paged", attention_fallback="torch_tiled",
    block_size=16, max_blocks=256)
engine = InferenceEngine(runner)
```

分页仍共享真实 PagedStatePool 的 prefix、分支 COW、容量驱逐、抢占重算与回滚。只读 lease 在成功和异常两条路径均等 GPU 同步后才归还，防止异步 kernel 尚在读旧页时复用其物理存储。页核只读取各页 view；必要时单页 contiguous，不创建历史 KV 大拼接。跨页合并采用 logsumexp 加权，不能平均各页 softmax 输出。窗口语义为 `key_position > query_position-window`，与原生 dense 模型一致。

## 官方参照（锁定版本）

- [Triton v3.4.0 fused-attention tutorial](https://github.com/triton-lang/triton/blob/c817b9b63d40ead1ed023b7663f5ea14f676f4bc/python/tutorials/06-fused-attention.py)，文件 SHA256 `5f312a051cf0f1b55d0aa64d04e76c74d7aa8622096ad77b75f5d444fd91b6a7`。
- [FlashAttention v2.8.3 forward kernel](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/flash_fwd_kernel.h)，文件 SHA256 `765dd3ef217bc9d79c9c0494ba52ea63767099be737c14604bec748d85f0dde3`。

官方源码用于核验算法、在线归约和反向重算机制，没有借用官方性能成绩。本仓库代码独立编写，未复制官方完整 kernel 或安装其运行时；Triton 本身只作为可选编译器。

## 尚待硬件验收

GPU 测试含 GQA F/B、完整遮蔽、绝对 offset、非整 tile、多种mask、真实 native model 和分页 COW；无 CUDA 时明确 skip。必须在真实支持设备执行这些 tests，记录 GPU/driver/CUDA/Triton/精度、warmup、编译排除口径，再做相同模型/cohort 下的端到端 TTFT/ITL/吞吐和 allocator 峰值比较。此处 `KernelWork` 的 tile/保存元素计数是算法内存证据，不是实测 GPU 内存，也不是公开 benchmark 分数。

`KernelWork.max_score_elements`在Torch中是含batch/head的一个调用tile，在Triton中是单program的逻辑tile；并发program数、编译器寄存器分配及缓存都不同，不能直接拿两个数字宣称显存减少比例。

## 顺带修复的共享CE错误边界

真实DP2测试复现了默认CrossEntropyObjective的风险：某rank到attention内部才发现非法mask时，ZeRO3另一rank可能已进入下一投影gather，导致通信错序。现在共享CE在已知Llama/Qwen2/Qwen3/Mistral/Mixtral/DeepSeekV3原生纯token路径上预检完整microbatch窗口，检查IDs、labels、mask、position、嵌入形状和caller cache禁用；之后由Trainer统一汇总错误，再进入任何forward/gather。此修复同时覆盖默认核与新核。

这不是通用多模态审计：QwenVL、BLIP2、OpenVLA、未知模型及非causal目标继续使用已有显式预处理/目标协议，不猜视觉位置或encoder/decoder长度。上述六族之外的纯token模型仍需逐族确认预检契约，不能将本包说成所有模型的输入安全已完成。
