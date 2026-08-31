# 原生逐页在线注意力

`PagedAttentionRunner` 已接通本仓库模型、页池、连续调度、prefix cache、采样和
HTTP/SSE，不是未被调用的数学函数。其注意力直接读取 block table 中的 KV 页，
不先 `materialize`/拼接完整历史，也不构造完整 `query_length × key_length` 分数。
旧 `ModelRunner` 连续状态路径保留，便于特殊模型兼容和独立正确性对照。

## 显式选择

```python
from aster.inference import PagedAttentionRunner, InferenceEngine, SamplingConfig

# model 是已从本地可信 artifact 加载的原生 Llama/Qwen2/Qwen3；这里不下载权重。
runner = PagedAttentionRunner(model, policy_artifact_id=artifact.id,
    backend="torch_online_paged", block_size=16, max_blocks=4096,
    query_block_size=32, key_block_size=64, tokenizer=tokenizer)
engine = InferenceEngine(runner, max_active=8, max_batch_tokens=128,
    prefill_chunk_size=64)
handle = await engine.submit(prompt_ids, SamplingConfig(max_new_tokens=128))
result = await handle.collect()
await engine.close()
```

未知 backend、特殊模型、非零 attention dropout 都显式拒绝。也可使用
`PagedAttentionRunner.from_artifact(store, id, loader=load_model,
backend="torch_online_paged", ...)` 先校验制品再加载。此页池内部布局含 padding
叶，不能直接作为模型 `forward(state=...)` 的标准 KVState；由本 runner 独占解释。

## 数据流与内存

每层复用原生的 Q/K/V 投影、QK Norm、RoPE、输出投影、残差与 MLP；只投影当前
chunk。旧 KV 保留在真实 `PagedStatePool` 页内，`read_pages` 返回零拷贝 view 和
逻辑 offset。query 切为 Qb，key 页按 Kb 上限切分。GQA 使用显式 KV head/group
维，不复制完整历史 K/V 到所有 query head。

一个 tile 的状态是最大值 `m`、指数和 `l`、加权 value 分子 `u`：

`m=max(m_old,m_tile)`

`l=exp(m_old-m)*l_old + exp(m_tile-m)*l_tile`

`u=exp(m_old-m)*u_old + exp(m_tile-m)*u_tile`

最后输出 `u/l`。全 mask 行返回零，不用 `-inf-(-inf)`。独立 key 分片使用相同
重标定公式合并，不能直接平均各片已归一化输出。输入/点积溢出显式失败，不拿
clamp 改写注意力数学。FP16/BF16 输入采用 FP32 accumulator，FP64 输入保留 FP64。

额外 score 空间上限是 `B × Hq × Qb × Kb`，而不是 `B × Hq × Q × K`；输出及
统计仍需 `O(B × Hq × Q × Dv)`，当前 chunk 各层新 KV 也暂存至提交。所有历史
KV 本身仍需页缓存容量。内核明确 `no_grad`：当前不提供 FlashAttention backward，
防止 autograd 保存每个 score tile 后破坏前向低内存约定。

`AttentionWork` 记录实际算过的 tile 数和最大 score 元素数。它不是 CUDA allocator
峰值、显存占用、TTFT、ITL 或吞吐测量，不能据此宣称性能领先。全因果未来或整个
窗口之外的页不做 QK；运行器当前逐请求扫描 attention，投影可同批，但不是 fused
ragged-batch CUDA kernel。

## 可见性与缓存

位置是绝对物理 token 轴：causal 条件 `key_position <= query_position`；左窗口
条件 `key_position > query_position-window`。Qwen2/Qwen3 每层真实 window 由
各自配置决定。当前仍保存窗口外历史页，支持可靠 prefix、COW、截断重算及普通
full-attention 层；尚未实现仅保留活动窗口的紧凑页回收。

`forward_batch(..., padding_masks=...)` 支持本次 chunk 的二值 key padding，旧
mask 与 KV 存在同一页并一起 fork/截断/COW。query 位置不擅自按 padding 重编号；
此 runner 与原生模型一致，padding query 不强制零输出。单独数学核另有明确的
`query_padding` 可使无效 query 行为零。在线 engine 接收无 batch padding 的请求
token 列表，不会猜测 token ID 0 是否是 padding；离线 padded 序列末端不能直接
拿无效 query 的 logits 当采样结果，也不要把 padded cache 发布到无 padding
请求的 prefix namespace。

`append_delta` 与既有 `append(full_state)` 共用一个容量/COW/事务实现；新路径
仅提交当前 chunk，不生成完整历史 tensor。`read_pages` 持有固定页代数和读 lease，
读者完成前 release 不复用页，追加共享尾页会复制写入。runner 在退出 lease 前
同步设备计算；直接低层使用者也必须保证异步设备读取确已结束。PyTorch 没有
只读 tensor 类型，零拷贝 view 是受信内核接口，不暴露给 Agent 或任意外部代码。

## 已测与未测

### 前缀索引：页对齐压缩 Radix 树

`PrefixCache` 现按完整 `PrefixIdentity` 分域，以物理页 token 序列为步长匹配
压缩边。共同前缀分裂时转移页所有权，延长单一路径时只追加后缀；不再为每个
长前缀重复保存从零开始的所有 token tuple 和页引用。`max_entries` 明确表示
压缩树节点上限，物理 KV 容量仍由 `PagedStatePool.max_blocks` 控制。

4096 token、block_size=16 的单一路径只保存 4096 个 token ID、256 个缓存页
引用、1 个压缩节点；这证明元数据规模，不是吞吐/显存测量。命中仍返回共享页表，
分叉延长继续使用原 COW；冷叶淘汰释放缓存引用，不破坏活动请求与读 lease。
默认留最后一个 prompt token 供 logits 重算，与此前对外语义一致。

`stats()` 给出节点/域/引用/token 元数据计数和命中/淘汰统计。随机最长前缀
oracle、租户与模型身份隔离、并发读取、分裂引用、淘汰和延迟回收均有测试。
设计参考 [SGLang RadixCache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)，
不是调用其实现，也不宣称包括它全部优先级、层级 offload 或调度策略。

已测：FP64 独立完整 softmax oracle、GQA、多 value 宽度、跨页/窗口/因果边界、
padding/all-masked、不同 query/key tile、独立分片合并及倒序遍历；FP32 真实
Llama/Qwen2/Qwen3 logits、历史 padding、长前缀+chunk、新页/COW/rollback、prefix
命中、并发 greedy、HTTP/SSE、容量抢占重算、超容量有界失败、草稿拒绝回滚。
测试直接禁止 `pool.materialize` 和 `self_attn.forward`；长前缀还在算子分发层
禁止 rank4 KV 时间轴 cat，避免只用一个“分页”名称掩盖连续重建。

本机PyTorch为CPU构建。已有本仓库Triton F/B与逐页LSE合并路径，见[NATIVE_FLASH_ATTENTION.md](NATIVE_FLASH_ATTENTION.md)；
CUDA用例仍没有硬件执行证据。新增真实INT8/FP8热KV、逐tile解码、分页CPU交换和在线多LoRA组合，
详见[推理文档](INFERENCE.md)。它们有独立公式、全模型/调度工作流侧证，不冒充GPU吞吐、
单核GPU页表或distributed paged attention。MLA/DSA/Gemma4/Delta/Mamba/多模态decoder的
此分页runner支持仍未完成；不能称为vLLM全栈功能等价。

## 官方依据

2026-08-30 只读核对 [vLLM Paged Attention 设计](https://docs.vllm.ai/en/latest/design/paged_attention/)：
逻辑 token 通过 block table 定位物理 KV 页，head/token 布局与实际 cache 所有权
是数据流边界。该文描述的具体 CUDA 内核还会使用分数暂存，本实现没有照搬它的
CUDA 布局或声称其硬件性能。

稳定逐 tile 归一化参照 [FlashAttention 官方前向源码](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py)：
分块计算 QK、在线最大值/指数和及 value accumulator。该文件本身标为实验性
Triton 实现；此处只复核数学机制，自行实现 PyTorch 前向，不调用或复制其 kernel。
本轮锁定的 vLLM commit 源链接未能取回，因此没有假称对该提交做了源码逐行验收。
