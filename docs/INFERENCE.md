# 原生推理与部署

本目录拥有调度、分页状态、采样、传输和部署状态机，不启动vLLM进程来冒充自主实现。
CPU上的公式/存储语义验证不等于官方CUDA吞吐；没有实际基准记录就没有加速成绩。

## 模块边界

|模块|已实现路径|明确边界|
|---|---|---|
|`state`|真实tensor页、generation引用、refcount、借用确认、COW、压缩Radix前缀；可选INT8/FP8热KV|ModelRunner兼容路径重建连续状态，PagedAttentionRunner逐页消费|
|`runner`|独立私有参数、原生模型prefill/decode、等长状态批处理|未把窗口/循环/视觉状态误当普通KV|
|`engine`|连续加入/完成请求、decode优先、chunked prefill、token预算、容量抢占归档/恢复或显式重算|不同状态长度/身份分组，不是ragged fused attention|
|`sampling`|模型原始与行为logp、temperature/top-k/top-p/penalty/grammar|采样变换顺序固定并随rollout保存|
|`speculative`|两个真实模型、多token draft验证、精确拒绝采样、实际回滚/重放|独立同步API；尚未和continuous HTTP/TP×PP组合|
|`http`|loopback HTTP、completions/chat、SSE、cancel、health/ready/metrics|没有公网认证/TLS，拒绝公网绑定；不是全OpenAI兼容API|
|`deployment`|ArtifactStore校验、warmup、原子切换、旧请求保留旧策略、rollback|单机进程内版本路由，不是容器编排平台|
|`distributed`|dense Llama/Qwen2/Qwen3 TP×PP、DP独立请求、stage-local KV|同步请求collective；没有多rank HTTP协调或高吞吐PP流水|
|`checkpoint`|HF safetensors单文件/分片、strict key/shape、meta逐tensor加载、TP/PP读slice|支持上述三族；不执行remote code，不下载，不导入官方模型|
|`task_runners`|field、latent codec、物理动作块、RSSM、encoder内容缓存、typed token snapshot|非token任务不强行使用token scheduler或TTFT指标|
|`offload`|完整typed归档；新增PagedStateArchive逐页原码交换，接入在线抢占/恢复/取消/host满重算|传输在线程执行并等待完成；没有计算与PCIe预取重叠证据|
|`adapters`|共享base的在线多LoRA、A/B低秩计算、内容哈希、请求pin、HTTP选择、原生训练接入|按adapter分组；不是mixed-LoRA融合GEMM、DoRA/QLoRA/TP-PP的在线组合|
|`optimization`|真实4/8bit权重打包、RTN、GPTQ、线性层AWQ、SmoothQuant重参数化|反量化后float计算，不是int4/FP8融合kernel|

## 最小本地服务

```python
from aster.models import load_model
from aster.data import load_tokenizer
from aster.inference import DeploymentRouter, HTTPServer, ChatTemplate

router = DeploymentRouter(store, loader=load_model,
    tokenizer_loader=load_tokenizer,
    chat_template_loader=ChatTemplate.from_pretrained)
await router.deploy(artifact_id, warmup_prompt_ids=[1, 3])
async with HTTPServer(router) as server:
    print(server.url)  # 默认仅127.0.0.1
    # 宿主在此运行自己的生命周期，不靠隐式后台常驻进程。
```

制品必须带实际tokenizer与chat template；服务不会猜测特殊token约定。chat当前仅文本，
未知字段拒绝；`FiniteJSONGrammar`支持有限枚举/const/有界整数/固定对象/有界数组，
无限字符串、递归schema和超预算变体拒绝，不宣称完整JSON Schema支持。

`InferenceEngine.submit(prompt, SamplingConfig(...), timeout_s=...)`返回handle。
流式消费者遍历handle；非流式消费者用`await handle.collect()`持续排空事件队列。
只等待`result()`不会被视为正在消费stream，慢消费者会触发背压。取消必须等待当前
native forward完成确认才能释放页；Python不能安全强杀正在执行的C++计算线程。

请求参数绑定`PrefixIdentity(policy_artifact_id, adapter, processor, position,
multimodal_digest, tenant)`。前缀缓存不得跨这些域共享；warmup缓存与时间不混入生产测量。
在线LoRA用下文`MultiLoRARunner`显式注册；普通runner仍不会仅根据一个adapter字符串改变权重。

## 热低比特KV、交换和在线LoRA组合

```python
from aster.inference import (PagedAttentionRunner, KVQuantization, MultiLoRARunner,
    PagedStateArchive, InferenceEngine, SamplingConfig)

base = PagedAttentionRunner(model, policy_artifact_id=base_artifact_id,
    backend="torch_online_paged", kv_quantization=KVQuantization("int8"),
    block_size=16, max_blocks=256)
runner = MultiLoRARunner(base, max_adapters=8, max_adapter_bytes=256*1024**2)
# trained_model来自本仓库inject_lora + 统一Trainer；部署时核对base全部权重。
adapter_id = runner.register_trained_adapter(trained_model, base_artifact_id=base_artifact_id)
archive = PagedStateArchive(runner.pool, max_bytes=1024**3)
engine = InferenceEngine(runner, offload_archive=archive)
handle = await engine.submit([1, 3, 7], SamplingConfig(max_new_tokens=16),
    identity=runner.resolve_model_identity(adapter_id))
result = await handle.collect()
await engine.close()
runner.remove_adapter(adapter_id)  # 有排队/执行/被抢占的请求时拒绝卸载。
```

也可用`register_adapter({完整Linear路径: LoRAWeights(A,B,alpha)}, base_artifact_id=...)`。
注册复制低秩权重，内容变化得到新ID；不会复制第二份base或merge/unmerge它。仅支持
原生dense Llama/Qwen2/Qwen3的Linear目标。`HTTPServer`的`model`可选宿主已注册的
adapter ID；普通响应的`model`仍是base制品，`aster.adapter_id`明确记录增量身份。
没有远端路径加载接口或自动下载，宿主负责adapter来源与授权。注册/选择受锁保护，
不宣称多模型并发执行；同一轮不同adapter分组计算。

`KVQuantization`支持`int8`、`fp8_e4m3fn`、`fp8_e5m2`；scale按token/head的最后维最大
绝对值计算，INT8采用最近偶数舍入。FP8是真实float8码存储；并非FP8 TensorCore attention。
新chunk和旧页都采用相同量化语义，避免结果依赖chunk边界；COW与offload直接复制码和scale，
不反复量化历史。注意力每次只解码当前tile，不物化完整float历史。使用FP32 scale有额外字节
成本，`pool.storage_metrics()`记录实际持久tensor字节；这不是GPU峰值。量化不是无损替代FP16。

`PagedStateArchive`满时明确回退重算并计数；不偷偷驱逐尚未恢复的请求。恢复先预留全部页，
拷贝确认后一次发布；取消在途操作先等待，再回收handle/页。可选pinned host内存要求CUDA
allocator；当前CPU证据覆盖生命周期与原码一致，不代表真实DMA、预取重叠或带宽测量。

## 并行与加载

```python
from aster.training import ParallelConfig, ParallelContext
from aster.inference import load_hf_safetensors, CollectiveGenerator, SamplingConfig

# 每个rank由宿主先初始化torch.distributed，并以相同配置建立同一组网。
context = ParallelContext(ParallelConfig(tensor_parallel=2, pipeline_parallel=2))
model = load_hf_safetensors(local_snapshot, parallel=context)
generator = CollectiveGenerator(model, context, policy_artifact_id=artifact_id)
result = generator.generate([1, 3, 7], SamplingConfig(max_new_tokens=8))
```

TP按Q/K/V头、FFN中间维分片，输出归约；词表logits显式gather。PP只发hidden并由尾阶段
广播logits，各阶段保留自己的KV。模型组leader采样一次并广播token/原始logp/行为logp；
取消也先广播再同步退出。通信域复用`ParallelContext.tp_pp/tp/pp`，不私建隐藏通信网格。
GQA头数不整除TP、CP、窗口、MoE、MLA等不支持组合明确拒绝。

HF导入严格映射配置，未知计算字段、重复JSON键、remote code、外部量化布局、索引逃逸、
额外/遗漏/错shape权重、tie冲突和加载期间文件变化都拒绝。`safe_open.get_slice`读取
本地所需tensor切片；meta构造避免完整参数随机初始化。`load_report`中的最大单次物化
tensor大小不是RSS峰值，mmap与OS页缓存仍需独立硬件测量。tokenizer/chat处理不由权重
导入器猜测。`load_hf_artifact(store,id)`先执行统一制品校验。

## 其他状态与模型类型

`StatefulTokenRunner`支持dense/window/MLA、DSA indexed MLA、Qwen3Next hybrid Delta、Mamba、
DeepSeek V4 compressed-window MQA和Qwen3-VL完整视觉状态。
它保存每请求`rope_delta`，不把它展开后丢失；循环/窗口状态只fork或从完整输入replay，
不伪造任意位置truncate。这是实际cached decoding路径，但尚未接在线连续调度。

DSA分页codec显式保留latent/RoPE/index三叶，各叶末维可不同但序列轴同步；
Mamba/V4只走完整snapshot，不把卷积记忆、未关闭压缩窗与overlap状态硬切成KV。

`FieldRunner.predict(sample,time,condition)`保持epsilon/x0/v/score/velocity/edm_residual/consistency_residual参数化；求解器
属于methods。`LatentRunner`显式应用制品中的scale/shift，默认posterior mode，采样需要seed。
`ActionRunner`输出物理单位动作、valid mask、ActionSpec和策略ID；执行horizon含padding
就拒绝，不复用陈旧控制命令。本框架没有在测试中驱动真实机器人。
`DynamicsRunner`的observe/imagine状态有策略版本绑定；想象reward不是实际环境成绩。
`EncoderRunner`缓存键包含实际tensor内容、processor、grid/时间参数和tenant；返回独立副本。

`StateArchive`默认无损CPU归档；可选择dense/window/MLA INT8浮点叶子存储。
循环/视觉/DSA索引组合有损状态明确不支持。恢复后仍由原生float模型计算，不能把存储节省
称为推理延迟提升；量化误差必须随实际质量评估通过后才允许部署。

## 量化组合

```python
calibration = collect_calibration(model, batches, targets=names,
    dataset_fingerprint=data_id, max_rows=2048)
candidate = quantize_model(model, targets=names, algorithm="gptq",
    bits=4, group_size=128, calibration=calibration)
save_optimized_model(candidate, output, base_artifact_id=teacher_id,
    transformation_metadata={"calibration": data_id})
reloaded = load_optimized_model(output)
```

校准使用真实activation与有效padding mask；数据身份不能只写“random”。GPTQ实现阻尼
Hessian逆Cholesky顺序误差反馈；AWQ实现单个线性层的scale/clip搜索，不宣称完整
非线性Transformer block级官方配置。SmoothQuant实现等价通道重参数化并权重量化，
尚无W8A8激活量化/fused部署。相互组合需显式配方和质量/资源promotion gate，不自动
把所有优化串联后宣称效果改善。

## 测量与验证

`GenerationResult.metrics()`标记server emit单调时钟；TTFT从收到请求到首token发送，
queue单独计算，ITL逐token时间差。`measure_http`是真TCP/SSE客户端，记录客户端首token
到达和ITL、失败数、观察窗口吞吐及显式SLO goodput；不把排队时间冒充模型计算时间。
单token没有ITL；无首token没有TTFT，不用0伪造。失败/提前结束保留在分母中。

本地测试覆盖分页生命周期、COW/前缀隔离、容量抢占重算、缓存/full forward对照、
原生多请求SSE/取消/错误/退出、制品切换/rollback、spec接受/拒绝、量化重载、
原生Agent工具回合、三种真实Gloo并行布局、动作/RSSM/视觉/循环状态。
可选official oracle验证使用已安装Transformers wheel，其版本单独记录，
不声称wheel就是源码锁。小型随机权重只证明实现/协议正确，绝不是公开benchmark模型质量。

原生QAT→packed、结构化剪枝→KD，以及实际torch.compile/CUDA Graph执行器和带误差审计的
DiT近似残差缓存见[OPTIMIZATION.md](OPTIMIZATION.md)。CUDA Graph只有实现与设备门禁，
此CPU环境没有GPU执行证据，也未自动接入所有在线请求的动态状态。

仍缺：单核paged/ragged GPU页表内核、在线动态KV的CUDA Graph集成、计算/传输重叠预取、完整低精度融合kernel、
多rank连续HTTP调度、分布式故障重启、多LoRA的融合/分布式执行、全JSON语法、全模态在线接口、
跨优化组合的真实大型checkpoint/硬件/公开基准验收。

公式、工作流与硬件测试分层见[测试指南](TESTING.md)；
上述实际代码缺口仍保留，不能据此声明全部vLLM功能等价。

## 官方来源

在线LoRA的低秩公式核对[Microsoft作者实现](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)，
请求选择/共享base设计核对[vLLM v0.18.2 LoRA说明](https://docs.vllm.ai/en/v0.18.2/features/lora/)。
热KV的per-token/head格式核对[vLLM torch_utils](https://github.com/vllm-project/vllm/blob/main/vllm/utils/torch_utils.py)，
抢占生命周期核对[vLLM scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)。
这些2026-08-31只读核对的main页面不是固定commit的逐行执行oracle；本仓库有独立数值/状态机测试。

设计参考[vLLM固定源码](https://github.com/vllm-project/vllm/tree/8c51b92654100aa1d698aeef862cad09c8cc5df8)、
[prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)、
[speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)、
[指标定义](https://docs.vllm.ai/en/latest/design/metrics/)。这里独立实现对应数学/生命周期子集。

量化公式分别参考[GPTQ固定源码](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py)、
[AWQ固定源码](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py)、
[SmoothQuant固定源码](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py)。
各上游许可需按仓库保留，不由本文件假定统一MIT许可。
安全权重切片遵循[safetensors官方API](https://huggingface.co/docs/safetensors/main/en/index)。
# Gemma4 共享 owner 状态接入

新增 `Gemma4SnapshotRunner`，用于本仓库 `Gemma4ForCausalLM` 和 `Gemma4ForConditionalGeneration` 的文本/图像/视频文本输出。它复用 `InferenceEngine` 的真实连续调度、背压、SSE/取消和容量抢占，但不是普通 `KVStateCodec` 或 CUDA paged-attention 的别名。

Gemma4 只存前面独立 owner 层的 KV；后续 shared 层不存副本。local/global 的头数、头宽和历史长度可能不同，local 窗口已经丢掉旧位置，故**原生 state 不可 truncate**。本接入逐 owner 校验实际形状/完整模型身份，快照只存真实 owner 张量。`Gemma4SnapshotPool` 用引用计数共享快照；`fork` 不复制张量，随后分支提交生成独立快照，模型始终读取独立副本。

`pool.truncate(sequence, length)` 的语义是明确的 **rollback by replay**：从该请求保存的 token 历史和冻结视觉输入重放到目标长度，不切已有滑窗张量。不能回滚到半个图像/视频视觉块。事务式整批提交先检查总缓存字节，容量不足不先提交半批请求；调度器随后清理前缀/抢占，重算后继续原采样 RNG 序列。`SpeculativeDecoder` 目前仍要求真正可 truncate codec，所以不自动接这一路径，更不声称已经实现高效 Gemma4 speculative CUDA verification。

```python
from aster.inference import Gemma4SnapshotRunner, InferenceEngine, SamplingConfig
from aster.models import load_model

runner = Gemma4SnapshotRunner.from_artifact(
    store, gemma4_artifact_id, loader=load_model,
    processor_id=processor_artifact_id,
    max_cache_bytes=256*1024**2,
    max_modality_bytes=64*1024**2,
)
engine = InferenceEngine(runner, max_batch_tokens=1024, prefill_chunk_size=128)
handle = await engine.submit(
    prompt_token_ids, SamplingConfig(max_new_tokens=32, temperature=0),
    modality_inputs=packed_visual_tensors,
)
result = await handle.collect()
await engine.close()
```

`modality_inputs` 使用模型真实字段：图片 `pixel_values/image_position_ids/image_batch_indices`；视频 `pixel_values_videos/video_position_ids/video_batch_indices`；可选当前 prompt 的 `mm_token_type_ids`。只接受张量，禁止覆盖 state/position/attention/cache 控制。实际像素、二维坐标、素材归属、视觉 placeholder 位置、processor/model/layout、tenant 共同构成自动核验的 prefix identity；调用者传入假的 multimodal digest 会被拒绝。输入接纳时复制冻结，之后修改原 tensor 不改变请求。

第一次 prefill 必须包含当前请求所有完整视觉块；其长度超过 `max_batch_tokens` 会在接纳时拒绝，不能拆半块再称为双向视觉注意力。后续 decode 不重传媒体。前缀只在实际已有或可完整重放的边界存储；如需保存 prompt-1 状态，会真实重放并计入 `forward_calls/input_tokens_computed`，不假称免费截断。相同像素但不同 placeholder 布局、不同 tenant 或 processor 不共享命中。

当前 Gemma4 多模态 token 输入可能有超过文本输出词表范围的 image/video placeholder。只有这两个模型明确声明的越界 placeholder 从 repetition-penalty 上下文中排除，并在结果 `sampling_transform_order` 记录该步骤；其他越界 token 仍报错，原始 prompt token IDs 保留。`StatefulTokenRunner` 和无损 `StateArchive` 也已支持 `gemma4_shared_kv`；有损 INT8 状态仍拒绝。

边界与测试：计算目前逐请求执行，不声称异长/多模态 GPU 融合 batch；预算只覆盖持久快照和被引用的输入，不包含 forward 临时激活/拷贝峰值。`used_blocks` 为兼容诊断字段，表示完整 snapshot 对象数，不是 GPU 页数。HTTP 文本接口可直接使用；媒体 tensor 输入目前经受控 `engine.submit`，没有开放任意 JSON/文件下载的网络媒体入口。测试覆盖与 dense greedy/逐步 logits 一致、跨窗口 rollback、COW、真实容量抢占后重算、图片/视频布局隔离、输入冻结、SSE 与断开取消。CPU 已验证，GPU 性能未测。
