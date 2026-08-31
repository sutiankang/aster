# 训练到部署的原生优化

本模块把压缩和执行优化写成显式阶段，与共享 `Trainer`、蒸馏目标、制品存储和质量门禁组合。它不依赖执行 torchao、ModelOpt 或 vLLM 的模型实现，也不把打包文件、配置开关、随机模型测试等同于真实的质量/吞吐提升。

## 已实现与边界

| 能力 | 真正执行的计算 | 必须保留的边界 |
| --- | --- | --- |
| 分组权重 QAT | FP32 分组观察器、4/8bit 前向量化格点、按取整码越界裁剪的直通梯度、冻结观察器 | weight-only；不宣称完整 W8A8、FP8 QAT 或 QLoRA |
| QAT 部署 | 使用训练时 scale 真正打包 uint8/nibble，保存/重载 PackedLinear | 当前反量化后浮点 GEMM，不是低 bit 融合内核 |
| 结构化 MLP 剪枝 | SwiGLU gate/up 的同一组输出行与 down 输入列同时裁剪；真实改变 config/张量/参数数目 | 仅明确的 Llama/Qwen2/Qwen3/Mistral dense decoder；不是完整 Wanda/SparseGPT |
| 恢复训练 | 原生 teacher/student KL，共享训练器执行更新与断点恢复 | 本地小模型只证明梯度和流程；公开模型效果需另做协议评测 |
| torch.compile | prepare 时调用真实 torch.compile 并执行首次编译，比较 eager；bucket 失败后拒绝调用 | 默认 inductor 需要相应编译环境；CPU 测试 aot_eager 仅为真实图编译参考，不证明加速 |
| CUDA Graph | 单 GPU warmup/capture/replay，稳定输入/输出地址，复制新输入，串行保护静态缓冲区 | 当前 CUDA 环境未配置，GPU 测试明确 skip；不是已测 GPU 性能结论 |

## QAT 与剪枝组合

`prepare_qat(model, *, targets, bits=4, group_size=128)` 返回独立模型，目标必须是显式的未包装 `nn.Linear` 路径。共享权重或已包装模块会拒绝，防止一份 embedding/head 权重被变成两个不同参数。反向仍更新浮点主权重，scale 不承担梯度。

`configure_qat(model, observe=False)` 冻结 scale；`fake_quant=False` 可以显式做不量化前向，但此时拒绝导出“与前向一致”的 packed 模型。观察器与 fake-quant 状态是 checkpoint buffer，bits/group_size/梯度语义通过 `precision_contract()` 纳入共享训练器配置校验。恢复时需先构造相同的 QAT 拓扑，再调用 `Trainer.load_checkpoint`。不能直接把带 QAT buffer 的 `save_pretrained` 文件交给普通模型 factory 当作完整恢复。

`convert_qat(model)` 返回冻结的 PackedLinear 部署副本；原训练模型与优化器保留。导出复用 `inference.save_optimized_model`，再用 `load_optimized_model` 加载。冻结观察器后，导出绝不重新估计 scale，否则会改变训练所优化的量化函数。

`mlp_importance(model, *, batches=None, dataset_fingerprint=None, max_rows=2048)` 支持权重范数乘积，或者真实收集 post-SwiGLU 输入的 RMS × down 列范数。后者遵守有效 token mask。`prune_mlp(model, *, intermediate_size, importance=None, parent_artifact_id, calibration_fingerprint=None)` 返回 `PruningResult(model, manifest)`，记录每层保留的通道与父制品。它构造更小的原生配置并实际剪切权重，不是零掩码稀疏。先剪枝，再 KD 恢复，再 QAT；目前拒绝在 LoRA/QAT/packed MLP 上猜测复合变换。

建议流程是“固定 teacher/data/tokenizer → 结构剪枝 → KD 恢复 → QAT → packed 导出 → 原生推理 → 相同公开质量协议及实测资源门禁”。质量基准与候选必须共享评测协议；校准/恢复数据不得混入测试全集。对训练时间、部署字节、峰值显存、端到端延迟分别比较，不能用参数量降低代替延迟提升。

## 静态执行器

`CompileProvider(model, *, policy_artifact_id, backend='inductor', max_buckets=4, atol=1e-5, rtol=1e-4)` 与 `CUDAGraphProvider(model, *, policy_artifact_id, ...)` 提供相同生命周期：

1. `prepare(name, example_inputs)` 固定命名 Tensor 的 shape/dtype/device/stride，执行真实编译或捕获以及 eager 对照。
2. `provider(name, **inputs)` 仅接受预备的精确签名，不悄悄重编译成未知 bucket 或回落 eager。返回独立 Tensor，避免 CUDA 重放覆盖前一次调用的输出。
3. `observation()` 分开记录准备时间与同步后的调用时间、失败、调用次数、真实 backend 和 Torch 版本。
4. `close()` 清理当前实例的 bucket，拒绝后续调用；不重置其他实例的全局编译缓存。

执行器私有复制模型并冻结参数，普通 in-place 更改或参数替换令缓存失效。这不是防恶意 `.data` 写入的安全沙箱。当前只接受纯 Tensor 输入和单 Tensor 输出，不能直接接任意 Python KV 对象；应对明确的张量计算块使用它，并由上层处理状态。CUDA Graph 不允许跨请求并发写同一静态缓冲区。其同步计时包含当前实现的输入/输出复制开销，不是单独 kernel 计时，也不是 HTTP TTFT/ITL。

## 可复现检查

`tests/unit/test_optimization.py` 使用 PyTorch 原生 fake-quant 算子作为独立数值/梯度 oracle，覆盖 scale 冻结、别名拒绝、结构剪枝与全尺寸 mask 的对照、原生配置与文件重载、真实 aot_eager 图编译、bucket 生命周期和 CUDA 缺失拒绝。CUDA 捕获测试仅在真实设备可用时执行。

`tests/integration/test_optimization_recovery.py` 真正执行剪枝→KL 恢复→QAT 更新→训练状态保存/恢复（含下一次更新逐张量一致）→packed 保存/重载→缓存自回归推理。该测试的短 token 序列不是公开 benchmark，也不提供部署质量承诺。

## DiT 近似残差缓存

`ResidualCacheCalibration` 绑定模型制品、校准数据指纹和具体 probe；`fit_residual_calibration` 从实际成对观测距离拟合多项式，不生成虚构校准样本。`DiTStepCacheSession(model, *, policy_artifact_id, condition, schedule, calibration, threshold=.1, max_skip=2, audit_every=2, max_relative_error=.05)` 拥有单次采样的固定条件、时间序列和私有权重。`predict(sample, step=i)` 真实计算 patch/时间调制/输出层，在预算内复用 Transformer 主干残差；首末步和周期审计真实执行主干。步序或形状变化、模型变化都拒绝。

这采用 [TeaCache 官方实现](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py) 中首块调制输入距离、累计距离阈值及主干残差复用思路，但**不是 FLUX/Wan 模型实现，不沿用其硬编码拟合系数**。当前只适配本仓库 DiT。每次需要完整计算时，会在有待审计复用的情况下比较真正的输出空间误差；超限永久禁用本 session 的复用，并记录 `quality_status=failed_guard`。无超限也始终为 `requires_end_to_end_evaluation`，因为未审计步骤没有误差保证，整条采样轨迹仍需外部质量门禁。

`observation()` 记录条件/校准指纹、固定 schedule、复用次数、实际主干调用和逐步误差。主干调用减少不等于墙钟加速；审计还增加计算，必须另测端到端时间。本地测试使用显式非零输出的 native DiT，覆盖精确无复用对照、实际跳过主干、错误守卫禁用、跨制品/形状/步序拒绝，不声称取得任何公开生成质量分数。

## 官方依据

QAT 的 prepare/训练/convert 生命周期及 fake-quant 原理参照 [torchao QAT 工作流](https://docs.pytorch.org/ao/stable/workflows/qat.html) 和 [PyTorch per-channel fake quantize](https://docs.pytorch.org/docs/2.11/generated/torch.fake_quantize_per_channel_affine.html)，核心计算由本仓库基本 Torch 算子实现。它不与 torchao 全部 dtype、分组和融合后端保持全功能等价。

编译和捕获的真实运行时接口采用 [PyTorch 2.11 torch.compile](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html) 与 [CUDA Graph 官方说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)。结构化 SwiGLU 剪枝依据三个 Linear 的代数耦合，并通过完整模型 mask oracle 验证；不套用未实现论文的名字。GPTQ/AWQ/SmoothQuant 的已实现范围及官方源码链接另见 [INFERENCE.md](INFERENCE.md)。
