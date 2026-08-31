<p align="center">
  <img src="docs/assets/aster-banner.svg" width="960" alt="Aster：从原理到完整工作流的原生 PyTorch 框架">
</p>

<p align="center"><strong>看懂模型，掌握训练，再追踪它如何进入推理和评测。</strong></p>

<p align="center">
  <a href="README.md">English</a> · 简体中文<br>
  <a href="docs/GETTING_STARTED.md">快速开始</a> ·
  <a href="docs/LEARNING_PATH.md">学习路线</a> ·
  <a href="docs/ALGORITHMS.md">算法与论文</a> ·
  <a href="docs/STATUS.md">现状与待完成</a> ·
  <a href="docs/ROADMAP.md">开发路线图</a> ·
  <a href="docs/README.md">文档目录</a> ·
  <a href="CONTRIBUTING.md">参与贡献</a>
</p>

---

Aster 是一个原生 PyTorch 框架，把**模型、损失、训练、微调、蒸馏、推理、智能体和评测**放进相互衔接的工作流。你既可以沿着代码学习原理，也可以从小实验开始，逐步理解真实工程中的状态管理、资源约束与数值验证。

**[v0.1.0 — 首次发布](https://github.com/sutiankang/aster/releases/tag/v0.1.0)。** 从下面的原生实现和可运行示例开始。API 仍可能调整；支持范围与待完成内容见[当前状态](docs/STATUS.md)和[开发路线图](docs/ROADMAP.md)。

| 已实现的主要路径 | 尚需实现 | 尚需验证 | 许可及维护待办 |
| --- | --- | --- | --- |
| 原生模型与 Loss、共享训练、线性 LoRA、已支持的推理和评测流程 | 高级微调方法、更广的并行/服务组合、剩余 Agent 能力 | 多机 GPU 行为、硬件性能、公开预训练效果 | 许可确认、逐文件来源核查、维护治理 |

## 按你的目标开始

| 我想做什么 | 从这里进入 |
| --- | --- |
| 没有 GPU，先运行一个例子 | [LoRA 训练与合并](examples/quickstart.py) |
| 初学者，想知道先读什么 | [学习路线：问题、源码、实验](docs/LEARNING_PATH.md) |
| 查找某个算法及其官方出处 | [算法 → 代码 → 测试 → 来源](docs/ALGORITHMS.md) |
| 理解不同模块怎样组合 | [完整工作流示例](docs/EXAMPLES.md) |
| 判断是否适合实际任务 | [支持范围](docs/ROADMAP.md)、[模型细节](docs/MODELS.md) |

## 先跑一个真实的小实验

需要 Python 3.11+ 和适合本机的 PyTorch。首次使用先克隆仓库并进入项目目录；已有仓库时，跳过前两条命令：

~~~bash
git clone https://github.com/sutiankang/aster.git
cd aster
python -m pip install -e ".[test]"
python -m aster doctor
python examples/quickstart.py
~~~

这个例子会训练小型 LoRA 适配器，检查基础模型权重没有被更新，并比较合并前后的输出。不需要下载预训练权重、申请 API Key 或使用 GPU。

阅读顺序：[LoRA 原理与代码](src/aster/methods/distillation.py) → [监督损失](src/aster/methods/supervised.py) → [训练器](src/aster/training/trainer.py) → [对应测试](tests/unit/test_repository.py)。

## 模块如何真正连接

![模型和目标进入共享训练器，模型制品连接压缩、推理与评测。](docs/assets/workflow.svg)

| 示例 | 可以学到什么 | 运行入口 |
| --- | --- | --- |
| LoRA → 在线共享基础模型 | 适配器生命周期、INT8 KV 缓存、换出恢复与资源释放 | python examples/online_adapter_stack.py --kv int8 |
| 教师训练 → 学生蒸馏 → 评测 | 角色切换、不可变模型制品、统一评价协议 | python -m aster run examples/recipes/language_chain.json --output runs/language-001 --store artifacts |
| 小模型 → Loss → 更新 → 合并 | 微调的最小完整闭环 | python examples/quickstart.py |

示例使用小型模型和合成数据，验证的是工作流，不是公开榜单成绩或 GPU 加速比。数据要求、预期观察和修改入口见[示例导览](docs/EXAMPLES.md)。

## 这里有哪些内容

| 方向 | 代表内容 | 文档 |
| --- | --- | --- |
| LLM 与注意力 | GPT-2、Llama/Qwen、DeepSeek MLA/MoE、Mamba、混合循环注意力 | [模型说明](docs/MODELS.md) |
| VLM 与 VLA | CLIP/SigLIP、LLaVA、Qwen-VL、BLIP-2、OpenVLA、ACT、动作流模型 | [算法索引](docs/ALGORITHMS.md#multimodal-and-action-models) |
| 扩散与生成模型 | DDPM/DDIM、EDM、Flow Matching、Consistency、MeanFlow、Shortcut、Drifting | [方法与组合](docs/METHODS.md) |
| World Model 与规划 | RSSM、PlaNet、JEPA/LeWM、MuZero、搜索与模型预测控制 | [算法索引](docs/ALGORITHMS.md#world-models-and-planning) |
| 训练、微调与优化 | LoRA、Muon、梯度累积、EMA、检查点、已支持的并行与 ZeRO 组合 | [训练](docs/TRAINING.md)、[微调](docs/FINE_TUNING.md) |
| 强化学习与蒸馏 | DPO/IPO/SimPO、PPO/GRPO/RLOO、离线 RL、Token/特征/生成蒸馏 | [Loss 目录](docs/LOSSES.md) |
| 推理与智能体 | 分页缓存、前缀复用、连续批处理、推测解码、工具权限与 MCP 子集 | [推理](docs/INFERENCE.md)、[Agent](docs/AGENTS.md) |
| 评价体系 | 固定样本集合、公开评测适配、质量与资源配对比较 | [评测说明](docs/BENCHMARKS.md) |

各个名称都有具体范围，不应仅凭一个文件名判断“完整支持”。[算法索引](docs/ALGORITHMS.md)连接原理、实现、测试与官方来源；更细的状态见[能力清单](docs/scope/capabilities.json)。

## 面向学习者的使用方式

不要第一次就打开最大的模型文件。先选一个问题，读一个实现，运行一个测试，然后只改一个变量：

1. 为什么 LoRA 的 B 矩阵从零开始？
2. 为什么梯度累积时不能简单平均每个小批次的平均 Loss？
3. 为什么使用 KV 缓存后，输出应与完整序列前向保持一致？
4. 为什么冻结教师参数不一定意味着可以使用 no_grad？
5. TP、DP 和 ZeRO 分别切分了什么？
6. 为什么少步采样或缓存复用仍要做质量对照？

[学习路线](docs/LEARNING_PATH.md)给出了对应的阅读入口、命令、预期结果和练习。

## 当前边界与验证

QLoRA、DoRA、rsLoRA、IA³ 尚未实现；多 rank 连续 HTTP 服务、完整 MCP、广泛硬件性能和公开预训练效果验证也尚未完成。Aster 不承诺替代所有上游框架。详见[路线图](docs/ROADMAP.md)。

~~~bash
python tools/check_repository.py
python -m pytest tests/unit tests/integration -q
python -m pytest tests/distributed -q
~~~

部分参考对照与 GPU 测试需要独立依赖、设备或明确授权的数据，跳过不会记成通过。参见[测试说明](docs/TESTING.md)。

欢迎贡献可复现的小例子、独立的正确性测试或更清晰的解释：[贡献指南](CONTRIBUTING.md) · [安全说明](SECURITY.md) · [获取帮助](SUPPORT.md)。

如果 Aster 对你的学习或项目有帮助，欢迎[点一个 ⭐ Star](https://github.com/sutiankang/aster)，让更多人发现这个项目。也欢迎分享你的使用场景和建议。

## 来源与许可

算法作者、官方工程和本地实现对应关系见[算法索引](docs/ALGORITHMS.md)。借鉴不代表官方背书。

仓库尚未授予统一许可证。复用或再分发代码前，请查看[第三方许可与来源核查说明](NOTICE.md)。仓库不附带模型权重和基准数据集，它们另有使用条件。
