# 原生工具 Agent 强化学习

`aster.agents.agent_rl` 完成一个有界但真实的闭环：当前原生模型快照 → 原生推理队列
随机生成 → 单次权限绑定 → 实际读取指定文件 → 独立验证 → RLOO/GRPO → 共享 Trainer。
没有调用 verl、TRL、Agent Lightning 训练器，也没有把已有 verified trace 的 SFT
重新命名为 RL。SFT 仍是独立的 `methods.agent_learning.AgentSFTMethod`。

## 实际支持边界

- 环境为 `ReadFileTask`：读取宿主指定的一个 UTF-8 文件，随后精确回答内容；文件、
  答案、提示、任务版本、工作区共同固定身份。只有真实完整读取回执和正确终答同时
  存在才得 1 分；正常预算耗尽、错误答案、非法动作得 0 分并保留在完整样本组。
- 只暴露 `workspace.read`，参数必须正好指向该任务文件。没有命令、网络、写入、
  MCP、任意用户奖励回调。路径及单次权限检查不是 OS 隔离；未宣称 Windows sandbox。
- 模型为本仓库 factory 可构造、原生 `ModelRunner` 支持缓存的文本模型；当前控制器
  单 rank、FP32 存储/训练、dropout 关闭。不会静默把混合精度、非标准状态或多模态
  输入当成已支持。多机异步 Agent rollout、多 Agent credit assignment 尚未实现。
- 采样严格温度 1、无 top-k/top-p 截断、无 repetition/bias/grammar 修改。EOS 和
  有限 token/step horizon 明确固定；greedy 轨迹不能拿来伪称此处的 on-policy 数据。

## 调用接口

```python
from aster.agents.agent_rl import NativeAgentRLMethod, ReadFileTask
from aster.agents.runtime import AgentConfig
from aster.core import digest_json
from aster.inference import SamplingConfig

# trainer/reference/tokenizer 都由受信宿主创建；reference 必须是独立参数对象。
method = NativeAgentRLMethod(
    trainer, reference, tokenizer, work_directory="./private-agent-rollouts",
    reference_tokenizer_fingerprint=digest_json(tokenizer.to_dict()),
    algorithm="rloo", group_size=4,
    agent_config=AgentConfig(max_steps=4, max_action_tokens=256,
        max_total_action_tokens=768, max_context_tokens=8192),
    sampling=SamplingConfig(seed=17, eos_token_ids=(tokenizer.eos_token_id,)),
    kl_weight=0.02,
)
task = ReadFileTask(id="read-001", prompt="读取 note.txt 并回答完整内容。",
    workspace=absolute_workspace, path="note.txt", sha256=actual_file_sha256,
    expected_answer=expected_utf8_content, revision="train-split-v1")
result = await method.update([task])
trainer.save_checkpoint("./private-checkpoints/agent.json")
```

也可分开 `cohort = await method.rollout(tasks)` 和 `method.optimize(cohort)`，以便
更新前检查证据。返回的 `AgentRolloutBatch` 包含不可变规范 JSON 和宿主 HMAC。
每个 cohort 只能消费一次。过期或损坏的 cohort 可显式 `method.discard(cohort)`；
discard 不训练、不回退采样计数。默认消息处理器为规范 JSON；自定义 renderer
必须提供明确 `processor_fingerprint`，tokenizer 必须完整导出 `to_dict()`。

## 数学目标与掩码

每次生成都保存实际 prompt/action IDs、原始模型 logp、实际行为 logp、采样参数、
policy 权重身份、tokenizer/processor 身份。下一次 prompt 包含的工具观察、历史
动作、验证反馈全部无 loss；历史 action 不会因为再次进入 prompt 而重复计分。

RLOO 的一个样本是完整工具交互，而不是某一次模型调用。其比值为：

`ratio_i = exp(sum_over_all_decisions_and_action_tokens(logπ_new - logπ_behavior))`

同任务的 G 条完整轨迹先计算 `A_i = reward_i - mean(other rewards)`，再在整个
轨迹比值上裁剪。可选参考 KL 是所有 action token 的采样 KL 和，作为 reward
惩罚。不能先切 microbatch 再计算 baseline，也不能把每次调用独立裁剪。

GRPO 使用完整同任务组的均值和样本标准差（correction=1、epsilon=1e-4）；每个
action token 使用所属整轨迹的同一优势，逐 token 比值裁剪与 k3 参考 KL，最后按
整轨迹的 action token 总数归一化。跨回合长度不是平均回合长度。共享 Trainer
只按 `LossTerm(unit='trajectory')` 聚合分子/分母，microbatch 不拆一条轨迹。

更新前重算当前策略的 teacher-forced action logp，验证它与在线 cache 生成的
记录一致。策略的训练 step 和完整参数 hash 都必须仍匹配；仅 step 相同但参数
被改动也拒绝更新。故障/缺失 token 概率导致整个 cohort 拒绝更新，不筛掉失败
episode 来美化学习分布。

## 工具证据与恢复

原生 AgentLoop 的生命周期、工具批准/开始/提交及原始回执仍全部落盘。控制器还
保留采样时进程内追加的原始事件，签封前与磁盘比较；因此重新计算磁盘 hash 链
不能冒充 live rollout。更新时再次核验签封、事件字节 hash、事件顺序、实际模型
动作对应的工具调用、回执路径/hash/binding/model_view、文件身份、完整样本组。
HMAC 不是公开签名，也不能抵御能读取宿主内存或私有 checkpoint 的攻击者。

完整 checkpoint 由共享 Trainer 保存所有策略/参考参数、优化器、scheduler/EMA
（若启用）、RNG、控制器任务身份、更新/采样计数和签封密钥。只能在无在途 cohort
的成功事务边界提交；没有半次工具执行的透明继续或自动重做。日志 UUID/墙钟和
路径不要求逐字相同；恢复后下一组 action IDs、概率、奖励及更新后参数逐位一致。
检查点含密钥，日志含任务内容，二者都是私有训练数据，不能直接公开发布。

## 验证与官方参照

测试使用真实 tiny Llama、真实文件读取与真实 AdamW/RLOO/GRPO。有限动作词表把
单条 JSON 动作映射为一个 token，明确只验证“选工具—读观察—给答案”的学习链路，
不是预训练语言模型的通用 JSON 能力，更不是 SWE-bench/GAIA 成绩。测试包含独立
多回合公式/梯度 oracle、4 次更新后解析任务成功概率改善、新控制器精确恢复、
伪造/重算日志链、伪造回执、过期参数、重复消费和预算失败全集。

2026-08-30 只读核对的官方源：

- [verl AgentLoop](https://github.com/volcengine/verl/blob/main/verl/experimental/agent_loop/agent_loop.py)：
  `AgentLoopOutput` 分开 response IDs/mask/logprobs；模型生成与工具观察不能混成同一动作。
- [verl ToolAgentLoop](https://github.com/volcengine/verl/blob/main/verl/experimental/agent_loop/tool_agent_loop.py)：
  生成/处理工具/终止状态转换及明确的回合、响应长度预算。
- [Agent Lightning trace adapter](https://github.com/microsoft/agent-lightning/blob/main/agentlightning/adapter/triplet.py)：
  真实调用 token、reward span 对齐和重复调用去重的边界，不能仅凭最终聊天文本推断轨迹。

这些是机制参考而非运行依赖；链接的 main 可变化，本轮没有取得可信 upstream commit
锁，故不宣称与某个锁定官方版本逐行一致。运行记录固定本地控制器/runtime/tools/
permissions 源文件 SHA；算法独立公式及真实闭环测试才是本实现的验收依据。
