# MuZero 搜索：PUCT 与 Gumbel

`aster.planning` 是原生 Torch 搜索实现，不调用 JAX/mctx 搜索器，也不执行环境
动作、下载模型或拥有优化器。公开公式依据来自 Google DeepMind 的
[mctx](https://github.com/google-deepmind/mctx)。与模型/学习方法通过两个窄协议组合：

```python
RootOutput(prior_logits, value, embedding)
RecurrentOutput(reward, discount, prior_logits, value, embedding)
```

`prior_logits` 为 `[B,A]`，value/reward/discount 为 `[B]`，embedding 为 `[B,D]`。
值和奖励须先由模型自己的 categorical-support decoder 转成标量；不能把 value
logits 当作标量价值。`recurrent_fn(action[B], embedding[B,D])` 必须返回
`RecurrentOutput`，每轮 simulation 恰好调用一次完整 batch。

```python
import torch
from aster.planning import gumbel_muzero_policy

model.eval()                         # 回调不得隐式训练或改变模型状态
generator = torch.Generator(device=device).manual_seed(17)
root = model.search_root(observations)
result = gumbel_muzero_policy(root, model.search_step,
    num_simulations=64, max_depth=8, max_num_considered_actions=16,
    invalid_actions=invalid_mask, generator=generator)

action = result.action               # B；调用者决定是否执行环境动作
policy_target = result.action_weights  # B,A；可进入同一MuZero训练方法
value_target = result.search_tree.summary()['value']
```

## 不是两个名称对应同一种搜索

PUCT 用先验概率、访问次数与 Q 值探索奖励选边。Q 按当前 parent 的搜索 V 和
已访问兄弟的范围归一化；未访问边得分补零。可显式混合根 Dirichlet 噪声。
返回的训练 target 是原始访问分布，temperature **只改变最终动作抽样**。
temperature 为零时，最多访问项并列仍均匀抽样，而不是强行返回最小 action ID。

Gumbel MuZero 在根使用 Gumbel 排序和 sequential halving 访问预算，在内部节点
使用确定性的改进策略频率调度。缺失 Q 按 raw V 和先验加权的已访问 Q 混合补全，
再进行范围/访问数缩放。raw V 不等于后来的搜索均值。最终从最多访问的候选中
选择最高 `Gumbel + prior logit + transformed Q`；训练 target 则是改进 logits 的
softmax，不是 PUCT 的访问分布。非 2 次幂候选数、预算不足整轮、仅一个有效动作
都使用同一明确预算规则。

两者都使用均值 backup：边价值是 `reward + discount * V(child)`，初始 root V
作为第一次访问计入均值。discount 允许 `[-1,1]`，显式负号可表示零和玩家视角；
零 discount 截断价值传播。达到 max_depth 时会重新评估已有叶子，不偷偷改为
只读缓存值。终止状态应由调用者明确处理，本实现**拒绝全 invalid root**；mctx
在该边界返回 action 0 的行为不会被冒充成一个合法环境动作。

## 随机性、资源和错误边界

必须提供与搜索设备匹配的 `torch.Generator`。Dirichlet/gumbel/tie-break/最终
categorical 均使用它，不修改全局 Torch RNG。保存 `generator.get_state()`，
恢复时 `generator.set_state(saved)`；返回结果也含搜索完成后的 `rng_state`。
若 recurrent 回调自身有随机性，其 RNG 由回调所有者一同保存；应优先使用
eval 模式、无副作用的模型回调，不能把搜索 generator 当作所有环境随机流。

`max_tree_bytes` 在分配/抽样前约束树的持久张量存储（默认 512 MiB），包含
`B × (simulations+1) × A` 边数组与 embedding；不是进程 RSS/临时 workspace
的硬上限。所有张量形状、设备、有限值、动作掩码、discount 范围均明确校验。
不支持任意 embedding PyTree、连续动作采样 MuZero、机会节点 stochastic MuZero
或自动模型并行搜索。回调错误直接传播，不静默切换模型/算法、不重试副作用。

当前是数学和存储参考实现，Python 控制流与设备同步不会自动成为高吞吐 GPU
搜索。没有 CUDA/NCCL、多机、公开环境质量或与 mctx 的吞吐等价证据。

## 测试与官方证据的区别

`tests/unit/test_mcts.py` 使用独立标量 dict 树逐节点/边对照、非整除 halving
向量、深度截断重评、负 discount、真实 batched callback、显式 RNG 精确恢复、
内存/输入拒绝。它证明这些公式与状态转移，不是“mctx 已在本机执行”的替身。

`tests/parity/test_planning_mctx.py` 在独立的 JAX/mctx 对照环境运行真实
官方搜索，比较整棵树访问/拓扑、价值、策略 target 并记录版本。缺依赖时明确
skip，测试本身不自动安装。2026-08-31 已另建隔离环境，使用 mctx 0.0.71、
JAX/JAXlib 0.11.1 实际执行 MuZero/Gumbel 两项对照，均通过。范围为固定向量
动力学、零 root noise、13 次模拟、深度 2；不是 Atari 或棋类公开成绩。
Torch 与 JAX 的 PRNG 不同，相同 seed 不能声称跨框架抽样逐位一致。

可选安装入口：`python -m pip install -e ".[test,planning-oracle]"`。上游发行版
[mctx v0.0.71](https://github.com/google-deepmind/mctx/tree/e9d249eb0ed5c455a48f7ce5f0476ea5ece9b63a)
与实际安装源码摘要记录在本轮验证记录中；运行时仍不导入 mctx/JAX。

2026-08-30 实际读取的官方依据：

- [action_selection.py](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/action_selection.py)
- [qtransforms.py](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py)
- [seq_halving.py](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/seq_halving.py)
- [search.py](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/search.py)
- [policies.py](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py)
- [tree.py](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/tree.py)

这些链接为动态 main，不冒充不可变 commit pin。上游许可为
[Apache-2.0](https://github.com/google-deepmind/mctx/blob/main/LICENSE)，源文件注明
DeepMind Technologies Limited 2021；这里保留出处，不复制完整作者仓库。未来
引入具体上游源码文件时仍须逐文件保留其 copyright、license 与 NOTICE。
