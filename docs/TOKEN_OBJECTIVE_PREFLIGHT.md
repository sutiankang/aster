# Token学习目标：完整累积窗口的输入预检

本包不是新的训练算法，而是把既有监督、蒸馏、偏好优化和GRPO目标的输入错误，
在任何模型forward／ZeRO参数收集之前对称拒绝。它同时修复DPO／IPO原有的梯度
中断：参考分数仍stop-gradient，但从策略分数减去参考分数的运算不再位于no_grad中。
损失数值不变，策略梯度和真实optimizer更新得到独立公式检查。
另修复已审计纯token KD的特征mask来源：嵌套`model_inputs.attention_mask`与顶层
输入具有同样语义，padding不再误入特征loss；未审计rich／多模态分支不改动。

## 调用与职责边界

`methods.supervised.preflight_causal_microbatches(model, batches, causal=True)`只读取
输入张量和已审计原生模型的静态配置，不执行模型、不读取完整参数值、不发通信，
也不偷偷修正标签／截断序列。`Trainer`仍拥有完整窗口物化、WORLD错误汇总和之后的
优化器生命周期。本包不修改训练runner。

| 目标 | 在第一次forward前检查的真实分支 |
| --- | --- |
| CrossEntropyObjective | 当前模型的全部微批；原有行为抽为共享helper |
| DistillationObjective | 学生全部输入、教师同一批实际输入，各自vocab／长度／hidden边界；KL维度和显式feature层索引 |
| PreferenceObjective | chosen和rejected的策略分支；DPO／IPO还检查实际参考模型；SimPO不检查未执行参考模型的配置 |
| GRPOObjective | 实际策略分支及shift后逐token行为／参考logp、逐序列advantages；这里没有隐含reference模型forward |

已审计纯token静态族为Llama、Qwen2、Qwen3、Mistral、Mixtral和DeepSeekV3的明确
原生类型／配置组合。未知wrapper、多模态族、非causal目标保留原有显式协议，不能把
helper无操作返回理解为这些域也已通过此项认证。该名单也不宣称任意TP／PP provider、
自定义hook或设备／编译器故障都可静态预测。

## 数据规则

- 每个完整窗口预先检查IDs／labels范围、物理长度、binary mask、非负int64位置、
  输入／监督shape及设备一致性；训练输入拒绝caller-owned cache/past。
- `inputs_embeds`必须有限、维度匹配；嵌套`model_inputs`不得和顶层输入混用。
  嵌套输入须明确提供顶层labels，不能事后猜另一份监督来源。
- chosen／rejected可有不同token长度，但样本对数相同，每条response有有效监督。
  不要求DPO的策略／参考输出词表维度相同；只分别验证真实输入范围。KL蒸馏则必须
  两个输出分布维度对齐。维度相同本身不证明tokenizer语义相同。
- GRPO的行为／参考logp必须是有限浮点`[B,T-1]`，advantages为有限`[B]`，与监督
  同设备；包括被mask的位置也不允许NaN（NaN乘0仍可污染结果）。每条completion
  必须非空，固定分母模式不得小于有效长度。
- GRPO新增`config_dict()`，把clip、KL、reduction和固定分母纳入新的目标身份及
  checkpoint验证。不事后为缺少配置codec的旧记录补猜配置。

## 验证

`tests/unit/test_token_objective_preflight.py`验证真实分支配置、无forward拒绝、
GRPO分母、DPO／IPO／SimPO独立损失与全参数梯度，以及真实权重更新和恢复配置拒绝。
KD的MSE／cosine／relation特征分支还检查嵌套／顶层完全等价、padding输入梯度为零，
并在改动padding embedding数值后验证特征loss不变。

`tests/distributed/test_token_objective_preflight_dp.py`用真实两个Gloo进程、ZeRO0／3，
在rank1的第二个微批注入错误；rank0两条和rank1第一条始终合法。四个算法共38组
拒绝／更正序列，检查两端零forward、参数／成功更新记录／clock不变，修正后成功
更新；四种ZeRO3目标还检查checkpoint后下一步完整参数逐bit一致。

这些是小模型公式、通信和恢复测试，不是公开benchmark、模型质量或GPU性能成绩。
GPU执行、任意自定义目标、RLOO／PPO及其他多模态目标的完整静态预检不在本包范围。
既有NativeFlash固定源码报告保持原样，本次共享helper重构另行取证。
