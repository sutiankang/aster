# 冻结接口 v1

以下是新实现协作契约，不是全部算法的伪代码。核心依赖为Python标准库、PyTorch、NumPy；默认执行不导入Transformers/DeepSpeed/Megatron/vLLM/Diffusers/TRL/LeRobot。

## 模型

所有模型是`nn.Module`，有纯数据`config.to_dict()`，本地`save_pretrained(path)`/`from_pretrained(path)`。读取外部checkpoint须安全格式或`weights_only=True`，不得自动执行remote code。

- `TokenOutput(logits, state=None, hidden_states=None, auxiliary=None)`。
- `TokenPredictor.forward(input_ids=None, *, inputs_embeds=None, attention_mask=None, position_ids=None, state=None, use_cache=False, output_hidden_states=False, **declared_modality_inputs) -> TokenOutput`。只接受显式支持的模态字段，拒绝未知参数。
- `FieldOutput(prediction, prediction_type)`；`ConditionalField.forward(sample,time,condition=None) -> FieldOutput`。
- `RepresentationEncoder.encode(inputs, **declared_metadata)`返回表示与有效位置。
- `LatentDynamics.observe/imagine/step`输入显式episode与latent状态，不持有全局环境。
- `ActionPolicy.predict_chunk(observation,state=None)`返回动作块与有效mask，ActionSpec固定单位/坐标/执行horizon。

共享核心在`aster.core`；算子在`aster.nn`，模型在`aster.models`。配置按模型家族单独dataclass，不制作能接收任何键的巨型config。模型与state不持有optimizer或进程组全局单例。

## 训练

`LossTerm(numerator, denominator, unit, name="loss", weight=1.0)`；分子是标量tensor，分母是非负无梯度标量。`LossBundle(terms)`用于不同单位目标，不能把分母相加后统一除。

`Objective.forward(model,batch) -> LossTerm | LossBundle`是单角色便利协议。多角色Method显式提供phase：角色、optimizer key、输入batch、目标回调、detach/freeze策略、更新顺序；训练engine拥有实际反向/归约/更新。不得在模型forward更新EMA/router bias/replay等跨步状态。

`Trainer(model, objective, *, optimizer=None, lr=..., device="cpu", accumulation_steps=1, max_grad_norm=1, precision=...)`提供`step(microbatches)`/`fit(batches,steps)`/`evaluate(batches)`/`save_checkpoint(path)`/`load_checkpoint(path,trusted=False)`。完整快照只在声明的逻辑边界提交。

角色方法可调用同一个engine phase接口；PPO/SAC/KD各自不再维护孤立checkpoint格式。单角色便利接口也返回具有sample/token等单位的loss记录。

## 制品与工作流

`ArtifactStore(root).publish(directory, *, kind, metadata, parents=()) -> Artifact`。

`ArtifactStore.get(artifact_id, verify=True) -> Artifact`。

`Artifact`有`id/path/kind/metadata/parents`；制品含所有输入语义，不仅一个weight文件。`publish`复制成独立不可变快照，不允许软链接/重解析点逃逸。

`StageResult(artifacts:dict[str,str], metrics:dict[str,float], details:dict)`；工作流依赖制品ID，不能依赖可变外部目录别名。阶段重入必须校验配置、代码/来源、输入与输出哈希。

## 推理

ModelRunner提供明确的模型种类和状态codec。token runner调用上面的TokenPredictor，服务不假设state一定是KV tuple。初始KV codec可支持tuple，但必须按tag选择；循环state回滚需checkpoint/replay能力。

请求绑定`policy_artifact_id`。输出记录`token_ids`、`raw_model_logprobs`、`behavior_logprobs`、sampling transform顺序、接受的draft token及计数、stop reason、单调时间戳。decoder/processor由制品提供。

基础采样/调度/页管理在`aster.inference`；不调用官方服务进程充当主实现。

## Kernel ABI

`KernelSpec`固定op/provider/version/device/dtypes/layouts/masks/backward/workspace/side_effects/tolerance。provider不得拥有optimizer、模型下载器、server或自动依赖安装。参考数学、参考存储、加速kernel分别标记。每次运行记录实际provider，不静默降级。

## Agent

事件和审批用标准库实现；单写者日志。工具生命周期`prepared/approved/started/result_committed/ambiguous`。权限broker先验证工具版本、参数摘要、cwd、scope/expiry，再让executor执行。只读replay不执行任何工具；resume重新检验权限。

## 评价

`ComparisonProtocol`固定数据/任务/评测器/控制变量；`EvaluationRun`记录candidate artifact与变换、硬件、原始结果及完整分母。评价既可本地指标也可官方执行器。没有原始证据不能将status置为verified；缺失样本和NaN不得静默丢弃。
