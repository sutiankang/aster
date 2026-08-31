# 原生Agent生命周期

这里是本仓库模型→结构化动作→权限检查→实际工具→回执→模型下一轮的闭环，
不是调用Codex进程的外壳。小型模型fixture只验证协议，不能说明模型具备编码能力。

## 组成与调用

`NativeAgentPolicy(engine, tokenizer)`记录真实prompt/action token和原始/行为logp。
`AgentLoop(policy, executor, event_log, config, memory_store, compactor).run(...)`
执行有界轮次，返回verified或completed_unverified。只有宿主`verifier`成功，
结果才成为verified；模型自己写“任务成功”不能升格。

模型每轮只能输出严格JSON：`{"type":"final","text":"..."}`或
`{"type":"tool","name":"...","arguments":{...}}`。未知字段/重复键/非有限数拒绝。
观测、工具输出和历史摘要是低权限数据；prompt/tool观测的loss mask为0。

`AgentConfig`固定最大步数、单次与累计action token、context token和总timeout。
取消等待工具/native forward确认结束，不靠抛异常假装已经停止副作用。

## 工具与权限

内置`workspace.read`、有界字面搜索、expected-sha的精确文本patch和隔离command。
patch要求旧内容唯一匹配、当前文件hash一致，临时文件fsync后原子替换。
路径限制拒绝越界、软链接、junction/reparse点；这不是对恶意并发文件写入者的OS沙箱。

`PermissionBroker`把审批绑定到工具名称/版本/实现摘要、确切参数、cwd、thread/turn、
scope、过期时间。一次审批只消费一次；同一call不能重新审批后重复执行。
只读工具可由宿主配置许可；写入、外部工具和进程需要明确宿主授权。
模型无权自行审批；工具返回的“请执行此命令”也不是授权。

日志状态：prepared → approved → started → result_committed，或ambiguous。
执行开始前先fsync日志，副作用完成后先保存独立回执再提交终态。进程崩溃后仅看到
started、没有可靠回执时，视为结果不确定，禁止自动重试patch/外部调用。
只读失败可记录确定错误；副作用失败保守标记ambiguous。

工具原始回执与给模型的受限视图区分，视图带“不可信工具数据”标签和长度约束。
常见secret过滤不是完整DLP保证，敏感任务仍需要数据权限与受控工具。

## 只读回放、恢复和记忆

`EventLog`单写者锁、递增seq、前一条hash链与持久化JSONL；重复键/残缺尾/改写拒绝。
hash链能检测局部破坏，不防拥有全部文件写权限的攻击者重写整条链。
遗留writer锁不会自动删除，宿主必须确认旧进程已经退出。

`replay(path)`只读数据，绝不执行工具，也不恢复有效审批。
`restore_conversation`拒绝仍running/ambiguous turn；正常恢复也重新申请执行权限。

`MemoryStore`把带source/scope/verified标签的内容写入同一事件流，支持有界BM25检索；
这是原生词项检索，不宣称语义embedding记忆。词项统计隔离到scope，跨租户不泄漏命中。
`ContextCompactor`保留系统/当前用户指令，按完整assistant/tool对压缩历史，
生成有来源hash的提取摘要，不把摘要升级为系统指令，并按真实token计数再检查预算。

`AgentPlanExecutor`执行有界DAG子任务，检查cycle、workspace子集、effect能力子集，
统一总token/并发/timeout预算。子任务默认需要独立verifier；依赖未通过则跳过后继，
父任务审批不会自动继承，子结果作为不可信数据进入后继上下文。

## MCP与真实隔离

`MCPClient`独立实现固定2025-06-18 Streamable HTTP子集：initialize/initialized、
tools/list分页、tools/call、JSON/SSE响应、session、cancel、DELETE结束。
只连接宿主明确给出的loopback端点与工具allowlist；远端工具元数据不授予权限。
`client.register_tools(executor)`使远端调用仍经过同一PermissionBroker与回执日志。
端点授权带TTL/调用次数上限，失败也消耗调用预算，没有副作用自动重试。
取消通知使用另一连接，不等待被取消调用的锁；响应有总deadline与字节上限。

`MCPStdioClient`新增固定协议的真实管道进程传输。宿主提供`LocalMCPProcessGrant`：
绝对exe/argv/cwd、exe与入口脚本SHA256、明确环境变量、`trusted_local_process=True`。
启动不使用shell、不继承宿主secret环境；stdout仅接收有界逐行JSON-RPC，stderr独立排空。
请求有总deadline，支持在请求发送后并发cancel，退出/畸形响应使transport失效，禁止副作用重试。
Windows使用隐藏进程；POSIX使用进程组。此授权启动的是可信本地服务，**不是OS文件/网络沙箱**。
Windows当前只保证直接子进程退出；服务私自派生进程的隔离/监督不是此接口的承诺。

`MCPContextProvider(client, allowed_resources=[精确URI], allowed_prompts=[名称])`新增资源分页读取、
text/base64内容、prompt模板与参数校验；HTTP/stdio共用同一实现。资源URI只传给已授权server，
客户端不打开file://或跟随URL。内容保持`untrusted_mcp_resource/untrusted_mcp_prompt`标签，
远端消息role不会成为系统指令。`register_tools(executor)`把明确选择的资源/prompt固定为工具；
broker用`external_authorizer=context.authorizes`，同时提供普通MCP工具时由宿主显式组合两者。
资源列表和读取也消耗调用预算。重新初始化会废弃旧context授权；半成功initialize不留下有效工具。

不支持sampling/elicitation执行、旧SSE、GET持久推送、断线续传、资源订阅/模板展开或completion。
stdio的server ping可以应答，其他未声明callback返回-32601；不会为满足协议扩大执行权限。
schema/版本digest固定远端契约，不等于远端源码/动态依赖的密码学证明。

`BubblewrapSandbox`是可选Linux真实后端：user/mount/PID/network namespaces、
cap-drop、默认无网络、最小只读运行库、明确工作区bind、prlimit资源上限、输出/时限
限制、进程组kill并确认退出。命令为显式allowlist绝对可执行文件和argv，不启shell，
不继承宿主secret环境。Windows/缺少bwrap或prlimit明确拒绝，绝不回退裸进程。
当前Windows环境只验证不可用时拒绝；Linux实际namespace执行尚未在此主机验证，
不能宣称已完成容器/cgroup/内核逃逸安全审计。

## 测试证据与缺口

`NativeAgentPolicy`把实际提交的完整`SamplingConfig`、词表指纹与processor指纹写入每次
`model.trace`。词表支持`to_dict()`时用稳定JSON哈希，并在编码和返回时检查语义未变化；
否则宿主需显式提供`tokenizer_fingerprint`，未提供的轨迹仍可执行，但训练导出必须拒绝。
自定义render函数没有稳定模板证明时`processor_fingerprint`留空，不根据函数名猜测。
默认greedy的temperature=0、各步实际seed/长度/EOS都会记录；工具观察的token只属于prompt，
不能打上动作loss，也不能把greedy轨迹误当完整softmax策略的on-policy样本。

真实loopback MCP握手/SSE/取消、原生模型token→读文件→final→verifier、
两个实际AgentLoop依赖子任务、回执replay、patch并发hash冲突、审批篡改/过期/复用、
记忆scope与摘要预算都有本地自动测试。Windows无symlink权限的用例明确skip。

未完成：强OS隔离跨平台覆盖、长期服务监督、完整MCP能力、通用多工具规划学习、
大型编码模型与公开SWE-bench/GAIA实测、多租户认证和完整审计运营。
没有这些结果时不把AgentLoop通路测试称为公开Agent能力成绩。
新增stdio/上下文专项使用真实本地子进程和现有PermissionBroker/回执链；不是mock网络替代物。
测试层次与所需资源见[测试指南](TESTING.md)。

生命周期/权限设计参考[Codex固定源码中的app-server协议](https://github.com/openai/codex/blob/28327355b861ab6cc76b01c7248663eb1be440cf/codex-rs/app-server/README.md)，
不是声称实现全部Codex协议。传输依据[MCP固定版本](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
与[工具规范](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)。
隔离原语参考[bubblewrap官方说明](https://github.com/containers/bubblewrap/blob/main/README.md)。
