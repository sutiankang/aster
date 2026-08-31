# 跨领域任务评测接口与证据边界

`evaluation.suites`是原生回合/Agent执行；`evaluation.adapters`允许官方任务环境或评测器。
它们复用`ComparisonProtocol/EvaluationRun`，不修改训练或模型实现，也不把公开库名称
放进注册表就称为“已经跑过”。当前没有LIBERO、LMMS、SWE-bench完整公共成绩。

## 协议与批准

每个比较协议固定：数据内容fingerprint/revision/split、任务与样本全集、seed、
评测器版本/源码hash、预处理/控制频率/horizon、成功判据、主指标方向及failure score。
候选artifact ID只进入运行记录，不混入比较协议，否则候选和基线永远无法同协议比较。

`EvaluationGrant(protocol.id, effects, expires_at)`由受信宿主构造，有协议范围和有效期。
环境、Agent、官方评测器、不可信代码、Docker分别授权；模型输出不能生成许可。
这不是OS隔离。官方依赖应运行在预先准备并审核的环境，网络访问/数据与模型许可/费用
由宿主确认。默认调用不会安装依赖或下载权重。

`OfficialModulePin`核验发行版本、完整commit、实际模块文件sha256和协议中的源码hash。
它不是整个Python依赖闭包的密码学证明，不能替代可信环境与供应链审核。

## 实际控制回合

`episode_protocol(cases, ...)`把每项`EpisodeCase(id,task_id,seed,max_steps,initial_state_id,options)`
写入协议。`evaluate_episodes`每回合新建环境与策略状态，执行真实reset/step、累加回报、
记录动作/观测hash/终止/截断/成功轨迹，始终关闭环境。policy回调须声明真实
`policy_artifact_id`；成功由外部判据决定，不能由策略自己声称。

当前成功聚合为`any_post_step`，horizon、timeout语义都在协议中。step阻塞期间不能
强杀同进程环境，故timeout明确是step之间的合作式限制；需要硬时限的环境须由独立
受监督进程实现，不可把它宣称为已完成。崩溃、NaN、关闭错误计failure；未完成/缺失样本
不会被删除。对可能为负的环境回报应选择明确保守的failure score，不默认用0获益。

`GymnasiumFactory`调用真实`gymnasium.make`与五元组接口，并核验版本；拒绝
`module:Env`形式的隐式模块注册。`LiberoFactory`调用官方benchmark/OffScreenRenderEnv，
核验任务BDDL、初始化状态内容和显式相机尺寸，使用`check_success()`，不把done当成功。
MuJoCo、渲染器、robot控制频率、动作归一化与数据许可由评测配方固定。

它们适用于VLA/RL/world-model控制策略的真实环境回报/成功率。RSSM想象奖励、像素预测
PSNR/FVD等不能代替环境成功率；预测任务与控制任务应分别建协议。多seed比较应保留每个
task×seed×初始状态样本，再用配对区间/IQM汇总，不先挑选最好seed。

## VLM官方任务

`evaluate_lmms`实际调用`lmms_eval.evaluator.simple_evaluate`，桥接本仓库生成/likelihood
回调，官方定义负责文档→问题/图像/目标及判分。Task对象必须固定dataset revision、
实际split fingerprint和完整doc_id，`limit=None`；原始results/samples独立保存。
仅实际逐样本标量指标进入统一paired记录，不能把聚合MMBench分数复制给每个样本。
多轮任务、未实现的视觉likelihood和非标量指标明确拒绝或记录缺失错误。

MMBench/VQAv2/MMMU等需要各自题目split、prompt、选项解析、图像尺寸和多语言协议。
开放式模型judge须另外固定judge版本/模板/采样seed、保存原始判定、考虑judge偏差与费用；
没有批准不连接收费模型。公开测试污染和训练数据重叠需随模型卡披露，不能仅凭分数断言
泛化能力。LMMS适配已实现，但没有在本环境运行这些官方数据集。

## Agent与代码验证

`evaluate_agents`要求真实`AgentLoop`，核验policy artifact、任务工作树fingerprint、seed与
verifier ID，再实际执行模型/工具回合。任务树外保存日志和回执，避免污染工作树快照。
独立verifier成功才计resolved；预算耗尽、执行错误、丢失结果都保留分母。

`SandboxTestVerifier`要求实际只读、无网络的Linux BubblewrapSandbox，先后核验冻结测试
文件hash，依据真实测试进程exit_code/可选stdout摘要判定，返回可追溯回执hash。
没有隔离环境就拒绝；不会用一个`isolation=True`字符串或普通subprocess冒充安全执行。
当前主机为Windows，未验证Linux隔离中的真实测试执行。

`evaluate_swebench`可真实调用官方`harness.main`，只用内容固定的本地JSON数据和候选
JSONL patch，要求当前harness的容器image digest，避免随tag漂移；Docker与不可信测试
代码分别授权，禁用云运行。官方原始report归一化保留resolved/unresolved/error/
empty-patch/missing等情况。SWE-bench大仓库构建、镜像拉取、磁盘和CPU资源成本较高，
需要先准备允许的镜像、runtime及网络策略。接口存在不代表此处跑过公共SWE-bench。

GAIA/BFCL等未在这个模块伪造计分实现：它们仍需各自工具环境、最终答案标准化、
工具调用验证与公开数据许可。共享Agent回合原语不等于这些榜单已经复现。

## 当前可验证事实

本地自动测试实际执行小型控制环境步进/失败分母/轨迹保存，以及原生token模型→
workspace工具→外部verifier的Agent回合；这些报告标记为protocol fixture，非公开成绩。
官方结果parser测试只验证缺失/错误计分，不伪装为实际执行官方harness。
目前没有为评测安装大型依赖、下载公开测试数据、驱动机器人或启动Docker/外部收费服务。

参考官方接口：[Gymnasium Env](https://gymnasium.farama.org/api/env/)、
[LIBERO task与初始化示例](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/README.md)、
[LMMS evaluator](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/evaluator.py)、
[LMMS模型接口](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/api/model.py)、
[SWE-bench评价说明](https://www.swebench.com/SWE-bench/guides/evaluation/)。
这些链接用于接口出处；每次真实运行仍必须提供固定revision与实际源码hash，不把main当锁。
