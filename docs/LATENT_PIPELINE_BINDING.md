# 潜空间管线：采样日程与真实训练来源

`LatentGenerationPipeline`的离散扩散现在在构造时固定完整原始beta和模型时间映射，
推理只对这条原链respacing；保存后不再凭名字和采样steps创建一条不同训练链。
这遵守[OpenAI原始时间映射/respacing工程](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/respace.py)
的数学语义，但本接口的均匀索引取整规则显式保留为原有linspace-round，未伪称支持
所有官方ddimN选点字符串。

## 两类来源不能混淆

普通构造器仍可用于实验，`diffusion_schedule=`可显式传本仓库DiffusionSchedule。
来源分别记录为 `caller_config` 和 `caller_explicit_schedule`，都明确
`training_semantics_bound=False`。手动声明不能变成“已验证训练参数”。
构造器复制beta/map，不保留传入对象的Tensor引用；外部修改该对象不改变管线。

```python
pipeline = LatentGenerationPipeline.from_artifacts(
    store, autoencoder_artifact_id, field_artifact_id,
    LatentPipelineConfig(method='diffusion', solver='ddim', steps=50, diffusion_steps=1000),
)
```

上述训练绑定入口要求field制品保存实际 `LatentFieldObjective.config_dict()`：
`encoder_identity`必须等于VAE内容寻址ID，内部目标必须与管线方法一致。
并且必须保存 `successful_update.json`，对应Trainer真正最后一次成功phase的目标
class/codec/config和角色更新时钟；缺失的legacy声明不能进入训练绑定入口。
普通VAE/field实验仍可显式用caller-owned构造器，不会替它自动补造训练证据。
普通像素目标不自动推断为此VAE坐标下训练；beta/map、学习方差、flow方向、EDM
sigma_data不匹配时拒绝。模型参数化和输出通道也预先检查。
这核验内容和声明的一致性，不是密码学上的训练历史/数据合规证明。
最后成功更新记录也不证明所有早期步骤采用同一目标；全过程约束需要单独历史策略。

原链长度指实际betas数组长度，而非模型time的最大值；已respaced训练链可以有
非连续timestep_map。采样选的是原数组索引，模型收到对应原map中的time。
DDIM eta、clip_clean、learned_variance均显式保存并实际传给sampler；DDPM不能
携带被忽略的非零eta，flow/EDM也不能携带未消费的离散扩散控制参数。

## 保存与恢复

`pipeline.json`升级schema2，包含完整原始日程、原链hash、实际respacing索引、
来源声明、配置身份和潜空间scale/shift。训练绑定的保存还包含实际field/VAE权重
指纹；加载、采样、再次导出会核验，改过权重后须显式创建新来源，不能沿用旧ID。
此完整CPU指纹检查偏向正确性，不能作为高速生产服务的性能实现；生产服务可在
不可变artifact加载边界验证后由专门runtime持有，不能静默关闭来源校验。

schema1缺少实际原链，明确拒绝自动升级。迁移时用户必须选择重新从训练制品加载，
或按旧配置显式构造caller-owned新管线；不得把重新生成的日程称作原训练事实。
保存只包含推理管线，Trainer完整checkpoint负责优化器、训练RNG和数据游标恢复。

集成测试覆盖真实VAE冻结/field训练后checkpoint恢复，非连续model time、原链
respacing、学习方差/DDIM随机eta、caller对象修改隔离、保存逐值恢复，以及错误
VAE/像素目标/参数化、元数据篡改、权重变更、legacy schema和非有限配置拒绝。
