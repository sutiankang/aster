# MeanFlow 与 Shortcut：原生少步生成训练

实现依据作者的 [MeanFlow](https://github.com/Gsunshine/meanflow/blob/main/meanflow.py)、
[MeanFlow DiT](https://github.com/Gsunshine/meanflow/blob/main/models/models_dit.py)、
[Shortcut targets](https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py)和
[Shortcut DiT](https://github.com/kvfrans/shortcut-models/blob/main/model.py)。本轮实读日期
2026-08-30，运行时不导入这些仓库。两者共用区间条件DiT组件，但关键差异明确记录：

| 契约 | MeanFlow | Shortcut |
|---|---|---|
| 时间方向 | data=0 → noise=1；采样逆向 | noise=0 → data=1；采样正向 |
| 网络第三输入 | 区间长度`t-r` | `-log2(dt)`，不是dt |
| attention缩放 | `1/sqrt(head_dim)` | 作者实现的`1/head_dim` |
| block调制初值 | 全零 | Xavier；只有最终层调制/输出全零 |
| 核心监督 | 精确JVP平均速度恒等式 | 两个半步的停止梯度目标 + FM锚点 |
| 损失归约 | 每图像元素求和，再自适应加权 | 每图像元素平均 |
| guidance | 训练目标内guidance，含kappa和时间窗口 | 最细自举层选定样本CFG；推理可显式CFG |
| 推理更新 | `x -= (t-r)*u(x,t,t-r)` | `x += dt*u(x,t,-log2(dt))` |

`IntervalDiTConfig(variant=...)`固定上述模型差异，输出类型为`average_velocity`。
不能交给只接收瞬时`velocity`的普通ODE sampler，或把两个方法仅通过名字互换。

## MeanFlow

```python
from aster.models.interval_dit import IntervalDiTConfig, IntervalDiT
from aster.methods.meanflow import MeanFlowObjective, sample_meanflow
from aster.training import Trainer

model = IntervalDiT(IntervalDiTConfig(variant='meanflow'))
engine = Trainer(model, MeanFlowObjective(), lr=1e-4, ema_decay=.999)
# sample为真实BCHW图像/潜变量，labels为对应类别；时间和noise也可显式提供。
# engine.step([{'sample': sample, 'labels': labels}])
# images = sample_meanflow(deployed_model, noise, labels=labels, timesteps=(1., 0.))
```

路径为`z_t=(1-t)*data+t*noise`；JVP输入切向量为`(v_guided,1,1)`，第三项来自
`h=t-r`对t的导数。漏掉这一项会改变目标。`torch.autograd.forward_ad`在no-grad上下文
计算精确JVP，另一次普通前向保留参数梯度。不是有限差分，也不是通过反向图求高阶
梯度。此分离实现多一次primal forward，但无需完整模型拷贝，已与原生ZeRO3逐层
物化配合。模型没有随机dropout，两个primal使用相同权重/输入。

采样时间支持logit-normal/uniform，先排序，再将当前microbatch前`floor(B*p)`个
样本设r=t。类别丢弃使用作者的随机数量、固定前缀规则。自适应权重分母停止梯度；
所有计数为int64。默认采样器不做两分支CFG，训练guidance已经进入学到的平均速度。
默认的时间/类别抽样按当前microbatch作用；改变DP/accum布局不会保持相同随机样本序列。
若要严格比较完整batch与不等长microbatch，显式提供noise/time/reference_time/drop_count。

## Shortcut

```python
from aster.methods.shortcut import ShortcutMethod, sample_shortcut

model = IntervalDiT(IntervalDiTConfig(variant='shortcut'))
engine = Trainer(model, lr=1e-4, zero_stage=0, accumulation_steps=1)
method = ShortcutMethod(engine, base_steps=128, bootstrap_every=8,
    bootstrap_ema=True, ema_decay=.999, bootstrap_cfg=True, cfg_scale=1.5)
# method.update([{'sample': sample, 'labels': labels}])
# images = sample_shortcut(deployed_model, noise, labels=labels, steps=4)
```

bootstrap与flow的配比由`bootstrap_every`固定；每个microbatch大小必须是它的正整数
倍。全DP、全accum窗口只生成一份官方步长分层，按实际局部样本数切分，避免小rank
各自舍入后只学到最大步长。数据置乱按rank/microbatch独立执行，不能声称和上游单个
全局随机置乱逐位相同。分层频数、条件目标和损失公式有独立测试。

数据路径保留1e-5端点噪声；自举用两次半步，第二半步状态和最终平均速度clip[-4,4]。
选中的最细层CFG把有条件与追加null样本合并成一次前向，因此每个半步一次模型调用。
EMA可关闭，此时两个半步使用更新前模型。EMA目标是单独冻结角色，不拥有优化器。
当前冻结EMA角色持完整权重，成功step后通过完整权重导出更新，存在额外峰值内存；
这不是所有角色参数均完全分片的声明。EMA阶段中断后不得保存半轮状态。

## 验证与限制

独立函数覆盖两种DiT前向及全部参数梯度；解析场测试同时验证MeanFlow的三项JVP、
guidance和停止目标梯度，以及Shortcut半步、clip、分层和CFG。真实DP2测试覆盖不等
rank批量、ZeRO0–3、模型/EMA与精确恢复；MeanFlow直接对照相同完整batch的梯度范数
和更新，Shortcut比较各ZeRO布局，局部随机置乱序列固定。

CPU FP32/BF16、单机多进程已经验证；没有GPU/NCCL性能结论。MeanFlow两类微型数据
120更新的一步生成MSE约从1.20降到0.069，仅作为可学习回归，不是ImageNet FID。
Shortcut尚无公开质量复现。生产权重映射、iMF/pMF、随机dropout变体、整条真实数据
训练配方、模型TP/PP和已测GPU优化仍未完成。上游unused logvar诊断头不参与本实例的
预测/训练目标，因此省略；不把这个选择包装成上游所有诊断输出完全覆盖。

生成制品必须包含variant、训练guidance、区间语义和采样步数；在相应制品消费者完成
验证前，只使用上述显式sampler，不强塞到通用flow或DMD消费者。
