# 原生模型层：结构、状态与证据

本目录默认执行纯 PyTorch 计算图。`transformers` 仅在 `tests/parity` 中充当可选 oracle；模型构造、前向、存储和推理不调用它，也不下载权重或远程代码。

## 一个 factory，不是一个万能模型

```python
from aster.models import build_model, load_model, LlamaConfig

model = build_model(LlamaConfig(vocab_size=128, hidden_size=64,
    intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2))
output = model(input_ids, attention_mask=valid_mask, use_cache=True)
next_output = model(next_ids, state=output.state, use_cache=True)
model.save_pretrained("local_model")
restored = load_model("local_model")
```

配置必须明确架构。未知类型、未实现字段和不支持的状态操作直接报错；不允许任意 `kwargs` 静默丢弃差异。模型不内置 optimizer、下载器、HTTP 队列或训练损失，`TokenOutput.logits` 供 SFT/KD/RL/推理共享使用。

`forward(input_ids=None, *, inputs_embeds=None, attention_mask=None, position_ids=None, state=None, use_cache=False, output_hidden_states=False)` 是因果语言模型的公共接口。mask 中 1/True 表示有效；使用缓存时二维 mask 覆盖完整过去和当前位置，即使窗口缓存已裁剪。`position_ids` 只描述本次 token。训练默认不建立缓存。

## 已落盘的真实结构

| 架构 | 与基础解码器的实质差异 | 证据和边界 |
|---|---|---|
| Llama | RMSNorm、split-half RoPE、GQA、SwiGLU | 同权重前向、全部参数梯度、增量缓存、存储重载 |
| Qwen2 | Q/K/V 有 bias，可按层选择滑动窗口 | 独立配置/真实参数，不是 Llama 名称别名 |
| Qwen3 | 每头 Q/K RMSNorm、可声明 local/global 层 | 同权重主路径；不包含 Qwen3-VL 或 Qwen3.5 |
| Mistral | 滑动窗口，包括 prefill 后裁剪 | 跨窗口缓存/不可直接回滚语义 |
| Mixtral | top-k softmax router、稀疏专家矩阵、归一化路由 | 与官方 packed 参数布局对应，计算实际选中专家 |
| DeepSeekV3 | MLA、独立内容/RoPE 维度、query LoRA、分组路由、共享专家 | `absorbed`/`expanded` 值和梯度对照；缓存确实是压缩 latent |
| BERT MLM | 绝对位置/segment embedding、双向注意力、post-LN、GELU、MLM 变换头 | 非因果编码器；拒绝 append-only cache；不包含 NSP/分类任务头 |
| T5 | encoder/decoder、相对距离桶、无 QK 点积缩放、跨层共享偏置、交叉注意力 | ReLU/gated-GELU/gated-SiLU；独立 seq2seq 状态与 decoder shift-right |
| CLIP vision | patch Conv、CLS、绝对二维位置、QuickGELU、CLS pooler norm | 视觉塔/位置插值/像素梯度；不是完整 CLIP 文本塔与对比学习 |
| LLaVA 固定网格 | 选视觉隐藏层、MLP 投影、逐样本图像占位替换、共享文本 decoder | 单/固定 N 图张量；官方 Llama+CLIP 路径同权重；不是 AnyRes/LLaVA-NeXT |
| Qwen3Next | GatedDeltaNet、部分 RoPE gated attention、zero-centered norm、门控共享专家 | 原生递推与官方 chunk 公式及梯度对照；包含 dense/MoE 配置；不包含 MTP |
| Gemma3 text | embedding 乘 sqrt(H)、四个分支 norm、local/global RoPE、可选输出 softcap | 因果文本分支；不是 Gemma3 多模态或 Gemma4 |
| Llama4 text | RoPE/NoPE 层、复数偶奇旋转、固定块 attention、长位置温度、输入侧专家路由缩放 | Scout/Maverick 共用公开文本公式的可配置分支；不包含视觉塔 |
| Qwen3-VL | Conv3D时空patch、merge顺序、packed视觉attention、位置插值、DeepStack、交错MRoPE | 完整图像路径同权重/全梯度/cache；视觉多帧对照；不是旧Qwen2-VL别名 |
| Kimi K2.5 | MoonViT时空位置、xy-RoPE、全clip注意力、temporal pooling、保留子patch的MLP连接器、MLA文本 | 完整图像路径与视频视觉塔对照；官方当前混合帧数/完整视频split限制见下 |
| Kimi K3 text | 按key通道遗忘的KDA+安全gate、NoPE/输出门MLA、深度AttnRes块bank、SiTU双支路软限制、Stable LatentMoE | SGLang/FLA公开公式独立前向/全梯度、block1/2、padding/缓存、ZeRO0/3单rank真实训练和精确恢复；未运行SGLang GPU整包或载入MXFP4发布权重 |
| Kimi K3 image/video | 专用MoonViT-V2宽QKV、complex偶奇xy-RoPE、无bias、dtype-epsilon RMS、bilinear位置、时间pool、无preNorm连接器+postRMS | 完整图文公式全部梯度；vision FP32/BF16对照，BF16同数值路径逐位一致；冻结LM视觉反传、ZeRO0/3单rank15步训练/精确恢复/缓存/重载 |
| SigLIP | 无CLS视觉塔、所有patch末端LN、learned-query池化、双向文本末格pool、可学习温度/偏置 | 真正双塔相似度与全部梯度对照；patch-only可接Pi；不是CLIP文本塔 |
| Janus | 理解视觉与生成视觉解耦；视觉MLP连接器、原生VQ codec、生成embedding/aligner/head、共享Llama | 完整理解路径与image-code head/cache对照；VQ编码/离散decode/全梯度；不是JanusFlow |
| GPT-2 | learned absolute positions、Conv1D `[in,out]`权重、GELU-new、残差深度初始化、可选QK upcast/层缩放 | 因果主干严格同权重/全梯度/cache；不含cross-attention微调变体或分类头 |
| Mamba-1 | causal depthwise conv、输入依赖dt/B/C、selective SSM、门控与skip、fp32 residual | 完整序列及单步与官方对照、原生分块继续一致；不是Mamba2 SSD/Mamba3 |
| LLaDA | 官方独立参数布局、双向RoPE/GQA、RMS/SwiGLU、Mitchell初始化、同位置mask预测 | 原作者源码核对；与官方Llama显式双向mask子图对照，不冒称执行原作者旧运行时 |
| DeepSeek-V3.2 | MLA上接lightning indexer、half-split index RoPE、ReLU加权打分、top-k稀疏集合 | strict官方主干/梯度/cache对照；indexer另用teacher-attention KL训练；不是FP8/kernel吞吐对齐 |
| DeepSeek-V4 | mHC多残差流、共享K=V的MQA、attention sink/逆RoPE、分组低秩输出、CSA/HCA闭窗压缩、sqrtsoftplus/hash-MoE | strict官方默认/YaRN同权重、全梯度及跨闭窗cache；不含FP4/FP8、MTP或ragged压缩kernel |
| GR00T N1.7 | Qwen3-VL截断主干、独立embodiment三组MLP、text-cross/self/image-cross/self DiT、离散时间条件、RTC衔接 | 原作者源码公式与独立functional全部梯度对照；TF视觉语言组合oracle；不是π别名，不冒称已执行整个官方Diffusers环境 |
| Gemma4 text | PLE逐层token/context支路、global/local不同head尺寸、value RMS、proportional RoPE、后续层真实KV共享、dense+MoE双分支及expert scale | 已运行官方dense/MoE/无共享/无PLE四配置全部梯度与跨window cache |
| Gemma4 image/video | 独立二维频谱RoPE、可裁剪线性层、按xy池化、FP32标准化、无scale RMS视觉投影、PAD identity PLE、local-only视觉双向mask | 真实TF视觉与完整图像+视频+文本oracle全部参数/pixel梯度、完整视觉前缀后缓存续算；不包含音频或Unified分支 |
| BLIP-2 | 真实packed-QKV ViT、BERT post-LN Q-Former、learned query定长压缩、query-only交叉注意力与独立文本FFN、语言投影 | Q-Former/vision官方子图；T5和Llama两种完整官方图文模型全部梯度/缓存；共享Trainer训练、冻结LM反传及随机续跑。尚未实现OPT或第一阶段整套ITC/ITM生成预训练配方 |
| Qwen3.5 dense/MoE | 独立QKV/Z/B/A投影的DeltaNet、partial interleaved MRoPE、无DeepStack的视觉塔、混合循环图文cache | dense/MoE文本与完整图像同权重/全梯度/cache；官方说明Qwen3.6共用architecture，未验证下载权重 |
| Qwen MTP | embedding/前级hidden双RMS后拼接FC、每步按索引循环选择一层full attention、共享embedding/head | 实际TF dense/MoE主干与decoder子图全部梯度；vLLM公开MTP连接公式独立重述；联合/冻结训练、独立draft cache、别名保存恢复。未执行vLLM服务整包 |
| Qwen3.8-Flash-Next / Qwen4Exp | GDN+实际可见token微块QSA、四流GatedResidual、EOS重置N-gram PLE、独立图文状态 | 锁定官方新增源码的独立全梯度公式、TF真实GDN/视觉子图、训练+索引器KD+缓存+重载；不是Qwen3.5别名，不含GPU稀疏kernel/异步词表prefetch |
| DeepSeek-OCR2 | SAM窗口/全局网格→独立Qwen2因果阅读queries→local/global/separator→非MLA softmax-MoE语言 | SAM/Qwen2真实TF子图全梯度，语言真实TF attention+原作者MoE公式；共享训练/续跑/文档生成；不冒称已运行原作者旧TF/torchvision整包或取得OmniDocBench成绩 |
| Cosmos3 MoT | 因果理解路与双向生成路独立参数，视频/声学latent/动作联合attention，3D interleaved mRoPE | 独立锁源码全参数公式、真实TF理解子图、联合训练/采样、DP2 ZeRO0–3；无波形AVAE或发布质量声明 |
| Cosmos-Predict1 | FA/CA/MLP各自AdaLN、共享AdaLN-LoRA、FPS/NTK 3D RoPE、Kendall不确定性头 | 原生GeneralDIT/可训练net+logvar组合，独立公式全梯度、真实Euler与DP2；非Wan，未宣称原包TE/CUDA执行 |
| DINOv2 register vision | CLS+4register、无register位置、LayerScale、GELU/pre-LN、指定block原始patch | timm布局到官方TF映射（CLS位置折叠）、全梯度；固定网格，不含DINO自监督训练recipe |
| OpenVLA原版 | DINO-reg4+SigLIP倒数第二block原始特征融合、3FC projector插BOS后、尾词表action bins、quantile mask反归一 | 训练→cached greedy动作→保存重载；官方子模块组合oracle，未执行原包timm0.9.x/TF4.40.1环境 |

统一 factory 还接入其他施工包的 `UNet2D`、`DiT`、`AutoencoderKL`、`RSSMWorldModel`、`ACTPolicy`、`DiffusionPolicy1D`、`PiActionExpert`、`PiVLA`、`JEPAEncoder`、`JEPAModel`。它们保留 `FieldOutput`、世界状态、动作或特征输出协议，不被强行包装成 token logits；其算法证据见对应测试与方法文档。

上述表不是“所有规模/硬件/预训练权重已验证”的声明。默认 tiny 配置验证数学与工程路径，不代表已达到公开大模型的基准分数。

## 状态不是统一的四维 K/V

| 状态类型 | 每层载荷 | 安全操作 |
|---|---|---|
| `KVState(kind='dense_kv')` | K/V `[B,Hkv,S,D]` | fork、reorder、truncate |
| `KVState(kind='window_kv')` | 最后 W-1 个 K/V；不同层可有不同窗口 | fork、reorder；回滚必须 checkpoint+replay |
| `KVState(kind='mla_latent')` | latent `[B,1,S,rank]` 和 RoPE key `[B,1,S,rope_dim]` | fork、reorder、truncate；两张量最后维度不同 |
| `Seq2SeqState` | self K/V 与固定 encoder cross K/V；独立 encoder 条件/mask | 仅 self 时间轴 truncate，条件不能被后续调用替换 |
| `HybridState(kind='hybrid_delta')` | full 层 K/V；linear 层 conv `[B,C,K-1]` + memory `[B,Hv,Dk,Dv]` | fork、reorder、checkpoint+replay；循环 memory 不能 truncate |
| `VisionLanguageState(kind='qwen3_vl_kv')` | decoder KV + 每请求独立 `[B,1]` MRoPE delta | fork、reorder、checkpoint+replay；截断跨视觉片段不能只切KV |
| `MambaState(kind='mamba_ssm')` | conv `[B,I,K-1]` + selective memory `[B,I,N]` | fork、reorder、checkpoint+replay；两者均没有可截断的token轴 |
| `IndexedMLAState(kind='indexed_mla')` | latent、RoPE key、index key三张量，均 `[B,1,S,D]` | 三者同步fork/reorder/truncate；不同末维不能混装 |
| `CompressedAttentionState(kind='compressed_window_mqa')` | rolling `[B,1,W-1,D]`；compressor/indexer各自的完成条目、未闭窗投影、CSA前窗Ca | fork/reorder、checkpoint+replay；不能按dense KV规则截断/打包 |
| `Qwen35VisionState(kind='qwen3_5_vl_hybrid')` | HybridState与每请求MRoPE delta | fork/reorder、checkpoint+replay；不是Qwen3-VL的dense KV |

所有模型 forward 都以新张量返回状态，不原地更新调用方状态。`model_key` 锁定配置；外层推理服务还必须绑定权重制品版本，配置哈希不是权重身份。循环 state 的数值维度不能当成 sequence axis；状态编解码器必须按类型/能力调度。

T5 第一次调用：`model(encoder_ids, decoder_input_ids=decoder_ids, attention_mask=encoder_mask, use_cache=True)`。之后传 `decoder_input_ids` 和 `state`，不能重传另一组 encoder 条件。`shift_right(labels)` 仅构造 decoder 输入；原 labels 和有效位置仍由外部方法管理。当前官方 T5 配置把旧 `tie_word_embeddings` 参数用于历史缩放兼容并实际共享词嵌入，本实现将独立输出尺度显式命名为 `scale_decoder_outputs`，避免误解为解绑权重。

## 图像与共享连接器

`CLIPVisionModel(pixel_values, output_hidden_states=True)` 返回视觉 `last_hidden_state`、CLS `pooler_output` 和隐藏层列表。`normalize_clip_pixels` 接受已经完成几何变换的 RGB BCHW，uint8 按 255 缩放，float 必须在 [0,1]。resize/crop 不自动发生，必须记录在数据配方中。非标准分辨率需要显式 `interpolate_pos_encoding=True`。

`LlavaForConditionalGeneration` 用相同 decoder 执行器接上视觉输出；图像 token 数量逐样本核对，而不是仅核对全 batch 总数。图像注入只发生于 prefill；增量 decode 只传新文本和 state。embedding-only 图文输入必须传显式 `image_token_mask`，不能靠浮点向量相等猜测占位位置。标准 LLaVA 配置默认删除视觉 CLS token；`full` 和多视觉层拼接也是独立测试分支。

## 多模态与循环/稀疏模型补充契约

`Qwen3VLForConditionalGeneration` 要求图像/视频的显式 `grid_thw` 与 `mm_token_type_ids`，视频时间戳是文本token，视觉块和grid逐段核对。`pack_qwen_pixels` 保留merge邻域顺序，不能改成普通raster flatten。MRoPE delta存在每个请求state，而不是model的可变全局字段。

`KimiK25ForConditionalGeneration` 的 MoonViT 与 Qwen视觉不同：同clip所有帧共同注意力，先时间平均再保留空间kernel子patch给projector。已对照完整图像路径和分别固定T的多帧视觉路径。Transformers 5.16.1 的temporal-merge索引拼接不能混合不同T，完整多帧 `get_image_features` split计数也与时间平均后的长度不符；本地严格按时间平均后的空间token数执行，但不能称这些上游失败路径已取得full-model parity。

`SigLIPVisionModel(...).last_hidden_state` 是可复用的patch特征，`vision_use_head=False` 不生成无用pooler。`SigLIPModel` 输出双塔单位向量、`logits_per_text`与转置图像logits；外部方法实现逐pair sigmoid目标。文本固定max_length的最后一个格子就是pool位置，即使该位置是padding，也不能自行改成最后非padding token。像素标准化为`(RGB/255-0.5)/0.5`，几何处理仍由数据配方声明。

`JanusForConditionalGeneration` 通过 `output_kind='image_codes'` 使用生成head，`prepare_embeddings_for_image_generation`把离散图像ID变为共享语言主干的输入。`decode_image_tokens`按统一图像协议返回BCHW（官方便利函数为BHWC）。`JanusVQModel.encode`返回逐元素commitment/codebook误差；历史官方beta乘**codebook**项。离散decode会L2归一化码本，而`reconstruct`是保持直通梯度的独立训练路径。冻结VQ用于语言图像训练由外部角色图设置，不由模型隐藏调用`requires_grad_`。当前Janus `use_qk_norm=True`在上游有head/LN尺寸冲突，本地明确拒绝；不把临时改公式当成官方分支。

Mamba没有位置embedding，要求`position_ids=None`。padding只将输入和卷积后特征置零；它不等于把循环状态复位，episode/reset必须由上层显式处理。原生scan统一接收initial memory以支持多token分块继续；官方5.16.1普通scan fallback从零开始，因此oracle缓存比较限定真实单tokenupdate路径，分块继续另有全序列一致性测试。

LLaDA不支持append-only缓存；同一个MASK槽在后续迭代改变会影响全部双向上下文。`methods.masked_diffusion.MaskedDiffusionObjective`与`sample_masked_diffusion`复用该模型，目标不作next-token shift。本轮权重映射测试使用Transformers Llama+显式4D全可见bias验证公开源码规定的子图，证据类别不同于“直接运行原作者LLaDA 4.38运行时”。

V3.2 `output.auxiliary['indexer']`暴露每层scores、visible与indices，`methods.sparse_indexer.indexer_distillation`接受teacher注意力并按有效query数归一KL。离散top-k的CE梯度不会训练indexer，测试明确断言这一点；轻量indexer训练停止对teacher和backbone输入的梯度。Hadamard正交变换的精确点积等价不代表FP8量化排序等价。参考实现仍以dense布尔mask执行稀疏集合，不宣称实际稀疏kernel性能。

V4不是V3 MLA的改名。局部attention的K和V是同一张量，trailing通道RoPE在输出端逆旋转；sink只进softmax分母。HCA条目在完整不重叠窗口关闭后可见，CSA把前窗Ca与本窗Cb联合压缩后按indexer选择。缓存同时保存条目、窗口余数和Ca，使3→6→9等跨边界分块与整段计算一致。当前明确拒绝含padding/非连续position的压缩输入，服务按有效长度分组。`set_hash_routes(layer, table)`加载真实token→expert表；官方随机初始化表为零只是待加载状态，不能把默认表当作已训练分配，也不能用取模替代真实checkpoint。mHC残差混合必须使用矩阵转置，有限Sinkhorn迭代仅近似双随机。训练默认不建缓存；indexer离散选择不从CE得到梯度，分离的indexer目标可复用公开scores，但V4此处未强制冻结backbone，角色图需要显式管理。

Qwen3.5的four-projection DeltaNet与Qwen3Next按头packed参数不是相同checkpoint布局；两者在`nn.delta`共享递推、卷积状态和归一化。partial MRoPE只旋转显式比例的头维度。Qwen3.5图像不包含Qwen3-VL DeepStack层级残差；复用相同视觉原语经过严格视觉与整链权重测试。文本dense/MoE分别有配置，完整图文配置也可选择对应文本分支。当前官方文档称Qwen3.6与3.5共享model_type，因此不编造一个结构不同的新名称。

OpenVLA固定融合路径的DINO与SigLIP分别使用自己的归一化。`normalize_openvla_pixels`只处理已经完成几何变换的RGB，输出[B,6,H,W]；resize/crop/letterbox仍需数据配方锁定。倒数第二block特征无最后LayerNorm，DINO的CLS/register不作为图像patch送给语言模型。`align_labels`在BOS后插入-100标签区，对应真实logits长度S+N；cache按物理S+N计数，`position_ids`由模型管理，调用者传None。模型未额外创建独立trainer/采样服务器。`action_tokens`与`decode_actions`复用统一ActionSpec/ActionNormalizer/UniformActionTokenizer；词表大小先扣除补齐区域，选定训练数据的q01/q99与mask后恢复动作单位。默认strict拒绝非action token，`strict=False`才复现原作者预测函数对任意token的clip行为。`convert_prismatic_state_dict`显式拆SigLIP packed-QKV并覆盖DINO/timm与Llama权重；只允许不参与特征前向的attention-pool参数列为ignored，其余未知键/shape失败。当前训练演示是单样本接口闭环，不是LIBERO成功率或真实机器人验证。

## Gemma4文本、视觉与共享KV

Gemma4文本使用`Gemma4TextConfig`/`Gemma4ForCausalLM`。PLE由token表乘sqrt(ple_dim)与主embedding投影归一化相加再乘1/sqrt(2)；每层另用GELU门控和输出RMS残差注入。embedding-only输入必须同时传`per_layer_inputs=model.get_per_layer_inputs(ids)`，不执行官方便利入口昂贵且易歧义的反向词表匹配。`project_per_layer_inputs`显式提供上下文投影计算供后续多模态连接器使用。

Gemma4的Q/K norm不是Gemma3的zero-centered参数化，value还带无scale RMS，attention logits缩放为1。global层的proportional RoPE在完整head维度频谱后补零；不能换成较小head上的标准partial RoPE。`attention_k_eq_v`只影响全局attention的投影以及全局KV头数。后`num_kv_shared_layers`层不创建K/V权重，复用各类型最后一个owner的当前KV，状态`Gemma4State(kind='gemma4_shared_kv')`仅存owner层。local状态保留W-1条，global保留全长，支持fork/reorder/replay但不能任意truncate。MoE路由读dense分支前的残差，dense/sparse输出分别norm后相加再做总norm，不能用普通Mixtral替代。

`Gemma4VisionModel`消费浮点[0,1]的HWC顺序patch与xy坐标；`pack_gemma4_images`只做可微patch化与padding，不隐藏resize/crop或再次应用外部均值归一化。模型内执行2*(pixels-.5)，完整矩形patch网格必须被pool kernel整除。每个空间轴独立使用head_dim/2频谱，视觉attention缩放为1。输出是压紧的`PackedVisionOutput.last_hidden_state[sum(valid soft tokens),H]`和逐图`counts`，不是假定同长度的[B,S,H]。pool=1在实际oracle中有mask语义缺陷，当前明确拒绝；真实配置的pool>1路径已含有限clipping与standardize测试。

`Gemma4Config`/`Gemma4ForConditionalGeneration`将视觉特征经无scale RMS+线性映射填入placeholder；PLE身份分支用PAD、上下文分支读取注入后的图像特征。视频逐帧经过同一视觉塔，时间戳/帧顺序由输入token配方负责，不假装使用3D视觉主干。每条样本的placeholder数量和素材归属分别校验，多图/视频需显式`image_batch_indices`/`video_batch_indices`。设置`text_config.use_bidirectional_attention='vision'`并传0/1/2的`mm_token_type_ids`时，仅local层放开同一连续视觉块，global仍因果；local窗口只限制key>query-W，不能误换为对称窗口。完整视觉块须在同次prefill，之后支持文本缓存生成；不允许半张图像分批先后进入双向视觉块。图文embedding-only便利入口不猜图像身份，可直接使用已合并embedding与显式PLE。新增测试使用实际TF5.16.1整包模型，图像+视频、dense+MoE、mask及所有梯度均对照，但不代表大型权重公开benchmark成绩。

## BLIP-2查询式图文连接

`Blip2QFormerModel(query_embeds, query_length, attention_mask, encoder_hidden_states, encoder_attention_mask)`返回`VisionOutput`特征，并不是语言token logits。查询前缀在指定频率的层与图像cross-attend；`use_qformer_text_input=True`时文本后缀有独立FFN，且不直接cross-attend图像。输入可以显式指定[B,1,Q,K]可见关系，用于不让查询读取文本的阶段一屏蔽图；此功能不等于已经实现完整ITC/ITM/LM训练配方。Q-Former没有自己的token embedding，也没有可追加的因果缓存。

`Blip2VisionModel`保留真实CLS、带bias patch卷积、packed QKV、无pre-LN，以及末端所有token的LN；pooler对CLS再次用同一LN。模型接已标准化RGB，几何变换和归一化策略由数据配方固定；非原尺寸明确开启bicubic位置插值。BLIP-2 eager attention使用工作dtype softmax，而非强行沿用LLM默认FP32 softmax。

`Blip2Config`/`Blip2ForConditionalGeneration`组合真实视觉→可学习查询→Q-Former→语言投影→已有T5或明确的dense causal decoder。两种语言结构分别执行实际TF5.16.1完整模型对照；当前未增加OPT名字。输入每张图像对应一条文本，并已展开`num_query_tokens`个image placeholder，因此文本物理长度不被隐藏改变。T5通过新增的encoder `inputs_embeds`窄接口消费图像条件，目标使用`CrossEntropyObjective(causal=False)`且明确传`decoder_input_ids`；causal decoder仍用next-token监督。冻结语言权重不等于语言前向no_grad，测试确认梯度经过被冻结语言计算回到查询/投影。

`Blip2State(kind='blip2_language_state')`包裹对应语言state，配置身份绑定整个图文模型。T5 state保存固定视觉编码条件，seen_tokens仅计decoder；causal分支计展开后的文本长度。图像仅在prefill编码，续算拒绝更换图像；fork/reorder/truncate委托真实语言state，不伪装统一KV布局。共享Trainer测试覆盖真实训练、保存重载、全模态dropout精确续跑和冻结权重的反传路径。

证据边界：TF5.16.1的Q-Former wrapper对“query+text混合且有encoder padding”的cross-mask错误地使用总长度，会自身shape报错；该组合的oracle明确直接执行官方norm+encoder子图并传可广播mask。纯查询、纯文本、视觉和完整生成模型则直接执行原包，不修改安装的源码。当前没有大型BLIP-2权重或COCO/VQA质量成绩。

## GR00T图像语言动作与复用条件

`GrootConfig` / `GrootVLA`是真正的Qwen3-VL→N1.7动作DiT组合；`select_layer`为保留的LLM层数（保留最终RMSNorm），不是隐藏层索引。`GrootActionConfig` / `GrootActionHead`也可独立消费`GrootCondition(features, attention_mask, image_mask, proprio, embodiment_id)`。图文mask、固定本体状态历史、动作horizon逐项校验，机器人ID选用`[embodiment,input,output]`权重。完整模型只接已处理的图像patch/grid和token，不自动猜prompt、下载Cosmos权重或转换机器人坐标。

`GrootFlowObjective`共享`FlowPath`和`Trainer`；t从0噪声走向1动作，目标action-noise，默认时间分布`(1-Beta(1.5,1))*.999`，模型用`floor(t*1000)`编码。普通OpenPI的时间方向相反，不能混用。有效动作标量数是独立分母；原版`count+1e-6`空mask保护改为引擎空项处理和精确有效计数，避免microbatch/DP分片数量改变目标。它与官方非空单批的标量有不超过约1e-6相对差异，不宣称该分母逐位相同。

动作编码器时间基使用sin/cos、分母width/2；DiT时间编码器固定256维cos/sin、分母127。block调制输出按scale/shift拆分，最后输出按shift/scale拆分。FFN支持gelu/gelu-approximate/geglu；AdaLN与普通LN、可选位置编码均有独立公式对照。N1.7默认交替图文cross，其他块做动作self attention。原版普通DiT忽略encoder padding，本地该分支拒绝带padding输入；默认AlternateVLDiT正确保留图文valid masks。当前不包含可选额外`vl_self_attention_cfg`重处理塔，也不冒称旧N1/N1.5/N1.6 Eagle骨干兼容。

`sample_actions`默认4步Euler；只在当前请求内复用图文/本体特征，且可预投影各cross层K/V。缓存仅允许eval/no_grad；参数改变后必须重新编码。`previous_actions + overlap_steps + frozen_steps + ramp_rate`实现原版RTC已有动作重叠、延迟前缀冻结和指数斜坡；只操作归一化动作，不控制真实机器人。`predict_chunk`返回动作与无padding标志，物理单位仍由统一ActionSpec/ActionNormalizer和控制器负责。

测试包含60步动作头训练、100步完整视觉语言动作训练、同噪声缓存/重算采样一致、RTC冻结前缀保持、保存重载和Beta/noise/多处dropout的共享Trainer恢复。均为CPU微型工程验证，不是LIBERO成功率、Cosmos预训练能力或生产硬件延迟结果。

## Kimi K3文本与深度状态

`KimiK3TextConfig` / `KimiK3ForCausalLM`是独立的公开K3文本结构；小配置保留KDA/MLA混合、块间AttnRes、latent专家和SiTU，不沿用K2.5别名。KDA记忆`[B,H,Dk,Dv]`按key坐标独立衰减，安全门为`-5*sigmoid(exp(A_log)*(f+dt_bias))`，再作beta控制的delta纠错；普通GDN按head标量遗忘不是同一算法。输出为每头RMS后乘sigmoid的全秩门，不是SiLU。padding不写入卷积历史或记忆，等价于按样本打包后的有效序列。

AttnRes在每次forward内沿深度构建bank：块首先聚合旧bank+当前prefix、再存下**原始prefix**并开始新的块累计；MLP支路和最终输出各有自己的norm/score。RMS只用于打分，混合的value仍是原残差。bank不detach，也不属于跨token KV cache。Stable LatentMoE中router/shared读原H维hidden，routed experts读down-projected latent；先混合各专家，再RMS、up-project、加shared，不能把归一化挪到各专家内部。

`KimiK3State(kind='kimi_k3_hybrid')`的KDA叶子是卷积history和FP32循环memory，MLA叶子是latent和独立shared key通道（此处不做旋转）。只支持fork/reorder/快照重放，不允许任意truncate。attention默认为expanded计算图以保持真实叶子forward的分片所有权，但保存的仍是压缩latent。当前测试包含独立functional公式对照、20步训练、ZeRO3单rank训练/存储/恢复；不外推为专家多卡、原版MXFP4内核吞吐或公开基准效果。

`KimiK3Config/KimiK3ForConditionalGeneration`连接真正K3专用视觉塔，而不是K2.5塔。视觉head_dim由`qkv_hidden_size/num_attention_heads`决定，可以与hidden宽度不同；所有线性层无bias，xy位置表bilinear插值，静态图片不加时间cos项。各clip全部帧共同注意但不同clip隔离；时间维平均后保留空间merge内各patch。encoder使用`nn.RMSNorm(eps=None)`，因此epsilon是工作dtype的机器精度，不能换成文本1e-5。projector为biasfree Linear→GELU→Linear→RMS，没有K2.5的pre-LayerNorm。

输入`pixel_values[N,3,P,P] + grid_thw[items,3]`沿用共享`pack_kimi_patches`的行优先打包；`media_batch_indices`显式指定多个媒体项属于哪条文本。每条文本已展开的media placeholder数量分别验证，不能只比较全batch总数。模型不自动resize、猜聊天模板或下载权重。`KimiK3VisionState(kind='kimi_k3_multimodal')`保留完整模型身份和KDA/MLA语言状态；seen_tokens就是包含已展开视觉占位的物理token数。新媒体要求新prefill，缓存只支持快照重放，不假装普通可切片KV。

视觉BF16测试同时使用上游的复数乘法RoPE和Torch SDPA：只使用实数等价表达或在eager中先round注意力分数，会改变低精度梯度。当前微型BF16对照做到输出、pixel梯度及全部参数梯度逐位一致；这仍是特定CPU计算图验证，不代表整个模型在所有设备/精度下逐位相同。K3视觉权重路径对应公开`patch_embed/encoder.blocks/mm_projector`，文本routed专家和KDA打包布局需显式转换；未声称可直接读取发布的MXFP4 checkpoint。

## Qwen3.8-Flash-Next / Qwen4Exp 独立分支

`Qwen4ExpTextConfig/Qwen4ExpForCausalLM`保留公开Flash-Next的真实差异：GDN输出门为sigmoid；QSA按每个query实际可见的token依次分组，原始index key先FP32求微块均值、RMS归一化，再按块首token做partial MRoPE。多head正部得分相加后选完整微块，未满的尾部无条件保留。它不同于DSA的逐token索引，也不能把普通attention加一个任意稀疏mask就称为同结构。当前为透明Torch数学参考，仍构造显式mask/分数，不声称拥有稀疏CUDA kernel的吞吐或复杂度。

四条残差流的GatedResidual使用分组RMS、低秩门控混入、按流的`2*sigmoid`注入；不是DeepSeek mHC的Sinkhorn矩阵。默认第二层（`ple_layer_ids`是**1-based**）启用PLE：独立质数表大小、SplitMix64奇数乘子、EOS分段哈希、多头N-gram embedding、signed-sqrt门，以及dilation=`ngram_size`的因果卷积。EOS重置的是N-gram上下文，不擅自重置attention或GDN。PLE卷积按官方初始化为零；测试另外赋非零权重检查真实卷积和缓存历史，而非只测试零分支。

`Qwen4ExpState(kind='qwen4_exp_hybrid')`显式保存每层GDN的卷积/记忆，或QSA的K/V/**未归一化index key**；PLE层还保存其独立空洞卷积历史和最近N-1个token。全局保存`position_ids[3,B,physical_tokens]`，因为后续query的完整微块可能从历史位置开始。支持fork/reorder/快照重放，拒绝任意truncate；不能交给通用paged KV编解码器。`forward(inputs_embeds=...)`若启用PLE必须给`ple_input_ids`，不通过全词表比较去猜不可逆的视觉替换前token。

`Qwen4ExpVisionConfig/Qwen4ExpConfig/Qwen4ExpForConditionalGeneration`已连接图像和逐帧视频。固定官方源确认视觉Conv3d、逐帧packed attention、双线性位置表和pre-shuffle merger与已验证的无DeepStack Qwen视觉原语相同，因此共享这些计算；文本仍为上述独立结构。原token进入PLE，视觉特征只替换语言embedding，三轴位置和解码delta一起保存于`Qwen4ExpVisionState(kind='qwen4_exp_multimodal')`。输入格式与`pack_qwen_pixels`一致，视频各帧span须有文本/时间戳分隔，视觉数据只能在fresh prefill给出。

本地组合参数保留`language_model`与`vision_tower`两个真实所有者，`official_weight_name`显式映射为官方`model.language_model/model.visual/lm_head`路径。共享视觉的位置表和像素dtype访问已放入真正Embedding/Conv叶子forward，使ZeRO3能够物化参数；并未放宽训练器的安全校验。N-gram表的叶子可在CPU查行、仅将所取embedding移回输入设备；没有实现或宣称官方异步prefetch、分片巨表加载和GPU overlap。

`methods.sparse_indexer.QSAIndexerObjective(layers=(3,), ...)`在同一个模型forward/Trainer中组合CE和显式教师微块KL。`batch['qsa_teacher_attention']`以实际层号给出`[B,H,Q,K]`或`[B,Q,K]`矩阵；教师停止梯度，先在真实可见微块内聚合token概率，尾部不参与排序监督，每层按有效query独立计数。输出top-k索引不可微，**CE本身不会更新indexer**；测试明确区分CE-only和KL组合，不能以“参数存在”代替训练连通性。此为公开公式的KD工程组合，不冒称还原未公开的完整Flash-Next预训练配方。

测试：`test_models_qwen4exp*.py`包含哈希EOS隔离、非零PLE卷积、分段/单token解码、三轴位置、两种状态叉分/重排、训练导出和精确续跑；固定源functional oracle独立按token累加MoE、Python哈希、GDN递推并核对全部参与参数梯度。视觉子图和sigmoid GDN子图执行已安装Transformers5.16.1真实模块，完整Qwen4Exp公式锁定下列源码；本机wheel没有Qwen4Exp整类，所以不宣称运行过该整类。生产权重、Flash-Next hybrid MTP、公开质量基准和异步加速均不由这些测试推导为已验收。

## DeepSeek-OCR2文档链与可变切图

`OCR2SAMConfig/OCR2SAMEncoder`实现真正SAM图像路径：patch卷积、可插值绝对位置、窗口/全局注意力、分解相对位置、二维LayerNorm neck和两次stride-2卷积。窗口补零仍参与注意力，这是原作者公式，不能补一个看似合理的padding mask。相对位置bias使用未按sqrt(d)缩放的query；原生Torch SDPA另负责点积缩放。公开源的实际末端通道为896，不按发布配置中遗留1024字段猜测。

`OCR2VisualConfig/OCR2VisualEncoder`接上有独立local/global查询表的Qwen2。图像token只读整个图像，查询token读取图像与之前/当前查询；这不是普通BLIP-2 cross-attention。生产分辨率768/1024对应144/256查询，tiny配置保留两种分辨率和查询表，只缩小网格/宽度。公开权重路径`sam_model`与`qwen2_model.model.model.layers`经显式parameter codec保留；参数访问留在真实叶子forward，便于ZeRO3物化和统一导出。

`OCR2TextConfig/OCR2ForCausalLM`是该模型实际使用的非MLA语言分支：Llama attention、FP32 router线性+softmax、greedy top-k（默认不重新归一化）、共享专家相加。不是DeepSeekV3的sigmoid/correction router。未选中的专家也执行零token矩阵运算，使梯度为零而不是None，保留优化器动量/权重衰减语义及跨rank调用顺序。路由平衡项作为`output.auxiliary['router_aux']`显式返回；`CrossEntropyObjective(auxiliary_weights={'router_aux': .001})`组合公开默认权重，避免原作者自定义autograd隐式加损失。各层相加，每项按有效sequence计数；padding不计入router统计。

`OCR2Config/OCR2ForConditionalGeneration`按照实际forward拼接：按行优先的local图块→global图块→单个`view_seperator`，而不是按JSON中`global_view_pos=head`字段改变顺序。每条文档单独核对占位数；所有local和global分别一次批处理，空local仍执行零batch图，不伪造哨兵图片。视觉参数默认可训练；`freeze_vision/freeze_projector`显式冻结权重但不切断输入梯度，不照搬公开推理wrapper的硬编码CUDA/no_grad。训练label在展开后的图像占位处设-100，普通文本按next-token监督；模型不另建训练器。

状态`OCR2State(kind='ocr2_multimodal')`保存真实语言state和整个配置身份；seen_tokens是已展开视觉占位后的物理token数。图像只在fresh prefill提供，decode传新文本和原state。支持fork/reorder/快照重放，任意truncate明确拒绝；不能未经认证送进通用paged KV codec。

`data.ocr.prepare_document_views`接受uint8 RGB CHW：global保持比例居中补127，local按原作者宽高比规则选2–6图块并行优先排列，归一化到[-1,1]。当前只用Torch AA bicubic，**没有把未安装/执行的PIL整数重采样声称为逐像素完全一致**。`prepare_ocr_inputs(image, prompt, encode_text, config)`展开唯一`<image>`并返回`model_inputs`与`DocumentViews`；编码回调必须来自外层已锁定tokenizer，不能猜编码/模板。多页、文件读取和任务调度仍属于外部工作流。

`generate_document(model, inputs, decode_tokens, image_size=...)`复用公共采样器和原生state，以单次视觉prefill+单token decode返回`DocumentResult(text, token_ids, regions, stopped_on_eos)`；模型模式在结束/异常时恢复。禁止无像素的新增image token，EOS可显式设None禁用。`parse_grounding`把ref/det标记转为0–1000归一化框及原图像素框，只允许受限literal数值，拒绝代码/调用/反向框/无限数；不执行原作者示例中的eval。此入口不是另造HTTP推理服务，也不代表生成文本已有真实OCR质量。

测试分别核对真实Transformers5.16.1 SAM/Qwen2子图全部参数和输入梯度、实际Llama attention加独立原作者MoE、local→global→separator连接、训练降损失、ZeRO0/3单rank精确续跑/保存重载和真实缓存生成。没有执行原作者依赖旧Transformers/torchvision的整个wrapper。公开文档评测（如OmniDocBench）需要固定数据、tokenizer和真正预训练权重，应独立记录质量，而不能从tiny训练测试推出分数。

`test_models_ocr2_distributed.py`另执行真实Gloo DP2：rank0没有local、rank1有两个local，包含不等长文本padding。ZeRO0–3各自对照单进程完整batch的加权CE+router目标和全部参数更新，并验证checkpoint后下一步逐位一致。这只认证CPU DP与所测布局，不外推OCR专家并行、跨机GPU吞吐或巨大生产分辨率内存表现。

## Qwen多token预测与草稿状态

`QwenMTPConfig(text_config=Qwen35TextConfig(...), num_mtp_layers=1, share_embeddings=True)`构造共同拥有主干与草稿头的模型。`forward(ids, mtp_depth=N)`正常输出主干logits，并在`auxiliary['mtp_logits']`和`mtp_offsets`输出N条移位预测；offset=d时使用`logits[:, :-1]`预测`ids[:, d+1:]`。目标有效mask也必须相应移动，不能给不同深度相同分母。前一深度最终归一化hidden与下一已知token embedding输入下一深度，不是各头独立读同一个隐藏张量。

真实Qwen MTP公式按embedding在左、hidden在右做双norm与拼接FC；每次仅执行`layers[spec_step_idx % num_mtp_layers]`，不能一次把所有MTP层串起来。默认主干/草稿embedding和lm_head是相同Parameter对象，存储重载保持共享身份。`detach_mtp_base=True`只截断主干hidden支路，不冻结共享embedding/head；如果要求主干参数完全不更新，需显式冻结`model.backbone`，草稿FC/decoder仍能收到梯度。

推理直接调用`model.mtp(input_ids, hidden_states=..., position_ids=..., spec_step_idx=..., state=..., use_cache=True)`。`QwenMTPState`保存选中层的独立K/V、seen_tokens和layer_index，支持fork/reorder/truncate；不接受主模型混合递归状态，也不允许在另一个草稿层复用同一状态。调用者分别保存各层草稿状态，拒绝采样时同时回滚对应历史。当前这是可训练/可调用的模型能力，不等于部署服务已完成MTP speculative调度。

Qwen3.8公开配置仍采用Qwen3.5计算架构。官方SGLang中`output_gate_type='swish'`控制GDN的归一化输出门；full-attention输出门仍为sigmoid。不能仅凭配置字段把普通attention门替换成swish。已验证这里的MTP计算，不代表下载/验证过3.8发布权重或Qwen3.8-Flash-Next新架构。

GDN的`A_log/dt_bias`由独立`decay_gate.forward`叶子实际拥有并计算，使ZeRO3能够逐单元gather/重算/release，而不是父层读取裸分片。内部参数名带`decay_gate`，公开state_dict继续使用官方原名；`_aster_parameter_key_map`及成对存储hooks显式映射，双命名/碰撞拒绝。oracle依然严格加载同名state_dict，梯度对照显式使用同一映射。

## Cosmos3：联合理解/生成，而不是另一个 Wan 别名

`models/cosmos3.py` 的 `Cosmos3MoT` 保留两套真实注意力投影、输入/输出 RMSNorm、MLP：理解路只做因果自注意力；生成路同时读取理解 K/V 和全部生成模态 K/V。视频、声音、动作是**同一次联合生成注意力**中的三种 token，不是三个独立模型。`hidden_act='silu'` 对应 Qwen 型 SwiGLU，`relu2` 对应 Nemotron ReLU² 与不同 RMS 精度顺序；可显式关闭理解 Q/K norm 并为生成读取理解 K 添加独立 norm。这里的 `moe_gen` 是固定模态路，不是假定为 top-k 路由专家。

公共接口：

```python
from aster.models import Cosmos3Config, Cosmos3MoT, Cosmos3Vision, Cosmos3Sequence
from aster.methods.cosmos3 import Cosmos3FlowObjective, sample_cosmos3

model = Cosmos3MoT(Cosmos3Config())
# vision: Cosmos3Vision(sample[B,C,T,H,W], positions[3,B,N], timesteps[B,T], noisy_frames[B,T])
# sound/action: Cosmos3Sequence(sample[B,T,D], positions[3,B,T], timesteps[B,T], noisy_frames[B,T])
# 动作额外要求 domain_ids[B]，选择真实 embodiment 专属 input/output 参数表。
output = model(input_ids, vision=vision, sound=sound, action=action)
# output.text 是 TokenOutput，其余字段是 FieldOutput(prediction_type='velocity')。
# 训练时 sample 是干净目标；采样时 noisy 帧放高斯噪声、clean 帧放条件 latent。
objective = Cosmos3FlowObjective(text_weight=.2, time_distribution='logit_normal')
latents = sample_cosmos3(model, model_inputs, steps=20, solver='heun', shift=10.)
```

空间 patch 顺序为 `[T,Hp,Wp,p,p,C]`，奇数尺寸只在右/下补零并在输出裁掉。`cosmos3_positions` 显式声明 FPS、VAE 时间压缩倍率和基准倍率；三轴 interleaved mRoPE 按视频/声音各自真实采样率构造，声画是同一起点的时间轴，不把声音时间接到视频末尾。时间单位是官方 scheduler 的 `[0,1000]`，模型缩放 `.001`。`noisy_frames` 只向待预测帧加入 time embedding，条件帧输出速度恒零。动作投影按官方 `[input,output]` embedding 行布局计算，不能当普通 Linear 权重转置加载。

`Cosmos3State(kind='cosmos3_understanding')` 的每层是理解 K、供生成读取的 K、V 三个 `[B,KV,S,D]` 张量，外带完整 bool mask 和 `seen_tokens`。只缓存因果理解前缀，支持 fork/reorder/replay，禁止任意 truncate；随 ODE 时间改变的双向生成 token 每一步必须重算。`forward_text` 暴露同一模型的 TokenModel 角色，不复制语言主干。该状态不是通用 paged KV 已认证声明。

训练使用共享 `FlowPath(direction='data_to_noise')`，`x_s=(1-s)data+s*noise`、目标 `noise-data`。推理从 `s=1` 走到 `0`；共享 Euler/Heun/RK4 求解器将三种 latent 拼成同一个 ODE 状态，而不是分别采样。文本 CE 和每种模态 MSE 分别累计真实有效 token/scalar 分子与 int64 分母，再由统一 Trainer 在 DP 域归一化。支持固定显式噪声/时间测试和随机 uniform/logit-normal 时间，统一 checkpoint 保存随机状态；这些是明确可复现的本地配方，不冒称 NVIDIA 未公开的大规模预训练数据/优化日程。

证据：`test_models_cosmos3.py` 验证联合训练下降、ZeRO0/3、模型导出、随机 time/noise 精确续跑、cached/recomputed 联合采样及条件帧不变；`test_models_cosmos3_distributed.py` 在真实 Gloo DP2 对照全部 ZeRO0–3 的 dense 更新和精确恢复，包括一个 rank 某模态零监督帧。`test_models_cosmos3_parity.py` 按锁定源码独立重建 packed 单样本公式，校验 batch 扩展、FP32 全参数/latent 梯度和两分支 BF16 前向；理解路另执行真正的 Transformers5.16.1 Qwen3VL 子图严格权重映射、梯度和缓存对照。**整包 Diffusers 未安装/未执行，独立公式不是整包 oracle；没有下载发布权重或证明视频/音频/机器人基准质量。**

当前边界是固定 batch、每样本每种生成模态一段连续 latent；任意多段素材 packing、Qwen3-VL 像素理解完整 processor、Cosmos3 AVAE 波形 codec、官方 4-step 蒸馏权重、UniPC 全管线和专用 context-parallel kernels 不在该模型包的已验证范围。可输入显式 `inputs_embeds`，但不会因此宣称视觉理解 processor 已完成。Nano 的真实 Wan2.2-TI2V-5B 视频codec已在下述独立分支完成，并连接到同一个MoT；不是拿旧Wan2.1改名。官方声音 codec 为48kHz stereo、hop1920的 AVAE2，使用 STFT/ConvNeXt encoder 与 SnakeBeta/Oobleck decoder；这不是一个通用线性投影可以替代的缺口。

来源固定为 [Diffusers Cosmos3 模型源码](https://github.com/huggingface/diffusers/blob/c1bf18c92c6285334adcaac7e75ef8946a227f49/src/diffusers/models/transformers/transformer_cosmos3.py)、[生成管线与位置坐标](https://github.com/huggingface/diffusers/blob/c1bf18c92c6285334adcaac7e75ef8946a227f49/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py)、[NVIDIA Cosmos 发布](https://github.com/NVIDIA/cosmos/tree/e7ad5e77eecd47acadf17db47d6eb56282a099cc)、[Nano 公开配置锁](https://huggingface.co/nvidia/Cosmos3-Nano/tree/7a312c868bcce8e40b3eb40861300a9d0ba3fde1)。前两份源码 Apache-2.0，Copyright NVIDIA/HuggingFace；Cosmos3 发布权重按 OpenMDW1.1 条款单独核验，不将代码许可当作模型权重许可。

### Wan2.2残差视频codec与Cosmos3像素闭环

`Wan22VideoVAE(Wan22VAEConfig(...))` 是独立 `wan22_vae` 架构。共享经过核对的因果卷积与时间重采样原语，但独立实现2.2的AvgDown/DupUp残差捷径、保持输出宽度的upsampler、patch2展开和不同encoder/decoder宽度。权重名保持Diffusers `encoder.down_blocks.*.resnets.*`、`decoder.up_blocks.*`、`quant_conv` / `post_quant_conv` 布局；不复用Wan2.1的权重映射。公开Nano配置是160/256基础宽度、48潜变量通道、空间压缩16、时间压缩4；默认小配置保留公式但不附会公开模型效果。

`encode(video)`接收归一化RGB `[B,3,1+4k,H,W]`，首帧独立，其后每4帧编码一个latent；`decode_chunks`逐次输出1、4、4帧。每次调用的缓存是局部字典，允许交错解码两个视频；训练缓存不detach，后续片段能反传至历史片段。2.2低精度RMS先在FP32归一化，QKV使用锁定源的连续stride布局。`latent()`按posterior mode及codec统计归一化，显式选择才采后验噪声。

`Wan22VAEConfig.from_diffusers_config(mapping)`只翻译已经核对的残差RGB分支，拒绝未知数学字段和不一致的压缩倍率。官方Nano配置带`clip_output=false`，但固定Diffusers版本的`_decode`实际无条件clamp；native默认照源码clamp，训练显式`clip_output=False`保留越界梯度。归一化严格先在codec dtype求`inv_std`，编码`(z-mean)*inv_std`，解码`z/inv_std+mean`，不把BF16下不同舍入顺序当作逐位等价。

`Wan22AutoencoderObjective(sequence_length=5, kl_weight=1e-6, sample_posterior=True)`是明确的最小L1+KL训练目标，不声称完整复刻Wan的感知/GAN训练配方。使用共享Trainer与全局sample分母；固定T进入objective身份并在所有rank任何参数gather之前检查，允许不同B/H/W。不同rank不同chunk数量会对称拒绝，而不是挂起后再检查梯度。

`Cosmos3VideoPipeline(model, vae)`将真实codec与**同一个**联合MoT相接：`training_batch(video, model_inputs, ...)`冻结条件codec并返回`Cosmos3FlowObjective`直接消费的batch；`generate(model_inputs, noise, condition_video=..., ...)`完成显式噪声→共享ODE→视频解码。条件视频为因果前缀，清洁latent帧保持不变；按每条有效文本长度、FPS及codec时间倍率构造生成坐标，不让padding推进mRoPE。codec禁用外层AMP并在计算后恢复原dtype与train/eval模式。返回`Cosmos3VideoOutput.video`和各模态`latents`，不把尚未实现的声音codec伪装为波形。

证据共12项：`test_models_wan22_vae_parity.py`独立重建完整公开权重名函数图，9帧覆盖全部cache阶段，FP32/BF16前向、全部参数及输入梯度对照；`test_models_wan22_vae.py`验证真实学习、ZeRO0/3后验/dropout随机状态精确续跑、保存重载、因果前缀与交错cache，以及BF16近零KL和FP16大logvar负例。KL统计提升至FP32并用`expm1`避免相消，不改变codec前向公式；这不等于FP16参数/梯度不再需要finite检查。`test_models_wan22_distributed.py`执行真Gloo DP2全ZeRO0–3与dense更新、精确恢复、不等T前置拒绝；`test_models_cosmos3_video_pipeline.py`执行像素→codec→flow训练→两个模型导出重载→条件生成→解码闭环及BF16归一化对照。环境没有安装或执行Diffusers整包，没有下载公开大权重，没有宣称视觉基准效果达标。空间tiling及完整官方UniPC/蒸馏采样不属于此包。

来源：[锁定Wan VAE源码](https://github.com/huggingface/diffusers/blob/c1bf18c92c6285334adcaac7e75ef8946a227f49/src/diffusers/models/autoencoders/autoencoder_kl_wan.py)、[真实Nano VAE配置](https://huggingface.co/nvidia/Cosmos3-Nano/blob/7a312c868bcce8e40b3eb40861300a9d0ba3fde1/vae/config.json)。源码Copyright2025 Wan/HuggingFace，Apache-2.0；公开权重另依对应模型条款。

## Cosmos-Predict1：GeneralDIT、官方训练损失与 Euler 边界

`models/cosmos_predict1.py` 的裸网络 `CosmosPredict1DiT(CosmosPredict1Config(...))` 实现原作者 GeneralDIT。每一层包含独立的全注意力 FA、文本交叉注意力 CA 和非门控精确 GELU MLP，每个子块都有自己的无仿射 LayerNorm 与 shift/scale/gate。AdaLN-LoRA 版本将原始 sin/cos 时间向量送入各层低秩调制，同时从共享 time MLP 产生 `3*hidden` 附加调制；不能按普通 DiT 的单个时间向量处理。理解/生成固定双路属于 Cosmos3，不套到 Predict1 上。

原作者 patch 输入顺序 `[channel,temporal_patch,spatial_h,spatial_w]` 与最终输出 `[spatial_h,spatial_w,temporal_patch,channel]` 不同；测试使用 `patch_temporal=2` 明确检查这一点。Q/K 每头 RMSNorm，3D RoPE 只施加于视频 self-attention，不施加于文本 cross-attention；时间坐标按 `24/fps` 缩放，三轴 NTK 参数独立。可选每块添加 learned 轴位置，其归一化分母是 `RMS+eps` 而非 `sqrt(mean(square)+eps)`。额外 `padding_mask` 是网络输入 channel，不是 attention mask；原作者发布主路径不使用文本 padding attention mask，调用者必须提供已经按约定编码的完整文本 context，不能假装遮罩生效。

```python
from aster.models import CosmosPredict1ModelConfig, CosmosPredict1Condition, build_model
from aster.methods.cosmos_predict1 import CosmosPredict1Objective, sample_cosmos_predict1
from aster.training import Trainer

model = build_model(CosmosPredict1ModelConfig())  # net + 可学习的 logvar
condition = CosmosPredict1Condition(text_embeddings, fps, padding_mask)
trainer = Trainer(model, CosmosPredict1Objective(), lr=1e-4)
trainer.step([{'sample': clean_latent, 'condition': condition}])
generated = sample_cosmos_predict1(model, unit_noise, condition,
    negative_condition=negative_condition, guidance=1.5, steps=35)
```

`forward(sample, time, condition)` 返回 `FieldOutput(edm_residual)`；这里 sample 已经过 EDM 的 `c_in` 缩放，time 是 `log(sigma)/4`，不是 sigma、本步编号或 `[0,1000]` 时间。`CosmosPredict1Condition` 显式带 `[B,L,D]` 文本嵌入、`[B]` FPS 和空间 padding channel。视频批次遵守源头的同 FPS 限制；裸 `CosmosPredict1Config` 可直接接共享 `EDMObjective`，但这只代表标准 EDM 配方。

官方训练组合 `CosmosPredict1Model` 另持有随机 Fourier128→Linear 的 `logvar` 头，持久保存随机频率/相位。`CosmosPredict1Objective` 使用共享 `edm_denoise`：`x0=c_skip*x+c_out*net(c_in*x,log(sigma)/4,condition)`，先算 sigma 权重 MSE，再按 `exp(-logvar)*weightedMSE+logvar` 得到 Kendall 损失。默认 `log_mean=0/log_std=1/loss_reduce='sum'/loss_add_logvar=True/sigma_data=.5` 与已核对的训练默认一致；支持明确关闭 logvar、选择 mean、额外 sample 权重和逐 latent 损失权重。损失 mask 是重要性权重，不改变分母；logvar 项仍按源公式另加。原作者 EDMSDE 虽保存 sigma_min/max，训练抽样实际上不截断 log-normal；本地同样不暗加截断。Torch 正态与上游 NumPy uniform→NormalDist inverse-CDF 分布等价，不声称随机种子到噪声的跨库逐 bit 对齐。

`sample_cosmos_predict1` 是明确的 **EDM Euler**，不是公共 `sample_edm` 的 Heun：默认 Karras `rho=7`、`sigma_max=80`、`sigma_min=.0002`、末端 0；FP64 构建日程后转换 FP32。初始单位噪声乘 `sqrt(sigma_max²+1)`，每步 FP32 Euler 更新后回到网络输出 dtype。官方 guidance 是额外引导量：`conditional + guidance*(conditional-unconditional)`；默认1.5必须显式提供 negative condition。只要传了 negative，每个 sigma 调网络两次；单条件显式 guidance=0 时一次。没有末端二阶校正。网络/目标/采样的参数化统一，但 Euler 与 Heun 必须在评价记录中分开。

证据为 `test_models_cosmos_predict1.py`、`test_models_cosmos_predict1_parity.py`、`test_models_cosmos_predict1_distributed.py`：训练下降、单卡 ZeRO0/3 随机精确恢复、模型/不确定性头导出、BF16有界前向/梯度/存储；非零 gate 下独立公式的全部参数/像素latent/文本/time 梯度；Kendall 的 sum/mean、权重和 Fourier 公式；Euler 每一步与 NFE 检查；真实 Gloo DP2、各 rank 不同 batch/视频空间时间尺寸/文本长度，ZeRO0–3 全局 sample 分母和逐 bit 下一步恢复。没有执行上游 TransformerEngine、Apex、Megatron 整包，不把 CPU 公式测试当作生产 CUDA 性能或公开视频基准成绩。

当前只实现公开 GeneralDIT text-conditioned latent 主路径及上述训练组合。Cosmos tokenizer/VAE、T5 字符串处理、VideoExtend 的条件帧/augment-sigma、插帧、多视角、模型专属 CP 内核和原始数据训练配方仍是独立项；不因为输入/输出形状类似就挂现有 Wan VAE 或通用视频模型并宣称权重兼容。

原作者锁定 [GeneralDIT](https://github.com/nvidia-cosmos/cosmos-predict1/blob/724daa1b2df5ec96bdf111bb947479d2216b3b08/cosmos_predict1/diffusion/networks/general_dit.py)、[子块](https://github.com/nvidia-cosmos/cosmos-predict1/blob/724daa1b2df5ec96bdf111bb947479d2216b3b08/cosmos_predict1/diffusion/module/blocks.py)、[注意力](https://github.com/nvidia-cosmos/cosmos-predict1/blob/724daa1b2df5ec96bdf111bb947479d2216b3b08/cosmos_predict1/diffusion/module/attention.py)、[3D位置](https://github.com/nvidia-cosmos/cosmos-predict1/blob/724daa1b2df5ec96bdf111bb947479d2216b3b08/cosmos_predict1/diffusion/module/position_embedding.py)、[训练公式](https://github.com/nvidia-cosmos/cosmos-predict1/blob/724daa1b2df5ec96bdf111bb947479d2216b3b08/cosmos_predict1/diffusion/training/models/model_image.py)、[官方训练默认](https://github.com/nvidia-cosmos/cosmos-predict1/blob/724daa1b2df5ec96bdf111bb947479d2216b3b08/cosmos_predict1/diffusion/training/config/base/model.py)、[Euler管线](https://github.com/nvidia-cosmos/cosmos-predict1/blob/724daa1b2df5ec96bdf111bb947479d2216b3b08/cosmos_predict1/diffusion/model/model_t2w.py)。Euler 数值定义另锁定 [Diffusers EDMEulerScheduler](https://github.com/huggingface/diffusers/blob/c1bf18c92c6285334adcaac7e75ef8946a227f49/src/diffusers/schedulers/scheduling_edm_euler.py)。上述代码 Apache-2.0；发布权重条款需独立核验，未下载。

## 本地制品（存储契约）

`save_pretrained` 保存严格 JSON 配置、配置 SHA256、权重 SHA256 和逐 tensor dtype 清单。权重只用 `torch.load(..., weights_only=True)` 读取张量字典；不反序列化任意模型对象。加载保留张量 dtype，并恢复真正共享的 Parameter 身份。错误 checksum、配置不符、共享参数保存值不一致、未知字段均拒绝。

每个文件原子替换不等于跨文件事务。保存中途崩溃时 loader 会拒绝 hash 不一致；最终不可变发布由外层 `ArtifactStore` 管理。本地模型目录不是远程 Hub ID；禁止把路径自动解释为下载请求。

## 可执行对照与版本

设计指定参考锁是 Transformers `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`。已实际读取并执行的 oracle 是隔离环境中 **Transformers 5.16.1 / PyTorch 2.11.0 CPU**，不是自动声称该 wheel 等于设计锁。后续通过只读远程ref查询确认目标修订并可读新Qwen4Exp源码，但未升级本机wheel；因此“可读的源码锁”与“实际执行的安装快照”继续保持分开。

本机已读源码 SHA256 示例：

- `models/deepseek_v3/modeling_deepseek_v3.py`：`da66249787ddac6ba2dd603d3d39d791011d2385751da68248ac9d03fad07fd2`
- `models/qwen3_next/modeling_qwen3_next.py`：`46baea7085f2fbde062e546e7511dfc9d01217963d1c695310256069e725603b`
- `models/qwen3_vl/modeling_qwen3_vl.py`：`e925233666f43299ec9e2a1c6bc2a5d16b0d6fc06aaa3ef817485a14997de327`
- `models/kimi_k25/modeling_kimi_k25.py`：`34db965be72d5df859ce18ee9cf69aecc1b972317ce5201d0c074dc9015e969d`
- `models/siglip/modeling_siglip.py`：`0a0d2b55f4fcb977b74549540211e42a3c3f41d0d446dc474dcc244a925791aa`
- `models/janus/modeling_janus.py`：`680a3b3b7bcac84c5e9568680a41e5e8b94fbe1a13ca9ef37fe154fc42883983`
- `models/gpt2/modeling_gpt2.py`：`2d77d6a0260f1ee3779e418fef228d98914a32f33b35198be0fd3ebec9700b1a`
- `models/mamba/modeling_mamba.py`：`76643b79767f5bf0aa563a4e547c5526868733c97fcda8239c87f054fb6aaa2d`
- `models/deepseek_v32/modeling_deepseek_v32.py`：`bbc83144985bb3669b102d611dda9261a97dfad29cc73228a36d2da1d09ddddc`
- `models/deepseek_v4/modeling_deepseek_v4.py`：`31ec07d1baacdfd7e2dc2414cbee7d7b346f84d100b8b219915d91545a9635fa`
- `models/qwen3_5/modeling_qwen3_5.py`：`788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263`
- `models/qwen3_5_moe/modeling_qwen3_5_moe.py`：`682f99cb07b1df9d57b8a68006cc8e74cb4b1aa5263fd53f5f1bf4b886d6f55e`
- `models/dinov2_with_registers/modeling_dinov2_with_registers.py`：`fa3a93c01aabe0c9c8638252e56c4ac404b00fc6c81b8400b9cd8f858ba4e4bf`
- `models/gemma4/modeling_gemma4.py`：`3f6a049b83b79be651693db5a10f229a33e81812a01ced73470fee2b5eeac86f`
- `models/blip_2/modeling_blip_2.py`：`1b738cb433bd07251bdc9db9f7fb3555893d882fe24e814ff69edbfb293a2a1d`

`tests/unit/test_models*.py` 不需要 Transformers。`tests/parity/test_models*_parity.py` 在隔离 oracle 可用时构造 tiny 官方模型，严格检查 state_dict 参数映射、FP32 logits、全部参与参数梯度和增量缓存。对照容差通常前向 `atol=2e-6..3e-6`，梯度 `atol=2e-5..5e-5`，各测试明确给出；全 mask 行本地定义为零注意力，不能把它当成有效位置与上游未定义行为做同等承诺。

还包括：MLA 权重吸收、Qwen Delta chunk/递推、RoPE linear/Llama3/YaRN、跨窗口缓存、Llama4 输入侧专家路由、CLIP 官方像素标准化、混合状态错误回滚、严格存储 dtype/共享参数测试。后续加速 kernel 必须复用这些计算图/状态基准，不绕过它们。

## 官方来源与仍未实现的边界

本轮实际实现直接核对安装的官方源码，以下链接供源码锁审计与公式追溯：

- [Transformers 锁定源码根](https://github.com/huggingface/transformers/tree/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models)：`llama`、`qwen2`、`qwen3`、`mistral`、`mixtral`、`deepseek_v3`、`bert`、`t5`、`clip`、`llava`、`qwen3_next`、`gemma3`、`llama4` 各自的 `configuration_*.py` / `modeling_*.py`。
- [DeepSeek-V3 原作者实现](https://github.com/deepseek-ai/DeepSeek-V3)、[Qwen3-Next 官方说明](https://qwenlm.github.io/blog/qwen3_next/)、[Qwen3Next Transformers 接口](https://huggingface.co/docs/transformers/model_doc/qwen3_next)。
- [LLaVA 原作者工程](https://github.com/haotian-liu/LLaVA)、[LLaVA 官方库接口与预处理说明](https://huggingface.co/docs/transformers/model_doc/llava)、[OpenAI CLIP](https://github.com/openai/CLIP)。
- [T5 官方库接口](https://huggingface.co/docs/transformers/model_doc/t5)、[Google T5](https://github.com/google-research/text-to-text-transfer-transformer)、[Google BERT](https://github.com/google-research/bert)。

- [SigLIP官方库接口](https://huggingface.co/docs/transformers/model_doc/siglip)、[Google big_vision双塔源码](https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/image_text/two_towers.py)。SigLIP、Kimi、Qwen-VL和Janus本轮直接对照的Transformers源码文件头为Apache-2.0；不要把代码许可证自动套给预训练权重。
- [Janus原作者工程](https://github.com/deepseek-ai/Janus)、[Janus官方库接口](https://huggingface.co/docs/transformers/model_doc/janus)：原作者仓库分别列出LICENSE-CODE（MIT）与LICENSE-MODEL，本轮未下载模型权重。

- [Mamba原作者模块](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)、[OpenAI GPT-2](https://github.com/openai/gpt-2)；本轮直接运行的Transformers对应文件均Apache-2.0。
- [LLaDA原作者源码锁](https://huggingface.co/GSAI-ML/LLaDA-8B-Base/blob/0f2787f2d87eac5eed8a087d5ecd24277e6255b2/modeling_llada.py)、[配置](https://huggingface.co/GSAI-ML/LLaDA-8B-Base/blob/0f2787f2d87eac5eed8a087d5ecd24277e6255b2/config.json)、[训练指引](https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md)。原作者公开模型页标记MIT；训练框架与数据未公开，本轮不声称复刻预训练结果。

- [V4官方Transformers源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py)、[配置](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py)。本轮直接执行5.16.1快照；`deepseek-ai/DeepSeek-V4`仓库读取返回404，不把第三方“guess V4”PR当作原厂证据。

- [Qwen3.5/3.6官方库说明](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、[MoE说明](https://huggingface.co/docs/transformers/model_doc/qwen3_5_moe)；本轮公式以已执行5.16.1源码为准。
- [OpenVLA原作者HF模型](https://github.com/openvla/openvla/blob/main/prismatic/extern/hf/modeling_prismatic.py)、[action tokenizer](https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py)、[原版timm v0.9.16视觉源码](https://github.com/huggingface/pytorch-image-models/blob/v0.9.16/timm/models/vision_transformer.py)、[Meta DINOv2](https://github.com/facebookresearch/dinov2)。OpenVLA整包指定旧timm/Transformers环境，本轮只执行官方子模块组合oracle，不把组合验证误称原包直接运行。
- [GR00T N1.7发布锁](https://github.com/NVIDIA/Isaac-GR00T/commit/23ace64f17aa5015259b8609d371eb61a357c776)、[动作头](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/gr00t_n1d7/gr00t_n1d7.py)、[DiT](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/modules/dit.py)、[多本体MLP](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/modules/embodiment_conditioned_mlp.py)。这些上游文件Apache-2.0、Copyright2026 NVIDIA；模型权重许可证另行核验。本地没有Diffusers，因此动作测试证据是独立公式重述，与真正运行Transformers视觉语言oracle分开记录。
- [Gemma4官方接口及PLE说明](https://huggingface.co/docs/transformers/model_doc/gemma4)、[Gemma4官方模型源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)。实际oracle为上述5.16.1快照，Apache-2.0；不把代码许可证当作Google Gemma模型权重条款。
- [BLIP-2官方库源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py)、[Salesforce原作者Q-Former](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/Qformer.py)。实际权重/公式oracle执行上述Transformers5.16.1 Apache-2.0快照；原作者工程的完整预训练配方是独立验收项。
- [Qwen MTP官方vLLM源码锁](https://github.com/vllm-project/vllm/blob/87b9b5b8d9dadc3edb31efa6ea71ee7d49d0bdcd/vllm/model_executor/models/qwen3_5_mtp.py)、[SGLang Qwen3.5门控实现](https://github.com/sgl-project/sglang/blob/fe694986a296787cb0ada3b8cd7b6dccd21de72a/python/sglang/srt/models/qwen3_5.py)、[Qwen3.8公开配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)。前两个源码Apache-2.0；本轮完整读取vLLM MTP公式并重新核验不可变提交，但未执行其分布式运行时，oracle为已安装Transformers5.16.1的真实主干/层加独立MTP拼接公式。
- [Kimi K3公开模型配置](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)、[官方推荐SGLang文本源码锁](https://github.com/sgl-project/sglang/blob/fe694986a296787cb0ada3b8cd7b6dccd21de72a/python/sglang/srt/models/kimi_k3.py)、[AttnRes公式](https://github.com/sgl-project/sglang/blob/fe694986a296787cb0ada3b8cd7b6dccd21de72a/python/sglang/srt/layers/attn_residual.py)、[SiTU](https://github.com/sgl-project/sglang/blob/fe694986a296787cb0ada3b8cd7b6dccd21de72a/python/sglang/srt/layers/activation.py)、[FLA KDA参考](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/naive.py)、[安全门](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/gate.py)。SGLang源码Apache-2.0、FLA源码MIT；K3发布权重和原作者仓库另按Kimi K3 License核验。测试是独立公式重述，不是执行整个SGLang运行时。
- [K3专用视觉与连接器源码锁](https://github.com/sgl-project/sglang/blob/fe694986a296787cb0ada3b8cd7b6dccd21de72a/python/sglang/srt/models/kimi_k3_vl.py)、[时间池化/空间merge](https://github.com/sgl-project/sglang/blob/fe694986a296787cb0ada3b8cd7b6dccd21de72a/python/sglang/srt/models/kimi_vl_moonvit.py)。源头明确要求不要与K2.5视觉混用；本地保持独立配置和真实参数布局，仅共享已核对的packing/分段attention原语。
- [Qwen3.8-Flash-Next原作者发布说明](https://github.com/QwenLM/Qwen3.8-Flash-Next)、[实际公开配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)、[Qwen4Exp锁定模型源码](https://github.com/huggingface/transformers/blob/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)、[锁定配置源码](https://github.com/huggingface/transformers/blob/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py)。本轮逐项读取并按Apache-2.0源码核对，权重条款另行核验；没有把5.16.1安装包冒认为包含此新增模型。
- [DeepSeek-OCR2原作者仓库锁](https://github.com/deepseek-ai/DeepSeek-OCR-2/tree/2f3699ebbb96fa8af32212e8c170f2cc28730fad)、[DeepEncoderV2锁定源码](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/blob/aaa02f3811945a91062062994c5c4a3f4c0af2b0/deepencoderv2.py)、[图文模型与预处理](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/blob/aaa02f3811945a91062062994c5c4a3f4c0af2b0/modeling_deepseekocr2.py)、[实际V2语言分支](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/blob/aaa02f3811945a91062062994c5c4a3f4c0af2b0/modeling_deepseekv2.py)、[许可证](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/blob/aaa02f3811945a91062062994c5c4a3f4c0af2b0/LICENSE.txt)。上述源Apache-2.0，视觉还标注Meta SAM/ViTDet来源；本地保留这些归属说明，权重未下载。

后续独立施工范围包括 Gemma4的音频/Unified分支、Kimi K2.6及其他 scope 清单。没有把这些名称映射成已实现的 V3/普通 LLaVA。闭源权重、训练数据和未公开工程细节依然是资料缺口，不能用一份技术报告宣称完整复刻。

量化、LoRA、MTP部署调度、长序列加速、专家并行和真实基准质量是独立能力/配方验收，不由“模型类已经存在”推导为已完成。
# Cosmos3 Qwen视觉理解与同一MoT生成路

2026-08-31本轮固定模型回归：从441个src/tests/pyproject文件逐hash核对的独立
副本运行`tests/unit tests/parity -k models`，**250 passed / 682 deselected**，
331.23秒；仅两个官方PyTorch weight_norm弃用警告。该结果不是全仓/公开大模型
benchmark结果。音频后验scale/KL反向强化与语义buffer负例另经7项复测通过。

`Cosmos3VLMConfig/Cosmos3VLM`（`cosmos3_vlm`）并非独立VLM输出再拼一个生成器：
原生Qwen3Vision合并patch特征，按真实占位token散射回文本embedding，并在各指定
理解层后注入DeepStack；视频、声学、动作latent仍读取这同一MoT的理解前缀。
`forward_text`返回`TokenOutput`，联合`forward`返回`Cosmos3Output`。
`Cosmos3VLMState`保留物理KV长度、K/Kgen/V和mRoPE delta，缓存后不重复编码图像；
支持fork/reorder/replay，拒绝把视觉坐标状态当成普通文本KV任意截断。

`Cosmos3VisualFlowObjective(visual_prefill='image'|'video'|'none', ...)`复用既有
逐token CE和逐模态flow分母。媒体角色、网格、占位段界标、生成字段与noise在任何
分片参数收集前验证；图像路径和无图像路径不能在同一次ZeRO3调用图里隐式混用。
`Cosmos3VideoPipeline`直接接该模型：图像理解与真实Wan2.2编码的视频共同训练，
然后复用理解缓存做联合ODE并解码视频。无另造专用Trainer。

证据：8项新测试覆盖真实Transformers5.16.1图像/视频子图的全部对应参数、像素
梯度和缓存，单卡ZeRO0/3训练下降及RNG精确恢复，Gloo DP2所有ZeRO0–3全局
分母/不等长样本/异步错误前置拒绝，FP32/BF16安全发布重载。联合旧视频pipeline
2项及共享序列化2项负例，合计12 passed。只执行小随机权重CPU测试；没有下载
公开大权重、执行完整NVIDIA运行时或声称达到官方基准质量。

锁定源：
[NVIDIA Qwen视觉融合](https://github.com/NVIDIA/cosmos-framework/blob/0e034bc98ffa3c3dfa19f037871f3a8bbc1c4d05/cosmos_framework/model/generator/reasoner/qwen3_vl/utils.py)、
[同一MoT与DeepStack](https://github.com/NVIDIA/cosmos-framework/blob/0e034bc98ffa3c3dfa19f037871f3a8bbc1c4d05/cosmos_framework/model/generator/mot/unified_mot.py)。
该NVIDIA源码标注OpenMDW-1.1；不得误沿用其他Diffusers文件的Apache许可。
包装器消费明确预处理后的packed pixels，不隐式下载processor，不实现Nemotron/
SigLIP2 Edge视觉分支，也不将此包的音频latent head说成已经拥有波形codec。

共享模型制品新增独立`runtime_buffers.pt`及hash/dtype/schema检查。官方
`state_dict`参数键不变，但`model.to(BF16)`已舍入的RoPE等非持久语义buffer
被精确保存；重载不再从FP32频率错误重建。当前model buffers为定形语义状态，
动态推理cache属于独立typed state。`semantic_buffers(model)`只返回模块显式
`_aster_semantic_buffers`声明，额外负例验证任意动态buffer不会被自动打包。
旧schema1制品仍按原有配置重建语义读取，
不能声称它没有保存过的BF16辅助状态可精确恢复。

# Cosmos3 AVAE2与真实声画联合生成

`Cosmos3AudioConfig/Cosmos3AudioCodec`（`cosmos3_avae2`）完整实现公开AVAE2
波形编码器与解码器，不只是声学latent head：STFT实/虚频率与双声道排列、
SpecConvNeXt深度卷积、FP32 LayerNorm、SnakeBeta、显式weight_g/weight_v，
以及反序Oobleck上采样和1/3/9膨胀残差。奇数stride的output_padding不可省略。
相同公开参数键可strict加载；weight_norm为自主叶子公式，没有隐藏hook破坏ZeRO3。

`encode(waveform, force_pad=False)`返回`AudioGaussian`：后验
`std=softplus(scale)+1e-4`，不是视频后验logvar；默认公开KL是通常高斯KL的两倍，
按声学帧取mean。`decode(BDT|DT)`返回clamp[-1,1]的波形；`forward`返回
`(waveform, posterior)`，显式`sample_posterior`和generator控制采样。
Diffusers默认mode，NVIDIA上游采样行为需显式`sample_posterior=True`。
配置解析器核验真实Cosmos3-Nano的1920倍压缩、48kHz→25Hz声学latent；不自动
启用未实现causal/anti-alias/其他bottleneck，也不把decoder-only模型说成可编码。

源的`normalize_volume=True`取**整个batch**峰值，改变microbatch便改变输入。
因此`Cosmos3AudioAutoencoderObjective`明确要求False（归一化留给前置数据阶段），
用真实未clamp波形L1和FP32稳定声学KL，分别按波形标量数与latent帧数全局归一化。
这个目标是最小可训练验证配方，不声称复现原厂完整感知/对抗codec训练。

`Cosmos3AudioVideoPipeline`把Wan2.2视频codec、AVAE2音频codec与同一Cosmos3MoT
连接：图像理解可走真实Qwen视觉/DeepStack，声画latent参加同一次注意力和联合ODE。
声学位置采用`fps=sample_rate/hop, temporal_compression=1`，与视频同起点；
噪声长度严格按视频时长与hop校验，解码返回补齐的波形及采样率，不隐式重采样。
训练特征准备冻结两套codec，主模型继续用共享CE/逐模态flow和统一Trainer。
这一组合支持原生Euler/Heun/RK4，不宣称完整官方UniPC/CFG模板/蒸馏权重已执行。

验证包：13项覆盖FP32/BF16独立全图公式与全部参数/波形梯度、真实PyTorch
weight_norm算子、声道/填充/高斯语义、变时长梯度累积、ZeRO0/3真实损失下降和
随机后验精确恢复，Gloo DP2所有ZeRO0–3全局更新/恢复/非对称坏输入前置拒绝，
以及图像+原始视频+波形→共同训练→三制品重载→声画联合采样。额外测试改变声音
初噪声确实改变视频latent，排除两个孤立生成器的伪组合。小CPU示例的240Hz时间
坐标被显式标为toy；公开模型的48kHz参数未经大权重质量benchmark验证。

源锁定（Apache-2.0，公开权重许可需另按OpenMDW审计）：
[完整AVAE2](https://github.com/huggingface/diffusers/blob/c1bf18c92c6285334adcaac7e75ef8946a227f49/src/diffusers/models/autoencoders/autoencoder_cosmos3_audio.py)、
[声学高斯定义](https://github.com/huggingface/diffusers/blob/c1bf18c92c6285334adcaac7e75ef8946a227f49/src/diffusers/models/autoencoders/autoencoder_oobleck.py)、
[声画拼接与时间坐标](https://github.com/huggingface/diffusers/blob/c1bf18c92c6285334adcaac7e75ef8946a227f49/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py)、
[真实发布配置](https://huggingface.co/nvidia/Cosmos3-Nano/blob/7a312c868bcce8e40b3eb40861300a9d0ba3fde1/sound_tokenizer/config.json)。

# Qwen3-VL原始图像/视频：处理器、训练、缓存推理的真实连接

`data.qwen_vl`从**已解码RGB**开始，不调用AutoProcessor或torchvision运行时。
`RawImage`接受uint8 CHW Tensor、HWC NumPy或PIL；`RawVideo`接受uint8 TCHW及
必须明确的`VideoMetadata(fps,total_num_frames,frame_indices)`。模块不读取网络URL、
不隐式解码视频、猜FPS或重采样已有采样帧。采样索引保留原视频坐标，包括重复索引。

`QwenMediaConfig`显式固定smart resize预算、patch/temporal/merge大小、标准化、
PIL/Torch后端以及版本相关`video_cap_pixels_per_frame`。视频末帧重复补齐到temporal
patch，grid使用实际ceil(T/temporal)，不能抄用官方某个计数辅助函数的floor(T/temporal)
而与真实patchify输出冲突。时间组取第一/最后原索引的中点秒数，以源规定一位小数
文本放在各`vision_start/video_pad*/vision_end`前；不是把frame index直接塞进mRoPE。
PIL rescale的FP64中间结果与Torch fused FP32标准化分开实现，后端不同即身份不同。

`Qwen3VLProcessor(config,encode_text=...,tokenizer_id=...).prepare(examples,model.config)`
返回`PreparedQwenBatch`，内含真实packed像素/grid、右padding mask、视觉占位token、
对齐labels、三轴position_ids和rope_deltas。每个样本是显式
`[str | tuple[int,...] | RawImage | RawVideo]`；tuple表示调用者已定义的BOS/角色
token边界。文本分词器、词表与chat模板属于独立制品，不用byte测试词表假装兼容
Qwen公开checkpoint。视觉marker与padding的labels为-100，普通文本含timestamp默认
可监督；回答范围用显式loss_mask进一步筛选。相邻普通文本直到真正特殊token边界
才一起tokenize，避免随意拆分BPE边界。

`save_pretrained/from_pretrained`仅写/读严格JSON：包含全部处理选项、固定源revision、
tokenizer身份、显式模板边界与整体fingerprint，不序列化/执行callable。`PackedMedia`
记录原始/实际尺寸、真实帧索引、FPS/timestamps、后端及像素hash，不用文件名冒充输入身份。

`methods.qwen_vl.RawQwenObjective(processor,visual_prefill=...,objective=...,generation_fields=...)`
是统一Trainer的前置桥梁，复用原生CE或Cosmos3VisualFlowObjective，不另起优化器或
独立训练器。批次为`{'examples': [...], 'model_inputs': {...latent字段...}, 'noise': ...}`。
纯Qwen支持image/video/both/none；Cosmos3使用明确image/video/none路径和同一MoT的
vision/sound/action字段。目标身份锁定这些角色图。全累积窗口在任何视觉参数gather前
本地准备，并由Trainer对称汇总错误；某个rank不能悄悄缺图或改变生成域。

部署用已有`StatefulTokenRunner(model,policy_artifact_id=...,processor_id=processor.fingerprint)`：
把准备后的input_ids和其余model_inputs交给forward，后续decode只携带typed state。
真实mRoPE delta随状态fork/replay，不把视觉KV当普通文本长度；当前不是宣称已经为
Qwen多模态接通通用连续batch分页调度。Cosmos桥接则直接接已有VideoPipeline和
VisualFlowObjective，原始RGB理解与真实Wan2.2潜视频共同训练，之后复用理解前缀采样。

验证包括实际Transformers **5.16.1** PIL完整预处理逐像素严格相等；视频执行其真实
sample_frames/patchify/timestamps子图并独立核验Torch插值与cap公式（没有安装并冒充
执行torchvision整包）；同权重Qwen图像/视频logits、全部参数梯度、mRoPE与缓存生成；
模型/处理器安全重载、真实损失下降、精确恢复、不等长累积归一化；Gloo DP2的全部
ZeRO0–3全局更新和单rank坏媒体/缺图在model forward计数为0时一致失败；原始图像→
Cosmos视觉flow训练→Wan视频解码闭环。所有效果证据是小CPU模型测试，不是公开
Qwen大权重或视频理解benchmark分数。

锁定规范源码（Apache-2.0）与执行oracle版本分别记录，不混称同一快照：
[Qwen图像smart resize/pack](https://github.com/huggingface/transformers/blob/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)、
[Qwen视频采样与预算](https://github.com/huggingface/transformers/blob/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py)、
[占位token与时间戳](https://github.com/huggingface/transformers/blob/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)、
[Torchvision v0.24.0 bicubic参考](https://github.com/pytorch/vision/blob/v0.24.0/torchvision/transforms/v2/functional/_geometry.py)。

## LeWorldModel：联合梯度、全局统计与实际规划

新增`models/vit.py`和`models/lewm.py`，不是给既有EMA-JEPA换名。
`LeWMConfig.official_tiny(action_dim=10)`声明作者发布的192维、12层ViT、
6层动作AdaLN predictor、16个64维attention head与2048维BN投影，实数参数18,034,478
（不沿用论文摘要约15M作为实际计数）；普通默认配置
是测试小宽度。两者都走`build_model`、`config_from_dict`、`save_pretrained/load_model`。
ViT保留CLS、绝对位置双三次插值、GELU、前置LayerNorm和Q/K/V独立投影；
predictor特别保留外侧non-affine LN与attention/MLP内部affine LN，不因看似重复而删减。

模型入口`LeWorldModel.forward(pixels,actions)`要求`[B,H+1,C,W,W]`的显式预处理
图像与`[B,H,A]`标准化动作，返回`LeWMOutput(embeddings,predictions)`。
`encode`/`predict`/`rollout_latents`分别提供视觉表征、一步因果预测和滑动历史自回归。
`LeWMObjective`仅组合下一步embedding MSE与SIGReg：**目标没有stop-gradient，也没有
EMA教师**。SIGReg按`[T,B,D]`逐时间计算B个样本的经验特征函数，投影列单位化，
频率0..3用17点梯形积分，再乘B、平均时间与投影方向；源训练权重是0.09。
因此不能把T合并到B，也不能先算小批次SIGReg后求平均。

`LeWMMethod(engine,objective=...,seed=...,max_batch_bytes=...).update(chunks)`复用
统一Trainer/优化器/checkpoint。它在任何模型forward前对称检查输入，然后把整个
logical batch按rank→chunk→row顺序合并，每个DP replica计算同一完整批次和投影。
这是**复制全局激活的数学reference**，不是节省显存的SyncBN性能实现。
引擎accumulation必须固定为1；DP绕过Method直接平均本地目标会明确拒绝。
TP/PP/CP/EP/ETP/GTP未支持，不能悄悄fallback成不同统计目标。BN统计父节点与
可分片affine叶子分离，ZeRO3只重算纯affine、不二次更新running statistics；
公开权重仍是标准`weight/bias/running_mean/running_var/num_batches_tracked`。

`data/lewm.py`复用ActionSpec/ActionNormalizer，并按作者使用**无偏动作std**。
`lewm_windows`每行只是一条episode：窗口可以终止于真实terminal observation，
但不能跨terminal/truncation；终止之后是padding，reset必须新起一行。
输入NaN和恒定动作维导致的零std直接拒绝，不默认把坏动作改成0。

`planning/lewm.py`的`LeWMCEM.solve(pixels,goal_pixels,init_action=...)`和`LeWMMPC.act`
复用同一个原生模型。CEM与固定stable-worldmodel 0.0.5源码保持以下细节：
高斯候选没有裁剪；候选0总是当前均值；elite做普通平均与无偏std（源变量虽叫var，
存的其实是std）；最终返回elite均值而不是最佳单条候选；目标成本是末步latent的
**sum平方差**。MPC只把未执行的均值尾部用于warm-start，每次重置std；action block
拆成连续控制动作而不是重复同一动作。rng、动作队列、warm-start可以精确恢复。
规划临时eval后恢复每个子模块原有train/eval状态，不更新BN。
源H>1候选会优化整个历史窗口的动作，本实现明确保留该语义，不能冒充固定真实
历史动作的规划变体；小闭环使用H=1。实体控制仍需独立安全限幅/坐标系审查。

验证证据：真实Transformers **5.16.1** ViT同权重forward/全参数梯度，包括动态
分辨率和CPU BF16 autocast；实际Torch BN多轮全梯度与running统计；哈希锁定的作者
完整JEPA子图、全部参数/图像/动作/target梯度、滑动rollout和SIGReg同RNG逐值；
固定官方CEM逐轮候选、最终actions/std/elite_cost严格相等。原生DP2的ZeRO0–3
更新与全局大batch一致，ZeRO3新实例恢复连BN与方法RNG都严格相等，单rank坏输入
在模型forward前全rank失败。旧HF4.57 ViT键名有显式无碰撞映射与全schema预检，
但未把旧版整包或公开大权重加载声称为已验证；运行oracle仍是5.16.1。

`examples/lewm_pipeline.py`用共享TensorTreeDataset/StatefulSampler训练、保存采样
游标、发布ArtifactStore并重载后执行独立图像目标控制。默认训练600步；报告同时
保留每个目标的误差、成功阈值和汇总指标，不能只看均值。固定seed743的一次本地
CPU运行：loss 2.3462→0.4854，heldout latent std 0.8315，4个目标误差均值
1.125→0.0445，阈值0.15成功4/4，累计return均值−0.4352（零动作−9）。
不同CPU数学后端可能产生不同的训练与规划轨迹；这些数字是该次运行记录，
不是逐位复现或其他硬件上的质量保证。
这是声明范围内的一维渲染环境，不是PushT/TwoRooms/Reacher公开成绩。训练数据
覆盖完整可达动作区间；此前仅邻域动作数据在远目标规划失败，说明低latent MSE
本身不能证明控制性能。没有下载公开大权重或官方环境数据，也没有声称GPU吞吐。

官方来源与版本：
[LeWM论文](https://arxiv.org/html/2603.19312v1)、
[作者架构及SIGReg](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py)、
[作者联合模型与rollout](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/jepa.py)、
[官方依赖的CEM](https://github.com/galilai-group/stable-worldmodel/blob/3a85ac6888c39db90af648993fd0b23ac4c0a51d/stable_worldmodel/solver/cem.py)、
[MPC队列](https://github.com/galilai-group/stable-worldmodel/blob/3a85ac6888c39db90af648993fd0b23ac4c0a51d/stable_worldmodel/policy.py)、
[旧HF ViT布局](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/vit/modeling_vit.py)。
LeWM为MIT ©2026 Lucas Maes；stable-worldmodel为MIT ©2025 AI.QED Group @ Brown；
ViT来源为Apache-2.0。作者module/jepa与CEM源码SHA256在oracle测试中逐文件固定；
测试默认不访问网络，只有显式`ASTER_RUN_REMOTE_LEWM_ORACLE=1`执行锁定源码定义。

## Gemma4 DSpark 独立草稿结构

`models/dspark_gemma4.py`提供`Gemma4DSparkConfig(target=Gemma4TextConfig(...),...)`
与`Gemma4DSparkDraft`，架构标识`dspark_gemma4`。中央factory、配置与安全制品读写
使用lazy route；它不是Qwen3类的继承或别名。固定作者分支明确拒绝MoE与PLE，
所以目标需`enable_moe_block=False, hidden_size_per_layer_input=0`；不能悄悄删掉
公开Gemma4某些权重分支再称完整目标模型兼容。

草稿每层都是真正全局attention，使用`global_head_dim`；K=V时选择全局KV头数，
否则使用普通GQA头数。Q/K是带scale的FP32 RMS，V是无scale的RMS，attention
scale固定为1，采用global proportional RoPE。每层保留四处RMS、两次残差和
`layer_scalar`。目标的共享KV结构**不传给草稿**；只有MLP保留作者配置继承规则：
`threshold=draft_layers-target_shared_layers`，仅在`index>=threshold>0`时加宽。
阈值为零/负数并不意味着所有草稿MLP都加宽。

`initialize_from_target(target)`绑定真实目标embedding/head与完整权重身份。
`forward(input_ids,target_hidden_states,loss_mask,target_last_hidden_states=None,*,
anchor_positions=None,block_keep_mask=None)`返回共用`DSparkOutput`。
context特征为选中目标层拼接后经过共享fc/hidden_norm；每个block只读anchor之前
的context，并在自身block内双向attention，不能误改成普通causal mask。标签从
anchor+1开始，监督mask遇到首个空洞后取连续有效前缀。基础词表logits先softcap，
然后才加vanilla/gated/RNN Markov残差；teacher对齐logits只有相同softcap，
没有Markov。置信头复用共享实现的监督语义，概率接受率标签必须detach。

原生模型不依赖Transformers、DeepSpec、Flex或Triton runtime。CPU使用显式数学
attention，沿用作者Flex训练分支不施加attention-dropout的语义；它不是随意选择
其它fallback训练dropout。可选解冻embedding/head有独立全梯度对照；实际teacher
logit监督要求冻结目标head，禁止把变化中的head误当固定教师。

验证包括真实原生Gemma4 teacher产出特征，完整batch与不等长累积ZeRO0–3更新，
FP32/BF16随机anchor新实例精确续训，坏末microbatch前置拒绝，零监督mask有限值，
以及FP32/BF16模型制品重载逐值一致。这里的专项测试壳显式验证**global-window**
分母；不把它称为作者默认逐microbatch归一化训练。正式DSparkMethod的两种
归一化profile由方法层单独声明和测试。本模型包不声称服务吞吐或真实大权重
接受率已经验证。

`DSparkTeacherFeatures`现在只允许两种经过明确验证的配对：native Qwen3及其
DSparkConfig，或native Gemma4及Gemma4DSparkConfig。保持冻结教师副本、权重身份、
no-autocast提取和原来的history索引；Gemma4的embedding已经乘sqrt(width)，
不能在提取时重复放大。`extraction_profile`给发布者标记准确家族，不把Gemma
特征伪装成Qwen特征。特征缓存发布/校验与完整部署由相应方法和推理包负责。

`Gemma4DSparkDraft.backbone_cached(noise_ids,new_context_features,state=None)`返回
`(hidden, tuple[layer](K,V))`，每个叶子是`[B,KV_heads,S,global_head_dim]`，序列轴2。
该state是**草稿自己的独立全局context KV**，不是Gemma4目标的shared-owner/window
state。只投影新增教师特征，位置从缓存真实context长度延续；query在本block内
双向可见，但返回缓存时完全丢弃。缓存必须clone仅context，连底层storage都不
保留query；不能以一个切片view假装释放临时查询。空context、无新增context的
重新query、截短context后增量补齐都有原生FP32/BF16及K=V/独立KV验证，原缓存
不被修改。接口禁止训练中的模型和ambient-autocast精度漂移，拒绝错dtype、
非有限特征/状态；上层PagedStatePool负责制品身份和页所有权，当前数学attention
仍物化历史，不宣称是融合分页kernel。该hidden之后须经过`compute_logits`的
Gemma softcap，再加Markov，不能在通用执行器中直接调用裸lm_head。

`tests/integration/test_dspark_gemma4_pipeline.py`把上述模块实际接成小CPU闭环：
真实Gemma4教师→不可变特征cache→正式DSparkMethod默认
`official_microbatch_mean`→ZeRO0/3训练→新实例checkpoint精确恢复→带成功更新
receipt和cache lineage发布→安全加载→Gemma4 snapshot投机生成→成对baseline评价。
随机anchor恢复后的loss、每个模型tensor和语义runtime buffer逐值相等；正常
目标验证贪心token与softcap后的原始logprob一致。另用**明确未训练的合成fixture**
强制8次实际拒绝，检查丢失旧滑窗的目标确实执行8次完整prefix replay，而不是
伪truncate；取消、客户端回调异常后目标snapshot、草稿页与binding leases全释放。
小规模验证保持`public_quality=not_evaluated`和`deployment_promoted=False`，
成对延迟只是本机测量，不把它改写为生产加速比。新桥接+模型+集成21项通过，
6项实际作者oracle另包含cached增量直接对照作者完整backbone，合计27项此包证据。
制品配置自身先规范为JSON-compatible lists，不能放宽严格合同检查来掩盖tuple/
list落盘差异；这正是端到端缓存测试发现并修复的边界问题。

实际上游oracle固定DeepSpec提交`005e03b81cec38b7da6399833d609ee89a2587f2`，
执行作者config/model及共同Markov源定义，所有文件先核对SHA256。其requirements
指定**Transformers 5.10.2**，不是本机5.16.1：oracle另取固定5.10.2真实RMS、
MLP、RoPE、scaled embedding和PreTrainedModel子类数学源码。5.16.1仅提供通用
生命周期/配置外壳；明确恢复旧flat global-head字段并允许读取它们，文档装饰器
无操作，不替换计算。CPU oracle固定SDPA math后端，不宣称执行官方CUDA Flex。
覆盖K=V/独立KV、双宽阈值两侧、比例RoPE、softcap、三种Markov以及可训练head，
同权重对照全部输出和每个可训练参数梯度；默认测试不访问网络，执行需显式
`ASTER_RUN_REMOTE_DSPARK_ORACLE=1`。

官方来源：[DeepSpec配置](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/gemma4/config.py)、
[完整Gemma4草稿](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/gemma4/modeling.py)、
[作者依赖锁定](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/requirements.txt)、
[TF5.10.2 Gemma4](https://github.com/huggingface/transformers/blob/v5.10.2/src/transformers/models/gemma4/modeling_gemma4.py)、
[TF5.10.2比例RoPE](https://github.com/huggingface/transformers/blob/v5.10.2/src/transformers/modeling_rope_utils.py)。
DeepSpec为MIT，许可证见`docs/third_party/DEEPSPEC_LICENSE.txt`；Gemma4源为Apache-2.0。
