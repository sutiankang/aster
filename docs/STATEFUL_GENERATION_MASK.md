# 原生snapshot生成的历史mask生命周期

本包修复`StatefulTokenRunner.generate`把所有`modality_inputs`都当作一次性prefill
数据的问题：图像／视频应只输入一次，但历史`attention_mask`必须贯穿后续cached
decode，否则已有padding位置会重新进入注意力，Qwen-VL还可能使用错误的位置坐标。

只读复现使用seed55的原生Qwen3Next两层（linear／full）、prompt
`[0,0,1,4,6]`及mask`[0,0,1,1,1]`。原实现第二token的原始logp为
`-3.1771414`，保留完整mask的重算值为`-3.1965160`；虽然该例greedy token碰巧
相同，概率已经发生变化，不能据“输出文字一样”判断缓存正确。

现在生成前按确切原生类型验证物理token协议，把已展开prompt对应的二值mask
复制到请求内部。每次追加一个生成token，就增加一个有效位置，并把完整mask传给
模型；位置坐标仍由各模型自己的原生state／rope_delta生成，不在通用runner中猜测。
原始prompt IDs保留，不擅自删除padding或改写既有sampling transform。

## 明确边界

- mask必须是模型设备上的固定二值`[1,prompt_length]`；末尾须为有效query。
  trailing padding不自动裁掉，不从padding query的logits采样。
- 自定义`position_ids`、`cache_position`、token-type／额外bias等未声明的跨步
  语义在首次forward前拒绝。需要这些能力应使用显式`forward`逐步提供参数。
  `position_ids=None`和`attention_mask=None`与省略同义。
- Qwen3-VL和Gemma4图像／视频字段按各自明确列表只用于prefill，后续只传token、
  native state和完整mask。不扩大Gemma专属runner或DSpark的功能范围。
- 已验证物理长度等于input_ids长度的原生dense／window／MLA／DSA／hybrid／Mamba／
  GPT2／Qwen3-VL／Gemma4类型才允许此mask生成协议。插入额外视觉位置的OpenVLA等
  不能仅凭返回dense_kv就套用本协议；未知类型带mask提前拒绝。
- DeepSeekV4当前原生模型只允许无padding输入；本入口允许全1mask，拒绝包含0。
- 原有无mask生成、手工`forward`／`replay`接口不变。本包不代表hybrid state
  已进入连续批处理，也不代表新增GPU性能或质量benchmark验收。

## 验证

`tests/unit/test_inference_stateful_generation.py`包含九类文本模型、left／interior
padding、至少四步的全logits／原始logp／greedy对照，以及Qwen图像、Gemma图像与
视频三步对照。forward hook核实每步实际mask和一次性媒体字段；回调修改调用者的
原mask／prompt不改变在途请求。非法shape／binary值／grad／complex／无效末尾、
未知物理布局和自定义位置在零模型调用时失败。

公式依据为[Transformers Qwen3Next v5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3_next/modeling_qwen3_next.py)
中的实际mask传递与缓存逻辑；本次计算和测试使用本仓库模型，没有调用官方模型
runtime、下载权重或运行公开benchmark。Qwen3-VL位置来自本仓库已检验的每请求
rope_delta状态，没有以普通文本绝对位置替换多模态位置。
