# Fine-tuning

## Current support

| Method | Status | Interface / boundary |
| --- | --- | --- |
| Full-parameter fine-tuning | Implemented | Native model + objective + shared Trainer |
| Freeze selected parameters | Implemented | Explicit requires_grad flags before Trainer construction |
| Linear LoRA | Implemented | LoRALinear / inject_lora; exact named Linear targets |
| LoRA merge | Implemented | merge_lora returns an independent evaluation model |
| Online multi-LoRA | Implemented subset | Llama/Qwen2/Qwen3 dense or paged single-worker runners |
| QLoRA / NF4 training | Not implemented | Packed inference weights and QAT are not QLoRA |
| DoRA | Not implemented | No magnitude/direction decomposition |
| rsLoRA | Not implemented | Existing scaling is alpha/r, not alpha/sqrt(r) |
| IA³ / prefix / prompt tuning | Not implemented | No corresponding trainable adapters |

The native implementation is not a wrapper around PEFT and does not promise PEFT checkpoint compatibility.

## LoRA formula and ownership

For a base matrix W, LoRA computes:

~~~text
y = x W^T + (alpha / rank) dropout(x) A^T B^T
A: [rank, input_features]
B: [output_features, rank]
~~~

Base parameters are frozen. A is Kaiming-initialized; B starts at zero, so injection initially preserves the base function. Only the adapter branch receives dropout.

Pass exact module paths: target selection does not guess projection names or match arbitrary suffixes. Inspect model.named_modules() when changing architecture.

~~~python
from aster.methods import CrossEntropyObjective, inject_lora, merge_lora
from aster.training import Trainer

model = inject_lora(model, targets=["lm_head"], rank=4, alpha=8.0)
trainer = Trainer(model, CrossEntropyObjective(), lr=1e-3)
result = trainer.step([batch])
merged = merge_lora(trainer.model)
~~~

A complete runnable example is [quickstart.py](../examples/quickstart.py).

## Training and deployment are connected

[online_adapter_stack.py](../examples/online_adapter_stack.py) trains with the same native Trainer, verifies the frozen base, and registers the learned A/B matrices in MultiLoRARunner.

Registration checks the base configuration and every base parameter. Adapter content identities isolate prefix caches. Queued, running, and preempted requests keep adapters pinned until completion; unloading a pinned adapter fails.

The HTTP interface selects host-registered model identities. It does not accept arbitrary adapter paths, download weights, or merge into a shared base between requests.

## Practical boundaries

- Validate all target names and hyperparameters on a disposable model before constructing the trainer.
- Inject adapters before the optimizer is created; otherwise the optimizer may not own them.
- Use merge_lora for an independent deployment copy. A stochastic dropout branch is not an evaluation-time merge.
- Checkpoint the shared trainer for exact resume. A merged model does not preserve optimizer state.
- Do not assume every parallel layout, quantized base, tied projection, or multimodal family supports adapter injection.
- Evaluate merged and unmerged results with identical preprocessing and weights.

Primary references: [LoRA](https://github.com/microsoft/LoRA), [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora), and [vLLM LoRA serving](https://docs.vllm.ai/en/v0.18.2/features/lora/).
