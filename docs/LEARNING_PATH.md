# Learn Aster by following one experiment

[Home](../README.md) · [简体中文首页](../README.zh-CN.md) · [Algorithm map](ALGORITHMS.md) · [Workflow gallery](EXAMPLES.md)

You do not need to read the whole framework first. Start with one small model, follow its tensors into a loss, watch which parameters change, then trace its state into inference. Each stop below pairs a question with code, a test, and an experiment.

Prerequisites: basic Python, tensor shapes, matrix multiplication, and the idea of gradients. [PyTorch's beginner tutorials](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) provide a starting point if those are new. The first six stops use CPU PyTorch; no pretrained model downloads are required.

## Your route

| Stop | Question | What to demonstrate |
| --- | --- | --- |
| 0. Run | What does a complete tiny workflow do? | An adapter updates; base weights stay fixed |
| 1. Model | How does a sequence become next-token scores? | Causal masking and the expected tensor shapes |
| 2. Adapt | Why can LoRA start without changing the model? | Zero update at initialization; merge agreement |
| 3. Train | Which denominator does a loss need? | Accumulated updates match the declared full objective |
| 4. Serve | What can safely be reused? | Cached/full agreement and released request state |
| 5. Generate | What is the model predicting? | A known field reaches the expected endpoint |
| 6. Learn from another model | What does “frozen” actually mean? | Correct teacher and input-gradient boundaries |
| 7. Scale | What is being split across workers? | Actual collective/update agreement |
| 8. Branch out | How do actions, world models, and tools connect? | An explicit input/output/ownership contract |

Run commands from the repository root. Keep experiments in your own branch or scratch copy; change one setting at a time and retain the original test as your reference.

## 0. See the entire loop first

~~~bash
python -m pip install -e ".[test]"
python -m aster doctor
python examples/quickstart.py
~~~

Read [quickstart.py](../examples/quickstart.py). It builds a small random Llama-shaped model, adapts its output projection, performs eight updates, and checks merge equivalence. It is deliberately a mechanics experiment, not a useful pretrained chatbot.

Look for `base_unchanged: true`, `updates: 8`, fewer trainable than total parameters, and a small `merge_max_absolute_error`. Do not require an exact printed decimal across platforms.

**Try:** change only the adapter rank from 4 to 2. Explain why the trainable parameter count changes. A smaller count alone does not prove better quality or speed.

## 1. Trace the shapes before reading model variants

Read [gpt.py](../src/aster/models/gpt.py), then [attention.py](../src/aster/nn/attention.py). After that, compare the rotary-position decoder in [decoder.py](../src/aster/models/decoder.py).

For batch size B, sequence length T, hidden size D, and vocabulary size V, follow:

~~~text
input_ids [B, T] -> hidden states [B, T, D] -> logits [B, T, V]
~~~

Logits are scores before softmax. In causal next-token training, a position predicts the following token; labels, masks, and shifts must describe the same positions. A causal mask prevents earlier predictions from using later input tokens.

~~~bash
python -m pytest tests/unit/test_models_gpt.py tests/unit/test_models_decoder.py -q
~~~

**Try:** inspect the causality test, change only a future token, and compare earlier logits. Then locate the check that blocks an incompatible cached state. Understanding one decoder's state contract is more useful than memorizing model aliases.

## 2. Understand LoRA with one equation

Read `LoRALinear`, `inject_lora`, and `merge_lora` in [distillation.py](../src/aster/methods/distillation.py).

For a linear layer with weight W, Aster uses:

~~~text
W_effective = W + (alpha / rank) * B @ A
~~~

A has shape [rank, input_features]; B has shape [output_features, rank]. W is frozen, A is initialized, and B starts at zero. Therefore B @ A initially adds nothing to the original projection. Gradients can first change B and subsequently both adapter factors.

~~~bash
python -m pytest tests/unit/test_methods.py -k lora -q
python -m pytest tests/unit/test_repository.py -k quickstart -q
~~~

**Try:** compare base, unmerged, and merged outputs before and after training. Use evaluation mode; stochastic adapter dropout changes the function during training. The test uses a floating-point tolerance, not bitwise equality. See [fine-tuning](FINE_TUNING.md) for supported targets and methods that are not implemented.

## 3. Do not average averages blindly

Read `LossTerm` in [contracts.py](../src/aster/core/contracts.py), the target masks in [supervised.py](../src/aster/methods/supervised.py), and reductions in [trainer.py](../src/aster/training/trainer.py).

Suppose one microbatch has 2 valid tokens with mean loss 1, and another has 20 valid tokens with mean loss 3:

~~~text
Token-weighted mean = (2 * 1 + 20 * 3) / (2 + 20) = 62 / 22
Mean of means       = (1 + 3) / 2                = 2
~~~

These are different objectives. For a global token mean, accumulate loss sums and valid-token counts. A separate image-, pair-, or sample-level term needs its own denominator.

~~~bash
python -m pytest tests/unit/test_training.py -k "independent_denominators or zero_count" -q
~~~

**Try:** change the mask in the test, compute the denominator by hand, and compare the full-window update with the accumulated update. A window with zero valid targets must not take an optimizer step or apply weight decay.

A mathematical reduction check can establish the declared objective. It cannot establish multi-node throughput. Continue with [loss contracts](LOSSES.md).

## 4. Follow a request's cached state

Read the full-versus-cached test in [test_inference.py](../tests/unit/test_inference.py), then [state.py](../src/aster/inference/state.py) and [engine.py](../src/aster/inference/engine.py).

KV means the attention keys and values retained from previous tokens. Reusing them saves repeated computation only when weights, token positions, processing rules, and cache identity still match. A different adapter or tenant must not accidentally receive another request's state.

~~~bash
python -m pytest tests/unit/test_inference.py -k "cache_matches_full or pages_hold or prefix_complete" -q
python examples/online_adapter_stack.py --kv int8
~~~

**Try:** follow the base-model and adapter request identities in [the example](../examples/online_adapter_stack.py). At shutdown, inspect `remaining_pages`, `remaining_host_bytes`, and `remaining_adapter_bytes`: all should be zero.

Full/cached equality tests use their declared precision and tolerances. INT8 state is a lossy representation; do not extend exact-cache conclusions to quantized quality. See [paged attention](PAGED_ATTENTION.md).

## 5. Separate the generative target from the sampler

Read `FlowPath`, `FlowObjective`, and `sample_flow` in [generation.py](../src/aster/methods/generation.py). First use the simple noise-to-data convention:

~~~text
x(t) = (1 - t) * noise + t * data
target velocity = data - noise
~~~

Aster also supports other time conventions. Reversing the path without changing the target sign or integration direction is an error. Diffusion epsilon/x0 targets and preconditioned EDM residuals are not velocity aliases.

~~~bash
python -m pytest tests/unit/test_generative.py -k "flow_time or parameterizations" -q
~~~

**Try:** read the constant-field fixture. Euler, Heun, and RK4 should reach its known endpoint. Change the direction using the fixture's paired sign. This validates integration conventions; it says nothing about the quality of a learned field.

Then follow [generation and compression](ALGORITHMS.md#generative-models-and-compression). A sensible study sequence is baseline sampler → one distillation method → fewer steps → optional caching → fixed quality comparison. Fewer network calls do not by themselves prove equal quality or lower end-to-end latency.

## 6. Inspect the gradient boundary

Read [distillation.py](../src/aster/methods/distillation.py), the role-freezing logic in [trainer.py](../src/aster/training/trainer.py), and [methods](METHODS.md).

In ordinary token distillation, teacher logits are fixed targets. In actor/critic or generator/score compositions, a frozen network may still need to differentiate its output with respect to its input. Freezing parameters and placing an entire forward pass inside `no_grad` are therefore not interchangeable.

~~~bash
python -m pytest tests/unit/test_methods.py -k "kl_orientation or shared_engine" -q
python -m pytest tests/unit/test_training.py -k freeze_preserves_input_gradient -q
~~~

**Try:** trace which tensors should have gradients before changing any code. Then explain why swapping teacher/student arguments changes forward versus reverse KL. Check the temperature scaling in the test rather than copying a generic KL snippet.

Run the teacher → student recipe in [the gallery](EXAMPLES.md#3-teacher-to-student-to-evaluation) when you are ready to connect the pieces.

## 7. Scale the same contract

Read [training](TRAINING.md) before launching processes.

| Mechanism | What is split? | Question to ask |
| --- | --- | --- |
| Data parallelism | Samples across replicas | Which counts/gradients are globally reduced? |
| Tensor parallelism | Tensor operations/parameters within a layer | Which axes and collectives reconstruct the result? |
| Pipeline parallelism | Layers/stages across workers | Who owns each microbatch's activations and gradients? |
| ZeRO | Optimizer state, gradients, and/or parameters | Where is the authoritative value before export or resume? |

Tensor parallelism and ZeRO describe different axes of ownership and can be composed in supported layouts. They are not two independent trainers stepping the same weights.

~~~bash
python -m pytest tests/distributed/test_training_causal_parallel.py -q
~~~

This test starts real processes and needs a working distributed backend. Read [test requirements](TESTING.md); a skip is not a distributed pass, and CPU collectives are not multi-node GPU evidence.

**Try:** follow the reference unsharded update and the distributed update in the test. Write down the logical parameter name, owner, and reduction group for one projection. Do not assume a layout valid for Llama also supports every world or action model.

## 8. Choose a branch

| Interest | Read first | Small reading/verification entry |
| --- | --- | --- |
| Vision-language | [Qwen-VL model](../src/aster/models/qwen_vl.py) | [Preprocessing tests](../tests/unit/test_models_qwen_preprocessing.py): token/image alignment |
| Robot actions | [Action models](../src/aster/models/actions.py) | [Action/world tests](../tests/unit/test_world_actions.py): chunk shapes and masked targets |
| World prediction | [LeWM](../src/aster/models/lewm.py) | [LeWM tests](../tests/unit/test_models_lewm.py): prediction, state, and gradients |
| Search and planning | [MCTS](../src/aster/planning/mcts.py) | [Search tests](../tests/unit/test_mcts.py): legal actions and backed-up values |
| Tool-using agents | [Agent lifecycle](AGENTS.md) | [Agent tests](../tests/unit/test_agents.py): exact approval and durable receipts |
| Reproducible evaluation | [Benchmark protocols](BENCHMARKS.md) | [Evaluation tests](../tests/unit/test_evaluation_suites.py): fixed cohorts and failure counts |

Do not connect a robot, permit external tool writes, or download restricted datasets just to follow a learning exercise. Those require their own environment and permission decisions.

## A small glossary

| Term | In this repository |
| --- | --- |
| Logits | Model scores before conversion to probabilities |
| Objective | A loss definition plus masks, counts, and gradient rules |
| Gradient | How a small change in a differentiable value changes the objective |
| Oracle | An independent reference calculation used for comparison, not a quality certificate |
| Artifact | Exported model content plus identity/processing metadata |
| Checkpoint | State required to resume training, including more than model weights |
| Cache | Reused computation/state whose validity depends on an explicit identity |
| Quality gate | A declared acceptance rule for a fixed evaluation protocol |

## Further learning and contributing

The chapter-to-code navigation in [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch), concept-to-source organization in [LabML's annotated implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations), and background/algorithm/exercise structure in [Spinning Up](https://github.com/openai/spinningup) informed this guide's organization. Their content and implementations remain their authors' work.

A good first contribution is one precise explanation or one counterexample: show the expected tensor shape, the invariant, the smallest reproducible input, and the test. Use English for code comments and public API docstrings. See [contributing](../CONTRIBUTING.md).
