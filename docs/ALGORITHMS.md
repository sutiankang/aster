# Algorithms, papers, and implementation map

[Home](../README.md) · [Learning path](LEARNING_PATH.md) · [Runnable workflows](EXAMPLES.md) · [Full model catalog](MODELS.md)

Find a concept, open its implementation, and read the corresponding test beside it. The tables are a curated reading map, not a claim that every upstream feature, checkpoint, or kernel is supported. Model variants and exact configuration limits live in [MODELS](MODELS.md) and the [capability manifest](scope/capabilities.json).

**How to read the evidence.** A linked test is an entry point, not a statement that it ran on every platform. Unit tests check formulas and invariants; integration tests connect components; distributed tests require real communicating processes; optional parity tests need their declared reference packages or source files. Hardware speed and trained-model quality require separate [benchmark protocols](BENCHMARKS.md). A source link credits an idea or implementation reference; it does not establish licensing clearance or universal numerical equivalence. See [NOTICE](../NOTICE.md).

## Contents

- [Attention and language models](#attention-and-language-models)
- [Training, optimizers, and parallelism](#training-optimizers-and-parallelism)
- [Fine-tuning, losses, and distillation](#fine-tuning-losses-and-distillation)
- [Inference and cache systems](#inference-and-cache-systems)
- [Generative models and compression](#generative-models-and-compression)
- [Multimodal and action models](#multimodal-and-action-models)
- [World models and planning](#world-models-and-planning)
- [Reinforcement learning](#reinforcement-learning)
- [Agents and evaluation](#agents-and-evaluation)

## Attention and language models

Start with a dense decoder before comparing sparse or recurrent alternatives. Different attention families carry different state; their caches are not interchangeable.

| Concept and idea | Native implementation | Reading test | Primary reference |
| --- | --- | --- | --- |
| GPT-2: causal token prediction with learned positions | [gpt.py](../src/aster/models/gpt.py) | [GPT tests](../tests/unit/test_models_gpt.py) | [Transformers GPT-2](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt2) |
| Llama-style decoder: RMSNorm, rotary positions, grouped KV heads | [decoder.py](../src/aster/models/decoder.py), [position.py](../src/aster/nn/position.py) | [Decoder/cache tests](../tests/unit/test_models_decoder.py) | [Transformers Llama](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama) |
| DeepSeek MLA: cache a compressed latent and absorb projections where valid | [latent_attention.py](../src/aster/nn/latent_attention.py) | [MLA absorption and storage](../tests/unit/test_models_decoder.py) | [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) |
| Mixture of experts: route tokens to a subset of experts; distinguish selection bias from mixing weights | [experts.py](../src/aster/nn/experts.py) | [Router behavior](../tests/unit/test_models_decoder.py) | [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) |
| DeepSeek sparse attention: learn an indexer and restrict the attention candidate set | [sparse.py](../src/aster/nn/sparse.py), [DSA objective](../src/aster/methods/sparse_indexer.py) | [DSA training](../tests/unit/test_dsa_training.py) | [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| Gated DeltaNet / KDA: update a recurrent matrix state instead of retaining every token's KV | [delta.py](../src/aster/nn/delta.py), [kda.py](../src/aster/nn/kda.py) | [Hybrid models](../tests/unit/test_models_hybrid.py) | [FLA reference implementations](https://github.com/fla-org/flash-linear-attention), [Kimi Linear](https://github.com/MoonshotAI/Kimi-Linear) |
| Mamba: selective state-space recurrence with explicit state handling | [ssm.py](../src/aster/nn/ssm.py), [mamba.py](../src/aster/models/mamba.py) | [Mamba tests](../tests/unit/test_models_mamba.py) | [Mamba](https://github.com/state-spaces/mamba) |
| Multi-token prediction: auxiliary future-token heads with separate valid-target counts | [mtp.py](../src/aster/methods/mtp.py) | [MTP objective](../tests/unit/test_mtp_objective.py) | [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) |

For Qwen, Kimi, DeepSeek, Gemma, encoder/decoder families, multimodal variants, and checkpoint mapping details, use the [complete model notes](MODELS.md). A matching architecture does not supply pretrained weights.

## Training, optimizers, and parallelism

The training design combines model-parallel decomposition with data-parallel state sharding. This is a native implementation of supported contracts, not an embedded Megatron or DeepSpeed runtime. Not every combination is valid for every model; consult [training](TRAINING.md).

| Concept and idea | Native implementation | Reading test | Primary reference |
| --- | --- | --- | --- |
| Loss-aware accumulation: sum losses and counts before taking the global mean | [trainer.py](../src/aster/training/trainer.py) | [Independent denominators](../tests/unit/test_training.py) | [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html) |
| Tensor parallelism: split projections while preserving forward and backward collectives | [parallel.py](../src/aster/training/parallel.py), [causal_parallel.py](../src/aster/training/causal_parallel.py) | [Multi-process causal TP](../tests/distributed/test_training_causal_parallel.py) | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| Pipeline parallelism: stage the network and schedule microbatches | [pipeline.py](../src/aster/training/pipeline.py), [causal_pipeline.py](../src/aster/training/causal_pipeline.py) | [Pipeline training](../tests/distributed/test_training_causal_pipeline.py) | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| ZeRO stages 1–3 and offload: shard optimizer state, gradients, then parameters | [sharding.py](../src/aster/training/sharding.py) | [Stage/update/resume checks](../tests/unit/test_training.py) | [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) |
| Sequence/context parallelism: distribute sequence work and combine attention statistics | [sequence.py](../src/aster/training/sequence.py), [ring.py](../src/aster/training/ring.py) | [Attention/gradient checks](../tests/unit/test_training.py) | [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) |
| Expert parallelism: exchange routed tokens and reconcile expert updates | [moe_parallel.py](../src/aster/training/moe_parallel.py) | [Multi-process MoE](../tests/distributed/test_training_moe_parallel.py) | [Megatron MoE](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe) |
| Muon: orthogonalize matrix updates; keep non-matrix parameter rules explicit | [muon.py](../src/aster/training/muon.py) | [Muon equations](../tests/unit/test_training_muon.py), [recipes](../tests/unit/test_training_muon_recipe.py) | [Muon](https://github.com/KellerJordan/Muon) |
| FP8: scaled low-precision representation with explicit numerical contracts | [fp8.py](../src/aster/training/fp8.py) | [FP8 checks](../tests/unit/test_training_fp8.py) | [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) |
| Recompute, EMA, and exact resume: recover the next update, not only model weights | [trainer.py](../src/aster/training/trainer.py) | [RNG/EMA/checkpoint tests](../tests/unit/test_training.py) | [PyTorch checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html) |

FP8 representation/formula checks do not demonstrate Tensor Core throughput. A single-rank ZeRO test does not substitute for the multi-process tests.

## Fine-tuning, losses, and distillation

[Fine-tuning](FINE_TUNING.md) explains supported projection targets and adapter ownership. [Losses](LOSSES.md) records masks, reduction units, and gradient boundaries.

| Concept and idea | Native implementation | Reading test | Primary reference |
| --- | --- | --- | --- |
| LoRA: add a trainable low-rank update to a frozen linear projection; optionally merge it | [LoRALinear and merge](../src/aster/methods/distillation.py) | [Injection/merge invariance](../tests/unit/test_methods.py) | [Microsoft LoRA](https://github.com/microsoft/LoRA) |
| Supervised cross-entropy: align prediction/target positions and count valid targets | [supervised.py](../src/aster/methods/supervised.py) | [Native CE and shared engine](../tests/unit/test_methods.py) | [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) |
| DPO / IPO / SimPO: optimize preferred versus rejected responses with method-specific reference/normalization rules | [preference.py](../src/aster/methods/preference.py) | [Preference sanity check](../tests/unit/test_methods.py) | [TRL objectives](https://github.com/huggingface/trl), [SimPO](https://github.com/princeton-nlp/SimPO) |
| Token knowledge distillation: align teacher/student distributions with explicit KL direction and temperature | [distillation.py](../src/aster/methods/distillation.py) | [KL orientation/gradient](../tests/unit/test_methods.py) | [Distillation paper](https://arxiv.org/abs/1503.02531) |
| TinyBERT / MiniLM: transfer hidden states, attention scores, or head relations | [encoder_distillation.py](../src/aster/methods/encoder_distillation.py) | [Encoder distillation](../tests/unit/test_encoder_distillation.py) | [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT), [MiniLM](https://github.com/microsoft/unilm/tree/master/minilm) |
| On-policy distillation: let the student generate contexts, then obtain teacher supervision | [rollout_distillation.py](../src/aster/methods/rollout_distillation.py) | [Rollout → update → resume](../tests/integration/test_rollout_distillation.py) | [TRL GKD](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py) |
| DSpark draft training: combine the draft's declared classification/regression objectives | [dspark.py](../src/aster/methods/dspark.py) | [DSpark losses](../tests/unit/test_dspark_training.py), [normalization](../tests/unit/test_dspark_normalization.py) | [DeepSpec / DSpark](https://github.com/deepseek-ai/DeepSpec) |

Standard linear LoRA is implemented. QLoRA/NF4 training, DoRA, rsLoRA, IA³, prompt tuning, and prefix tuning are not implied by this entry.

## Inference and cache systems

Follow one request from admission to cleanup. Correct outputs are only part of the contract: memory ownership, identity isolation, cancellation, and error handling matter too.

| Concept and idea | Native implementation | Reading test | Primary reference |
| --- | --- | --- | --- |
| Online softmax attention: merge block maxima, normalizers, and weighted values without a full attention matrix | [online_attention.py](../src/aster/optimization/online_attention.py) | [Dense/online comparison](../tests/unit/test_inference_online_attention.py) | [FlashAttention](https://github.com/Dao-AILab/flash-attention) |
| Fused attention kernels: combine attention work in a supported native Triton path | [fused_attention.py](../src/aster/optimization/fused_attention.py), [kernel](../src/aster/optimization/_triton_attention.py) | [Conditional GPU tests](../tests/unit/test_optimization_fused_attention.py) | [FlashAttention](https://github.com/Dao-AILab/flash-attention), [Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) |
| Paged KV / copy-on-write / prefix reuse: share immutable history while isolating writes and identities | [state.py](../src/aster/inference/state.py), [paged_attention.py](../src/aster/inference/paged_attention.py) | [Page lifetime and cache isolation](../tests/unit/test_inference.py) | [vLLM](https://github.com/vllm-project/vllm) |
| Continuous batching and chunked prefill: admit requests into a bounded token budget | [engine.py](../src/aster/inference/engine.py) | [Batching, backpressure, cancellation](../tests/unit/test_inference.py) | [vLLM scheduler](https://github.com/vllm-project/vllm/tree/main/vllm/v1/core/sched) |
| Quantized KV and host swapping: reduce stored state and move inactive pages with explicit ownership | [kv_quantization.py](../src/aster/optimization/kv_quantization.py), [offload.py](../src/aster/inference/offload.py) | [Paged offload](../tests/unit/test_paged_offload.py) | [vLLM](https://github.com/vllm-project/vllm) |
| Multi-LoRA serving: share a base model without sharing incompatible adapter cache state | [adapters.py](../src/aster/inference/adapters.py) | [Online LoRA](../tests/unit/test_online_lora.py) | [vLLM LoRA](https://github.com/vllm-project/vllm/tree/main/vllm/lora) |
| Speculative decoding: accept draft tokens and correct rejections against a target distribution | [speculative.py](../src/aster/inference/speculative.py) | [Target-distribution check](../tests/unit/test_inference.py) | [Speculative decoding paper](https://proceedings.mlr.press/v202/leviathan23a.html) |
| DSpark inference: integrate the draft, target verification, and supported cache contracts | [dspark.py](../src/aster/inference/dspark.py) | [DSpark inference](../tests/unit/test_dspark_inference.py) | [DeepSpec / DSpark](https://github.com/deepseek-ai/DeepSpec) |

An exact online-softmax identity is not a speed measurement. Low-bit cache formats introduce numerical error; approximate draft/cache paths must state their supported acceptance and state rules. See [inference](INFERENCE.md), [paged attention](PAGED_ATTENTION.md), and [optimization](OPTIMIZATION.md).

## Generative models and compression

A noise predictor, a clean-data predictor, a velocity field, an EDM residual, and a consistency residual are different output contracts. Match the model, objective, schedule, and solver before composing optimizations.

| Concept and idea | Native implementation | Reading test | Primary reference |
| --- | --- | --- | --- |
| DDPM / DDIM: learn denoising targets and sample with explicit schedules and parameterizations | [generation.py](../src/aster/methods/generation.py) | [Schedules and perfect-predictor checks](../tests/unit/test_generative.py) | [OpenAI improved diffusion](https://github.com/openai/improved-diffusion), [DDIM](https://github.com/ermongroup/ddim) |
| UNet / DiT: convolutional versus patch-transformer denoising backbones | [generative.py](../src/aster/models/generative.py) | [Shared training loop](../tests/unit/test_generative.py) | [Guided diffusion](https://github.com/openai/guided-diffusion), [DiT](https://github.com/facebookresearch/DiT) |
| EDM: precondition the denoiser and integrate over a noise-level schedule | [EDM objective and sampler](../src/aster/methods/generation.py) | [EDM gradients](../tests/unit/test_generative.py) | [NVIDIA EDM](https://github.com/NVlabs/edm) |
| Flow matching: regress a conditional path's velocity and integrate its field | [FlowPath and FlowObjective](../src/aster/methods/generation.py) | [Time conventions/integrators](../tests/unit/test_generative.py) | [Meta Flow Matching](https://github.com/facebookresearch/flow_matching) |
| Gaussian paths / SB-CFM / OT coupling: select compatible endpoints, path noise, and conditional targets | [stochastic_flow.py](../src/aster/methods/stochastic_flow.py) | [Gaussian paths](../tests/unit/test_stochastic_flow.py), [transport](../tests/unit/test_flow_transport.py) | [TorchCFM](https://github.com/atong01/conditional-flow-matching) |
| Consistency training/distillation: align predictions along a shared denoising trajectory | [consistency.py](../src/aster/methods/consistency.py) | [Consistency tests](../tests/unit/test_consistency.py) | [OpenAI consistency models](https://github.com/openai/consistency_models) |
| Reflow and distribution-matching distillation: preserve teacher endpoint coupling or alternate score/generator roles | [generative_distillation.py](../src/aster/methods/generative_distillation.py) | [Gradient boundaries](../tests/unit/test_generative.py) | [Rectified Flow](https://github.com/gnobitab/RectifiedFlow), [DMD paper](https://arxiv.org/abs/2311.18828) |
| MeanFlow: learn interval-average velocity with its directional-derivative correction | [meanflow.py](../src/aster/methods/meanflow.py) | [MeanFlow tests](../tests/unit/test_meanflow.py) | [MeanFlow](https://github.com/Gsunshine/meanflow) |
| Shortcut models: condition on step size and bootstrap a larger step from smaller ones | [shortcut.py](../src/aster/methods/shortcut.py) | [Shortcut targets](../tests/unit/test_shortcut.py) | [Shortcut Models](https://github.com/kvfrans/shortcut-models) |
| Drifting: train a generator from an attraction/repulsion field in the declared feature space | [drifting.py](../src/aster/methods/drifting.py) | [Drifting method](../tests/unit/test_drifting_method.py) | [Drifting](https://github.com/lambertae/drifting) |
| Latent diffusion: keep encoder scale/shift, latent field, and decoder in one pipeline contract | [pipelines.py](../src/aster/pipelines.py) | [Latent pipeline](../tests/integration/test_latent_pipeline.py) | [Latent Diffusion](https://github.com/CompVis/latent-diffusion) |
| Wan video generation: combine a spatiotemporal field with a causal video VAE | [video_generation.py](../src/aster/methods/video_generation.py) | [Video generation](../tests/integration/test_video_generation.py) | [Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| TeaCache-style reuse: skip selected backbone work using a residual cache; evaluate end-to-end error | [step_cache.py](../src/aster/optimization/step_cache.py), [wan_teacache.py](../src/aster/optimization/wan_teacache.py) | [DiT cache](../tests/unit/test_optimization_step_cache.py), [Wan cache](../tests/unit/test_wan_teacache.py) | [TeaCache](https://github.com/ali-vilab/TeaCache) |

**Compose, then measure.** A useful experiment is teacher → distillation → fewer sampling steps → optional residual cache → paired quality/resource evaluation. Do not enable every approximation at once: establish a baseline, change one component, and apply a quality gate. [Methods](METHODS.md), [generation evaluation](GENERATIVE_EVALUATION.md), and [interval generation](INTERVAL_GENERATION.md) explain the connected paths.

## Multimodal and action models

Image preprocessing, inserted visual tokens, camera order, action scaling, and padding masks are part of a model's semantics, not incidental input plumbing.

| Concept and idea | Native implementation | Reading test | Primary reference |
| --- | --- | --- | --- |
| SigLIP: image/text encoders with pairwise sigmoid supervision | [siglip.py](../src/aster/models/siglip.py), [supervised.py](../src/aster/methods/supervised.py) | [SigLIP](../tests/unit/test_models_siglip.py), [objective](../tests/unit/test_sigmoid_objective.py) | [Big Vision SigLIP](https://github.com/google-research/big_vision) |
| LLaVA / Qwen-VL: connect visual features to language while preserving token/position alignment | [multimodal.py](../src/aster/models/multimodal.py), [qwen_vl.py](../src/aster/models/qwen_vl.py) | [Qwen-VL tests](../tests/unit/test_models_qwen_vl.py) | [LLaVA](https://github.com/haotian-liu/LLaVA), [Transformers](https://github.com/huggingface/transformers) |
| BLIP-2: use learned queries to bridge vision and language representations | [blip2.py](../src/aster/models/blip2.py) | [BLIP-2 tests](../tests/unit/test_models_blip2.py) | [LAVIS](https://github.com/salesforce/LAVIS) |
| ACT: predict action chunks with a conditional variational model | [actions.py](../src/aster/models/actions.py) | [Action/world tests](../tests/unit/test_world_actions.py) | [ACT](https://github.com/tonyzhaozh/act) |
| OpenVLA: fuse visual encoders and predict discretized actions through language tokens | [openvla.py](../src/aster/models/openvla.py) | [OpenVLA tests](../tests/unit/test_models_openvla.py) | [OpenVLA](https://github.com/openvla/openvla) |
| π0 / π0.5: condition an action-flow expert on vision, language, and declared state inputs | [pi_vla.py](../src/aster/models/pi_vla.py) | [Pi VLA integration](../tests/integration/test_pi_vla.py) | [OpenPI](https://github.com/Physical-Intelligence/openpi) |
| GR00T: combine embodiment conditioning and a flow-based action head | [groot.py](../src/aster/models/groot.py) | [GR00T tests](../tests/unit/test_models_groot.py) | [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) |

These are model/training interfaces, not authorization to control a robot. Public robot success rates require the matching environment, action protocol, and trained weights.

## World models and planning

Latent prediction, pixel generation, reward prediction, and search solve different problems. Pick the representation and control objective before selecting a planner.

| Concept and idea | Native implementation | Reading test | Primary reference |
| --- | --- | --- | --- |
| RSSM: combine deterministic memory with stochastic latent state | [world.py](../src/aster/models/world.py) | [World/action tests](../tests/unit/test_world_actions.py) | [DreamerV3](https://github.com/danijar/dreamerv3) |
| PlaNet: learn latent dynamics and use cross-entropy-method planning | [planet.py](../src/aster/models/planet.py), [planner](../src/aster/planning/planet.py) | [PlaNet](../tests/unit/test_planet.py), [loop](../tests/unit/test_planet_loop.py) | [PlaNet](https://github.com/google-research/planet) |
| JEPA: predict masked target representations with an EMA target encoder | [jepa.py](../src/aster/models/jepa.py), [objective](../src/aster/methods/jepa.py) | [JEPA tests](../tests/unit/test_jepa.py) | [JEPA](https://github.com/facebookresearch/jepa) |
| LeWorldModel: learn action-conditioned latent prediction and use it for planning | [lewm.py](../src/aster/models/lewm.py), [planner](../src/aster/planning/lewm.py) | [LeWM tests](../tests/unit/test_models_lewm.py) | [LeWorldModel](https://github.com/lucas-maes/le-wm) |
| TD-MPC2: learn latent dynamics, rewards and values; plan with policy-guided sampling | [tdmpc2.py](../src/aster/methods/tdmpc2.py) | [TD-MPC2 tests](../tests/unit/test_tdmpc2.py) | [TD-MPC2](https://github.com/nicklashansen/tdmpc2) |
| MuZero / PUCT / Gumbel search: combine learned dynamics/value with a bounded tree search | [muzero.py](../src/aster/methods/muzero.py), [mcts.py](../src/aster/planning/mcts.py) | [MuZero](../tests/unit/test_muzero.py), [search](../tests/unit/test_mcts.py) | [mctx](https://github.com/google-deepmind/mctx) |
| VAE + MDN-RNN + controller: separate visual compression, predictive memory, and control | [vmc.py](../src/aster/models/vmc.py), [planning](../src/aster/planning/vmc.py) | [World Models tests](../tests/unit/test_vmc.py) | [WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments) |

The last entry is the non-driving pathway. Autonomous-driving integrations are outside this repository's intended scope. Other video/world variants and their precise boundaries are listed in [model notes](MODELS.md).

## Reinforcement learning

| Concept and idea | Native implementation | Reading test | Primary reference |
| --- | --- | --- | --- |
| PPO / GRPO: control policy changes with explicit probability ratios and advantage conventions | [reinforcement.py](../src/aster/methods/reinforcement.py) | [RL objectives](../tests/unit/test_reinforcement.py) | [Spinning Up](https://github.com/openai/spinningup), [DeepSeekMath](https://arxiv.org/abs/2402.03300) |
| RLOO / online GRPO: generate a complete rollout group before forming group-relative advantages | [policy_gradient.py](../src/aster/methods/policy_gradient.py) | [Online policy integration](../tests/integration/test_policy_gradient.py) | [TRL RLOO](https://github.com/huggingface/trl/blob/main/trl/trainer/rloo_trainer.py) |
| TD3 / TD3+BC / IQL: learn from transitions with method-specific targets and actor updates | [offline.py](../src/aster/methods/offline.py) | [Offline RL tests](../tests/unit/test_offline.py) | [TD3](https://github.com/sfujim/TD3), [TD3+BC](https://github.com/sfujim/TD3_BC), [IQL](https://github.com/ikostrikov/implicit_q_learning) |
| CQL: penalize overly optimistic action values using the declared proposal distribution | [conservative.py](../src/aster/methods/conservative.py) | [CQL tests](../tests/unit/test_conservative.py) | [CQL](https://github.com/aviralkumar2907/CQL) |

A loss formula is not a trained policy. Read [methods](METHODS.md) for rollout restrictions, termination semantics, frozen targets, multi-role ownership, and recovery.

## Agents and evaluation

| Concept and idea | Native implementation | Reading test | Primary reference |
| --- | --- | --- | --- |
| Tool lifecycle and permissions: bind approval to an exact action; commit its receipt before trusting completion | [runtime.py](../src/aster/agents/runtime.py), [permissions.py](../src/aster/agents/permissions.py) | [Agent integration](../tests/integration/test_agents_native.py) | [Codex app-server](https://github.com/openai/codex/tree/main/codex-rs/app-server) |
| MCP transports and context: parse bounded protocol messages without treating remote content as authority | [mcp.py](../src/aster/agents/mcp.py), [mcp_stdio.py](../src/aster/agents/mcp_stdio.py), [mcp_context.py](../src/aster/agents/mcp_context.py) | [stdio](../tests/unit/test_mcp_stdio.py), [context](../tests/unit/test_mcp_context.py) | [MCP 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18) |
| Verified agent data → SFT: learn from approved action tokens without supervising tool observations | [agent_learning.py](../src/aster/methods/agent_learning.py) | [Agent learning integration](../tests/integration/test_agents_rl.py) | [Codex lifecycle reference](https://github.com/openai/codex) |
| Fixed evaluation cohorts and language metrics: preserve dataset/configuration identity and count failures | [protocol.py](../src/aster/evaluation/protocol.py), [language.py](../src/aster/evaluation/language.py) | [Evaluation suites](../tests/unit/test_evaluation_suites.py) | [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| Generative quality gates: compare the same cohort with fixed feature extraction and resource measurements | [generative.py](../src/aster/evaluation/generative.py), [generation_gate.py](../src/aster/evaluation/generation_gate.py) | [Generation gate](../tests/unit/test_generation_gate.py) | [Clean-FID](https://github.com/GaParmar/clean-fid) |

Agent orchestration is an engineering state machine, not an assertion that Aster implements all Codex functionality. MCP support is a fixed subset. Evaluation adapters do not supply benchmark scores without running the protocol. See [agents](AGENTS.md) and [benchmarks](BENCHMARKS.md).

## Keep this map useful

When adding an algorithm, contribute its mathematical contract, native implementation, one independent check, and the primary source. Record approximations and unsupported inputs next to the implementation. If only a model alias or import wrapper exists, do not present it as a new native algorithm.

For a first contribution, pick one row, run its test, and explain a boundary or add a small counterexample. See [the learning path](LEARNING_PATH.md) and [contributing](../CONTRIBUTING.md).
