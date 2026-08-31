# Roadmap

[Home](../README.md) · [中文首页](../README.zh-CN.md) · [Current status](STATUS.md) · [Documentation](README.md)

Aster v0.1.0 has experimental APIs. This roadmap defines priorities and completion criteria; [current status](STATUS.md) describes supported features, and [releases](https://github.com/sutiankang/aster/releases) record verification for each version. Numerical correctness, hardware performance, and pretrained quality require different evidence.

路线图分成发布基线、微调、并行训练、推理部署、生成与世界模型、智能体、公开评测及开源许可。下面的完成标准是验收条件，不表示所有项目已完成；目前缺口也不会因为发布版本而消失。

## Milestones and acceptance criteria

The order below expresses priorities, not promised delivery dates. Features advance together with their tests, examples, documentation, and supported-configuration limits.

| Milestone | Current scope / remaining work | Completion criterion | Implementation and evidence entry |
| --- | --- | --- | --- |
| M0 · Reproducible release baseline | Native workflows exist; each release must pass its own clean-environment gates | Installable package, executable examples, matching source/tag/package identity, valid documentation links, and successful CPU CI on Python 3.11/3.13 | [Testing](TESTING.md), [examples](EXAMPLES.md), [releases](https://github.com/sutiankang/aster/releases) |
| M1 · Practical fine-tuning | Linear LoRA exists; QLoRA/NF4, DoRA, rsLoRA, IA³, prompt/prefix adapters remain | Declared target layers, independent update/merge tests, frozen-base checks, resumable checkpoints, and held-out task comparison | [Fine-tuning](FINE_TUNING.md), [losses](LOSSES.md) |
| M2 · Broader parallel training | Selected TP/PP/DP/CP and ZeRO paths exist; architecture combinations and elastic migration remain | Multi-process gradient/update equivalence, resumed-next-step equivalence, explicit rejection of unsupported layouts, and multi-node GPU measurements before performance claims | [Training](TRAINING.md), [Muon](MUON_RECIPES.md) |
| M3 · Production serving combinations | Single-worker continuous paging exists; multi-rank continuous serving, mixed-state scheduling, and low-bit kernel integration remain | Cancellation and resource cleanup under load, checkpoint/adaptor identity checks, supported grammar/state contracts, and paired quality/latency/memory measurements | [Inference](INFERENCE.md), [paged attention](PAGED_ATTENTION.md), [optimization](OPTIMIZATION.md) |
| M4 · Connected generative acceleration | Selected flow/diffusion and distillation paths exist; wider model/decoder/cache combinations remain | Teacher → student → sampler → evaluator workflow with fixed data, feature identity, quality floors, and measured resource use | [Methods](METHODS.md), [generation evaluation](GENERATIVE_EVALUATION.md) |
| M5 · Multimodal, actions, and world models | Native families and planners exist; broader deployment combinations and public task/control evidence remain | Correct preprocessing/state/episode contracts, held-out downstream evaluation, and matched environment protocols without autonomous-driving integrations | [Models](MODELS.md), [world-model algorithms](ALGORITHMS.md#world-models-and-planning) |
| M6 · Robust agents | Bounded tools and selected MCP paths exist; full protocol coverage, isolation, and service supervision remain | Permission-denial/cancellation tests, crash-safe receipts, independently verified task completion, and supported-platform isolation tests | [Agents](AGENTS.md), [security](../SECURITY.md) |
| M7 · Public evaluation and open-source clearance | Evaluation contracts and source notices exist; benchmark runs and redistribution clearance remain | Licensed weights/data, reproducible public protocols, file-level source clearance, repository-wide license, and private vulnerability reporting | [Benchmarks](BENCHMARKS.md), [NOTICE](../NOTICE.md), [release checklist](RELEASING.md) |

## Working paths

- Native models and objective families listed in the main README.
- Single-worker LoRA training, merge, and supported online adapters.
- Shared trainer with declared multi-role state and supported parallel providers.
- Native paging, prefix sharing, bounded host swap, and continuous single-worker serving.
- Bounded agent tools with approvals, receipts, and selected MCP capabilities.
- Explicit local artifacts and complete-cohort evaluation protocols.

## Remaining engineering work

- QLoRA/NF4, DoRA, rsLoRA, IA³, and prompt/prefix adapters.
- Cross-architecture TP/PP/CP/SP/GTP combinations and DeepSeek-specific parallel providers.
- Elastic data-cursor and multi-role migration.
- Multi-rank continuous HTTP serving and DSpark continuous-scheduler integration.
- Single-kernel low-bit GPU page-table execution and compute/transfer prefetch overlap.
- Generic mixed recurrent-state scheduling, full multimodal serving, and complete JSON grammar.
- Remaining MCP extension capabilities, cross-platform strong isolation, long-lived service supervision, and multi-tenant authentication.
- Broader public pretrained-model and downstream-task evaluation.

## Evidence boundaries

CPU formulas and gradient checks can establish numerical behavior for their tested inputs. They cannot prove GPU compilation, stream ordering, hardware throughput, kernel isolation, or public model quality.

Approximate methods such as quantization, distillation, pruning, and residual caches require paired evaluation with the same data and preprocessing. Tiny random-model examples are workflow checks only.

The [capability manifest](scope/capabilities.json) retains finer-grained implementation paths. It is not an upstream compatibility certificate.

## Licensing and release readiness

Public repository access and private security reporting are enabled. Resolve [license and attribution issues](../NOTICE.md), establish contributor governance and a repository-wide license, run CI for each release candidate, and complete the [release checklist](RELEASING.md). Public visibility does not complete source or licensing review.
