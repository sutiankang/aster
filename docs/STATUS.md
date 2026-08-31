# Release status — v0.1.0

[Home](../README.md) · [中文首页](../README.zh-CN.md) · [Roadmap](ROADMAP.md) · [Algorithms](ALGORITHMS.md) · [Testing](TESTING.md)

Aster v0.1.0 provides native implementations and connected workflows for the configurations described below. APIs are experimental. Use this page to check whether a feature fits your task, and the [roadmap](ROADMAP.md) to find planned improvements.

The [release page](https://github.com/sutiankang/aster/releases/tag/v0.1.0) provides downloads, the source revision, checksums, and version-specific test results. This is not a guarantee of complete upstream compatibility or pretrained-model quality.

## Status at a glance

| Category | Current position |
| --- | --- |
| Native implementation | Model, objective, training, compression, inference, agent, and evaluation components exist for the declared configurations |
| Connected workflows | Tiny CPU LoRA/merge, shared-base serving, and teacher → student → evaluation examples are runnable |
| Completeness | Broader architecture/layout combinations and several named methods still require implementation |
| Verification | Local numerical/workflow tests are available; GPU, multi-node, and public quality evidence is incomplete |
| Security | Bounded permissions and receipts exist; this is not a hardened multi-tenant platform |
| Availability and licensing | Public repository on `main` and public version downloads; repository-wide licensing and file-level clearance remain open |

## What works, and what is still missing

“Implemented” below means the documented native path exists, not unrestricted compatibility with the entire upstream repository.

| Area | Implemented path | Remaining work / limitation | Details |
| --- | --- | --- | --- |
| LLM and attention | Native decoder/encoder families; dense, MLA/MoE, sparse and recurrent components | Architecture-specific checkpoint/state/provider constraints; not every combination is supported | [Models](MODELS.md), [algorithms](ALGORITHMS.md#attention-and-language-models) |
| VLM and VLA | Native visual-language and action model paths, preprocessing and training objectives | Broader serving combinations, pretrained-task quality, environment-specific action evaluation | [Models](MODELS.md), [methods](METHODS.md) |
| Generative modeling | Diffusion/flow, consistency, interval methods, drifting and selected compression paths | Approximate acceleration needs paired quality checks; no blanket upstream-quality or speed claim | [Methods](METHODS.md), [generation evaluation](GENERATIVE_EVALUATION.md) |
| World models / planning | RSSM, PlaNet, JEPA/LeWM, TD-MPC2, MuZero and selected planners | Public control/long-horizon quality and remaining model-specific deployment combinations | [Algorithm map](ALGORITHMS.md#world-models-and-planning) |
| Shared training | Multi-role updates, explicit loss counts, checkpointing, supported parallel/ZeRO layouts and Muon recipes | Broader cross-architecture parallel providers, elastic cursor/role migration, multi-node GPU validation | [Training](TRAINING.md), [Muon](MUON_RECIPES.md) |
| Fine-tuning | Full/frozen-parameter tuning, linear LoRA, merge, selected online multi-LoRA | QLoRA/NF4 training, DoRA, rsLoRA, IA³, prompt and prefix adapters are not implemented | [Fine-tuning](FINE_TUNING.md) |
| Reinforcement learning / distillation | Declared preference, online/offline RL, token/feature/generative distillation workflows | No generic asynchronous multi-node rollout service; methods retain their own sampling/gradient constraints | [Methods](METHODS.md), [losses](LOSSES.md) |
| Inference | Paging, prefix sharing, host swapping, continuous single-worker batching and selected speculative paths | Multi-rank continuous HTTP, DSpark scheduler integration, generic mixed recurrent/multimodal scheduling and full JSON grammar | [Inference](INFERENCE.md), [DSpark](DSPARK.md) |
| Hardware optimization | Native operator formulas, supported Triton paths, low-bit storage and residual reuse | Low-bit single-kernel page-table execution, transfer/compute overlap, broad hardware compilation/performance evidence | [Optimization](OPTIMIZATION.md), [native attention](NATIVE_FLASH_ATTENTION.md) |
| Agents | Tool permissions, durable receipts, bounded planning, HTTP/stdio MCP subsets and verified training data | Complete MCP, cross-platform strong isolation, long-lived supervision, multi-tenant authentication and public agent benchmarks | [Agents](AGENTS.md), [security](../SECURITY.md) |
| Evaluation | Fixed cohorts, failure accounting, native metrics and optional official adapters | Licensed benchmark inputs/weights and actual public quality/throughput runs | [Benchmarks](BENCHMARKS.md) |

Autonomous-driving integrations are outside the project scope.

## How to interpret the capability manifest

The [machine-readable manifest](scope/capabilities.json) currently contains **171 required capability entries**, all conservatively marked `partial` at the full-title level. This does not mean 171 empty modules: entries carry concrete source paths, tests, and limits, and broad titles include unverified architecture/configuration/hardware combinations.

The recorded comparison categories are 149 formula/workflow entries, 15 installed-official-reference entries, one same-weight entry, two fixed-source-reference entries, one public-mechanism entry, and three unverified entries. These categories are not percentages of engineering completion, universal correctness proofs, or evidence that every linked test ran on this machine.

An entry is complete only when its stated implementation and verification requirements are met.

## Verification for this version

- The source/documentation check covers English code prose, private-path patterns, relative links, HTML navigation/images, and static SVG assets.
- Formatting and static Python checks cover source, tests, examples, and tools.
- A regression test verifies that the CPU test extra includes Pillow and safetensors without making either a core runtime dependency.
- GitHub checks run on Python 3.11 and 3.13 with pinned action revisions and read-only permissions. See [Actions](https://github.com/sutiankang/aster/actions/workflows/ci.yml) and the exact commit linked in each [release](https://github.com/sutiankang/aster/releases). A configured workflow or a run from a different commit is not evidence for the current release.
- Release checks include wheel/source-package contents, metadata, installed-package examples, and downloadable checksums. Results for the published source revision are recorded in its release notes.
- Optional CUDA, upstream-source, restricted-data, or platform tests keep explicit skip reasons. A skipped check is not a pass.

Public benchmark scores, multi-node GPU throughput, and broad device coverage are not claimed. See [testing](TESTING.md) for environments and test tiers.

## What is included in Git

| Included | Excluded |
| --- | --- |
| Native source, tests, reproducible examples and tiny demonstration data | Model weights, checkpoints, private or benchmark datasets |
| Homepages, algorithm references, learning material and support boundaries | Local research records, validation receipts, temporary previews and publication backups |
| Packaging metadata, repository checks and CI configuration | Virtual environments, build output, caches, logs and test result files |
| Required third-party license/attribution notices | Credentials, environment secrets and unrelated workspace projects |

Tests and attribution notices are intentionally retained: they are essential to learning, verification, and source provenance.

## Licensing and deployment

Repository-wide licensing and file-level source review remain open, including non-commercial, share-alike, custom, and unknown source relationships. Read [NOTICE](../NOTICE.md) before reusing or redistributing code. Model weights, benchmark data, and optional dependencies have their own terms.

Private vulnerability reporting is available through the repository's Security page. Before deployment, validate your hardware, model quality, and security requirements; the tiny examples do not establish production readiness. See [Security](../SECURITY.md), [Benchmarks](BENCHMARKS.md), and the [release checklist](RELEASING.md).
