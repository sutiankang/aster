# Changelog

## 0.1.0 — First release

### Models and methods

- Native PyTorch implementations for language, multimodal, action, generative, and world-model workflows.
- Supervised, preference, reinforcement-learning, and distillation objectives.
- Linear LoRA fine-tuning, merge, and supported shared-base adapters.

### Training and inference

- Shared training with explicit loss normalization, checkpoints, EMA, Muon, and supported parallel/ZeRO layouts.
- Paged KV storage, prefix reuse, continuous single-worker batching, INT8/FP8 KV, and bounded host swap.
- Bounded agent tools, HTTP/stdio MCP subsets, and cancellation-safe receipts.
- Artifact-backed workflows and evaluation protocols connecting training, compression, and serving.

### Getting started

- Three connected CPU examples: LoRA training and merge, shared-base serving, and teacher → student → evaluation.
- Bilingual homepages, an algorithm and paper index, and a guided learning path.
- Source tests, CPU CI, contributor guidance, and wheel/source packages.

APIs are experimental. Supported configurations and remaining work are described in [Status](docs/STATUS.md) and [Roadmap](docs/ROADMAP.md). See [NOTICE](NOTICE.md) for licensing and attribution.
