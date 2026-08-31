# Getting started

Start from the repository root with Python 3.11+ and PyTorch installed for your machine. A CPU environment is enough for the first examples.

~~~bash
python -m pip install -e ".[test]"
python -m aster doctor
python examples/quickstart.py
~~~

No model downloads, remote code, API credentials, or hardware actions are needed. The supported dependency range is broader than the locally tested environment; CI and [testing](TESTING.md) record what actually runs.

## 1. Fine-tune a tiny model

The [quickstart](../examples/quickstart.py) builds a small Llama-shaped native model, injects LoRA into its output projection, runs supervised updates, and checks base-weight immutability and merged outputs.

Use [fine-tuning](FINE_TUNING.md) to select real projection targets. A model architecture is not a pretrained model: meaningful quality requires licensed data, sufficient training, and appropriate evaluation.

## 2. Serve the trained adapter

~~~bash
python examples/online_adapter_stack.py --kv int8
~~~

This example uses one base model, separate adapter identities, a small KV page budget, and a bounded host archive. It verifies that all request-owned resources are released. Optional storage formats are fp8_e4m3fn and fp8_e5m2. Low-bit storage does not imply a low-bit GPU matrix kernel.

## 3. Run the artifact workflow

~~~bash
python -m aster run examples/recipes/language_chain.json \
  --output runs/language-001 --store artifacts
~~~

The bundled data is synthetic demonstration text. The workflow trains, distills, publishes model artifacts, and evaluates native language loss. Replace it with licensed, correctly split data for real experiments.

Local language JSONL accepts text records, or explicit input_ids with labels. The value -100 excludes positions from supervision. Sequences longer than the configured maximum are rejected instead of silently truncating answers.

## Resume carefully

Completed workflow stages are reusable only when code, configuration, data, and upstream artifacts still match. After changing them, choose a new output directory. Do not force reuse of an incompatible run.

A deployment artifact is not a full training checkpoint. Checkpoints include declared optimizer, RNG, EMA, sampler, replay, and role state; portable resharding and exact rank-local resume are distinct operations.

## Where to go next

- [Learning path](LEARNING_PATH.md): trace tensors, gradients, caches, and ownership step by step.
- [Algorithm map](ALGORITHMS.md): native implementations, tests, papers, and official sources.
- [Workflow gallery](EXAMPLES.md): inputs, expected checks, and experiment prerequisites.
- [Fine-tuning](FINE_TUNING.md): trainable parameters, merge, and online adapters.
- [Losses](LOSSES.md): objectives and normalization.
- [Architecture](ARCHITECTURE.md): extend the common lifecycle.
- [Roadmap](ROADMAP.md): exact boundaries before production use.
- [Technical notes](README.md): domain-specific implementation details.
