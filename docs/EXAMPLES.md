# Runnable workflow gallery

[Home](../README.md) · [Learning path](LEARNING_PATH.md) · [Algorithm map](ALGORITHMS.md)

These examples connect components rather than only constructing a model. They run on CPU with synthetic inputs and do not download weights, call paid APIs, or control hardware.

Install once from the repository root:

~~~bash
python -m pip install -e ".[test]"
python -m aster doctor
~~~

| Workflow | Input | What it checks | Writes |
| --- | --- | --- | --- |
| Tiny LoRA | Two short integer-token sequences | Adapter update, frozen base, merge equivalence | None |
| Shared-base serving | Base and adapter requests | Paged KV, bounded swapping, cleanup | None |
| Teacher → student | Bundled synthetic JSONL text | Training, distillation, artifacts, evaluation | Your run and artifact directories |

## 1. Train and merge a tiny adapter

~~~bash
python examples/quickstart.py
~~~

**Read:** [example](../examples/quickstart.py) → [LoRA implementation](../src/aster/methods/distillation.py) → [Trainer](../src/aster/training/trainer.py) → [regression test](../tests/unit/test_repository.py).

The JSON result should report eight updates, unchanged base weights, and a small merge error. The adapter is attached to `lm_head` to keep the first example compact; it is not a claim that output-only tuning is the best recipe for your task.

Change the rank or projection targets to study parameter efficiency. For an actual fine-tuning run, supply licensed data, a suitable pretrained checkpoint, and a separate evaluation split. See [fine-tuning](FINE_TUNING.md).

## 2. Serve a base model and its adapter

~~~bash
python examples/online_adapter_stack.py --kv int8
~~~

**Read:** [example](../examples/online_adapter_stack.py) → [adapter runner](../src/aster/inference/adapters.py) → [request engine](../src/aster/inference/engine.py) → [online adapter tests](../tests/unit/test_online_lora.py).

The example trains an adapter, registers it against the base identity, and submits two requests with a small page budget. It uses the native torch-online paged path, INT8 cache storage, and a bounded host archive.

Check that both requests complete and these three fields are zero after shutdown:

~~~text
remaining_pages
remaining_host_bytes
remaining_adapter_bytes
~~~

Alternative storage formats are `--kv fp8_e4m3fn` and `--kv fp8_e5m2`. They are storage choices, not proof that the whole model executes in FP8 or uses a faster GPU kernel. Compare outputs and resource measurements before selecting a lossy format.

This is an in-process serving workflow, not an authenticated public HTTP service. See [inference](INFERENCE.md) for deployment boundaries.

## 3. Teacher to student to evaluation

~~~bash
python -m aster run examples/recipes/language_chain.json --output runs/language-001 --store artifacts
~~~

**Read:** [recipe](../examples/recipes/language_chain.json) → [bundled text](../examples/data/tiny_text.jsonl) → [language recipes](../src/aster/recipes.py) → [artifact store](../src/aster/core/artifacts.py).

The recipe runs three named stages: `teacher`, `student`, and `student_eval`. The student depends on the teacher artifact and uses forward-KL distillation. The evaluator consumes the student artifact.

Check the stage receipts and artifact identities, not only the last printed loss. After changing code, data, or configuration, choose a new output directory; an incompatible completed stage must not be silently reused.

The demonstration deliberately reuses the tiny text file to keep setup minimal. Its evaluation loss is a pipeline check, **not held-out generalization evidence**. Replace the data and evaluation split before making a model-quality claim.

## Move from a demonstration to an experiment

| Next goal | Entry point | Additional requirements |
| --- | --- | --- |
| Real supervised fine-tuning | [Fine-tuning](FINE_TUNING.md), [losses](LOSSES.md) | Licensed weights/data, preprocessing identity, held-out split |
| Distributed language training | [Training](TRAINING.md), [Muon recipes](MUON_RECIPES.md) | Valid model/layout combination and communicating workers |
| Sparse attention or DSpark | [DSA training](DSA_TRAINING.md), [DSpark](DSPARK.md) | Matching model/draft configuration and cache contracts |
| Latent/image/video generation | [Methods](METHODS.md), [generation evaluation](GENERATIVE_EVALUATION.md) | Encoder/latent conventions, licensed data, fixed feature extractor |
| Action learning or world planning | [Model map](MODELS.md), [PlaNet](PLANET.md), [MuZero](MUZERO.md) | Observation/action spec, episode boundaries, environment evaluation |
| Tool-using agent | [Agents](AGENTS.md), [agent learning](AGENT_RL.md) | Host permissions, bounded tools, independent completion verification |

For performance work, report baseline and modified quality, latency/throughput, peak memory, hardware, precision, batch size, and warmup/measurement rules. Do not present tiny synthetic outputs as public benchmark scores.

## Re-run the small checks

~~~bash
python -m pytest tests/unit/test_repository.py tests/unit/test_online_lora.py -q
python tools/check_repository.py
~~~

Longer or optional tests are documented in [testing](TESTING.md). If a platform requirement is unavailable, preserve the skip reason.
