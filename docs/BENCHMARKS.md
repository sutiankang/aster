# Benchmarks and reproducible claims

Aster provides evaluators, not fabricated leaderboard results. No public pretrained quality or GPU throughput number is advertised here.

| Domain | Evaluation direction | Implementation |
| --- | --- | --- |
| Language | Token NLL/perplexity; official task definitions and scoring | Native scorer + optional lm-eval |
| Vision-language | Task-specific public prompts, preprocessing, and scoring | Optional lmms-eval adapter |
| Generation | FID/KID, video feature distributions, paired quality/resource comparisons | Native metrics + approved local extractors |
| Actions / control | Complete episodes, success/return, fixed initial states | Declared Gymnasium/LIBERO protocols |
| World models | Prediction/control protocols, paired controllability | Native world/Genie evaluation |
| Agents | Independent task resolution, complete task denominator | Native verifier; optional SWE-bench harness |
| Serving | Client TTFT/ITL, latency, throughput, actual storage | Native request/measurement interfaces |

## Report enough to reproduce the result

Include the artifact and preprocessing identity, source revision, dataset split and license, complete sample IDs, seeds, software/hardware, warmup, measurement boundaries, errors, and all comparison settings.

Keep failed samples in the planned denominator. Report quality alongside approximation speedups. Count real model calls rather than deriving a favorable NFE from a solver label.

A CPU toy run, fewer calls, packed bytes, or a formula comparison does not establish public quality or GPU speed. Do not compare FID/FVD values produced by different extractors or preprocessing.

For a first community benchmark, prefer one reproducible narrow task with a downloadable licensed dataset and a script that emits all required metadata.
