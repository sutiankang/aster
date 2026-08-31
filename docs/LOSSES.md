# Loss catalog

A loss is an executable objective, not just a configuration label. This table points to the implementation that computes it.

| Family | Implemented objectives / terms | Source |
| --- | --- | --- |
| Supervised | Causal/aligned CE, ignored-position masks, label smoothing, regression MSE | [supervised.py](../src/aster/methods/supervised.py) |
| Preference | DPO, IPO, SimPO | [preference.py](../src/aster/methods/preference.py) |
| Text policy gradients | GRPO sequence/token/fixed-length reductions; RLOO full-response ratios | [reinforcement.py](../src/aster/methods/reinforcement.py), [policy_gradient.py](../src/aster/methods/policy_gradient.py) |
| Actor-critic | PPO clipped policy/value and entropy; SAC policy/Q/temperature; Double-DQN TD Huber | [reinforcement.py](../src/aster/methods/reinforcement.py) |
| Offline RL | TD3, TD3+BC, IQL expectile/value/advantage-weighted BC, CQL conservative Q | [offline.py](../src/aster/methods/offline.py), [conservative.py](../src/aster/methods/conservative.py) |
| Token distillation | Forward KL, reverse KL, mixed KL, Jensen-Shannon, temperature scaling, CE mixture | [distillation.py](../src/aster/methods/distillation.py) |
| Feature distillation | MSE, cosine, relation matching; TinyBERT and MiniLM | [distillation.py](../src/aster/methods/distillation.py), [encoder_distillation.py](../src/aster/methods/encoder_distillation.py) |
| Contrastive | Symmetric CLIP cross-entropy, pairwise SigLIP binary objective | [supervised.py](../src/aster/methods/supervised.py) |
| Model auxiliaries | Mixtral router balancing, multi-token prediction, DSA/QSA indexer KL | [experts.py](../src/aster/nn/experts.py), [mtp.py](../src/aster/methods/mtp.py), [sparse_indexer.py](../src/aster/methods/sparse_indexer.py) |
| Speculative drafts | DSpark token CE, probability L1, confidence BCE | [dspark.py](../src/aster/methods/dspark.py) |
| Diffusion | Epsilon/x0/v targets, learned-variance variational terms, Min-SNR weighting, EDM | [generation.py](../src/aster/methods/generation.py) |
| Flow | Flow matching, conditional Gaussian/OT/SB paths, MeanFlow, Shortcut | [generation.py](../src/aster/methods/generation.py), [stochastic_flow.py](../src/aster/methods/stochastic_flow.py), [meanflow.py](../src/aster/methods/meanflow.py), [shortcut.py](../src/aster/methods/shortcut.py) |
| Generative distillation | Consistency training/distillation, progressive distillation, DMD, Drifting | [consistency.py](../src/aster/methods/consistency.py), [generative_distillation.py](../src/aster/methods/generative_distillation.py), [solvers.py](../src/aster/methods/solvers.py) |
| Representation / codecs | JEPA prediction, LeWM SIGReg, masked reconstruction, VQ, VAE KL, LPIPS, PatchGAN | [methods directory](../src/aster/methods) |
| Action / world models | ACT reconstruction/KL, action flow, RSSM observation/reward/continue/KL, PlaNet overshooting, MuZero policy/value/reward | [actions.py](../src/aster/methods/actions.py), [world_model.py](../src/aster/methods/world_model.py), [planet.py](../src/aster/methods/planet.py), [muzero.py](../src/aster/methods/muzero.py) |

This is not a claim to implement every TRL loss or every paper-specific variation. For example, KTO, ORPO, a general router z-loss, and arbitrary fused/chunked vocabulary CE kernels are not exposed as completed native objectives.

## Normalization is part of the algorithm

Every objective returns a LossTerm or LossBundle:

~~~python
from aster.core import LossTerm

term = LossTerm(
    numerator=per_token_loss.masked_select(valid).sum(),
    denominator=valid.sum().to(per_token_loss.dtype),
    unit="token",
    name="language",
)
~~~

The denominator is a detached valid count. The trainer accumulates numerators/counts over the appropriate data group; tensor-parallel replicas are not extra samples.

For batches containing 2 and 20 valid tokens, the correct aggregate is (sum1 + sum2) / 22, not (mean1 + mean2) / 2. Terms measured in tokens, pairs, queries, transitions, or latent coordinates are normalized separately before weighting.

## Gradient boundaries matter

- DPO/IPO freeze the reference, not the policy difference.
- Distillation targets usually stop gradients; DMD and perceptual actor/generator paths may require input gradients through frozen networks.
- PPO/RL behavior probabilities and advantages are recorded data, not recomputed policy targets.
- Padding and prompt tokens are excluded from action/token supervision.
- MeanFlow uses a real directional derivative. Replacing it with detached arithmetic changes the objective.
- Global-statistic methods require their declared logical batch. Arbitrary microbatching can change SIGReg or advantage statistics.

See [the trainer contract](ARCHITECTURE.md) and the independent formula/gradient tests under [tests](../tests). Reference implementations inform the formulas; runtime objectives remain native.
