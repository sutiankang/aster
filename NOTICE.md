# Licensing and third-party notices

**No repository-wide license has been granted yet. File-level license clearance remains incomplete.**

Review the source-specific terms below before reusing or redistributing code. Public access does not grant a repository-wide license. Existing third-party terms and notices remain in force.

Local implementations can still carry obligations concerning reference-source expression, copyright notices, weights, datasets, or dependencies. A mathematical reference alone does not determine the license of an implementation.

Core runtime code does not download or execute the following repositories. Optional reference tests may import official packages or use explicitly enabled, fixed-source comparisons.

## Reference-source inventory

This inventory preserves identified origins and reported terms; it is not exhaustive or a legal determination. A reference relationship does not automatically assign the same license to every mathematical formula, and changing variable names does not establish independence.

| Reference | Recorded terms | Handling |
| --- | --- | --- |
| [NVIDIA EDM](https://github.com/NVlabs/edm/blob/main/LICENSE.txt) | CC BY-NC-SA 4.0; NVIDIA 2022 | Non-commercial/share-alike reference; file-level distribution clearance remains open. |
| [Meta DiT](https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt) | CC BY-NC 4.0 | Non-commercial reference; file-level distribution clearance remains open. |
| [Meta JEPA](https://github.com/facebookresearch/jepa/blob/main/LICENSE) | CC BY-NC 4.0 | Non-commercial video/prediction reference; clearance remains open. |
| [OpenPI](https://github.com/Physical-Intelligence/openpi/blob/main/LICENSE) | Apache-2.0 | Action-flow and expert-architecture reference; weights need separate review. |
| [Big Vision](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py) | Apache-2.0; Big Vision Authors 2024 | SigLIP objective and vision architecture reference. |
| [ACT](https://github.com/tonyzhaozh/act/blob/main/LICENSE) | MIT; Tony Z. Zhao 2023 | CVAE, action-chunk and temporal-ensemble reference. |
| [TD-MPC2](https://github.com/nicklashansen/tdmpc2/blob/main/LICENSE) | MIT; Nicklas Hansen 2023 | SimNorm, Q/prior and MPPI formula reference. |
| [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/general_distill.py) | Apache-2.0; Huawei 2019-2020; retain Google/Hugging Face/NVIDIA notices in source | Feature/attention distillation reference; weights have separate terms. |
| [Wan2.1](https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt) | Apache-2.0; Alibaba Wan Team Authors 2024-2025 | Video field, causal VAE and image conditioning; weights/data have separate terms. |
| [Meta Flow Matching](https://github.com/facebookresearch/flow_matching/blob/main/LICENSE) | CC BY-NC 4.0 | Non-commercial continuous-flow reference; clearance remains open. |
| [Drifting](https://github.com/lambertae/drifting) | Complete license not verified | No license is not permission to redistribute; resolve before release. |
| [CQL](https://github.com/aviralkumar2907/CQL) | Complete license evidence not verified | Native formula comparison does not resolve source-expression obligations; review remains open. |
| [TorchCFM](https://github.com/atong01/conditional-flow-matching) | MIT in referenced path; Alex Tong and Kilian Fatras | Conditional/bridge paths and OT reference; no external runtime delegation. |
| [mctx](https://github.com/google-deepmind/mctx) | Apache-2.0; DeepMind Technologies Limited 2021 | PUCT/Gumbel search reference; no JAX runtime delegation. |
| [PlaNet](https://github.com/google-research/planet/tree/c04226b6db136f5269625378cd6a0aa875a92842) | Apache-2.0; PlaNet Authors 2019 | Gaussian RSSM, likelihood, overshooting and CEM reference. |
| [TensorFlow GRUBlockCell](https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/contrib/rnn/python/ops/gru_ops.py) | Apache-2.0; TensorFlow Authors 2016 | Reset-before GRU equations and initialization reference. |
| [WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments/tree/fd982b9691a941b52c6addbde29bc801ca6202c8) | README states MIT; standalone complete license not verified | Non-driving World Models reference; distribution clearance remains open. |
| [TensorFlow LayerNormBasicLSTMCell](https://github.com/tensorflow/tensorflow/blob/v1.8.0/tensorflow/contrib/rnn/python/ops/rnn_cell.py) | Apache-2.0; TensorFlow Authors | LSTM gate order, normalization epsilon and dropout semantics. |
| [CMA-ES purecma](https://github.com/CMA-ES/pycma/blob/master/cma/purecma.py) | Source declares public domain; Nikolaus Hansen | Positive-weight CMA-ES equation reference. |
| [MaskGIT](https://github.com/google-research/maskgit/tree/1db23594e1bd328ee78eadcd148a19281cd0f5b8) | Apache-2.0; Google LLC 2022 | Discrete iterative sampling and remasking reference. |
| [Sonnet VQ-VAE](https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py) | Apache-2.0; Sonnet Authors 2018 | Non-EMA VQ formula/gradient reference. |
| [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity/tree/082bb24f84c091ea94de2867d34c4544f68e0963) | BSD-2-Clause; Richard Zhang et al. 2018 | Full notice: docs/third_party/LPIPS_LICENSE.txt; no perceptual weights bundled. |
| [CompVis latent-diffusion](https://github.com/CompVis/latent-diffusion/tree/a506df5756472e2ebaf9078affdde2c4f1502cd4) | MIT; Machine Vision and Learning Group, LMU Munich 2022 | Full notice: docs/third_party/LATENT_DIFFUSION_LICENSE.txt. |
| [taming-transformers](https://github.com/CompVis/taming-transformers/tree/3ba01b241669f5ade541ce990f7650a3b8f65318) | MIT; Patrick Esser, Robin Rombach, Bjorn Ommer 2020 | Full notice: docs/third_party/TAMING_TRANSFORMERS_LICENSE.txt; no weights bundled. |
| [DeepSpec / DSpark](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2) | MIT; The DeepSpec Authors 2026 | Full notices: docs/third_party/DEEPSPEC_LICENSE.txt and DEEPSPEC_NOTICE.txt; weight terms are separate. |
| [LeWorldModel](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac) | MIT; Lucas Maes 2026 | Full notice: docs/third_party/LEWM_LICENSE.txt; no visual weights or PushT data bundled. |

Additional source-level notices, including NVIDIA GR00T and Cosmos/OpenMDW references, remain in the corresponding modules and technical documentation. Preserve them when moving or modifying code. Do not assume their terms are interchangeable with Apache-2.0.

## Outstanding licensing requirements

1. Review files with non-commercial, share-alike, unknown, or custom-license reference relationships.
2. Obtain permission, retain required notices, or replace restricted derivative expression through an appropriate independently documented process.
3. Select the license for the original work with its copyright owner.
4. Review model weights, evaluation data, and optional dependencies separately.
5. Confirm the final publish set and run the [release checklist](docs/RELEASING.md).

Existing full notices are in [docs/third_party](docs/third_party). No model weights or benchmark datasets are bundled. Source hashes identify content; they are not legal clearance or third-party signatures.
