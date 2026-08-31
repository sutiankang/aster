# Release checklist

Public release is intentionally manual. No workflow publishes packages or changes repository visibility.

This is a reusable release and licensing checklist, not a live progress report. The repository is publicly accessible; this alone does not complete licensing review. See [current status](STATUS.md) for completed checks and unresolved requirements, and [versioned releases](https://github.com/sutiankang/aster/releases) for immutable build identities. Recheck the applicable items for every new release candidate.

## Requirements for a license-cleared release

- [ ] Resolve file-level reference/derivative-code licensing, including non-commercial or unverified sources.
- [ ] Choose the repository-wide license with the copyright owner and retain third-party notices.
- [ ] Confirm repository owner/name and maintainer identity.
- [ ] Configure private security and conduct reporting channels.
- [ ] Review the complete publish set for private documents, tokens, personal paths, and data.
- [ ] Run the configured GitHub workflows on the final repository and inspect every failure/skip.
- [ ] Build and inspect wheel and source distribution contents.
- [ ] Verify quickstart commands from a clean checkout/environment.
- [ ] Check both homepages, algorithm/source links, and learning exercises after packaging.
- [ ] Preview the banner and workflow diagram in light/dark themes and at a narrow width.
- [ ] Review support boundaries and avoid unsupported compatibility/performance claims.

## Recommended repository settings

Use a concise description: **Native PyTorch workflows for training, fine-tuning, inference, and evaluation.**

Relevant topics, if they accurately describe the release: pytorch, machine-learning, llm, vlm, vla, multimodal, vision-language-models, vision-language-action, lora, distributed-training, inference, diffusion-models, world-models, reinforcement-learning.

Protect the default branch with the configured checks. Enable issues and optionally discussions, select a social preview only when an actual asset exists, and make a small focused first release.

The original [banner](assets/aster-banner.svg) and [workflow diagram](assets/workflow.svg) are self-contained SVG assets used by both homepages. Keep their accessible titles/descriptions. If the hosting platform requires a raster social preview, export and inspect it separately; do not assume the SVG is an accepted upload format.

Create the repository from the Aster project boundary, not an enclosing workspace containing other projects. Review the staged file list before any push; exclude publication backups, local environments, model weights, run output, and private records. Choose the destination and visibility explicitly.

Do not add fake stars, badges, testimonials, benchmark scores, affiliations, or a DOI. Add repository URLs and a license badge only after the real repository and license exist.

## Packaging

For future releases, keep published version identities stable: do not move a published tag or replace its package bytes to update documentation. Record later documentation changes on `main`; code or packaging changes require a new version.

~~~bash
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/*
~~~

Inspect archives for only intended source, documentation, tests, examples, and required notices. Model weights, checkpoints, run output, private research records, and local environments do not belong in a source release.

The source distribution includes both homepages, the learning guides, and SVG assets. Before publishing on PyPI, replace or transform repository-relative README links/images using the confirmed repository URL and release revision; local relative links alone do not provide a working PyPI project page. No remote URL is invented before the destination exists.

## References behind the repository structure

The homepage uses task-oriented navigation and a short first experiment, informed by [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [Lightning](https://github.com/Lightning-AI/pytorch-lightning). The learning route follows a question → code → experiment pattern, informed by [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch), [LabML's annotated implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations), and [Spinning Up](https://github.com/openai/spinningup).

The contributor/test organization draws from [TRL](https://github.com/huggingface/trl) and [vLLM](https://github.com/vllm-project/vllm). Community files follow [GitHub's guidance](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/creating-a-default-community-health-file). Aster's text and visual assets are original; references do not imply affiliation, endorsement, or permission to reuse upstream branding.

These choices improve usability and contribution readiness; they do not guarantee popularity.
