# Design overview

Aster shares one native lifecycle across model domains while retaining their different tensor and state semantics.

The current public design is described in [Architecture](ARCHITECTURE.md). Start with [Getting started](GETTING_STARTED.md), [Losses](LOSSES.md), and [Support boundaries](ROADMAP.md).

## Principles

- Models compute; objectives describe losses; the trainer owns updates.
- Each loss declares its valid unit and reduction domain.
- Deployment artifacts bind weights and preprocessing without executable remote loaders.
- Different architectures require actual layers and state semantics, not aliases.
- Approximate transforms need quality evaluation as well as resource measurements.
- Failed, cancelled, and partially committed operations are explicit lifecycle states.
- Credentials, private research records, checkpoints, and generated runs stay outside the source repository.

[Technical interfaces](INTERFACES.md) provide lower-level contracts.
