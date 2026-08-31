# Architecture decisions

## Native core, optional reference tools

Use PyTorch and NumPy as computational foundations. Upstream model/train/serve frameworks are references, not hidden runtime owners. Official evaluators may be explicitly enabled.

## Narrow model and state contracts

Token predictors, representation encoders, conditional fields, latent dynamics, and action policies share lifecycle tools without pretending they all have token KV state. Unsupported state operations fail explicitly.

## One owner for training state

Methods declare roles and phases; the trainer owns backward, communication, optimizers, schedulers, EMA, and checkpoints. Parameters have stable logical names and unique ownership.

## Independent loss denominators

Each term is normalized in its actual sample-reduction group. Token, pair, query, transition, and latent-coordinate counts are not interchangeable.

## Immutable deployment identities

Weights, configuration, processors, and lineage define an artifact. Serving requests bind a version instead of reading a base model while another request mutates it.

## Explicit failure and security boundaries

Approvals do not survive replay. Ambiguous effects are not silently retried. An unavailable isolation backend does not imply permission to run unrestricted commands.

## Comparable optimization measurements

Use the same workload, preprocessing, quality protocol, and failure denominator for baseline/candidate comparisons. Distinguish client/server clocks and cold/warm execution.
