# Architecture

Aster separates mathematical computation from lifecycle ownership.

| Component | Owns | Must not own |
| --- | --- | --- |
| Model | Parameters, tensor computation, typed state | Optimizer steps, HTTP, implicit downloads |
| Objective | Loss numerator, valid count, gradient semantics | Backward, hidden optimizer state |
| Trainer | Roles, reductions, updates, EMA, checkpoint boundaries | Task-specific transport |
| Artifact store | Immutable content and preprocessing identity | An implicit executable model loader |
| Runner | Model execution and admitted state layout | Network permissions |
| Scheduler | Requests, capacity, cancellation, cache ownership | Model architecture aliases |
| Transport | HTTP/SSE or MCP messages | Granting permissions from model text |
| Evaluator | Fixed cohorts, metrics, failed-sample accounting | Invented pretrained scores |

## Add an objective

Return LossTerm or LossBundle and keep counts explicit. Use the same Trainer for backward and state. If a method needs several roles, declare the order of phases and which parameters freeze; do not add a second hidden optimizer.

A frozen network may still need input gradients. Use parameter freezing, not no_grad, when the upstream actor or generator depends on those gradients.

## Add a model or execution layout

Implement the actual architecture and configuration. Reuse operators only when tensor layouts, normalization, position semantics, and cache behavior match. Add same-weight forward/gradient/cache tests over a declared configuration range.

Parallel providers must define logical parameter names, ownership, reduction domains, and export rules. A CPU multi-process result does not establish multi-node GPU performance.

## Make cancellation safe

Threads and native kernels may continue after an asyncio task is cancelled. Hold leases until the worker settles, reclaim state or persist tool receipts, then propagate cancellation. Repeated cancellation must not bypass this sequence.

## Extend without a second framework

The preferred path is:

~~~text
new model/objective -> existing Trainer -> existing artifact format
                   -> native Runner -> existing evaluator
~~~

A new name alone is not a new implementation. Keep unsupported combinations explicit and document the smallest meaningful supported workflow.
