# Testing

## Fast local checks

~~~bash
python -m pip install -e ".[test]"
python tools/check_repository.py
python -m pytest tests/unit tests/integration -q
~~~

## Test tiers

| Tier | Purpose | Requirements |
| --- | --- | --- |
| Unit | Formulas, gradients, masks, state ownership, errors | CPU PyTorch and the test extra (pytest, Pillow, safetensors) |
| Integration | Training, artifact, inference, evaluation, agent workflows | The test extra and local temporary storage |
| Distributed | Real multi-process collectives, updates, export, resume | Gloo or separately configured NCCL |
| Parity | Independent formulas and optional upstream-model comparisons | Declared oracle packages/source versions |
| GPU / benchmark | Hardware kernels and public quality/resource evidence | Explicit devices, weights, data, and approvals |

~~~bash
python -m pytest tests/distributed -q
python -m pytest tests/parity -q
~~~

Some parity cases skip without their optional packages, fixed source resources, or CUDA. Keep the skip reasons visible. Do not relabel a skip as an executed pass.

Pillow creates image fixtures; safetensors writes independent checkpoint fixtures for native loading and distributed inference tests. Both belong to the test extra, not the core runtime. A development machine with optional packages already installed is not a substitute for checking the declared extras in a clean environment.

## CI scope

Pull-request CI checks repository hygiene and runs CPU unit/integration tests. A separate manually dispatched workflow runs the longer distributed/parity suite. Both have read-only repository permissions and no publication credentials.

CI configuration is not itself proof that a hosted run passed. Add a passing-status badge only after the workflow has actually run in the final repository.

## Numerical changes

Use independent equations, same-weight comparisons, or full-window versus accumulated/distributed updates. Cover gradients and resumed next updates where relevant, not only forward shapes.

For prose-only changes, compare parsed executable syntax trees after removing docstrings, then run regression tests. Keep tokenization test data and user-facing strings intact when changing comments.

## Environments

The development environment used Python 3.13 and CPU PyTorch 2.11.0, with separately installed optional reference libraries. Other supported dependency versions need their own CI evidence. No blanket GPU speed or public benchmark result is claimed.
