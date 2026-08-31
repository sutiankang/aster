# Contributing to Aster

Thank you for improving the project. Start with a concrete workflow, reproducible problem, or narrowly scoped proposal.

**Release status:** repository-wide licensing is not yet resolved. Do not contribute copied or restricted third-party implementation code without identifying its origin and permission. See [NOTICE.md](NOTICE.md).

## Development setup

~~~bash
python -m pip install -e ".[test,dev]"
python tools/check_repository.py
ruff check src tests examples tools
ruff format --check src tests examples tools
python -m pytest tests/unit tests/integration -q
~~~

Optional reference and GPU dependencies belong in their test tiers, not in the core model runtime.

The default branch is `main`. Open focused pull requests against it; numbered tags identify release snapshots. Do not move an existing release tag when making a fix: record a new patch version and its verification results.

## Before opening a pull request

- Explain the behavior changed and the supported input/model/device domain.
- Link primary references and preserve required copyright notices.
- Add an independent correctness test, including gradients or resume when applicable.
- Route new objectives through the shared Trainer and new deployments through existing artifacts/runners.
- Document unsupported combinations and the actual dependency boundary.
- Keep runtime credentials, private data, local paths, weights, and generated run output out of the repository.

Use English for code comments and public API docstrings. Explain formulas, tensor layouts, ownership, and non-obvious decisions. Remove patch chronology and repetitive claims such as “audited in this round”; do not remove security constraints or attribution.

## Test changes

Start with the smallest affected test, then run the relevant integration/parallel suite. A skip is acceptable when its dependency is genuinely absent and its reason is explicit. Do not weaken assertions to make a reference comparison pass.

No task requires public leaderboard claims. Performance changes should include a reproducible benchmark and a quality comparison when approximation is involved.

## Scope

Prefer one coherent change per pull request. Discuss new dependencies and major public APIs before implementation. Good first contributions include missing regression cases, runnable examples, and translations that preserve technical meaning.
