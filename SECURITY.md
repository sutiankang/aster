# Security policy

Aster v0.1.0 has experimental APIs and is not a hardened multi-tenant service. See [the status page](docs/STATUS.md) for supported deployment boundaries.

## Report privately

Private vulnerability reporting is enabled. Use **Security → Report a vulnerability** or [open a private advisory](https://github.com/sutiankang/aster/security/advisories/new) to contact the repository maintainer without disclosing the issue publicly.

If that channel is unavailable, request a private contact without posting exploit details, credentials, private data, or a working destructive payload in a public issue. No security email address is invented by this repository.

## Important boundaries

- Model output and tool-returned content are untrusted data.
- Tool execution requires explicit host capabilities and approvals.
- MCP process grants authorize trusted local processes; they are not an OS sandbox.
- Linux bubblewrap isolation is optional; unavailable isolation must not silently fall back to unrestricted execution.
- Hashes establish content identity, not trust in a source or permission to redistribute it.
- Do not load untrusted pickle files or enable remote-code execution.
- Do not expose development HTTP endpoints to an untrusted network without authentication, isolation, quotas, and operational supervision.

Include the affected revision, environment, minimal reproduction, and expected versus actual boundary in a private report.
