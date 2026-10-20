# Security

Forge-Mind builds and inspects CAD. Its local execution mode is intended for trusted users and trusted inputs on a private computer.

## Trust boundaries

- Authored Python CAD can execute code. Treat it like a program you received from another person.
- Keep `trusted_local` workspaces bound to loopback. Do not expose this development service to the Internet.
- Loopback access is not a sandbox. Local mode uses one local owner and trusts processes on the same computer.
- Treat uploaded documents, images, imported models, and model responses as untrusted inputs.
- Keep credentials and private projects outside commits and shared example bundles.
- Artifact hashes bind saved bytes. They do not prove safe geometry, safe code, ownership of imported material, or physical fitness.
- CAD and engineering checks have limited scopes. Review unverified findings before any physical use.

The public snapshot does not provide a hardened hosted-service deployment. Do not use its local setup as a multi-user production configuration.

## Report a vulnerability

Do not put secrets, private models, or exploit details in a public issue. If this repository has GitHub private vulnerability reporting enabled, use **Security > Report a vulnerability**. If that option is unavailable, open an issue requesting a private reporting channel without disclosing the vulnerability.

Include the affected revision, a small reproduction, the expected trust boundary, and the observed result. Remove credentials and customer data. A response deadline is not currently promised.

## Supported versions

Report issues against the latest published snapshot. Older private development snapshots are not supported release branches. State whether the issue also occurs on the public snapshot.
