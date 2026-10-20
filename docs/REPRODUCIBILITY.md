# Reproduce this release

`SNAPSHOT.json` records the exact source, tests, fixtures, documents and example files in this public snapshot. `RELEASE_QUALIFICATION.json` records the exercised checks, platform, results and scope. An inventory alone is not a passing test result.

## Locked environment

The reference local environment uses Windows, CPython 3.11.9 and uv 0.12.5. The repository declares Python 3.11 and 3.12 support. A separate Linux CI run appears on the repository's Actions page; consult its actual result.

From the repository root:

```powershell
uv sync --locked --all-extras --python 3.11
uv run --locked --all-extras ruff check .
uv run --locked --all-extras ruff format --check .
uv run --locked --all-extras mypy src/forge_mind
uv run --locked --all-extras pytest -n 2 --dist loadfile --tb=short
```

The lock has platform and Python-version conditions. For example, it selects different tifffile versions for Python 3.11 and 3.12. Preserve those conditions; a version from another Python branch is not a lock mismatch.

The base quickstart and full development environment have different dependencies. Follow the README's consistent extras for the workspace and MCP bridge. Native library requirements and optional dependency licenses still apply.

## What each check covers

- The full suite exercises deterministic code and native CAD controls. The default selection excludes tests marked `llm`.
- The CLI quickstart builds the included NEMA 17 fixture and exports CAD files.
- The MCP smoke uses a real stdio client, the loopback HTTP service and a fresh workspace. It exercises edits, saved inspections, exports and sampled motion. It requests no built-in model calls and does not accept the candidate.
- Generated examples contain their own evidence and limits. Offline test totals do not establish a generation success rate.

The release retains failures and unknowns. Physical fit, load capacity and manufactured performance need separate evidence.

## File identity

The repository preserves exact file bytes through Git with `* -text` in `.gitattributes`. This keeps source and CAD hashes stable across clones. Text files can retain either LF or CRLF line endings.

Each manifest member has a SHA-256 digest and byte count. The release checks actual Git blobs and a fresh checkout against those values. Qualification records are added separately; the source inventory does not contain its own test result or hash.

The public repository starts with a clean source snapshot. Private project stores, model transcripts, account configuration, deployment setup and previous repository history are excluded. Historical benchmark specifications have explicit availability notes; they are not current results.