# Contributing

Forge-Mind accepts changes that improve editable CAD generation, inspection, and reliable delivery. Keep each change small enough to review. Preserve failure evidence and the scope of every check.

## Development setup

Use Python 3.11 or 3.12 and uv. Run commands from the repository root. Start with the locked base environment:

```powershell
uv sync --locked --python 3.11
```

Workspace and MCP tests require the `server`, `mcp`, and `fea` extras. The full development environment uses all extras:

```powershell
uv sync --locked --all-extras --python 3.11
```

Review optional dependency terms before installation or distribution. In particular, `cad_analysis` adds Gmsh; its licensing needs separate review. Native libraries can have platform requirements.

## Make and verify a change

1. Describe the observed problem and the intended result.
2. Inspect the existing implementation before adding a new path.
3. Add a focused regression when behavior changes.
4. Run the smallest relevant checks first.
5. Run the complete gate before proposing integration.

```powershell
uv run --locked --all-extras ruff check .
uv run --locked --all-extras ruff format --check .
uv run --locked --all-extras mypy src/forge_mind
uv run --locked --all-extras pytest -s
```

The default pytest selection excludes tests marked `llm`. Other tests can still build native CAD and take time. Do not enable provider-backed tests or benchmark scripts without reviewing their calls, time limits, and cost. Tests that depend on unavailable local archives must report a skip, not a fabricated pass.

For an initial focused check:

```powershell
uv run --locked --all-extras pytest tests/unit/workspace/test_author_memory.py
```

Use an existing relevant test path for your change. Include the exact commands and outcomes in the pull request. Report skips, platform limits, and unresolved failures. A passing unit test does not establish generation quality or physical performance.

## Review expectations

Preserve immutable revision identities, ownership checks, cancellation, artifact hashes, and approval boundaries. Do not make a failure pass by lowering a valid threshold. New checks must state inputs, assumptions, units, pass/fail rules, and unsupported cases. Return unverified when the evidence cannot establish the claim.

Keep documentation consistent with behavior. Do not commit credentials, user projects, private prompts, model transcripts, generated bulk outputs, or assets without clear redistribution rights. Add a dependency only when the existing code cannot meet the requirement simply.

The project license is MIT. Submit only material you have the right to contribute; identify any third-party license or attribution requirements.

