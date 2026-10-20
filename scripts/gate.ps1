#!/usr/bin/env pwsh
# The full local gate, mirroring CI (.github/workflows/ci.yml): lint, format, types, then the
# full non-LLM geometry suite in parallel (xdist). Run before every commit that touches shared
# code. The `-m "not llm"` here is explicit (it is also the pyproject addopts default) so a bare
# run never spends against the live API key in .env -- opt into spend with `-m llm`.
#
# Fast inner loop for focused work (sub-30s):
#   uv run pytest tests/unit -m "not slow and not llm" -n auto
$ErrorActionPreference = "Stop"

Write-Host "== ruff check ==" -ForegroundColor Cyan
uv run ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== ruff format --check ==" -ForegroundColor Cyan
uv run ruff format --check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== mypy src/forge_mind ==" -ForegroundColor Cyan
uv run mypy src/forge_mind
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== pytest -m 'not llm' -n auto ==" -ForegroundColor Cyan
uv run pytest -m "not llm" -n auto
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "GATE GREEN" -ForegroundColor Green
