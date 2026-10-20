"""Toolbelt + preflight v0 (ADR-0048, P6): a deterministic preflight that reports each tool honestly,
blocks on a missing required tool, and -- with no model wired -- keeps the OFFLINE build available
while marking LIVE concept generation blocked and never implying a fake fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.invention.toolbelt import REGISTRY, ToolStatus, run_preflight


def _clear_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FORGE_FORBID_API_SPEND", raising=False)


def test_offline_build_available_with_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_model_env(monkeypatch)
    report = run_preflight()
    assert report.offline_build_available is True
    assert report.ready is True
    assert report.live_concept_generation == "blocked"
    assert any("NO fake fallback" in n for n in report.notes)
    assert any("Offline build mode is AVAILABLE" in n for n in report.notes)
    by_id = {c.tool_id: c for c in report.checks}
    assert by_id["cad_engine"].status is ToolStatus.AVAILABLE
    assert by_id["validators"].status is ToolStatus.AVAILABLE
    assert by_id["benchmark_fixtures"].status is ToolStatus.AVAILABLE
    assert by_id["model_client"].status is ToolStatus.NOT_CONFIGURED


def test_a_configured_model_unblocks_live(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "claude_code")
    report = run_preflight()
    assert report.live_concept_generation == "available"
    by_id = {c.tool_id: c for c in report.checks}
    assert by_id["model_client"].status is ToolStatus.AVAILABLE


def test_spend_ban_marks_model_unavailable_but_offline_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("FORGE_FORBID_API_SPEND", "1")
    report = run_preflight()
    by_id = {c.tool_id: c for c in report.checks}
    assert by_id["model_client"].status is ToolStatus.UNAVAILABLE
    assert report.live_concept_generation == "blocked"
    assert report.offline_build_available is True  # the model is not required for the offline build


def test_a_missing_required_tool_blocks_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_model_env(monkeypatch)
    report = run_preflight(fixtures_path=tmp_path / "absent.json")
    assert report.offline_build_available is False
    assert report.ready is False
    assert any("benchmark_fixtures" in b for b in report.blockers)


def test_vendor_fetch_is_not_configured_with_a_fail_closed_fallback() -> None:
    report = run_preflight()
    vendor = next(c for c in report.checks if c.tool_id == "vendor_fetch")
    assert vendor.status is ToolStatus.NOT_CONFIGURED
    assert vendor.fallback is not None and "fail-closed" in vendor.fallback


def test_registry_covers_the_lifecycle_tools() -> None:
    ids = {entry.tool_id for entry in REGISTRY}
    assert {
        "cad_engine",
        "validators",
        "artifact_storage",
        "vendor_fetch",
        "model_client",
        "benchmark_fixtures",
    } <= ids
