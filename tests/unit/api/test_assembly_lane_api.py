"""API wiring of the assembly lane: flag-gated, ships only its own gate PASS, and is
byte-inert when the flag is unset (money-free -- the agentic engine is stubbed)."""

from __future__ import annotations

from pathlib import Path

import pytest

import forge_mind.api.core as core


class _MustNotReachAgenticError(RuntimeError):
    pass


@pytest.fixture()
def no_agentic(monkeypatch: pytest.MonkeyPatch):
    def _boom(*a: object, **k: object) -> object:
        raise _MustNotReachAgenticError("agentic engine must not run in this test")

    monkeypatch.setattr(core, "produce_part", _boom)


def test_flag_off_is_byte_inert(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_agentic: None
) -> None:
    monkeypatch.delenv("FORGE_ASSEMBLY_LANE", raising=False)
    calls: list[str] = []
    real = core._generate_assembly

    def _spy(prompt: str, job_dir: Path, on_phase: object) -> object:
        calls.append(prompt)
        return real(prompt, job_dir, on_phase)  # pragma: no cover - must not happen

    monkeypatch.setattr(core, "_generate_assembly", _spy)
    res = core.generate_part("a pan-tilt camera head for two MG996R servos", tmp_path)
    # The lane was never consulted; the prompt fell to the (stubbed) agentic engine.
    assert calls == []
    assert res.status == "error" and "MustNotReachAgentic" in (res.error.message or "")


def test_flag_on_ships_the_arm_without_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_agentic: None
) -> None:
    monkeypatch.setenv("FORGE_ASSEMBLY_LANE", "1")
    res = core.generate_part("a pan-tilt camera head for two MG996R servos", tmp_path)
    assert res.status == "ok"
    ok = res.ok
    assert ok is not None
    assert ok.gen_path == "assembly:servo_arm"
    assert ok.cost_usd == 0.0  # deterministic: zero LLM spend
    assert ok.validation.passed
    assert ok.confidence is not None
    assert ok.confidence.basis == "validated"
    # dims anchor to the PROVISIONAL registry row: sourced, never verified
    assert ok.confidence.dimension_tiers.verified == 0
    assert ok.confidence.dimension_tiers.sourced > 0
    assert (tmp_path / "part.glb").exists()
    assert (tmp_path / "part.stl").exists()
    assert (tmp_path / "parts.json").exists()
    assert list((tmp_path / "parts").glob("**/*.step"))  # per-part named-B-rep STEPs
    # ADR-0044 increment: positioned assembly STEP + manifest block + populated BOM.
    assert (tmp_path / "part.step").exists()
    assert ok.assembly is not None
    assert any(not p.printed and p.source for p in ok.assembly.parts)  # cited hardware
    assert all(j.checked == "swept" for j in ok.assembly.joints)
    assert ok.kit.bom.lines  # never an empty BOM under the flagship lane again


def test_flag_on_non_matching_prompt_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_agentic: None
) -> None:
    monkeypatch.setenv("FORGE_ASSEMBLY_LANE", "1")
    # The gripper family (ADR-0044) has its OWN flag; without it a gripper prompt
    # still falls through -- the arm matcher remains byte-identical and declines.
    monkeypatch.delenv("FORGE_GRIPPER", raising=False)
    res = core.generate_part("a two-finger gripper driven by an MG996R servo", tmp_path)
    # the lane declined (different mechanism class); the stubbed agentic engine was hit
    assert res.status == "error" and "MustNotReachAgentic" in (res.error.message or "")
