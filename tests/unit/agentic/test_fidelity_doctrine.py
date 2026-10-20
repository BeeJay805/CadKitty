"""Form-fidelity levers (the L88 measured gap: every part accepted, yet the humanoid
assembly reads as a flat-plate lab prototype -- box limbs, bare pins, no proportion
spec, no form target in the judge's reference). GENERALIZED at founder direction
(2026-07-23): crafted-form rules for EVERY object class + a humanoid annex.

Two independent flags, both default OFF and byte-identical off: FORGE_FIDELITY_DOCTRINE
(plan layer) and FORGE_FIDELITY_BRIEF (brief layer). The tapered_limb_shell /
joint_module idioms are unflagged cheatsheet vocabulary, execution-verified through the
real sandbox by test_cheatsheet_idioms.py like every idiom."""

from __future__ import annotations

import pytest

from forge_mind.agentic.brief import _FIDELITY_BRIEF_ADDENDUM, fidelity_brief_enabled
from forge_mind.agentic.functional import (
    _DECOMPOSE_SYSTEM,
    _FIDELITY_DOCTRINE,
    _PLAN_DOCTRINE,
    _decompose_system,
)


def _clear_overlay_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for flag in (
        "FORGE_PLAN_DOCTRINE",
        "FORGE_FIDELITY_DOCTRINE",
        "FORGE_ENG_DOCTRINE",
        "FORGE_CHIEF",
    ):
        monkeypatch.delenv(flag, raising=False)


def test_fidelity_doctrine_is_general_not_robot_only() -> None:
    # Founder directive 2026-07-23: crafted form for EVERY object, not a robot rule.
    assert "FORM-FIDELITY DOCTRINE (every object)" in _FIDELITY_DOCTRINE
    assert "CARDS STATE THE FORM LANGUAGE" in _FIDELITY_DOCTRINE
    assert "PRIMITIVES ARE SCAFFOLDING, NOT FINAL FORM" in _FIDELITY_DOCTRINE
    assert "SURFACE DETAIL IS PART OF THE PART" in _FIDELITY_DOCTRINE
    # The prismatic escape hatch (a flat plate is honestly flat) must survive edits --
    # without it the doctrine taxes every honest prism into a false defect.
    assert "genuinely prismatic" in _FIDELITY_DOCTRINE


def test_fidelity_doctrine_humanoid_annex_states_the_table() -> None:
    assert "HUMANOID / FIGURE ANNEX" in _FIDELITY_DOCTRINE
    for station in ("knee axis 105", "hip axis 180", "shoulder axis 290", "head 45"):
        assert station in _FIDELITY_DOCTRINE
    assert "0.30H -> 0.25H" in _FIDELITY_DOCTRINE  # the shoulder -> waist taper
    assert "RECTANGULAR-BOX LIMBS ARE FORBIDDEN" in _FIDELITY_DOCTRINE
    # It routes authors to the EXECUTED idioms by name, and fit still beats styling.
    assert "tapered_limb_shell" in _FIDELITY_DOCTRINE
    assert "joint_module" in _FIDELITY_DOCTRINE
    assert "FORM NEVER OVERRIDES FIT" in _FIDELITY_DOCTRINE


def test_decompose_system_off_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_overlay_flags(monkeypatch)
    assert _decompose_system() == _DECOMPOSE_SYSTEM


def test_decompose_system_flags_compose_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_overlay_flags(monkeypatch)
    monkeypatch.setenv("FORGE_FIDELITY_DOCTRINE", "1")
    assert _decompose_system() == _DECOMPOSE_SYSTEM + _FIDELITY_DOCTRINE
    monkeypatch.setenv("FORGE_PLAN_DOCTRINE", "1")
    assert _decompose_system() == _DECOMPOSE_SYSTEM + _PLAN_DOCTRINE + _FIDELITY_DOCTRINE
    monkeypatch.delenv("FORGE_FIDELITY_DOCTRINE")
    assert _decompose_system() == _DECOMPOSE_SYSTEM + _PLAN_DOCTRINE


def test_brief_addendum_flag_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_FIDELITY_BRIEF", raising=False)
    assert fidelity_brief_enabled() is False
    monkeypatch.setenv("FORGE_FIDELITY_BRIEF", "1")
    assert fidelity_brief_enabled() is True


def test_brief_addendum_carries_the_crafted_form_language() -> None:
    assert "every object" in _FIDELITY_BRIEF_ADDENDUM
    assert "DESIGNED, CRAFTED" in _FIDELITY_BRIEF_ADDENDUM
    assert "flat plates" in _FIDELITY_BRIEF_ADDENDUM
    assert "primitives-only" in _FIDELITY_BRIEF_ADDENDUM
    assert "comparison target" in _FIDELITY_BRIEF_ADDENDUM
