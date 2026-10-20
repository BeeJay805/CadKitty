"""Tendon-routability audit v1: the R4 finger's per-part gate, pinned to the flagship
brief's own dimensions (d1.2 channel through a 32 mm proximal phalanx)."""

from __future__ import annotations

import pytest

from forge_mind.agentic.servo_fit_audit import Hole
from forge_mind.agentic.tendon_audit import match_tendon_channel

_Z = (0.0, 0.0, 1.0)


def _hole(d: float) -> Hole:
    return Hole(point=(0.0, 0.0, 0.0), axis=_Z, diameter=d)


def test_the_flagship_phalanx_channel_passes() -> None:
    # The proximal phalanx: 32 mm long, d1.2 channel running its full length.
    assert match_tendon_channel([_hole(1.2)], 32.0, [(0.0, 32.0)]) is None


def test_a_dimple_is_not_a_through_channel() -> None:
    # Same diameter, 6 mm deep pocket in a 32 mm part: gates.
    assert match_tendon_channel([_hole(1.2)], 32.0, [(0.0, 6.0)]) is not None


def test_no_channel_at_all_gates() -> None:
    # The solid-phalanx class (the desk_grommet law at finger scale).
    assert match_tendon_channel([], 32.0, []) is not None


def test_oversize_channel_gates() -> None:
    # A d3.4 screw hole is not a tendon fair-lead.
    assert match_tendon_channel([_hole(3.4)], 32.0, [(0.0, 32.0)]) is not None


def test_undersize_channel_gates() -> None:
    assert match_tendon_channel([_hole(0.8)], 32.0, [(0.0, 32.0)]) is not None


def test_unmeasurable_extent_fails_closed() -> None:
    # In-band diameter but no measurable axial span: cannot certify -> gates.
    assert match_tendon_channel([_hole(1.2)], 32.0, [None]) is not None


def test_flag_off_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from forge_mind.agentic.tendon_audit import tendon_findings

    monkeypatch.delenv("FORGE_TENDON_AUDIT", raising=False)
    brief = SimpleNamespace(features=[SimpleNamespace(name="tendon_channel", standard=None)])
    assert tendon_findings(brief, part=None, part_longest_mm=32.0) == []
