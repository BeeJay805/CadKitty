"""Coaxial SEGMENT counting: two clevis ears drilled on one pivot axis are ONE coaxial
stack but TWO physical wall penetrations. Counting stacks alone audited a drilled ear
pair as 1-of-2 (measured live on the v2d turret) and sent the author chasing a phantom
missing hole. Segments are computed from member-face axial extents; unmeasurable extents
conservatively collapse to 1 (never fabricate a separation)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from build123d import Box, Cylinder, Pos, Rot

from forge_mind.agentic.spatial import measured_hole_feedback
from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

pytestmark = pytest.mark.slow  # builds tiny B-reps (build123d/OCCT)


def _clevis(ear_gap: float = 30.0, bore_d: float = 6.0):
    """Two parallel ears on a base, drilled by ONE coaxial bore along Y."""
    part = Box(40, 60, 10)  # base, z -5..5
    for y in (-(ear_gap / 2 + 4), ear_gap / 2 + 4):  # ears 8 thick, inner faces +/-15
        part += Pos(0, y, 15) * Box(24, 8, 30)
    # one drill through BOTH ears: a Y-axis cylinder spanning the whole width
    part -= Pos(0, 0, 25) * Rot(90, 0, 0) * Cylinder(bore_d / 2, 80)
    return part


def test_two_ears_one_axis_count_two_segments() -> None:
    part = _clevis()
    stacks = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]
    assert len(stacks) == 1  # ONE coaxial stack (same axis, same radius)...
    assert stacks[0].segments == 2  # ...but TWO separated wall penetrations


def test_single_wall_bore_stays_one_segment() -> None:
    part = Box(60, 60, 8) - Pos(0, 0, 0) * Cylinder(3, 20)
    stacks = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]
    assert len(stacks) == 1
    assert stacks[0].segments == 1


def test_hole_audit_credits_ear_pairs() -> None:
    """A brief planning 2 pivot bores is SATISFIED by one through-drilled ear pair."""
    part = _clevis()
    brief = SimpleNamespace(
        features=[
            SimpleNamespace(
                name="pivot bores",
                approx_dims_mm="Ø6 mm dia through each ear",
                count=2,
                standard=None,
            )
        ]
    )
    assert measured_hole_feedback(part, brief) == []  # no phantom deficit


def test_hole_audit_still_names_a_real_deficit() -> None:
    """One drilled ear pair does NOT satisfy a 4-bore plan -- the deficit stays honest."""
    part = _clevis()
    brief = SimpleNamespace(
        features=[
            SimpleNamespace(name="pivot bores", approx_dims_mm="Ø6 mm dia", count=4, standard=None)
        ]
    )
    lines = measured_hole_feedback(part, brief)
    assert len(lines) >= 1 and "MISSING 2 x" in lines[0]
