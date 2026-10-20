"""Regression pins for the six sense-fixes (engine deep review, 2026-07-04).

The trajectory autopsies proved the persistent both-arm bench failures were manufactured by
FALSE FINDINGS -- the validator's measurement/accounting layers, not author capability. Each
pin below reproduces one measured false finding synthetically and asserts it no longer fires,
plus the matching TRUE-positive so the fix cannot have weakened the gate.
"""

from __future__ import annotations

from typing import Any

import pytest
from build123d import Box, BuildPart, Cone, Cylinder, Locations, Mode, Pos, Rot, Sphere

from forge_mind.agentic.brep_checks import brep_report, edge_adjacency
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.spatial import (
    _DIA_RE,
    _expected_through_holes,
    measured_hole_feedback,
)
from forge_mind.agentic.standards_audit import audit_standards
from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

pytestmark = pytest.mark.slow  # a few B-rep builds; keep out of the hot loop


def _feat(name: str, dims: str = "", count: int = 1, method: str = "bore") -> BriefFeature:
    return BriefFeature(name=name, purpose="p", approx_dims_mm=dims, method=method, count=count)


def _brief(feats: list[BriefFeature]) -> DesignBrief:
    return DesignBrief(
        object_name="part",
        summary="s",
        visual_description="v",
        overall_envelope_mm="50 x 50 x 10 mm",
        features=feats,
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


# --- S1: degenerate pole/apex edges are not naked (hook_coat false-FAIL) ----------------


def test_sphere_and_cone_are_sealed() -> None:
    """Sphere(5) alone reported 2 naked edges and hard-gated `passed` -- any part with a
    bulb tip / dome / revolve apex was permanently unpassable."""
    for solid in (Sphere(5), Cone(5, 0, 10)):
        r = brep_report(solid)
        assert r.open_edges == 0 and not r.open_or_nonmanifold


def test_genuinely_open_boundary_still_flags() -> None:
    face = Box(10, 10, 10).faces()[0]
    open_e, _, _ = edge_adjacency(face)
    assert open_e > 0  # the fix must not have blinded the real-opening detector


# --- S2: hole-plan census reads the feature HEAD, honors depth (phantom demands) --------


def test_solid_head_compound_plans_no_hole() -> None:
    # 'bolt ear' (a solid tab) once counted as 2 planned holes -> pipe clamp demanded 6-of-5
    b = _brief(
        [
            _feat("bolt ears", count=2),
            _feat("M5 ear clearance holes", count=2),
            _feat("wall mounting holes", count=2),
        ]
    )
    assert _expected_through_holes(b) == 4


def test_modifier_head_compound_plans_no_hole() -> None:
    # 'bore chamfer' x2 once demanded genus 3 of a correct genus-1 pulley
    b = _brief([_feat("central bore"), _feat("bore chamfer", count=2, method="chamfer")])
    assert _expected_through_holes(b) == 1


def test_stated_depth_without_through_is_blind() -> None:
    # the knob's 'M3 set-screw radial hole 5 mm deep' was ordered to pierce
    b = _brief([_feat("M3 set-screw radial hole", dims="M3, 5 mm deep")])
    assert _expected_through_holes(b) == 0


def test_bracket_bug_pin_bolt_holes_with_pocket_method_still_count() -> None:
    # the ORIGINAL measured bug this census exists for: name says holes, method says pocket
    b = _brief([_feat("bolt holes", count=4, method="pocket")])
    assert _expected_through_holes(b) == 4


def test_through_keyword_overrides_depth() -> None:
    b = _brief([_feat("pivot holes", dims="10 mm dia through, 25 mm deep", count=2)])
    assert _expected_through_holes(b) == 2


# --- S3: diameter parser never reads a depth as a diameter (knob false order) ------------


def test_depth_is_not_a_diameter() -> None:
    m = _DIA_RE.search("set-screw bore, depth 5 mm")
    assert m is None or float(m.group(1) or m.group(2)) != 5.0


def test_through_bore_stated_diameter_wins_over_countersink() -> None:
    # hook_coat: '5 mm through-bore, 10 mm countersink dia' once demanded a Ø10 bore
    m = _DIA_RE.search("5 mm through-bore, 10 mm countersink dia")
    assert m is not None and float(m.group(1) or m.group(2)) == 5.0


def test_plain_diameter_forms_still_parse() -> None:
    for text, want in [("dia 6 mm", 6.0), ("6 mm diameter", 6.0), ("Ø4.4 mm", 4.4)]:
        m = _DIA_RE.search(text)
        assert m is not None and float(m.group(1) or m.group(2)) == want


# --- S4: a 180-degree opposed hole pair measures as TWO penetrations (mech_star) ---------


def _hub_with_opposed_setscrews() -> Any:
    with BuildPart() as bp:
        Cylinder(radius=11, height=20)
        for ang in (0, 180):
            with Locations(Rot(0, 0, ang) * Pos(11, 0, 0)):
                Cylinder(radius=2.0, height=8, rotation=(0, 90, 0), mode=Mode.SUBTRACT)
    return bp.part


def test_opposed_radial_pair_counts_two_segments() -> None:
    part = _hub_with_opposed_setscrews()
    stacks = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]
    d4 = [s for s in stacks if abs(2 * s.radius - 4.0) <= 0.3]
    assert d4, "the set-screw stack must be measured"
    assert sum(max(1, s.segments) for s in d4) >= 2  # was 1: unmeasurable-by-construction


def test_measured_audit_is_silent_on_the_correct_opposed_pair() -> None:
    part = _hub_with_opposed_setscrews()
    b = _brief([_feat("radial set-screw through holes", dims="4 mm dia through", count=2)])
    assert measured_hole_feedback(part, b) == []


# --- S5: nearest-diameter assignment; specs cannot steal each other's holes (cam) --------


def test_no_cross_spec_hole_theft() -> None:
    # a Ø5.0 spec once consumed the Ø4.5 corner holes (both in the 0.6 window) and reported
    # the CORNER holes missing -- the author rewrote a healthy block five times
    with BuildPart() as bp:
        Box(60, 40, 6)
        with Locations((-15, 0, 0)):
            Cylinder(radius=2.5, height=10, mode=Mode.SUBTRACT)  # Ø5.0
        with Locations((15, 0, 0)):
            Cylinder(radius=2.25, height=10, mode=Mode.SUBTRACT)  # Ø4.5
    b = _brief(
        [
            _feat("detent holes", dims="5 mm dia through", count=1),
            _feat("corner mounting holes", dims="4.5 mm dia through", count=1),
        ]
    )
    assert measured_hole_feedback(bp.part, b) == []  # each spec finds its own hole


# --- S6: the auditor accepts the helper the cheatsheet teaches (mech_star refusal) -------


def test_thread_hole_d_is_a_recognized_standards_fetch() -> None:
    cites, fails = audit_standards("d = thread_hole_d('M4')", ["M4"])
    assert fails == []
    assert any("4.00" in c for c in cites)  # cited at NOMINAL (ADR-0081), not tap/clearance
