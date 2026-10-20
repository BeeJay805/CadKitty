"""ADR-0120 interface audits: stated-OD + bore-ligament, pinned to the L68 measured cases.

The ligament numbers below are the ACTUAL manifold i4 geometry (ports r9.5 at x=+/-17,
bolt holes r3.3 at (+/-20, +/-12.5)): center distance 12.85 vs summed radii 12.80 =
0.05 mm web -- the defect the founder spotted in the render and no gate caught.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.interface_audit import check_bore_ligaments, check_stated_ods


def _brief(features: list[BriefFeature]) -> DesignBrief:
    return DesignBrief(
        object_name="part",
        summary="s",
        visual_description="v",
        overall_envelope_mm="150 x 120 x 180",
        features=features,
        min_wall_mm=2.0,
        print_orientation_note="flange down",
    )


def _stack(d: float, at, direction=(0.0, 0.0, 1.0), concave=True):
    return SimpleNamespace(radius=d / 2.0, axis_point=at, axis_dir=direction, concave=concave)


# ---- audit B: ligaments -----------------------------------------------------------


def test_l68_bolt_hole_into_port_gates() -> None:
    stacks = [
        _stack(19.0, (17.0, 0.0, -2.0)),
        _stack(6.6, (20.0, 12.5, -3.0)),
    ]
    findings = check_bore_ligaments(stacks, np)
    gating = [f for f in findings if f.gates]
    assert len(gating) == 1 and gating[0].check == "bore_ligament"
    assert "0.0" in gating[0].message  # the ~0.05 mm web is named


def test_intersecting_bores_stay_advisory() -> None:
    stacks = [
        _stack(10.0, (0.0, 0.0, 0.0)),
        _stack(10.0, (6.0, 0.0, 0.0)),  # ligament -4: full overlap (cross-drill class)
    ]
    findings = check_bore_ligaments(stacks, np)
    assert findings and not any(f.gates for f in findings)
    assert "INTERSECT" in findings[0].message


def test_far_slab_bores_never_pair() -> None:
    stacks = [
        _stack(19.0, (17.0, 0.0, -2.0)),
        _stack(6.6, (17.0, 5.0, 173.0)),  # the outlet flange, 175 mm away along the axis
    ]
    assert check_bore_ligaments(stacks, np) == []


def test_healthy_spacing_is_silent() -> None:
    stacks = [
        _stack(19.0, (17.0, 0.0, -2.0)),
        _stack(6.6, (34.0, 12.5, -3.0)),  # web ~8.6 mm
    ]
    assert check_bore_ligaments(stacks, np) == []


# ---- audit A: stated ODs ----------------------------------------------------------


def _runner_brief() -> DesignBrief:
    return _brief(
        [
            BriefFeature(
                name="runner tubes",
                purpose="carry exhaust",
                approx_dims_mm="24 mm outer diameter with 2.5 mm walls, sweeping",
                method="sweep",
                count=4,
            )
        ]
    )


def test_stated_od_off_gates() -> None:
    stacks = [_stack(31.0, (0.0, 0.0, 50.0), concave=False)]
    findings = check_stated_ods(_runner_brief(), stacks)
    gating = [f for f in findings if f.gates]
    assert len(gating) == 1 and "31.0" in gating[0].message and "24" in gating[0].message


def test_stated_od_in_band_is_silent_on_diameter() -> None:
    stacks = [_stack(24.5, (0.0, 0.0, 50.0), concave=False) for _ in range(4)]
    findings = check_stated_ods(_runner_brief(), stacks)
    assert not any(f.gates for f in findings)


def test_no_convex_geometry_is_a_loud_advisory() -> None:
    """The tapered-strut evasion: declared tubes, zero measurable cylinders."""
    stacks = [_stack(19.0, (17.0, 0.0, -2.0), concave=True)]
    findings = check_stated_ods(_runner_brief(), stacks)
    assert len(findings) == 1
    assert findings[0].check == "stated_od_unmeasured" and not findings[0].gates


def test_features_without_stated_od_contribute_nothing() -> None:
    b = _brief(
        [BriefFeature(name="plate", purpose="m", approx_dims_mm="150 x 45 x 8", method="box")]
    )
    assert check_stated_ods(b, [_stack(40.0, (0, 0, 0), concave=False)]) == []
