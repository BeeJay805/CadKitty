"""ADR-0012 E1: the stated-dim owning check + the area-significant wall gate (money-free).

Row-40 evidence pinned here: 12-15% extent drifts off stated dims passed the 20% envelope
gate (91% false-PASS, n=49), and sub-floor WALLS escaped because the point-min wall check is
advisory (a knife edge legitimately measures thin at a point). The stated gate owns the band
under 20%; the wall gate reads the thin AREA fraction, so knife edges stay advisory-only.
"""

from __future__ import annotations

import pytest
import trimesh

from forge_mind.agentic.brief import DesignBrief
from forge_mind.agentic.validate import (
    _envelope_mismatch,
    _stated_extent_mismatch,
    _thin_wall_gate,
)
from forge_mind.types import BBox, Vec3

pytestmark = pytest.mark.slow  # tiny meshes + ray casts


def _brief(envelope: str) -> DesignBrief:
    return DesignBrief.model_validate(
        {
            "object_name": "plate",
            "summary": "s",
            "visual_description": "v",
            "overall_envelope_mm": envelope,
            "features": [
                {
                    "name": "base plate",
                    "purpose": "body",
                    "approx_dims_mm": "60 x 40 x 5 mm",
                    "method": "box",
                    "count": 1,
                }
            ],
            "min_wall_mm": 2.0,
            "print_orientation_note": "flat",
        }
    )


def _bbox(x: float, y: float, z: float) -> BBox:
    return BBox(min=Vec3(x=0.0, y=0.0, z=0.0), max=Vec3(x=x, y=y, z=z))


_STATED = frozenset({60.0, 40.0, 5.0})  # the user's numbers (as stated_numbers would yield)


def test_stated_gate_owns_the_band_under_the_envelope_gate() -> None:
    brief = _brief("60 x 40 x 5 mm")
    # exact build passes
    ok, issue = _stated_extent_mismatch(_bbox(60, 40, 5), brief, _STATED)
    assert ok is False and issue is None
    # the audited mutant shape: +12% on one stated dim (67.2 vs 60) -- inside the 20%
    # envelope band, so it previously PASSED; the owning check fails it
    bad, issue = _stated_extent_mismatch(_bbox(67.2, 40, 5), brief, _STATED)
    assert bad is True
    assert issue is not None and "STATED DIM" in issue and "67.2" in issue
    # -15% (the other mutant multiplier) fails too
    bad, _ = _stated_extent_mismatch(_bbox(51, 40, 5), brief, _STATED)
    assert bad is True


def test_stated_gate_tolerates_build_noise_and_small_dims() -> None:
    brief = _brief("60 x 40 x 5 mm")
    # sub-1mm drift (the row-28 known-good noise class) never fires
    ok, _ = _stated_extent_mismatch(_bbox(60.8, 40.3, 5.0), brief, _STATED)
    assert ok is False
    # on a small stated dim the 1.0mm absolute floor governs: 5 -> 5.9 is 18% but <1mm
    ok, _ = _stated_extent_mismatch(_bbox(60, 40, 5.9), brief, _STATED)
    assert ok is False


def test_stated_gate_protects_only_user_numbers() -> None:
    """The first audit false-FAIL, pinned: clip_tube_saddle_clamp's 46mm envelope height was
    the MODEL's own stack estimate (not in the prompt) and the accepted build measured 40.1
    (-13%). A model estimate is not protected currency -- the tight band applies only to
    envelope numbers in the user's stated set; the 15% envelope gate bounds the rest."""
    brief = _brief("60 x 20 x 46 mm")
    stated = frozenset({60.0, 20.0, 22.0, 15.0})  # the user's numbers; 46 is NOT among them
    ok, issue = _stated_extent_mismatch(_bbox(60, 20, 40.1), brief, stated)
    assert ok is False and issue is None
    # but a USER-stated axis drifting the same way still fails: 60 -> 52.2 (-13%)
    bad, _ = _stated_extent_mismatch(_bbox(52.2, 20, 46), brief, stated)
    assert bad is True
    # and with NO user numbers at all the check is fully off
    ok, issue = _stated_extent_mismatch(_bbox(40, 15, 30), brief, frozenset())
    assert ok is False and issue is None


def test_model_derived_envelope_rejects_the_audited_eighteen_percent_drift() -> None:
    brief = _brief("100 x 40 x 5 mm")
    mismatch, issue = _envelope_mismatch(_bbox(118, 40, 5), brief)
    assert mismatch is True
    assert issue is not None and "+18%" in issue


def test_stated_gate_fails_open_without_a_parseable_envelope() -> None:
    ok, issue = _stated_extent_mismatch(_bbox(60, 40, 5), _brief("roughly palm sized"), _STATED)
    assert ok is False and issue is None


def test_wall_gate_fires_on_an_area_thin_wall_not_a_knife_edge() -> None:
    # a whole 1.0mm plate vs a 2.0mm floor: nearly every sample is thin -> gate
    plate = trimesh.creation.box(extents=(40.0, 40.0, 1.0))
    gated, issue = _thin_wall_gate(plate, 2.0)
    assert gated is True
    assert issue is not None and "WALL GATE" in issue
    # a 6mm-thick block: nothing thin -> no gate
    block = trimesh.creation.box(extents=(40.0, 40.0, 6.0))
    gated, issue = _thin_wall_gate(block, 2.0)
    assert gated is False and issue is None
    # the knife-EDGE class the advisory over-flags (ADR-0029): a body tapering to a point.
    # A cone's apex measures thin at a POINT (the advisory min flags it) but the thin AREA
    # is ~0.2% of the surface -> the gate stays quiet. This is the false-FAIL class the
    # area-fraction discriminator exists to protect.
    cone = trimesh.creation.cone(radius=20.0, height=20.0)
    gated, _ = _thin_wall_gate(cone, 2.0)
    assert gated is False


def test_wall_gate_fails_open_without_a_floor() -> None:
    plate = trimesh.creation.box(extents=(40.0, 40.0, 1.0))
    assert _thin_wall_gate(plate, None) == (False, None)
    assert _thin_wall_gate(plate, 0.0) == (False, None)


def test_wall_gate_sees_a_quarter_mm_membrane_below_the_old_noise_band() -> None:
    membrane = trimesh.creation.box(extents=(40.0, 40.0, 0.25))
    gated, issue = _thin_wall_gate(membrane, 1.0, 0.8**2)
    assert gated is True
    assert issue is not None and "0.25 mm" in issue


def test_absolute_area_catches_a_small_load_bearing_wall_below_three_percent() -> None:
    from forge_mind.validator.feature_checks import WallMeasurement, wall_gate_triggered

    measured = WallMeasurement(0.02, 0.8, 12.0, 20, 1000)
    assert measured.thin_fraction < 0.03
    assert wall_gate_triggered(
        measured.thin_fraction,
        measured.thin_area_mm2,
        measured.thin_sample_count,
        0.8**2,
    )
