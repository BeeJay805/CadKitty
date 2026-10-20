"""Feature-at-station geometry check (Sol P0.3 / task 3): is a declared bore realized in
the built geometry at its declared station and size? Hand-authored CylinderStack fixtures
(the cad-validator-designer discipline: known-bad must FAIL, known-good must PASS, and an
unmeasurable input is not_run). The pure core runs with no kernel/model.

Fixtures mirror the real H3 hip interface: a Ø14.3 bore, axis +Z through (0, 26, 167)."""

from __future__ import annotations

from forge_mind.validator.feature_geom import bore_at_station, bore_at_station_in_stacks
from forge_mind.validator.geometry import CylinderStack


def _hole(dia: float, axis_point, axis_dir=(0.0, 0.0, 1.0), span=6.0) -> CylinderStack:
    # A realistic bore straddles the station along its axis: axial_extent centered on the
    # station's axial projection so the station falls inside the measured span.
    proj = axis_point[0] * axis_dir[0] + axis_point[1] * axis_dir[1] + axis_point[2] * axis_dir[2]
    return CylinderStack(
        radius=dia / 2.0,
        axis_point=axis_point,
        axis_dir=axis_dir,
        concave=True,
        axial_extent=(proj - span, proj + span),
        n_faces=1,
    )


def _boss(dia: float, axis_point, axis_dir=(0.0, 0.0, 1.0)) -> CylinderStack:
    s = _hole(dia, axis_point, axis_dir)
    return CylinderStack(
        radius=s.radius,
        axis_point=s.axis_point,
        axis_dir=s.axis_dir,
        concave=False,  # a boss/pin, not a hole
        axial_extent=s.axial_extent,
        n_faces=1,
    )


def test_declared_bore_realized_passes() -> None:
    stacks = [_hole(14.3, (0.0, 26.0, 167.0))]
    v = bore_at_station_in_stacks(
        stacks, diameter_mm=14.3, axis_point=(0.0, 26.0, 167.0), axis_dir=(0.0, 0.0, 1.0)
    )
    assert v.status == "pass", v.message
    assert v.measured["radial_offset_mm"] < 1.0
    assert abs(v.measured["measured_dia_mm"] - 14.3) < 0.4


def test_bore_present_but_off_station_fails() -> None:
    # A Ø14.3 bore exists, but on the wrong axis line (14 mm off in Y) -- present, misplaced.
    stacks = [_hole(14.3, (0.0, 40.0, 167.0))]
    v = bore_at_station_in_stacks(
        stacks, diameter_mm=14.3, axis_point=(0.0, 26.0, 167.0), axis_dir=(0.0, 0.0, 1.0)
    )
    assert v.status == "fail", v.message
    assert v.measured["radial_offset_mm"] > 1.0
    assert v.measured["dia_err_mm"] < 0.4  # right diameter, wrong place


def test_bore_on_axis_but_wrong_axial_position_fails() -> None:
    # Sol P0.3 false-PASS catch: a Ø14.3 bore on the SAME +Z axis line through (0,26,*) but
    # physically at z=300..310, while the declared station is z=167. Radial offset is 0 (same
    # infinite line) -- the OLD check PASSED this. It must FAIL: the station is outside the span.
    s = CylinderStack(
        radius=14.3 / 2.0,
        axis_point=(0.0, 26.0, 305.0),
        axis_dir=(0.0, 0.0, 1.0),
        concave=True,
        axial_extent=(300.0, 310.0),
        n_faces=1,
    )
    v = bore_at_station_in_stacks(
        [s], diameter_mm=14.3, axis_point=(0.0, 26.0, 167.0), axis_dir=(0.0, 0.0, 1.0)
    )
    assert v.status == "fail", v.message
    assert "axial span" in v.message


def test_bore_with_unmeasurable_axial_span_is_not_run() -> None:
    # A diameter/axis/radial match whose axial span could not be measured cannot confirm the
    # station is inside the bore -> fail closed not_run, never a silent PASS.
    s = CylinderStack(
        radius=14.3 / 2.0,
        axis_point=(0.0, 26.0, 167.0),
        axis_dir=(0.0, 0.0, 1.0),
        concave=True,
        axial_extent=None,
        n_faces=1,
    )
    v = bore_at_station_in_stacks(
        [s], diameter_mm=14.3, axis_point=(0.0, 26.0, 167.0), axis_dir=(0.0, 0.0, 1.0)
    )
    assert v.status == "not_run", v.message


def test_wrong_diameter_bore_fails_naming_nearest() -> None:
    stacks = [_hole(6.2, (0.0, 26.0, 167.0))]  # only a small bore where a Ø14.3 was declared
    v = bore_at_station_in_stacks(
        stacks, diameter_mm=14.3, axis_point=(0.0, 26.0, 167.0), axis_dir=(0.0, 0.0, 1.0)
    )
    assert v.status == "fail", v.message
    assert abs(v.measured["nearest_dia_mm"] - 6.2) < 0.1


def test_no_hole_fails() -> None:
    stacks = [_boss(14.3, (0.0, 26.0, 167.0))]  # a boss, not a bore
    v = bore_at_station_in_stacks(
        stacks, diameter_mm=14.3, axis_point=(0.0, 26.0, 167.0), axis_dir=(0.0, 0.0, 1.0)
    )
    assert v.status == "fail" and v.measured["n_holes"] == 0.0


def test_wrong_axis_direction_reports_no_parallel_bore() -> None:
    stacks = [_hole(14.3, (0.0, 26.0, 167.0), axis_dir=(1.0, 0.0, 0.0))]  # bore runs +X
    v = bore_at_station_in_stacks(
        stacks, diameter_mm=14.3, axis_point=(0.0, 26.0, 167.0), axis_dir=(0.0, 0.0, 1.0)
    )
    assert v.status == "fail" and "no bore parallel" in v.message


def test_part_extraction_failure_is_not_run_never_pass() -> None:
    # A non-part object makes extract_cylinders raise -> fail-closed not_run, never PASS.
    v = bore_at_station(
        object(), diameter_mm=14.3, axis_point=(0.0, 26.0, 167.0), axis_dir=(0.0, 0.0, 1.0)
    )
    assert v.status == "not_run"


def test_degenerate_declared_axis_is_not_run() -> None:
    # A zero declared axis reads as "parallel to nothing" -> would spuriously FAIL; the malformed
    # query must be not_run instead (cross-examiner: no defensible-looking verdict on bad input).
    stacks = [_hole(14.3, (0.0, 26.0, 167.0))]
    v = bore_at_station_in_stacks(
        stacks, diameter_mm=14.3, axis_point=(0.0, 26.0, 167.0), axis_dir=(0.0, 0.0, 0.0)
    )
    assert v.status == "not_run" and "axis" in v.message


def test_nonpositive_diameter_is_not_run() -> None:
    stacks = [_hole(14.3, (0.0, 26.0, 167.0))]
    v = bore_at_station_in_stacks(
        stacks, diameter_mm=0.0, axis_point=(0.0, 26.0, 167.0), axis_dir=(0.0, 0.0, 1.0)
    )
    assert v.status == "not_run" and "diameter" in v.message
