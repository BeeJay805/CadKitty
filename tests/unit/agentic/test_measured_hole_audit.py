"""Size-aware measured-hole audit: plan diameters vs measured B-rep bores (money-free).

The genus gate is COUNT-only; these pin the size half: target diameters resolved from
the standards registry (M4 -> Ø4.5 clearance), missing-by-size named with the planned
label, and right-count/wrong-size holes called out instead of waved through."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from build123d import Box, Cylinder, Pos

from forge_mind.agentic.spatial import (
    _planned_hole_specs,
    measured_hole_conformance_issues,
    measured_hole_feedback,
    measured_holes_short,
    mesh_genus,
)
from forge_mind.materials import resolve_material

pytestmark = pytest.mark.slow  # builds tiny B-reps (build123d/OCCT)


def _plate_with_holes(hole_d_mm: float, n: int):
    """A 60x60x5 plate with n through-holes of the given diameter."""
    part = Box(60, 60, 5)
    spots = [(-20, -20), (-20, 20), (20, -20), (20, 20)][:n]
    for x, y in spots:
        part -= Pos(x, y, 0) * Cylinder(hole_d_mm / 2.0, 10)
    return part


def _brief(count: int = 4, standard: str | None = "M4", dims: str = "M4 clearance"):
    return SimpleNamespace(
        features=[
            SimpleNamespace(
                name="mounting holes",
                approx_dims_mm=dims,
                count=count,
                standard=standard,
            )
        ]
    )


def test_planned_specs_resolve_metric_from_the_registry() -> None:
    specs = _planned_hole_specs(_brief())
    assert len(specs) == 1
    assert specs[0].count == 4
    assert specs[0].target_d_mm == pytest.approx(4.5, abs=0.2)  # ISO 273 M4 clearance
    assert "ISO 273" in specs[0].source


def test_planned_specs_parse_stated_diameter() -> None:
    specs = _planned_hole_specs(_brief(count=1, standard=None, dims="Ø6 mm bore, through"))
    assert specs[0].target_d_mm == pytest.approx(6.0)
    specs = _planned_hole_specs(_brief(count=2, standard=None, dims="about 30 long"))
    assert specs[0].target_d_mm is None  # unstated size stays None, never guessed


def test_missing_by_size_names_target_and_label() -> None:
    part = _plate_with_holes(4.5, n=2)  # 2 of 4 drilled at the right size
    lines = measured_hole_feedback(part, _brief())
    assert lines, "expected a deficit line"
    assert "MISSING 2 x Ø4.5" in lines[0]
    assert "mounting holes" in lines[0]
    assert "Ø4.50mm at" in lines[0]  # the measured ones are listed with positions


def test_right_count_wrong_size_is_named_not_waved_through() -> None:
    part = _plate_with_holes(2.0, n=4)  # the genus COUNT is satisfied; every size is wrong
    lines = measured_hole_feedback(part, _brief())
    assert any("MISSING 4 x Ø4.5" in ln for ln in lines)
    assert any("SIZE MISMATCH" in ln and "Ø2.00" in ln for ln in lines)


def test_quiet_when_plan_is_satisfied() -> None:
    part = _plate_with_holes(4.5, n=4)
    assert measured_hole_feedback(part, _brief()) == []


def test_assignment_window_does_not_certify_an_undersized_m5_bore() -> None:
    pla = resolve_material("PLA")
    brief = _brief(count=4, standard="M5", dims="M5 clearance")
    part = _plate_with_holes(5.0, n=4)
    # 5.0 is within the legacy 0.6 mm association window of ISO target 5.5,
    # but below the material-derived 5.3 mm fit floor.
    assert measured_holes_short(part, brief) is False
    issues = measured_hole_conformance_issues(part, brief, pla)
    assert issues and "needs at least 5.30 mm" in issues[0]


def test_iso_m5_clearance_bores_pass_the_separate_fit_gate() -> None:
    pla = resolve_material("PLA")
    brief = _brief(count=4, standard="M5", dims="M5 clearance")
    assert measured_hole_conformance_issues(_plate_with_holes(5.5, n=4), brief, pla) == []


def test_mesh_genus_counts_through_holes() -> None:
    import tempfile
    from pathlib import Path

    import trimesh
    from build123d import export_stl

    part = _plate_with_holes(4.5, n=3)
    with tempfile.TemporaryDirectory() as d:
        stl = Path(d) / "p.stl"
        export_stl(part, str(stl))
        mesh = trimesh.load(str(stl), force="mesh")
        assert mesh_genus(mesh) == 3


def test_measured_holes_short_gates_on_bore_count_not_genus() -> None:
    # H8 (ADR-0065): the completeness GATE must count measured concave bores, not mesh genus -- a
    # handle/ring inflates genus so a dropped required hole could pass the genus gate. Bore count
    # can't be fooled that way.
    assert measured_holes_short(_plate_with_holes(4.5, n=2), _brief()) is True  # 2 of 4 -> short
    assert measured_holes_short(_plate_with_holes(4.5, n=4), _brief()) is False  # complete
    # fail open: nothing planned -> never short (no false FAIL on unplanned/unmeasurable geometry)
    assert measured_holes_short(_plate_with_holes(4.5, n=4), SimpleNamespace(features=[])) is False


def _plate_with_mixed_holes(holes: list[tuple[float, float, float]]):
    """A 100x100x5 plate with a through-hole of diameter d at each (d, x, y)."""
    part = Box(100, 100, 5)
    for d, x, y in holes:
        part -= Pos(x, y, 0) * Cylinder(d / 2.0, 10)
    return part


def test_short_when_extras_mask_a_missing_sized_spec() -> None:
    """TEST_LOG row 38 (nr200_hdd35 golden): the tray shipped with its 4 planned Ø3.4 rail
    attachment holes NEVER cut because 16 unplanned Ø7 vent bores satisfied the TOTAL bore
    count. Shortness must reconcile per-SPEC by diameter, not compare totals."""
    # right TOTAL count, but every measured size misses the planned Ø4.5 -> short
    assert measured_holes_short(_plate_with_holes(7.0, n=4), _brief()) is True
    # 2 of 4 planned Ø4.5 cut; 2 off-plan Ø7 extras keep the total at 4 -> still short
    part = _plate_with_mixed_holes([(4.5, -30, -30), (4.5, -30, 30), (7.0, 30, -30), (7.0, 30, 30)])
    assert measured_holes_short(part, _brief()) is True
    # a SATISFIED plan with extra off-plan bores is NOT short (extras alone never gate)
    part = _plate_with_mixed_holes(
        [(4.5, -30, -30), (4.5, -30, 30), (4.5, 30, -30), (4.5, 30, 30), (7.0, 0, 0)]
    )
    assert measured_holes_short(part, _brief()) is False


def _brief_two_same_size(count_a: int = 2, count_b: int = 2):
    """Two hole specs stating the SAME diameter (the hdd35 golden's case-mount +
    side-mount pair, both Ø3.4)."""
    return SimpleNamespace(
        features=[
            SimpleNamespace(
                name="case mount hole",
                approx_dims_mm="3.4 mm diameter, 76.2 mm center-to-center",
                count=count_a,
                standard="M3",
            ),
            SimpleNamespace(
                name="side mount hole",
                approx_dims_mm="3.4 mm diameter, 30 mm pitch",
                count=count_b,
                standard="M3",
            ),
        ]
    )


def test_same_diameter_specs_share_the_pool() -> None:
    """hdd35 golden re-run (2026-07-07): with two Ø3.4 specs, 'nearest spec wins' broke the
    tie by list order -- the FIRST spec won every stack, the second could never claim a bore,
    and a COMPLETE part reported its full count phantom-missing for six iterations. Ties must
    share the pool: earlier specs consume first, later same-size specs take the leftovers."""
    # 4 measured Ø3.4 bores satisfy 2 + 2 planned across two same-size specs -> complete
    part = _plate_with_mixed_holes([(3.4, -30, -30), (3.4, -30, 30), (3.4, 30, -30), (3.4, 30, 30)])
    assert measured_hole_feedback(part, _brief_two_same_size()) == []
    assert measured_holes_short(part, _brief_two_same_size()) is False
    # a REAL deficit across the tie group still reports and gates: only 2 of 2+2 cut
    part = _plate_with_mixed_holes([(3.4, -30, -30), (3.4, -30, 30)])
    lines = measured_hole_feedback(part, _brief_two_same_size())
    assert any("MISSING 2 x Ø3.40" in ln for ln in lines)
    assert measured_holes_short(part, _brief_two_same_size()) is True


def test_stated_diameter_outranks_the_fastener_standard() -> None:
    """Bench row 17, ratchet pawl: the prompt stated 'pivot bore 6 mm diameter', the brief
    tagged it M6, and the registry conversion silently redrilled it to the 6.6 clearance.
    A STATED number is the user's dimension -- it must win over the standard tag."""
    specs = _planned_hole_specs(
        _brief(count=1, standard="M6", dims="6 mm diameter through-hole, center 10 mm from end")
    )
    assert specs[0].target_d_mm == pytest.approx(6.0)
    assert specs[0].source == "stated in the brief"
    # the registry still resolves when NO diameter is stated
    specs = _planned_hole_specs(_brief(count=1, standard="M6", dims="clearance for bolt"))
    assert specs[0].target_d_mm == pytest.approx(6.6, abs=0.2)
    assert "ISO 273" in specs[0].source


def test_threaded_hole_targets_nominal_not_clearance() -> None:
    """ADR-0081 (bench row 18): a threaded / tapped / set-screw hole is identified by its
    NOMINAL major diameter, not the clearance (M8->8.0, not 8.4) and not the tap drill."""
    specs = _planned_hole_specs(
        _brief(count=1, standard="M8", dims="M8 threaded bore through the grip")
    )
    assert specs[0].target_d_mm == pytest.approx(8.0)
    assert "nominal" in specs[0].source
    # a set-screw reads as threaded too
    specs = _planned_hole_specs(
        _brief(count=2, standard="M4", dims="M4 set-screw hole on the hub OD")
    )
    assert specs[0].target_d_mm == pytest.approx(4.0)
    # a plain bolt CLEARANCE hole is unchanged -> ISO 273 clearance
    specs = _planned_hole_specs(_brief(count=4, standard="M4", dims="M4 clearance for mounting"))
    assert specs[0].target_d_mm == pytest.approx(4.5, abs=0.2)
    assert "clearance" in specs[0].source
