"""Environment-fit checks (ADR-0087 S5), mutation-style: the correct part passes, the seeded
defect FLAGS, and missing geometry/parameters flag fail-closed. All advisory (P13) — asserted by
the result type itself (pass/flag only, no gate)."""

from __future__ import annotations

import trimesh
from build123d import Box, Cylinder, Pos

from forge_mind.invention.compile_requirements import CheckRequest
from forge_mind.invention.env_checks import run_env_checks

_ENVELOPE = CheckRequest(
    check_id="envelope_fits",
    params={"available_length": 110.0, "available_width": 80.0, "available_height": 25.0},
    domain="fit_envelope",
    description="fits the bay",
)
_KEEPOUT = CheckRequest(
    check_id="keepout_clear",
    params={"fan_frame_size": 120.0, "airflow_clearance": 15.0},
    domain="thermal_airflow",
    description="airflow clearance",
)
_SECTION = CheckRequest(
    check_id="path_cross_section",
    params={"fan_frame_size": 120.0},
    domain="thermal_airflow",
    description="flow cross-section",
)
_PATTERN = CheckRequest(
    check_id="interface_pattern",
    params={"mount_hole_pitch": 76.2},
    domain="mount_interface",
    description="drive hole pattern",
)


def _bracket_mesh(h: float = 10.0) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(100.0, 70.0, h))


def test_envelope_fits_sorted_not_posed() -> None:
    # built 70 x 100 x 10 (axes swapped vs stated 110/80/25): sorted compare must PASS (v1.1 lesson)
    mesh = trimesh.creation.box(extents=(70.0, 100.0, 10.0))
    (res,) = run_env_checks([_ENVELOPE], mesh)
    assert res.status == "pass"


def test_envelope_flags_the_oversize_defect() -> None:
    (res,) = run_env_checks([_ENVELOPE], trimesh.creation.box(extents=(130.0, 70.0, 10.0)))
    assert res.status == "flag" and "exceeds" in res.explanation


def test_keepout_margin_convention_stated_and_enforced() -> None:
    (ok,) = run_env_checks([_KEEPOUT], _bracket_mesh(h=10.0))  # 10 <= 120-15
    assert ok.status == "pass" and "convention:" in ok.measured
    (bad,) = run_env_checks([_KEEPOUT], trimesh.creation.box(extents=(150.0, 140.0, 110.0)))
    assert bad.status == "flag"  # thinnest extent 110 > 105 budget


def test_path_cross_section_floor() -> None:
    (ok,) = run_env_checks([_SECTION], _bracket_mesh(h=10.0))
    # silhouette 100x70=7000 vs 14400 -> free 0.51 < 0.6 floor? 1-0.486=0.514 -> FLAG. Use smaller part:
    small = trimesh.creation.box(extents=(60.0, 60.0, 10.0))  # 3600/14400 -> free 0.75
    (ok2,) = run_env_checks([_SECTION], small)
    assert ok2.status == "pass" and "no flow simulated" in ok2.measured
    assert ok.status == "flag"  # the wide bracket chokes the 120mm path below the floor


def test_interface_pattern_measured_on_the_brep() -> None:
    plate = Box(100, 30, 5)
    hole = Cylinder(radius=1.7, height=20)
    part = plate - Pos(-38.1, 0, 0) * hole - Pos(38.1, 0, 0) * hole  # pitch exactly 76.2
    (ok,) = run_env_checks([_PATTERN], None, brep_part=part)
    assert ok.status == "pass" and "76.2" in ok.measured

    off = plate - Pos(-38.1, 0, 0) * hole - Pos(30.0, 0, 0) * hole  # pitch 68.1: seeded defect
    (bad,) = run_env_checks([_PATTERN], None, brep_part=off)
    assert bad.status == "flag" and "no hole pair matches" in bad.explanation


def test_fail_closed_on_missing_geometry_params_and_unknown_ids() -> None:
    (no_mesh,) = run_env_checks([_ENVELOPE], None)
    assert no_mesh.status == "flag" and "could not verify" in no_mesh.explanation
    incomplete = CheckRequest(
        check_id="envelope_fits", params={}, domain="fit_envelope", description=""
    )
    (missing,) = run_env_checks([incomplete], _bracket_mesh())
    assert missing.status == "flag"
    (no_brep,) = run_env_checks([_PATTERN], _bracket_mesh(), brep_part=None)
    assert no_brep.status == "flag" and "could not be measured" in no_brep.explanation
    drift = CheckRequest(check_id="envelope_fits", params={}, domain="x", description="")
    rogue = drift.model_copy(update={"check_id": "cfd_sim"})
    (unknown,) = run_env_checks([rogue], _bracket_mesh())
    assert unknown.status == "flag" and "vocabulary drift" in unknown.explanation
