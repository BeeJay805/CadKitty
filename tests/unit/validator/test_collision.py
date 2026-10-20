"""Validation tests for the two-mesh interference report (forge_mind.validator.collision).

Uses primitive boxes where the overlap volume and clearance are known exactly, so the engine's
numbers are checked against ground truth (a 10 mm cube overlapping 5 mm in x = 500 mm^3).
"""

from __future__ import annotations

import trimesh

from forge_mind.validator.collision import interference_report


def _cube() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))  # centered at origin, watertight


def test_overlap_reports_exact_volume() -> None:
    r = interference_report(_cube(), _cube(), b_offset_mm=(5.0, 0.0, 0.0))
    assert r.collide is True
    assert r.penetration_volume_mm3 == 500.0  # 5 mm * 10 mm * 10 mm overlap
    assert r.method == "boolean+proximity"


def test_full_overlap_volume() -> None:
    r = interference_report(_cube(), _cube())  # identical, coincident
    assert r.collide is True
    assert r.penetration_volume_mm3 == 1000.0  # whole 10^3 cube


def test_disjoint_reports_clearance() -> None:
    r = interference_report(_cube(), _cube(), b_offset_mm=(20.0, 0.0, 0.0))
    assert r.collide is False
    assert r.min_clearance_mm == 10.0  # a:[-5,5], b:[15,25] -> 10 mm gap
    assert r.method == "proximity-only"


def test_touching_is_not_collision() -> None:
    # Faces meet at x=5: contact, not interference -> zero penetration volume.
    r = interference_report(_cube(), _cube(), b_offset_mm=(10.0, 0.0, 0.0))
    assert r.collide is False
    assert r.penetration_volume_mm3 == 0.0


def test_non_solid_mesh_refuses_volume() -> None:
    # An open (non-watertight) mesh has no defined overlap volume -> report None, never a fake number.
    solid = _cube()
    box = _cube()
    open_mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2])
    assert not open_mesh.is_volume
    r = interference_report(solid, open_mesh)
    assert r.penetration_volume_mm3 is None
    assert r.method in ("proximity-only", "refused")
    assert any("not a closed solid" in c for c in r.caveats) or r.method == "refused"


def test_unverified_overlap_is_not_assessed() -> None:
    # Regression (audit M1): bboxes overlap but the boolean volume is undefined (an open mesh), so
    # collision CANNOT be ruled out. collide=False must read as "unverified", not "safe" -- the CLI
    # gates on `assessed` and exits non-zero here (audit M2).
    solid = _cube()
    box = _cube()
    open_mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2])
    r = interference_report(solid, open_mesh)
    assert r.collide is False
    assert r.assessed is False


def test_assessed_true_for_volume_backed_and_disjoint() -> None:
    # A volume-backed verdict and a bbox-disjoint verdict are both trustworthy (assessed).
    assert interference_report(_cube(), _cube(), b_offset_mm=(5.0, 0.0, 0.0)).assessed is True
    assert interference_report(_cube(), _cube(), b_offset_mm=(20.0, 0.0, 0.0)).assessed is True
