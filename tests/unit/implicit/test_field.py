"""SDF primitives, boolean ops, and the marching-cubes mesher (ADR-0026)."""

from __future__ import annotations

import pytest

from forge_mind.implicit import field as sdf


def test_sphere_is_watertight() -> None:
    m = sdf.sdf_to_mesh(sdf.sphere(8.0), (-10, -10, -10), (10, 10, 10), 0.5)
    assert m.is_watertight
    assert m.volume > 0


def test_box_volume_matches() -> None:
    # box half-extents 5 -> a 10mm cube -> ~1000 mm^3 (flat faces mesh closely).
    m = sdf.sdf_to_mesh(sdf.box(5.0, 5.0, 5.0), (-7, -7, -7), (7, 7, 7), 0.4)
    assert m.is_watertight
    assert 900 < m.volume < 1100


def test_difference_is_watertight() -> None:
    diff = sdf.op_difference(sdf.sphere(8.0), sdf.box(5.0, 5.0, 12.0))
    m = sdf.sdf_to_mesh(diff, (-10, -10, -10), (10, 10, 10), 0.4)
    assert m.is_watertight
    assert m.volume > 0


def test_gyroid_walled_box_is_watertight() -> None:
    fld = sdf.op_intersect(sdf.gyroid(8.0, 1.4), sdf.box(10.0, 10.0, 10.0))
    m = sdf.sdf_to_mesh(fld, (-12, -12, -12), (12, 12, 12), 0.3)
    assert m.is_watertight
    assert m.volume > 0


def test_strut_lattice_is_watertight() -> None:
    fld = sdf.op_intersect(sdf.strut_lattice(8.0, 2.0), sdf.box(10.0, 10.0, 10.0))
    m = sdf.sdf_to_mesh(fld, (-12, -12, -12), (12, 12, 12), 0.3)
    assert m.is_watertight
    assert m.volume > 0


def test_sample_guard_rejects_runaway_grid() -> None:
    with pytest.raises(ValueError, match="too fine"):
        sdf.sdf_to_mesh(sdf.sphere(1.0), (-100, -100, -100), (100, 100, 100), 0.1)


def test_mesh_is_in_world_coordinates() -> None:
    # A sphere of radius 8 centred at the origin: bbox extents ~16mm, centred near 0.
    m = sdf.sdf_to_mesh(sdf.sphere(8.0), (-10, -10, -10), (10, 10, 10), 0.4)
    ext = m.extents
    assert all(14.0 < e < 18.0 for e in ext)
    assert all(abs(c) < 1.0 for c in m.bounds.mean(axis=0))
