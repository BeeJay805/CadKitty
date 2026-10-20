"""The stand-or-tip wiring in server.app: _stability_view's dict shape + Z-up frame, and that
_mesh_metrics carries a stability block only when given the Z-up STL mesh. The physics itself
(margins, refusal on no flat base) is covered by tests/unit/physics/test_stability.py."""

from __future__ import annotations

import trimesh

from forge_mind.server.app import _mesh_metrics, _stability_view


def _zup_box() -> trimesh.Trimesh:
    """A 20x20x10 box resting on z in [0, 10], centred in xy. Z-up, watertight: CoM sits over the
    base centre, so it stands with a healthy margin."""
    m = trimesh.creation.box(extents=(20.0, 20.0, 10.0))
    m.apply_translation((0.0, 0.0, 5.0))
    return m


def test_stability_view_shape_and_zup_frame() -> None:
    view = _stability_view(_zup_box())
    assert view is not None
    assert view["stable"] is True
    assert view["margin_mm"] > 0
    # com_mm is Z-up: centred in xy, half-height in z.
    assert abs(view["com_mm"]["z"] - 5.0) < 0.5
    assert abs(view["base_z_mm"] - 0.0) < 0.5
    assert view["com_height_mm"] > 0
    footprint = view["footprint_mm"]
    assert len(footprint) >= 3
    assert all(len(pt) == 2 for pt in footprint)  # 2D xy at base_z


def test_mesh_metrics_includes_stability_with_stl() -> None:
    box = _zup_box()
    metrics = _mesh_metrics(box, "PLA", stl_mesh=box)
    assert metrics is not None
    assert metrics["stability"] is not None
    assert metrics["stability"]["stable"] is True


def test_mesh_metrics_stability_null_without_stl() -> None:
    metrics = _mesh_metrics(_zup_box(), "PLA")
    assert metrics is not None
    assert metrics["stability"] is None
