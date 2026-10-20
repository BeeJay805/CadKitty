"""Shape signature: pose/scale-invariant descriptors that capture what the D2 score misses
(axial asymmetry + concavity), report-only. Hand-verified on primitives, and the value-prop
test proves it separates a solid disc from a tube where D2 saturates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import trimesh

from forge_mind.agentic.shape_match import shape_similarity
from forge_mind.agentic.shape_signature import (
    conformance_advisory,
    signature_deltas,
    signature_of,
)


def _save(tmp: Path, name: str, mesh: Any) -> Path:
    p = tmp / name
    mesh.export(str(p))
    return p


def test_box_is_isotropic_and_solid(tmp_path: Path) -> None:
    sig = signature_of(_save(tmp_path, "box.stl", trimesh.creation.box(extents=[10, 10, 10])))
    assert sig is not None and sig.axes is not None
    # a cube's surface points spread equally on all three axes -> ~(1/3, 1/3, 1/3)
    for e in sig.axes:
        assert abs(e - 1 / 3) < 0.06
    assert sig.solidity is not None and sig.solidity > 0.98  # a box IS its own convex hull


def test_rod_is_elongated(tmp_path: Path) -> None:
    sig = signature_of(_save(tmp_path, "rod.stl", trimesh.creation.box(extents=[40, 3, 3])))
    assert sig is not None and sig.axes is not None
    assert sig.axes[0] > 0.8  # nearly all spread on the long axis
    assert sig.axes[2] < 0.12


def test_cylinder_is_axisymmetric_but_box_is_not(tmp_path: Path) -> None:
    cyl = signature_of(_save(tmp_path, "cyl.stl", trimesh.creation.cylinder(radius=5, height=30)))
    rect = signature_of(_save(tmp_path, "rect.stl", trimesh.creation.box(extents=[30, 10, 4])))
    assert cyl is not None and rect is not None
    assert cyl.asymmetry is not None and rect.asymmetry is not None
    # a cylinder's two transverse axes spread equally (axisymmetric); a rectangular bar's do not
    assert cyl.asymmetry < 0.03
    assert rect.asymmetry > 0.05


def test_solidity_captures_concavity(tmp_path: Path) -> None:
    # an annulus (tube) is watertight and NON-convex: solidity = 1 - (r_in/r_out)^2 = 1 - 0.25 = 0.75
    tube = trimesh.creation.annulus(r_min=5.0, r_max=10.0, height=8.0)
    sig = signature_of(_save(tmp_path, "tube.stl", tube))
    assert sig is not None and sig.solidity is not None
    assert 0.70 < sig.solidity < 0.80  # hand value 0.75


def test_signature_separates_disc_from_tube_where_d2_saturates(tmp_path: Path) -> None:
    """The value prop: a solid disc and a tube of the same envelope read as ~the same form to D2
    (it captures proportion, not concavity), but the signature's solidity cleanly separates them."""
    disc = _save(tmp_path, "disc.stl", trimesh.creation.cylinder(radius=10, height=8))
    tube = _save(tmp_path, "tube.stl", trimesh.creation.annulus(r_min=5.0, r_max=10.0, height=8.0))
    d2 = shape_similarity(disc, tube)
    assert d2 is not None and d2 > 0.80  # D2 says "close" -- the saturation we measured
    deltas = signature_deltas(disc, tube)
    assert (
        deltas is not None and deltas.get("solidity_delta", 0.0) > 0.18
    )  # signature separates them


def test_identical_mesh_has_near_zero_deltas(tmp_path: Path) -> None:
    m = _save(tmp_path, "a.stl", trimesh.creation.box(extents=[12, 6, 6]))
    deltas = signature_deltas(m, m)
    assert deltas is not None
    for v in deltas.values():
        assert v < 1e-6


def test_concavity_zero_for_convex_positive_for_concave(tmp_path: Path) -> None:
    box = signature_of(_save(tmp_path, "box.stl", trimesh.creation.box(extents=[10, 10, 10])))
    tube = signature_of(_save(tmp_path, "tube.stl", trimesh.creation.annulus(5.0, 10.0, 8.0)))
    assert box is not None and box.concavity is not None and box.concavity < 0.03  # convex -> ~0
    assert tube is not None and tube.concavity is not None and tube.concavity > 0.10  # hollow bore


def test_non_watertight_reference_keeps_axes_and_concavity_not_solidity(tmp_path: Path) -> None:
    """The eval's real situation: a non-watertight artist mesh yields NO exact solidity, but the
    watertight-free axes AND concavity descriptors still compute -- the concavity signal survives."""
    # an open box with a face removed -> not watertight
    box = trimesh.creation.box(extents=[10, 10, 10])
    open_mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2], process=False)
    sig = signature_of(_save(tmp_path, "open.ply", open_mesh))
    assert sig is not None
    assert sig.solidity is None  # not watertight -> no reliable volume
    assert sig.axes is not None  # surface points still give the spread
    assert sig.concavity is not None  # THE FIX: concavity survives a non-watertight reference


def test_conformance_advisory_fires_against_a_non_watertight_reference(tmp_path: Path) -> None:
    """The point of the concavity descriptor: a watertight solid build vs a NON-watertight concave
    reference still yields a CONCAVITY advisory (solidity would be None; concavity carries it)."""
    disc = _save(tmp_path, "disc.stl", trimesh.creation.cylinder(radius=10, height=8))
    tube = trimesh.creation.annulus(r_min=5.0, r_max=10.0, height=8.0)  # concave reference...
    open_tube = trimesh.Trimesh(
        vertices=tube.vertices, faces=tube.faces[:-4], process=False
    )  # ...not watertight
    ref = _save(tmp_path, "open_tube.ply", open_tube)
    assert (
        signature_of(ref) is not None and signature_of(ref).solidity is None
    )  # ref not watertight
    msg = conformance_advisory(disc, ref)
    assert msg is not None and "CONCAVITY" in msg and "too CONVEX" in msg


def test_conformance_advisory_names_the_concavity_direction(tmp_path: Path) -> None:
    # build = solid disc, reference = tube -> the build is too convex, advisory says add concavity
    disc = _save(tmp_path, "disc.stl", trimesh.creation.cylinder(radius=10, height=8))
    tube = _save(tmp_path, "tube.stl", trimesh.creation.annulus(r_min=5.0, r_max=10.0, height=8.0))
    msg = conformance_advisory(disc, tube)
    assert msg is not None and "CONCAVITY" in msg and "too CONVEX" in msg


def test_conformance_advisory_none_when_matched(tmp_path: Path) -> None:
    m = _save(tmp_path, "m.stl", trimesh.creation.cylinder(radius=8, height=20))
    assert conformance_advisory(m, m) is None  # nothing differs -> no note


def test_fail_closed_on_bad_path(tmp_path: Path) -> None:
    assert signature_of(tmp_path / "nope.stl") is None
    assert signature_deltas(tmp_path / "nope.stl", tmp_path / "no.stl") is None
    assert conformance_advisory(tmp_path / "nope.stl", tmp_path / "no.stl") is None
