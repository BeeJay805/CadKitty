"""GLB exports must be Y-up (the glTF convention); the kernel mesh is Z-up (CAD / build123d), so a
part exported as-is renders lying on its back in a glTF viewer. Every GLB export rotates -90 deg about
X first ((x, y, z) -> (x, z, -y)); the STL stays Z-up for slicers."""

from __future__ import annotations

from pathlib import Path

import trimesh

from forge_mind.api import core
from forge_mind.viz.glb import to_gltf_yup


def _tall_z_box() -> trimesh.Trimesh:
    # Distinct extents so each axis is identifiable; tallest on Z (Z-up): X=10, Y=12, Z=30.
    return trimesh.creation.box(extents=[10.0, 12.0, 30.0])


def test_to_gltf_yup_maps_height_from_z_to_y() -> None:
    mesh = _tall_z_box()
    to_gltf_yup(mesh)
    ex, ey, ez = (round(float(v)) for v in mesh.bounding_box.extents)
    # -90 about X: (x, y, z) -> (x, z, -y). The 30 mm height moves Z -> Y; the 12 mm depth Y -> Z.
    assert (ex, ey, ez) == (10, 30, 12)


def test_stl_to_glb_writes_a_y_up_glb(tmp_path: Path) -> None:
    stl = tmp_path / "p.stl"
    _tall_z_box().export(str(stl))  # the STL is Z-up (tall on Z)
    glb = tmp_path / "p.glb"
    assert core._stl_to_glb(stl, glb) is True

    reloaded = trimesh.load(str(glb), force="mesh")
    _ex, ey, ez = (round(float(v)) for v in reloaded.bounding_box.extents)
    # The exported GLB is Y-up: the 30 mm height is now on Y, not Z.
    assert ey == 30
    assert ez == 12
    # The STL it was built from is untouched (still Z-up: tall on Z).
    assert round(float(trimesh.load(str(stl), force="mesh").bounding_box.extents[2])) == 30
