"""M3 servo bridge + 3-D SIMP + the 3-D mesh path (the founder's 'build M2 and M3, tackle 3-D').

Bridge: a declared MG996R becomes the PlacedHardware the assembly validators already eat, and a
body<->servo joint SURVIVES the adapter's endpoint filter (it used to be dropped as 'did not
build'). 3-D: the numerically-built hex stiffness is provably a stiffness matrix; small-grid
SIMP behaves; marching the field yields a watertight mesh, fail-closed on empty.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forge_mind.structures import mesh3d, topopt3d
from forge_mind.structures.servo_bridge import (
    HardwareDecl,
    known_hardware_kinds,
    placed_hardware_for,
)

trimesh = pytest.importorskip("trimesh")


# ---- M3: the servo bridge ------------------------------------------------------------------


def test_bridge_builds_mg996r_with_datasheet_ports() -> None:
    hw = placed_hardware_for(HardwareDecl(kind="mg996r_servo", hw_id="shoulder_servo"))
    assert hw.id == "shoulder_servo"
    assert hw.kind == "mg996r_servo"  # the key hardware_mounts.patterns_for matches on
    assert len(hw.ports) >= 2  # datasheet ports (horn + mounts), not invented
    assert hw.transform.shape == (4, 4)


def test_bridge_unknown_kind_fails_closed() -> None:
    with pytest.raises(KeyError):
        placed_hardware_for(HardwareDecl(kind="warp_coil", hw_id="x"))
    assert "mg996r_servo" in known_hardware_kinds()


def test_adapter_keeps_body_to_hardware_joint(tmp_path: Path) -> None:
    """The A7 unlock: a joint from a printed part to a DECLARED servo resolves instead of being
    dropped as 'part did not build', and the model carries the hardware."""
    from forge_mind.agentic.assembly_adapter import from_part_results
    from forge_mind.assembly.joints import Joint, JointType, Port
    from forge_mind.types import Vec3

    stl = tmp_path / "upper_arm.stl"
    trimesh.creation.box(extents=(60, 20, 20)).export(str(stl))
    lr = SimpleNamespace(final_stl=stl, final_step=None)
    bolt = Port(
        name="shoulder-bore",
        origin=Vec3(x=0.0, y=0.0, z=0.0),
        axis=Vec3(x=1.0, y=0.0, z=0.0),
        normal=Vec3(x=0.0, y=0.0, z=1.0),
    )
    hw = placed_hardware_for(HardwareDecl(kind="mg996r_servo", hw_id="shoulder_servo"))
    horn_port = hw.ports[0].name
    joint = Joint(
        type=JointType.REVOLUTE,
        a=f"upper arm.{bolt.name}",
        b=f"shoulder_servo.{horn_port}",
        range_deg=(0.0, 90.0),
    )

    notes: list[str] = []
    model = from_part_results(
        [("upper arm", lr)],  # type: ignore[list-item]
        [joint],
        {"upper arm": [bolt]},
        notes,
        hardware=[hw],
    )
    assert [h.id for h in model.hardware] == ["shoulder_servo"]
    assert len(model.joints) == 1, f"joint dropped: {notes}"
    assert not notes


def test_adapter_without_hardware_unchanged(tmp_path: Path) -> None:
    """No hardware declared -> exactly the old behavior (a joint to an unknown id drops)."""
    from forge_mind.agentic.assembly_adapter import from_part_results
    from forge_mind.assembly.joints import Joint, JointType

    stl = tmp_path / "p.stl"
    trimesh.creation.box(extents=(10, 10, 10)).export(str(stl))
    lr = SimpleNamespace(final_stl=stl, final_step=None)
    joint = Joint(type=JointType.REVOLUTE, a="p.x", b="ghost_servo.horn", range_deg=(0.0, 90.0))
    notes: list[str] = []
    model = from_part_results([("p", lr)], [joint], {"p": []}, notes)  # type: ignore[list-item]
    assert model.hardware == []
    assert model.joints == []
    assert notes  # the drop is explained, not silent


# ---- 3-D SIMP ------------------------------------------------------------------------------


def test_hex_stiffness_is_a_stiffness_matrix() -> None:
    ke = topopt3d.hex_stiffness()
    assert ke.shape == (24, 24)
    assert np.allclose(ke, ke.T)
    w = np.linalg.eigvalsh(ke)
    assert np.sum(np.abs(w) < 1e-9) == 6  # exactly the 6 rigid-body modes
    assert np.all(w > -1e-9)  # positive semidefinite


def test_simp3d_reduces_compliance_and_respects_regions() -> None:
    dom = topopt3d.cantilever3d(nelx=14, nely=6, nelz=6, volfrac=0.35, rmin=1.2)
    pocket = dom.elements_in_box(6, 2, 2, 9, 4, 4)
    mount = dom.elements_in_box(12, 2, 2, 14, 4, 4)
    dom.keep_void = pocket
    dom.keep_solid = mount
    res = topopt3d.optimize3d(dom, max_iter=10)
    assert res["compliance"][-1] < res["compliance"][0] * 0.7
    d = res["density"]  # (nelx, nely, nelz)
    assert abs(float(d.mean()) - 0.35) < 0.08
    for e in pocket:
        ix, r = divmod(e, dom.nely * dom.nelz)
        iy, iz = divmod(r, dom.nelz)
        assert d[ix, iy, iz] < 0.05
    for e in mount:
        ix, r = divmod(e, dom.nely * dom.nelz)
        iy, iz = divmod(r, dom.nelz)
        assert d[ix, iy, iz] > 0.95


# ---- the 3-D mesh path ---------------------------------------------------------------------


def test_density3d_to_mesh_watertight_solid(tmp_path: Path) -> None:
    d = np.zeros((20, 12, 12))
    d[3:17, 3:9, 3:9] = 1.0  # a solid bar
    res = mesh3d.density3d_to_mesh(d, out_dir=tmp_path, cell_mm=(1.0, 1.0, 1.0))
    assert res is not None
    assert res.watertight
    mesh = trimesh.load(str(res.stl), force="mesh")
    assert mesh.is_watertight
    assert float(mesh.volume) > 100.0


def test_density3d_to_mesh_fail_closed_on_empty(tmp_path: Path) -> None:
    assert (
        mesh3d.density3d_to_mesh(np.zeros((8, 8, 8)), out_dir=tmp_path, cell_mm=(1, 1, 1)) is None
    )
