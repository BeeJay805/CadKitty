"""Nominal thread crests must not disappear behind simplified catalog roots."""

import math

import build123d as b
import pytest
import trimesh

from forge_mind.workspace.hardware import hardware_shape
from forge_mind.workspace.mesh_evidence import assembly_mesh_evidence
from forge_mind.workspace.tessellation import STL_ANGULAR_TOLERANCE, STL_LINEAR_TOLERANCE


@pytest.mark.parametrize(
    "catalog,diameter,length",
    [("button_m3x6", 3, 6), ("shcs_m4x25", 4, 25), ("hexbolt_m8x20", 8, 20)],
)
def test_envelope_includes_nominal_thread_crests_and_preserves_head(catalog, diameter, length):
    original = hardware_shape(catalog)
    result = hardware_shape(catalog, envelope=True)
    probe = b.Pos(0, 0, -length / 2) * b.Cylinder(diameter / 2 - 0.001, 0.2)
    assert result.is_valid and len(result.solids()) == 1
    assert (result & probe).volume == pytest.approx(probe.volume, abs=1e-6)
    assert (original & probe).volume < probe.volume * 0.95
    head = b.Pos(0, 0, 10.01) * b.Box(100, 100, 20)
    assert (result & head).volume == pytest.approx((original & head).volume, abs=1e-6)
    assert pytest.approx(-length) == result.bounding_box().min.Z


@pytest.mark.parametrize("catalog", ["csk_m3x8", "setscrew_m3x6", "hexnut_m3"])
def test_unsupported_envelopes_fail_closed(catalog):
    with pytest.raises(ValueError, match="envelope mode"):
        hardware_shape(catalog, envelope=True)


def test_small_rotated_button_head_meets_unchanged_volume_gate(tmp_path):
    # Original catalog shape reproduces the old ~0.137% volume discrepancy.
    shape = b.Pos(16, 16, 45) * b.Rot(-90, 0, 0) * hardware_shape("button_m3x6")
    step = tmp_path / "button.step"
    b.export_step(shape, step)
    exported = b.import_step(step)
    stl = tmp_path / "button.stl"
    b.export_stl(
        exported, stl, tolerance=STL_LINEAR_TOLERANCE, angular_tolerance=STL_ANGULAR_TOLERANCE
    )
    mesh = trimesh.load(stl, force="mesh")
    assert math.isclose(mesh.volume, exported.volume, rel_tol=0.001, abs_tol=0.001)
    raw = trimesh.load(stl, force="mesh", process=False)
    assert assembly_mesh_evidence(exported, raw)["valid"]
