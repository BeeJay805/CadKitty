"""Internal (ring) gear meshes run at the pitch-radius DIFFERENCE, external at the sum
(strictness-ratchet change, north-star gear-train iteration 1: three correct ring meshes
at |48-18|*2/2 = 30 mm false-FAILed against the sum formula, 66 mm). Two-sided by the
ratchet's rules: the correct spacing passes ONLY against its own formula, and a seeded
mis-spacing still fails. External pairs are byte-unaffected."""

from __future__ import annotations

from pathlib import Path

import trimesh

from forge_mind.agentic.assembly_adapter import AuthoredPart, build_assembly_model
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.types import Vec3
from forge_mind.validator.assembly_checks import _gear_mesh_checks
from forge_mind.validator.report import CheckStatus


def _axis_port(x: float) -> Port:
    return Port(
        name="axis",
        origin=Vec3(x=x, y=0.0, z=0.0),
        axis=Vec3(x=0.0, y=0.0, z=1.0),
        normal=Vec3(x=1.0, y=0.0, z=0.0),
    )


def _model(tmp: Path, planet_x: float, *, internal: bool) -> object:
    parts = []
    for name, x in (("ring", 0.0), ("planet", planet_x)):
        stl = tmp / f"{name}.stl"
        trimesh.creation.box(extents=(6, 6, 6)).apply_translation((x, 0, 0)).export(str(stl))
        parts.append(AuthoredPart(id=name, stl_path=stl, step_path=None, ports=[_axis_port(x)]))
    joint = Joint(
        type=JointType.GEAR_MESH,
        a="ring.axis",
        b="planet.axis",
        module=2.0,
        teeth_a=48,
        teeth_b=18,
        internal=internal,
    )
    return build_assembly_model(parts, [joint])


def _spacing(tmp: Path, planet_x: float, *, internal: bool) -> CheckStatus:
    checks: list = []
    _gear_mesh_checks(_model(tmp, planet_x, internal=internal), checks, [])
    return next(c for c in checks if c.id == "assembly.gear_center_distance").status


def test_internal_mesh_passes_at_the_pitch_radius_difference(tmp_path: Path) -> None:
    # |48-18| * 2 / 2 = 30 mm -- the correct ring-planet spacing
    assert _spacing(tmp_path, 30.0, internal=True) is CheckStatus.PASS


def test_internal_mesh_seeded_misspacing_still_fails(tmp_path: Path) -> None:
    """The ratchet's seeded defect: the formula fix must not stop catching real errors."""
    assert _spacing(tmp_path, 25.0, internal=True) is CheckStatus.FAIL
    # and the OLD (external-sum) spacing is now correctly rejected for an internal pair
    assert _spacing(tmp_path, 66.0, internal=True) is CheckStatus.FAIL


def test_external_formula_is_unchanged(tmp_path: Path) -> None:
    # external 48T+18T at module 2 meshes at (48+18)*2/2 = 66 mm, and only there
    assert _spacing(tmp_path, 66.0, internal=False) is CheckStatus.PASS
    assert _spacing(tmp_path, 30.0, internal=False) is CheckStatus.FAIL


def test_internal_pair_sim_scope_is_honest(tmp_path: Path) -> None:
    """No internal tooth sim exists yet: the gear_mesh check must SAY so, never count a
    ring pair as simulated (an unearned claim is a false PASS in disguise)."""
    checks: list = []
    _gear_mesh_checks(_model(tmp_path, 30.0, internal=True), checks, [])
    mesh = next(c for c in checks if c.id == "assembly.gear_mesh")
    assert mesh.status is CheckStatus.PASS
    assert "spacing/skew-gated only" in (mesh.actual or "")
