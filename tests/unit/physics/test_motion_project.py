"""MotionProof projector (ADR-0086): claims only for measured motion, refusals stated,
advisory severity everywhere (motion can never move the verdict)."""

from __future__ import annotations

import numpy as np
import trimesh

from forge_mind.physics.report.motion_project import project_motion


def test_single_rotor_projects_a_spin_claim() -> None:
    ring = trimesh.creation.annulus(r_min=5.0, r_max=15.0, height=8.0)
    motion, findings = project_motion(ring)
    assert motion is not None
    assert [c.kind for c in motion.claims] == ["spin"]
    claim = motion.claims[0]
    assert claim.status == "free" and claim.moving == ["body_0"]
    assert claim.axis is not None and claim.axis.frame == "stl_z_up"
    d = np.array([claim.axis.direction.x, claim.axis.direction.y, claim.axis.direction.z])
    assert abs(abs(float(d[2])) - 1.0) < 1e-3
    assert all(f.severity == "advisory" for f in findings)


def test_single_bracket_projects_no_claim() -> None:
    motion, findings = project_motion(trimesh.creation.box(extents=(20.0, 30.0, 8.0)))
    assert motion is None
    assert len(findings) == 1 and findings[0].id == "motion.none"
    assert "nothing animates" in findings[0].explanation


def test_two_body_clearance_pair_projects_pair_claim() -> None:
    shaft = trimesh.creation.cylinder(radius=4.0, height=30.0)
    ring = trimesh.creation.annulus(r_min=5.0, r_max=10.0, height=8.0)
    mesh = trimesh.util.concatenate([shaft, ring])
    motion, findings = project_motion(mesh)
    assert motion is not None
    claim = motion.claims[0]
    assert claim.kind == "pair_rotation" and claim.status == "free"
    roles = {b.id: b.role for b in motion.bodies}
    assert roles[claim.moving[0]] == "rotor" and roles[claim.static[0]] == "stator"
    assert findings[0].measured is not None and "clearance" in findings[0].measured.unit


def test_unrecognized_three_bodies_refuse_with_reason() -> None:
    bodies = []
    for i in range(3):
        b = trimesh.creation.box(extents=(5.0, 8.0, 4.0))
        b.apply_translation((i * 20.0, 0.0, 0.0))
        bodies.append(b)
    motion, findings = project_motion(trimesh.util.concatenate(bodies))
    assert motion is None
    assert findings[0].status == "not_run"
    assert findings[0].reason_not_run is not None  # the mechanism refusal, stated


def test_bearing_projects_a_coupled_claim_with_measured_ratio() -> None:
    outer = trimesh.creation.annulus(r_min=14.0, r_max=18.0, height=6.0)
    inner = trimesh.creation.annulus(r_min=4.0, r_max=8.0, height=6.0)
    bodies = [outer, inner]
    for k in range(6):
        ball = trimesh.creation.icosphere(subdivisions=2, radius=2.8)
        a = np.deg2rad(60.0 * k)
        ball.apply_translation((11.0 * np.cos(a), 11.0 * np.sin(a), 0.0))
        bodies.append(ball)
    motion, findings = project_motion(trimesh.util.concatenate(bodies))
    assert motion is not None
    claim = motion.claims[0]
    assert claim.kind == "coupled_rotation" and claim.status == "free"
    assert claim.couplings is not None and len(claim.couplings) == 6
    assert all(0.34 < c.ratio < 0.41 for c in claim.couplings)
    roles = {b.id: b.role for b in motion.bodies}
    assert sum(1 for r in roles.values() if r == "roller") == 6
    assert sum(1 for r in roles.values() if r == "stator") == 1
    assert findings[0].severity == "advisory"
