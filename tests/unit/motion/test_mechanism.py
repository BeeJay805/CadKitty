"""T3 measured mechanisms (ADR-0086 slice 3): a bearing's coupled motion is certified at the
MEASURED epicyclic ratio; meshed gears earn coupled counter-rotation at the MEASURED tooth
ratio; unrecognized patterns refuse."""

from __future__ import annotations

import numpy as np
import trimesh

from forge_mind.motion import gear_pair_coupled, pair_rotation, rolling_mechanism


def _bearing(ball_r: float = 2.8) -> list[trimesh.Trimesh]:
    """Outer race + inner race + 6 balls on the orbit circle. ball_r=2.8 leaves 0.2 mm of
    radial clearance each side; 3.2 interpenetrates both races."""
    outer = trimesh.creation.annulus(r_min=14.0, r_max=18.0, height=6.0)
    inner = trimesh.creation.annulus(r_min=4.0, r_max=8.0, height=6.0)
    bodies = [outer, inner]
    for k in range(6):
        ball = trimesh.creation.icosphere(subdivisions=2, radius=ball_r)
        a = np.deg2rad(60.0 * k)
        ball.apply_translation((11.0 * np.cos(a), 11.0 * np.sin(a), 0.0))
        bodies.append(ball)
    return bodies


def _paddle_wheel(phase_deg: float) -> trimesh.Trimesh:
    """A 4-pin 'gear': hub + pins reaching r=10 (pins overlap INTO the hub — tangent-only
    unions split into touching shells, measured earlier on the spider fixture)."""
    hub = trimesh.creation.cylinder(radius=5.0, height=3.0)
    parts = [hub]
    for k in range(4):
        pin = trimesh.creation.box(extents=(6.0, 1.2, 3.0))
        pin.apply_translation((7.0, 0.0, 0.0))  # spans x 4..10
        pin.apply_transform(
            trimesh.transformations.rotation_matrix(np.deg2rad(phase_deg + 90.0 * k), (0, 0, 1))
        )
        parts.append(pin)
    wheel = trimesh.boolean.union(parts)
    assert wheel.is_watertight
    return wheel


def test_bearing_certified_free_at_the_measured_orbit_ratio() -> None:
    mm = rolling_mechanism(_bearing())
    assert mm.status == "free"
    assert mm.kind == "rolling"
    # inner race drives; outer is the stator; all six balls are coupled
    assert len(mm.couplings) == 6
    assert mm.poses_tested == 24
    # measured epicyclic ratio Ri/(Ri+Ro) with r_orbit=11, r_ball=2.8 -> 8.2/22 = 0.373
    for c in mm.couplings:
        assert 0.34 < c.ratio < 0.41
    assert "Ri" in mm.evidence and "epicyclic" in mm.evidence
    d = np.asarray(mm.axis_direction, dtype=float)
    assert abs(abs(float(d[2])) - 1.0) < 1e-3  # the main axis is Z


def test_bearing_with_oversized_balls_is_locked() -> None:
    mm = rolling_mechanism(_bearing(ball_r=3.2))
    assert mm.status == "locked"
    assert mm.interference_volume_mm3 is not None and mm.interference_volume_mm3 > 0.0
    assert "cannot move" in mm.explanation


def test_unrecognized_multibody_refuses() -> None:
    boxes = []
    for i in range(3):
        b = trimesh.creation.box(extents=(6.0, 9.0, 4.0))
        b.apply_translation((i * 20.0, 0.0, 0.0))
        boxes.append(b)
    mm = rolling_mechanism(boxes)
    assert mm.status == "refused"
    assert (
        "nothing animates" in mm.explanation or "share a measured rotation axis" in mm.explanation
    )


def test_meshed_paddle_gears_earn_coupled_rotation() -> None:
    # A pins at 45deg phase, B pins at 0 phase, centers 18 apart: interleaved. Turning A
    # alone MUST hit B's pin (arc-limited); the coupled counter-rotation at the measured
    # 4/4 ratio clears a full turn.
    a = _paddle_wheel(45.0)
    b = _paddle_wheel(0.0)
    b.apply_translation((18.0, 0.0, 0.0))
    pair = pair_rotation([a, b])
    assert pair.status == "arc_limited"  # uncoupled: honest partial arc

    mm = gear_pair_coupled([a, b], pair)
    assert mm is not None and mm.status == "free"
    assert mm.kind == "gear_pair"
    assert len(mm.couplings) == 1
    assert abs(mm.couplings[0].ratio + 1.0) < 0.01  # external mesh, z/z -> -1
    assert "tooth counts" in mm.evidence


def test_gear_coupling_never_invented_for_a_wall() -> None:
    # the arc-limited spider-vs-wall pair is NOT a gear mesh: the wall has no measurable
    # tooth count / rotor axis, so no coupling may be invented and the honest arc stands.
    hub = trimesh.creation.cylinder(radius=6.0, height=3.0)
    parts = [hub]
    for k in range(3):
        arm = trimesh.creation.box(extents=(11.0, 2.4, 3.0))
        arm.apply_translation((10.5, 0.0, 0.0))
        arm.apply_transform(
            trimesh.transformations.rotation_matrix(np.deg2rad(120.0 * k), (0, 0, 1))
        )
        parts.append(arm)
    rotor = trimesh.boolean.union(parts)
    wall = trimesh.creation.box(extents=(4.0, 4.0, 10.0))
    wall.apply_translation((13.0 * np.cos(np.deg2rad(60.0)), 13.0 * np.sin(np.deg2rad(60.0)), 0.0))
    pair = pair_rotation([rotor, wall])
    assert pair.status == "arc_limited"
    assert gear_pair_coupled([rotor, wall], pair) is None
