"""T2 pair rotation (ADR-0086): free spins free, a press fit locks, a wall limits the arc,
and anything uncertifiable refuses — measured, never guessed."""

from __future__ import annotations

import numpy as np
import trimesh

from forge_mind.motion import pair_rotation


def _spider_rotor() -> trimesh.Trimesh:
    """A k=3 rotor: hub + three arms at 120° (tip radius 16). One watertight body."""
    hub = trimesh.creation.cylinder(radius=6.0, height=3.0)
    parts = [hub]
    for k in range(3):
        arm = trimesh.creation.box(extents=(11.0, 2.4, 3.0))
        # spans x 5..16: overlaps INTO the hub (r=6) so the boolean truly fuses one solid —
        # a merely-tangent arm unions into separate touching shells that split apart
        arm.apply_translation((10.5, 0.0, 0.0))
        arm.apply_transform(
            trimesh.transformations.rotation_matrix(np.deg2rad(120.0 * k), (0, 0, 1))
        )
        parts.append(arm)
    rotor = trimesh.boolean.union(parts)
    assert rotor.is_watertight
    return rotor


def test_clearance_pair_is_certified_free() -> None:
    # a Ø8 shaft in a Ø10 bore ring: 1 mm radial clearance, full turn clear.
    shaft = trimesh.creation.cylinder(radius=4.0, height=30.0)
    ring = trimesh.creation.annulus(r_min=5.0, r_max=10.0, height=8.0)
    pm = pair_rotation([shaft, ring])
    assert pm.status == "free"
    assert pm.poses_tested == 24
    assert pm.clearance_mm is not None and 0.8 < pm.clearance_mm < 1.2
    d = np.asarray(pm.axis_direction, dtype=float)
    assert abs(abs(float(d[2])) - 1.0) < 1e-3  # spins about Z


def test_interpenetrating_pair_is_locked_with_measured_overlap() -> None:
    # a Ø11 shaft "in" a Ø10 bore: bodies overlap as built -- press-fit/fused, cannot rotate.
    shaft = trimesh.creation.cylinder(radius=5.5, height=30.0)
    ring = trimesh.creation.annulus(r_min=5.0, r_max=10.0, height=8.0)
    pm = pair_rotation([shaft, ring])
    assert pm.status == "locked"
    assert pm.interference_volume_mm3 is not None and pm.interference_volume_mm3 > 1.0
    assert "cannot rotate" in pm.explanation


def test_wall_limits_the_arc_and_reports_where() -> None:
    # a 3-arm spider next to a wall at 60°: as-built clear, rotation blocked every 120°.
    rotor = _spider_rotor()
    wall = trimesh.creation.box(extents=(4.0, 4.0, 10.0))
    wall.apply_translation((13.0 * np.cos(np.deg2rad(60.0)), 13.0 * np.sin(np.deg2rad(60.0)), 0.0))
    pm = pair_rotation([rotor, wall])
    assert pm.status == "arc_limited"
    assert pm.blocked_at_deg, "the blocked poses must be reported"
    assert pm.allowed_arc_deg is not None
    lo, hi = pm.allowed_arc_deg
    assert -60.0 <= lo <= -15.0 and 15.0 <= hi <= 60.0
    assert lo < 0.0 < hi  # the as-built pose sits inside the allowed arc


def test_open_shell_refuses() -> None:
    ring = trimesh.creation.annulus(r_min=5.0, r_max=10.0, height=8.0)
    open_shell = trimesh.Trimesh(vertices=ring.vertices, faces=ring.faces[:-40], process=False)
    pm = pair_rotation([open_shell, trimesh.creation.cylinder(radius=4.0, height=30.0)])
    assert pm.status == "refused"
    assert "closed solid" in pm.explanation


def test_axisless_pair_refuses() -> None:
    # two boxes: no measured spin axis on either body -> nothing may animate.
    a = trimesh.creation.box(extents=(20.0, 30.0, 8.0))
    b = trimesh.creation.box(extents=(40.0, 25.0, 6.0))
    b.apply_translation((60.0, 0.0, 0.0))
    pm = pair_rotation([a, b])
    assert pm.status == "refused"
    assert "no measured spin axis" in pm.explanation
