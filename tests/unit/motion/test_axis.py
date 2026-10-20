"""T1 spin-axis discovery (ADR-0086): rotors earn a measured axis, brackets never do."""

from __future__ import annotations

import numpy as np
import trimesh

from forge_mind.motion import spin_axis


def _z_aligned(direction: tuple[float, float, float]) -> bool:
    d = np.asarray(direction, dtype=float)
    return abs(abs(float(d @ np.array([0.0, 0.0, 1.0]))) - 1.0) < 1e-3


def test_annulus_earns_a_spin_axis_along_z() -> None:
    # a pulley/bearing-race shape: k=inf rotor. The unique principal axis is Z.
    ring = trimesh.creation.annulus(r_min=5.0, r_max=15.0, height=8.0)
    axis = spin_axis(ring)
    assert axis is not None
    assert axis.method == "inertia_rotor"
    assert _z_aligned(axis.direction)
    assert axis.balance.balanced is True  # symmetric rotor: certified balanced


def test_plain_box_never_spins() -> None:
    # a bracket-class body has no rotor signature -- no claim, no fake motion.
    assert spin_axis(trimesh.creation.box(extents=(20.0, 30.0, 8.0))) is None


def test_open_shell_fails_closed() -> None:
    # inertia on a non-solid is untrustworthy: no axis rather than a guessed one.
    ring = trimesh.creation.annulus(r_min=5.0, r_max=15.0, height=8.0)
    open_shell = trimesh.Trimesh(vertices=ring.vertices, faces=ring.faces[:-40], process=False)
    assert not open_shell.is_watertight
    assert spin_axis(open_shell) is None


def test_near_cube_has_no_meaningful_unique_axis() -> None:
    # all three moments nearly equal: the distinctness gate refuses to invent an axis.
    assert spin_axis(trimesh.creation.box(extents=(20.0, 20.0, 20.0))) is None
