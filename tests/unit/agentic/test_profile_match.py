"""Axial profile (ADR-0073 Stage 2): hand-verifiable on primitives -- a cylinder is flat and
on-axis, a cone ramps, an offset lug shows in the offset channel, scale does not matter, and the
distance is flip-invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from forge_mind.agentic.profile_match import axial_profile, profile_distance


def _save(tmp: Path, name: str, mesh: Any) -> Path:
    p = tmp / name
    mesh.export(str(p))
    return p


def test_cylinder_is_flat_and_on_axis(tmp_path: Path) -> None:
    p = axial_profile(_save(tmp_path, "cyl.stl", trimesh.creation.cylinder(radius=8, height=60)))
    assert p is not None
    mid = np.array(p.r_max[4:-4])  # end slices include the flat caps; judge the barrel
    assert float(mid.min()) > 0.9  # silhouette ~= the radius everywhere
    # axisymmetric -> centroid ON the axis: measured sampling-noise floor at 16k samples is
    # mean ~0.037 / worst-slice ~0.10 (vs a real crank-throw signal of 0.4+)
    off = np.array(p.offset)
    assert off.mean() < 0.06 and off.max() < 0.15


def test_cone_ramps(tmp_path: Path) -> None:
    p = axial_profile(_save(tmp_path, "cone.stl", trimesh.creation.cone(radius=10, height=50)))
    assert p is not None
    r = np.array(p.r_max)
    ramp = r if r[0] < r[-1] else r[::-1]  # axis direction is arbitrary
    assert ramp[-2] - ramp[1] > 0.5  # wide end >> tip


def test_offset_throws_read_in_offset_channel(tmp_path: Path) -> None:
    """A crank-like stack: journals ON the axis, two throws offset +X / -X (symmetric, so the
    principal axis stays the crank centerline -- as on a real crankshaft). The offset channel
    must light up AT the throws and stay near zero at the journals."""
    segs = []
    for i, dx in enumerate([0.0, 10.0, 0.0, -10.0, 0.0]):
        c = trimesh.creation.cylinder(radius=6, height=15)
        c.apply_translation([dx, 0, 15.0 * i])
        segs.append(c)
    p = axial_profile(_save(tmp_path, "crankish.stl", trimesh.util.concatenate(segs)))
    assert p is not None
    off = np.array(p.offset)
    n = len(off)
    throws = np.r_[off[int(n * 0.25) : int(n * 0.35)], off[int(n * 0.65) : int(n * 0.75)]]
    journals = np.r_[
        off[: int(n * 0.12)], off[int(n * 0.44) : int(n * 0.56)], off[-int(n * 0.12) :]
    ]
    assert throws.mean() - journals.mean() > 0.2  # throws clearly off-axis, journals on-axis


def test_identical_and_scaled_are_zero_distance(tmp_path: Path) -> None:
    small = _save(tmp_path, "s.stl", trimesh.creation.cylinder(radius=5, height=40))
    big = _save(tmp_path, "b.stl", trimesh.creation.cylinder(radius=15, height=120))
    ps, pb = axial_profile(small), axial_profile(big)
    assert ps is not None and pb is not None
    same = profile_distance(ps, ps)
    scaled = profile_distance(ps, pb)
    assert same is not None and same < 1e-9  # identical -> 0
    assert scaled is not None and scaled < 0.05  # scale-invariant: same FORM


def test_flip_invariance(tmp_path: Path) -> None:
    cone = trimesh.creation.cone(radius=10, height=50)
    flipped = trimesh.creation.cone(radius=10, height=50)
    flipped.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    pa = axial_profile(_save(tmp_path, "c1.stl", cone))
    pb = axial_profile(_save(tmp_path, "c2.stl", flipped))
    assert pa is not None and pb is not None
    d = profile_distance(pa, pb)
    assert d is not None and d < 0.05  # the same cone upside down is the same form


def test_different_forms_separate(tmp_path: Path) -> None:
    cyl = axial_profile(_save(tmp_path, "cy.stl", trimesh.creation.cylinder(radius=10, height=50)))
    cone = axial_profile(_save(tmp_path, "co.stl", trimesh.creation.cone(radius=10, height=50)))
    assert cyl is not None and cone is not None
    d = profile_distance(cyl, cone)
    assert d is not None and d > 0.1  # a cone is not a cylinder


def test_fail_soft(tmp_path: Path) -> None:
    assert axial_profile(tmp_path / "nope.stl") is None
