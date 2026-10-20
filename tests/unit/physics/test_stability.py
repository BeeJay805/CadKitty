"""Stand-or-tip stability: a centered body stands, a cantilevered mass tips, no flat base refuses."""

from __future__ import annotations

import math

import trimesh

from forge_mind.physics.stability import compute_stability


def test_centered_box_stands() -> None:
    r = compute_stability(trimesh.creation.box(extents=(20, 20, 10)))
    assert r is not None
    assert r.stable
    assert r.margin_mm > 0.0
    # CoM is centered over a 20x20 base -> ~10 mm to the nearest edge.
    assert 8.0 <= r.margin_mm <= 10.5
    assert r.tip_angle_deg > 0.0


def test_cantilevered_mass_tips() -> None:
    # A small base plate with a heavy column hung far out in +x: the CoM leaves the base.
    base = trimesh.creation.box(extents=(10, 10, 2))
    top = trimesh.creation.box(extents=(10, 10, 40))
    top.apply_translation((30.0, 0.0, 21.0))
    body = trimesh.util.concatenate([base, top])
    r = compute_stability(body)
    assert r is not None
    assert not r.stable
    assert r.margin_mm < 0.0
    assert r.tip_angle_deg == 0.0  # already past the edge


def test_edge_balance_refuses() -> None:
    # A box tipped onto a single bottom edge -> collinear contact, no footprint -> refuse.
    box = trimesh.creation.box(extents=(40, 10, 10))
    box.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 4, (1, 0, 0)))
    assert compute_stability(box) is None
