"""Robust selectors (lever-b): name a face/edge by axis+position so it survives a
boolean that renumbers the B-rep, instead of a raw index that silently retargets."""

from __future__ import annotations

from build123d import Axis, Box

from forge_mind.agentic.helpers import (
    bottom_face,
    face_facing,
    largest_face,
    top_edges,
    top_face,
    vertical_edges,
)


def test_top_and_bottom_face_pick_the_z_extremes() -> None:
    box = Box(40, 40, 20)  # spans z in [-10, 10]
    assert abs(top_face(box).center().Z - 10) < 1e-6
    assert abs(bottom_face(box).center().Z - (-10)) < 1e-6


def test_face_facing_picks_the_side_by_normal() -> None:
    box = Box(40, 30, 20)  # x in [-20, 20], y in [-15, 15]
    assert abs(face_facing(box, Axis.X).center().X - 20) < 1e-6
    assert abs(face_facing(box, Axis.X, positive=False).center().X - (-20)) < 1e-6
    assert abs(face_facing(box, Axis.Y).center().Y - 15) < 1e-6


def test_largest_face_is_the_biggest_by_area() -> None:
    box = Box(40, 40, 10)  # the 40x40 faces (area 1600) beat the 40x10 sides (400)
    assert abs(largest_face(box).area - 1600) < 1e-3  # OCP area carries tiny float error


def test_edge_selectors_pick_the_expected_sets() -> None:
    box = Box(40, 40, 20)
    verts = vertical_edges(box)
    assert len(verts) == 4
    assert all(abs(e.length - 20) < 1e-6 for e in verts)  # the four upright corners
    top = top_edges(box)
    assert len(top) == 4
    assert all(abs(v.Z - 10) < 1e-6 for e in top for v in e.vertices())  # the top rim
