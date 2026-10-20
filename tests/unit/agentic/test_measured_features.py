"""Measured feedback beyond holes (the lever-c pattern extended): convex BOSS/pin audit
by stated diameter, and the wall-thickness CLUSTER map. Both advisory, both fail-open --
a false measurement poisons the author worse than none."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import trimesh
from build123d import Box, Cylinder, Pos

from forge_mind.agentic.spatial import (
    _planned_boss_specs,
    measured_boss_feedback,
    wall_cluster_feedback,
)

pytestmark = pytest.mark.slow  # builds tiny B-reps (build123d/OCCT)


def _boss_brief(count: int = 2, dims: str = "Ø8 mm x 10 tall"):
    return SimpleNamespace(
        features=[
            SimpleNamespace(name="standoff boss", approx_dims_mm=dims, count=count, standard=None)
        ]
    )


def _plate_with_bosses(n: int, d_mm: float = 8.0):
    part = Box(60, 60, 5)
    spots = [(-20, -20), (20, 20)][:n]
    for x, y in spots:
        part += Pos(x, y, 7.5) * Cylinder(d_mm / 2.0, 10)
    return part


def test_boss_specs_require_a_stated_diameter() -> None:
    assert len(_planned_boss_specs(_boss_brief())) == 1
    # no diameter stated -> not auditable -> no spec (never guessed)
    assert _planned_boss_specs(_boss_brief(dims="short cylindrical posts")) == []
    # a 'pin hole' is a hole plan, not a boss plan
    hole_brief = SimpleNamespace(
        features=[
            SimpleNamespace(name="pin hole", approx_dims_mm="Ø4 mm dia", count=2, standard=None)
        ]
    )
    assert _planned_boss_specs(hole_brief) == []


def test_missing_bosses_are_named_with_measured_stacks() -> None:
    part = _plate_with_bosses(1)  # brief plans 2
    lines = measured_boss_feedback(part, _boss_brief(count=2))
    assert len(lines) == 1
    assert "MISSING 1 x Ø8.00mm" in lines[0]
    assert "standoff boss" in lines[0]
    assert "Ø8.00mm at" in lines[0]  # the measured stack is listed for placement


def test_complete_bosses_say_nothing() -> None:
    part = _plate_with_bosses(2)
    assert measured_boss_feedback(part, _boss_brief(count=2)) == []


def test_no_boss_plan_says_nothing() -> None:
    part = _plate_with_bosses(2)
    brief = SimpleNamespace(
        features=[SimpleNamespace(name="body", approx_dims_mm="60x60x5", count=1, standard=None)]
    )
    assert measured_boss_feedback(part, brief) == []


def _mesh_of(part) -> trimesh.Trimesh:
    import tempfile
    from pathlib import Path

    from build123d import export_stl

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.stl"
        export_stl(part, str(p))
        m = trimesh.load(str(p), force="mesh")
    assert isinstance(m, trimesh.Trimesh)
    return m


def test_wall_clusters_fire_only_under_minimum() -> None:
    thin = _mesh_of(Box(60, 60, 1.0))  # 1mm slab vs 2mm min -> fires
    lines = wall_cluster_feedback(thin, 2.0)
    assert len(lines) == 1 and "WALL MAP" in lines[0] and "1.0mm" in lines[0]

    thick = _mesh_of(Box(60, 60, 8.0))  # comfortably above the minimum -> silent
    assert wall_cluster_feedback(thick, 2.0) == []
