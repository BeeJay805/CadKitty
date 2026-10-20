"""Parametric implicit archetypes: each builds a watertight, positive-volume mesh (ADR-0026)."""

from __future__ import annotations

from forge_mind.implicit.archetypes import gyroid_block, schwarz_block, strut_lattice_block


def test_gyroid_block_watertight() -> None:
    p = gyroid_block(size_mm=24, cell_mm=8, wall_mm=1.4)
    assert p.archetype == "gyroid"
    assert p.watertight
    assert p.mesh.volume > 0
    assert "STEP is not available" in p.note  # honest mesh-only messaging


def test_schwarz_block_watertight() -> None:
    p = schwarz_block(size_mm=24, cell_mm=8, wall_mm=1.4)
    assert p.archetype == "schwarz_p"
    assert p.watertight
    assert p.mesh.volume > 0


def test_strut_lattice_watertight() -> None:
    p = strut_lattice_block(size_mm=24, cell_mm=8, strut_mm=2.0)
    assert p.archetype == "strut_lattice"
    assert p.watertight
    assert p.mesh.volume > 0


def test_size_fits_within_clamped_envelope() -> None:
    p = gyroid_block(size_mm=24, cell_mm=8, wall_mm=1.4)
    ext = p.mesh.extents
    # The lattice is clipped to a 24mm cube; allow a little marching-cubes slack.
    assert all(e <= 26.0 for e in ext)


def test_params_are_clamped() -> None:
    p = gyroid_block(size_mm=9999, cell_mm=0.1, wall_mm=99)
    assert p.params["size_mm"] <= 200
    assert p.params["cell_mm"] >= 3
    assert p.params["wall_mm"] <= 12
