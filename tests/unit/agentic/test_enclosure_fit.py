"""Enclosure in the fit tier (deep-review, the founder's pump-in-a-tube class): a part
surrounded by other bodies with DESIGNED clearance is seated-by-enclosure, never
'floating' -- while a genuinely adrift part still floats and interference still fails.
Plus the decomposer teaching + the judge occlusion note that ride the same detection."""

from __future__ import annotations

from pathlib import Path

import trimesh

from forge_mind.agentic.functional import _DECOMPOSE_SYSTEM, _assembly_fit


def _fit(meshes: list[trimesh.Trimesh]) -> dict | None:
    return _assembly_fit(meshes, keys=[(f"m{i}", 0.0) for i in range(len(meshes))], cache=None)


def test_enclosed_part_with_designed_clearance_is_seated(tmp_path: Path) -> None:
    tube = trimesh.creation.annulus(r_min=20.0, r_max=25.0, height=30.0)
    impeller = trimesh.creation.box(extents=(10.0, 10.0, 10.0))  # 15mm clearance all around
    fit = _fit([tube, impeller])
    assert fit is not None
    assert fit["enclosed_parts"] == [1]
    assert fit["floating_parts"] == []
    assert fit["seated"] is True


def test_genuinely_adrift_part_still_floats(tmp_path: Path) -> None:
    tube = trimesh.creation.annulus(r_min=20.0, r_max=25.0, height=30.0)
    stray = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    stray.apply_translation((120.0, 0.0, 0.0))  # nowhere near anything
    fit = _fit([tube, stray])
    assert fit is not None
    assert fit["enclosed_parts"] == []
    assert 1 in fit["floating_parts"]
    assert fit["seated"] is False


def test_interfering_part_never_reads_enclosed(tmp_path: Path) -> None:
    """No-false-PASS: enclosure never hides interpenetration evidence. A part that eats
    the wall TOUCHES it, so it is not a floating candidate at all -- the pair's measured
    interference stays on the report (this tier's mate semantics treat moderate overlap
    as seated; burial and the sweep own deep interference, unchanged by enclosure)."""
    tube = trimesh.creation.annulus(r_min=20.0, r_max=25.0, height=30.0)
    fat = trimesh.creation.cylinder(radius=22.0, height=10.0)  # pierces the inner wall
    fit = _fit([tube, fat])
    assert fit is not None
    assert fit["enclosed_parts"] == []  # touching-by-interference is not enclosure
    assert any(p["interference_mm3"] > 1.0 for p in fit["pairs"])


def test_decomposer_teaches_orbit_void_and_enclosure() -> None:
    """The three-iteration failure class (housings intruding into the planet orbit) and
    the enclosure archetype are TAUGHT with arithmetic, not hoped for."""
    assert "ORBIT ANNULUS" in _DECOMPOSE_SYSTEM
    assert "module*(teeth+2)/2" in _DECOMPOSE_SYSTEM
    assert "ENCLOSED PARTS" in _DECOMPOSE_SYSTEM
    assert "INSERTION path" in _DECOMPOSE_SYSTEM
