"""Export integrity (bench baseline row 17): the two failure classes that were EXPORT
artifacts, not geometry -- a STEP product named 'SOLID' (5 rubric failures on valid parts)
and STLs reading non-watertight only because OCC leaves duplicate vertices on face seams
(both no_build parts: single-body, correct volume, cracked mesh, 8 wasted iterations each).
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from forge_mind.agentic._sandbox_runner import _weld_stl_seams
from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code


def test_step_product_carries_the_part_name(tmp_path: Path) -> None:
    """An unlabeled Part must export as PRODUCT('<part_name>'), never PRODUCT('SOLID')."""
    res = run_build123d_code(
        "result = Box(20, 10, 5)",
        tmp_path,
        part_name="shelf_bracket",
        limits=SandboxLimits(timeout_s=120),
    )
    assert res.ok and res.step_path is not None
    text = res.step_path.read_text(encoding="utf-8", errors="replace")
    assert "PRODUCT('shelf_bracket'" in text
    assert "PRODUCT('SOLID'" not in text


def test_weld_closes_seam_duplicates_only(tmp_path: Path) -> None:
    """A solid whose export carries duplicate/degenerate seam faces reads non-watertight;
    the weld (merge exact vertices + drop invalid faces) restores it. A mesh with a REAL
    hole must stay open (the honesty bound: welding can never bridge a genuine gap)."""
    import numpy as np

    box = trimesh.creation.box(extents=[10, 10, 10])
    seamy = trimesh.Trimesh(
        vertices=box.vertices.copy(),
        faces=np.vstack([box.faces, box.faces[:1]]),  # one face duplicated, as OCC seams do
        process=False,
    )
    p = tmp_path / "seamy.stl"
    seamy.export(str(p))
    assert not bool(trimesh.load(str(p), force="mesh").is_watertight)  # broken as exported
    _weld_stl_seams(p)
    assert bool(trimesh.load(str(p), force="mesh").is_watertight)

    cracked = trimesh.creation.box(extents=[10, 10, 10])
    cracked.faces = cracked.faces[:-1]  # remove a triangle: a REAL hole
    q = tmp_path / "cracked.stl"
    cracked.export(str(q))
    _weld_stl_seams(q)
    assert not bool(trimesh.load(str(q), force="mesh").is_watertight)
