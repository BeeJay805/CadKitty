"""The guarded hole-repair backstop (ADR-0033 Stage 2): cuts a finished part's MISSING through-holes
deterministically where placement is defensible, re-validates, and refuses everything else. Real
kernel geometry, no LLM spend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import trimesh

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.repair import attempt_hole_repair


def _plate_files(tmp_path: Path) -> tuple[Path, Path]:
    """A holeless 60x60x6 plate exported as STL + STEP (the 'author dropped the holes' artifact)."""
    from build123d import Box, export_step, export_stl

    plate = Box(60, 60, 6)
    stl, step = tmp_path / "plate.stl", tmp_path / "plate.step"
    export_stl(plate, str(stl))
    export_step(plate, str(step))
    return stl, step


def _brief(dims: str, *, count: int = 4, standard: str | None = "M4") -> DesignBrief:
    return DesignBrief(
        object_name="mounting plate",
        summary="a plate",
        visual_description="a flat square plate",
        overall_envelope_mm="60 x 60 x 6",
        features=[
            BriefFeature(
                name="bolt holes",
                purpose="mounting",
                approx_dims_mm=dims,
                method="bolt_holes",
                count=count,
                standard=standard,
            )
        ],
        min_wall_mm=2.0,
        print_orientation_note="flat side down",
    )


def _genus(stl: Path) -> int:
    mesh: Any = trimesh.load(str(stl), force="mesh")
    return max(0, round(1 - mesh.euler_number / 2))


def test_repairs_a_stated_square_pattern(tmp_path: Path) -> None:
    stl, step = _plate_files(tmp_path)
    out = attempt_hole_repair(stl, step, _brief("M4 on a 40 mm square pattern"), tmp_path)
    assert out is not None
    assert out.holes_added == 4
    assert _genus(out.stl) == 4  # all four holes really go THROUGH
    assert out.validation.features_missing is False
    assert out.validation.body_count == 1


def test_repairs_a_stated_bolt_circle(tmp_path: Path) -> None:
    stl, step = _plate_files(tmp_path)
    out = attempt_hole_repair(stl, step, _brief("M4 on a 40 mm bolt circle"), tmp_path)
    assert out is not None
    assert _genus(out.stl) == 4
    assert out.validation.features_missing is False


def test_corner_layout_when_nothing_is_stated(tmp_path: Path) -> None:
    stl, step = _plate_files(tmp_path)
    out = attempt_hole_repair(stl, step, _brief("four M4 clearance holes"), tmp_path)
    assert out is not None
    assert _genus(out.stl) == 4


def test_multi_region_brief_is_vetoed(tmp_path: Path) -> None:
    # "two per leg": genus cannot verify POSITION across regions -- the repair must refuse.
    stl, step = _plate_files(tmp_path)
    out = attempt_hole_repair(stl, step, _brief("two M4 clearance holes per leg"), tmp_path)
    assert out is None


def test_unstated_diameter_is_refused(tmp_path: Path) -> None:
    # no standard + no M-size anywhere: never guess a dimension.
    stl, step = _plate_files(tmp_path)
    out = attempt_hole_repair(stl, step, _brief("four clearance holes", standard=None), tmp_path)
    assert out is None


def test_part_already_complete_is_untouched(tmp_path: Path) -> None:
    from build123d import Box, Cylinder, Pos, export_step, export_stl

    part = Box(60, 60, 6)
    for x, y in ((-20, -20), (20, -20), (20, 20), (-20, 20)):
        part -= Pos(x, y, 0) * Cylinder(radius=2.25, height=10)
    stl, step = tmp_path / "ok.stl", tmp_path / "ok.step"
    export_stl(part, str(stl))
    export_step(part, str(step))
    assert attempt_hole_repair(stl, step, _brief("M4 on a 40 mm square pattern"), tmp_path) is None


def test_pattern_too_big_for_the_face_is_refused(tmp_path: Path) -> None:
    # a 200 mm bolt circle cannot fit a 60 mm plate: the fit check must refuse, not clip.
    stl, step = _plate_files(tmp_path)
    out = attempt_hole_repair(stl, step, _brief("M4 on a 200 mm bolt circle"), tmp_path)
    assert out is None


@pytest.mark.parametrize("count,expected_genus", [(1, 1), (2, 2)])
def test_small_counts_use_canonical_layouts(
    tmp_path: Path, count: int, expected_genus: int
) -> None:
    stl, step = _plate_files(tmp_path)
    out = attempt_hole_repair(
        stl, step, _brief(f"{count} M4 clearance holes", count=count), tmp_path
    )
    assert out is not None
    assert _genus(out.stl) == expected_genus
