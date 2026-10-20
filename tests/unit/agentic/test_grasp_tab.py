"""Grasp-tab emission (ADR-0012 §6), CELL LANE ONLY.

The load-bearing test is TWO-SIDED (the leak that must never ship is a tab on a customer part):
cell profile ON -> a measurable grip fin is fused; profile OFF -> the pipeline output is
BYTE-IDENTICAL. Plus placement across shapes and the fail-closed refusal when no valid grasp exists.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import trimesh

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.grasp_tab import GRIP_H, GRIP_T, GRIP_W, emit_grasp_tab
from forge_mind.agentic.loop import LoopResult
from forge_mind.agentic.result import _maybe_emit_grasp_tab
from forge_mind.materials import resolve_material


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="block",
        summary="a block",
        visual_description="a block",
        overall_envelope_mm="40 x 30 x 20",
        features=[BriefFeature(name="body", purpose="bulk", approx_dims_mm="x", method="box")],
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


def _shape(tmp_path: Path, kind: str) -> tuple[Path, Path]:
    """Build a shape, drop it onto the bed (min-z = 0), export STEP + STL."""
    from build123d import Box, BuildPart, Cylinder, Pos, Sphere, export_step, export_stl

    with BuildPart() as bp:
        if kind == "plate":
            Box(40, 30, 5)
        elif kind == "cylinder":
            Cylinder(20, 10)
        elif kind == "tall_box":
            Box(25, 25, 45)
        elif kind == "sphere":
            Sphere(18)
        else:  # pragma: no cover
            raise ValueError(kind)
    part = bp.part
    part = Pos(0, 0, -part.bounding_box().min.Z) * part
    step, stl = tmp_path / f"{kind}.step", tmp_path / f"{kind}.stl"
    export_step(part, str(step))
    export_stl(part, str(stl))
    return step, stl


def _loop_for(step: Path, stl: Path) -> LoopResult:
    return LoopResult(
        prompt="p",
        brief=_brief(),
        final_code="code",
        final_stl=stl,
        final_step=step,
        final_render=None,
        validation=None,
        critique=None,
        geometry_passed=True,
        critic_approved=True,
        stopped_reason="accepted",
    )


# --- the two-sided safety property ---------------------------------------------------------------


def test_website_lane_is_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_CELL_PROFILE", raising=False)  # website lane
    step, stl = _shape(tmp_path, "plate")
    loop = _loop_for(step, stl)
    before = hashlib.sha256(stl.read_bytes()).hexdigest()
    disp = _maybe_emit_grasp_tab(loop, "shipped", tmp_path, resolve_material("PLA"))
    assert disp == "shipped"
    assert loop.final_stl == stl  # unchanged path
    assert hashlib.sha256(stl.read_bytes()).hexdigest() == before  # unchanged bytes


def test_cell_lane_fuses_a_measurable_tab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_CELL_PROFILE", "1")
    step, stl = _shape(tmp_path, "plate")
    loop = _loop_for(step, stl)
    original_vol = float(abs(trimesh.load(str(stl), force="mesh").volume))
    disp = _maybe_emit_grasp_tab(loop, "shipped", tmp_path, resolve_material("PLA"))
    assert disp == "shipped"
    assert loop.final_stl is not None and loop.final_stl != stl  # now the tabbed file
    tabbed = trimesh.load(str(loop.final_stl), force="mesh")
    assert tabbed.is_watertight and tabbed.body_count == 1
    assert float(abs(tabbed.volume)) > original_vol  # material was added
    # the fin rises through the 10-15 mm grip band (plate top was 5 mm) -> bbox grows in z
    assert tabbed.bounds[1][2] > 15.0


# --- placement across shapes -----------------------------------------------------------------------


def test_plate_top_mounts(tmp_path: Path) -> None:
    step, stl = _shape(tmp_path, "plate")
    r = emit_grasp_tab(
        step, stl, tmp_path / "out", material=resolve_material("PLA"), brief=_brief()
    )
    assert r.ok and r.mode == "top"
    assert r.grip_dims_mm == (GRIP_W, GRIP_H, GRIP_T)
    assert r.mass_g is not None and r.mass_g <= 200.0


def test_cylinder_side_mounts(tmp_path: Path) -> None:
    step, stl = _shape(tmp_path, "cylinder")
    r = emit_grasp_tab(
        step, stl, tmp_path / "out", material=resolve_material("PLA"), brief=_brief()
    )
    assert r.ok and r.mode == "side"


def test_tall_box_is_refused(tmp_path: Path) -> None:
    step, stl = _shape(tmp_path, "tall_box")
    r = emit_grasp_tab(
        step, stl, tmp_path / "out", material=resolve_material("PLA"), brief=_brief()
    )
    assert r.refused and not r.ok
    assert r.reason is not None and "placement" in r.reason


def test_sphere_is_refused(tmp_path: Path) -> None:
    step, stl = _shape(tmp_path, "sphere")
    r = emit_grasp_tab(
        step, stl, tmp_path / "out", material=resolve_material("PLA"), brief=_brief()
    )
    assert r.refused and not r.ok


# --- seam behaviour --------------------------------------------------------------------------------


def test_cell_lane_refusal_sets_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_CELL_PROFILE", "1")
    step, stl = _shape(tmp_path, "sphere")
    loop = _loop_for(step, stl)
    disp = _maybe_emit_grasp_tab(loop, "shipped", tmp_path, resolve_material("PLA"))
    assert disp == "refused_grasp_tab"
    assert loop.refusal_reasons and "placement" in loop.refusal_reasons[0]


def test_rejected_part_never_gets_a_tab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_CELL_PROFILE", "1")
    step, stl = _shape(tmp_path, "plate")
    loop = _loop_for(step, stl)
    disp = _maybe_emit_grasp_tab(loop, "rejected", tmp_path, resolve_material("PLA"))
    assert disp == "rejected"
    assert loop.final_stl == stl  # untouched
