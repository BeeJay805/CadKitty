"""The slicer-gate TIER wiring inside validate_authored_part (ADR-0012 SS3).

Pins the flag x lane matrix, patching the ``slice_part`` seam so no real binary or slow slice runs
(the wrapper + fact parser are unit-tested in test_slicer_wrapper / test_gcode_facts):

  - flag OFF (default)         -> never slices; output byte-identical to today.
  - flag ON, WEBSITE lane      -> REPORT-ONLY: slicer issues surfaced, ``passed`` unaffected.
  - flag ON, CELL lane         -> fail-closed: a would-gate fact OR a missing binary REFUSES.
  - GENERAL (non-print) mode   -> never slices (the slicer gate is print-lane only).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.validate import validate_authored_part
from forge_mind.fabrication import FabricationMode
from forge_mind.slicing import slicer as slicer_mod
from forge_mind.slicing.slicer import SliceOutcome

_FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "slicer" / "tiny.gcode"


@pytest.fixture(scope="module")
def box_part(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    from build123d import Box, BuildPart, export_step

    d = tmp_path_factory.mktemp("slicer_gate")
    stl, step = d / "part.stl", d / "part.step"
    mesh = trimesh.creation.box((20.0, 20.0, 10.0))
    mesh.apply_translation([0.0, 0.0, 5.0])
    mesh.export(stl)
    with BuildPart() as bp:
        Box(20, 20, 10)
    export_step(bp.part, str(step))
    return stl, step


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="block",
        summary="a solid block",
        visual_description="a plain rectangular block",
        overall_envelope_mm="20 x 20 x 10",
        features=[
            BriefFeature(
                name="body", purpose="bulk mass", approx_dims_mm="20 x 20 x 10", method="box"
            )
        ],
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


def _validate(
    box_part: tuple[Path, Path], mode: FabricationMode = FabricationMode.PRINT_3D
) -> object:
    stl, step = box_part
    return validate_authored_part(stl, step, _brief(), fabrication_mode=mode)


def _fixture_outcome() -> SliceOutcome:
    # ok slice whose gcode has a support block (+ a 10x10 silhouette vs the 20x20 mesh) -> would-gate
    return SliceOutcome(ok=True, gcode_path=_FIXTURE, warnings=["WARNING: overhang detected"])


def test_flag_off_never_slices(
    box_part: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FORGE_SLICER_GATE", raising=False)

    def _boom(*_a: object, **_k: object) -> SliceOutcome:
        raise AssertionError("slice_part must not be called when the gate is off")

    monkeypatch.setattr(slicer_mod, "slice_part", _boom)
    val = _validate(box_part)
    assert val.passed is True  # type: ignore[attr-defined]
    assert not any("SLICER" in i for i in val.issues)  # type: ignore[attr-defined]


def test_website_lane_is_report_only(
    box_part: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_SLICER_GATE", "1")
    monkeypatch.delenv("FORGE_CELL_PROFILE", raising=False)
    monkeypatch.setattr(slicer_mod, "slice_part", lambda *_a, **_k: _fixture_outcome())
    val = _validate(box_part)
    assert any("SLICER GATE" in i for i in val.issues)  # surfaced  # type: ignore[attr-defined]
    assert val.passed is True  # but NOT gated on the website lane  # type: ignore[attr-defined]


def test_cell_lane_gates_passed(
    box_part: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_SLICER_GATE", "1")
    monkeypatch.setenv("FORGE_CELL_PROFILE", "1")
    monkeypatch.setattr(slicer_mod, "slice_part", lambda *_a, **_k: _fixture_outcome())
    val = _validate(box_part)
    assert any("SLICER GATE" in i for i in val.issues)  # type: ignore[attr-defined]
    assert val.passed is False  # fail-closed on the cell lane  # type: ignore[attr-defined]


def test_cell_lane_missing_binary_refuses(
    box_part: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_SLICER_GATE", "1")
    monkeypatch.setenv("FORGE_CELL_PROFILE", "1")
    monkeypatch.setattr(
        slicer_mod,
        "slice_part",
        lambda *_a, **_k: SliceOutcome(ok=False, error="slicer unavailable"),
    )
    val = _validate(box_part)
    assert val.passed is False  # a slice that cannot run REFUSES on the cell lane
    assert any("UNVERIFIABLE" in i and "REFUSED" in i for i in val.issues)  # type: ignore[attr-defined]


def test_website_lane_missing_binary_is_report_only(
    box_part: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_SLICER_GATE", "1")
    monkeypatch.delenv("FORGE_CELL_PROFILE", raising=False)
    monkeypatch.setattr(
        slicer_mod,
        "slice_part",
        lambda *_a, **_k: SliceOutcome(ok=False, error="slicer unavailable"),
    )
    val = _validate(box_part)
    assert val.passed is True  # unverifiable is report-only on the website lane
    assert any("UNVERIFIABLE" in i for i in val.issues)  # type: ignore[attr-defined]


def test_general_mode_never_slices(
    box_part: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_SLICER_GATE", "1")

    def _boom(*_a: object, **_k: object) -> SliceOutcome:
        raise AssertionError("the slicer gate is print-lane only; GENERAL must not slice")

    monkeypatch.setattr(slicer_mod, "slice_part", _boom)
    val = _validate(box_part, mode=FabricationMode.GENERAL)
    assert not any("SLICER" in i for i in val.issues)  # type: ignore[attr-defined]
