"""Reviser loop with hand-authored patches (real compile+validate, no LLM).

Each test builds a part that fails exactly one dominant check, hands the loop a
deterministic ``propose`` function, and asserts the repaired part passes. This is
the patch-only revision mechanism proven end to end on real geometry.
"""

from __future__ import annotations

from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.revision.patch import Patch
from forge_mind.revision.reviser import ProposeResult, revise
from forge_mind.types import Vec3
from forge_mind.validator.report import ValidationReport

CAP = (60.0, 60.0, 40.0)


def _program(
    operations: list[Box],
    *,
    min_wall: float = 4.0,
    max_overhang: float = 45.0,
    max_bbox: tuple[float, float, float] = (60.0, 60.0, 40.0),
    orient: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> PartProgram:
    return PartProgram(
        metadata=Metadata(
            name="fixture",
            print_orientation=Vec3(x=orient[0], y=orient[1], z=orient[2]),
        ),
        operations=operations,
        constraints=Constraints(
            min_wall_thickness=min_wall,
            max_overhang_deg=max_overhang,
            max_bbox=Vec3(x=max_bbox[0], y=max_bbox[1], z=max_bbox[2]),
        ),
    )


def _validate(program: PartProgram, work: Path) -> ValidationReport:
    from forge_mind.validator.run import validate

    model = compile_program(program, work)
    return validate(model, program)


def _box(op_id: str, w: float, d: float, h: float, *, x=0.0, y=0.0, z=0.0) -> Box:
    return Box(
        op_id=op_id, mode=OpMode.ADD, position=Vec3(x=x, y=y, z=z), width=w, depth=d, height=h
    )


def test_repairs_thin_wall_by_thickening(tmp_path: Path) -> None:
    prog = _program([_box("slab", 40, 40, 2)])
    report = _validate(prog, tmp_path / "start")
    assert "features.min_wall_thickness" in report.failed_check_ids()

    def propose(p: PartProgram, r: ValidationReport) -> ProposeResult:
        h = p.operations[0].height
        return ProposeResult(
            patches=[
                Patch(
                    op_id="slab",
                    field_path="height",
                    old_value=h,
                    new_value=6.0,
                    reason_check_id="features.min_wall_thickness",
                    confidence=0.9,
                    explanation="thicken the slab past the 4mm minimum",
                )
            ]
        )

    result = revise(prog, report, propose, output_dir=tmp_path / "rev", envelope_cap_mm=CAP)
    assert result.improved
    assert result.end_passed
    assert "features.min_wall_thickness" not in result.end_failed_check_ids


def test_repairs_overhang_by_reorienting(tmp_path: Path) -> None:
    # A T-step: a narrow lower post and a wide upper cap. The cap's underside
    # overhangs in the +Z print orientation, but flipping to -Z puts the full
    # 40x40 cap top on the bed and removes the overhang.
    prog = _program([_box("lower", 20, 40, 20, z=10.0), _box("upper", 40, 40, 20, z=29.0)])
    report = _validate(prog, tmp_path / "start")
    assert "printability.overhang_max_angle" in report.failed_check_ids()

    def propose(p: PartProgram, r: ValidationReport) -> ProposeResult:
        cur = p.metadata.print_orientation
        return ProposeResult(
            patches=[
                Patch(
                    op_id=None,
                    field_path="metadata.print_orientation",
                    old_value={"x": cur.x, "y": cur.y, "z": cur.z},
                    new_value={"x": 0.0, "y": 0.0, "z": -1.0},
                    reason_check_id="printability.overhang_max_angle",
                    confidence=0.8,
                    explanation="print upside down so the wide cap rests on the bed",
                )
            ]
        )

    result = revise(prog, report, propose, output_dir=tmp_path / "rev", envelope_cap_mm=CAP)
    assert result.improved
    assert result.end_passed
    assert "printability.overhang_max_angle" not in result.end_failed_check_ids


def test_repairs_bbox_then_wall_in_two_iters(tmp_path: Path) -> None:
    # Fails BOTH bbox (70 > 60) and wall (2 < 4): exercises multi-step revision.
    prog = _program([_box("slab", 70, 40, 2)])
    report = _validate(prog, tmp_path / "start")
    failed = set(report.failed_check_ids())
    assert {"mesh.bbox_within_max", "features.min_wall_thickness"} <= failed

    def propose(p: PartProgram, r: ValidationReport) -> ProposeResult:
        op = p.operations[0]
        failing = set(r.failed_check_ids())
        if "features.min_wall_thickness" in failing:
            return ProposeResult(
                patches=[
                    Patch(
                        op_id="slab",
                        field_path="height",
                        old_value=op.height,
                        new_value=6.0,
                        reason_check_id="features.min_wall_thickness",
                        confidence=0.9,
                        explanation="thicken",
                    )
                ]
            )
        return ProposeResult(
            patches=[
                Patch(
                    op_id="slab",
                    field_path="width",
                    old_value=op.width,
                    new_value=55.0,
                    reason_check_id="mesh.bbox_within_max",
                    confidence=0.9,
                    explanation="shrink to fit the envelope",
                )
            ]
        )

    result = revise(
        prog, report, propose, output_dir=tmp_path / "rev", envelope_cap_mm=CAP, max_iters=3
    )
    assert result.end_passed
    assert result.iterations == 2
    assert 1 <= result.iterations <= 3


def test_anti_thrash_stops_on_no_improvement(tmp_path: Path) -> None:
    # A patch that does not clear the wall failure does not improve the score, so
    # the loop must stop and return the (still failing) best == start.
    prog = _program([_box("slab", 40, 40, 2)])
    report = _validate(prog, tmp_path / "start")

    def propose(p: PartProgram, r: ValidationReport) -> ProposeResult:
        h = p.operations[0].height
        return ProposeResult(
            patches=[
                Patch(
                    op_id="slab",
                    field_path="height",
                    old_value=h,
                    new_value=3.0,  # still below 4mm - won't clear the check
                    reason_check_id="features.min_wall_thickness",
                    confidence=0.5,
                    explanation="too-small bump",
                )
            ]
        )

    result = revise(prog, report, propose, output_dir=tmp_path / "rev", envelope_cap_mm=CAP)
    assert not result.end_passed
    assert result.iterations == 1
    assert result.steps[-1].stop_reason == "no score improvement"


def test_empty_proposal_stops_immediately(tmp_path: Path) -> None:
    prog = _program([_box("slab", 40, 40, 2)])
    report = _validate(prog, tmp_path / "start")

    def propose(p: PartProgram, r: ValidationReport) -> ProposeResult:
        return ProposeResult(patches=[])

    result = revise(prog, report, propose, output_dir=tmp_path / "rev", envelope_cap_mm=CAP)
    assert not result.end_passed
    assert result.steps[-1].stop_reason == "no patches proposed"
