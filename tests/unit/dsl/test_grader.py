"""Unit tests for the DSL grader (compile + validate), no LLM.

We grade the hand-authored fixtures: the good one passes; a broken one fails but
still compiles.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge_mind.dsl.grader import grade_program
from forge_mind.dsl.program import PartProgram

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures" / "handwritten"

REQUIRED = [
    "mesh.watertight",
    "mesh.single_body",
    "features.min_wall_thickness",
    "interfaces.motor_face.bolt_pattern_31mm_square",
    "interfaces.extrusion_face.two_m5_holes",
]


def _program(name: str) -> PartProgram:
    return PartProgram.model_validate(json.loads((FIXTURES / f"{name}.partdsl.json").read_text()))


def test_grade_good_fixture_passes(tmp_path: Path) -> None:
    grade = grade_program(_program("nema17_mount_good"), tmp_path, REQUIRED)
    assert grade.compiled
    assert grade.passed, grade.failures
    assert grade.stl_path is not None


def test_grade_broken_holes_fails_but_compiles(tmp_path: Path) -> None:
    grade = grade_program(_program("nema17_mount_broken_holes"), tmp_path, REQUIRED)
    assert grade.compiled  # geometry is valid, just wrong
    assert not grade.passed
    assert any("bolt_pattern_31mm_square" in f for f in grade.failures)
