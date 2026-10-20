"""Placement helpers (A2) compile to valid geometry. These COMPILE solids, so they are marked
@pytest.mark.slow and excluded from the fast spec suite (run in the full ``-m 'not llm'`` gate)."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.layout.hardware import (
    place_bearing_seat,
    place_captive_nut_pocket,
    place_clearance_hole,
    place_counterbored_hole,
    place_heat_set_boss,
    place_magnet_pocket,
)
from forge_mind.types import Vec3
from forge_mind.validator.run import validate

pytestmark = pytest.mark.slow

_BASE = Box(
    op_id="base",
    mode=OpMode.ADD,
    position=Vec3(x=0.0, y=0.0, z=0.0),
    width=80.0,
    depth=60.0,
    height=12.0,
)


def test_clearance_and_counterbore_compile(tmp_path: Path) -> None:
    ops = place_clearance_hole("M4", Vec3(x=-20.0, y=0.0, z=0.0), 12.0, op_id="m4")
    ops += place_counterbored_hole("M5", Vec3(x=20.0, y=0.0, z=0.0), 12.0, op_id_prefix="m5cb")
    model = compile_program(
        PartProgram(metadata=Metadata(name="holes"), operations=[_BASE, *ops]), tmp_path
    )
    assert model.part.volume > 0
    assert model.part.volume < 80.0 * 60.0 * 12.0  # holes removed material


def test_boss_seat_pocket_compile_one_body(tmp_path: Path) -> None:
    ops = place_heat_set_boss("M3", Vec3(x=-25.0, y=0.0, z=6.0), op_id_prefix="ins")
    ops += place_bearing_seat("608", Vec3(x=10.0, y=0.0, z=4.0), op_id="brg")
    ops += place_captive_nut_pocket(
        "M4", Vec3(x=28.0, y=0.0, z=0.0), op_id_prefix="nut", bolt_clearance_depth=12.0
    )
    ops += place_magnet_pocket(
        "6x3", Vec3(x=0.0, y=20.0, z=5.0), op_id="mag"
    )  # opens through the +Z face
    program = PartProgram(metadata=Metadata(name="features"), operations=[_BASE, *ops])
    model = compile_program(program, tmp_path)
    report = validate(model, program)
    sb = report.check("mesh.single_body")
    assert sb is not None and sb.status.value == "pass"  # boss + cuts fuse into one body
    assert model.part.volume > 0
