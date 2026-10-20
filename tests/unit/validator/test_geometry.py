"""Unit tests for B-rep feature extraction against a known compiled part."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.types import Rotation, Vec3
from forge_mind.validator.geometry import extract_cylinders, extract_planes, planar_distance


@pytest.fixture(scope="module")
def plate_with_two_holes(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    program = PartProgram(
        metadata=Metadata(name="two_holes"),
        operations=[
            Box(op_id="plate", width=40.0, depth=20.0, height=4.0, position=Vec3(z=2.0)),
            Cylinder(
                op_id="h1",
                mode=OpMode.SUBTRACT,
                radius=1.6,
                height=4.0,
                position=Vec3(x=-10.0, z=2.0),
            ),
            Cylinder(
                op_id="h2",
                mode=OpMode.SUBTRACT,
                radius=1.6,
                height=4.0,
                position=Vec3(x=10.0, z=2.0),
            ),
        ],
    )
    return compile_program(program, tmp_path_factory.mktemp("geom"))


def test_extract_cylinders_finds_both_holes(plate_with_two_holes) -> None:  # type: ignore[no-untyped-def]
    cyls = extract_cylinders(plate_with_two_holes.part)
    assert len(cyls) == 2
    for c in cyls:
        assert math.isclose(c.radius, 1.6, abs_tol=1e-3)
        # Axis is along Z (the holes go through the plate thickness).
        assert c.axis_is_parallel_to((0.0, 0.0, 1.0))


def test_cylinder_axis_points_are_20mm_apart(plate_with_two_holes) -> None:  # type: ignore[no-untyped-def]
    cyls = extract_cylinders(plate_with_two_holes.part)
    pts = [c.axis_point for c in cyls]
    d = planar_distance(pts[0], pts[1], (0.0, 0.0, 1.0))
    assert math.isclose(d, 20.0, abs_tol=0.05)


def test_extract_planes_reports_top_and_bottom(plate_with_two_holes) -> None:  # type: ignore[no-untyped-def]
    planes = extract_planes(plate_with_two_holes.part)
    up_faces = [p for p in planes if p.normal[2] > 0.99]
    down_faces = [p for p in planes if p.normal[2] < -0.99]
    assert up_faces and down_faces


def test_axis_parallel_handles_antiparallel() -> None:
    program = PartProgram(
        metadata=Metadata(name="y_hole"),
        operations=[
            Box(op_id="plate", width=20.0, depth=4.0, height=20.0, position=Vec3(z=10.0)),
            Cylinder(
                op_id="h",
                mode=OpMode.SUBTRACT,
                radius=2.6,
                height=4.0,
                position=Vec3(z=10.0),
                rotation=Rotation(x=90.0),
            ),
        ],
    )
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        model = compile_program(program, Path(d))
        cyls = extract_cylinders(model.part)
        assert len(cyls) == 1
        # Axis along Y; parallel check accepts both +Y and -Y.
        assert cyls[0].axis_is_parallel_to((0.0, 1.0, 0.0))
        assert cyls[0].axis_is_parallel_to((0.0, -1.0, 0.0))
