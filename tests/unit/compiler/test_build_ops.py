"""Per-primitive compile tests.

Each test builds ONE op alone and verifies the resulting build123d Part has
the expected bbox / volume / center. This isolates compiler issues from
multi-op composition issues.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from forge_mind.compiler.build_ops.box import build_box
from forge_mind.compiler.build_ops.computed import build_computed_solid
from forge_mind.compiler.build_ops.cylinder import build_cylinder
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, ComputedSolid, Cylinder, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.types import Rotation, Vec3

# ---------------------------------------------------------------------------
# Box op
# ---------------------------------------------------------------------------


def test_build_box_default_position() -> None:
    op = Box(op_id="b", width=10.0, depth=20.0, height=4.0)
    part = build_box(op)

    bb = part.bounding_box()
    assert math.isclose(bb.size.X, 10.0, abs_tol=1e-6)
    assert math.isclose(bb.size.Y, 20.0, abs_tol=1e-6)
    assert math.isclose(bb.size.Z, 4.0, abs_tol=1e-6)
    # Box default is centered at origin.
    assert math.isclose(bb.min.X, -5.0, abs_tol=1e-6)
    assert math.isclose(bb.min.Z, -2.0, abs_tol=1e-6)


def test_build_box_translation() -> None:
    op = Box(
        op_id="b",
        width=4.0,
        depth=4.0,
        height=4.0,
        position=Vec3(x=10.0, y=20.0, z=30.0),
    )
    part = build_box(op)
    bb = part.bounding_box()
    # Centered on position.
    assert math.isclose((bb.min.X + bb.max.X) / 2, 10.0, abs_tol=1e-6)
    assert math.isclose((bb.min.Y + bb.max.Y) / 2, 20.0, abs_tol=1e-6)
    assert math.isclose((bb.min.Z + bb.max.Z) / 2, 30.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Cylinder op
# ---------------------------------------------------------------------------


def test_build_cylinder_default_axis_z() -> None:
    op = Cylinder(op_id="c", radius=3.0, height=10.0)
    part = build_cylinder(op)
    bb = part.bounding_box()
    # Default axis = Z. So Z extent = height; X/Y extent = 2 * radius.
    assert math.isclose(bb.size.Z, 10.0, abs_tol=1e-6)
    assert math.isclose(bb.size.X, 6.0, abs_tol=1e-6)
    assert math.isclose(bb.size.Y, 6.0, abs_tol=1e-6)


def test_build_cylinder_rotated_to_y_axis() -> None:
    # Rot(X=90) takes the Z axis into the Y direction.
    op = Cylinder(
        op_id="c",
        radius=3.0,
        height=10.0,
        rotation=Rotation(x=90.0),
    )
    part = build_cylinder(op)
    bb = part.bounding_box()
    # After rotation, the cylinder axis is along Y.
    assert math.isclose(bb.size.Y, 10.0, abs_tol=1e-3)
    assert math.isclose(bb.size.X, 6.0, abs_tol=1e-3)
    assert math.isclose(bb.size.Z, 6.0, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# ComputedSolid op (ADR-0019) — the faceted_strut builder
# ---------------------------------------------------------------------------


def test_build_faceted_strut_hex() -> None:
    # A straight hex prism: length along X, across-flats = size*sqrt(3) on Z (flat-bottom),
    # across-corners = 2*size on Y. Obtuse (120 deg) dihedrals -> no acute slivers.
    op = ComputedSolid(op_id="s", kind="faceted_strut", section="hexagonal", size=6.0, length=80.0)
    part = build_computed_solid(op)
    bb = part.bounding_box()
    assert math.isclose(bb.size.X, 80.0, abs_tol=1e-3)
    assert math.isclose(bb.size.Z, 6.0 * math.sqrt(3.0), abs_tol=0.05)  # across-flats, flat-bottom
    assert math.isclose(bb.size.Y, 12.0, abs_tol=0.05)  # across-corners = 2*size
    assert part.volume > 0


def test_build_faceted_strut_rect() -> None:
    op = ComputedSolid(
        op_id="s", kind="faceted_strut", section="rectangular", size=6.0, length=80.0
    )
    part = build_computed_solid(op)
    bb = part.bounding_box()
    assert math.isclose(bb.size.X, 80.0, abs_tol=1e-3)
    assert part.volume > 0


def test_build_faceted_strut_translation() -> None:
    op = ComputedSolid(
        op_id="s",
        kind="faceted_strut",
        section="hexagonal",
        size=5.0,
        length=40.0,
        position=Vec3(x=10.0, y=20.0, z=30.0),
    )
    part = build_computed_solid(op)
    bb = part.bounding_box()
    assert math.isclose((bb.min.X + bb.max.X) / 2, 10.0, abs_tol=1e-3)
    assert math.isclose((bb.min.Y + bb.max.Y) / 2, 20.0, abs_tol=1e-3)
    assert math.isclose((bb.min.Z + bb.max.Z) / 2, 30.0, abs_tol=1e-3)


def test_compile_computed_strut(tmp_path: Path) -> None:
    program = PartProgram(
        metadata=Metadata(name="hex_strut"),
        operations=[
            ComputedSolid(
                op_id="s", kind="faceted_strut", section="hexagonal", size=6.0, length=80.0
            )
        ],
    )
    model = compile_program(program, tmp_path)
    assert model.stl_path.exists()
    assert model.part.volume > 0
    assert "s" in model.topology_index.op_ids()


# ---------------------------------------------------------------------------
# Compiler — minimal programs
# ---------------------------------------------------------------------------


def test_compile_single_box(tmp_path: Path) -> None:
    program = PartProgram(
        metadata=Metadata(name="just_a_box"),
        operations=[Box(op_id="b", width=10.0, depth=10.0, height=4.0)],
    )
    model = compile_program(program, tmp_path)
    assert model.stl_path.exists()
    assert model.stl_path.name == "just_a_box.stl"
    # 10 * 10 * 4 = 400 mm^3
    assert math.isclose(model.part.volume, 400.0, rel_tol=1e-4)
    assert "b" in model.topology_index.op_ids()


def test_compile_box_minus_cylinder(tmp_path: Path) -> None:
    """A 10x10x4 plate with a 2mm hole through it."""
    program = PartProgram(
        metadata=Metadata(name="plate_with_hole"),
        operations=[
            Box(
                op_id="plate",
                width=10.0,
                depth=10.0,
                height=4.0,
                position=Vec3(x=0.0, y=0.0, z=2.0),
            ),
            Cylinder(
                op_id="hole",
                mode=OpMode.SUBTRACT,
                radius=1.0,
                height=4.0,
                position=Vec3(x=0.0, y=0.0, z=2.0),
            ),
        ],
    )
    model = compile_program(program, tmp_path)
    expected_volume = 10.0 * 10.0 * 4.0 - math.pi * 1.0**2 * 4.0
    assert math.isclose(model.part.volume, expected_volume, rel_tol=1e-3)
    # Both ops tagged.
    assert "plate" in model.topology_index.op_ids()
    assert "hole" in model.topology_index.op_ids()


def test_compile_rejects_subtract_first(tmp_path: Path) -> None:
    # Caught by PartProgram validator; ValidationError raised at model build.
    with pytest.raises(Exception, match="first operation must have mode='add'"):
        PartProgram(
            metadata=Metadata(name="bad"),
            operations=[Cylinder(op_id="c", mode=OpMode.SUBTRACT, radius=1.0, height=1.0)],
        )
