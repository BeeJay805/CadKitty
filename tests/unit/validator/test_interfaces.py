"""Unit tests for interface checkers in isolation, including m5_tnut which the
integration fixtures don't exercise directly.
"""

from __future__ import annotations

from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.types import Rotation, Vec3
from forge_mind.validator.geometry import extract_cylinders
from forge_mind.validator.interfaces.registry import get_interface, known_interface_types
from forge_mind.validator.report import CheckStatus


def test_registry_knows_all_types() -> None:
    types = set(known_interface_types())
    assert types == {
        InterfaceType.NEMA17_FACE.value,
        InterfaceType.EXTRUSION_2020_SLOT.value,
        InterfaceType.M5_TNUT.value,
        InterfaceType.PI4_MOUNT.value,
        InterfaceType.PI4_PORTS.value,
        InterfaceType.FAN_FACE.value,
    }


def _plate_with_single_m5(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A plate with one M5 (5.2mm) hole through Y."""
    program = PartProgram(
        metadata=Metadata(name="m5_plate"),
        operations=[
            Box(op_id="plate", width=20.0, depth=4.0, height=20.0, position=Vec3(z=10.0)),
            Cylinder(
                op_id="m5",
                mode=OpMode.SUBTRACT,
                radius=2.6,
                height=4.0,
                position=Vec3(z=10.0),
                rotation=Rotation(x=90.0),
            ),
        ],
    )
    return compile_program(program, tmp_path)


def test_m5_tnut_passes_with_correct_hole(tmp_path: Path) -> None:
    model = _plate_with_single_m5(tmp_path)
    decl = InterfaceDeclaration(
        name="tnut",
        interface_type=InterfaceType.M5_TNUT,
        location=Vec3(y=2.0, z=10.0),
        normal=Vec3(y=1.0),
    )
    checker = get_interface(InterfaceType.M5_TNUT.value)
    assert checker is not None
    checks, fixes = checker.check(decl, extract_cylinders(model.part))
    assert all(c.status == CheckStatus.PASS for c in checks)
    assert not fixes


def test_m5_tnut_fails_when_hole_too_small(tmp_path: Path) -> None:
    # An M3-sized hole (1.6r) must not satisfy an M5 T-nut interface.
    program = PartProgram(
        metadata=Metadata(name="m3_plate"),
        operations=[
            Box(op_id="plate", width=20.0, depth=4.0, height=20.0, position=Vec3(z=10.0)),
            Cylinder(
                op_id="m3",
                mode=OpMode.SUBTRACT,
                radius=1.6,
                height=4.0,
                position=Vec3(z=10.0),
                rotation=Rotation(x=90.0),
            ),
        ],
    )
    model = compile_program(program, tmp_path)
    decl = InterfaceDeclaration(
        name="tnut",
        interface_type=InterfaceType.M5_TNUT,
        location=Vec3(y=2.0, z=10.0),
        normal=Vec3(y=1.0),
    )
    checker = get_interface(InterfaceType.M5_TNUT.value)
    assert checker is not None
    checks, fixes = checker.check(decl, extract_cylinders(model.part))
    assert any(c.status == CheckStatus.FAIL for c in checks)
    assert fixes  # should suggest adding a correctly-sized hole


def test_unknown_interface_type_returns_none() -> None:
    assert get_interface("not_a_real_interface") is None
