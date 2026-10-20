"""Hand-authored Raspberry Pi 4 open-tray case — bench-002 deterministic reference.

The bench-002 equivalent of the Phase 0 NEMA 17 ground truth: a known-good part,
built by hand, that compiles and passes the validator (including the new
pi4_mount + pi4_ports interface checks). It is the reference the future
Pi4CaseInterpreter must reproduce, and the fixture the checkers are tested
against.

Scope (per the task): open tray only — floor, four standoffs, board clearance,
the two required port openings. No lid, no snap fits, no decorative vents.

Geometry (centred at the XY origin, floor on the bed at Z=0, prints floor-down):
- floor + perimeter walls (2 mm) forming a board-sized cavity,
- 4 standoffs (OD 7 mm) on the 58 x 49 mm M2.5 hole rectangle, each with a blind
  M2.5 clearance hole and a >= 2 mm material land (OD 7, not 6, resolves the
  ADR-0005 standoff-land tension WITHOUT weakening any validator threshold),
- two port openings cut as top-open notches so their top edge does not overhang:
  one in a long wall (USB/Ethernet) and one in a short wall (USB-C / micro-HDMI).

Design notes (honest, for the interpreter):
- Board clearance is 2.0 mm here, not the 0.5 mm of the LayoutSpec fixture. At
  0.5 mm clearance an OD-7 corner standoff sits ~0.5 mm from the wall, a fragile
  near-collision. The interpreter must either merge corner standoffs into the
  walls or widen clearance; this reference uses 2.0 mm for a clean passing part.
"""

from __future__ import annotations

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Vec3

# Pi 4 + case parameters.
MIN_WALL = 2.0
CLEARANCE = 2.0
BOARD_L = 85.0
BOARD_W = 56.0
HOLE_DX = 58.0
HOLE_DY = 49.0
SCREW_R = 1.35  # M2.5 clearance, 2.7mm dia
STANDOFF_R = 3.5  # OD 7.0mm -> land 3.5 - 1.35 = 2.15mm >= MIN_WALL
STANDOFF_H = 4.0
CASE_H = 16.0

_HX = HOLE_DX / 2.0  # 29.0
_HY = HOLE_DY / 2.0  # 24.5
_INNER_L = BOARD_L + 2 * CLEARANCE  # 89
_INNER_W = BOARD_W + 2 * CLEARANCE  # 60
_OUTER_L = _INNER_L + 2 * MIN_WALL  # 93
_OUTER_W = _INNER_W + 2 * MIN_WALL  # 64


def build_pi4_case_reference(
    *,
    hole_dx: float = HOLE_DX,
    standoff_r: float = STANDOFF_R,
    include_short_port: bool = True,
) -> PartProgram:
    """Return the hand-authored Pi 4 open-tray reference PartProgram.

    The defaults build the canonical good part. The overrides exist only so tests
    can construct failure cases (wrong hole spacing, too-thin standoff land, a
    missing port) without duplicating the geometry.
    """
    hx = hole_dx / 2.0
    ops: list[Operation] = [
        # Outer solid block.
        Box(
            op_id="outer",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=CASE_H / 2.0),
            width=_OUTER_L,
            depth=_OUTER_W,
            height=CASE_H,
        ),
        # Hollow the cavity above the floor (floor stays Z[0, MIN_WALL]).
        Box(
            op_id="cavity",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=0.0, y=0.0, z=MIN_WALL + (CASE_H - MIN_WALL) / 2.0),
            width=_INNER_L,
            depth=_INNER_W,
            height=CASE_H - MIN_WALL,
        ),
    ]

    # 4 standoffs (posts) on the hole rectangle.
    for name, sx, sy in (
        ("standoff_nw", -1.0, 1.0),
        ("standoff_ne", 1.0, 1.0),
        ("standoff_sw", -1.0, -1.0),
        ("standoff_se", 1.0, -1.0),
    ):
        ops.append(
            Cylinder(
                op_id=name,
                mode=OpMode.ADD,
                position=Vec3(x=sx * hx, y=sy * _HY, z=MIN_WALL + STANDOFF_H / 2.0),
                radius=standoff_r,
                height=STANDOFF_H,
            )
        )

    # Blind M2.5 clearance hole in each standoff (down to the floor top, open up).
    for name, sx, sy in (
        ("screw_nw", -1.0, 1.0),
        ("screw_ne", 1.0, 1.0),
        ("screw_sw", -1.0, -1.0),
        ("screw_se", 1.0, -1.0),
    ):
        ops.append(
            Cylinder(
                op_id=name,
                mode=OpMode.SUBTRACT,
                position=Vec3(x=sx * hx, y=sy * _HY, z=MIN_WALL + 2.5),
                radius=SCREW_R,
                height=5.0,  # spans Z[2, 7]: blind at the floor, open into the cavity
            )
        )

    # Long-edge port (USB/Ethernet): a top-open notch in the +Y wall.
    ops.append(
        Box(
            op_id="port_long",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=0.0, y=_OUTER_W / 2.0 - MIN_WALL / 2.0, z=11.0),
            width=50.0,
            depth=2.0 * MIN_WALL,  # through the wall
            height=14.0,  # Z[4, 18]: reaches above the wall top -> no overhang
        )
    )
    # Short-edge port (USB-C / micro-HDMI): a top-open notch in the -X wall.
    if include_short_port:
        ops.append(
            Box(
                op_id="port_short",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=-(_OUTER_L / 2.0 - MIN_WALL / 2.0), y=0.0, z=11.0),
                width=2.0 * MIN_WALL,
                depth=44.0,
                height=14.0,
            )
        )

    interfaces = [
        InterfaceDeclaration(
            name="pi4_mount",
            interface_type=InterfaceType.PI4_MOUNT,
            location=Vec3(x=0.0, y=0.0, z=MIN_WALL + STANDOFF_H),
            normal=Vec3(x=0.0, y=0.0, z=1.0),
        ),
        InterfaceDeclaration(
            name="pi4_ports",
            interface_type=InterfaceType.PI4_PORTS,
            location=Vec3(x=0.0, y=0.0, z=CASE_H / 2.0),
            normal=Vec3(x=0.0, y=1.0, z=0.0),
        ),
    ]

    metadata = Metadata(name="pi4_case_reference", units="mm")
    constraints = Constraints(
        min_wall_thickness=MIN_WALL,
        max_overhang_deg=45.0,
        max_bbox=Vec3(x=95.0, y=70.0, z=30.0),
    )
    return PartProgram(
        metadata=metadata, operations=ops, interfaces=interfaces, constraints=constraints
    )


def main() -> None:
    from pathlib import Path

    from forge_mind.compiler.compile import compile_program
    from forge_mind.validator.run import validate

    program = build_pi4_case_reference()
    out = Path(__file__).parent
    (out / "handwritten_pi4_case.partdsl.json").write_text(
        program.model_dump_json(indent=2), encoding="utf-8"
    )
    model = compile_program(program, out)
    report = validate(model, program)
    print(f"bbox: {model.bounding_box.size.as_tuple()}")
    print(f"OVERALL: {report.overall.value.upper()}")
    for c in report.checks:
        if c.status.value != "pass":
            print(f"  [{c.status.value.upper()}] {c.id} -> {c.message}")


if __name__ == "__main__":
    main()
