"""Hand-authored NEMA17 -> 2020 adapter plate — bench-003 deterministic reference.

The bench-003 equivalent of bench-001's NEMA17 mount and bench-002's Pi 4 case:
a known-good part, built by hand, that compiles and passes the validator with
zero new validator code. ADR-0008 selects this as the third pattern; this file
is the deterministic reference the future `AdapterPlateInterpreter` must
reproduce, and the fixture the bench harness validates.

Scope (per the task gating in ADR-0008): the flat adapter only — one plate
hosting two known interfaces. NO new `Pattern` enum entry, NO `LayoutSpec`
schema, NO interpreter. Those land in a later commit only if this reference is
green.

Geometry (centred at the XY origin, plate sitting on the bed at Z=0,
prints flat):

- Plate: 65 mm (X) x 45 mm (Y) x 4 mm (Z), Z = [0, 4].
- Motor face (`nema17_face`, normal +Z) centred at (+11, 0, 4):
  - 4 M3 clearance holes on the 31 mm bolt square (radii from the shared
    `dsl/interfaces/nema17` facts),
  - central 22 mm shaft clearance bore.
- Extrusion mount (`extrusion_2020_slot`, normal -Z) centred at (-22, 0, 4):
  - 2 M5 T-nut clearance holes spaced 20 mm along Y (radius from
    `dsl/interfaces/m5`).

Lands (every hole keeps >= 4 mm to the nearest plate edge and >= 4 mm to its
neighbours in-plane):

- Motor M3 outermost (X = 26.5 + 1.6 = 28.1) -> plate +X edge (32.5): 4.4 mm.
- Motor M3 outermost (Y = 15.5 + 1.6 = 17.1) -> plate +/-Y edge (22.5): 5.4 mm.
- T-nut outermost (X = -22 - 2.6 = -24.6) -> plate -X edge (-32.5): 7.9 mm.
- M3-to-M3 (adjacent corners 31 mm apart): edge-to-edge 27.8 mm.
- Shaft bore (r=11) to nearest M3 (corner-distance 21.92 mm): edge-to-edge
  9.32 mm.
- T-nut-to-T-nut (20 mm centres): edge-to-edge 14.8 mm.
- Leftmost M3 edge (X=-6.1) to nearest T-nut edge (X=-19.4): 13.3 mm.

Run with:
    uv run python examples/handwritten_nema17_2020_adapter.py

NOTE on placer reuse: ``layout/interfaces.py`` already provides
``place_nema17_face`` (used as-is here; it drills along +Z, which is exactly
what a flat plate needs) and ``place_m5_pair`` (currently hardcoded to drill
along +Y for the bench-001 upright). The M5 pair on this flat plate is drilled
along -Z, so this reference inlines two Cylinder ops rather than calling the
helper. Generalising ``place_m5_pair`` to an arbitrary axis is in-scope for the
*interpreter* commit (it is the "second consumer" the placer's docstring already
defers to); the reference deliberately stays helper-free so it does not have to
change when the helper does.
"""

from __future__ import annotations

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM
from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.interfaces import place_nema17_face
from forge_mind.types import Vec3

# Plate body.
PLATE_W = 65.0  # X span
PLATE_D = 45.0  # Y span
PLATE_H = 4.0  # Z thickness (also the min-wall everywhere)

# Interface centres (in-plane).
MOTOR_CX = 11.0
TNUT_CX = -22.0
TNUT_SPACING = 20.0  # along Y, the extrusion's slot direction in this layout

# Envelope target from prompt.txt / ground_truth.json.
MAX_BBOX = Vec3(x=70.0, y=50.0, z=8.0)
MIN_WALL = 4.0


def build_adapter_plate_reference(
    *,
    motor_center_x: float = MOTOR_CX,
    tnut_center_x: float = TNUT_CX,
    tnut_spacing: float = TNUT_SPACING,
) -> PartProgram:
    """Return the hand-authored NEMA17 -> 2020 adapter-plate PartProgram.

    Defaults build the canonical good part. Overrides exist so tests can build
    failure cases (wrong T-nut spacing for the extrusion checker, etc.) without
    duplicating the geometry.
    """
    plate_top_z = PLATE_H  # plate spans Z = [0, PLATE_H]
    plate_mid_z = PLATE_H / 2.0

    ops: list[Operation] = [
        Box(
            op_id="plate",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=plate_mid_z),
            width=PLATE_W,
            depth=PLATE_D,
            height=PLATE_H,
        )
    ]

    # NEMA 17 face (4x M3 corners + central shaft bore), drilled +Z through the
    # plate. The helper reads the shared NEMA 17 facts so the geometry that gets
    # built and the geometry the checker validates are sourced from the same
    # constants.
    ops.extend(
        place_nema17_face(
            center=Vec3(x=motor_center_x, y=0.0, z=plate_mid_z),
            thickness=PLATE_H,
        )
    )

    # 2x M5 T-nut clearance holes drilled -Z (axis along Z, no rotation), spaced
    # along Y at (-tnut_spacing/2, +tnut_spacing/2). Inlined rather than going
    # through place_m5_pair: that helper is currently hardcoded to drill +Y for
    # bench-001's upright. Axis-generalising it is in-scope for the interpreter
    # commit, NOT for this reference.
    half = tnut_spacing / 2.0
    for op_id, sy in (("ext_bolt_back", -1.0), ("ext_bolt_front", 1.0)):
        ops.append(
            Cylinder(
                op_id=op_id,
                mode=OpMode.SUBTRACT,
                position=Vec3(x=tnut_center_x, y=sy * half, z=plate_mid_z),
                radius=M5_CLEARANCE_RADIUS_MM,
                height=PLATE_H,
            )
        )

    interfaces = [
        InterfaceDeclaration(
            name="motor_face",
            interface_type=InterfaceType.NEMA17_FACE,
            location=Vec3(x=motor_center_x, y=0.0, z=plate_top_z),
            normal=Vec3(x=0.0, y=0.0, z=1.0),
        ),
        InterfaceDeclaration(
            name="extrusion_face",
            interface_type=InterfaceType.EXTRUSION_2020_SLOT,
            location=Vec3(x=tnut_center_x, y=0.0, z=0.0),
            normal=Vec3(x=0.0, y=0.0, z=-1.0),
            # No `spacing` param override -> the checker uses its 20 mm default
            # (matches the canonical part). A build with non-default geometric
            # spacing then trips m5_spacing, which is what we want.
        ),
    ]

    metadata = Metadata(
        name="nema17_2020_adapter_reference",
        units="mm",
        description=(
            "bench-003 deterministic reference: flat NEMA17 -> 2020 adapter plate "
            "(ADR-0008). Two known interfaces composed on one plate, both checks "
            "reused from bench-001."
        ),
    )
    constraints = Constraints(
        min_wall_thickness=MIN_WALL,
        max_overhang_deg=45.0,
        max_bbox=MAX_BBOX,
    )
    return PartProgram(
        metadata=metadata, operations=ops, interfaces=interfaces, constraints=constraints
    )


def main() -> None:
    from pathlib import Path

    from forge_mind.compiler.compile import compile_program
    from forge_mind.validator.run import validate

    program = build_adapter_plate_reference()
    example_dir = Path(__file__).parent
    repo_root = example_dir.parent
    fixture_path = (
        repo_root / "tests" / "fixtures" / "handwritten" / "nema17_2020_adapter_good.partdsl.json"
    )
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(program.model_dump_json(indent=2) + "\n", encoding="utf-8")

    model = compile_program(program, example_dir)
    report = validate(model, program)
    print(f"wrote {fixture_path.relative_to(repo_root)}")
    print(f"bbox: {model.bounding_box.size.as_tuple()}")
    print(f"OVERALL: {report.overall.value.upper()}")
    for c in report.checks:
        if c.status.value != "pass":
            print(f"  [{c.status.value.upper()}] {c.id} -> {c.message}")


if __name__ == "__main__":
    main()
