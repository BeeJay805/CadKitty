"""Hand-authored fan-mount -> 2020 adapter plates — bench-006 deterministic reference.

The bench-006 Stage-1 foundation (ADR-0011): known-good flat fan-mount plates, built
by hand, that compile and pass the validator -- the fan analogue of bench-003's
NEMA17->2020 adapter. A fan mount reuses the *adapter_plate* shape (a flat plate
composing two face interfaces); only the +Z interface differs (a fan face instead of
a motor face). NO new Pattern enum entry, NO LayoutSpec schema, NO interpreter -- those
are later stages, gated on this reference being green.

Two sizes are built:

- ``build_fan_mount_reference(size="60mm")`` -- the roomy case. 60 mm fan: 50 mm bolt
  square, M3 corners, ~56 mm airflow bore. Lands are comfortable; passes cleanly.
- ``build_fan_mount_reference(size="40mm")`` -- the tight FEASIBILITY PROBE. 40 mm fan:
  32 mm bolt square, M3 corners, ~38 mm airflow bore. The bore-edge-to-bolt land is
  the binding constraint ADR-0011 predicted (see the module-level note below).

Geometry (centred at the XY origin, plate on the bed at Z=0, prints flat):

- Plate: ``plate_w`` (X) x ``plate_d`` (Y) x 4 mm (Z), Z = [0, 4].
- Fan face (``fan_face``, normal +Z) centred at (+fan_cx, 0, top):
  - 4 M3 clearance holes on the fan's bolt square,
  - central airflow bore (open, >= the fan's spec opening).
- Extrusion mount (``extrusion_2020_slot``, normal -Z) centred at (-tnut_cx, 0, top):
  - 2 M5 T-nut clearance holes spaced 20 mm along Y (reuses ``place_m5_pair``,
    ``ALONG_Y_DRILL_Z`` -- the exact bench-003 adapter approach).

FAN-BORE-TO-BOLT LAND (the binding land, computed in ``main``):

- 60 mm: bolt corner radial 35.36 mm, bore edge 28.0 mm, M3 inner edge 33.76 mm ->
  land ~5.76 mm >= 4 mm. Comfortable.
- 40 mm: bolt corner radial 22.63 mm, bore edge 19.0 mm, M3 inner edge 21.03 mm ->
  land ~2.03 mm. This is *just* over a 2 mm min_wall and BELOW a 4 mm one -- so the
  40 mm reference is built at min_wall = 2 mm (a thinner plate is realistic for a
  small fan). At min_wall = 4 mm the flat 40 mm fan mount at this bore is infeasible;
  see the printed report and the bench-006 plan finding.

Run with:
    uv run python examples/handwritten_fan_2020_adapter.py
"""

from __future__ import annotations

from typing import Literal

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.fan import (
    FAN_40MM_AIRFLOW_BORE_RADIUS_MM,
    FAN_40MM_BOLT_SQUARE_MM,
    FAN_60MM_AIRFLOW_BORE_RADIUS_MM,
    FAN_60MM_BOLT_SQUARE_MM,
    FAN_BOLT_CLEARANCE_RADIUS_MM,
)
from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.interfaces import TNutPair, place_m5_pair
from forge_mind.layout.rules import centered_rect_corners
from forge_mind.types import Vec3

FanSize = Literal["40mm", "60mm"]

# Per-size facts: (bolt_square, airflow_bore_radius, plate_w, plate_d, fan_cx,
# tnut_cx, min_wall, max_bbox). The 60 mm plate is roomy; the 40 mm plate is sized
# tight to the binding fan-bore-to-bolt land at a 2 mm wall.
_SIZES: dict[FanSize, dict[str, float]] = {
    "60mm": {
        "bolt_square": FAN_60MM_BOLT_SQUARE_MM,  # 50
        "airflow_bore_r": FAN_60MM_AIRFLOW_BORE_RADIUS_MM,  # 28
        "plate_w": 104.0,
        "plate_d": 64.0,
        "fan_cx": 20.0,
        "tnut_cx": -30.0,
        "min_wall": 4.0,
        "bbox_x": 108.0,
        "bbox_y": 68.0,
        "bbox_z": 8.0,
    },
    "40mm": {
        "bolt_square": FAN_40MM_BOLT_SQUARE_MM,  # 32
        "airflow_bore_r": FAN_40MM_AIRFLOW_BORE_RADIUS_MM,  # 19
        "plate_w": 82.0,
        "plate_d": 42.0,
        "fan_cx": 20.0,
        "tnut_cx": -30.0,
        "min_wall": 2.0,  # see module note: 4 mm land is infeasible at this bore
        "bbox_x": 86.0,
        "bbox_y": 46.0,
        "bbox_z": 8.0,
    },
}

TNUT_SPACING = 20.0  # along Y, the extrusion's slot direction in this layout


def _place_fan_face(
    *, center: Vec3, bolt_square: float, airflow_bore_r: float, thickness: float
) -> list[Operation]:
    """The fan face as SUBTRACT cylinders drilled +Z: 4 M3 corners on the bolt
    square + a central airflow bore. The fan analogue of ``place_nema17_face``;
    inlined here (not promoted to ``layout/interfaces.py``) because the interpreter
    that needs a shared placer is a later stage -- the reference stays helper-light,
    exactly as the bench-003 adapter reference did for ``place_m5_pair``.
    """
    ops: list[Operation] = []
    for tag, cx, cy in centered_rect_corners(bolt_square, bolt_square):
        ops.append(
            Cylinder(
                op_id=f"fan_bolt_{tag}",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=center.x + cx, y=center.y + cy, z=center.z),
                radius=FAN_BOLT_CLEARANCE_RADIUS_MM,
                height=thickness,
            )
        )
    ops.append(
        Cylinder(
            op_id="airflow_bore",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=center.x, y=center.y, z=center.z),
            radius=airflow_bore_r,
            height=thickness,
        )
    )
    return ops


def build_fan_mount_reference(
    *,
    size: FanSize = "60mm",
    tnut_spacing: float = TNUT_SPACING,
    airflow_bore_r: float | None = None,
    bolt_square: float | None = None,
) -> PartProgram:
    """Return the hand-authored fan-mount -> 2020 adapter-plate PartProgram.

    Defaults build the canonical good part for ``size``. ``airflow_bore_r`` and
    ``bolt_square`` overrides let tests build broken variants (oversized bore that
    eats the bolt land; wrong-size square) without duplicating the geometry.
    """
    s = _SIZES[size]
    plate_h = s["min_wall"]
    plate_mid_z = plate_h / 2.0
    eff_square = bolt_square if bolt_square is not None else s["bolt_square"]
    eff_bore_r = airflow_bore_r if airflow_bore_r is not None else s["airflow_bore_r"]

    ops: list[Operation] = [
        Box(
            op_id="plate",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=plate_mid_z),
            width=s["plate_w"],
            depth=s["plate_d"],
            height=plate_h,
        )
    ]
    ops.extend(
        _place_fan_face(
            center=Vec3(x=s["fan_cx"], y=0.0, z=plate_mid_z),
            bolt_square=eff_square,
            airflow_bore_r=eff_bore_r,
            thickness=plate_h,
        )
    )
    # 2x M5 T-nut clearance holes drilled -Z, spaced along Y -- the exact bench-003
    # adapter approach (place_m5_pair, ALONG_Y_DRILL_Z), reused unchanged.
    ops.extend(
        place_m5_pair(
            center=Vec3(x=s["tnut_cx"], y=0.0, z=plate_mid_z),
            spacing=tnut_spacing,
            thickness=plate_h,
            orientation=TNutPair.ALONG_Y_DRILL_Z,
            left_op_id="ext_bolt_back",
            right_op_id="ext_bolt_front",
        )
    )

    interfaces = [
        InterfaceDeclaration(
            name="fan_face",
            interface_type=InterfaceType.FAN_FACE,
            location=Vec3(x=s["fan_cx"], y=0.0, z=plate_h),
            normal=Vec3(x=0.0, y=0.0, z=1.0),
            # Per-fan-size nominals the checker validates against.
            params={"bolt_square": s["bolt_square"], "airflow_bore_radius": s["airflow_bore_r"]},
        ),
        InterfaceDeclaration(
            name="extrusion_face",
            interface_type=InterfaceType.EXTRUSION_2020_SLOT,
            location=Vec3(x=s["tnut_cx"], y=0.0, z=0.0),
            normal=Vec3(x=0.0, y=0.0, z=-1.0),
        ),
    ]
    metadata = Metadata(
        name=f"fan_{size}_2020_adapter_reference",
        units="mm",
        description=(
            f"bench-006 deterministic reference: flat {size} fan-mount -> 2020 adapter "
            "plate (ADR-0011). Fan face on +Z, 2020 M5 T-nut pair through to -Z; "
            "reuses the adapter_plate shape, one new fan_face interface."
        ),
    )
    constraints = Constraints(
        min_wall_thickness=s["min_wall"],
        max_overhang_deg=45.0,
        max_bbox=Vec3(x=s["bbox_x"], y=s["bbox_y"], z=s["bbox_z"]),
    )
    return PartProgram(
        metadata=metadata, operations=ops, interfaces=interfaces, constraints=constraints
    )


def _report_land(size: FanSize) -> None:
    import math

    s = _SIZES[size]
    bolt_radial = (s["bolt_square"] / 2.0) * math.sqrt(2)
    bolt_inner_edge = bolt_radial - FAN_BOLT_CLEARANCE_RADIUS_MM
    land = bolt_inner_edge - s["airflow_bore_r"]
    print(
        f"  {size}: bolt corner radial {bolt_radial:.2f}mm, M3 inner edge "
        f"{bolt_inner_edge:.2f}mm, bore edge {s['airflow_bore_r']:.2f}mm -> "
        f"bore-to-bolt LAND {land:.2f}mm (min_wall {s['min_wall']:.0f}mm) "
        f"-> {'OK' if land >= s['min_wall'] else 'BELOW MIN_WALL'}"
    )


def main() -> None:
    from pathlib import Path

    from forge_mind.compiler.compile import compile_program
    from forge_mind.validator.run import validate

    example_dir = Path(__file__).parent
    repo_root = example_dir.parent
    fixture_dir = repo_root / "tests" / "fixtures" / "handwritten"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    print("fan-bore-to-bolt land (the binding land):")
    for size in ("60mm", "40mm"):
        _report_land(size)  # type: ignore[arg-type]

    for size in ("60mm", "40mm"):
        program = build_fan_mount_reference(size=size)  # type: ignore[arg-type]
        fixture_path = fixture_dir / f"fan_{size}_2020_adapter_good.partdsl.json"
        fixture_path.write_text(program.model_dump_json(indent=2) + "\n", encoding="utf-8")
        model = compile_program(program, example_dir)
        report = validate(model, program)
        print(f"\n{size}: wrote {fixture_path.relative_to(repo_root)}")
        print(f"  bbox: {model.bounding_box.size.as_tuple()}")
        print(f"  OVERALL: {report.overall.value.upper()}")
        for c in report.checks:
            if c.status.value != "pass":
                print(f"    [{c.status.value.upper()}] {c.id} -> {c.message}")


if __name__ == "__main__":
    main()
