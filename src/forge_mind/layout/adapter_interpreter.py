"""AdapterPlateInterpreter — AdapterPlateSpec -> PartProgram (ADR-0008).

The third LayoutSpec interpreter, alongside the l_bracket one (bench-001) and
the electronics_tray one (bench-002). Reproduces the exact geometry decisions
the hand-authored adapter-plate reference (examples/handwritten_nema17_2020_
adapter.py) proved against the existing NEMA17 + extrusion-2020 checkers,
parameterised by the spec, with every relational invariant enforced BEFORE
geometry exists.

Flat plate only (no upright, no gusset, no walls). Box + Cylinder primitives
only. Reuses ``place_nema17_face`` as-is (it already drills +Z, which a flat
plate wants) and ``place_m5_pair`` in the new ``ALONG_Y_DRILL_Z`` orientation
the helper grew when this -- its second consumer -- came online (ADR-0006
Stage 2: earn the abstraction with a real second caller).

Key invariants guaranteed BEFORE any geometry is built:
- print orientation is floor-down (+Z up), so the plate prints flat with no
  overhang anywhere (cylindrical hole walls are vertical and ignored by the
  plane-only printability checker);
- every M3 bolt outer edge keeps >= min_wall to the nearest plate edge,
  in both X and Y;
- the shaft bore outer edge keeps >= min_wall to the nearest plate edge;
- every M5 hole outer edge keeps >= min_wall to the nearest plate edge,
  in both X and Y;
- the motor cluster and the T-nut pair never overlap in plan view: the
  closest M3 outer edge keeps >= min_wall to the closest M5 outer edge;
- the part's assembled bbox fits within the spec's envelope.

An infeasible spec is rejected up front with ``LayoutError`` rather than
emitted as a part that would fail validation.
"""

from __future__ import annotations

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM as M5_CLEARANCE_RADIUS
from forge_mind.dsl.interfaces.nema17 import (
    BOLT_PITCH_MM as NEMA17_BOLT_PITCH,
)
from forge_mind.dsl.interfaces.nema17 import (
    M3_CLEARANCE_RADIUS_MM as M3_CLEARANCE_RADIUS,
)
from forge_mind.dsl.interfaces.nema17 import (
    SHAFT_BORE_RADIUS_MM as SHAFT_BORE_RADIUS,
)
from forge_mind.dsl.primitives import Box, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.adapter_spec import AdapterPlateSpec
from forge_mind.layout.interfaces import TNutPair, place_m5_pair, place_nema17_face
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.rules import edge_land
from forge_mind.types import Vec3

# Derived from the shared NEMA17 facts so we cannot drift.
NEMA17_HALF_PITCH = NEMA17_BOLT_PITCH / 2.0  # 15.5
# Outermost in-plane extent of the M3 cluster from the motor centre.
NEMA17_M3_HALF_EXTENT = NEMA17_HALF_PITCH + M3_CLEARANCE_RADIUS  # 17.1

# Adapter-plate-specific constants (used to live on AdapterPlateSpec's
# now-removed `nema17` / `extrusion_2020` sub-specs; hardcoded here to keep
# the LLM-facing schema small enough to fit Anthropic's structured-output
# grammar limit -- see adapter_spec.py docstring for context).
MOTOR_FACE_NAME = "motor_face"
EXTRUSION_FACE_NAME = "extrusion_face"
ADAPTER_PLATE_M5_SPACING_MM = 20.0


def derive_adapter_offsets(envelope_x_mm: float, min_wall_mm: float) -> tuple[float, float]:
    """Smallest legal ``(motor_center_x_mm, extrusion_center_x_mm)`` pair
    given the envelope and min_wall (ADR-0009 Task 1).

    The motor sits at +X, the T-nut at -X by convention. The pair is
    chosen so that:

    1. Both sides consume equal plate half-width:
       ``|motor_x| + NEMA17_M3_HALF_EXTENT == |ext_x| + M5_CLEARANCE_RADIUS``.
       Equivalently, ``|ext_x| = |motor_x| + (17.1 - 2.6) = |motor_x| + 14.5``.
       Equalising the two land budgets minimises the derived plate width
       for a given motor-to-T-nut separation.
    2. The no-overlap rule holds at the floor:
       ``|motor_x| + |ext_x| >= NEMA17_M3_HALF_EXTENT + M5_CLEARANCE_RADIUS +
       min_wall_mm`` (= ``19.7 + min_wall``). Substituting (1) and solving
       for ``|motor_x|`` gives ``|motor_x| >= (5.2 + min_wall) / 2``.

    A small FP epsilon keeps the derived values strictly above the
    boundaries the strict `<` feasibility check would trip on (mirrors
    `derive_plate_dims` / `derive_l_bracket_dims`).

    Raises ``LayoutError`` if the envelope X cannot host the resulting
    plate at all -- the request is geometrically infeasible regardless of
    offset choice. The envelope check duplicates `_check_feasible`'s but
    keeps the helper honest as a standalone callable.
    """
    eps = 1e-6
    no_overlap_floor = NEMA17_M3_HALF_EXTENT + M5_CLEARANCE_RADIUS + min_wall_mm
    # Balanced solution from the docstring derivation.
    motor_abs = (no_overlap_floor - (NEMA17_M3_HALF_EXTENT - M5_CLEARANCE_RADIUS)) / 2.0
    motor_abs = max(motor_abs, 0.0) + eps
    ext_abs = motor_abs + (NEMA17_M3_HALF_EXTENT - M5_CLEARANCE_RADIUS)

    # Resulting plate half-width consumes equal land on both sides.
    half_w = motor_abs + NEMA17_M3_HALF_EXTENT + min_wall_mm
    envelope_half = envelope_x_mm / 2.0
    if half_w > envelope_half + 1e-6:
        raise LayoutError(
            f"adapter_plate envelope X {envelope_x_mm}mm is too small to host "
            f"a NEMA17 face + 2020 T-nut pair with min_wall {min_wall_mm}mm. "
            f"Required envelope X >= {2 * half_w:.2f}mm (smallest balanced layout)."
        )
    return (motor_abs, -ext_abs)


def _resolve_adapter_offsets(spec: AdapterPlateSpec) -> tuple[float, float]:
    """Pick ``(motor_x, ext_x)`` either from explicit spec values or by deriving them.
    Explicit values win when given (backward compat with the hand-authored fixture).
    """
    if spec.motor_center_x_mm is not None and spec.extrusion_center_x_mm is not None:
        return (spec.motor_center_x_mm, spec.extrusion_center_x_mm)
    derived_motor, derived_ext = derive_adapter_offsets(spec.envelope_mm.x, spec.min_wall_mm)
    motor_x = spec.motor_center_x_mm if spec.motor_center_x_mm is not None else derived_motor
    ext_x = spec.extrusion_center_x_mm if spec.extrusion_center_x_mm is not None else derived_ext
    return (motor_x, ext_x)


def derive_plate_dims(spec: AdapterPlateSpec) -> tuple[float, float]:
    """Tightest legal plate footprint given the resolved offsets + min_wall.

    Width is set so the farther-out of (motor cluster, T-nut pair) keeps
    a min_wall land to the nearest plate edge on each side. Depth is set
    so the larger of (motor M3 extent, T-nut M5 + half-spacing) keeps a
    min_wall land to the +/- Y edge.

    This is the "structure derives layout" half of the ADR-0008 perturbation
    fix; with ADR-0009 Task 1 the offsets themselves are also derivable, so
    this function now takes them through `_resolve_adapter_offsets`.
    """
    w = spec.min_wall_mm
    m5_half = ADAPTER_PLATE_M5_SPACING_MM / 2.0
    motor_x, ext_x = _resolve_adapter_offsets(spec)

    # Width: each side needs to clear whatever interface lives there. We
    # use abs() per side rather than max(abs(M), abs(E)) so a layout with
    # both centres on the same side still gets enough room.
    plus_x_extent = max(
        motor_x + NEMA17_M3_HALF_EXTENT,
        motor_x + SHAFT_BORE_RADIUS,
        ext_x + M5_CLEARANCE_RADIUS,
    )
    minus_x_extent = max(
        -motor_x + NEMA17_M3_HALF_EXTENT,
        -motor_x + SHAFT_BORE_RADIUS,
        -ext_x + M5_CLEARANCE_RADIUS,
    )
    # Centred plate: half-width must cover the worst side.
    half_w = max(plus_x_extent, minus_x_extent) + w

    # Depth: motor M3 cluster + T-nut M5 row both centred in Y.
    half_d = max(NEMA17_M3_HALF_EXTENT, m5_half + M5_CLEARANCE_RADIUS) + w

    return (2.0 * half_w, 2.0 * half_d)


def _resolve_plate_dims(spec: AdapterPlateSpec) -> tuple[float, float]:
    """Pick (width, depth) either from the spec's explicit values or by
    deriving them. Explicit values win when given."""
    derived_w, derived_d = derive_plate_dims(spec)
    width = spec.plate_width_mm if spec.plate_width_mm is not None else derived_w
    depth = spec.plate_depth_mm if spec.plate_depth_mm is not None else derived_d
    return (width, depth)


def _check_feasible(spec: AdapterPlateSpec) -> None:
    """Reject specs that cannot be realised with legal lands / envelope."""
    w = spec.min_wall_mm
    motor_x, ext_x = _resolve_adapter_offsets(spec)
    plate_width, plate_depth = _resolve_plate_dims(spec)
    half_w = plate_width / 2.0
    half_d = plate_depth / 2.0
    m5_half = ADAPTER_PLATE_M5_SPACING_MM / 2.0

    # Print orientation is no longer a spec field: adapter_plate always prints
    # flat (+Z up) and the interpreter assumes it. No runtime check required.

    problems: list[str] = []

    # --- NEMA17 cluster vs plate edges (X = plate.width / 2, offset by motor centre).
    if edge_land(half_w, abs(motor_x) + NEMA17_HALF_PITCH, M3_CLEARANCE_RADIUS) < w:
        problems.append(
            f"plate width {plate_width}mm too small: NEMA17 M3 hole-to-edge land < "
            f"min_wall {w}mm at motor X centre {motor_x}mm "
            f"(half-pitch {NEMA17_HALF_PITCH} + M3 r {M3_CLEARANCE_RADIUS} + wall {w})."
        )
    if edge_land(half_d, NEMA17_HALF_PITCH, M3_CLEARANCE_RADIUS) < w:
        problems.append(
            f"plate depth {plate_depth}mm too small: NEMA17 M3 hole-to-edge land < "
            f"min_wall {w}mm in Y."
        )
    # Shaft bore vs plate edges (worst case = motor centre at the side it leans toward).
    if edge_land(half_w, abs(motor_x), SHAFT_BORE_RADIUS) < w:
        problems.append(
            f"plate width {plate_width}mm too small for the {2 * SHAFT_BORE_RADIUS}mm "
            f"shaft bore + {w}mm land at motor X centre {motor_x}mm."
        )
    if edge_land(half_d, 0.0, SHAFT_BORE_RADIUS) < w:
        problems.append(
            f"plate depth {plate_depth}mm too small for the {2 * SHAFT_BORE_RADIUS}mm "
            f"shaft bore + {w}mm land in Y."
        )

    # --- T-nut pair vs plate edges (the pair is spaced along Y, drilled +Z).
    if edge_land(half_w, abs(ext_x), M5_CLEARANCE_RADIUS) < w:
        problems.append(
            f"plate width {plate_width}mm too small: M5 hole-to-edge land < "
            f"min_wall {w}mm at T-nut X centre {ext_x}mm."
        )
    if edge_land(half_d, m5_half, M5_CLEARANCE_RADIUS) < w:
        problems.append(
            f"plate depth {plate_depth}mm too small: M5 row + {w}mm land does not fit "
            f"(half-spacing {m5_half} + M5 r {M5_CLEARANCE_RADIUS} + wall {w})."
        )

    # --- Motor cluster vs T-nut pair: their nearest outer edges must clear
    # min_wall in plan view, otherwise the holes effectively merge.
    motor_inner_x = abs(motor_x - ext_x)
    closest_motor_edge = motor_inner_x - (NEMA17_HALF_PITCH + M3_CLEARANCE_RADIUS)
    if closest_motor_edge - M5_CLEARANCE_RADIUS < w:
        problems.append(
            "motor cluster and T-nut pair overlap or touch: distance "
            f"{motor_inner_x}mm leaves a land of "
            f"{closest_motor_edge - M5_CLEARANCE_RADIUS:.2f}mm < min_wall {w}mm. "
            "Move one centre further away or shrink the spacing."
        )

    # --- Envelope: plate footprint + thickness must fit.
    env = spec.envelope_mm
    if plate_width > env.x + 1e-6:
        problems.append(f"plate width {plate_width}mm exceeds envelope X {env.x}mm.")
    if plate_depth > env.y + 1e-6:
        problems.append(f"plate depth {plate_depth}mm exceeds envelope Y {env.y}mm.")
    if w > env.z + 1e-6:
        problems.append(f"plate thickness {w}mm exceeds envelope Z {env.z}mm.")

    if problems:
        raise LayoutError("infeasible adapter_plate spec: " + "; ".join(problems))


def interpret_adapter_plate(spec: AdapterPlateSpec) -> PartProgram:
    """Deterministically realise an AdapterPlateSpec as a validated PartProgram.

    Raises ``LayoutError`` if the spec is infeasible (checked before geometry).
    The emitted operation order + ids mirror the hand-authored bench-003
    reference (``examples/handwritten_nema17_2020_adapter.py``) so the
    interpreter and the reference produce equivalent parts.

    When ``fan_component_id`` is set the +Z face hosts an axial fan instead of a
    NEMA 17 motor (bench-006); the fan path lives in ``layout/fan_mount.py`` and
    reuses the -Z M5 mount. The motor path below is unchanged.
    """
    if spec.fan_component_id is not None:
        from forge_mind.layout.fan_mount import interpret_fan_mount

        return interpret_fan_mount(spec)

    _check_feasible(spec)

    w = spec.min_wall_mm
    motor_x, ext_x = _resolve_adapter_offsets(spec)
    plate_width, plate_depth = _resolve_plate_dims(spec)
    plate_mid_z = w / 2.0
    plate_top_z = w

    ops: list[Operation] = [
        Box(
            op_id="plate",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=plate_mid_z),
            width=plate_width,
            depth=plate_depth,
            height=w,
        )
    ]

    # NEMA 17 motor face: 4 M3 + central shaft bore, drilled +Z through the
    # plate. The helper already drills +Z so it works as-is for the flat
    # orientation; the same shared facts feed both placement and the checker.
    ops.extend(
        place_nema17_face(
            center=Vec3(x=motor_x, y=0.0, z=plate_mid_z),
            thickness=w,
        )
    )

    # 2x M5 T-nut clearance holes drilled +Z (the new ALONG_Y_DRILL_Z
    # orientation grown for this, the helper's second consumer). The op id
    # labels are back/front because the pair is spaced along Y.
    ops.extend(
        place_m5_pair(
            center=Vec3(x=ext_x, y=0.0, z=plate_mid_z),
            spacing=ADAPTER_PLATE_M5_SPACING_MM,
            thickness=w,
            orientation=TNutPair.ALONG_Y_DRILL_Z,
            left_op_id="ext_bolt_back",
            right_op_id="ext_bolt_front",
        )
    )

    interfaces = [
        InterfaceDeclaration(
            name=MOTOR_FACE_NAME,
            interface_type=InterfaceType.NEMA17_FACE,
            location=Vec3(x=motor_x, y=0.0, z=plate_top_z),
            normal=Vec3(x=0.0, y=0.0, z=1.0),
        ),
        InterfaceDeclaration(
            name=EXTRUSION_FACE_NAME,
            interface_type=InterfaceType.EXTRUSION_2020_SLOT,
            location=Vec3(x=ext_x, y=0.0, z=0.0),
            normal=Vec3(x=0.0, y=0.0, z=-1.0),
            params={"spacing": ADAPTER_PLATE_M5_SPACING_MM},
        ),
    ]

    metadata = Metadata(
        name=spec.name,
        units="mm",
        # Constant +Z: adapter_plate always prints flat.
        print_orientation=Vec3(x=0.0, y=0.0, z=1.0),
    )
    constraints = Constraints(
        min_wall_thickness=w,
        max_overhang_deg=spec.max_overhang_deg,
        max_bbox=spec.envelope_mm,
    )

    return PartProgram(
        metadata=metadata,
        operations=ops,
        interfaces=interfaces,
        constraints=constraints,
    )


__all__ = [
    "NEMA17_HALF_PITCH",
    "derive_adapter_offsets",
    "derive_plate_dims",
    "interpret_adapter_plate",
]
