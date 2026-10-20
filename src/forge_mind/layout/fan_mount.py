"""Fan-mount path of the adapter_plate pattern (bench-006, ADR-0011).

A fan mount is the *adapter_plate* shape -- a flat plate composing two face
interfaces -- with the +Z face hosting an axial FAN instead of a NEMA 17 motor.
``interpret_adapter_plate`` dispatches here when ``AdapterPlateSpec.fan_component_id``
is set; the -Z M5 T-nut mount is reused unchanged. This is NOT a new pattern: same
``AdapterPlateSpec``, same ``Pattern.ADAPTER_PLATE``, same validator tiers; only the
+Z interface (and its land math) differs.

The fan's bolt square + airflow bore are facts of the chosen fan, resolved by id.
Every structural value -- plate dims, bolt positions, the bore -- is DERIVED here
(or taken from explicit spec values for the byte-equivalent reference); the LLM
picks only the fan id (ADR-0009 "LLM picks categorical, code derives relational").

Fan-specific feasibility (beyond the adapter's edge/overlap/envelope rules): the
central airflow bore must keep >= min_wall to the corner bolts -- the bore-to-bolt
land ADR-0011 flagged. This is what makes a flat 40 mm mount infeasible at a 4 mm
wall (land ~2.03 mm) while a 60 mm mount is comfortable (~5.76 mm).
"""

from __future__ import annotations

import math

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.fan import FAN_BOLT_CLEARANCE_RADIUS_MM as M3_CLEARANCE_RADIUS
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM as M5_CLEARANCE_RADIUS
from forge_mind.dsl.primitives import Box, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.adapter_spec import AdapterPlateSpec
from forge_mind.layout.interfaces import TNutPair, place_fan_face, place_m5_pair
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.rules import edge_land
from forge_mind.types import Vec3

FAN_FACE_NAME = "fan_face"
EXTRUSION_FACE_NAME = "extrusion_face"
FAN_MOUNT_M5_SPACING_MM = 20.0


def resolve_fan_facts(component_id: str) -> tuple[float, float]:
    """(bolt_square_mm, airflow_bore_radius_mm) for a fan id, from the FanComponent
    registry (bench-006 Stage B). Raises ``LayoutError`` on an unknown id so an LLM
    that picks a bad id routes to the freeform fallback (ADR-0006 layer D)."""
    from forge_mind.dsl.components.registry import get_fan_component

    try:
        fan = get_fan_component(component_id)
    except ValueError as exc:
        raise LayoutError(str(exc)) from None
    return (fan.mount.bolt_square_mm, fan.mount.airflow_bore_radius_mm)


def _fan_half_extent(bolt_square: float, bore_r: float) -> float:
    """In-plane half-extent of the fan cluster from its centre: the larger of the
    corner-bolt reach (square/2 + M3 r) and the airflow bore radius."""
    return max(bolt_square / 2.0 + M3_CLEARANCE_RADIUS, bore_r)


def derive_fan_offsets(
    envelope_x_mm: float, min_wall_mm: float, bolt_square: float, bore_r: float
) -> tuple[float, float]:
    """Smallest balanced ``(fan_center_x, extrusion_center_x)`` -- the fan analogue of
    ``derive_adapter_offsets``. Fan on +X, T-nut on -X; both sides consume equal plate
    half-width, and the two clusters keep a >= min_wall no-overlap land at the centre.

    Balancing cancels the cluster size, so the fan offset is ``M5_r + min_wall/2``
    (same as the motor adapter); the T-nut offset and the plate width then absorb the
    fan's larger half-extent.
    """
    eps = 1e-6
    fan_half = _fan_half_extent(bolt_square, bore_r)
    fan_abs = M5_CLEARANCE_RADIUS + min_wall_mm / 2.0 + eps
    ext_abs = fan_abs + (fan_half - M5_CLEARANCE_RADIUS)
    half_w = fan_abs + fan_half + min_wall_mm
    if half_w > envelope_x_mm / 2.0 + 1e-6:
        raise LayoutError(
            f"fan-mount envelope X {envelope_x_mm}mm is too small to host the fan face "
            f"+ 2020 T-nut pair with min_wall {min_wall_mm}mm. Required envelope X >= "
            f"{2 * half_w:.2f}mm (smallest balanced layout)."
        )
    return (fan_abs, -ext_abs)


def _resolve_offsets(
    spec: AdapterPlateSpec, bolt_square: float, bore_r: float
) -> tuple[float, float]:
    """Explicit (motor_center_x serves as the +Z face centre) values win; else derive."""
    if spec.motor_center_x_mm is not None and spec.extrusion_center_x_mm is not None:
        return (spec.motor_center_x_mm, spec.extrusion_center_x_mm)
    d_fan, d_ext = derive_fan_offsets(spec.envelope_mm.x, spec.min_wall_mm, bolt_square, bore_r)
    fan_x = spec.motor_center_x_mm if spec.motor_center_x_mm is not None else d_fan
    ext_x = spec.extrusion_center_x_mm if spec.extrusion_center_x_mm is not None else d_ext
    return (fan_x, ext_x)


def derive_fan_plate_dims(
    spec: AdapterPlateSpec, bolt_square: float, bore_r: float
) -> tuple[float, float]:
    """Tightest legal plate footprint given the resolved offsets + min_wall + fan facts."""
    w = spec.min_wall_mm
    m5_half = FAN_MOUNT_M5_SPACING_MM / 2.0
    fan_x, ext_x = _resolve_offsets(spec, bolt_square, bore_r)
    fan_half = _fan_half_extent(bolt_square, bore_r)
    plus_x = max(fan_x + fan_half, ext_x + M5_CLEARANCE_RADIUS)
    minus_x = max(-fan_x + fan_half, -ext_x + M5_CLEARANCE_RADIUS)
    half_w = max(plus_x, minus_x) + w
    half_d = max(fan_half, m5_half + M5_CLEARANCE_RADIUS) + w
    return (2.0 * half_w, 2.0 * half_d)


def _resolve_plate_dims(
    spec: AdapterPlateSpec, bolt_square: float, bore_r: float
) -> tuple[float, float]:
    derived_w, derived_d = derive_fan_plate_dims(spec, bolt_square, bore_r)
    width = spec.plate_width_mm if spec.plate_width_mm is not None else derived_w
    depth = spec.plate_depth_mm if spec.plate_depth_mm is not None else derived_d
    return (width, depth)


def _check_feasible(spec: AdapterPlateSpec, bolt_square: float, bore_r: float) -> None:
    """Reject fan-mount specs that cannot be realised with legal lands / envelope."""
    w = spec.min_wall_mm
    fan_x, ext_x = _resolve_offsets(spec, bolt_square, bore_r)
    plate_width, plate_depth = _resolve_plate_dims(spec, bolt_square, bore_r)
    half_w = plate_width / 2.0
    half_d = plate_depth / 2.0
    half_square = bolt_square / 2.0
    m5_half = FAN_MOUNT_M5_SPACING_MM / 2.0
    problems: list[str] = []

    # --- Fan bolt cluster vs plate edges (corner bolts at +/- square/2 in each axis).
    if edge_land(half_w, abs(fan_x) + half_square, M3_CLEARANCE_RADIUS) < w:
        problems.append(
            f"plate width {plate_width:.1f}mm too small: fan M3 hole-to-edge land < "
            f"min_wall {w}mm at fan X centre {fan_x:.1f}mm."
        )
    if edge_land(half_d, half_square, M3_CLEARANCE_RADIUS) < w:
        problems.append(
            f"plate depth {plate_depth:.1f}mm too small: fan M3 hole-to-edge land < min_wall {w}mm in Y."
        )
    # --- Airflow bore vs plate edges.
    if edge_land(half_w, abs(fan_x), bore_r) < w:
        problems.append(
            f"plate width {plate_width:.1f}mm too small for the {2 * bore_r:.0f}mm airflow "
            f"bore + {w}mm land at fan X centre {fan_x:.1f}mm."
        )
    if edge_land(half_d, 0.0, bore_r) < w:
        problems.append(
            f"plate depth {plate_depth:.1f}mm too small for the {2 * bore_r:.0f}mm airflow bore + {w}mm land in Y."
        )
    # --- Fan-specific: the airflow bore must keep >= min_wall to the corner bolts
    # (the bore-to-bolt land ADR-0011 flagged; binding for small fans on thick walls).
    bolt_corner_radial = half_square * math.sqrt(2)
    bore_to_bolt_land = (bolt_corner_radial - M3_CLEARANCE_RADIUS) - bore_r
    if bore_to_bolt_land < w:
        problems.append(
            f"airflow bore (r {bore_r:.1f}mm) leaves only {bore_to_bolt_land:.2f}mm to the "
            f"corner bolts (need >= min_wall {w}mm). A flat fan mount this size cannot keep "
            "the bolt land at this wall -- shrink the bore, widen the bolt square, or thin the wall."
        )
    # --- T-nut pair (spaced along Y) vs the +/- Y plate edge.
    if edge_land(half_d, m5_half, M5_CLEARANCE_RADIUS) < w:
        problems.append(
            f"plate depth {plate_depth:.1f}mm too small: M5 row + {w}mm land does not fit "
            f"(half-spacing {m5_half} + M5 r {M5_CLEARANCE_RADIUS} + wall {w})."
        )
    # --- Fan cluster vs T-nut pair overlap (plan view).
    inner = abs(fan_x - ext_x)
    closest_fan_edge = inner - (half_square + M3_CLEARANCE_RADIUS)
    if closest_fan_edge - M5_CLEARANCE_RADIUS < w:
        problems.append(
            f"fan cluster and T-nut pair overlap or touch: distance {inner:.1f}mm leaves a land "
            f"of {closest_fan_edge - M5_CLEARANCE_RADIUS:.2f}mm < min_wall {w}mm."
        )
    # --- Envelope.
    env = spec.envelope_mm
    if plate_width > env.x + 1e-6:
        problems.append(f"plate width {plate_width:.1f}mm exceeds envelope X {env.x}mm.")
    if plate_depth > env.y + 1e-6:
        problems.append(f"plate depth {plate_depth:.1f}mm exceeds envelope Y {env.y}mm.")
    if w > env.z + 1e-6:
        problems.append(f"plate thickness {w}mm exceeds envelope Z {env.z}mm.")

    if problems:
        raise LayoutError("infeasible fan-mount spec: " + "; ".join(problems))


def interpret_fan_mount(spec: AdapterPlateSpec) -> PartProgram:
    """Deterministically realise a fan-mount AdapterPlateSpec as a validated PartProgram.

    Raises ``LayoutError`` if infeasible (checked before geometry). The op order + ids
    mirror the hand-authored 60 mm reference (``examples/handwritten_fan_2020_adapter.py``)
    so the interpreter and the reference produce equivalent parts on the explicit spec.
    """
    assert spec.fan_component_id is not None  # dispatch guard
    bolt_square, bore_r = resolve_fan_facts(spec.fan_component_id)
    _check_feasible(spec, bolt_square, bore_r)

    w = spec.min_wall_mm
    fan_x, ext_x = _resolve_offsets(spec, bolt_square, bore_r)
    plate_width, plate_depth = _resolve_plate_dims(spec, bolt_square, bore_r)
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
    ops.extend(
        place_fan_face(
            center=Vec3(x=fan_x, y=0.0, z=plate_mid_z),
            thickness=w,
            bolt_square_mm=bolt_square,
            airflow_bore_radius_mm=bore_r,
        )
    )
    ops.extend(
        place_m5_pair(
            center=Vec3(x=ext_x, y=0.0, z=plate_mid_z),
            spacing=FAN_MOUNT_M5_SPACING_MM,
            thickness=w,
            orientation=TNutPair.ALONG_Y_DRILL_Z,
            left_op_id="ext_bolt_back",
            right_op_id="ext_bolt_front",
        )
    )

    interfaces = [
        InterfaceDeclaration(
            name=FAN_FACE_NAME,
            interface_type=InterfaceType.FAN_FACE,
            location=Vec3(x=fan_x, y=0.0, z=plate_top_z),
            normal=Vec3(x=0.0, y=0.0, z=1.0),
            params={"bolt_square": bolt_square, "airflow_bore_radius": bore_r},
        ),
        InterfaceDeclaration(
            name=EXTRUSION_FACE_NAME,
            interface_type=InterfaceType.EXTRUSION_2020_SLOT,
            location=Vec3(x=ext_x, y=0.0, z=0.0),
            normal=Vec3(x=0.0, y=0.0, z=-1.0),
            params={"spacing": FAN_MOUNT_M5_SPACING_MM},
        ),
    ]
    metadata = Metadata(
        name=spec.name,
        units="mm",
        print_orientation=Vec3(x=0.0, y=0.0, z=1.0),  # always flat (+Z up)
    )
    constraints = Constraints(
        min_wall_thickness=w,
        max_overhang_deg=spec.max_overhang_deg,
        max_bbox=spec.envelope_mm,
    )
    return PartProgram(
        metadata=metadata, operations=ops, interfaces=interfaces, constraints=constraints
    )


__all__ = [
    "derive_fan_offsets",
    "derive_fan_plate_dims",
    "interpret_fan_mount",
    "resolve_fan_facts",
]
