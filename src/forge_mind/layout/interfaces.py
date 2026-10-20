"""Deterministic placement of known mechanical interfaces (ADR-0006 Stage 3).

The placement half of the interface single-source-of-truth. Each function emits
the PartDSL operations for a known interface, reading the SAME facts module the
validator checker reads (``dsl.interfaces.nema17``). An interpreter calls these
instead of open-coding hole math, so "where the holes go" is defined once and the
built geometry cannot drift from what the checker expects.

This is the layout-layer counterpart to ``validator.interfaces`` (checking) and
``dsl.interfaces`` (data). place() lives here, not next to the facts, because it
emits PartDSL ops and uses ``layout.rules`` -- behaviour belongs in the layout
layer; only the facts are shared neutral data.

First interface: NEMA 17 face. More interfaces move here only when a second
interpreter needs them (ADR-0006 Stage 2 rule: earn the abstraction with a real
second caller).
"""

from __future__ import annotations

from enum import StrEnum

from forge_mind.dsl.interfaces.fan import FAN_BOLT_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.nema17 import (
    BOLT_PITCH_MM,
    M3_CLEARANCE_RADIUS_MM,
    SHAFT_BORE_RADIUS_MM,
)
from forge_mind.dsl.primitives import Cylinder, Operation, OpMode
from forge_mind.layout.rules import centered_rect_corners
from forge_mind.types import Rotation, Vec3


class TNutPair(StrEnum):
    """Orientation of an M5 T-nut pair: which axis the two holes are spaced
    along, and which axis they drill.

    Two orientations are defined -- one per real consumer (ADR-0006 Stage 2:
    earn the abstraction with a second caller, do not over-generalise before
    the third). Add a member when a third interpreter needs a new pair.

    - ``ALONG_X_DRILL_Y``: bench-001 L-bracket upright. Holes offset +/-
      spacing/2 along X; cylinder axis rotated to +Y (X-rotation 90 deg) so
      the pair drills through a vertical wall facing +Y.
    - ``ALONG_Y_DRILL_Z``: bench-003 adapter plate. Holes offset +/-
      spacing/2 along Y; cylinder axis stays at the default +Z so the pair
      drills straight through a horizontal plate (the extrusion-2020 checker
      is axis-agnostic, so a -Z interface normal still matches +Z cylinders).
    """

    ALONG_X_DRILL_Y = "along_x_drill_y"
    ALONG_Y_DRILL_Z = "along_y_drill_z"


def place_nema17_face(
    center: Vec3,
    thickness: float,
    *,
    bolt_op_prefix: str = "motor_bolt",
    shaft_op_id: str = "shaft_clearance",
    bolt_pitch_mm: float = BOLT_PITCH_MM,
    bolt_clearance_radius_mm: float = M3_CLEARANCE_RADIUS_MM,
    pilot_bore_radius_mm: float = SHAFT_BORE_RADIUS_MM,
) -> list[Operation]:
    """A motor face as SUBTRACT cylinders drilled +Z through a plate.

    Emits the 4 corner bolt holes on the bolt square plus the central pilot-boss bore,
    all centred at ``center`` and ``thickness`` deep along +Z (the only orientation any
    current consumer needs; an arbitrary face normal is deferred until a second consumer
    requires it). The pitch and both radii DEFAULT to the ``dsl.interfaces.nema17`` facts
    -- so a default call is byte-identical to the legacy NEMA 17 geometry -- and are
    overridden per motor (NEMA 14/23) from the MotorComponent registry (ADR-0017).

    op ids are ``f"{bolt_op_prefix}_{tag}"`` for the four bolts (tag = nw/ne/sw/se)
    and ``shaft_op_id`` for the bore, matching the hand-authored bench-001 geometry
    so the emitted program is byte-for-byte what the interpreter produced inline.
    """
    ops: list[Operation] = []
    for tag, cx, cy in centered_rect_corners(bolt_pitch_mm, bolt_pitch_mm):
        ops.append(
            Cylinder(
                op_id=f"{bolt_op_prefix}_{tag}",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=center.x + cx, y=center.y + cy, z=center.z),
                radius=bolt_clearance_radius_mm,
                height=thickness,
            )
        )
    ops.append(
        Cylinder(
            op_id=shaft_op_id,
            mode=OpMode.SUBTRACT,
            position=Vec3(x=center.x, y=center.y, z=center.z),
            radius=pilot_bore_radius_mm,
            height=thickness,
        )
    )
    return ops


def place_fan_face(
    center: Vec3,
    thickness: float,
    *,
    bolt_square_mm: float,
    airflow_bore_radius_mm: float,
    bolt_op_prefix: str = "fan_bolt",
    bore_op_id: str = "airflow_bore",
) -> list[Operation]:
    """The axial-fan face as SUBTRACT cylinders drilled +Z through a plate.

    The fan analogue of ``place_nema17_face``: 4 M3 corner holes on a square of
    side ``bolt_square_mm`` plus a central airflow bore of ``airflow_bore_radius_mm``,
    all centred at ``center`` and ``thickness`` deep along +Z. Reuses the same
    ``centered_rect_corners`` corner ordering and the shared M3 clearance fact the
    fan_face checker validates against, so built geometry and checked geometry are
    sourced from the same numbers. Only the square side and the central bore radius
    differ from the NEMA17 face -- exactly the per-fan facts the registry supplies.
    """
    ops: list[Operation] = []
    for tag, cx, cy in centered_rect_corners(bolt_square_mm, bolt_square_mm):
        ops.append(
            Cylinder(
                op_id=f"{bolt_op_prefix}_{tag}",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=center.x + cx, y=center.y + cy, z=center.z),
                radius=FAN_BOLT_CLEARANCE_RADIUS_MM,
                height=thickness,
            )
        )
    ops.append(
        Cylinder(
            op_id=bore_op_id,
            mode=OpMode.SUBTRACT,
            position=Vec3(x=center.x, y=center.y, z=center.z),
            radius=airflow_bore_radius_mm,
            height=thickness,
        )
    )
    return ops


def place_m5_pair(
    center: Vec3,
    spacing: float,
    thickness: float,
    *,
    orientation: TNutPair = TNutPair.ALONG_X_DRILL_Y,
    left_op_id: str = "ext_bolt_left",
    right_op_id: str = "ext_bolt_right",
    clearance_radius_mm: float = M5_CLEARANCE_RADIUS_MM,
) -> list[Operation]:
    """The 2 M5 T-nut clearance holes of a 2020-extrusion face, as SUBTRACT cylinders.

    Two ``orientation`` choices are supported (see ``TNutPair``):

    - ``ALONG_X_DRILL_Y`` (default, bench-001 upright): holes offset +/-
      ``spacing``/2 along X at the shared y/z, cylinder rotated to drill +Y
      through a vertical wall.
    - ``ALONG_Y_DRILL_Z`` (bench-003 adapter plate): holes offset +/-
      ``spacing``/2 along Y at the shared x/z, cylinder at the default +Z axis
      so the pair drills straight through a horizontal plate.

    The radius DEFAULTS to ``dsl.interfaces.m5`` (so a default call is byte-identical
    to the legacy 2020 geometry) and is overridden per extrusion (M6 for 3030) from the
    ExtrusionComponent registry (ADR-0017) -- the same fact the checker validates against.

    The op id keyword args read ``left``/``right`` because that is the natural
    label for bench-001's X-spaced row (still the default). For the +Y-spaced
    flat orientation, pass labels that read more naturally for the layout
    (e.g. ``left_op_id="ext_bolt_back"``, ``right_op_id="ext_bolt_front"``);
    the geometry and the validator do not care about the names.
    """
    half = spacing / 2.0
    if orientation is TNutPair.ALONG_X_DRILL_Y:
        rotation = Rotation(x=90.0, y=0.0, z=0.0)
        offsets: tuple[tuple[str, float, float], ...] = (
            (left_op_id, -half, 0.0),
            (right_op_id, half, 0.0),
        )
    elif orientation is TNutPair.ALONG_Y_DRILL_Z:
        rotation = Rotation(x=0.0, y=0.0, z=0.0)
        offsets = (
            (left_op_id, 0.0, -half),
            (right_op_id, 0.0, half),
        )
    else:  # pragma: no cover - StrEnum exhaustively covered above
        raise ValueError(f"unsupported TNutPair orientation: {orientation!r}")

    ops: list[Operation] = []
    for op_id, dx, dy in offsets:
        ops.append(
            Cylinder(
                op_id=op_id,
                mode=OpMode.SUBTRACT,
                position=Vec3(x=center.x + dx, y=center.y + dy, z=center.z),
                rotation=rotation,
                radius=clearance_radius_mm,
                height=thickness,
            )
        )
    return ops


__all__ = ["TNutPair", "place_fan_face", "place_m5_pair", "place_nema17_face"]
