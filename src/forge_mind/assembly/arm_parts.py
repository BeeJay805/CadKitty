"""Printed-part builders for the servo arm family (MG996R serial chains).

Every part mounts its servo the same way: a 3 mm PLATE with a rectangular WINDOW the
servo body drops through and the CITED 4-hole tab pattern around it (49.5 x 10.0 mm,
from ``assembly.servo`` -- the same constants the mount-pattern gate measures against,
so the geometry and the check can never disagree). Orientation lives in the PORT
frames, not the solids: all boxes stay axis-aligned; only drilled holes rotate.

Local frame conventions:
  * ``servo_base``: plate on top of a printable skirt; cradle_mount port faces +Z.
    The servo body hangs inside the skirt cavity; the output shaft points up.
  * ``servo_link``: the horn hub is at the ORIGIN, axis +Z (it mates the servo's horn
    plane, so the link's z=0 plane sits ON the horn). The beam runs +X. The far end is
    one of: "tool" (two M3 holes), "cradle_perp" (a vertical plate whose servo axis is
    +X -- the pan->tilt yoke), "cradle_parallel" (a windowed slab whose servo axis is
    +Z again -- a planar elbow).
"""

from __future__ import annotations

from forge_mind.assembly.joints import Port
from forge_mind.assembly.servo import (
    MG996R_BODY_H_MM,
    MG996R_BODY_L_MM,
    MG996R_BODY_W_MM,
    MG996R_FLANGE_UNDERSIDE_BELOW_TOP_MM,
    MG996R_HORN_SCREW_BC_D_MM,
    MG996R_HORN_SCREW_D_MM,
    MG996R_HUB_CENTER_BORE_D_MM,
    MG996R_MOUNT_HOLE_SPAN_X_MM,
    MG996R_MOUNT_HOLE_SPAN_Y_MM,
    MG996R_SHAFT_OFFSET_MM,
)
from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.types import Rotation, Vec3

_WINDOW_CLEAR_MM = 0.4  # body window clearance per the running print fits
_TAB_HOLE_R_MM = 2.25  # 4.5 mm nominal tab screw clearance
_PLATE_T_MM = 3.0
_BODY_BELOW_FLANGE_MM = MG996R_BODY_H_MM - MG996R_FLANGE_UNDERSIDE_BELOW_TOP_MM  # 28.6 drop
_HOLE_HX = MG996R_MOUNT_HOLE_SPAN_X_MM / 2.0  # 24.75
_HOLE_HY = MG996R_MOUNT_HOLE_SPAN_Y_MM / 2.0  # 5.0
_WIN_HX = (MG996R_BODY_L_MM + _WINDOW_CLEAR_MM) / 2.0  # 20.55
_WIN_HY = (MG996R_BODY_W_MM + _WINDOW_CLEAR_MM) / 2.0  # 10.05


def _hub_ops(hub_r: float, hub_h: float) -> list[Operation]:
    """The horn hub: a disc at the origin (axis +Z) carrying the cited horn interface
    (center bore + 4 horn screws on the bolt circle) -- measured by the horn mount
    pattern. The builders re-partition every op list to ADD-then-SUBTRACT before
    compiling (operations apply sequentially; a beam added after a drilled hole would
    refill it), so concatenation order here never matters."""
    adds: list[Operation] = [
        Cylinder(
            op_id="hub",
            mode=OpMode.ADD,
            position=Vec3(z=hub_h / 2.0),
            radius=hub_r,
            height=hub_h,
        ),
    ]
    # NOT op_id "hub_bore": that id marks a NEMA shaft-GRIP hub and triggers the
    # shaft-coupling check; this is the horn's screw interface (clearance, not grip).
    subs: list[Operation] = [
        Cylinder(
            op_id="horn_bore",
            mode=OpMode.SUBTRACT,
            position=Vec3(z=hub_h / 2.0),
            radius=MG996R_HUB_CENTER_BORE_D_MM / 2.0,
            height=hub_h + 2.0,
        ),
    ]
    bc_r = MG996R_HORN_SCREW_BC_D_MM / 2.0
    for i, (dx, dy) in enumerate(((bc_r, 0.0), (-bc_r, 0.0), (0.0, bc_r), (0.0, -bc_r))):
        subs.append(
            Cylinder(
                op_id=f"horn_screw_{i}",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=dx, y=dy, z=hub_h / 2.0),
                radius=MG996R_HORN_SCREW_D_MM / 2.0,
                height=hub_h + 2.0,
            )
        )
    return adds + subs


def _adds_then_subtracts(ops: list[Operation]) -> list[Operation]:
    """Stable-partition to ADD-then-SUBTRACT. The compiler applies ops sequentially,
    so a solid added AFTER a drilled hole refills it; partitioning makes every drill
    land in the finished union regardless of authoring order."""
    return [o for o in ops if o.mode == OpMode.ADD] + [o for o in ops if o.mode != OpMode.ADD]


def servo_base(
    *,
    name: str,
    plate_x: float = 80.0,
    plate_y: float = 60.0,
    skirt_t: float = 4.0,
    foot_hole_r: float = 2.15,
) -> tuple[PartProgram, list[Port]]:
    """The desk base: a printable skirt with a windowed top plate; the pan servo drops
    in shaft-up, flange resting on the plate, body hanging inside the cavity."""
    top = _BODY_BELOW_FLANGE_MM + 3.0  # plate TOP height: body bottom floats 3 mm up
    ops: list[Operation] = [
        Box(
            op_id="plate",
            mode=OpMode.ADD,
            position=Vec3(z=top - _PLATE_T_MM / 2.0),
            width=plate_x,
            depth=plate_y,
            height=_PLATE_T_MM,
        ),
    ]
    # Skirt: four walls from the desk (z=0) up to the plate underside.
    wall_h = top - _PLATE_T_MM
    for op_id, px, py, w, d in (
        ("skirt_xp", plate_x / 2.0 - skirt_t / 2.0, 0.0, skirt_t, plate_y),
        ("skirt_xm", -plate_x / 2.0 + skirt_t / 2.0, 0.0, skirt_t, plate_y),
        ("skirt_yp", 0.0, plate_y / 2.0 - skirt_t / 2.0, plate_x, skirt_t),
        ("skirt_ym", 0.0, -plate_y / 2.0 + skirt_t / 2.0, plate_x, skirt_t),
    ):
        ops.append(
            Box(
                op_id=op_id,
                mode=OpMode.ADD,
                position=Vec3(x=px, y=py, z=wall_h / 2.0),
                width=w,
                depth=d,
                height=wall_h,
            )
        )
    # The servo window + the cited tab pattern, centered on the plate.
    ops.append(
        Box(
            op_id="window",
            mode=OpMode.SUBTRACT,
            position=Vec3(z=top - _PLATE_T_MM / 2.0),
            width=2.0 * _WIN_HX,
            depth=2.0 * _WIN_HY,
            height=_PLATE_T_MM + 2.0,
        )
    )
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        ops.append(
            Cylinder(
                op_id=f"tab_{i}",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=sx * _HOLE_HX, y=sy * _HOLE_HY, z=top - _PLATE_T_MM / 2.0),
                radius=_TAB_HOLE_R_MM,
                height=_PLATE_T_MM + 2.0,
            )
        )
    # Corner desk-mount holes through the plate.
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        ops.append(
            Cylinder(
                op_id=f"foot_{i}",
                mode=OpMode.SUBTRACT,
                position=Vec3(
                    x=sx * (plate_x / 2.0 - 7.0),
                    y=sy * (plate_y / 2.0 - 7.0),
                    z=top - _PLATE_T_MM / 2.0,
                ),
                radius=foot_hole_r,
                height=_PLATE_T_MM + 2.0,
            )
        )
    program = PartProgram(
        metadata=Metadata(name=name, units="mm"), operations=_adds_then_subtracts(ops)
    )
    ports = [
        Port(name="cradle_mount", origin=Vec3(z=top), axis=Vec3(z=1.0), normal=Vec3(x=1.0)),
    ]
    return program, ports


def servo_link(
    *,
    name: str,
    length_x: float,
    end: str = "tool",
    beam_w: float = 24.0,
    beam_t: float = 6.0,
    hub_r: float = 12.0,
    cradle_height_z: float = 60.0,
    tool_hole_r: float = 1.6,
) -> tuple[PartProgram, list[Port]]:
    """A driven link: horn hub at the origin (+Z), beam to +X, and the chosen far end.

    ``end``:
      * "tool": two M3 tool holes near the tip (the terminal link / tilt platform).
      * "cradle_perp": a vertical servo plate at x=length_x whose servo axis is +X,
        elevated to z=cradle_height_z, carried by two side cheeks that straddle the
        servo body (the pan->tilt yoke).
      * "cradle_parallel": the beam widens into a windowed slab at the far end whose
        servo axis is +Z (a planar elbow stage); the next servo hangs below the slab.
    """
    ops: list[Operation] = _hub_ops(hub_r, beam_t)
    ports: list[Port] = [
        Port(name="hub", origin=Vec3(), axis=Vec3(z=1.0), normal=Vec3(x=1.0)),
    ]
    if end == "cradle_perp":
        # The servo's flange rests on the plate's +X face (the mount plane at
        # x=length_x); the plate body sits BEHIND that plane (toward -X) so the lower
        # servo body passes through its window while the flange + case-top section
        # stay on the +X side -- the physical way a servo bolts to a plate.
        plate_x0 = length_x  # the mount plane (flange underside lands here)
        half_span = 31.0  # plate half-height: covers holes at +-24.75 with margin
        # Base slab: hub disc out to the cheeks' feet.
        ops.append(
            Box(
                op_id="slab",
                mode=OpMode.ADD,
                position=Vec3(x=(plate_x0 - 12.0) / 2.0, y=0.0, z=beam_t / 2.0),
                width=plate_x0 + 12.0,
                depth=2.0 * (_WIN_HY + 6.0) + 2.0,
                height=beam_t,
            )
        )
        # Two cheeks straddle the servo body (body |y| <= _WIN_HY) and carry the plate.
        cheek_w = 6.0
        for op_id, sy in (("cheek_p", 1.0), ("cheek_m", -1.0)):
            ops.append(
                Box(
                    op_id=op_id,
                    mode=OpMode.ADD,
                    position=Vec3(
                        x=plate_x0 - 5.0,
                        y=sy * (_WIN_HY + 1.0 + cheek_w / 2.0),
                        z=(cradle_height_z + half_span) / 2.0,
                    ),
                    width=10.0,
                    depth=cheek_w,
                    height=cradle_height_z + half_span,
                )
            )
        # The vertical servo plate (normal +X), behind the mount plane, with window +
        # cited tab holes drilled along X.
        ops.append(
            Box(
                op_id="plate",
                mode=OpMode.ADD,
                position=Vec3(
                    x=plate_x0 - _PLATE_T_MM / 2.0,
                    y=0.0,
                    z=cradle_height_z,
                ),
                width=_PLATE_T_MM,
                depth=2.0 * (_WIN_HY + 1.0 + cheek_w),
                height=2.0 * half_span,
            )
        )
        ops.append(
            Box(
                op_id="window",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=plate_x0 - _PLATE_T_MM / 2.0, y=0.0, z=cradle_height_z),
                width=_PLATE_T_MM + 2.0,
                depth=2.0 * _WIN_HY,
                height=2.0 * _WIN_HX,
            )
        )
        for i, (sy, sz) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            ops.append(
                Cylinder(
                    op_id=f"tab_{i}",
                    mode=OpMode.SUBTRACT,
                    position=Vec3(
                        x=plate_x0 - _PLATE_T_MM / 2.0,
                        y=sy * _HOLE_HY,
                        z=cradle_height_z + sz * _HOLE_HX,
                    ),
                    rotation=Rotation(y=90.0),  # drill along +X
                    radius=_TAB_HOLE_R_MM,
                    height=_PLATE_T_MM + 2.0,
                )
            )
        # The servo's +Y maps to the link's -Y under this mate; the pattern is
        # y-symmetric so the measured holes land on the cited points either way.
        ports.append(
            Port(
                name="cradle_mount",
                origin=Vec3(x=plate_x0, y=0.0, z=cradle_height_z),
                axis=Vec3(x=1.0),
                normal=Vec3(z=1.0),
            )
        )
    elif end == "cradle_parallel":
        # Widened far slab carrying the next servo shaft-up at x ~= length_x.
        body_cx = length_x - MG996R_SHAFT_OFFSET_MM
        slab_x0 = body_cx - (_WIN_HX + 11.0)
        slab_x1 = body_cx + (_WIN_HX + 11.0)
        ops.append(
            Box(
                op_id="beam",
                mode=OpMode.ADD,
                position=Vec3(x=slab_x1 / 2.0, y=0.0, z=beam_t / 2.0),
                width=slab_x1,
                depth=beam_w,
                height=beam_t,
            )
        )
        ops.append(
            Box(
                op_id="slab",
                mode=OpMode.ADD,
                position=Vec3(x=(slab_x0 + slab_x1) / 2.0, y=0.0, z=beam_t / 2.0),
                width=slab_x1 - slab_x0,
                depth=2.0 * (_HOLE_HY + 6.0),
                height=beam_t,
            )
        )
        ops.append(
            Box(
                op_id="window",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=body_cx, y=0.0, z=beam_t / 2.0),
                width=2.0 * _WIN_HX,
                depth=2.0 * _WIN_HY,
                height=beam_t + 2.0,
            )
        )
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            ops.append(
                Cylinder(
                    op_id=f"tab_{i}",
                    mode=OpMode.SUBTRACT,
                    position=Vec3(x=body_cx + sx * _HOLE_HX, y=sy * _HOLE_HY, z=beam_t / 2.0),
                    radius=_TAB_HOLE_R_MM,
                    height=beam_t + 2.0,
                )
            )
        ports.append(
            Port(
                name="cradle_mount",
                origin=Vec3(x=body_cx, y=0.0, z=beam_t),
                axis=Vec3(z=1.0),
                normal=Vec3(x=1.0),
            )
        )
    else:  # "tool"
        ops.append(
            Box(
                op_id="beam",
                mode=OpMode.ADD,
                position=Vec3(x=length_x / 2.0, y=0.0, z=beam_t / 2.0),
                width=length_x,
                depth=beam_w,
                height=beam_t,
            )
        )
        for i, sy in enumerate((1.0, -1.0)):
            ops.append(
                Cylinder(
                    op_id=f"tool_{i}",
                    mode=OpMode.SUBTRACT,
                    position=Vec3(x=length_x - 8.0, y=sy * 6.0, z=beam_t / 2.0),
                    radius=tool_hole_r,
                    height=beam_t + 2.0,
                )
            )
    program = PartProgram(
        metadata=Metadata(name=name, units="mm"), operations=_adds_then_subtracts(ops)
    )
    return program, ports


__all__ = ["servo_base", "servo_link"]
