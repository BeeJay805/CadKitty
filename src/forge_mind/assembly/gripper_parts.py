"""Printed-part builders for the two-finger gear-coupled gripper (MG996R, ADR-0044).

The mechanism: ONE servo drives the left finger; full involute gear discs at both
finger roots (module 1.5, 24T -- the catalog generator's geometry, never hand-drawn
teeth) couple the right finger in mirror motion; the right finger pivots on a cited
ISO 2338 dowel pressed into a palm boss. The fingers bow OUTBOARD around the payload
(pivot spacing 36.2 mm is far smaller than a 66 mm can, so straight fingers cannot
work -- the classic C-jaw), and their grip pads carry a vertical V-groove so a
cylinder self-centers on four contact lines.

Every dimension is one of:
  * a cited registry fact (MG996R interface, ISO 2338 dowel, the payload object's
    diameter from ``dsl.standards.payloads``), or
  * a geometric derivation from those facts, done HERE in code (ADR-0009): the
    aperture targets derive from the can diameter; the finger travel derives from
    the aperture targets and the pad lever geometry.

LOCAL FRAMES:
  * palm: desk plane z=0; pivot line along X; drive pivot (the servo shaft axis) at
    x=0, idler pivot at x=+CENTER_DISTANCE; fingers reach toward +Y.
  * finger: gear axis = local origin, gear disc z in [0, GEAR_THICKNESS]; the drive
    finger reaches outboard -X then forward +Y (the idler is its x-mirror).
"""

from __future__ import annotations

import math

from forge_mind.agentic.gears import mesh_clock_deg
from forge_mind.assembly.joints import Port
from forge_mind.assembly.servo import (
    MG996R_BODY_H_MM,
    MG996R_BODY_L_MM,
    MG996R_BODY_W_MM,
    MG996R_FLANGE_UNDERSIDE_BELOW_TOP_MM,
    MG996R_HORN_PLANE_Z_MM,
    MG996R_HORN_SCREW_BC_D_MM,
    MG996R_HORN_SCREW_D_MM,
    MG996R_HUB_CENTER_BORE_D_MM,
    MG996R_MOUNT_HOLE_SPAN_X_MM,
    MG996R_MOUNT_HOLE_SPAN_Y_MM,
    MG996R_SHAFT_OFFSET_MM,
)
from forge_mind.dsl.primitives import Box, Cylinder, GearSolid, Operation, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.dsl.standards import dowel_hole, payload_object
from forge_mind.types import Rotation, Vec3

# ---- the gear coupling (catalog geometry; the mesh gate reads the same numbers) ----
GRIPPER_GEAR_MODULE = 1.5  # printable tooth size for a 0.4 mm nozzle (root web > min wall)
GRIPPER_GEAR_TEETH = 24  # valid involute (>= 17, no undercut); pitch dia 36 mm
GRIPPER_GEAR_THICKNESS_MM = 6.0
# Designed center-distance backlash: +0.2 mm opens the tooth flanks so FDM gears run
# free; it sits well inside the gear gate's +-0.5 mm spacing tolerance.
GRIPPER_BACKLASH_MM = 0.2
GRIPPER_CENTER_DISTANCE_MM = GRIPPER_GEAR_MODULE * GRIPPER_GEAR_TEETH + GRIPPER_BACKLASH_MM

# ---- shared servo-cradle facts (same constants the mount-pattern gate measures) ----
_WINDOW_CLEAR_MM = 0.4
_TAB_HOLE_R_MM = 2.25  # 4.5 mm nominal tab screw clearance
_PLATE_T_MM = 3.0
_BODY_BELOW_FLANGE_MM = MG996R_BODY_H_MM - MG996R_FLANGE_UNDERSIDE_BELOW_TOP_MM
_HOLE_HX = MG996R_MOUNT_HOLE_SPAN_X_MM / 2.0
_HOLE_HY = MG996R_MOUNT_HOLE_SPAN_Y_MM / 2.0
_WIN_HX = (MG996R_BODY_L_MM + _WINDOW_CLEAR_MM) / 2.0
_WIN_HY = (MG996R_BODY_W_MM + _WINDOW_CLEAR_MM) / 2.0

# ---- palm vertical stack (derived, single-sourced for builders + the graph) ----
GRIPPER_PALM_TOP_MM = _BODY_BELOW_FLANGE_MM + 3.0  # plate top; servo body floats 3 mm up
GRIPPER_BOSS_TOP_MM = 52.0  # dowel boss top (1.5 mm under the finger plane)
GRIPPER_BOSS_BORE_DEPTH_MM = 15.0  # press-bore depth into the boss
GRIPPER_BOSS_BORE_BOTTOM_MM = GRIPPER_BOSS_TOP_MM - GRIPPER_BOSS_BORE_DEPTH_MM
# The finger plane: a finger's z=0 face sits ON the servo horn plane.
GRIPPER_FINGER_PLANE_MM = (
    GRIPPER_PALM_TOP_MM + MG996R_FLANGE_UNDERSIDE_BELOW_TOP_MM + MG996R_HORN_PLANE_Z_MM
)
GRIPPER_DOWEL_SIZE = "6"  # ISO 2338 key (cited diameter; bores DERIVE from it)
GRIPPER_DOWEL_LENGTH_MM = 40.0  # a standard ISO 2338 catalog length

# ---- finger geometry (derived from the payload facts in gripper_finger_design) ----
_STRUT_W_MM = 12.0
_FOREARM_X_INNER_MM = 28.0  # forearm inner face offset (outboard of the pad face)
_FOREARM_X_OUTER_MM = 36.0
_PAD_T_MM = 8.0  # pad thickness (its inner face is the grip surface)
_PAD_LEN_MM = 16.0  # pad length along the reach direction
_PAD_H_MM = 14.0  # pad height (vertical contact band on the payload)
_PAD_V_DEPTH_MM = 2.0  # V-groove depth (4-line self-centering contact)

GRIP_PAD_OP_ID = "grip_pad"  # the aperture gate finds the pad region by THIS op id


class GripperDesign:
    """The derived design numbers for one payload object -- computed ONCE from the
    cited facts and read by the palm/finger builders, the graph, and the tests."""

    def __init__(self, payload_key: str = "beverage_can_355ml") -> None:
        can = payload_object(payload_key)
        self.payload_key = payload_key
        self.grip_diameter_mm = can.grip_diameter_mm
        # Aperture targets: open clears the payload by 18 mm total; closed undercuts
        # it by 6 mm (the squeeze that makes the pads actually clamp). Both leave
        # margin inside the gate bounds (open >= dia+8, closed <= dia-2).
        self.aperture_open_mm = can.grip_diameter_mm + 18.0
        self.aperture_closed_mm = can.grip_diameter_mm - 6.0
        cd = GRIPPER_CENTER_DISTANCE_MM
        # Pad inner-face x (drive-finger local, negative = outboard) at the OPEN pose:
        # the two mirrored faces must span aperture_open about the pivot midline.
        self.pad_face_x_mm = (cd - self.aperture_open_mm) / 2.0  # e.g. -23.9
        # Grip center: where the payload axis sits (midline, forward of the pivots).
        self.grip_center_y_mm = 70.0
        self.pad_y0_mm = self.grip_center_y_mm - _PAD_LEN_MM / 2.0
        self.pad_y1_mm = self.grip_center_y_mm + _PAD_LEN_MM / 2.0
        # Close travel: solve the rotation that brings the pad's innermost point to
        # the closed aperture (the pad corner leads once the finger tilts).
        self.close_travel_deg = self._solve_travel(self.aperture_closed_mm)

    def _pad_max_x(self, theta_deg: float) -> float:
        """The drive pad's innermost x after closing by ``theta`` (CW about +Z):
        x' = x cos(t) + y sin(t) over the pad face corners."""
        t = math.radians(theta_deg)
        return max(
            self.pad_face_x_mm * math.cos(t) + y * math.sin(t)
            for y in (self.pad_y0_mm, self.pad_y1_mm)
        )

    def _solve_travel(self, target_aperture_mm: float) -> float:
        """The closing angle where the mirrored pads reach the target aperture
        (bisection on the monotone pad-advance function; deterministic)."""
        target_x = (GRIPPER_CENTER_DISTANCE_MM - target_aperture_mm) / 2.0
        lo, hi = 0.0, 45.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if self._pad_max_x(mid) < target_x:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2.0, 1)

    def drive_range_deg(self) -> tuple[float, float]:
        """The drive joint's DESIGNED range: home (0) = open, negative = closing
        (CW about +Z swings the left finger's pad toward the midline). Never the
        servo's full +-90 -- the mechanism's own travel is the declared range."""
        return (-self.close_travel_deg, 0.0)


def _finger_ops(design: GripperDesign, *, mirror: bool) -> list[Operation]:
    """One finger's ops in its LOCAL frame (gear axis at origin). ``mirror=False`` =
    the drive finger (reaches outboard -X); ``mirror=True`` = the idler (+X)."""
    s = -1.0 if not mirror else 1.0  # outboard sign
    pad_face = abs(design.pad_face_x_mm)  # 23.9: pad inner face |x|
    y0, y1 = design.pad_y0_mm, design.pad_y1_mm

    ops: list[Operation] = []
    # Strut: hub -> forearm root, a rotated box (fused over the gear web). Its near
    # end starts CLEAR of the hub-bore column -- the bores are re-cut as explicit
    # SUBTRACTs after every ADD, but keeping solids out of the bore column entirely
    # leaves the bore wall a clean full cylinder for the mount-pattern gate.
    x0, sy0 = s * 8.0, 12.0
    x1, sy1 = s * (_FOREARM_X_INNER_MM + 4.0), 48.0
    length = math.hypot(x1 - x0, sy1 - sy0) + _STRUT_W_MM
    angle = math.degrees(math.atan2(sy1 - sy0, x1 - x0))
    ops.append(
        Box(
            op_id="strut",
            mode=OpMode.ADD,
            position=Vec3(x=(x0 + x1) / 2.0, y=(sy0 + sy1) / 2.0, z=GRIPPER_GEAR_THICKNESS_MM / 2),
            rotation=Rotation(z=angle),
            width=length,
            depth=_STRUT_W_MM,
            height=GRIPPER_GEAR_THICKNESS_MM,
        )
    )
    # Forearm: outboard rail running forward past the grip zone.
    fx0, fx1 = s * _FOREARM_X_OUTER_MM, s * _FOREARM_X_INNER_MM
    ops.append(
        Box(
            op_id="forearm",
            mode=OpMode.ADD,
            position=Vec3(
                x=(fx0 + fx1) / 2.0, y=(44.0 + y1) / 2.0, z=GRIPPER_GEAR_THICKNESS_MM / 2
            ),
            width=abs(fx1 - fx0),
            depth=y1 - 44.0,
            height=GRIPPER_GEAR_THICKNESS_MM,
        )
    )
    # Grip pad: a wall whose INNER face is the grip surface, taller than the arm
    # band for a real vertical contact patch, V-grooved to self-center a cylinder.
    px_inner, px_outer = s * pad_face, s * (pad_face + _PAD_T_MM)
    ops.append(
        Box(
            op_id=GRIP_PAD_OP_ID,
            mode=OpMode.ADD,
            position=Vec3(x=(px_inner + px_outer) / 2.0, y=(y0 + y1) / 2.0, z=_PAD_H_MM / 2.0),
            width=_PAD_T_MM,
            depth=y1 - y0,
            height=_PAD_H_MM,
        )
    )
    # The V-groove: two boxes rotated +-30 deg about Z cut into the inner face, meeting
    # at a vertical apex -- the payload contacts the two remaining edge ridges.
    for i, rot in enumerate((30.0, -30.0)):
        ops.append(
            Box(
                op_id=f"pad_groove_{i}",
                mode=OpMode.SUBTRACT,
                position=Vec3(
                    x=px_inner,
                    y=(y0 + y1) / 2.0,
                    z=_PAD_H_MM / 2.0,
                ),
                rotation=Rotation(z=rot),
                width=2.0 * _PAD_V_DEPTH_MM,
                depth=(y1 - y0) - 6.0,
                height=_PAD_H_MM + 2.0,
            )
        )
    return ops


def gripper_finger(
    *,
    name: str,
    side: str = "drive",
    payload: str = "beverage_can_355ml",
) -> tuple[PartProgram, list[Port]]:
    """A finger: a full involute gear disc at the root (the catalog generator's
    geometry) + strut + forearm + V-pad. ``side``:

    * ``drive``: bore + 4 horn screws = the CITED MG996R horn interface (the horn
      mount-pattern gate measures these); tooth clock 0 (a tooth on +X).
    * ``idler``: a plain bore sized from the ISO 2338 dowel + slip fit (the bore
      DERIVES from the registry, never a literal); tooth clock = the meshing phase
      for a partner on its -X side (``mesh_clock_deg``).
    """
    design = GripperDesign(payload)
    if side == "drive":
        bore = MG996R_HUB_CENTER_BORE_D_MM
        clock = 0.0
    elif side == "idler":
        bore = dowel_hole(GRIPPER_DOWEL_SIZE, "slip")
        clock = mesh_clock_deg(GRIPPER_GEAR_TEETH)
    else:
        raise ValueError(f"side must be 'drive' or 'idler', got {side!r}")

    # The gear is built SOLID (bore_d=0); the hub bore is an explicit SUBTRACT after
    # every ADD, so the bore wall is one clean measurable cylinder no matter what
    # solids fuse over the hub -- the mount-pattern gate measures exactly this face.
    ops: list[Operation] = [
        GearSolid(
            op_id="gear",
            mode=OpMode.ADD,
            module=GRIPPER_GEAR_MODULE,
            teeth=GRIPPER_GEAR_TEETH,
            thickness=GRIPPER_GEAR_THICKNESS_MM,
            bore_d=0.0,
            clock_deg=clock,
        )
    ]
    ops += _finger_ops(design, mirror=(side == "idler"))
    ops.append(
        Cylinder(
            op_id="hub_bore_cut",
            mode=OpMode.SUBTRACT,
            position=Vec3(z=GRIPPER_GEAR_THICKNESS_MM / 2.0),
            radius=bore / 2.0,
            height=GRIPPER_GEAR_THICKNESS_MM + 2.0,
        )
    )
    if side == "drive":
        bc_r = MG996R_HORN_SCREW_BC_D_MM / 2.0
        for i, (dx, dy) in enumerate(((bc_r, 0.0), (-bc_r, 0.0), (0.0, bc_r), (0.0, -bc_r))):
            ops.append(
                Cylinder(
                    op_id=f"horn_screw_{i}",
                    mode=OpMode.SUBTRACT,
                    position=Vec3(x=dx, y=dy, z=GRIPPER_GEAR_THICKNESS_MM / 2.0),
                    radius=MG996R_HORN_SCREW_D_MM / 2.0,
                    height=GRIPPER_GEAR_THICKNESS_MM + 2.0,
                )
            )
    # ADD-then-SUBTRACT so drills land in the finished union regardless of order.
    ops = [o for o in ops if o.mode == OpMode.ADD] + [o for o in ops if o.mode != OpMode.ADD]
    program = PartProgram(metadata=Metadata(name=name, units="mm"), operations=ops)
    ports = [
        Port(name="hub", origin=Vec3(), axis=Vec3(z=1.0), normal=Vec3(x=1.0)),
    ]
    return program, ports


def gripper_palm(
    *,
    name: str,
    foot_hole_r: float = 2.15,
) -> tuple[PartProgram, list[Port]]:
    """The palm: a printable skirted plate that cradles the servo shaft-up at the
    drive pivot (the cited window + 4-tab pattern) and carries the dowel boss at the
    idler pivot. The grip zone is FORWARD (+Y) of the plate, so the payload never
    meets the palm."""
    cd = GRIPPER_CENTER_DISTANCE_MM
    body_cx = -MG996R_SHAFT_OFFSET_MM  # servo body center (shaft at x=0)
    top = GRIPPER_PALM_TOP_MM
    # Plate extents: cover the tab pattern on the left, the boss on the right.
    x_min, x_max = body_cx - _HOLE_HX - 8.0, cd + 14.0
    y_min, y_max = -30.0, 32.0
    plate_w, plate_d = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

    ops: list[Operation] = [
        Box(
            op_id="plate",
            mode=OpMode.ADD,
            position=Vec3(x=cx, y=cy, z=top - _PLATE_T_MM / 2.0),
            width=plate_w,
            depth=plate_d,
            height=_PLATE_T_MM,
        ),
    ]
    wall_h = top - _PLATE_T_MM
    skirt_t = 4.0
    for op_id, px, py, w, d in (
        ("skirt_xp", x_max - skirt_t / 2.0, cy, skirt_t, plate_d),
        ("skirt_xm", x_min + skirt_t / 2.0, cy, skirt_t, plate_d),
        ("skirt_yp", cx, y_max - skirt_t / 2.0, plate_w, skirt_t),
        ("skirt_ym", cx, y_min + skirt_t / 2.0, plate_w, skirt_t),
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
    # Servo window + the cited tab pattern, centered on the BODY center.
    ops.append(
        Box(
            op_id="window",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=body_cx, y=0.0, z=top - _PLATE_T_MM / 2.0),
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
                position=Vec3(x=body_cx + sx * _HOLE_HX, y=sy * _HOLE_HY, z=top - _PLATE_T_MM / 2),
                radius=_TAB_HOLE_R_MM,
                height=_PLATE_T_MM + 2.0,
            )
        )
    # Dowel boss at the idler pivot + the press bore (diameter DERIVES from ISO 2338).
    boss_wall_h = GRIPPER_BOSS_TOP_MM - top
    ops.append(
        Cylinder(
            op_id="dowel_boss",
            mode=OpMode.ADD,
            position=Vec3(x=cd, y=0.0, z=top + boss_wall_h / 2.0),
            radius=7.0,
            height=boss_wall_h,
        )
    )
    ops.append(
        Cylinder(
            op_id="dowel_bore",
            mode=OpMode.SUBTRACT,
            position=Vec3(
                x=cd, y=0.0, z=(GRIPPER_BOSS_BORE_BOTTOM_MM + GRIPPER_BOSS_TOP_MM + 1.0) / 2.0
            ),
            radius=dowel_hole(GRIPPER_DOWEL_SIZE, "press") / 2.0,
            height=GRIPPER_BOSS_TOP_MM - GRIPPER_BOSS_BORE_BOTTOM_MM + 1.0,
        )
    )
    # Corner desk-mount holes.
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        ops.append(
            Cylinder(
                op_id=f"foot_{i}",
                mode=OpMode.SUBTRACT,
                position=Vec3(
                    x=cx + sx * (plate_w / 2.0 - 7.0),
                    y=cy + sy * (plate_d / 2.0 - 7.0),
                    z=top - _PLATE_T_MM / 2.0,
                ),
                radius=foot_hole_r,
                height=_PLATE_T_MM + 2.0,
            )
        )
    ops = [o for o in ops if o.mode == OpMode.ADD] + [o for o in ops if o.mode != OpMode.ADD]
    program = PartProgram(metadata=Metadata(name=name, units="mm"), operations=ops)
    ports = [
        # Servo cradle: origin at the BODY center on the plate top (the servo's mount
        # port lands here; its shaft axis then sits exactly at x=0).
        Port(
            name="cradle_mount",
            origin=Vec3(x=body_cx, y=0.0, z=top),
            axis=Vec3(z=1.0),
            normal=Vec3(x=1.0),
        ),
        # Dowel seat: the press-bore BOTTOM (the dowel's base lands here).
        Port(
            name="dowel_seat",
            origin=Vec3(x=cd, y=0.0, z=GRIPPER_BOSS_BORE_BOTTOM_MM),
            axis=Vec3(z=1.0),
            normal=Vec3(x=1.0),
        ),
    ]
    return program, ports


__all__ = [
    "GRIPPER_BOSS_BORE_BOTTOM_MM",
    "GRIPPER_CENTER_DISTANCE_MM",
    "GRIPPER_DOWEL_LENGTH_MM",
    "GRIPPER_DOWEL_SIZE",
    "GRIPPER_FINGER_PLANE_MM",
    "GRIPPER_GEAR_MODULE",
    "GRIPPER_GEAR_TEETH",
    "GRIPPER_GEAR_THICKNESS_MM",
    "GRIP_PAD_OP_ID",
    "GripperDesign",
    "gripper_finger",
    "gripper_palm",
]
