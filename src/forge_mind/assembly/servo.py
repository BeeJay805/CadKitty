"""MG996R hobby-servo hardware facts: ports, mount pattern, and a collision envelope.

The servo is a BOUGHT part (a hardware node, ADR-0012): printed parts mate to it; it is
never printed. Unlike the original NEMA-17 spike node (frames only), the servo carries an
ENVELOPE PROGRAM -- a conservative solid (body + flange + horn stack) compiled like any
part so the swept-collision gate can include hardware in motion checks instead of skipping
them (the ADR-0012 deferral this module closes).

Every dimension here is cited to the TowerPro MG996R datasheet (nominal, standard-size
40x20 RC servo) and is PROVISIONAL until verified against a physical sample -- the
provenance tier for dimensions anchored here is therefore "sourced", never "verified"
(ADR-0040 anchor invariant). The numbers live ONLY here; the cradle generator and the
mount-pattern gate both read them, so geometry and check can never disagree.

LOCAL FRAME (the contract every port + the envelope share):
  * origin = the output-shaft axis where it exits the case top; +Z = shaft axis.
  * the case top is the z=0 plane; the body hangs below (z in [-BODY_H, 0]).
  * the shaft is OFFSET toward the case front: the body center sits at x = -SHAFT_OFFSET.
  * the mounting flange (tabs) underside is at z = -FLANGE_UNDERSIDE_BELOW_TOP.
  * the horn mating plane (where a printed link's hub face lands) is z = +HORN_PLANE_Z.
"""

from __future__ import annotations

from forge_mind.assembly.joints import Port
from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.types import Vec3

# --- MG996R facts (TowerPro datasheet, nominal; PROVISIONAL pending physical check) ---
MG996R_SOURCE = "TowerPro MG996R datasheet (towerpro.com.tw), nominal; provisional"
MG996R_BODY_L_MM = 40.7  # case length (x), the long horizontal axis
MG996R_BODY_W_MM = 19.7  # case width (y)
MG996R_BODY_H_MM = 42.9  # case bottom -> case top (z), excluding spline/horn
# The output spline sits near the case FRONT: its axis is offset from the body center.
MG996R_SHAFT_OFFSET_MM = 10.2  # body center is at x = -10.2 in the shaft frame
# Mounting flange (the screw tabs): a plate spanning past both case ends.
MG996R_FLANGE_SPAN_MM = 54.4  # overall length across both tabs (x)
MG996R_FLANGE_THICKNESS_MM = 2.5
MG996R_FLANGE_UNDERSIDE_BELOW_TOP_MM = 14.3  # case top -> flange underside (z drop)
# Tab screw holes: 4 holes on a 49.5 x 10.0 mm rectangle centered on the BODY center.
MG996R_MOUNT_HOLE_SPAN_X_MM = 49.5
MG996R_MOUNT_HOLE_SPAN_Y_MM = 10.0
MG996R_MOUNT_HOLE_DIAMETER_MM = (4.0, 4.8)  # acceptable clearance range, nominal 4.5
# Spline + round horn stack above the case top: the plane a horn-mounted hub mates to.
MG996R_HORN_PLANE_Z_MM = 7.6  # case top -> horn top face (spline 3.8 + horn 3.8, nominal)
MG996R_HORN_DISC_RADIUS_MM = 10.5  # 21 mm round horn, conservative envelope
# Horn screw interface a printed hub uses (round horn, nominal): center screw boss bore
# + 4 horn screws on a bolt circle.
MG996R_HUB_CENTER_BORE_D_MM = 8.1
MG996R_HORN_SCREW_D_MM = 2.2
# PHYSICALLY VERIFIED 14.0 by founder caliper 2026-07-22 (was 17.0, sourced -- wrong).
# The one dimension here at "verified" tier (ADR-0040); everything else stays sourced/
# provisional. Case study: single-source constants kept the hub generators and the
# mount-pattern gate consistent WITH EACH OTHER and wrong about the physical horn --
# only an external measurement could catch it. June-era horn-mated archive parts
# (gripper/arm families) embody the old 17.0 pattern; nothing printed/shipped.
MG996R_HORN_SCREW_BC_D_MM = 14.0
# Sweep capability (datasheet ~180 deg total): the family's default joint range.
MG996R_RANGE_DEG = (-90.0, 90.0)
# Stall torque (datasheet): the payload-sanity INFO check reads these; they are NEVER a
# PASS criterion ("can lift" depends on unmodeled friction/voltage/dynamics -- reported
# arithmetic only, ADR-0044). Cross-checked against the TowerPro product page and the
# components101/handsontec datasheet mirrors on 2026-06-10.
MG996R_STALL_TORQUE_KG_CM_4V8 = 9.4
MG996R_STALL_TORQUE_KG_CM_6V0 = 11.0


def mg996r_mount_hole_points_local() -> list[tuple[float, float, float]]:
    """The 4 tab-hole centers in the SERVO local frame, on the flange underside plane.
    The mount-pattern gate transforms these by the servo's world transform and demands a
    matching measured hole in the printed part bolted there."""
    cx = -MG996R_SHAFT_OFFSET_MM
    hx = MG996R_MOUNT_HOLE_SPAN_X_MM / 2.0
    hy = MG996R_MOUNT_HOLE_SPAN_Y_MM / 2.0
    z = -MG996R_FLANGE_UNDERSIDE_BELOW_TOP_MM
    return [(cx - hx, -hy, z), (cx - hx, hy, z), (cx + hx, -hy, z), (cx + hx, hy, z)]


def mg996r_ports() -> list[Port]:
    """The frames a MG996R presents.

    * ``horn``: the horn mating plane on the output axis (origin (0,0,+HORN_PLANE_Z),
      axis +Z). A printed link's hub port mates here (mate="align"), so the link lands
      coaxial with the output axis with the spline+horn stack visibly between case and
      hub -- the revolute anchor.
    * ``mount``: the flange frame (origin at body-center x/y on the flange underside,
      axis +Z). A printed cradle's mount port mates here (mate="align"), which drops the
      body into the pocket with the flange resting on the cradle ledges.
    """
    return [
        Port(
            name="horn",
            origin=Vec3(x=0.0, y=0.0, z=MG996R_HORN_PLANE_Z_MM),
            axis=Vec3(z=1.0),
            normal=Vec3(x=1.0),
        ),
        Port(
            name="mount",
            origin=Vec3(
                x=-MG996R_SHAFT_OFFSET_MM,
                y=0.0,
                z=-MG996R_FLANGE_UNDERSIDE_BELOW_TOP_MM,
            ),
            axis=Vec3(z=1.0),
            normal=Vec3(x=1.0),
        ),
    ]


def mg996r_envelope_program() -> PartProgram:
    """A conservative watertight envelope of the servo for swept-collision + display.

    Three ADD solids: the case box, the flange plate (full tab span), and ONE cylinder
    for the spline+horn stack. The horn cylinder is rotationally symmetric about the
    output axis, so its static envelope EXACTLY equals its swept volume under the
    revolute joint it drives -- no under-approximation there. The cable exit is not
    modeled (v1; conservative growth would add a side stub).
    """
    cx = -MG996R_SHAFT_OFFSET_MM
    ops: list[Operation] = [
        Box(
            op_id="case",
            mode=OpMode.ADD,
            position=Vec3(x=cx, y=0.0, z=-MG996R_BODY_H_MM / 2.0),
            width=MG996R_BODY_L_MM,
            depth=MG996R_BODY_W_MM,
            height=MG996R_BODY_H_MM,
        ),
        Box(
            op_id="flange",
            mode=OpMode.ADD,
            position=Vec3(
                x=cx,
                y=0.0,
                z=-(MG996R_FLANGE_UNDERSIDE_BELOW_TOP_MM - MG996R_FLANGE_THICKNESS_MM / 2.0),
            ),
            width=MG996R_FLANGE_SPAN_MM,
            depth=MG996R_BODY_W_MM,
            height=MG996R_FLANGE_THICKNESS_MM,
        ),
        Cylinder(
            op_id="horn_stack",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=MG996R_HORN_PLANE_Z_MM / 2.0),
            radius=MG996R_HORN_DISC_RADIUS_MM,
            height=MG996R_HORN_PLANE_Z_MM,
        ),
    ]
    return PartProgram(metadata=Metadata(name="mg996r_servo", units="mm"), operations=ops)


__all__ = [
    "MG996R_BODY_H_MM",
    "MG996R_BODY_L_MM",
    "MG996R_BODY_W_MM",
    "MG996R_FLANGE_SPAN_MM",
    "MG996R_FLANGE_THICKNESS_MM",
    "MG996R_FLANGE_UNDERSIDE_BELOW_TOP_MM",
    "MG996R_HORN_DISC_RADIUS_MM",
    "MG996R_HORN_PLANE_Z_MM",
    "MG996R_HORN_SCREW_BC_D_MM",
    "MG996R_HORN_SCREW_D_MM",
    "MG996R_HUB_CENTER_BORE_D_MM",
    "MG996R_MOUNT_HOLE_DIAMETER_MM",
    "MG996R_MOUNT_HOLE_SPAN_X_MM",
    "MG996R_MOUNT_HOLE_SPAN_Y_MM",
    "MG996R_RANGE_DEG",
    "MG996R_SHAFT_OFFSET_MM",
    "MG996R_SOURCE",
    "MG996R_STALL_TORQUE_KG_CM_4V8",
    "MG996R_STALL_TORQUE_KG_CM_6V0",
    "mg996r_envelope_program",
    "mg996r_mount_hole_points_local",
    "mg996r_ports",
]
