"""Generic parametric part archetypes — NOT objects, just shapes with ports.

``knuckled_plate`` (a plate carrying N coaxial knuckles on one edge) and ``rod``
(a pin) are the only two archetypes the hinge spike needs. A hinge is composed
from them in the graph; there is deliberately no ``hinge`` archetype. Each builder
returns a single-body ``PartProgram`` plus the typed ``Port``s it exposes, in the
part's local frame.

Local frame convention for ``knuckled_plate``: the pivot/bore axis is the world X
line at z = thickness/2 (the plate's top surface), y = 0; the plate hangs in -Y.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from forge_mind.assembly.joints import Port
from forge_mind.dsl.interfaces.nema17_shaft import (
    SET_SCREW_CLEARANCE_RADIUS_MM as _SET_SCREW_CLEARANCE_RADIUS_MM,
)
from forge_mind.dsl.interfaces.nema17_shaft import (
    SHAFT_BORE_RADIUS_MM as _SHAFT_BORE_RADIUS_MM,
)
from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.types import Rotation, Vec3

# Cylinders are authored along local +Z; this rotation turns the axis to +X.
_AXIS_TO_X = Rotation(y=90.0)


def knuckled_plate(
    *,
    name: str,
    length_x: float,
    width_y: float,
    thickness: float,
    knuckle_centers_x: list[float],
    knuckle_outer_r: float,
    bore_r: float,
    knuckle_width: float,
    clearance_notches_x: list[float] | None = None,
) -> tuple[PartProgram, list[Port]]:
    """A plate (-Y) with knuckle barrels on its +Y edge, each bored for a pin.

    The bore axis is the local X line at (y=0, z=thickness/2). Single connected
    body: the plate fuses with every knuckle barrel.

    ``clearance_notches_x`` cuts a pocket in the plate's hinge edge at each given
    x. A mating part's knuckle barrel (centered on the shared axis) bulges across
    the pivot line into THIS plate; the notch makes room so two interleaved
    knuckled_plates do not interpenetrate when coplanar. This is generic clearance
    geometry (where the partner's barrel sits), not hinge-specific logic.
    """
    pivot_z = thickness / 2.0
    ops: list[Operation] = [
        Box(
            op_id="plate",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=-width_y / 2.0, z=0.0),
            width=length_x,
            depth=width_y,
            height=thickness,
        )
    ]
    for i, xc in enumerate(knuckle_centers_x):
        ops.append(
            Cylinder(
                op_id=f"knuckle_{i}",
                mode=OpMode.ADD,
                position=Vec3(x=xc, y=0.0, z=pivot_z),
                rotation=_AXIS_TO_X,
                radius=knuckle_outer_r,
                height=knuckle_width,
            )
        )
    for i, xc in enumerate(knuckle_centers_x):
        ops.append(
            Cylinder(
                op_id=f"bore_{i}",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=xc, y=0.0, z=pivot_z),
                rotation=_AXIS_TO_X,
                radius=bore_r,
                height=knuckle_width + 2.0,  # overrun so the bore fully pierces
            )
        )
    notch_depth_y = knuckle_outer_r + 0.5  # clear the partner barrel's bulge across the axis
    notch_width_x = knuckle_width + 2.0
    for i, xc in enumerate(clearance_notches_x or []):
        ops.append(
            Box(
                op_id=f"notch_{i}",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=xc, y=-notch_depth_y / 2.0, z=0.0),
                width=notch_width_x,
                depth=notch_depth_y,
                height=thickness + 2.0,
            )
        )
    program = PartProgram(metadata=Metadata(name=name, units="mm"), operations=ops)
    ports = [
        Port(
            name="bore",
            origin=Vec3(x=0.0, y=0.0, z=pivot_z),
            axis=Vec3(x=1.0, y=0.0, z=0.0),
            normal=Vec3(x=0.0, y=-1.0, z=0.0),  # points toward the plate
        )
    ]
    return program, ports


def rod(*, name: str, length_x: float, radius: float) -> tuple[PartProgram, list[Port]]:
    """A pin: one cylinder along local X, exposing a ``shaft`` axis port at its center."""
    ops: list[Operation] = [
        Cylinder(
            op_id="shaft",
            mode=OpMode.ADD,
            position=Vec3(),
            rotation=_AXIS_TO_X,
            radius=radius,
            height=length_x,
        )
    ]
    program = PartProgram(metadata=Metadata(name=name, units="mm"), operations=ops)
    ports = [
        Port(
            name="shaft",
            origin=Vec3(),
            axis=Vec3(x=1.0, y=0.0, z=0.0),
            normal=Vec3(x=0.0, y=0.0, z=1.0),
        )
    ]
    return program, ports


def flat_link(
    *,
    name: str,
    length_x: float,
    width_y: float,
    thickness: float,
    bore_r: float,
    end_margin: float = 4.0,
) -> tuple[PartProgram, list[Port]]:
    """A flat bar with a pivot bore (axis +Z) at each end -- the building block of a
    planar linkage. Exposes two axis ports, ``pivot_a`` (proximal) and ``pivot_b``
    (distal). A rod pins through a bore exactly like a knuckled_plate's bore.

    The two ports sit on OPPOSITE faces (pivot_a on the bottom, pivot_b on the top) so
    that chaining link_a.pivot_b -> link_b.pivot_a staggers each link one thickness up
    in Z -- the layering that lets a serial linkage articulate without the links
    colliding.
    """
    half_l = length_x / 2.0
    hole_x = half_l - end_margin
    ops: list[Operation] = [
        Box(
            op_id="bar",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=thickness / 2.0),
            width=length_x,
            depth=width_y,
            height=thickness,
        ),
        Cylinder(
            op_id="bore_a",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=-hole_x, y=0.0, z=thickness / 2.0),
            radius=bore_r,
            height=thickness + 2.0,
        ),
        Cylinder(
            op_id="bore_b",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=hole_x, y=0.0, z=thickness / 2.0),
            radius=bore_r,
            height=thickness + 2.0,
        ),
    ]
    program = PartProgram(metadata=Metadata(name=name, units="mm"), operations=ops)
    ports = [
        Port(
            name="pivot_a",
            origin=Vec3(x=-hole_x, y=0.0, z=0.0),
            axis=Vec3(z=1.0),
            normal=Vec3(x=1.0),
        ),
        Port(
            name="pivot_b",
            origin=Vec3(x=hole_x, y=0.0, z=thickness),
            axis=Vec3(z=1.0),
            normal=Vec3(x=1.0),
        ),
    ]
    return program, ports


def tube(
    *, name: str, length_x: float, outer_r: float, bore_r: float
) -> tuple[PartProgram, list[Port]]:
    """A hollow cylinder along local X -- a sleeve a rod slides through. One annular
    body. Exposes a ``bore`` axis port (the slide / insert channel), like a
    knuckled_plate's bore, so a rod mates into it the same way."""
    ops: list[Operation] = [
        Cylinder(
            op_id="wall",
            mode=OpMode.ADD,
            position=Vec3(),
            rotation=_AXIS_TO_X,
            radius=outer_r,
            height=length_x,
        ),
        Cylinder(
            op_id="bore",
            mode=OpMode.SUBTRACT,
            position=Vec3(),
            rotation=_AXIS_TO_X,
            radius=bore_r,
            height=length_x + 2.0,  # overrun so the bore fully pierces
        ),
    ]
    program = PartProgram(metadata=Metadata(name=name, units="mm"), operations=ops)
    ports = [
        Port(
            name="bore",
            origin=Vec3(),
            axis=Vec3(x=1.0, y=0.0, z=0.0),
            normal=Vec3(x=0.0, y=0.0, z=1.0),
        )
    ]
    return program, ports


def output_arm(
    *,
    name: str,
    length_x: float,
    width_y: float,
    thickness: float,
    hub_outer_r: float,
    hub_height: float,
    bore_r: float | None = None,
    set_screw_r: float | None = None,
    far_hole_r: float = 2.6,
) -> tuple[PartProgram, list[Port]]:
    """A driven robot-arm link: a flat beam with a shaft HUB at the driven end (+X=0)
    and a simple mounting hole at the far end (bench-007, ADR-0012).

    The hub is a cylindrical boss whose GRIPPING bore (axis +Z) slides onto the motor
    shaft -- the opposite of a clearance bore: it grips the 5 mm shaft to transmit
    torque, held by a radial M3 set screw into the bore wall. The ``hub`` port (origin
    at the hub centre, axis +Z) frame-mates ``motor.shaft`` so the arm lands coaxial
    with the shaft and revolves about it. ``bore_r`` / ``set_screw_r`` default to the
    single-sourced NEMA 17 shaft-coupling facts (the checker reads the same numbers).

    Local frame: hub centred at the origin, beam extending +X, printed flat (beam +Z up
    to ``thickness``; the hub rises to ``hub_height`` above the bed). The far-end hole
    is the generic "mounting feature" end -- a through-hole at +X near the tip.
    """
    if bore_r is None:
        bore_r = _SHAFT_BORE_RADIUS_MM
    if set_screw_r is None:
        set_screw_r = _SET_SCREW_CLEARANCE_RADIUS_MM
    far_x = length_x - hub_outer_r  # far-end hole sits one hub-radius in from the tip
    ops: list[Operation] = [
        # Beam: from the hub (x=0) out to length_x, centred in Y, lying on the bed.
        Box(
            op_id="beam",
            mode=OpMode.ADD,
            position=Vec3(x=length_x / 2.0, y=0.0, z=thickness / 2.0),
            width=length_x,
            depth=width_y,
            height=thickness,
        ),
        # Hub boss at the driven end (a cylinder rising +Z from the bed).
        Cylinder(
            op_id="hub",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=hub_height / 2.0),
            radius=hub_outer_r,
            height=hub_height,
        ),
        # Gripping bore through the hub (+Z), grips the shaft.
        Cylinder(
            op_id="hub_bore",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=0.0, y=0.0, z=hub_height / 2.0),
            radius=bore_r,
            height=hub_height + 2.0,  # overrun so the bore fully pierces
        ),
        # Radial M3 set screw into the bore wall (axis along +Y, through the hub side).
        Cylinder(
            op_id="set_screw",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=0.0, y=0.0, z=hub_height / 2.0),
            rotation=Rotation(x=90.0),  # cylinder axis -> +Y
            radius=set_screw_r,
            height=hub_outer_r + 2.0,  # from outside the boss into the bore
        ),
        # Far-end mounting hole (the generic "feature" end), through the beam (+Z).
        Cylinder(
            op_id="far_hole",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=far_x, y=0.0, z=thickness / 2.0),
            radius=far_hole_r,
            height=thickness + 2.0,
        ),
    ]
    program = PartProgram(metadata=Metadata(name=name, units="mm"), operations=ops)
    ports = [
        # The hub axis: origin at the hub centre on the bed, axis +Z (mates motor.shaft).
        Port(name="hub", origin=Vec3(x=0.0, y=0.0, z=0.0), axis=Vec3(z=1.0), normal=Vec3(x=1.0)),
    ]
    return program, ports


def face_plate(
    *,
    name: str,
    size_xy: float = 56.0,
    thickness: float = 2.0,
    pilot_bore_r: float = 11.2,
    screw_r: float = 1.75,
    screw_half_pitch: float = 15.5,
) -> tuple[PartProgram, list[Port]]:
    """A NEMA-17-face mounting plate: a square plate with the 22 mm pilot-boss
    clearance bore and the cited 4x M3 pattern on the 31 mm square. The physically
    true way a printed part bolts to a motor face (the boss stands 2 mm proud, so the
    plate is boss-height thin and the boss registers in the pilot bore). Exposes
    ``face_mount`` at the plate's UNDERSIDE center (axis +Z): align-mate it onto
    ``motor.face`` and the plate lies on the face with every hole on the cited point."""
    ops: list[Operation] = [
        Box(
            op_id="plate",
            mode=OpMode.ADD,
            position=Vec3(z=thickness / 2.0),
            width=size_xy,
            depth=size_xy,
            height=thickness,
        ),
        Cylinder(
            op_id="pilot_bore",
            mode=OpMode.SUBTRACT,
            position=Vec3(z=thickness / 2.0),
            radius=pilot_bore_r,
            height=thickness + 2.0,
        ),
    ]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        ops.append(
            Cylinder(
                op_id=f"face_screw_{i}",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=sx * screw_half_pitch, y=sy * screw_half_pitch, z=thickness / 2.0),
                radius=screw_r,
                height=thickness + 2.0,
            )
        )
    program = PartProgram(metadata=Metadata(name=name, units="mm"), operations=ops)
    ports = [
        Port(name="face_mount", origin=Vec3(), axis=Vec3(z=1.0), normal=Vec3(x=1.0)),
    ]
    return program, ports


def _arm_builders() -> dict[str, Callable[..., tuple[PartProgram, list[Port]]]]:
    # Servo-arm part builders live in assembly.arm_parts (the MG996R family); imported
    # lazily here so the archetype registry stays the single dispatch point without a
    # circular import at module load.
    from forge_mind.assembly.arm_parts import servo_base, servo_link

    return {"servo_base": servo_base, "servo_link": servo_link}


def _gripper_builders() -> dict[str, Callable[..., tuple[PartProgram, list[Port]]]]:
    # Gripper part builders (assembly.gripper_parts, ADR-0044); lazy like the arm's.
    from forge_mind.assembly.gripper_parts import gripper_finger, gripper_palm

    return {"gripper_palm": gripper_palm, "gripper_finger": gripper_finger}


_ARCHETYPES: dict[str, Callable[..., tuple[PartProgram, list[Port]]]] = {
    "knuckled_plate": knuckled_plate,
    "rod": rod,
    "tube": tube,
    "flat_link": flat_link,
    "output_arm": output_arm,
    "face_plate": face_plate,
    **_arm_builders(),
    **_gripper_builders(),
}


def build_node(
    archetype: str, node_id: str, params: dict[str, Any]
) -> tuple[PartProgram, list[Port]]:
    """Dispatch an archetype name + params to its builder. The node id becomes the part name."""
    fn = _ARCHETYPES.get(archetype)
    if fn is None:
        raise ValueError(f"unknown archetype {archetype!r}; known: {sorted(_ARCHETYPES)}")
    return fn(name=node_id, **params)


__all__ = ["build_node", "flat_link", "knuckled_plate", "output_arm", "rod", "tube"]
