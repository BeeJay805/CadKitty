"""Non-printed hardware-component nodes for the AssemblyGraph (ADR-0012 spike).

A hardware node represents a real, bought part (a motor, later a bearing) that the
printed parts mate to but that is NOT printed: it contributes named frames (ports)
and a bounding box, no geometry program. This is the de-risk spike for the actuated
arm joint -- the smallest thing that lets a joint be anchored to a real hardware axis
(the motor SHAFT) rather than to a printed hole pattern.

Reuses the shared NEMA 17 facts (`dsl/interfaces/nema17.py`) so the motor's face and
shaft frames are single-sourced with everything else that knows the NEMA 17.

The node builder returns ``(None, ports)`` -- None program signals "not printed". The
solver places it by frame-mate exactly like any node (ports only); the compiler skips
it (no STL); the validator skips printability on it but still uses its placed frames.
"""

from __future__ import annotations

from forge_mind.assembly.joints import Port
from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.types import Vec3

# --- NEMA 17 hardware facts (single-source; bbox + shaft are hardware-node facts the
# face interface module doesn't carry). Standard single-shaft NEMA 17. ---
NEMA17_BODY_MM = 42.3  # square body footprint (NEMA 17 = 1.7 inch face)
NEMA17_BODY_LENGTH_MM = 40.0  # a common body length (varies by torque; coarse bbox only)
NEMA17_SHAFT_LENGTH_MM = 24.0  # usable shaft length from the face


NEMA17_BOSS_HEIGHT_MM = 2.0  # the 22 mm pilot boss stands proud of the face


def nema17_motor_ports() -> list[Port]:
    """The two named frames a NEMA 17 motor presents, coaxial by fact.

    - ``face``: the mounting-face plane. Origin at the face centre; axis +Z is the
      shaft/output direction (out of the face); normal +X is an in-plane reference.
    - ``shaft``: the output-shaft HUB-MATE plane: same axis as the face, origin
      lifted by the pilot-boss height. A hub mated at the bare face plane would sit
      INSIDE the 22 mm boss that stands proud of a real motor (the envelope mesh made
      that latent clash visible); a real coupling grips the shaft above the boss.

    Both frames share the face-centre x/y and +Z axis, so a part frame-mated to
    ``shaft`` is coaxial with the motor's output axis with no extra solver work.
    """
    return [
        Port(name="face", origin=Vec3(x=0.0, y=0.0, z=0.0), axis=Vec3(z=1.0), normal=Vec3(x=1.0)),
        Port(
            name="shaft",
            origin=Vec3(x=0.0, y=0.0, z=NEMA17_BOSS_HEIGHT_MM),
            axis=Vec3(z=1.0),
            normal=Vec3(x=1.0),
        ),
    ]


def nema17_motor_bbox_mm() -> Vec3:
    """Coarse motor bounding box (for later collision; not used by the static spike)."""
    return Vec3(x=NEMA17_BODY_MM, y=NEMA17_BODY_MM, z=NEMA17_BODY_LENGTH_MM)


def nema17_envelope_program() -> PartProgram:
    """A conservative watertight envelope of the motor (body + pilot boss + shaft) so the
    swept-collision gate can include the motor instead of skipping it (closes the
    ADR-0012 deferral for NEMA arms). Frame: face center at origin, shaft +Z, body -Z."""
    ops: list[Operation] = [
        Box(
            op_id="body",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=-NEMA17_BODY_LENGTH_MM / 2.0),
            width=NEMA17_BODY_MM,
            depth=NEMA17_BODY_MM,
            height=NEMA17_BODY_LENGTH_MM,
        ),
        # Pilot boss (22 mm register) + shaft, both on the face side (+Z).
        Cylinder(op_id="boss", mode=OpMode.ADD, position=Vec3(z=1.0), radius=11.0, height=2.0),
        Cylinder(
            op_id="shaft",
            mode=OpMode.ADD,
            position=Vec3(z=NEMA17_SHAFT_LENGTH_MM / 2.0),
            radius=2.5,
            height=NEMA17_SHAFT_LENGTH_MM,
        ),
    ]
    return PartProgram(metadata=Metadata(name="nema17_motor", units="mm"), operations=ops)


def dowel_pin_ports(length_mm: float, pivot_z_mm: float | None = None) -> list[Port]:
    """The frames an ISO 2338 dowel presents. Frame: origin at the press-end base,
    axis +Z along the pin.

    * ``seat``: the press end (origin at base) -- mates into a printed press-fit boss.
    * ``pivot``: where a printed part's bore rides as a plain-bearing pivot --
      ``pivot_z_mm`` above the base (default: 3/4 of the length).
    """
    pivot_z = pivot_z_mm if pivot_z_mm is not None else 0.75 * length_mm
    return [
        Port(name="seat", origin=Vec3(z=0.0), axis=Vec3(z=1.0), normal=Vec3(x=1.0)),
        Port(
            name="pivot",
            origin=Vec3(z=pivot_z),
            axis=Vec3(z=1.0),
            normal=Vec3(x=1.0),
        ),
    ]


def dowel_pin_program(diameter_mm: float, length_mm: float) -> PartProgram:
    """A dowel's collision solid IS the part (an exact cylinder, no approximation)."""
    return PartProgram(
        metadata=Metadata(name="dowel_pin", units="mm"),
        operations=[
            Cylinder(
                op_id="pin",
                mode=OpMode.ADD,
                position=Vec3(z=length_mm / 2.0),
                radius=diameter_mm / 2.0,
                height=length_mm,
            )
        ],
    )


def build_hardware_node(
    node_id: str, params: dict[str, object]
) -> tuple[PartProgram | None, list[Port]]:
    """Build a non-printed hardware node: ``(envelope_program | None, ports)``.

    The program of a hardware node is its collision/display ENVELOPE, never printed
    (PartNode.printed stays False; the compiler routes on that flag, not on the program).
    A None program means "frames only" -- the legacy ADR-0012 shape; the motion gate
    then reports the sweep as skipped (WARN) exactly as before.

    ``params['hardware']`` selects the kind: ``nema17_motor``, ``mg996r_servo``, or
    ``dowel_pin`` (which also reads ``size`` -- an ISO 2338 key like "6" -- and
    ``length_mm``; the diameter always comes from the cited registry, never a literal).
    """
    kind = params.get("hardware")
    if kind == "nema17_motor":
        return nema17_envelope_program(), nema17_motor_ports()
    if kind == "mg996r_servo":
        from forge_mind.assembly.servo import mg996r_envelope_program, mg996r_ports

        return mg996r_envelope_program(), mg996r_ports()
    if kind == "dowel_pin":
        from forge_mind.dsl.standards import DOWEL_PINS

        size = str(params.get("size", "6"))
        length = float(params.get("length_mm", 40.0))  # type: ignore[arg-type]
        pivot_raw = params.get("pivot_z_mm")
        pivot_z = float(pivot_raw) if pivot_raw is not None else None  # type: ignore[arg-type]
        diameter = DOWEL_PINS.get(size).diameter_mm
        return dowel_pin_program(diameter, length), dowel_pin_ports(length, pivot_z)
    raise ValueError(
        f"unknown hardware kind {kind!r}; known: ['nema17_motor', 'mg996r_servo', 'dowel_pin']"
    )


__all__ = [
    "NEMA17_BODY_LENGTH_MM",
    "NEMA17_BODY_MM",
    "NEMA17_SHAFT_LENGTH_MM",
    "build_hardware_node",
    "nema17_motor_bbox_mm",
    "nema17_motor_ports",
]
