"""First-arm interpreter — fuse a computed_body strut with a NEMA-17 motor mount into ONE
validated part (the north-star "mechanical interface + computed shell", single rigid arm).

A drone arm that HOLDS a motor is RIGID (the motor bolts on; nothing articulates), so it is a
fused PART, not an articulated assembly (that is the bench-007 track). The part validator
covers everything: mesh single-body+watertight (the fusion integrity), min_wall+overhang (the
strut), the nema17_face interface (the motor mount certified), and mass.

The LLM picks only the strut cross-section; every dimension is DERIVED here from coarse intent
(envelope + min_wall) + the shared NEMA-17 facts. An envelope that cannot host the motor mount
raises LayoutError up front.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.blueprint.arm_sizing import derive_nema17_arm_envelope
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.nema17 import (
    BOLT_PITCH_MM,
    M3_CLEARANCE_RADIUS_MM,
)
from forge_mind.dsl.primitives import Box, ComputedSolid, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.computed_interpreter import _derive_lattice
from forge_mind.layout.interfaces import place_nema17_face
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.spec import Pattern
from forge_mind.types import Vec3

_TIP_GAP_MM = 15.0  # clearance between the strut tip and the motor-face centre (clear pad)
_PAD_MARGIN_MM = 4.0  # pad extends this past the bolt square + land on each side
_STRUT_FRACTION = 0.15  # strut circumradius as a fraction of the mount pad width
_STRUT_FLOOR_MM = 4.0
_STRUT_CAP_MM = 8.0


class ArmSpec(BaseModel):
    """A single drone-arm: a structural strut carrying a NEMA-17 motor mount at the tip.

    Categorical-only LLM surface (structure beats scale): the LLM picks the strut cross-section
    and a coarse ``reach`` class; the envelope is DERIVED from the motor + reach (Part A), never
    LLM-sized. ``envelope_mm`` stays available for direct/explicit construction (tests, fixtures),
    and when given it is used verbatim (the existing arm geometry is unchanged). The motor is
    NEMA 17 (the first arm); other sizes are deferred.

    ``body`` chooses the arm's structural body (Part B): a ``solid`` faceted strut (the default,
    byte-identical to the original arm) or a ``lattice`` beam — a vertical-walled cellular grid
    (the LatticeOp Box-union reuse) that is lighter than the same beam solid. ``density`` tunes
    the lattice (ignored for a solid body)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: Literal[Pattern.MOTOR_ARM] = Pattern.MOTOR_ARM
    name: str = Field(..., min_length=1, max_length=128)
    envelope_mm: Vec3 | None = None
    material: str = "PLA"
    min_wall_mm: float = Field(..., gt=0)
    max_overhang_deg: float = Field(45.0, gt=0, le=90)
    cross_section: Literal["rectangular", "hexagonal"] = "hexagonal"
    reach: Literal["compact", "standard", "long"] = "standard"
    body: Literal["solid", "lattice"] = "solid"
    density: Literal["light", "medium", "dense"] = "medium"


def _solid_strut_body(
    cross_section: Literal["rectangular", "hexagonal"], body_len: float, pad: float
) -> tuple[list[Operation], float]:
    """The original arm body: one obtuse-dihedral faceted strut (validator-ready, modest section).
    Returns (ops, body half-height). Byte-identical to the pre-lattice arm."""
    strut_size = min(_STRUT_CAP_MM, max(_STRUT_FLOOR_MM, _STRUT_FRACTION * pad))
    # The strut's Z half-extent depends on the section (the builder orients hex flat-bottom,
    # across-flats = size*sqrt(3); the rectangle's Z-extent is 2*size). The pad underside must
    # sit on the body's bottom (== the bed) or it becomes a 90deg overhang.
    half_height = (
        (strut_size * math.sqrt(3.0)) / 2.0 if cross_section == "hexagonal" else strut_size
    )
    body: list[Operation] = [
        ComputedSolid(
            op_id="strut",
            mode=OpMode.ADD,
            kind="faceted_strut",
            section=cross_section,
            size=strut_size,
            length=body_len,
        ),
    ]
    return body, half_height


def _lattice_beam_body(
    body_len: float, pad: float, envelope: Vec3, min_wall: float, density: str
) -> tuple[list[Operation], float]:
    """The lattice arm body (Part B): a full-section beam latticed into a vertical-walled grid
    (the LatticeOp Box-union reuse) -> lighter than the same beam solid. Beam = full envelope
    height x the mount-pad width; the compiler unions the walls; the mount pad fuses at the tip.
    Returns (wall ops, beam half-height)."""
    beam_z = envelope.z
    walls = _derive_lattice("grid", density, Vec3(x=body_len, y=pad, z=beam_z), min_wall)
    return walls, beam_z / 2.0


def _derive_arm_ops(spec: ArmSpec, envelope: Vec3) -> tuple[list[Operation], InterfaceDeclaration]:
    min_wall = spec.min_wall_mm
    # The motor-mount pad must host the NEMA-17 bolt square + a min_wall land on each side.
    pad = BOLT_PITCH_MM + 2.0 * (M3_CLEARANCE_RADIUS_MM + min_wall) + 2.0 * _PAD_MARGIN_MM
    if pad > envelope.y + 1e-6:
        raise LayoutError(
            f"motor_arm: the NEMA-17 mount needs a {pad:.0f}mm-wide pad but envelope.y is {envelope.y:.0f}mm"
        )
    pad_thick = max(min_wall, 5.0)

    # The body spans the envelope minus the tip mount; the pad sits at the tip and extends past
    # the body so the bolt pattern + bore land in clear pad material.
    body_len = envelope.x - _TIP_GAP_MM - pad / 2.0
    if body_len <= pad / 2.0:
        raise LayoutError(
            f"motor_arm: envelope.x {envelope.x:.0f}mm too short for a body + the tip motor mount"
        )

    if spec.body == "lattice":
        body_ops, body_half_z = _lattice_beam_body(body_len, pad, envelope, min_wall, spec.density)
    else:
        body_ops, body_half_z = _solid_strut_body(spec.cross_section, body_len, pad)

    if body_half_z * 2.0 > envelope.z + 1e-6 or pad_thick > envelope.z + 1e-6:
        raise LayoutError(f"motor_arm: arm is taller than envelope.z {envelope.z:.0f}mm")

    face_cx = body_len / 2.0 + _TIP_GAP_MM
    pad_cz = (
        -body_half_z + pad_thick / 2.0
    )  # pad underside coplanar with the body bottom (on the bed)

    ops: list[Operation] = [
        *body_ops,
        Box(
            op_id="mount_pad",
            mode=OpMode.ADD,
            position=Vec3(x=face_cx, y=0.0, z=pad_cz),
            width=pad,
            depth=pad,
            height=pad_thick,
        ),
        *place_nema17_face(center=Vec3(x=face_cx, y=0.0, z=pad_cz), thickness=pad_thick),
    ]
    declaration = InterfaceDeclaration(
        name="motor_face",
        interface_type=InterfaceType.NEMA17_FACE,
        location=Vec3(x=face_cx, y=0.0, z=pad_cz + pad_thick / 2.0),
        normal=Vec3(x=0.0, y=0.0, z=1.0),
    )
    return ops, declaration


def interpret_arm(spec: ArmSpec) -> PartProgram:
    # Span-sizing (Part A): derive the envelope from the motor + reach class when the LLM/caller
    # supplied none. An explicit envelope (tests, fixtures) is honored verbatim -> the existing
    # arm geometry is byte-identical.
    envelope = (
        spec.envelope_mm
        if spec.envelope_mm is not None
        else derive_nema17_arm_envelope(spec.reach, spec.min_wall_mm)
    )
    ops, declaration = _derive_arm_ops(spec, envelope)
    return PartProgram(
        metadata=Metadata(name=spec.name, units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=ops,
        interfaces=[declaration],
        constraints=Constraints(
            min_wall_thickness=spec.min_wall_mm,
            max_overhang_deg=spec.max_overhang_deg,
            max_bbox=envelope,
        ),
    )


__all__ = ["ArmSpec", "interpret_arm"]
