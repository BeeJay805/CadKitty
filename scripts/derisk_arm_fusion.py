"""FIRST ARM Step-0 de-risk (deterministic, $0): fuse a computed_body strut with a NEMA-17
motor mount into ONE validated part, and confirm the fusion validates clean.

Finding that shapes this: the AssemblyGraph validator (validate_assembly) is a MECHANISM
validator — it REQUIRES a moving joint (assembly.range_of_motion fails on FIXED-only) and a
pin-in-bore (assembly.pin_coaxial), and it does NOT run printability/min_wall/interface/mass.
A drone arm that HOLDS a motor is RIGID (the motor bolts on; nothing articulates), so it is a
fused PART, not a mechanism. The PART validator gives exactly the required set: mesh
single-body+watertight (the fusion integrity = the strut + mount are one valid body),
min_wall+overhang (strut printability), the nema17_face interface (motor mount certified), and
mass. So we validate the fused arm as a part. (An ARTICULATED arm — bench-007 — is the
assembly track; this rigid structural arm is not.)

Run: uv run python scripts/derisk_arm_fusion.py
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.primitives import Box, ComputedSolid, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.interfaces import place_nema17_face
from forge_mind.types import Vec3
from forge_mind.validator.run import validate


def build_arm(
    strut_size: float = 6.0, strut_len: float = 90.0, pad: float = 44.0, pad_thick: float = 6.0
) -> PartProgram:
    """A computed_body hex strut along X + a NEMA-17 motor-mount pad at the +X tip.

    The pad rests on the bed (its underside coplanar with the strut's flat bottom, so it is
    not an overhang) and extends past the strut so the bolt pattern + pilot bore sit in clear
    pad material. The strut + pad union into one watertight body; the motor face is drilled
    through the pad and declared as a nema17_face interface."""
    half_flats = (
        strut_size * math.sqrt(3.0)
    ) / 2.0  # hex across-flats/2 = the flat-bottom half-height
    tip_x = strut_len / 2.0  # strut spans [-tip_x, +tip_x] along X (builder centers at position)
    face_cx = tip_x + 15.0  # motor face center, past the strut tip -> clear pad
    pad_cx = face_cx
    pad_cz = -half_flats + pad_thick / 2.0  # pad bottom == strut bottom (on the bed)

    ops = [
        ComputedSolid(
            op_id="strut",
            mode=OpMode.ADD,
            kind="faceted_strut",
            section="hexagonal",
            size=strut_size,
            length=strut_len,
        ),
        Box(
            op_id="mount_pad",
            mode=OpMode.ADD,
            position=Vec3(x=pad_cx, y=0.0, z=pad_cz),
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
    return PartProgram(
        metadata=Metadata(
            name="first_arm", units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)
        ),
        operations=ops,
        interfaces=[declaration],
        constraints=Constraints(
            min_wall_thickness=4.0, max_overhang_deg=45.0, max_bbox=Vec3(x=220.0, y=60.0, z=40.0)
        ),
    )


def main() -> None:
    program = build_arm()
    model = compile_program(program, Path(tempfile.mkdtemp(prefix="arm_")))
    report = validate(model, program, material="PLA")
    ck = {c.id: c.status.value for c in report.checks}
    print("=== FIRST ARM (computed strut + NEMA-17 motor mount) — fused PART validation ===")
    for cid in sorted(ck):
        print(f"  {ck[cid]:<5} {cid}")
    m = report.metrics
    print(
        f"\n  mass={m.mass_g:.1f} g  volume={m.volume_mm3:.0f} mm^3  com=({m.center_of_mass.x:.1f},{m.center_of_mass.y:.1f},{m.center_of_mass.z:.1f})"
        if m
        else "  (no metrics)"
    )
    print(f"\n  PASSED={report.passed}")
    print(
        "  FUSION CONFIRMED"
        if report.passed
        else "  *** FUSION DID NOT VALIDATE — inspect above ***"
    )


if __name__ == "__main__":
    main()
