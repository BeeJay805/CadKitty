"""Part B Step-0 de-risk (deterministic, $0): a LATTICE-bodied arm — a latticed beam fused with
a NEMA-17 motor mount into ONE rigid part — validates clean and is lighter than the SAME arm
built solid (equal envelope).

A thin faceted strut can't be latticed (too thin for cells), so a lattice arm's body is a
latticed BEAM (a vertical-walled cellular grid, the Step-B reuse of LatticeOp's Box-union) with
the mount pad fused at the +X tip. It's RIGID (the motor bolts on; nothing articulates), so it is
a fused PART validated by the part validator (NOT validate_assembly, which needs a moving joint).

Confirms: mesh single_body+watertight (the lattice walls + the pad fuse into one body),
min_wall+overhang (all walls vertical, 90deg -> validator-ready), the nema17_face certified, and
mass < the same beam built SOLID at equal envelope (the lattice payoff). Reuses the existing
_derive_lattice (Box-union) + place_nema17_face -- no new primitive (ADR-0006).

Run: uv run python scripts/derisk_lattice_arm.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.blueprint.arm_sizing import arm_pad_width, derive_nema17_arm_envelope
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.components.registry import NEMA_17
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.primitives import Box, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.computed_interpreter import _derive_lattice
from forge_mind.layout.interfaces import place_nema17_face
from forge_mind.types import Vec3
from forge_mind.validator.run import validate

_TIP_GAP_MM = 15.0  # matches arm_interpreter._TIP_GAP_MM


def _beam_dims(reach: str, min_wall: float) -> tuple[Vec3, float, float, float, float]:
    env = derive_nema17_arm_envelope(reach, min_wall)
    pad = arm_pad_width(NEMA_17, min_wall)
    beam_len = env.x - _TIP_GAP_MM - pad / 2.0
    return env, pad, beam_len, pad, env.z  # beam_y == pad (full-width beam), beam_z == envelope.z


def _mount_ops(
    beam_len: float, pad: float, beam_z: float, min_wall: float
) -> tuple[list[Operation], InterfaceDeclaration]:
    pad_thick = max(min_wall, 5.0)
    face_cx = beam_len / 2.0 + _TIP_GAP_MM
    pad_cz = (
        -beam_z / 2.0 + pad_thick / 2.0
    )  # pad underside coplanar with the beam bottom (on the bed)
    ops: list[Operation] = [
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
    decl = InterfaceDeclaration(
        name="motor_face",
        interface_type=InterfaceType.NEMA17_FACE,
        location=Vec3(x=face_cx, y=0.0, z=pad_cz + pad_thick / 2.0),
        normal=Vec3(x=0.0, y=0.0, z=1.0),
    )
    return ops, decl


def _program(
    name: str,
    body_ops: list[Operation],
    env: Vec3,
    min_wall: float,
    mount: list[Operation],
    decl: InterfaceDeclaration,
) -> PartProgram:
    return PartProgram(
        metadata=Metadata(name=name, units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=[*body_ops, *mount],
        interfaces=[decl],
        constraints=Constraints(min_wall_thickness=min_wall, max_overhang_deg=45.0, max_bbox=env),
    )


def build_lattice_arm(reach: str, min_wall: float, density: str) -> PartProgram:
    env, pad, beam_len, beam_y, beam_z = _beam_dims(reach, min_wall)
    walls = _derive_lattice("grid", density, Vec3(x=beam_len, y=beam_y, z=beam_z), min_wall)
    mount, decl = _mount_ops(beam_len, pad, beam_z, min_wall)
    return _program(f"lattice_arm_{density}", list(walls), env, min_wall, mount, decl)


def build_solid_beam_arm(reach: str, min_wall: float) -> PartProgram:
    env, pad, beam_len, beam_y, beam_z = _beam_dims(reach, min_wall)
    beam = Box(
        op_id="beam",
        mode=OpMode.ADD,
        position=Vec3(x=0.0, y=0.0, z=0.0),
        width=beam_len,
        depth=beam_y,
        height=beam_z,
    )
    mount, decl = _mount_ops(beam_len, pad, beam_z, min_wall)
    return _program("solid_beam_arm", [beam], env, min_wall, mount, decl)


_CERT = (
    "interfaces.motor_face.bolt_holes_present",
    "interfaces.motor_face.bolt_pattern_31mm_square",
    "interfaces.motor_face.shaft_bore_present",
)


def _check(program: PartProgram, tag: str, work: Path) -> tuple[bool, float | None]:
    model = compile_program(program, work / tag)
    report = validate(model, program, material="PLA")
    ck = {c.id: c.status.value for c in report.checks}
    certified = all(ck.get(c) == "pass" for c in _CERT)
    mass = report.metrics.mass_g if report.metrics else None
    key = [
        "mesh.single_body",
        "mesh.watertight",
        "features.min_wall_thickness",
        "printability.overhang_max_angle",
    ]
    print(
        f"  [{tag}] "
        + "  ".join(f"{c.split('.')[-1]}={ck.get(c, '-')}" for c in key)
        + f"  certified={certified}  passed={report.passed}  mass={mass and round(mass, 1)}g"
    )
    return bool(report.passed and certified), mass


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="lattice_arm_derisk_"))
    print(
        "=== LATTICE-bodied arm (latticed beam + NEMA-17 mount, fused) vs the SAME beam SOLID ==="
    )
    ok_all = True
    for reach in ("standard", "long"):
        for density in ("light", "medium"):
            lat_ok, lat_mass = _check(
                build_lattice_arm(reach, 4.0, density), f"{reach}/lattice-{density}", work
            )
            _sol_ok, sol_mass = _check(
                build_solid_beam_arm(reach, 4.0), f"{reach}/solid-beam", work
            )
            lighter = lat_mass is not None and sol_mass is not None and lat_mass < sol_mass
            saved = (
                f"{100 * (1 - lat_mass / sol_mass):.0f}%"
                if lighter and lat_mass and sol_mass
                else "-"
            )
            print(
                f"  -> {reach}/{density}: lattice {'LIGHTER' if lighter else 'NOT lighter'} than solid beam "
                f"(saved {saved})   valid={lat_ok}\n"
            )
            ok_all = ok_all and lat_ok and lighter
    print(
        f"{'ALL PASS — lattice arm validates + lighter than the solid beam at equal envelope' if ok_all else '*** SOME FAILED (HARD STOP)'}"
    )


if __name__ == "__main__":
    main()
