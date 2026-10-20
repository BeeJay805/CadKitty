"""ADR-0007 Stage C: a second rectangular board through the SAME Pi4 tray path.

A generic PCB (70x50 board, 60x40 M3 mount, no ports/vents) is built by the
EXISTING ``interpret_pi4`` under the EXISTING ``pi4_case`` pattern -- no new
interpreter, no new ``Pattern`` enum -- and its mount validates via the now
parametric ``pi4_mount`` checker (rectangle + screw radius read from the interface
declaration). This proves the mount facts generalize across boards.

The Pi4-specific ``pi4_ports`` interface is deliberately out of scope: a generic
PCB has no connector banks, yet ``interpret_pi4`` still declares ``pi4_ports``
unconditionally (it can't know the board has none). So this test asserts the mount
+ structural checks, not ``pi4_ports`` -- itself the Stage C signal that PORTS,
unlike the mount, still need board-specific knowledge (a registry concern).
"""

from __future__ import annotations

import json
from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.layout.pi4_interpreter import interpret_electronics_tray
from forge_mind.layout.pi4_spec import ElectronicsTraySpec
from forge_mind.layout.spec import Pattern
from forge_mind.validator.run import validate

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "layout" / "generic_pcb_tray.layoutspec.json"

# Mount + structural checks the generic PCB tray must pass. NOT pi4_ports (a
# generic PCB has no connector banks; that interface is Pi4-specific).
MOUNT_AND_STRUCTURE = [
    "mesh.watertight",
    "mesh.single_body",
    "mesh.bbox_within_max",
    "features.min_wall_thickness",
    "printability.overhang_max_angle",
    "interfaces.pi4_mount.mount_holes_present",
    "interfaces.pi4_mount.hole_rectangle_58x49",
    "interfaces.pi4_mount.standoff_land",
]


def _spec() -> ElectronicsTraySpec:
    return ElectronicsTraySpec.model_validate(json.loads(FIXTURE.read_text()))


def test_generic_pcb_uses_the_electronics_tray_pattern() -> None:
    # Same pattern, no new enum: the generic board rides the electronics_tray
    # discriminator (serialized value still "pi4_case").
    assert _spec().pattern is Pattern.ELECTRONICS_TRAY


def test_mount_declaration_carries_this_boards_params() -> None:
    # The interpreter declares the actual 60x40 / M3 mount it built (not the Pi 4
    # nominal), so the checker validates THIS board.
    mount = next(d for d in interpret_electronics_tray(_spec()).interfaces if d.name == "pi4_mount")
    assert mount.params["hole_dx"] == 60.0
    assert mount.params["hole_dy"] == 40.0
    assert mount.params["screw_radius"] == 1.6


def test_generic_pcb_mount_validates_through_the_interpreter(tmp_path: Path) -> None:
    # THE DELIVERABLE: the same interpreter + pattern builds a second rectangular
    # board, and the parametric pi4_mount checker accepts its 60x40 / M3 mount.
    program = interpret_electronics_tray(_spec())
    report = validate(compile_program(program, tmp_path), program)
    for cid in MOUNT_AND_STRUCTURE:
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid
