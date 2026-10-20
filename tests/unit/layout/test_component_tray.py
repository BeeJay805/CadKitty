"""ADR-0007 Stage D: component-driven electronics trays.

Both boards come from the board registry via ``board_component_id`` -- same
``interpret_electronics_tray``, same ``Pattern.ELECTRONICS_TRAY``. The Pi 4 gets its
ports from the component facts; the generic PCB has none and is a valid CLOSED tray
that does not claim (and is not validated against) pi4_ports.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.layout.pi4_interpreter import interpret_electronics_tray
from forge_mind.layout.pi4_spec import ElectronicsTraySpec
from forge_mind.validator.run import validate

REPO = Path(__file__).resolve().parents[3]
PI4 = REPO / "tests" / "fixtures" / "layout" / "pi4_tray_component.layoutspec.json"
GENERIC = REPO / "tests" / "fixtures" / "layout" / "generic_rect_pcb_tray_component.layoutspec.json"

_STRUCTURE = [
    "mesh.watertight",
    "mesh.single_body",
    "mesh.bbox_within_max",
    "features.min_wall_thickness",
    "printability.overhang_max_angle",
    "interfaces.pi4_mount.mount_holes_present",
    "interfaces.pi4_mount.hole_rectangle_58x49",
    "interfaces.pi4_mount.standoff_land",
]


def _spec(path: Path) -> ElectronicsTraySpec:
    return ElectronicsTraySpec.model_validate(json.loads(path.read_text()))


def test_pi4_component_tray_validates_with_ports(tmp_path: Path) -> None:
    program = interpret_electronics_tray(_spec(PI4))
    assert any(d.name == "pi4_ports" for d in program.interfaces)
    report = validate(compile_program(program, tmp_path), program)
    assert report.passed
    for cid in [
        *_STRUCTURE,
        "interfaces.pi4_ports.long_edge_opening",
        "interfaces.pi4_ports.short_edge_opening",
    ]:
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid


def test_generic_component_tray_validates_without_ports(tmp_path: Path) -> None:
    program = interpret_electronics_tray(_spec(GENERIC))
    # No ports claimed: a generic PCB is a closed tray, not forced through pi4_ports.
    assert not any(d.name == "pi4_ports" for d in program.interfaces)
    report = validate(compile_program(program, tmp_path), program)
    assert report.passed
    for cid in _STRUCTURE:
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid
    # The port checks never ran (the interface was not declared).
    assert report.check("interfaces.pi4_ports.long_edge_opening") is None


def test_pi4_component_path_matches_legacy_geometry(tmp_path: Path) -> None:
    # The component path reproduces the legacy bench-002 Pi 4 tray bbox (93 x 64 x 16).
    model = compile_program(interpret_electronics_tray(_spec(PI4)), tmp_path)
    size = tuple(round(v, 1) for v in model.bounding_box.size.as_tuple())
    assert size == (93.0, 64.0, 16.0)
