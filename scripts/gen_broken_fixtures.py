"""Generate the three deliberately-broken NEMA 17 fixtures from the good one.

Each broken fixture is the good fixture with EXACTLY ONE isolated defect, so the
validator's failure is unambiguously attributable to that defect:

- broken_walls:    motor flange 3mm thick (was 4mm)  -> tier 2 wall thickness
- broken_holes:    M3 bolt pattern at 25mm (was 31mm) -> tier 4 NEMA17 square
- broken_overhang: print up = +Y (was +Z)             -> tier 3 overhang

Re-run after changing the good fixture:
    uv run python scripts/gen_broken_fixtures.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "handwritten"
GOOD = FIXTURES / "nema17_mount_good.partdsl.json"


def _op(program: dict, op_id: str) -> dict:
    for op in program["operations"]:
        if op["op_id"] == op_id:
            return op
    raise KeyError(op_id)


def make_broken_walls(good: dict) -> dict:
    p = copy.deepcopy(good)
    p["metadata"]["name"] = "nema17_mount_broken_walls"
    p["metadata"]["description"] = "BROKEN: motor flange thinned to 3mm. Should fail tier 2."
    flange = _op(p, "motor_flange")
    flange["height"] = 3.0
    flange["position"]["z"] = 1.5  # keep it sitting on the bed (z=0..3)
    # The 5 motor-flange holes were drilled at z=2 over a 4mm plate; refit to 3mm.
    for op_id in (
        "motor_bolt_nw",
        "motor_bolt_ne",
        "motor_bolt_sw",
        "motor_bolt_se",
        "shaft_clearance",
    ):
        hole = _op(p, op_id)
        hole["position"]["z"] = 1.5
        hole["height"] = 3.0
    return p


def make_broken_holes(good: dict) -> dict:
    p = copy.deepcopy(good)
    p["metadata"]["name"] = "nema17_mount_broken_holes"
    p["metadata"]["description"] = "BROKEN: bolt pattern at 25mm not 31mm. Should fail tier 4."
    # Move the 4 M3 holes from +/-15.5 (31mm) to +/-12.5 (25mm pitch).
    moves = {
        "motor_bolt_nw": (-12.5, 12.5),
        "motor_bolt_ne": (12.5, 12.5),
        "motor_bolt_sw": (-12.5, -12.5),
        "motor_bolt_se": (12.5, -12.5),
    }
    for op_id, (x, y) in moves.items():
        hole = _op(p, op_id)
        hole["position"]["x"] = x
        hole["position"]["y"] = y
    return p


def make_broken_overhang(good: dict) -> dict:
    p = copy.deepcopy(good)
    p["metadata"]["name"] = "nema17_mount_broken_overhang"
    p["metadata"]["description"] = (
        "BROKEN: printed +Y up, ext flange becomes a ceiling. Should fail tier 3."
    )
    p["metadata"]["print_orientation"] = {"x": 0.0, "y": 1.0, "z": 0.0}
    return p


def main() -> None:
    good = json.loads(GOOD.read_text())
    for maker in (make_broken_walls, make_broken_holes, make_broken_overhang):
        broken = maker(good)
        out = FIXTURES / f"{broken['metadata']['name']}.partdsl.json"
        out.write_text(json.dumps(broken, indent=2) + "\n")
        print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
