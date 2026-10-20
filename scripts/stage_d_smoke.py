"""Tiny Stage D LLM smoke (NOT a benchmark).

Does the LayoutSpecGenerator pick the right ``board_component_id`` through the
AI-facing path? n=3 raspberry_pi_4 + n=3 generic_rect_pcb. For each: run the LLM,
read the chosen board id, then deterministically interpret + compile + validate.
Reports id-correct rate, validate rate, cost, latency, and the ids chosen.

    uv run python scripts/stage_d_smoke.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from forge_mind.blueprint.schema import Blueprint
from forge_mind.layout.generator import LayoutSpecGenerator
from forge_mind.layout.pi4_spec import ElectronicsTraySpec
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.llm.client import api_key_available

PI4_BP = {
    "name": "pi4_case",
    "summary": (
        "An open-top case for a Raspberry Pi 4 Model B single-board computer, with "
        "its standard USB/Ethernet and USB-C/HDMI port openings."
    ),
    "envelope_mm": {"x": 95.0, "y": 70.0, "z": 30.0},
    "print_orientation": {"x": 0.0, "y": 0.0, "z": 1.0},
    "material": "PLA",
    "min_wall_mm": 2.0,
    "max_overhang_deg": 45.0,
    "features": [
        {
            "name": "tray",
            "kind": "wall",
            "purpose": "Floor + perimeter walls holding the Pi 4 on standoffs.",
            "approx_size_mm": {"x": 93.0, "y": 64.0, "z": 16.0},
        }
    ],
    "required_interfaces": [
        {"name": "pi4_mount", "interface_type": "pi4_mount", "on_feature": "tray", "facing": "+Z"},
        {"name": "pi4_ports", "interface_type": "pi4_ports", "on_feature": "tray", "facing": "+Y"},
    ],
}

GENERIC_BP = {
    "name": "generic_pcb_tray",
    "summary": (
        "An open tray to hold a generic 70 x 50 mm rectangular PCB on M3 standoffs. "
        "The board has no fixed connectors, so the tray needs NO port cutouts."
    ),
    "envelope_mm": {"x": 85.0, "y": 65.0, "z": 25.0},
    "print_orientation": {"x": 0.0, "y": 0.0, "z": 1.0},
    "material": "PLA",
    "min_wall_mm": 2.0,
    "max_overhang_deg": 45.0,
    "features": [
        {
            "name": "tray",
            "kind": "wall",
            "purpose": "Floor + perimeter walls holding the PCB on standoffs.",
            "approx_size_mm": {"x": 78.0, "y": 58.0, "z": 16.0},
        }
    ],
    "required_interfaces": [
        {"name": "pi4_mount", "interface_type": "pi4_mount", "on_feature": "tray", "facing": "+Z"},
    ],
}


def run_trials(label: str, bp_dict: dict, expected_id: str, gen: LayoutSpecGenerator) -> dict:
    bp = Blueprint.model_validate(bp_dict)
    ids: list[str | None] = []
    id_ok = 0
    val_ok = 0
    cost = 0.0
    lat = 0.0
    for i in range(3):
        t0 = time.time()
        spec, usage = gen(bp)
        dt = time.time() - t0
        board_id = spec.board_component_id if isinstance(spec, ElectronicsTraySpec) else None
        res = run_layout_pipeline(spec, Path(tempfile.mkdtemp()))
        ids.append(board_id)
        id_ok += int(board_id == expected_id)
        val_ok += int(res.passed)
        cost += usage.cost_usd
        lat += dt
        print(
            f"  [{label} {i + 1}/3] type={type(spec).__name__} board_id={board_id!r} "
            f"expected={expected_id!r} validate={'PASS' if res.passed else 'FAIL'} "
            f"cost=${usage.cost_usd:.4f} {dt:.1f}s" + (f" err={res.error}" if res.error else "")
        )
    return {"ids": ids, "id_ok": id_ok, "val_ok": val_ok, "cost": cost, "avg_lat": lat / 3}


def main() -> None:
    if not api_key_available():
        print("NO API KEY in .env — aborting smoke.")
        return
    gen = LayoutSpecGenerator()
    print("Stage D LLM smoke (n=3 each). LayoutSpecGenerator chooses board_component_id.\n")
    pi4 = run_trials("pi4", PI4_BP, "raspberry_pi_4", gen)
    print()
    generic = run_trials("generic", GENERIC_BP, "generic_rect_pcb", gen)
    print("\n=== SUMMARY ===")
    for name, r in (("raspberry_pi_4", pi4), ("generic_rect_pcb", generic)):
        print(
            f"{name}: id-correct {r['id_ok']}/3, validate {r['val_ok']}/3, "
            f"cost ${r['cost']:.4f}, avg latency {r['avg_lat']:.1f}s, ids={r['ids']}"
        )
    tot = pi4["cost"] + generic["cost"]
    print(f"total cost: ${tot:.4f} for 6 LLM generations")


if __name__ == "__main__":
    main()
