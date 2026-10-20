"""Render a per-feature CONTACT SHEET of the composition breadth ($0, deterministic).

A curated 3x3 of composed parts -- single hardware features + face+hardware and hardware+hardware
combos -- each rendered + titled with its mass + validated state. To the Desktop folder.

Run: uv run python scripts/composition_contact_sheet.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forge_mind.compiler.compile import compile_program
from forge_mind.layout.composable_interpreter import interpret_composable_plate
from forge_mind.layout.composable_spec import (
    BaseArchetype,
    ComposablePlateSpec,
    Face,
    InterfacePlacement,
    InterfaceRef,
)
from forge_mind.types import Vec3
from forge_mind.validator.run import validate
from forge_mind.viz.render import render_part_to_png

DESK = Path.home() / "Desktop" / "forge-mind-adr0017-renders"
WORK = Path(tempfile.mkdtemp(prefix="composition_sheet_"))
_ENV = Vec3(x=260.0, y=180.0, z=110.0)


def _p(ref: InterfaceRef, cid: str | None, fit: str | None = None) -> InterfacePlacement:
    return InterfacePlacement(ref=ref, face=Face.PLATE_TOP, component_id=cid, fit=fit)


_SAMPLES: list[tuple[str, list[InterfacePlacement]]] = [
    ("clearance hole M5", [_p(InterfaceRef.CLEARANCE_HOLE, "M5", "normal")]),
    ("heat-set boss M4", [_p(InterfaceRef.HEAT_SET_BOSS, "M4")]),
    ("bearing seat 608", [_p(InterfaceRef.BEARING_SEAT, "608", "press")]),
    ("captive nut M5", [_p(InterfaceRef.CAPTIVE_NUT_POCKET, "M5")]),
    ("magnet pocket 8x3", [_p(InterfaceRef.MAGNET_POCKET, "8x3")]),
    ("counterbore M6", [_p(InterfaceRef.COUNTERBORE, "M6")]),
    (
        "NEMA17 + heat-set M3",
        [_p(InterfaceRef.NEMA17_FACE, None), _p(InterfaceRef.HEAT_SET_BOSS, "M3")],
    ),
    (
        "clearance M3 + bearing 625",
        [
            _p(InterfaceRef.CLEARANCE_HOLE, "M3", "normal"),
            _p(InterfaceRef.BEARING_SEAT, "625", "press"),
        ],
    ),
    (
        "dowel 6 + magnet 6x3",
        [_p(InterfaceRef.DOWEL_HOLE, "6", "press"), _p(InterfaceRef.MAGNET_POCKET, "6x3")],
    ),
]


def main() -> None:
    DESK.mkdir(parents=True, exist_ok=True)
    cells: list[tuple[str, Path]] = []
    for i, (label, ifaces) in enumerate(_SAMPLES):
        spec = ComposablePlateSpec(
            name=f"c{i}",
            envelope_mm=_ENV,
            material="PLA",
            min_wall_mm=4.0,
            base=BaseArchetype.FLAT_PLATE,
            interfaces=ifaces,
        )
        program = interpret_composable_plate(spec)
        model = compile_program(program, WORK / f"c{i}")
        report = validate(model, program, material="PLA")
        mass = f"{report.metrics.mass_g:.1f}g" if report.metrics else "?"
        png = WORK / f"c{i}.png"
        render_part_to_png(model.stl_path, png, title="")
        cells.append((f"{label}\nmass={mass}  validated={report.passed}", png))

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle(
        "Forge-Mind composition engine - standards-backed hardware composed onto a plate "
        "(all validated, 4 tiers + conformance)",
        fontsize=12,
    )
    for ax, (label, png) in zip(axes.flat, cells, strict=True):
        ax.imshow(plt.imread(str(png)))
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = DESK / "composition_contact_sheet.png"
    fig.savefig(str(out), dpi=110)
    plt.close(fig)
    print(f"contact sheet -> {out}")
    for label, _ in cells:
        print("  " + label.replace("\n", " | "))


if __name__ == "__main__":
    main()
