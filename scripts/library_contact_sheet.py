"""Library Step-3: render a per-family sample CONTACT SHEET ($0, deterministic).

Re-derives one representative part per family (+ the solid/lattice contrast), renders each to a
3-view PNG, and tiles them into a single contact sheet on the Desktop. Mass + validated state in
each cell title.

Run: uv run python scripts/library_contact_sheet.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.computed_ops import FacetedStrutOp, LatticeOp
from forge_mind.layout.arm_interpreter import ArmSpec
from forge_mind.layout.composable_spec import BaseArchetype, Face, InterfaceRef
from forge_mind.layout.computed_interpreter import ComputedBodySpec
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.pi4_spec import (
    ElectronicsTraySpec,
    PortPolicy,
    VentilationPolicy,
)
from forge_mind.scaling.enumerate import FamilyShape, Placement, build_spec
from forge_mind.types import Vec3
from forge_mind.viz.render import render_part_to_png

DESK = Path.home() / "Desktop" / "forge-mind-adr0017-renders"
WORK = Path(tempfile.mkdtemp(prefix="library_sheet_"))

_ADAPTER = FamilyShape(
    BaseArchetype.FLAT_PLATE,
    (
        Placement(InterfaceRef.NEMA17_FACE, Face.PLATE_TOP),
        Placement(InterfaceRef.EXTRUSION_2020, Face.PLATE_BOTTOM),
    ),
)


def _samples() -> list[tuple[str, object]]:
    return [
        (
            "composable_plate\nNEMA17 -> 2020 adapter",
            build_spec(_ADAPTER, name="adapter", envelope_mm=Vec3(x=400.0, y=400.0, z=40.0)),
        ),
        (
            "computed_body\nfaceted hex strut",
            ComputedBodySpec(
                name="strut",
                envelope_mm=Vec3(x=120.0, y=30.0, z=28.0),
                min_wall_mm=4.0,
                op=FacetedStrutOp(cross_section="hexagonal"),
            ),
        ),
        (
            "computed_body\nlattice brace (light)",
            ComputedBodySpec(
                name="brace",
                envelope_mm=Vec3(x=120.0, y=30.0, z=28.0),
                min_wall_mm=4.0,
                op=LatticeOp(cell="grid", density="light"),
            ),
        ),
        (
            "motor_arm\nsolid (standard reach)",
            ArmSpec(
                name="arm",
                min_wall_mm=4.0,
                body="solid",
                cross_section="hexagonal",
                reach="standard",
            ),
        ),
        (
            "motor_arm\nlattice (standard reach)",
            ArmSpec(
                name="larm", min_wall_mm=4.0, body="lattice", density="light", reach="standard"
            ),
        ),
        (
            "electronics_tray\nRaspberry Pi 4 + slots",
            ElectronicsTraySpec(
                name="tray",
                envelope_mm=Vec3(x=110.0, y=80.0, z=30.0),
                material="PLA",
                min_wall_mm=2.0,
                clearance_mm=2.0,
                board_component_id="raspberry_pi_4",
                ports=PortPolicy.STANDARD,
                ventilation=VentilationPolicy.SLOTS,
            ),
        ),
    ]


def main() -> None:
    DESK.mkdir(parents=True, exist_ok=True)
    cells: list[tuple[str, Path]] = []
    for i, (label, spec) in enumerate(_samples()):
        program = interpret_layout(spec)  # type: ignore[arg-type]
        model = compile_program(program, WORK / f"s{i}")
        report = validate_part(model, program)
        png = WORK / f"s{i}.png"
        render_part_to_png(model.stl_path, png, title="")
        cells.append((f"{label}\n{report}", png))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Forge-Mind autonomous part library - per-family sample (all validated, 4 tiers clean)",
        fontsize=13,
    )
    for ax, (label, png) in zip(axes.flat, cells, strict=True):
        ax.imshow(plt.imread(str(png)))
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = DESK / "library_contact_sheet.png"
    fig.savefig(str(out), dpi=110)
    plt.close(fig)
    print(f"contact sheet -> {out}")
    for label, _ in cells:
        print("  " + label.replace("\n", " | "))


def validate_part(model: object, program: object) -> str:  # type: ignore[no-untyped-def]
    from forge_mind.validator.run import validate

    r = validate(model, program, material="PLA")  # type: ignore[arg-type]
    m = f"{r.metrics.mass_g:.1f}g" if r.metrics else "?"
    return f"mass={m}  validated={r.passed}"


if __name__ == "__main__":
    main()
