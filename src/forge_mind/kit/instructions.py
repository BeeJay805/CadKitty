"""Make-it instructions — deterministic, ADVISORY fabrication notes for a finished part.

Derived purely from the part's own data: the validated print orientation (``metadata``), the
declared minimum wall + overhang limit (``constraints``), the source material, and which hardware
the BOM lists. NO LLM, no geometry change, no I/O.

Everything here is GUIDANCE, not a gate. The numbers that are genuinely heuristic (layer height,
infill, insert install) are marked advisory and carry a caveat; the part's pass/fail verdict comes
only from the validator, never from these notes (no-false-PASS: a make-note can't pass a part, and
a conservative note can't fail one either).
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.dsl.program import PartProgram
from forge_mind.kit.bom import BillOfMaterials
from forge_mind.types import Vec3
from forge_mind.validator.metrics import DEFAULT_MATERIAL
from forge_mind.validator.report import CheckStatus, ValidationReport

# A common FDM default; reported as advisory, not derived from a fake formula.
DEFAULT_LAYER_HEIGHT_MM = 0.2
# Typical nozzle, used only to suggest a perimeter count for the declared min wall.
NOZZLE_WIDTH_MM = 0.4
_OVERHANG_CHECK_ID = "printability.overhang_max_angle"

_CAVEAT = (
    "Fabrication notes are advisory guidance derived from the part's geometry and declared "
    "constraints -- REPORTED, never a validation gate. Verify material choice, print orientation, "
    "and any PROVISIONAL hardware against a test print before committing to a production run."
)


class MakeInstructions(BaseModel):
    """How to fabricate one part. Advisory throughout; never gates the part."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    print_orientation: Vec3 = Field(
        ..., description="The orientation the part was validated in (its up vector while printing)."
    )
    orientation_note: str = Field(
        ..., description="Support / overhang guidance for that orientation."
    )
    material: str = Field(..., description="Suggested print material (from the source blueprint).")
    recommended_min_wall_mm: float | None = Field(
        default=None, description="The part's declared minimum wall thickness, if any (mm)."
    )
    layer_height_mm: float = Field(
        default=DEFAULT_LAYER_HEIGHT_MM, description="Advisory layer height (mm)."
    )
    wall_note: str = Field(..., description="Perimeter / wall guidance to realize the min wall.")
    infill_note: str = Field(..., description="Advisory infill guidance.")
    insert_install_note: str | None = Field(
        default=None, description="Heat-set install note; present iff the BOM lists inserts."
    )
    steps: list[str] = Field(default_factory=list, description="Ordered fabrication steps.")
    advisory: bool = Field(
        default=True, description="Always True: these notes never gate the part."
    )
    caveat: str = Field(default=_CAVEAT, description="Honest scope of the guidance.")


def _axis_name(v: Vec3) -> str:
    """Name a unit-axis orientation (+Z, -X, ...); fall back to the vector for an off-axis one."""
    for axis, (x, y, z) in {
        "+X": (1, 0, 0),
        "-X": (-1, 0, 0),
        "+Y": (0, 1, 0),
        "-Y": (0, -1, 0),
        "+Z": (0, 0, 1),
        "-Z": (0, 0, -1),
    }.items():
        if (v.x, v.y, v.z) == (x, y, z):
            return axis
    return f"({v.x:g}, {v.y:g}, {v.z:g})"


def _orientation_note(program: PartProgram, report: ValidationReport | None) -> str:
    up = _axis_name(program.metadata.print_orientation)
    limit = program.constraints.max_overhang_deg
    if report is not None:
        check = report.check(_OVERHANG_CHECK_ID)
        if check is not None and check.status == CheckStatus.PASS:
            return (
                f"Print with {up} up: validated support-free in this orientation "
                f"(no unsupported overhangs beyond {limit:.0f} deg)."
            )
        if check is not None and check.status == CheckStatus.FAIL:
            return (
                f"Print with {up} up, but overhangs were flagged here ({check.actual}); add support "
                "or reorient so large flat faces sit on the bed."
            )
    return (
        f"Print with {up} up; overhangs were checked against a {limit:.0f} deg limit. "
        "Reorient if your slicer flags unsupported faces."
    )


def _wall_note(min_wall: float | None) -> str:
    if min_wall is None:
        return "No explicit minimum wall declared; default slicer perimeters are fine."
    perimeters = max(2, math.ceil(min_wall / NOZZLE_WIDTH_MM))
    return (
        f"Design minimum wall is {min_wall:g} mm. At a {NOZZLE_WIDTH_MM:g} mm nozzle use about "
        f"{perimeters} perimeters (or solid walls) so thin features print fully dense."
    )


def _count(bom: BillOfMaterials, kind: str) -> int:
    return sum(line.quantity for line in bom.lines if line.kind == kind)


def _steps(
    program: PartProgram, bom: BillOfMaterials, material: str, orientation_note: str
) -> list[str]:
    """Ordered assembly steps, derived categorically from which hardware the BOM lists."""
    steps: list[str] = [
        f"Print the part in {material} at {DEFAULT_LAYER_HEIGHT_MM:g} mm layers. {orientation_note}"
    ]
    if (n := _count(bom, "insert")) > 0:
        steps.append(
            f"Heat-set the {n} threaded insert(s) into their bosses while the plastic is hot."
        )
    if (n := _count(bom, "bearing")) > 0:
        steps.append(f"Press the {n} bearing(s) into their seats until each is flush.")
    if (n := _count(bom, "dowel")) > 0:
        steps.append(f"Fit the {n} dowel pin(s) into their locating holes.")
    if (n := _count(bom, "magnet")) > 0:
        steps.append(f"Bond the {n} magnet(s) into their pockets (check polarity before gluing).")
    if (n := _count(bom, "nut")) > 0:
        steps.append(f"Seat the {n} hex nut(s) in their capture pockets.")
    if (n := _count(bom, "fastener")) > 0:
        steps.append(
            f"Fasten with the {n} screw(s) in the bill of materials (pick lengths to suit your stack-up)."
        )
    return steps


def build_instructions(
    program: PartProgram,
    bom: BillOfMaterials,
    *,
    material: str = DEFAULT_MATERIAL,
    report: ValidationReport | None = None,
) -> MakeInstructions:
    """Build advisory fabrication instructions for a finished part.

    ``bom`` decides the assembly steps + whether an insert-install note appears; ``report`` (if
    given) sharpens the orientation note with the actual overhang verdict. Pure + deterministic.
    """
    min_wall = program.constraints.min_wall_thickness
    orientation_note = _orientation_note(program, report)
    insert_note = (
        "Install heat-set inserts with a temperature-controlled soldering iron after printing (set "
        "the tip just above the filament's softening point) and seat each flush. The pilot bosses "
        "follow the insert datasheet (PROVISIONAL -- verify with a test print)."
        if _count(bom, "insert") > 0
        else None
    )
    return MakeInstructions(
        print_orientation=program.metadata.print_orientation,
        orientation_note=orientation_note,
        material=material,
        recommended_min_wall_mm=min_wall,
        layer_height_mm=DEFAULT_LAYER_HEIGHT_MM,
        wall_note=_wall_note(min_wall),
        infill_note=(
            "20-40% infill suits a typical bracket/mount; raise it (or use more perimeters) for a "
            "load-bearing part."
        ),
        insert_install_note=insert_note,
        steps=_steps(program, bom, material, orientation_note),
    )


__all__ = [
    "DEFAULT_LAYER_HEIGHT_MM",
    "NOZZLE_WIDTH_MM",
    "MakeInstructions",
    "build_instructions",
]
