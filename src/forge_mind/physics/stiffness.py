"""Beam-theory stiffness/stress estimate — a deterministic, REPORTED-NOT-GATED metric.

Parallel to ``forge_mind.validator.metrics.compute_physics`` (the mass metric), but a
SEPARATE module the measure/report layer calls; it is NEVER wired into ``validate()`` and
never gates. Pure function in, dataclass out — NO LLM, no paid calls, no I/O.

What it does
------------
Reduce a beam-like part to an Euler-Bernoulli cantilever and report, for a unit tip load:

* cross-section area + second moment of area ``I`` (rectangle about its horizontal
  neutral axis, ``I = b*h^3 / 12``),
* bending stiffness ``EI``,
* cantilever tip stiffness ``k = 3*E*I / L^3`` (N/mm),
* peak bending stress per unit tip load ``sigma = F*L*c / I`` (MPa per N, F = 1 N,
  c = h/2).

Load + section convention (also stated in ``StiffnessEstimate.caveat``)
----------------------------------------------------------------------
* The part is a cantilever: built-in at one end, a point load F = 1 N at the free tip.
* The load acts along the THICKNESS direction (the bending-soft axis) — local Z for a
  strut/arm, the plate thickness for a bracket. So ``h`` = thickness (the dimension in
  the bending/load direction), ``b`` = width (the perpendicular in-plane dimension),
  ``c`` = h/2 (extreme fibre).
* SOLID, homogeneous, isotropic section — ignores infill, lattice voids, fillets,
  holes, and stress concentrations. Small-deflection Euler-Bernoulli theory.
This is a COMPARATIVE estimate (variant-vs-variant), NOT a certified FEA result.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.dsl.interfaces.declaration import InterfaceType
from forge_mind.dsl.primitives import Box, ComputedSolid, OpMode
from forge_mind.dsl.program import PartProgram

# Young's modulus E (MPa) per filament — TYPICAL FDM print values (printed parts are softer
# than the bulk polymer; treat as order-of-magnitude comparative moduli, not datasheet bulk).
# Keyed by upper-cased material; unknown materials fall back to PLA. Parallel to
# ``MATERIAL_DENSITY_G_CM3`` / ``density_for`` in the mass metric.
YOUNG_MODULUS_MPA: dict[str, float] = {
    "PLA": 3500.0,
    "PETG": 2100.0,
    "ABS": 2300.0,
    "TPU": 80.0,
    "NYLON": 2000.0,
}
DEFAULT_MATERIAL = "PLA"

# Rectangle aspect of the faceted strut builder (compiler/build_ops/computed.py):
# the section is Rectangle(2*size, 1.4*size) on a plane whose normal is the strut axis,
# so the two in-plane extents map to local Y (= 2*size) and local Z (= 1.4*size).
_RECT_WIDTH_FACTOR = 2.0  # local-Y extent / size
_RECT_THICK_FACTOR = 1.4  # local-Z extent / size  (the load/bending direction)

_CAVEAT = (
    "Euler-Bernoulli cantilever estimate: built-in at one end, unit point load (F=1 N) at "
    "the free tip along the thickness direction (h = thickness, b = width, c = h/2). "
    "Assumes a SOLID, homogeneous, isotropic section -- ignores infill, lattice voids, "
    "fillets, holes, and stress concentrations. E is a typical FDM-print modulus, not a "
    "datasheet bulk value. COMPARATIVE estimate (variant-vs-variant), not a certified FEA "
    "result; for hex sections the cross-section is reduced to an equal-area equivalent square."
)


def youngs_for(material: str) -> float:
    """Young's modulus (MPa) for a material, case-insensitive; unknown -> PLA."""
    return YOUNG_MODULUS_MPA.get(material.upper(), YOUNG_MODULUS_MPA[DEFAULT_MATERIAL])


class StiffnessEstimate(BaseModel):
    """A beam-theory stiffness/stress estimate for a beam-like part. REPORTED, never gated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str = Field(..., description="Beam family modelled: 'strut', 'arm', or 'bracket'.")
    length_mm: float = Field(..., description="Cantilever length L (the bending span), mm.")
    width_mm: float = Field(..., description="Cross-section width b (perpendicular to load), mm.")
    height_mm: float = Field(
        ..., description="Cross-section height h = thickness in the load direction, mm."
    )
    area_mm2: float = Field(..., description="Cross-section area used for the estimate, mm^2.")
    second_moment_mm4: float = Field(
        ..., description="Second moment of area I = b*h^3/12 about the neutral axis, mm^4."
    )
    young_modulus_mpa: float = Field(..., description="Young's modulus E used (MPa).")
    # Field names below are the public API contract (units in the name); the mixedCase is
    # intentional (E*I, N, MPa-per-N), so the pep8-naming check is suppressed per field.
    bending_stiffness_EI_Nmm2: float = Field(  # noqa: N815
        ..., description="Bending stiffness E*I (N*mm^2)."
    )
    tip_stiffness_N_per_mm: float = Field(  # noqa: N815
        ..., description="Cantilever tip stiffness k = 3*E*I / L^3 (N/mm)."
    )
    max_bending_stress_mpa_per_N: float = Field(  # noqa: N815
        ..., description="Peak bending stress per unit tip load, sigma = L*(h/2)/I (MPa per N)."
    )
    material: str = Field(..., description="Material name as supplied.")
    caveat: str = Field(..., description="Honest scope/assumptions of the estimate.")


def _rect_estimate(
    family: str, length_mm: float, width_mm: float, height_mm: float, material: str
) -> StiffnessEstimate:
    """Assemble the estimate from a resolved rectangular b x h cross-section + length L."""
    b, h, length = width_mm, height_mm, length_mm
    area = b * h
    second_moment = b * h**3 / 12.0  # about the horizontal neutral axis (load along h)
    e_mod = youngs_for(material)
    ei = e_mod * second_moment
    tip_stiffness = 3.0 * ei / length**3  # k = 3EI / L^3 (N/mm)
    c = h / 2.0  # extreme fibre from the neutral axis
    # sigma = F*L*c / I with F = 1 N -> MPa per N (N*mm * mm / mm^4 = N/mm^2 = MPa).
    max_stress_per_n = length * c / second_moment
    return StiffnessEstimate(
        family=family,
        length_mm=length,
        width_mm=b,
        height_mm=h,
        area_mm2=area,
        second_moment_mm4=second_moment,
        young_modulus_mpa=e_mod,
        bending_stiffness_EI_Nmm2=ei,
        tip_stiffness_N_per_mm=tip_stiffness,
        max_bending_stress_mpa_per_N=max_stress_per_n,
        material=material,
        caveat=_CAVEAT,
    )


def _computed_solid_section(op: ComputedSolid) -> tuple[float, float]:
    """Resolved (width b, thickness h) of a faceted strut's cross-section, mm.

    Rectangular: matches the builder -- local-Y extent (= 2*size) is the width, local-Z
    extent (= 1.4*size) is the thickness/load direction.
    Hexagonal: reduce the hexagon to an EQUAL-AREA equivalent square (b = h = sqrt(area)),
    so the reported b x h is self-consistent (area = b*h, I = b*h^3/12). Hex area for a
    circumradius ``size`` = (3*sqrt(3)/2) * size^2.
    """
    if op.section == "rectangular":
        return _RECT_WIDTH_FACTOR * op.size, _RECT_THICK_FACTOR * op.size
    hex_area = (3.0 * math.sqrt(3.0) / 2.0) * op.size**2
    side = math.sqrt(hex_area)
    return side, side


def _first_beam_computed_solid(program: PartProgram) -> ComputedSolid | None:
    """The first ADD ComputedSolid op (a computed strut/arm prism), if any."""
    for op in program.operations:
        if isinstance(op, ComputedSolid) and op.mode == OpMode.ADD:
            return op
    return None


def _has_motor_interface(program: PartProgram) -> bool:
    """True if the part declares a motor mount -> the beam is an 'arm', not a bare 'strut'."""
    return any(d.interface_type == InterfaceType.NEMA17_FACE for d in program.interfaces)


def _dominant_base_box(program: PartProgram) -> Box | None:
    """The base plate of a flat-plate part: the largest ADD Box by footprint, if the part
    reduces to a plate (a dominant thin slab). Returns None when no ADD Box exists."""
    boxes = [op for op in program.operations if isinstance(op, Box) and op.mode == OpMode.ADD]
    if not boxes:
        return None
    return max(boxes, key=lambda bx: bx.width * bx.depth)


def _is_flat_plate(box: Box) -> bool:
    """A plate is a slab: its thickness (Z height) is the smallest of its three extents and
    is meaningfully thinner than its in-plane span. Guards against treating a chunky block or
    a tall box as a bending plate."""
    in_plane_min = min(box.width, box.depth)
    return box.height <= in_plane_min and box.height * 2.0 <= max(box.width, box.depth)


def estimate_stiffness(
    program: PartProgram, material: str = DEFAULT_MATERIAL
) -> StiffnessEstimate | None:
    """Estimate cantilever stiffness/stress for a beam-like part, else ``None``.

    Returns ``None`` when the program is NOT reducible to a prismatic beam (e.g. an
    electronics tray / enclosure, or anything with no dominant beam/plate body). NEVER
    gates: this is a pure estimate the measure/report layer consumes; ``validate()`` does
    not call it.

    Detection (conservative, in priority order):
      1. A computed strut/arm prism (``ComputedSolid``) -> cantilever beam. ``length`` is
         the long axis; the cross-section comes from the resolved section. family = "arm"
         if the part also declares a motor mount, else "strut".
      2. Else a dominant flat base Box (a bracket/adapter plate) -> cantilever plate.
         width = the smaller in-plane dim (bending width), height = plate thickness,
         length = the larger in-plane dim. family = "bracket".
      3. Else -> ``None`` (not beam-like).
    """
    solid = _first_beam_computed_solid(program)
    if solid is not None:
        width_mm, thick_mm = _computed_solid_section(solid)
        family = "arm" if _has_motor_interface(program) else "strut"
        return _rect_estimate(family, solid.length, width_mm, thick_mm, material)

    box = _dominant_base_box(program)
    if box is not None and _is_flat_plate(box):
        in_plane = sorted((box.width, box.depth))  # [smaller, larger]
        return _rect_estimate(
            "bracket",
            length_mm=in_plane[1],  # bending span = the larger in-plane dimension
            width_mm=in_plane[0],  # bending width = the smaller in-plane dimension
            height_mm=box.height,  # plate thickness = the load/bending direction
            material=material,
        )

    return None


__all__ = [
    "YOUNG_MODULUS_MPA",
    "StiffnessEstimate",
    "estimate_stiffness",
    "youngs_for",
]
