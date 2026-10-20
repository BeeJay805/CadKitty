"""FDM filament facts: cited thermal/physical properties for MATERIAL PLANNING (ADR-0087 N1).

The consumer thesis needs the system to reason about material ("some stuff needs heat resistant
materials" — founder, 2026-07-07) without ever CLAIMING thermal sufficiency: we have no thermal
instrument (L6), so these rows power ADVISORY ranking only ("PETG has the highest cited HDT of
your set"), always with the honesty ceiling riding the card verbatim.

Citations (researched 2026-07-07 via web search — the L2 discipline: every value below traces to
the sources listed; values are taken at the CONSERVATIVE (low) end of the cited bands, which
disagree by ±5-15 °C across test standards and formulations):

  * Simplify3D Ultimate Material Properties Table
    (simplify3d.com/resources/materials-guide/properties-table/) — Tg per material; TPU 95A Tg
    marked "n/a (rubbery)".
  * Formlabs FDM materials chart (formlabs.com/blog/strongest-3d-printing-materials/) — HDT @
    0.45 MPa: PLA 55, ABS 95, Nylon PA6/12 band 90-140 (low end taken).
  * UltiMaker PETG/PLA/ABS strength comparison (ultimaker.com/learn/...) — PETG HDT ~70,
    ABS HDT ~98 (Formlabs' 95 taken as the conservative value).
  * Xometry filament types table (xometry.com/resources/3d-printing/types-of-3d-printer-filaments/)
    — densities; ASA Tg band 98-112.
  * siraya.tech heat-resistance guide + simplemachining.com FDM guide — ASA HDT in the ABS band
    (mid-90s, conservative 95); Tg values.
  * Prusa forum ("glass transition temperatures: not what they seem") — nylon Tg is a poor
    service indicator; the wide PA6 spread (45-60) is real, 50 taken.

``hdt_045_c`` is the primary heat field (service-relevant deflection under load);
``tg_c`` is secondary. TPU's heat fields are honestly ``None`` — 95A elastomers have no
published HDT and Simplify3D declines a Tg; a value here would be invented (never do that).
Every entry is ``provisional=True``: cited, but NOT physically verified.
"""

from __future__ import annotations

from pydantic import Field

from forge_mind.dsl.standards.framework import Registry, StandardSpec

_SIMPLIFY3D = "Simplify3D Ultimate Material Properties Table (simplify3d.com)"
_FORMLABS = "Formlabs FDM materials chart (formlabs.com/blog/strongest-3d-printing-materials)"
_ULTIMAKER = "UltiMaker PETG/PLA/ABS strength comparison (ultimaker.com/learn)"
_XOMETRY = "Xometry filament types table (xometry.com)"
_HEAT_GUIDES = "siraya.tech heat-resistance guide; simplemachining.com FDM materials guide"
_PRUSA_FORUM = "Prusa forum: nylon Tg caveat (glass-transition-temperatures-not-what-they-seem)"


class FilamentSpec(StandardSpec):
    """One FDM filament's cited planning facts. ``None`` = not published for this material class
    (TPU 95A heat fields) — surfaced as unknown, never invented."""

    tg_c: float | None = Field(
        default=None, description="Glass transition (°C), conservative low end of cited band."
    )
    hdt_045_c: float | None = Field(
        default=None,
        description="Heat deflection @ 0.45 MPa (°C), conservative low end of cited band.",
    )
    density_g_cm3: float = Field(..., gt=0)
    flexible: bool = Field(default=False, description="Elastomeric (TPU-class) material.")


FILAMENTS: Registry[FilamentSpec] = Registry(
    "filament",
    (
        FilamentSpec(
            key="PLA",
            tg_c=57.0,
            hdt_045_c=55.0,
            density_g_cm3=1.24,
            source=f"Tg 55-60: {_SIMPLIFY3D}; HDT 55 @0.45MPa: {_FORMLABS}; density: {_XOMETRY}",
            provisional=True,
        ),
        FilamentSpec(
            key="PETG",
            tg_c=80.0,
            hdt_045_c=70.0,
            density_g_cm3=1.27,
            source=f"Tg ~80: {_SIMPLIFY3D}; HDT ~70: {_ULTIMAKER} + {_FORMLABS}; density: {_XOMETRY}",
            provisional=True,
        ),
        FilamentSpec(
            key="ABS",
            tg_c=105.0,
            hdt_045_c=95.0,
            density_g_cm3=1.04,
            source=f"Tg ~105: {_SIMPLIFY3D}; HDT 95 (Formlabs) / 98 (UltiMaker), low end taken; density: {_XOMETRY}",
            provisional=True,
        ),
        FilamentSpec(
            key="ASA",
            tg_c=100.0,
            hdt_045_c=95.0,
            density_g_cm3=1.08,
            source=f"Tg 98-112 low end: {_XOMETRY}; HDT in the ABS band, 95 taken: {_HEAT_GUIDES}; density: {_XOMETRY}",
            provisional=True,
        ),
        FilamentSpec(
            key="TPU",
            tg_c=None,  # Simplify3D: "n/a (rubbery)" — a number here would be invented
            hdt_045_c=None,  # no published HDT for 95A elastomers — never invented
            density_g_cm3=1.22,
            flexible=True,
            source=f"Tg/HDT not published for 95A elastomers: {_SIMPLIFY3D}; density 1.20-1.25: {_XOMETRY}",
            provisional=True,
        ),
        FilamentSpec(
            key="Nylon",
            tg_c=50.0,
            hdt_045_c=90.0,
            density_g_cm3=1.13,
            source=(
                f"PA6-type: Tg 45-60 (wide spread, poor service indicator: {_PRUSA_FORUM}); "
                f"HDT 90-140 @0.45MPa low end: {_FORMLABS}; density ~1.13"
            ),
            provisional=True,
        ),
    ),
)


def filament(key: str) -> FilamentSpec:
    return FILAMENTS.get(key)


__all__ = ["FILAMENTS", "FilamentSpec", "filament"]
