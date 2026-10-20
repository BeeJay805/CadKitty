"""Categorical fit -> radial metric (mm): the single source of truth for clearances.

Mirrors the ``dsl/interfaces`` mechanical-fact modules: BOTH the solver (which
sizes an inserted pin from the bore it mates into) and the validator (which
measures the realized clearance) read these constants, so the geometry that gets
built can't drift from the geometry that gets checked. A negative value is an
interference (the pin is fatter than the bore) -- a deliberate defect class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge_mind.dsl.interfaces.fdm import PRINT_IN_PLACE_GAP_MM

if TYPE_CHECKING:
    from forge_mind.materials import Material

CLEARANCE_CLASS_MM = {
    "running_fit": 0.4,  # turns freely
    "loose_fit": 0.6,  # sloppy but assembles
    "interference": -0.1,  # press/seized -- pin larger than bore (a defect for a pivot)
    # A print-in-place moving gap, DERIVED from the nozzle (dsl/interfaces/fdm.py), not a
    # mechanical-fit constant (ADR-0009/0013). ~ running_fit today, but a separate class
    # because its source differs and the two diverge for any other nozzle.
    "print_in_place": PRINT_IN_PLACE_GAP_MM,
}


def clearance_mm(clearance_class: str) -> float:
    """Radial clearance (mm) for a fit class; raises on an unknown class."""
    try:
        return float(CLEARANCE_CLASS_MM[clearance_class])
    except KeyError:
        raise ValueError(
            f"unknown clearance_class {clearance_class!r}; known: {sorted(CLEARANCE_CLASS_MM)}"
        ) from None


def running_fit_clearance_mm(material: Material | None = None) -> float:
    """Slip-fit (running) clearance (mm) for the CHOSEN material.

    A running fit turns freely. Each material carries its OWN recommended slip-fit band
    (``Material.clearance_mm``); a tight-tolerance metal slips on far less gap than a stringy
    TPU. This returns that material's band midpoint so the gap tracks the material instead of
    one global number.

    Falls back to the global ``running_fit`` default (``CLEARANCE_CLASS_MM["running_fit"]``)
    when ``material`` is None, so existing callers that pass no material are unchanged.
    """
    if material is None:
        return float(CLEARANCE_CLASS_MM["running_fit"])
    return float(material.recommended_clearance_mm)


def minimum_running_diametral_clearance_mm(material: Material) -> float:
    """Minimum manufacturable bore-minus-pin diameter for a moving fit.

    This is the F2 law shared by plan-time joint checks and measured bore
    conformance. The 0.05 mm radial hysteresis prevents a marginally tight fit
    from becoming a hard failure; 0.05 mm is the absolute line-to-line floor.
    """
    return max(round(2.0 * (material.clearance_mm[0] - 0.05), 3), 0.05)


__all__ = [
    "CLEARANCE_CLASS_MM",
    "clearance_mm",
    "minimum_running_diametral_clearance_mm",
    "running_fit_clearance_mm",
]
