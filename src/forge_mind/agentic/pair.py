"""Fit-pairs: two parts that must MATE, certified by the fit engine (complex-parts arc P4).

Rung B of the arc (plan: complex-parts-arc.md). The multi-part seam already exists
(`produce_functional_part`); what was missing is the machinist's discipline this module adds:

  1. A ``MatingContract`` names WHICH feature of part A mates with WHICH feature of part B
     at WHICH ISO 286 fit class (e.g. "25 H7/g6" -- a 25 mm slide fit).
  2. Part B's mating dimension is derived from part A's AS-BUILT measured geometry, never
     from the nominal: you fit the part you HAVE. (Print processes drift; the drawing lies.)
  3. The pair is certified by the SHIPPED fit engine (`physics/fits.py`, ISO 286 + Lamé) on
     the MEASURED dims + exact two-mesh interference (`validator/collision.py`).

Tier law (ADR-0063/0065 lineage): the fit verdict ships REPORT-tier by default;
``FORGE_FIT_GATE=1`` promotes it to gating for bench/cell use. Every unmeasurable path
returns "unverifiable", never a silent pass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.physics.fits import evaluate, parse_designation


def fit_gate_enabled() -> bool:
    return os.environ.get("FORGE_FIT_GATE", "").strip().lower() in {"1", "true", "on"}


class MatingContract(BaseModel):
    """One mating relationship between a bore-like feature on A and a shaft-like on B."""

    model_config = ConfigDict(extra="forbid")

    designation: str = Field(..., description="ISO 286 fit, e.g. '25 H7/g6' or '12 H7/p6'.")
    bore_part: Literal["a", "b"] = Field("a", description="Which part carries the BORE.")
    bore_feature: str = Field(..., description="The bore feature's name in its brief.")
    shaft_feature: str = Field(..., description="The shaft feature's name in its brief.")
    engagement_mm: float | None = Field(
        None, description="Axial engagement length for press-fit math (None = clearance-only)."
    )
    hub_od_mm: float | None = Field(
        None, description="Hub outer diameter at the bore for Lamé (None = skip press math)."
    )


@dataclass
class FitVerdict:
    """The pair's measured fit certification -- honest three-state."""

    status: str  # "verified" | "violated" | "unverifiable"
    designation: str
    nominal_mm: float
    measured_bore_mm: float | None
    measured_shaft_mm: float | None
    measured_clearance_um: float | None
    band_clearance_um: tuple[float, float] | None
    detail: str
    press_fit: dict[str, Any] | None = None
    interference: dict[str, Any] | None = None
    gate: bool = field(default=False)  # True only when FORGE_FIT_GATE promotes a violation


def measure_cylinder_diameter(step_path: Path, *, near_mm: float, concave: bool) -> float | None:
    """The as-built diameter of the cylinder stack nearest ``near_mm`` (bore: concave=True;
    shaft/boss: concave=False). None when unmeasurable -- callers must treat None as
    'unverifiable', never as a pass."""
    try:
        from build123d import import_step

        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        stacks = [
            s
            for s in cluster_coaxial(extract_cylinders(import_step(str(step_path))))
            if s.concave is bool(concave)
        ]
        if not stacks:
            return None
        best = min(stacks, key=lambda s: abs(2.0 * float(s.radius) - near_mm))
        d = 2.0 * float(best.radius)
        # a "nearest" match wildly off the nominal is a different feature, not a measurement
        return d if abs(d - near_mm) <= max(2.0, 0.25 * near_mm) else None
    except Exception:
        return None


def derive_mating_dim(measured_bore_mm: float, designation: str) -> float:
    """Part B's target shaft dimension from part A's AS-BUILT bore: keep the designation's
    intended mid-band offset, re-anchored on the measured bore (fit the part you HAVE)."""
    return round(measured_bore_mm + _mid_shaft_offset(designation), 4)


def _mid_shaft_offset(designation: str) -> float:
    """The mid-band shaft offset (mm) the designation intends relative to the bore nominal."""
    from forge_mind.physics.fits import shaft_deviation_um

    nominal, _hole, (letter, grade) = parse_designation(designation)
    lo_um, hi_um = shaft_deviation_um(letter, grade, nominal)
    return ((lo_um + hi_um) / 2.0) / 1000.0


def certify_fit(
    contract: MatingContract,
    bore_step: Path,
    shaft_step: Path,
) -> FitVerdict:
    """Measure both mating features as-built and certify against the contract's fit band."""
    nominal, _hole_code, _shaft_code = parse_designation(contract.designation)
    bore_d = measure_cylinder_diameter(bore_step, near_mm=nominal, concave=True)
    shaft_d = measure_cylinder_diameter(shaft_step, near_mm=nominal, concave=False)
    if bore_d is None or shaft_d is None:
        which = "bore" if bore_d is None else "shaft"
        return FitVerdict(
            status="unverifiable",
            designation=contract.designation,
            nominal_mm=nominal,
            measured_bore_mm=bore_d,
            measured_shaft_mm=shaft_d,
            measured_clearance_um=None,
            band_clearance_um=None,
            detail=f"the {which} feature could not be measured on the as-built B-rep",
        )
    fit = evaluate(
        contract.designation,
        hub_od_mm=contract.hub_od_mm,
        engagement_mm=contract.engagement_mm,
    )
    clearance_um = (bore_d - shaft_d) * 1000.0
    lo_um = float(fit.fit.min_clearance_um)
    hi_um = float(fit.fit.max_clearance_um)
    ok = lo_um <= clearance_um <= hi_um
    detail = (
        f"measured bore {bore_d:.3f} / shaft {shaft_d:.3f} mm -> clearance "
        f"{clearance_um:+.0f} um vs the {contract.designation} band "
        f"[{lo_um:+.0f}, {hi_um:+.0f}] um: {'WITHIN' if ok else 'OUTSIDE'}"
    )
    return FitVerdict(
        status="verified" if ok else "violated",
        designation=contract.designation,
        nominal_mm=nominal,
        measured_bore_mm=bore_d,
        measured_shaft_mm=shaft_d,
        measured_clearance_um=clearance_um,
        band_clearance_um=(lo_um, hi_um),
        detail=detail,
        press_fit=(fit.press.model_dump() if fit.press is not None else None),
        gate=fit_gate_enabled() and not ok,
    )


__all__ = [
    "FitVerdict",
    "MatingContract",
    "certify_fit",
    "derive_mating_dim",
    "fit_gate_enabled",
    "measure_cylinder_diameter",
]
