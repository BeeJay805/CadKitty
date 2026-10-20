"""Physics metrics (ADR-0019) — the first "Einstein tool".

Deterministic mass + center-of-mass from the compiled body (build123d gives solid volume
and mass-center directly) x a material->density table. REPORTED on the ValidationReport,
never a gate. The LLM never computes this — it is measured from geometry, and a future
optimize loop minimizes ``mass_g`` against it.

HONEST CAVEAT (also on ``PhysicsMetrics``): mass = volume x density assumes 100% solid, so
it is a COMPARATIVE estimate (variant-vs-variant), not a scale-accurate infill-print mass.
"""

from __future__ import annotations

from typing import Any

from build123d import CenterOf

from forge_mind.types import Vec3
from forge_mind.validator.report import PhysicsMetrics

# Filament densities (g/cm^3), a small deterministic registry parallel to the component
# registry. Keyed by upper-cased material name; unknown materials fall back to PLA.
MATERIAL_DENSITY_G_CM3: dict[str, float] = {
    "PLA": 1.24,
    "PETG": 1.27,
    "ABS": 1.04,
    "TPU": 1.21,
    "NYLON": 1.14,
}
DEFAULT_MATERIAL = "PLA"


def density_for(material: str) -> float:
    return MATERIAL_DENSITY_G_CM3.get(material.upper(), MATERIAL_DENSITY_G_CM3[DEFAULT_MATERIAL])


def compute_physics(part: Any, material: str = DEFAULT_MATERIAL) -> PhysicsMetrics:
    """Measure volume + mass + center-of-mass of a compiled build123d body."""
    volume_mm3 = float(part.volume)
    com = part.center(CenterOf.MASS)
    density = density_for(material)
    mass_g = (volume_mm3 / 1000.0) * density  # mm^3 -> cm^3, x g/cm^3
    return PhysicsMetrics(
        volume_mm3=volume_mm3,
        mass_g=mass_g,
        center_of_mass=Vec3(x=float(com.X), y=float(com.Y), z=float(com.Z)),
        material=material,
        density_g_cm3=density,
    )


__all__ = ["MATERIAL_DENSITY_G_CM3", "compute_physics", "density_for"]
