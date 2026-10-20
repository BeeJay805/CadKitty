"""Material-aware evaluation of a measured part: the deterministic rules that turn a
:class:`~forge_mind.materials.Material` + measured geometry into author-facing issues and a
mass/cost/spec summary.

This is the bridge between the pure material library and the agentic validator. It is itself
PURE (no geometry kernel, no I/O) -- it takes ALREADY-MEASURED numbers (volume, minimum wall)
and the chosen material, so it unit-tests in microseconds and the same geometry can be shown
to PASS under one material and FAIL under a stricter one without touching build123d.

Only HARD, defensible limits gate ``passed``: a wall thinner than the material can hold is a
real "won't survive in this material" defect. Mass, cost, and the spec sheet are REPORTED.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge_mind.materials import Material, mass_grams, material_cost_usd, material_spec

# A wall within this of the material minimum still passes -- matches the validator's wall
# measurement tolerance (feature_checks.WALL_MEASUREMENT_TOL_MM) so jitter is not double-penalized.
WALL_TOL_MM = 0.2


@dataclass(frozen=True)
class MaterialReport:
    """The material verdict for one part: the spec numbers + any blocking limit violations."""

    material_id: str
    material_name: str
    mass_g: float
    cost_usd: float
    spec: dict[str, object]
    issues: list[str] = field(default_factory=list)
    blocks: bool = False  # True iff a hard material limit (sub-min wall) fails `passed`


def min_wall_issue(material: Material, measured_min_wall_mm: float | None) -> str | None:
    """Author-facing message iff the thinnest measured wall is below the material minimum, else None.

    Deterministic: the SAME geometry (measured wall) returns None for a lenient material and a
    message for a stricter one. None when the wall is unmeasurable (never invent a fault).
    """
    if measured_min_wall_mm is None:
        return None
    if measured_min_wall_mm < material.min_wall_mm - WALL_TOL_MM:
        return (
            f"thinnest wall {measured_min_wall_mm:.2f}mm is below {material.name}'s "
            f"{material.min_wall_mm:.2f}mm minimum -- {material.name} cannot reliably hold a wall "
            f"that thin; thicken the thinnest wall/rib to at least {material.min_wall_mm:.2f}mm "
            "or choose a finer-detail material."
        )
    return None


def evaluate_material(
    material: Material,
    *,
    volume_mm3: float,
    measured_min_wall_mm: float | None = None,
) -> MaterialReport:
    """Evaluate a part of ``volume_mm3`` against ``material``: spec + mass/cost + wall gate.

    ``measured_min_wall_mm`` (when provided) is checked against the material minimum -- the one
    HARD, blocking material rule. Mass/cost/spec are always computed and reported.
    """
    issues: list[str] = []
    blocks = False
    wall = min_wall_issue(material, measured_min_wall_mm)
    if wall is not None:
        issues.append(wall)
        blocks = True

    m = mass_grams(volume_mm3, material)
    return MaterialReport(
        material_id=material.id,
        material_name=material.name,
        mass_g=m,
        cost_usd=material_cost_usd(m, material),
        spec=material_spec(material, volume_mm3),
        issues=issues,
        blocks=blocks,
    )


__all__ = ["WALL_TOL_MM", "MaterialReport", "evaluate_material", "min_wall_issue"]
