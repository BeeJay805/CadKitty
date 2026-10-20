"""Semantic grading for IntentSpecs.

Different phrasings of the same request must extract to the same facts. We grade
on invariants: the required interfaces are present, and the stated constraints
(wall, envelope, material) were extracted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge_mind.intent.schema import IntentSpec

WALL_TOL_MM = 0.1
ENVELOPE_TOL_MM = 0.5


@dataclass
class IntentGrade:
    passed: bool
    failures: list[str] = field(default_factory=list)


def grade_intent(
    spec: IntentSpec,
    *,
    required_interface_types: set[str],
    expected_min_wall_mm: float | None = None,
    expected_envelope_mm: tuple[float, float, float] | None = None,
    expected_material: str | None = None,
) -> IntentGrade:
    failures: list[str] = []

    present = {m.interface_type.value for m in spec.mating_interfaces}
    missing = required_interface_types - present
    if missing:
        failures.append(f"missing interfaces: {sorted(missing)}")

    if expected_min_wall_mm is not None:
        actual = spec.constraints.min_wall_mm
        if actual is None or abs(actual - expected_min_wall_mm) > WALL_TOL_MM:
            failures.append(f"min_wall_mm={actual} != expected {expected_min_wall_mm}")

    if expected_envelope_mm is not None:
        env = spec.constraints.max_envelope_mm
        if env is None or len(env) != 3:
            failures.append(f"max_envelope_mm not extracted (got {env})")
        else:
            # Orientation-independent: compare sorted dims.
            for a, e in zip(sorted(env), sorted(expected_envelope_mm), strict=True):
                if abs(a - e) > ENVELOPE_TOL_MM:
                    failures.append(f"envelope {sorted(env)} != {sorted(expected_envelope_mm)}")
                    break

    if expected_material is not None:
        mat = (spec.constraints.material or "").lower()
        if expected_material.lower() not in mat:
            failures.append(f"material {spec.constraints.material!r} != {expected_material!r}")

    return IntentGrade(passed=not failures, failures=failures)


__all__ = ["IntentGrade", "grade_intent"]
