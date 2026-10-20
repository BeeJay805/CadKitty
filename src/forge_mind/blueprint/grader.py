"""Semantic grading for Blueprints.

The LLM won't match a reference blueprint field-for-field, so we grade on
invariants that capture "is this blueprint usable for the part": the right
interfaces are present, the envelope fits and is non-degenerate, the wall/overhang
constraints are sane, features exist, and every interface sits on a real feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge_mind.blueprint.schema import Blueprint

WALL_TOL_MM = 0.5
ENVELOPE_TOL_MM = 0.5
MIN_FEATURE_EXTENT_MM = 10.0  # a real part dimension, guards against a 0/tiny envelope


@dataclass
class BlueprintGrade:
    passed: bool
    failures: list[str] = field(default_factory=list)


def grade_blueprint(
    bp: Blueprint,
    *,
    required_interface_types: set[str],
    envelope_cap_mm: tuple[float, float, float],
    expected_min_wall_mm: float,
    min_features: int = 2,
) -> BlueprintGrade:
    failures: list[str] = []

    # 1. Required interface types all present.
    present = {i.interface_type.value for i in bp.required_interfaces}
    missing = required_interface_types - present
    if missing:
        failures.append(f"missing required interfaces: {sorted(missing)}")

    # 2. Envelope is non-degenerate and fits the cap under SOME orientation.
    # Axis assignment is arbitrary at the blueprint stage (the part can be
    # rotated), so compare sorted dimensions rather than per-axis.
    env = bp.envelope_mm.as_tuple()
    for axis, value in zip("xyz", env, strict=True):
        if value <= MIN_FEATURE_EXTENT_MM:
            failures.append(f"envelope {axis}={value} is degenerate (<= {MIN_FEATURE_EXTENT_MM}mm)")
    for env_dim, cap_dim in zip(sorted(env), sorted(envelope_cap_mm), strict=True):
        if env_dim > cap_dim + ENVELOPE_TOL_MM:
            failures.append(
                f"envelope does not fit the cap under any orientation "
                f"(sorted {sorted(env)} vs {sorted(envelope_cap_mm)})"
            )
            break

    # 3. Wall thickness matches intent.
    if abs(bp.min_wall_mm - expected_min_wall_mm) > WALL_TOL_MM:
        failures.append(
            f"min_wall_mm={bp.min_wall_mm} != expected {expected_min_wall_mm} (+/-{WALL_TOL_MM})"
        )

    # 4. Overhang constraint is sane.
    if not (0 < bp.max_overhang_deg <= 60):
        failures.append(f"max_overhang_deg={bp.max_overhang_deg} is out of the sane (0, 60] range")

    # 5. Enough features.
    if len(bp.features) < min_features:
        failures.append(f"only {len(bp.features)} features, need >= {min_features}")

    # 6. Print orientation is a single dominant axis (a clean build direction).
    o = bp.print_orientation.as_tuple()
    if max(abs(c) for c in o) < 0.9:
        failures.append(f"print_orientation {o} is not a clean axis direction")

    # 7. Every interface sits on a declared feature.
    feature_names = {f.name for f in bp.features}
    for iface in bp.required_interfaces:
        if iface.on_feature not in feature_names:
            failures.append(
                f"interface {iface.name!r} references unknown feature {iface.on_feature!r}"
            )

    return BlueprintGrade(passed=not failures, failures=failures)


__all__ = ["BlueprintGrade", "grade_blueprint"]
