"""Deterministic beam-theory stiffness/stress estimate (a parallel "Einstein tool").

A SEPARATE, REPORTED-NOT-GATED physics estimate, mirroring how
``forge_mind.validator.metrics`` measures mass — but living in its own package and
called by the measure/report layer, never by ``validate()``. It is a pure,
deterministic function (NO LLM, no I/O) that reduces a beam-like part (a strut, an
arm, a bracket/plate) to an Euler-Bernoulli cantilever and reports its tip
stiffness and bending stress per unit tip load. Non-beam parts (enclosures,
electronics trays) return ``None`` — the estimate is conservative about what it
claims to model.
"""

from forge_mind.physics.stiffness import (
    YOUNG_MODULUS_MPA,
    StiffnessEstimate,
    estimate_stiffness,
    youngs_for,
)

__all__ = [
    "YOUNG_MODULUS_MPA",
    "StiffnessEstimate",
    "estimate_stiffness",
    "youngs_for",
]
