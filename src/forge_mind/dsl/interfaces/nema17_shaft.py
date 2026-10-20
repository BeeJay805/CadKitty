"""NEMA 17 output-shaft coupling facts — single source of truth (bench-007, ADR-0012).

The DRIVEN side of the motor, distinct from ``nema17.py``'s ``SHAFT_BORE_RADIUS_MM``
(11 mm = a 22 mm CLEARANCE bore so the shaft + boss pass through the mounting plate).
A coupling does the opposite: it GRIPS the 5 mm shaft to transmit torque. Both the
``output_arm`` hub geometry and the ``shaft_coupling`` checker read these, so built
geometry and the checked geometry can't drift (ADR-0006 Stage 3).

Standard single-shaft NEMA 17:
- shaft diameter 5.0 mm (radius 2.5 mm),
- commonly a D-flat (~4.5 mm across the flat) for positive drive -- the D-bore variant
  is DEFERRED (a non-cylindrical bore the cylinder-feature validator can't measure
  easily); tier 1 is a round running-fit bore + a radial set screw,
- usable shaft length ~24 mm.

Nominal facts only; the checker's acceptance tolerances stay in the checker.
"""

from __future__ import annotations

from forge_mind.dsl.interfaces.nema17 import M3_CLEARANCE_RADIUS_MM

SHAFT_RADIUS_MM = 2.5
"""NEMA 17 output shaft radius (5 mm diameter)."""

SHAFT_RUNNING_FIT_MM = 0.2
"""Radial running-fit clearance: the hub bore is the shaft radius + this, so the hub
turns on the shaft before the set screw is tightened."""

SHAFT_BORE_RADIUS_MM = SHAFT_RADIUS_MM + SHAFT_RUNNING_FIT_MM
"""Hub gripping-bore radius (2.7 mm): grips the 5 mm shaft with a running fit."""

SHAFT_USABLE_LENGTH_MM = 24.0
"""Usable shaft length from the face (round-shaft NEMA 17 ~20-24 mm)."""

SET_SCREW_CLEARANCE_RADIUS_MM = M3_CLEARANCE_RADIUS_MM
"""Radial M3 set-screw clearance hole into the hub bore wall (reuses the M3 fact)."""

__all__ = [
    "SET_SCREW_CLEARANCE_RADIUS_MM",
    "SHAFT_BORE_RADIUS_MM",
    "SHAFT_RADIUS_MM",
    "SHAFT_RUNNING_FIT_MM",
    "SHAFT_USABLE_LENGTH_MM",
]
