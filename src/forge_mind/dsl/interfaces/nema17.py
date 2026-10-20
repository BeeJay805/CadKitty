"""NEMA 17 motor-face mechanical facts — the single source of truth (ADR-0006 Stage 3).

Pure data: the nominal dimensions of a NEMA 17 mounting face. BOTH the placement
helper (``layout.interfaces.place_nema17_face``) and the validator checker
(``validator.interfaces.nema17_face.Nema17FaceInterface``) import these, so the
geometry that gets built and the geometry that gets validated can never drift
apart -- they read the same constants.

Nominal facts only. The checker's ACCEPTANCE TOLERANCES (how far a real part may
deviate and still pass) stay in the checker: those are validator-authority policy,
not properties of the interface. This module sits in ``dsl/interfaces`` alongside
``declaration.py`` because, like it, it is data with no behaviour -- so both the
layout layer and the validator layer can depend on it without a cycle.
"""

from __future__ import annotations

from forge_mind.dsl.standards.fasteners import clearance_hole
from forge_mind.dsl.standards.framework import Fit

BOLT_PITCH_MM = 31.0
"""Center-to-center pitch of the 4 M3 mounting holes; they sit on a square."""

# Routed to the standards registry (A3): the existing 3.2 mm dia IS the ISO 273 close-fit M3
# clearance, so this derives from it byte-identically (3.2 / 2 = 1.6). The registry is now the
# single source of truth; this stays a named constant for back-compatible imports.
M3_CLEARANCE_RADIUS_MM = clearance_hole("M3", Fit.CLOSE) / 2.0
"""Clearance-hole radius for the M3 mounting bolts (ISO 273 close, 3.2 mm dia)."""

SHAFT_BORE_RADIUS_MM = 11.0
"""Central shaft clearance bore radius (22 mm dia)."""

__all__ = ["BOLT_PITCH_MM", "M3_CLEARANCE_RADIUS_MM", "SHAFT_BORE_RADIUS_MM"]
