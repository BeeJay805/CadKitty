"""FDM printer facts -- the single source of truth for print-in-place gaps (ADR-0013).

A print-in-place mechanism comes off the bed already assembled and moving: adjacent
surfaces that must move relative to each other are separated by a gap of roughly one
nozzle width, so the slicer lays them as SEPARATE shells (they do not fuse) yet they sit
close enough to articulate without rattling.

Per ADR-0009 (categorical-in -> code-derived-metric): the graph/LLM picks the categorical
``clearance_class: print_in_place``; this module derives the millimetres from a printer
fact (the nozzle), NOT from a hand-picked mechanical-fit constant. Numerically the design
gap is ~ today's ``running_fit`` (0.4 mm), but it is kept a SEPARATE class because the
DERIVATION differs (a printer fact, not a fit table) and the two diverge for any other
nozzle.
"""

from __future__ import annotations

NOZZLE_DIAMETER_MM = 0.4
"""Standard FDM nozzle diameter. The print-in-place gap is derived from this."""

PRINT_IN_PLACE_GAP_MM = NOZZLE_DIAMETER_MM
"""Design clearance for a print-in-place moving interface (~ one nozzle width). The solver
shrinks an inserted pin by this (pin radius = bore - gap), leaving the bore load-bearing."""

FDM_MIN_GAP_MM = 0.30
"""Below this, two adjacent surfaces fuse into one solid when sliced (no joint). The
print_in_place_gap check FAILs an interface gapped tighter than this."""

FDM_MAX_GAP_MM = 0.60
"""Above this, a print-in-place joint is sloppy / rattles. The check FAILs a functional
(pin-in-bore) gap looser than this."""


__all__ = [
    "FDM_MAX_GAP_MM",
    "FDM_MIN_GAP_MM",
    "NOZZLE_DIAMETER_MM",
    "PRINT_IN_PLACE_GAP_MM",
]
