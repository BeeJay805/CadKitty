"""Deterministic arm sizing (Part A) — derive a motor-arm's envelope from CATEGORICAL intent.

The structure-beats-scale fix for the FIRST ARM gap: the LLM under-sizes an arm's mount-driven
dimensions (it sizes to the motor footprint, not the reach + the mount land), so a raw arm prompt
fell back to direct-DSL. Here deterministic code derives EVERY number from two categoricals —
the motor class (its mount facts, from the registry) and a coarse REACH class — so the LLM
supplies no arm dimensions at all.

  - span (envelope X) = the reach class -> a drone-arm reach table.
  - lateral (envelope Y) = the motor's mount pad width (bolt square + a min_wall land each side)
    + a little headroom, so the NEMA face always fits.
  - height (envelope Z) = a fraction of the motor body, floored so the derived strut / lattice
    beam always fits.

This lives in the blueprint layer (it is the blueprint-stage sizing policy); the arm interpreter
consumes the result. It depends only on the motor registry + Vec3 (no layout import -> no cycle).
"""

from __future__ import annotations

from forge_mind.dsl.components.motor import MotorComponent
from forge_mind.dsl.components.registry import NEMA_17
from forge_mind.types import Vec3

# Coarse reach class -> overall arm span (mm). Absolute drone-arm reaches; all comfortably host
# a NEMA face + a real strut. The LLM picks the class only; these numbers are never LLM-authored.
REACH_SPAN_MM: dict[str, float] = {"compact": 120.0, "standard": 180.0, "long": 250.0}
DEFAULT_REACH = "standard"

# Mount-pad land margin (matches the arm interpreter's _PAD_MARGIN_MM; the feasibility test
# tests/unit/layout/test_arm_interpreter.py::test_derived_envelope_always_feasible guards drift).
_PAD_MARGIN_MM = 4.0
_LATERAL_SLACK_MM = 2.0  # envelope.y headroom beyond the exact mount pad
_HEIGHT_FACTOR = 0.6  # arm height as a fraction of the motor body...
_HEIGHT_FLOOR_MM = 24.0  # ...but never thinner than this (fits the derived strut + a lattice beam)


def arm_pad_width(motor: MotorComponent, min_wall: float) -> float:
    """The motor-mount pad width: the bolt square + a min_wall land + a margin on each side."""
    return (
        motor.mount.bolt_pitch_mm
        + 2.0 * (motor.mount.bolt_clearance_radius_mm + min_wall)
        + 2.0 * _PAD_MARGIN_MM
    )


def derive_arm_envelope(motor: MotorComponent, reach: str, min_wall: float) -> Vec3:
    """Derive an arm envelope from the motor class + the coarse reach class (no LLM numbers)."""
    span = REACH_SPAN_MM.get(reach, REACH_SPAN_MM[DEFAULT_REACH])
    lateral = arm_pad_width(motor, min_wall) + _LATERAL_SLACK_MM
    height = max(_HEIGHT_FLOOR_MM, round(motor.body_mm * _HEIGHT_FACTOR, 1))
    return Vec3(x=span, y=lateral, z=height)


def derive_nema17_arm_envelope(reach: str, min_wall: float) -> Vec3:
    """The arm's motor is NEMA 17 today (the only size with a printable arm interface)."""
    return derive_arm_envelope(NEMA_17, reach, min_wall)


__all__ = ["REACH_SPAN_MM", "arm_pad_width", "derive_arm_envelope", "derive_nema17_arm_envelope"]
