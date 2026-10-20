"""Deterministic matcher for the gripper lane (no LLM) -- ADR-0044.

Additive sibling of ``arm_match`` (which stays byte-identical and still declines
every gripper prompt -- THIS module owns them). The family claims only what it can
cite: the prompt must name the MG996R (the one registered servo) AND a two-finger /
parallel-finger gripper mechanism. Everything it cannot build honestly -- 3+ fingers,
hands, soft/vacuum grippers, arm+gripper combos (a later composition) -- declines to
the agentic engine (strictly-no-worse).

The payload is DECLARED when the prompt names the cited can ("soda can", "beverage
can", "355 ml", "12 oz"): only then does the aperture gate fire (a bare 'gripper'
makes no aperture claim).
"""

from __future__ import annotations

import re

from forge_mind.assembly.gripper_family import GripperSpec

_MG996R = re.compile(r"\bmg\s*[- ]?996r?\b", re.IGNORECASE)
_GRIPPER = re.compile(r"\b(gripper|claw)\b", re.IGNORECASE)
# Mechanisms/variants this family does NOT build -- decline, never approximate.
_EXCLUDE = re.compile(
    r"\b(three|3|four|4|five|5)\s*[- ]?finger|\bhand\b|\bsoft\b|\bvacuum\b|\bsuction\b|"
    r"\bparallel[- ]jaw\b|\brack\b|\blinear\b|\barm\b|\bpan\s*[-/ ]?\s*tilt\b",
    re.IGNORECASE,
)
# The cited payload object (the one registered can).
_CAN = re.compile(r"\b(soda|beverage|drink)\s+can\b|\b355\s*ml\b|\b12\s*oz\b", re.IGNORECASE)


def gripper_match(prompt: str) -> GripperSpec | None:
    """A GripperSpec when the prompt is a cited-servo two-finger gripper this family
    can build honestly, else None (the caller falls through)."""
    text = prompt.strip()
    if not _MG996R.search(text):
        return None  # no cited servo, no claim
    if not _GRIPPER.search(text):
        return None
    if _EXCLUDE.search(text):
        return None
    declared = bool(_CAN.search(text))
    return GripperSpec(payload_declared=declared, name="gripper")


__all__ = ["gripper_match"]
