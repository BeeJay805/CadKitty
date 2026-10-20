"""Deterministic matcher + spec extraction for the servo-arm lane (no LLM).

The arm family only claims what it can cite: the lane matches ONLY prompts that name
the MG996R explicitly (the one registered servo) and ask for a serial-chain mechanism
(arm / pan-tilt / turntable). Everything else -- grippers, linkages, other servos --
falls through to the agentic engine untouched (strictly-no-worse, ADR-0034 semantics).

Extraction is pure regex over the prompt text: DOF from "N-DOF/axis", a stated servo
count as a cross-check (a prompt that asks for 4 DOF with two servos is OVER-CONSTRAINED
for this all-servo family -- the lane declines rather than silently building either
reading). Paraphrases that preserve numbers and component names (the benchmark's
paraphrase rule, and any natural rewording) extract identically by construction.
"""

from __future__ import annotations

import re

from forge_mind.assembly.arm_family import ServoArmSpec

_MG996R = re.compile(r"\bmg\s*[- ]?996r?\b", re.IGNORECASE)
_PAN_TILT = re.compile(r"\bpan\s*[-/ ]?\s*tilt\b", re.IGNORECASE)
_ARM = re.compile(r"\b(robot(ic)?\s+)?arm\b", re.IGNORECASE)
_TURNTABLE = re.compile(r"\bturntable\b", re.IGNORECASE)
# Mechanisms the family does NOT build; their presence means this is not a serial arm.
_EXCLUDE = re.compile(
    r"\b(gripper|claw|finger|hand|linear|lead\s*screw|gantry|delta|scara|gear(ed|box)?|"
    r"belt|pulley|wheel|track|leg)\b",
    re.IGNORECASE,
)
_DOF = re.compile(r"\b(\d)\s*[- ]?\s*(dof|axis|axes|degrees?\s+of\s+freedom)\b", re.IGNORECASE)
_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
}
_SERVO_COUNT = re.compile(
    r"\b(one|two|three|four|[1-4])\s*(?:x\s*)?(?:mg\s*[- ]?996r?\s*)?servos?\b",
    re.IGNORECASE,
)
_SERVO_COUNT_X = re.compile(r"\b([1-4])\s*x\s*mg\s*[- ]?996r?\b", re.IGNORECASE)


def arm_match(prompt: str) -> ServoArmSpec | None:
    """The lane's gatekeeper: a ServoArmSpec when the prompt is a cited-servo serial
    chain this family can build honestly, else None (the caller falls through)."""
    text = prompt.strip()
    if not _MG996R.search(text):
        return None  # no cited servo, no claim -- the family builds only what it cites
    if _EXCLUDE.search(text):
        return None
    is_pan_tilt = bool(_PAN_TILT.search(text))
    if not (is_pan_tilt or _ARM.search(text) or _TURNTABLE.search(text)):
        return None

    dof_m = _DOF.search(text)
    dof = int(dof_m.group(1)) if dof_m else None

    count: int | None = None
    cm = _SERVO_COUNT_X.search(text) or _SERVO_COUNT.search(text)
    if cm:
        count = _COUNT_WORDS.get(cm.group(1).lower())

    if dof is not None and count is not None and dof != count:
        # Over-constrained for an all-servo serial chain (e.g. "4-DOF ... two servos"):
        # decline rather than silently drop one of the user's numbers.
        return None
    resolved = dof if dof is not None else count
    if resolved is None:
        resolved = 2 if is_pan_tilt else (1 if _TURNTABLE.search(text) else 3)
    if not 1 <= resolved <= 4:
        return None
    if is_pan_tilt and dof is None and count is None:
        resolved = 2

    name = "pan_tilt" if (is_pan_tilt and resolved == 2) else f"servo_arm_{resolved}dof"
    return ServoArmSpec(dof=resolved, name=name)


def joint_display_names(dof: int) -> dict[str, str]:
    """Human names for the chain's moving joints, keyed by the link part id."""
    by_dof = {
        1: ["pan"],
        2: ["pan", "tilt"],
        3: ["pan", "shoulder", "elbow"],
        4: ["pan", "shoulder", "elbow", "wrist"],
    }
    return {f"link_{i + 1}": n for i, n in enumerate(by_dof.get(dof, []))}


__all__ = ["arm_match", "joint_display_names"]
