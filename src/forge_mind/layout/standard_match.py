"""Standard-parts lane matcher (ADR-0034, vocabulary widened per ADR-0035 A1).

The agentic author is measurably ~25-50% reliable on hole-bearing standard parts (ADR-0033),
while the deterministic LayoutSpec families are measured ~100% (bench-001/002/003/005-7) -- and
a standard part should come out the SAME every time. This matcher gates entry to the
deterministic lane.

ADR-0034 shipped a 12-substring gate that covered a strict minority of the families the
interpreters support: "a 60mm fan mount" (bench-006, 30/30) and "a 608 bearing block"
(composition engine, Wilson LB 90%) never reached the lane that builds them. ADR-0035
decision (c) replaces it with vocabulary GENERATED from the component/standards registries,
compiled to word-boundary regexes:

- ECOSYSTEM tokens come from registry presence INTERSECT interpreter reachability: motors
  (``known_motor_component_ids``), extrusions (``known_extrusion_component_ids``), boards,
  fans, and the bearing registry. A coverage test asserts every reachable registry id has a
  token, so a new registry row forces a vocabulary decision instead of silently widening
  nothing.
- NUMERIC tokens (2020, 2040, 3030, 608...) are valid ONLY adjacent to their hardware noun
  ("2020 extrusion", spelled as phrase regexes) -- "built in 2020" and "1625 mm" never route.
- The gate stays a conservative AND: ecosystem token AND part-form token, word-bounded.

Precision is deliberately second to recall here because the lane is strictly-no-worse: a false
POSITIVE costs one cheap (~$0.02-0.25 / ~20 s) deterministic attempt whose strict gates +
intent-fidelity check + UNSUPPORTED classifier all fall through to the agentic engine
(api/core._generate_standard); a false NEGATIVE sends a ~100% part to the ~25-50% author.
"""

from __future__ import annotations

import re

from forge_mind.dsl.components.registry import (
    known_extrusion_component_ids,
    known_motor_component_ids,
)

# ---------------------------------------------------------------------- ecosystem tokens ---


def _motor_patterns() -> dict[str, str]:
    """nema14/nema17/nema23 -> word-bounded spellings (nema 17 / nema-17 / nema17)."""
    out: dict[str, str] = {}
    for cid in known_motor_component_ids():
        size = cid.removeprefix("nema")
        out[cid] = rf"\bnema\s*-?\s*{size}\b"
    return out


def _extrusion_patterns() -> dict[str, str]:
    """ext_2020/ext_2040/ext_3030 -> numeric-with-noun phrases (ADR-0035 decision c: a bare
    profile number never routes -- it must sit next to its hardware noun). The material word
    alone satisfies adjacency ("2020 aluminum", an ADR-0034 spelling) as does the profile noun
    ("2020 extrusion", "2040 v-slot")."""
    out: dict[str, str] = {}
    for cid in known_extrusion_component_ids():
        size = cid.removeprefix("ext_")
        out[cid] = (
            rf"\b{size}\s*(?:mm\s*)?-?\s*"
            r"(?:aluminum|aluminium|alu\b|extrusions?|profiles?|v-?slots?|rails?)"
        )
    return out


# Boards / fans / bearings / generic spellings. Boards and fans are registry classes with two
# entries each; their tokens are the class vocabulary (the tray is board-generic, the fan
# family is size-generic -- the spec carries the exact component). The bearing word covers
# every registry designator ("608 bearing", "625 bearing"): the NUMBER alone never routes.
_STATIC_PATTERNS: dict[str, str] = {
    "raspberry_pi_4": r"\braspberry\s*pi\b|\bpi\s*-?\s*4\b|\bpi4\b",
    "generic_rect_pcb": r"\bpcb\b|\bcircuit\s+board\b|\barduino\b",
    "fan": r"\bfans?\b",
    # (?<!load-) keeps the English compound "load-bearing" from reading as hardware.
    "bearing": r"(?<!load-)\bbearings?\b",
    "stepper": r"\bsteppers?\b",
    "extrusion_generic": r"\b(?:aluminum|aluminium|alu)\s+extrusions?\b|\bv-?slots?\b",
}


def _ecosystem_patterns() -> dict[str, str]:
    return {**_motor_patterns(), **_extrusion_patterns(), **_STATIC_PATTERNS}


# ---------------------------------------------------------------------- part-form tokens ---

# The mount/plate/enclosure shapes the interpreters build. Measured grounding per token:
# mount/plate/bracket/adapter (bench-001/003/006 + composable), case/tray/enclosure/shroud
# (bench-002 + electronics_tray), grille (composable config family), arm (motor_arm, 20/20),
# block/holder/stand/housing (composable bases: flat_plate / open_box / walled grammar).
_FORM_WORDS: tuple[str, ...] = (
    "mount",
    "mounting",
    "plate",
    "bracket",
    "adapter",
    "case",
    "tray",
    "enclosure",
    "shroud",
    "grille",
    "arm",
    "block",
    "holder",
    "stand",
    "housing",
)

_ECOSYSTEM_RE = re.compile("|".join(_ecosystem_patterns().values()))
_FORM_RE = re.compile(r"\b(?:" + "|".join(_FORM_WORDS) + r")s?\b")


def standard_match(prompt: str | None) -> bool:
    """True iff the prompt names a standard hardware ecosystem AND a part form for it."""
    p = (prompt or "").lower()
    if not p:
        return False
    return _ECOSYSTEM_RE.search(p) is not None and _FORM_RE.search(p) is not None


__all__ = ["standard_match"]
