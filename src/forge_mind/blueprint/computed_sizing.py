"""Deterministic computed-body sizing (Step 2): derive a computed member's envelope from a coarse
CATEGORICAL size class, so the LLM picks no dimensions (ADR-0009; mirrors blueprint/arm_sizing.py).

The member's long axis is local X (the span); the cross-section fills the minor dims. These match the
library builder's size table, so the offline catalog and the LLM path size identically. Lives in the
blueprint layer (the sizing policy); depends only on Vec3 (no layout import -> no cycle).
"""

from __future__ import annotations

from forge_mind.types import Vec3

# Coarse size class -> envelope (span x minor x minor). Same numbers as the library builder's
# _COMPUTED_SIZES; the LLM picks the CLASS only, these are never LLM-authored.
SIZE_CLASS_ENVELOPE: dict[str, Vec3] = {
    "compact": Vec3(x=80.0, y=24.0, z=24.0),
    "standard": Vec3(x=120.0, y=30.0, z=28.0),
    "large": Vec3(x=160.0, y=40.0, z=32.0),
}
DEFAULT_SIZE_CLASS = "standard"


def derive_computed_envelope(size_class: str) -> Vec3:
    """Envelope for a coarse size class (the grow target; never LLM-authored numbers)."""
    return SIZE_CLASS_ENVELOPE.get(size_class, SIZE_CLASS_ENVELOPE[DEFAULT_SIZE_CLASS])


__all__ = ["DEFAULT_SIZE_CLASS", "SIZE_CLASS_ENVELOPE", "derive_computed_envelope"]
