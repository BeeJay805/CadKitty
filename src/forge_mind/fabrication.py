"""Fabrication mode — what the part is being MADE with, which gates apply.

Forge-Mind is not only a 3D-printer pipeline. A part can be destined for FDM printing (today's
default, with its support-free / minimum-wall constraints) or for general fabrication where those
FDM-specific limits do not apply. The mode is a USER SETTING threaded deterministically through the
pipeline (``run_pipeline -> validate``); it is NEVER picked by the LLM.

What the mode changes in the validator (and ONLY this):

  - printability tier (overhang / bridging) -- FDM-physics-specific -> RUN only in ``PRINT_3D``.
  - minimum wall floor -- 4 mm is an FDM printability limit; ``GENERAL`` relaxes it to a
    structural-sanity floor that still catches genuinely degenerate walls.
  - mesh (watertight/manifold), interfaces, and hardware conformance are fabrication-agnostic and
    run in EVERY mode.

NO-FALSE-PASS: a ``GENERAL`` part is validated FOR general fabrication, never claimed 3D-printable.
The mode is recorded on the ``ValidationReport`` so a ``GENERAL`` PASS is never read as "printable".
``PRINT_3D`` is the default everywhere, so every pre-existing caller is byte-for-byte unchanged.
"""

from __future__ import annotations

from enum import StrEnum

# A wall thinner than this (mm) is almost certainly a modeling error even when FDM's 4 mm
# printability floor does not apply, so GENERAL still flags it. Deliberately loose -- it is a
# sanity floor, not a printability limit. Tunable as more fabrication methods are added.
GENERAL_MIN_WALL_MM = 1.0


class FabricationMode(StrEnum):
    """How a part will be made. Drives which validator gates apply. Room to grow (cnc, sheet_metal)."""

    PRINT_3D = "print_3d"
    GENERAL = "general"

    @property
    def enforces_printability(self) -> bool:
        """True iff overhang/bridging must be checked. Those are FDM-specific, so PRINT_3D only."""
        return self is FabricationMode.PRINT_3D

    def min_wall_override(self, declared: float | None) -> float | None:
        """The min-wall threshold to validate against, or None to use the part's declared wall as-is.

        PRINT_3D returns None (no override) so the feature check is byte-identical to before the mode
        existed. GENERAL relaxes a declared wall to the structural-sanity floor but is NEVER stricter
        than what the part declared, and adds no check to a part that declared none (None stays None).
        """
        if self is FabricationMode.PRINT_3D or declared is None:
            return None
        return min(declared, GENERAL_MIN_WALL_MM)

    def effective_min_wall(
        self, declared: float | None, material_floor: float | None
    ) -> float | None:
        """Deterministic wall floor used by validation and author self-check.

        Printed parts may not weaken the selected material's process floor through an
        LLM-written brief value. GENERAL preserves its established structural-sanity
        semantics and does not inherit an FDM-specific material floor.
        """
        if self is FabricationMode.PRINT_3D:
            values = [v for v in (declared, material_floor) if v is not None and v > 0]
            return max(values) if values else None
        return self.min_wall_override(declared)


__all__ = ["GENERAL_MIN_WALL_MM", "FabricationMode"]
