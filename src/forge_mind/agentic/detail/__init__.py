"""Designed-finish detail library (founder directive: real parts are DESIGNED, not bare
functional primitives, and the library needs TONS of breadth for tons of parts).

Public API: ``FinishOp`` (LLM-directed directive), ``apply_finish`` (fail-closed pass),
``finish_enabled`` (FORGE_FINISH flag), ``op_names`` (the registered breadth).

Importing this package registers every op. Categories:
  textures   -- surface_texture (diamond/ribbed/grid/dimple), perforate (round/hex/slot), knurl
  structure  -- ribs, cooling_fins, lightening (pockets/holes/honeycomb), fastener_seat, edge_treatment
  markings   -- emboss_text, direction_arrow

Every op is deterministic, printability-clamped (>=1mm), and re-validated by the core so a
broken texture is skipped, never shipped. Flag-gated + dark until the finish plan is wired.
"""

from __future__ import annotations

from forge_mind.agentic.detail import markings, structure, textures  # noqa: F401 (register)
from forge_mind.agentic.detail.core import (
    FinishOp,
    FinishResult,
    apply_finish,
    finish_enabled,
    op_names,
)

__all__ = [
    "FinishOp",
    "FinishResult",
    "apply_finish",
    "finish_enabled",
    "op_names",
]
