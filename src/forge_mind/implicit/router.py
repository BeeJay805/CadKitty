"""Route a prompt to the implicit lane -- ONLY when it clearly asks for a lattice/TPMS structure we
can actually build (ADR-0026). Conservative by design: a normal part never matches, so the existing
B-rep/agentic paths stay byte-identical. Params are parsed leniently with sane defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from forge_mind.implicit.archetypes import (
    ImplicitPart,
    gyroid_block,
    schwarz_block,
    strut_lattice_block,
)

# Ordered: the most specific keyword wins (schwarz before gyroid; "strut lattice" before "lattice").
_ARCH_KEYWORDS: list[tuple[str, str]] = [
    ("schwarz", "schwarz_p"),
    ("gyroid", "gyroid"),
    ("tpms", "gyroid"),
    ("triply periodic", "gyroid"),
    ("minimal surface", "gyroid"),
    ("strut lattice", "strut_lattice"),
    ("lattice", "strut_lattice"),
    ("truss", "strut_lattice"),
    ("honeycomb", "strut_lattice"),
]

_NUM = r"(\d+(?:\.\d+)?)"

_DEFAULT_SIZE = 40.0
_DEFAULT_CELL = 10.0
_DEFAULT_WALL = 1.6
_DEFAULT_STRUT = 2.4


@dataclass(frozen=True)
class ImplicitRequest:
    archetype: str  # gyroid | schwarz_p | strut_lattice
    size_mm: float
    cell_mm: float
    feature_mm: float  # TPMS wall thickness, or strut diameter


def _find(text: str, *patterns: str) -> float | None:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


def implicit_match(prompt: str | None) -> ImplicitRequest | None:
    """An ImplicitRequest if the prompt names a buildable lattice/TPMS structure, else None."""
    p = (prompt or "").lower()
    archetype: str | None = next((arch for kw, arch in _ARCH_KEYWORDS if kw in p), None)
    if archetype is None:
        return None

    # Keyworded params are unambiguous; parse them first.
    cell = _find(p, rf"{_NUM}\s*mm\s*cell", rf"cell\s*(?:size\s*)?(?:of\s*)?{_NUM}\s*mm")
    wall = _find(p, rf"wall\s*(?:thickness\s*)?(?:of\s*)?{_NUM}\s*mm", rf"{_NUM}\s*mm\s*wall")
    strut = _find(
        p,
        rf"strut\s*(?:diameter|dia|d|thickness)?\s*(?:of\s*)?{_NUM}\s*mm",
        rf"{_NUM}\s*mm\s*strut",
    )
    feature = strut if archetype == "strut_lattice" else wall

    # Overall size: an explicit "NxN" / "N mm cube" wins; otherwise the largest mm value that exceeds
    # both the cell and the feature (a block is always bigger than its cells and struts).
    size = _find(
        p,
        rf"{_NUM}\s*x\s*\d",
        rf"{_NUM}\s*mm\s*(?:cube|block|box|cell-?block)",
        rf"(?:cube|block|box)\s*(?:of\s*)?{_NUM}\s*mm",
    )
    if size is None:
        all_mm = [float(v) for v in re.findall(rf"{_NUM}\s*mm", p)]
        floor = max(cell or 0.0, feature or 0.0)
        bigger = [v for v in all_mm if v > floor]
        size = max(bigger) if bigger else None

    default_feature = _DEFAULT_STRUT if archetype == "strut_lattice" else _DEFAULT_WALL
    return ImplicitRequest(
        archetype=archetype,
        size_mm=size if size is not None else _DEFAULT_SIZE,
        cell_mm=cell if cell is not None else _DEFAULT_CELL,
        feature_mm=feature if feature is not None else default_feature,
    )


def build_implicit(req: ImplicitRequest) -> ImplicitPart:
    """Dispatch a request to its deterministic archetype generator."""
    if req.archetype == "gyroid":
        return gyroid_block(req.size_mm, req.cell_mm, req.feature_mm)
    if req.archetype == "schwarz_p":
        return schwarz_block(req.size_mm, req.cell_mm, req.feature_mm)
    return strut_lattice_block(req.size_mm, req.cell_mm, req.feature_mm)


__all__ = ["ImplicitRequest", "build_implicit", "implicit_match"]
