"""Computed-geometry op -> build123d Part (ADR-0019).

The deterministic builders that own every FORMULA for a computed-geometry op,
dispatched by ``kind`` from a registry (parallel to box/cylinder). The op carries
only RESOLVED scalars the interpreter derived from coarse intent — never anything
the LLM authored.

First op (`faceted_strut`): a straight structural prism. The cross-section is
obtuse-dihedral (hexagon = 120 deg, rectangle = 90 deg) so the conservative
min_wall measures the true thickness (no acute-edge sliver), and the hexagon is
oriented flat-bottom so its downward faces sit at 30 deg overhang (printable).
Taper is DEFERRED (a tapered loft introduces an acute end-corner the conservative
min_wall over-rejects — ADR-0019 Step-0 / ADR-0018 deferred SDF min_wall).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from build123d import Plane, Pos, Rectangle, RegularPolygon, Rot, extrude

from forge_mind.dsl.primitives import ComputedSolid

# Hexagon rotation that puts a flat edge on the bed (verified overhang-clean in the
# ADR-0019 Step-0 de-risk: downward faces at 30 deg < 45 deg limit).
_HEX_FLAT_BOTTOM_DEG = 30.0
# Rectangle aspect: width = 2*size, depth = RECT_ASPECT*size (a real rectangle, not a square).
_RECT_ASPECT = 1.4


def _cross_section(section: str, size: float, plane: Plane) -> Any:
    """The strut's cross-section as a planar face on ``plane`` (normal = strut axis)."""
    if section == "hexagonal":
        return plane * RegularPolygon(radius=size, side_count=6, rotation=_HEX_FLAT_BOTTOM_DEG)
    return plane * Rectangle(2.0 * size, _RECT_ASPECT * size)


def build_faceted_strut(op: ComputedSolid) -> Any:
    """A straight prism of length ``op.length`` along local X, centered at origin,
    then rotated/translated like every other op. All geometry from resolved params."""
    half = op.length / 2.0
    section = _cross_section(op.section, op.size, Plane(origin=(-half, 0, 0), z_dir=(1, 0, 0)))
    body = extrude(section, amount=op.length)  # extrudes along +X (plane normal) to +half
    if any((op.rotation.x, op.rotation.y, op.rotation.z)):
        body = Rot(X=op.rotation.x, Y=op.rotation.y, Z=op.rotation.z) * body
    if any((op.position.x, op.position.y, op.position.z)):
        body = Pos(op.position.x, op.position.y, op.position.z) * body
    return body


# kind -> deterministic builder. New computed ops register here (the menu grows one
# earned op at a time — lattice/curved are DEFERRED until the SDF min_wall, ADR-0019 Q5).
_BUILDERS: dict[str, Callable[[ComputedSolid], Any]] = {
    "faceted_strut": build_faceted_strut,
}


def build_computed_solid(op: ComputedSolid) -> Any:
    builder = _BUILDERS.get(op.kind)
    if builder is None:
        raise ValueError(f"no builder registered for computed-op kind {op.kind!r}")
    return builder(op)


__all__ = ["build_computed_solid"]
