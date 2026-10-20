"""Surface textures + perforations + grip: the detail that makes a face DESIGNED and
functional (anti-slip, grippable, vented). All boolean add/subtract -- robust on any part.

Each op raises on sub-printable params; the core dispatcher guards + re-validates."""

from __future__ import annotations

import math
from typing import Any

from forge_mind.agentic.detail.core import _MIN_FEATURE_MM, register


@register("surface_texture")
def surface_texture(
    part: Any,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z_top: float,
    kind: str = "diamond",
    pitch: float = 3.0,
    depth: float = 0.8,
) -> Any:
    """Recess an anti-slip pattern into a top-face region. kind in
    {diamond (crosshatch), ribbed (parallel), grid (orthogonal), dimple (bumps)}."""
    from build123d import Box, Cylinder, Pos, Rot

    if depth < 0.6 or pitch < _MIN_FEATURE_MM + 0.5:
        raise ValueError("texture below the printable floor")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, d = (x1 - x0), (y1 - y0)
    diag = math.hypot(w, d)
    cut: Any = None
    if kind == "dimple":
        nx, ny = max(1, int(w / pitch)), max(1, int(d / pitch))
        for i in range(nx):
            for j in range(ny):
                fx = x0 + (i + 0.5) * w / nx
                fy = y0 + (j + 0.5) * d / ny
                dot = Pos(fx, fy, z_top) * Cylinder(pitch * 0.3, 2 * depth)
                cut = dot if cut is None else cut + dot
        return part - cut
    angles = {"diamond": (45.0, -45.0), "ribbed": (0.0,), "grid": (0.0, 90.0)}.get(kind)
    if angles is None:
        raise ValueError(f"unknown texture kind {kind!r}")
    for a in angles:
        n = int(diag / pitch) + 2
        for k in range(-n, n + 1):
            rib = (
                Pos(cx, cy, z_top)
                * Rot(0, 0, a)
                * Pos(k * pitch, 0, 0)
                * Box(0.8, diag * 1.5, 2 * depth)
            )
            cut = rib if cut is None else cut + rib
    return part - cut


@register("perforate")
def perforate(
    part: Any,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z_top: float,
    thickness: float,
    kind: str = "round",
    size: float = 4.0,
    cols: int = 3,
    rows: int = 2,
) -> Any:
    """Array of through-openings over a region (grip / drainage / vents / weight). kind in
    {round, hex, slot}. Caller MUST place these on a non-mating panel."""
    from build123d import Box, Cylinder, Pos, RegularPolygon, Rot, extrude

    if size < _MIN_FEATURE_MM + 1.0 or cols < 1 or rows < 1:
        raise ValueError("perforation below the printable floor")
    cut: Any = None
    for i in range(cols):
        for j in range(rows):
            fx = x0 + (i + 0.5) * (x1 - x0) / cols
            fy = y0 + (j + 0.5) * (y1 - y0) / rows
            if kind == "round":
                hole: Any = Pos(fx, fy, z_top - thickness / 2) * Cylinder(size / 2, thickness + 4)
            elif kind == "hex":
                sk = Pos(fx, fy, z_top + 2) * Rot(0, 0, 0) * RegularPolygon(size / 2, 6)
                hole = extrude(sk, amount=-(thickness + 4))
            elif kind == "slot":
                hole = Pos(fx, fy, z_top - thickness / 2) * Box(size, size * 0.5, thickness + 4)
            else:
                raise ValueError(f"unknown perforation kind {kind!r}")
            cut = hole if cut is None else cut + hole
    return part - cut


@register("knurl")
def knurl(
    part: Any,
    *,
    cx: float,
    cy: float,
    z0: float,
    z1: float,
    radius: float,
    kind: str = "straight",
    grooves: int = 16,
    depth: float = 0.6,
) -> Any:
    """Grip texture around a Z-axis cylinder (knob/handle/wheel). kind in
    {straight (axial grooves), cross (axial + a mid ring band)}."""
    from build123d import Box, Cylinder, Pos, Rot

    spacing = 2 * math.pi * radius / max(1, grooves)
    if depth < 0.4 or grooves < 4 or radius <= depth or spacing < _MIN_FEATURE_MM:
        raise ValueError("knurl below the printable floor")
    cutters: Any = None
    span = z1 - z0
    for i in range(grooves):
        ang = 360.0 * i / grooves
        blade = Pos(cx + radius, cy, (z0 + z1) / 2) * Box(2 * depth, 0.8, span)
        blade = Pos(cx, cy, 0) * (Rot(0, 0, ang) * (Pos(-cx, -cy, 0) * blade))
        cutters = blade if cutters is None else cutters + blade
    out = part - cutters
    if kind == "cross":
        ring = Pos(cx, cy, (z0 + z1) / 2) * (
            Cylinder(radius + depth, 0.8) - Cylinder(radius - depth, 2.0)
        )
        out = out - ring
    return out
