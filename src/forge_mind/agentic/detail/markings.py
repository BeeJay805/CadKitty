"""Markings: embossed / debossed text + labels + directional arrows. The strongest
"designed, not prototype" signal, and functional (part numbers, flow direction, warnings)."""

from __future__ import annotations

from typing import Any

from forge_mind.agentic.detail.core import register


@register("emboss_text")
def emboss_text(
    part: Any,
    *,
    x: float,
    y: float,
    z_top: float,
    text: str,
    size: float = 6.0,
    depth: float = 0.8,
    raised: bool = False,
) -> Any:
    """Emboss (raised) or deboss (recessed) text on a top face at Z=z_top. Labels, part
    numbers, branding -- reads unmistakably as designed."""
    from build123d import Pos, Text, extrude

    if not text or size < 3.0 or depth < 0.4:
        raise ValueError("text below the printable floor")
    sk = Pos(x, y, z_top) * Text(text[:24], font_size=size)
    if raised:
        solid = extrude(sk, amount=depth)
        return part + solid
    solid = extrude(sk, amount=-depth)
    return part - solid


@register("direction_arrow")
def direction_arrow(
    part: Any,
    *,
    x: float,
    y: float,
    z_top: float,
    length: float = 10.0,
    depth: float = 0.8,
    angle: float = 0.0,
    raised: bool = False,
) -> Any:
    """A recessed/raised directional arrow (flow, rotation, assembly order) -- a real
    functional marking on pumps, valves, mechanisms."""
    from build123d import Location, Plane, Pos, Rot, Triangle, extrude

    if length < 4.0 or depth < 0.4:
        raise ValueError("arrow below the printable floor")
    _ = (Location, Plane)  # tolerate build123d version import differences
    shaft = Pos(x, y, z_top) * Rot(0, 0, angle) * _rect(length * 0.6, length * 0.22)
    head = (
        Pos(x, y, z_top)
        * Rot(0, 0, angle)
        * Pos(length * 0.4, 0, 0)
        * Triangle(a=length * 0.5, b=length * 0.5, C=90)
    )
    sk = shaft + head
    solid = extrude(sk, amount=depth if raised else -depth)
    return part + solid if raised else part - solid


def _rect(w: float, h: float) -> Any:
    from build123d import Rectangle

    return Rectangle(w, h)
