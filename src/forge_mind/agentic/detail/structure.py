"""Structural + assembly + edge detail: ribs, cooling fins, lightening, fastener seats,
edge treatment. What makes a part read as ENGINEERED (ribbed not blocky) and assemble
cleanly (chamfered entries, countersinks). Boolean ops + guarded edge ops."""

from __future__ import annotations

from typing import Any

from forge_mind.agentic.detail.core import _MIN_FEATURE_MM, register


@register("ribs")
def ribs(
    part: Any,
    *,
    x0: float,
    x1: float,
    y_center: float,
    z_base: float,
    z_top: float,
    count: int = 3,
    thickness: float = 2.0,
    width: float = 20.0,
) -> Any:
    """Add a row of stiffening ribs (ADD) spanning z_base..z_top, spaced along X. Reads as
    engineered and adds real stiffness."""
    from build123d import Box, Pos

    if thickness < _MIN_FEATURE_MM or count < 1:
        raise ValueError("rib below the printable floor")
    add: Any = None
    for i in range(count):
        fx = x0 + (i + 0.5) * (x1 - x0) / count
        rib = Pos(fx, y_center, (z_base + z_top) / 2) * Box(thickness, width, z_top - z_base)
        add = rib if add is None else add + rib
    return part + add


@register("cooling_fins")
def cooling_fins(
    part: Any,
    *,
    cx: float,
    cy: float,
    z0: float,
    z1: float,
    bore_r: float,
    fin_r: float,
    count: int = 6,
    fin_thick: float = 1.5,
) -> Any:
    """Stack annular cooling fins on a Z-axis cylinder body (ADD) -- the radial/engine look
    and real surface area. Fins from bore_r out to fin_r, evenly spaced z0..z1."""
    from build123d import Cylinder, Pos

    if fin_thick < _MIN_FEATURE_MM or count < 2 or fin_r <= bore_r:
        raise ValueError("fin below the printable floor")
    add: Any = None
    span = z1 - z0
    for i in range(count):
        fz = z0 + span * i / (count - 1)
        fin = Pos(cx, cy, fz) * (Cylinder(fin_r, fin_thick) - Cylinder(bore_r, fin_thick + 2))
        add = fin if add is None else add + fin
    return part + add


@register("lightening")
def lightening(
    part: Any,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z_top: float,
    kind: str = "pockets",
    depth: float = 3.0,
    cols: int = 2,
    rows: int = 2,
    thickness: float = 8.0,
) -> Any:
    """Remove non-structural material to read as engineered + save print. kind in
    {pockets (blind recesses), holes (through), honeycomb (hex pocket array)}."""
    from build123d import Box, Cylinder, Pos, RegularPolygon, Rot, extrude

    if depth < _MIN_FEATURE_MM or cols < 1 or rows < 1:
        raise ValueError("lightening below the printable floor")
    cut: Any = None
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
    for i in range(cols):
        for j in range(rows):
            fx = x0 + (i + 0.5) * cw
            fy = y0 + (j + 0.5) * ch
            if kind == "pockets":
                f: Any = Pos(fx, fy, z_top - depth / 2) * Box(cw * 0.7, ch * 0.7, depth)
            elif kind == "holes":
                f = Pos(fx, fy, z_top - thickness / 2) * Cylinder(min(cw, ch) * 0.3, thickness + 4)
            elif kind == "honeycomb":
                sk = Pos(fx, fy, z_top + 1) * Rot(0, 0, 30) * RegularPolygon(min(cw, ch) * 0.4, 6)
                f = extrude(sk, amount=-(depth + 1))
            else:
                raise ValueError(f"unknown lightening kind {kind!r}")
            cut = f if cut is None else cut + f
    return part - cut


@register("fastener_seat")
def fastener_seat(
    part: Any,
    *,
    x: float,
    y: float,
    z_top: float,
    thickness: float,
    kind: str = "countersink",
    bolt_d: float = 4.4,
    head_d: float = 8.0,
    head_h: float = 3.0,
) -> Any:
    """Seat a bolt head so it sits flush/recessed. kind in {countersink (cone),
    counterbore (flat pocket)}. Subtracts the clearance hole + the head recess."""
    from build123d import Cone, Cylinder, Pos

    if bolt_d < _MIN_FEATURE_MM or head_d <= bolt_d:
        raise ValueError("fastener seat below the printable floor")
    shank = Pos(x, y, z_top - thickness / 2) * Cylinder(bolt_d / 2, thickness + 4)
    if kind == "countersink":
        head: Any = Pos(x, y, z_top - head_h / 2) * Cone(head_d / 2, bolt_d / 2, head_h)
    elif kind == "counterbore":
        head = Pos(x, y, z_top - head_h / 2) * Cylinder(head_d / 2, head_h)
    else:
        raise ValueError(f"unknown fastener seat kind {kind!r}")
    return part - shank - head


@register("edge_treatment")
def edge_treatment(
    part: Any,
    *,
    size: float = 0.8,
    kind: str = "chamfer",
    zmin: float | None = None,
    zmax: float | None = None,
) -> Any:
    """Break FREE edges (chamfer or fillet) whose midpoint Z is in [zmin, zmax] (default
    all). GUARDED -- build123d edge ops fail on over-constrained geometry; the core catches
    it and keeps the clean part. NEVER target a mating edge (the plan owns that)."""
    from build123d import chamfer, fillet

    if size < 0.4:
        raise ValueError("edge treatment below the printable floor")
    edges = list(part.edges())
    if zmin is not None or zmax is not None:
        lo = -1e9 if zmin is None else zmin
        hi = 1e9 if zmax is None else zmax
        edges = [e for e in edges if lo <= e.center().Z <= hi]
    if not edges:
        raise ValueError("no edges in the region")
    if kind == "chamfer":
        return chamfer(edges, length=size)
    if kind == "fillet":
        return fillet(edges, radius=size)
    raise ValueError(f"unknown edge treatment kind {kind!r}")
