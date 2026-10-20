"""Swept-form archetypes: the curved-geometry vocabulary the author keeps faking with prisms.

The 2026-07-25 visual review (docs/research/2026-07-25-visual-engineering-gap-review.md, class D)
measured the pattern: whenever a part's true form is a swept or blended shape, the author reaches
for extrude/revolve primitives -- the "centrifugal impeller" shipped as flat radial plates on a
disc (a toy fan), crank webs as plain discs, cooling fins as square annular plates. This is the
same class L87 closed for threads and gears: prose cannot teach a form, an EXECUTABLE helper can
(`IsoThread`/`spur_gear` took those classes to 3/4 first-contact ships).

Same contract as ``agentic.gears``: every function returns ONE watertight build123d solid,
Z-axis, Z-centered where the natural frame allows, callable from sandboxed author code as
``from forge_mind.agentic.flow_forms import ...``. Deterministic -- no randomness, no LLM.
"""

from __future__ import annotations

import math
from typing import Any

from build123d import (
    Axis,
    Cylinder,
    Line,
    Plane,
    Polyline,
    Pos,
    Rot,
    ThreePointArc,
    extrude,
    make_face,
    revolve,
)

__all__ = ["backward_curved_impeller", "crank_web", "tapered_fin_stack"]


def _blade_face(r_in: float, r_out: float, sweep_deg: float, blade_t: float) -> Any:
    """One backward-curved blade FACE on the XY plane.

    The centerline runs from radius ``r_in`` at angle 0 to radius ``r_out`` at angle
    ``-sweep_deg`` (backward = trailing the rotation direction), bulged through a midpoint so the
    blade is a real curve, not a chord. The face is the centerline offset +-t/2 along the local
    normal -- built from sampled points, so no kernel offset op is involved (offsets on free
    curves are the fussiest op in the stack; three-point arcs through computed points are not).
    """
    sweep = math.radians(sweep_deg)

    def center(u: float) -> tuple[float, float, float]:
        # radius grows linearly; angle grows quadratically -> curvature reads "backward-swept"
        r = r_in + (r_out - r_in) * u
        a = -sweep * (u**1.6)
        return (r * math.cos(a), r * math.sin(a), 0.0)

    def offset_pt(u: float, side: float) -> tuple[float, float, float]:
        eps = 1e-4
        x0, y0, _ = center(max(0.0, u - eps))
        x1, y1, _ = center(min(1.0, u + eps))
        tx, ty = x1 - x0, y1 - y0
        n = math.hypot(tx, ty) or 1.0
        # left normal of the tangent, scaled to half thickness
        nx, ny = -ty / n, tx / n
        cx, cy, _ = center(u)
        return (cx + side * blade_t / 2 * nx, cy + side * blade_t / 2 * ny, 0.0)

    left = [offset_pt(u, +1.0) for u in (0.0, 0.5, 1.0)]
    right = [offset_pt(u, -1.0) for u in (0.0, 0.5, 1.0)]
    edges = (
        ThreePointArc(*left)
        + Line(left[2], right[2])
        + ThreePointArc(right[2], right[1], right[0])
        + Line(right[0], left[0])
    )
    return make_face(edges)


def backward_curved_impeller(
    disc_r: float,
    disc_t: float,
    blade_h: float,
    blades: int,
    *,
    hub_r: float,
    hub_h: float,
    bore: float = 0.0,
    r_in: float | None = None,
    sweep_deg: float = 35.0,
    blade_t: float = 2.4,
) -> Any:
    """A centrifugal impeller with genuinely BACKWARD-CURVED blades, one watertight solid.

    Base disc (radius ``disc_r``, thickness ``disc_t``, bottom at z=0), ``blades`` curved blades
    standing ``blade_h`` on the disc, a center hub (``hub_r`` x ``hub_h`` above the disc), and an
    optional through ``bore`` (diameter). Blades run from ``r_in`` (default just outside the hub)
    to the disc rim, each swept ``sweep_deg`` BACKWARD -- the direction a real blower curves them,
    and the visible difference from the flat-plate toy-fan the review measured.

    Blades lean on the disc and hub only through fusion -- no mating surface is touched, so the
    fit checks see exactly the bore and nothing else.
    """
    if blades < 2:
        raise ValueError(f"an impeller needs >= 2 blades (got {blades})")
    if hub_r >= disc_r:
        raise ValueError(f"hub_r {hub_r} must be < disc_r {disc_r}")
    start_r = r_in if r_in is not None else min(hub_r * 1.05, disc_r * 0.5)
    part: Any = Pos(0, 0, disc_t / 2) * Cylinder(disc_r, disc_t)
    part += Pos(0, 0, disc_t + hub_h / 2) * Cylinder(hub_r, hub_h)
    blade_face = _blade_face(start_r, disc_r - blade_t / 2, sweep_deg, blade_t)
    blade = extrude(Pos(0, 0, disc_t) * blade_face, amount=blade_h)
    for i in range(blades):
        part += Rot(0, 0, 360.0 * i / blades) * blade
    if bore > 0:
        part -= Cylinder(bore / 2, 4 * (disc_t + hub_h + blade_h))
    return part


def crank_web(
    web_t: float,
    throw: float,
    *,
    main_boss_r: float,
    pin_boss_r: float,
    cw_r: float,
    cw_span_deg: float = 120.0,
    main_bore: float = 0.0,
    pin_bore: float = 0.0,
) -> Any:
    """One crankshaft WEB with a crescent counterweight, the shape a real crank carries.

    Frame: main-journal axis through the ORIGIN (+Z), crankpin boss at ``(+throw, 0)``, the
    counterweight lobe (radius ``cw_r``, spanning ``cw_span_deg``) opposite the pin at 180 deg --
    which is what balances the throw. Extruded ``web_t`` from z=0. The review measured webs
    shipping as plain discs; the crescent IS the read of a real crank.

    ``main_bore``/``pin_bore`` (diameters) cut through-holes so the web can be pinned onto
    journal stock in a printed build-up crank.
    """
    if cw_r <= main_boss_r:
        raise ValueError(f"cw_r {cw_r} must exceed main_boss_r {main_boss_r} to read as a lobe")
    half = math.radians(cw_span_deg / 2)
    part: Any = Pos(0, 0, web_t / 2) * Cylinder(main_boss_r, web_t)
    part += Pos(throw, 0, web_t / 2) * Cylinder(pin_boss_r, web_t)
    # straight web connecting the bosses: a slab as wide as the smaller boss pair
    w = 2 * min(main_boss_r, pin_boss_r) * 0.9
    slab = make_face(
        Polyline((0, -w / 2), (throw, -w / 2), (throw, w / 2), (0, w / 2), (0, -w / 2))
    )
    part += extrude(slab, amount=web_t)
    # crescent counterweight: a sector of cw_r centered opposite the pin
    a = (math.pi - half, math.pi + half)
    p0 = (cw_r * math.cos(a[0]), cw_r * math.sin(a[0]), 0.0)
    p1 = (-cw_r, 0.0, 0.0)
    p2 = (cw_r * math.cos(a[1]), cw_r * math.sin(a[1]), 0.0)
    sector = make_face(ThreePointArc(p0, p1, p2) + Line(p2, (0, 0, 0)) + Line((0, 0, 0), p0))
    part += extrude(sector, amount=web_t)
    if main_bore > 0:
        part -= Cylinder(main_bore / 2, 4 * web_t)
    if pin_bore > 0:
        part -= Pos(throw, 0, 0) * Cylinder(pin_bore / 2, 4 * web_t)
    return part


def tapered_fin_stack(
    core_r: float,
    bore_r: float,
    length: float,
    fins: int,
    *,
    fin_r: float,
    t_root: float,
    t_tip: float | None = None,
) -> Any:
    """An air-cooled barrel: core tube + ``fins`` annular fins that TAPER root-to-tip.

    Frame: axis +Z, base at z=0, overall ``length``. Fins are evenly pitched along the core,
    each a revolved trapezoid -- ``t_root`` thick where it meets the core, thinning to ``t_tip``
    (default half the root) at radius ``fin_r``. Square constant-thickness fins are the review's
    class-D tell; the taper is both the real look and the stronger FDM print (wider root).
    """
    if fin_r <= core_r:
        raise ValueError(f"fin_r {fin_r} must exceed core_r {core_r}")
    if bore_r >= core_r:
        raise ValueError(f"bore_r {bore_r} must be < core_r {core_r}")
    tip = t_tip if t_tip is not None else t_root / 2
    part: Any = Pos(0, 0, length / 2) * Cylinder(core_r, length)
    pitch = length / fins
    fin_profile_z = [
        (-t_root / 2, core_r),
        (-tip / 2, fin_r),
        (tip / 2, fin_r),
        (t_root / 2, core_r),
    ]
    for i in range(fins):
        zc = pitch * (i + 0.5)
        pts = [(r, z + zc) for (z, r) in fin_profile_z]
        outline: Any = Plane.XZ * Polyline(*pts, pts[0])  # closed trapezoid, XZ half-plane
        part += revolve(make_face(outline), Axis.Z)
    part -= Cylinder(bore_r, 4 * length)
    # keep the bore honestly OPEN: the review's barrel had a domed closed end the piston
    # could never enter; a fin stack helper must never close its own tube.
    return part
