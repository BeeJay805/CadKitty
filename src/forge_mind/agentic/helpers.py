"""Deterministic hardware helpers for the agentic author (importable in the sandbox).

These produce EXACT geometry for hardware-mating features so the author never
hand-computes bolt-circle trig or stepped-bore fits (the error-prone part). Each
SUBTRACT tool is built with its open/top face on the z=0 plane and extends in -Z,
so the author positions it onto a real face with ``Pos(...)`` / a workplane and
subtracts it from the body. Pure geometry: imports only build123d + math, nothing
else from forge_mind, so importing it in the sandbox subprocess stays light.

Conventions:
  - dimensions are mm; diameters are diameters (not radii)
  - cutting tools (bolt_holes / counterbore_tool / bearing_seat_tool): top face at
    z=0, body in -Z. Make depths slightly greater than the wall to cut cleanly.
  - finished solids (rounded_box): centered at the origin.
"""

from __future__ import annotations

from math import cos, radians, sin
from typing import Any

from build123d import (
    Axis,
    Box,
    Circle,
    Cylinder,
    Plane,
    Pos,
    SortBy,
    chamfer,
    fillet,
    offset,
    sweep,
)


def bolt_circle_points(
    pcd: float, count: int, start_angle: float = 0.0
) -> list[tuple[float, float]]:
    """(x, y) centers of ``count`` features equally spaced on a bolt circle of pitch
    diameter ``pcd``. Use with build123d ``Locations(*bolt_circle_points(...))``."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if pcd <= 0:
        raise ValueError("pcd must be > 0")
    r = pcd / 2.0
    return [
        (
            r * cos(radians(start_angle + i * 360.0 / count)),
            r * sin(radians(start_angle + i * 360.0 / count)),
        )
        for i in range(count)
    ]


def bolt_holes(
    pcd: float, hole_d: float, count: int, depth: float, start_angle: float = 0.0
) -> Any:
    """SUBTRACT tool: ``count`` cylindrical holes of diameter ``hole_d`` on a bolt
    circle of pitch diameter ``pcd``. Open face at z=0, extends -Z by ``depth``."""
    r = hole_d / 2.0
    tool: Any = None
    for x, y in bolt_circle_points(pcd, count, start_angle):
        cyl = Pos(x, y, -depth / 2.0) * Cylinder(radius=r, height=depth)
        tool = cyl if tool is None else tool + cyl
    return tool


def counterbore_tool(thru_d: float, head_d: float, head_depth: float, thru_depth: float) -> Any:
    """SUBTRACT tool for a counterbored screw hole: a ``thru_d`` through bore (depth
    ``thru_depth``) with a coaxial ``head_d`` head recess (depth ``head_depth``) at
    the top. Top face at z=0, extends -Z. Subtract from a face."""
    thru = Pos(0, 0, -thru_depth / 2.0) * Cylinder(radius=thru_d / 2.0, height=thru_depth)
    head = Pos(0, 0, -head_depth / 2.0) * Cylinder(radius=head_d / 2.0, height=head_depth)
    return thru + head


def bearing_seat_tool(bore_d: float, seat_d: float, seat_depth: float, thru_depth: float) -> Any:
    """SUBTRACT tool for a bearing pocket: a ``seat_d`` counterbore (depth
    ``seat_depth``) the bearing OD presses into, over a smaller ``bore_d`` through
    bore (depth ``thru_depth``) for the shaft. Seat opening at z=0, extends -Z."""
    seat = Pos(0, 0, -seat_depth / 2.0) * Cylinder(radius=seat_d / 2.0, height=seat_depth)
    bore = Pos(0, 0, -thru_depth / 2.0) * Cylinder(radius=bore_d / 2.0, height=thru_depth)
    return seat + bore


def rounded_box(length: float, width: float, height: float, radius: float) -> Any:
    """A box centered at the origin with its four vertical (Z) edges filleted to
    ``radius`` (must be < half the smaller XY dimension). Ubiquitous for enclosures
    and brackets."""
    solid = Box(length, width, height)
    return fillet(solid.edges().filter_by(Axis.Z), radius=radius)


def radial_pattern(
    feature: Any,
    count: int,
    *,
    axis: Any = Axis.Z,
    start_angle: float = 0.0,
    fuse: bool = True,
) -> Any:
    """``count`` copies of a PRE-BUILT solid ``feature`` rotated evenly (360/count apart) about
    ``axis`` (through the origin), unioned into one solid (``fuse=True``) or returned as a list.

    The 'build ONE unit, array it' idiom for repeated-unit parts -- gear teeth, impeller/fan
    blades, crank throws, spokes, cooling fins, lugs -- so the author models one correct unit
    (offset from ``axis``) instead of hand-placing N and getting the trig/fusion wrong. For a PHASED
    pattern (e.g. a crankshaft's throws at 0/180/180/0 deg) set ``fuse=False`` and place/rotate the
    copies yourself, or call once per phase group."""
    if count < 1:
        raise ValueError("count must be >= 1")
    step = 360.0 / count
    copies = [feature.rotate(axis, start_angle + i * step) for i in range(count)]
    if not fuse:
        return copies
    out = copies[0]
    for c in copies[1:]:
        out = out + c
    return out


def linear_pattern(
    feature: Any,
    count: int,
    spacing: float,
    *,
    axis: Any = Axis.X,
    fuse: bool = True,
) -> Any:
    """``count`` copies of a PRE-BUILT solid ``feature`` spaced ``spacing`` mm apart along ``axis``,
    unioned (``fuse=True``) or returned as a list. The linear 'build one, array it' idiom -- a row
    of fins/ribs/bosses, rack teeth, a bank of blades. The first copy sits where ``feature`` is."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if spacing <= 0:
        raise ValueError("spacing must be > 0")
    d = axis.direction
    copies = [
        Pos(d.X * i * spacing, d.Y * i * spacing, d.Z * i * spacing) * feature for i in range(count)
    ]
    if not fuse:
        return copies
    out = copies[0]
    for c in copies[1:]:
        out = out + c
    return out


# ADR-0133 fix 3 (FORGE_EDGE_FINISH): the silent-degradation event log. Study 0
# measured 5/7 archived fillet-calling shipped parts with ZERO toroidal faces --
# safe_fillet silently no-oped and every gate passed the un-rounded part. Recording is
# unconditional (in-memory, per-process); SERIALIZATION into result.json is job-flag
# gated in the sandbox runner, so flag-off artifacts stay byte-identical.
_EDGE_FINISH_EVENTS: list[dict[str, Any]] = []


def drain_edge_finish_events() -> list[dict[str, Any]]:
    """Return-and-clear the safe_fillet/safe_chamfer outcome log (sandbox-runner use;
    drained before each job so warm workers never leak events across parts)."""
    out = list(_EDGE_FINISH_EVENTS)
    _EDGE_FINISH_EVENTS.clear()
    return out


def _edge_count(edges: Any) -> int | None:
    try:
        return len(list(edges))
    except Exception:
        return None


def safe_fillet(part: Any, edges: Any, radius: float, *, min_radius: float = 0.2) -> Any:
    """Fillet ``edges`` of ``part`` to ``radius``, AUTO-BACKING-OFF the radius on a
    kernel raise -- the #1 author failure is a radius larger than the local wall, which
    makes ``fillet`` throw and kills the whole script. This halves the radius and retries
    down to ``min_radius``; if even that fails (or ``edges`` is empty), it returns the
    ORIGINAL ``part`` UNCHANGED so a cosmetic round never destroys an otherwise-valid
    build. NEVER raises -- but every outcome is RECORDED (requested vs realized radius,
    or a skip), so the author loop can see a silent no-op (ADR-0133 fix 3).

    The broad ``except`` is safe here: this is GENERATION, not certification -- the
    validator re-measures the real geometry downstream and is the sole verdict authority,
    so a silently-skipped fillet cannot manufacture a false PASS.
    """
    n_edges = _edge_count(edges)
    r = float(radius)
    while r >= min_radius:
        try:
            out = fillet(edges, radius=r)
            _EDGE_FINISH_EVENTS.append(
                {"op": "fillet", "requested": float(radius), "realized": r, "edges": n_edges}
            )
            return out
        except Exception:  # degrade gracefully; the validator still gates downstream
            r *= 0.5
    _EDGE_FINISH_EVENTS.append(
        {"op": "fillet", "requested": float(radius), "realized": None, "edges": n_edges}
    )
    return part


def safe_chamfer(part: Any, edges: Any, length: float, *, min_length: float = 0.2) -> Any:
    """Chamfer ``edges`` of ``part`` by ``length``, AUTO-BACKING-OFF on a kernel raise
    (a chamfer wider than the local feature throws). Halves the length and retries down to
    ``min_length``; if even that fails (or ``edges`` is empty), returns the ORIGINAL
    ``part`` UNCHANGED. NEVER raises; every outcome is RECORDED like ``safe_fillet``'s.
    See ``safe_fillet`` for why the broad ``except`` is safe (the validator re-checks the
    real geometry)."""
    n_edges = _edge_count(edges)
    length_ = float(length)
    while length_ >= min_length:
        try:
            out = chamfer(edges, length=length_)
            _EDGE_FINISH_EVENTS.append(
                {"op": "chamfer", "requested": float(length), "realized": length_, "edges": n_edges}
            )
            return out
        except Exception:  # degrade gracefully; the validator still gates downstream
            length_ *= 0.5
    _EDGE_FINISH_EVENTS.append(
        {"op": "chamfer", "requested": float(length), "realized": None, "edges": n_edges}
    )
    return part


def safe_shell(part: Any, thickness: float, *, openings: Any = None) -> Any:
    """Hollow ``part`` to ``thickness``-thick walls via ``offset``, opening ``openings``
    (default: the top face). UNLIKE ``safe_fillet`` this does NOT silently degrade -- a
    vessel that must be hollow must never come back solid, and thinning the wall would
    violate the min-wall spec -- so it RAISES a clear, repairable message on the two ways
    shelling fails: a kernel raise (the wall is too thick for the local geometry, or the
    surface self-intersects) and the SILENT sliver (``offset`` "succeeds" but yields a
    near-zero-volume or multi-body shell). Surfacing the sliver here turns a confusing
    downstream validator reject into an actionable author fix.

    Build the SOLID outer form first, then ``safe_shell`` it -- never hand-draw an
    inner+outer two-rail profile (it self-intersects or collapses to a sliver).
    """
    faces = openings if openings is not None else top_face(part)
    try:
        hollow = offset(part, amount=-abs(float(thickness)), openings=faces)
    except Exception as exc:
        raise ValueError(
            f"safe_shell: cannot hollow to a {thickness} mm wall -- it is too thick for the "
            "local geometry or the surface self-intersects. Use a thinner wall, a cleaner "
            f"solid, or open a different face. (kernel: {exc})"
        ) from exc
    solids = hollow.solids()
    if hollow.volume <= 1e-6 or len(solids) != 1:
        raise ValueError(
            f"safe_shell: hollowing to a {thickness} mm wall collapsed the part to a sliver "
            f"or split it into {len(solids)} pieces. Use a thinner wall or a cleaner solid."
        )
    return hollow


def safe_sweep(
    path: Any,
    section: Any = None,
    *,
    radius: float | None = None,
    inner_radius: float = 0.0,
    is_frenet: bool = False,
) -> Any:
    """Sweep a closed section along ``path`` with the path-start plane built FOR you.

    The archive's #1 sweep killer (103 of 151 failures) is an open hand-drawn outline
    swept into a SHELL; #2 (8) is a section off the path start or too large for the
    bend. This constructs ``Plane(origin=path @ 0, z_dir=path % 0)`` so the section is
    perpendicular AND on the path by construction. ``radius`` sweeps a solid rod (the
    tube case: ``inner_radius`` > 0 sweeps the bore separately and subtracts AFTER --
    never a hand-drawn annulus); ``section`` is a closed 2D Sketch/Face drawn at the
    origin on Plane.XY, relocated here.

    FAIL-CLOSED (safe_shell's contract, deliberately NOT safe_fillet's silent degrade
    -- the fillet study proved silent degradation ships undetected, and a hook without
    its hook is the wrong part): raises a repairable ValueError instead of returning a
    shell, an empty shape, a one-branch sweep of a disconnected path, or a multi-solid
    scatter."""
    # Path normalization, verified against the kernel (review-caught): an algebra Edge's
    # .wire() returns None, and a DISCONNECTED BuildLine's .wire() silently returns only
    # the FIRST run -- a one-branch sweep that would pass every downstream check. Take
    # wires() ourselves and refuse ambiguity.
    wire: Any = path
    if hasattr(wire, "wires"):
        runs = list(wire.wires())
        if len(runs) > 1:
            raise ValueError(
                f"safe_sweep: the path contains {len(runs)} DISCONNECTED runs -- every "
                "segment must start exactly where the previous ended (l2 = Line(l1 @ 1, "
                "...)). One continuous path only."
            )
        if len(runs) == 1:
            wire = runs[0]
    plane = Plane(origin=wire @ 0, z_dir=wire % 0)
    if radius is not None:
        outer: Any = plane * Circle(radius)
        try:
            swept = sweep(outer, path=wire, is_frenet=is_frenet)
            if inner_radius > 0:
                bore_sect: Any = plane * Circle(inner_radius)
                bore = sweep(bore_sect, path=wire, is_frenet=is_frenet)
                swept = swept - bore
        except Exception as exc:
            raise ValueError(
                f"safe_sweep: the r={radius} section failed to sweep -- the profile is "
                "too large for the path's tightest bend, or the path segments do not "
                f"chain. Shrink the radius or ease the bend. (kernel: {exc})"
            ) from exc
    elif section is not None:
        try:
            sect: Any = plane * section
            swept = sweep(sect, path=wire, is_frenet=is_frenet)
        except Exception as exc:
            raise ValueError(
                "safe_sweep: the section failed to sweep -- it must be ONE closed planar "
                "sketch (drawn at the origin on Plane.XY; this helper relocates it to the "
                "path start). Close the outline, or shrink it below the path's tightest "
                f"bend. (kernel: {exc})"
            ) from exc
    else:
        raise ValueError(
            "safe_sweep: pass radius=<r> for a round rod/tube, or section=<closed 2D "
            "sketch drawn at the origin> for any other profile."
        )
    solids = swept.solids() if hasattr(swept, "solids") else []
    if not solids or len(solids) != 1 or float(abs(swept.volume)) <= 1e-6:
        raise ValueError(
            f"safe_sweep: sweeping produced {len(solids)} solid(s) with volume "
            f"{float(abs(swept.volume)) if hasattr(swept, 'volume') else 0:.4f} -- the "
            "section self-intersected around a bend or collapsed. Shrink the profile or "
            "increase the bend radius."
        )
    return swept


# --- Robust selectors -------------------------------------------------------------
# A boolean or fillet RENUMBERS the B-rep, so a raw list index (faces()[3]) silently
# grabs the wrong face after an edit and the next cut lands in the wrong place or
# vanishes. These name the face/edge by AXIS + POSITION (the stable way) so the author
# RE-SELECTS fresh right before each cut. Each accepts a BuildPart context OR a finished
# part/solid (anything with .faces()/.edges()).


def top_face(part: Any) -> Any:
    """The single face highest along +Z. Sketch the next feature on it -- e.g.
    ``with BuildSketch(top_face(bp)):`` -- re-selecting fresh so a prior boolean that
    renumbered the topology cannot point you at a stale face."""
    return part.faces().sort_by(Axis.Z)[-1]


def bottom_face(part: Any) -> Any:
    """The single face lowest along -Z (the print bed / underside)."""
    return part.faces().sort_by(Axis.Z)[0]


def face_facing(part: Any, axis: Any, *, positive: bool = True) -> Any:
    """The planar face whose outward normal points along ``axis``: the +``axis`` side
    when ``positive`` (the default), else the -``axis`` side. ``face_facing(bp, Axis.X)``
    is the right face; ``face_facing(bp, Axis.Y, positive=False)`` is the front. Mount a
    feature on a SIDE without hand-counting faces. Raises if no such planar face exists."""
    faces = part.faces().filter_by(axis)
    return faces.sort_by(axis)[-1] if positive else faces.sort_by(axis)[0]


def largest_face(part: Any) -> Any:
    """The face with the greatest area -- usually the natural face to lay out a hole
    pattern or a label on. Stable across edits (area, not index)."""
    return part.faces().sort_by(SortBy.AREA)[-1]


def vertical_edges(part: Any) -> Any:
    """All edges parallel to Z (the four upright corners of a box-like body). Feed to
    ``safe_fillet``/``fillet`` to round the vertical corners."""
    return part.edges().filter_by(Axis.Z)


def top_edges(part: Any) -> Any:
    """The ring of edges at the highest Z (the top rim). For blunting or chamfering the
    top lip -- ``safe_chamfer(part, top_edges(part), length=1)``."""
    return part.edges().group_by(Axis.Z)[-1]


__all__ = [
    "bearing_seat_tool",
    "bolt_circle_points",
    "bolt_holes",
    "bottom_face",
    "counterbore_tool",
    "drain_edge_finish_events",
    "face_facing",
    "largest_face",
    "linear_pattern",
    "radial_pattern",
    "rounded_box",
    "safe_chamfer",
    "safe_fillet",
    "safe_shell",
    "safe_sweep",
    "top_edges",
    "top_face",
    "vertical_edges",
]
