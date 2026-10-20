"""Deterministic constraint inference over a rough sketch (ADR-0025, Phase 3).

A hand sketch has noisy coordinates. The accuracy lever (Autodesk AutoConstrain: 8.9% -> 93%
fully-constrained; the real enemy is leftover degrees of freedom) is to detect which geometric
relationships are INTENTIONAL -- right angles, equal sizes, shared centers, symmetry -- so the
builder PARAMETERIZES them (one variable for two equal edges; one shared center) instead of emitting
independent magic numbers that distort on the first edit.

This module does *detection* only, on the strokes themselves -- pure Python, no solver dependency.
We author build123d from intent (not from snapped coordinates), so the high-value output is the
constraint LIST the author must honor, not solved coordinates. A constraint *solver* (py-slvs) is the
documented future upgrade if we ever feed sketch-derived coordinates straight to geometry.

Thresholds follow Onshape/SolveSpace auto-inference practice: ~5 deg for axis/parallel/perpendicular,
~8-10% for equal length/radius, ~2% of the canvas diagonal for coincident points.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from forge_mind.intake.schema import (
    ConstrainedState,
    ConstraintKind,
    InferredConstraint,
    Sketch,
    SketchStroke,
)

_ANGLE_TOL_DEG = 5.0
_EQUAL_TOL = 0.08  # relative length difference to call two edges "equal"
_EQUAL_RADIUS_TOL = 0.10
_COINCIDENT_FRAC = 0.02  # fraction of the canvas diagonal that counts as "the same point"
_MIN_LINE_FRAC = 0.04  # a segment shorter than this (of the diagonal) is noise -> skip
_CIRCLE_ROUNDNESS = 0.25  # max radius coefficient-of-variation to accept a stroke as a circle
_STRAIGHTNESS = 1.15  # path_len <= STRAIGHTNESS * chord  -> straight enough to be a line
_MAX_CONSTRAINTS = 32


@dataclass(frozen=True)
class _Line:
    ref: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def angle(self) -> float:
        """Direction in [0, 180) degrees (a line and its reverse are the same)."""
        a = math.degrees(math.atan2(self.y1 - self.y0, self.x1 - self.x0)) % 180.0
        return a

    @property
    def mid(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


@dataclass(frozen=True)
class _Circle:
    ref: str
    cx: float
    cy: float
    r: float


def _angle_diff(a: float, b: float) -> float:
    """Smallest angle between two undirected line directions, in [0, 90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _rel_diff(a: float, b: float) -> float:
    m = max(abs(a), abs(b))
    return abs(a - b) / m if m > 0 else 0.0


def _pts(stroke: SketchStroke) -> list[tuple[float, float]]:
    return [(p.x, p.y) for p in stroke.points]


def _polyline_length(pts: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    )


def _looks_like_line(pts: list[tuple[float, float]]) -> bool:
    if len(pts) < 2:
        return False
    chord = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    path = _polyline_length(pts)
    return chord > 0 and path <= _STRAIGHTNESS * chord


def _fit_circle(ref: str, pts: list[tuple[float, float]]) -> _Circle | None:
    """Centroid + mean radius; accept only if the radius is roughly constant (a real ring, not a blob)."""
    if len(pts) < 5:
        return None
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    radii = [math.hypot(x - cx, y - cy) for x, y in pts]
    mean_r = sum(radii) / len(radii)
    if mean_r <= 0:
        return None
    var = sum((r - mean_r) ** 2 for r in radii) / len(radii)
    if math.sqrt(var) / mean_r > _CIRCLE_ROUNDNESS:
        return None
    return _Circle(ref=ref, cx=cx, cy=cy, r=mean_r)


def _rect_edges(ref: str, pts: list[tuple[float, float]]) -> list[_Line]:
    """Expand a `rect` stroke into its 4 axis-aligned bounding edges -- the generic line logic then
    recovers the 2 horizontal/2 vertical/perpendicular/equal constraints a rectangle implies."""
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [
        _Line(ref=f"{ref}:top", x0=x0, y0=y0, x1=x1, y1=y0),
        _Line(ref=f"{ref}:bottom", x0=x0, y0=y1, x1=x1, y1=y1),
        _Line(ref=f"{ref}:left", x0=x0, y0=y0, x1=x0, y1=y1),
        _Line(ref=f"{ref}:right", x0=x1, y0=y0, x1=x1, y1=y1),
    ]


def _primitives(sketch: Sketch) -> tuple[list[_Line], list[_Circle]]:
    diag = math.hypot(max(sketch.canvas_width, 1), max(sketch.canvas_height, 1))
    min_len = _MIN_LINE_FRAC * diag
    lines: list[_Line] = []
    circles: list[_Circle] = []

    for stroke in sketch.strokes:
        pts = _pts(stroke)
        if len(pts) < 2:
            continue
        stype = (stroke.type or "").strip().lower()
        ref = stroke.id

        if stype == "text":
            continue
        if stype == "rect":
            lines.extend(e for e in _rect_edges(ref, pts) if e.length >= min_len)
            continue
        if stype == "circle":
            c = _fit_circle(ref, pts)
            if c is not None:
                circles.append(c)
            continue
        if stype in ("line", "arrow"):
            seg = _Line(ref=ref, x0=pts[0][0], y0=pts[0][1], x1=pts[-1][0], y1=pts[-1][1])
            if seg.length >= min_len:
                lines.append(seg)
            continue

        # pen / untyped: classify geometrically.
        circ = _fit_circle(ref, pts)
        if circ is not None and not _looks_like_line(pts):
            circles.append(circ)
        elif _looks_like_line(pts):
            seg = _Line(ref=ref, x0=pts[0][0], y0=pts[0][1], x1=pts[-1][0], y1=pts[-1][1])
            if seg.length >= min_len:
                lines.append(seg)

    return lines, circles


def _conf(error: float, tol: float) -> float:
    """Higher confidence the closer the measurement is to exact; clamped to a useful band."""
    return round(max(0.5, min(1.0, 1.0 - (error / tol) * 0.4)), 2)


def infer_constraints(sketch: Sketch) -> list[InferredConstraint]:
    """Detect intentional geometric relationships in a sketch. Deterministic + order-stable."""
    lines, circles = _primitives(sketch)
    diag = math.hypot(max(sketch.canvas_width, 1), max(sketch.canvas_height, 1))
    coincident_r = _COINCIDENT_FRAC * diag

    found: dict[tuple[str, tuple[str, ...]], InferredConstraint] = {}

    def add(kind: ConstraintKind, refs: list[str], conf: float) -> None:
        key = (kind.value, tuple(sorted(refs)))
        if key not in found:
            found[key] = InferredConstraint(
                kind=kind, refs=sorted(refs), source="sketch", confidence=conf
            )

    # Per-line: horizontal / vertical.
    for ln in lines:
        h_err = min(ln.angle, 180.0 - ln.angle)
        if h_err <= _ANGLE_TOL_DEG:
            add(ConstraintKind.HORIZONTAL, [ln.ref], _conf(h_err, _ANGLE_TOL_DEG))
        v_err = abs(ln.angle - 90.0)
        if v_err <= _ANGLE_TOL_DEG:
            add(ConstraintKind.VERTICAL, [ln.ref], _conf(v_err, _ANGLE_TOL_DEG))

    # Line pairs: parallel / perpendicular / equal length.
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a, b = lines[i], lines[j]
            d = _angle_diff(a.angle, b.angle)
            if d <= _ANGLE_TOL_DEG:
                add(ConstraintKind.PARALLEL, [a.ref, b.ref], _conf(d, _ANGLE_TOL_DEG))
            elif abs(d - 90.0) <= _ANGLE_TOL_DEG:
                add(
                    ConstraintKind.PERPENDICULAR,
                    [a.ref, b.ref],
                    _conf(abs(d - 90.0), _ANGLE_TOL_DEG),
                )
            rd = _rel_diff(a.length, b.length)
            if rd <= _EQUAL_TOL:
                add(ConstraintKind.EQUAL, [a.ref, b.ref], _conf(rd, _EQUAL_TOL))

    # Circle pairs: concentric / equal radius.
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            ca, cb = circles[i], circles[j]
            if math.hypot(ca.cx - cb.cx, ca.cy - cb.cy) <= coincident_r:
                add(ConstraintKind.CONCENTRIC, [ca.ref, cb.ref], 0.9)
            rd = _rel_diff(ca.r, cb.r)
            if rd <= _EQUAL_RADIUS_TOL:
                add(ConstraintKind.EQUAL, [ca.ref, cb.ref], _conf(rd, _EQUAL_RADIUS_TOL))

    # Coincident endpoints across distinct lines (a closed/connected outline).
    endpoints = [(ln.ref, ln.x0, ln.y0) for ln in lines] + [(ln.ref, ln.x1, ln.y1) for ln in lines]
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            ra, xa, ya = endpoints[i]
            rb, xb, yb = endpoints[j]
            if ra != rb and math.hypot(xa - xb, ya - yb) <= coincident_r:
                add(ConstraintKind.COINCIDENT, [ra, rb], 0.85)

    _add_symmetry(lines, circles, coincident_r, add)

    out = sorted(found.values(), key=lambda c: (c.kind.value, c.refs))
    return out[:_MAX_CONSTRAINTS]


def _add_symmetry(
    lines: list[_Line],
    circles: list[_Circle],
    coincident_r: float,
    add: Callable[[ConstraintKind, list[str], float], None],
) -> None:
    """Conservative mirror detection: anchor points (line mids + circle centers) that pair up across
    the bbox center axis. Require >=2 mirrored pairs covering most anchors before claiming symmetry."""
    anchors = [ln.mid for ln in lines] + [(c.cx, c.cy) for c in circles]
    if len(anchors) < 4:
        return
    cx = sum(x for x, _ in anchors) / len(anchors)
    cy = sum(y for _, y in anchors) / len(anchors)

    def mirrored_fraction(axis_vertical: bool) -> float:
        matched = 0
        for x, y in anchors:
            mx, my = (2 * cx - x, y) if axis_vertical else (x, 2 * cy - y)
            if any(
                math.hypot(mx - ox, my - oy) <= coincident_r
                for ox, oy in anchors
                if (ox, oy) != (x, y)
            ):
                matched += 1
        return matched / len(anchors)

    if mirrored_fraction(axis_vertical=True) >= 0.5:
        add(ConstraintKind.SYMMETRY, ["axis:vertical"], 0.7)
    if mirrored_fraction(axis_vertical=False) >= 0.5:
        add(ConstraintKind.SYMMETRY, ["axis:horizontal"], 0.7)


def summarize_constraints(constraints: list[InferredConstraint]) -> list[str]:
    """Render constraints as short build-intent lines for the generation prompt -- by KIND, not opaque
    stroke ids (the author needs to know 'right angles are intended', not which strokes)."""
    by_kind: dict[ConstraintKind, list[InferredConstraint]] = {}
    for c in constraints:
        by_kind.setdefault(c.kind, []).append(c)

    out: list[str] = []
    for c in by_kind.get(ConstraintKind.SYMMETRY, []):
        axis = "vertical" if any("vertical" in r for r in c.refs) else "horizontal"
        out.append(f"Symmetric about a {axis} axis -- keep the mirrored features identical.")
    if (n := len(by_kind.get(ConstraintKind.CONCENTRIC, []))) > 0:
        out.append(f"{n} circular feature pair(s) are concentric -- share one center.")
    if (n := len(by_kind.get(ConstraintKind.PERPENDICULAR, []))) > 0:
        out.append(f"Edges meet at right angles ({n} perpendicular pair(s)).")
    if (n := len(by_kind.get(ConstraintKind.PARALLEL, []))) > 0:
        out.append(f"{n} edge pair(s) are parallel.")
    if (n := len(by_kind.get(ConstraintKind.EQUAL, []))) > 0:
        out.append(f"{n} feature pair(s) are equal in size -- drive them from one dimension.")
    return out


@dataclass(frozen=True)
class DofAudit:
    """Advisory degrees-of-freedom summary for a sketch (REPORTED-only, never gates).

    ``under_pinned`` lists the stroke/primitive refs that participate in NO inferred constraint -- the
    features that will move independently on the next edit, which the planner can name in a clarifying
    question. ``free_dof`` is their count. This is a coarse, solver-free proxy (a primitive with one
    constraint is treated as "pinned" even if a real solver would still find residual freedom): it is
    deliberately conservative about claiming FULLY and never asserts a part is correct.
    """

    state: ConstrainedState
    free_dof: int
    under_pinned: list[str]


def audit_dof(sketch: Sketch) -> DofAudit:
    """Heuristic DoF auditor over the detected primitives + inferred constraints (AutoConstrain idea:
    leftover degrees of freedom are the real accuracy enemy). ADVISORY -- it is NOT a constraint solver
    and never gates; it flags primitives with no detected relationship so they can be pinned.
    Deterministic + order-stable."""
    lines, circles = _primitives(sketch)
    constraints = infer_constraints(sketch)
    constrained_refs = {ref for c in constraints for ref in c.refs}
    all_refs = [ln.ref for ln in lines] + [c.ref for c in circles]
    under_pinned = sorted(ref for ref in all_refs if ref not in constrained_refs)
    state = ConstrainedState.UNDER if under_pinned else ConstrainedState.FULLY
    return DofAudit(state=state, free_dof=len(under_pinned), under_pinned=under_pinned)


__all__ = ["DofAudit", "audit_dof", "infer_constraints", "summarize_constraints"]
