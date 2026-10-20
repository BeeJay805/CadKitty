"""Deterministic placement patterns for composed flat-plate features (sensible layout by
construction, ADR-0009 spirit: the LLM picks a categorical PATTERN, code derives every coordinate).

The composition path used to lay EVERY feature in a single centerline ROW along X -- four mounting
holes came out in a line, not a bolt pattern. This module turns a categorical pattern + the feature
sizes + the min-wall margin into in-plane (x, y) centres. NO coordinates come from the LLM and there
is NO aesthetic judge: the returned positions are sized to keep features apart and clear of a central
feature, and the EXISTING edge-land / overlap check still gates validity (a pattern that cannot fit
with proper margin hits the honest infeasible refusal, never a silently-bad layout).

Margin basis: ``min_wall`` is the only edge/separation rule the validator enforces (``rules.edge_land``
>= min_wall); there is no separate registry "edge distance" fact, so min_wall IS the margin here.

All sizes are TIGHT (feature-driven, not envelope-driven): the composition engine passes a generous
envelope and relies on the interpreter to derive a compact plate, so a pattern must not inflate to
the envelope.
"""

from __future__ import annotations

import math
from enum import StrEnum

Pos = tuple[float, float]


class LayoutPattern(StrEnum):
    """The categorical placement patterns the LLM can pick (else the interpreter defaults).

    Multi-instance (mounting holes / inserts / dowels / magnets): ``corner`` (the rectangular DEFAULT),
    ``grid``, ``row``, ``perimeter``, ``bolt_circle`` (N on a pitch circle -- the round-plate default).
    Single primary (a bearing seat): ``centered`` (DEFAULT), ``end_offset``. A single-only pattern
    asked for a group, or vice-versa, falls back to the sensible default rather than failing."""

    CENTERED = "centered"
    END_OFFSET = "end_offset"
    ROW = "row"
    CORNER = "corner"
    GRID = "grid"
    PERIMETER = "perimeter"
    BOLT_CIRCLE = "bolt_circle"


_SINGLE_ONLY = frozenset({LayoutPattern.CENTERED, LayoutPattern.END_OFFSET})


def derive_positions(
    pattern: LayoutPattern,
    n: int,
    feature_radius: float,
    *,
    min_wall: float,
    central_radius: float = 0.0,
    slack: float = 0.0,
) -> list[Pos]:
    """Deterministic in-plane (x, y) centres for ``n`` features of ``feature_radius`` arranged by
    ``pattern``, kept >= ``min_wall`` apart and clear of a central disk of ``central_radius``.

    Pure geometry. The caller sizes the plate from the returned centres + the real per-feature radii,
    and the existing edge-land / overlap check GATES validity -- this only proposes a sensible layout.
    ``feature_radius`` should be the LARGEST footprint radius in the group, so uniform spacing is
    conservative (smaller features keep more than the minimum)."""
    if n <= 0:
        return []
    r, m = feature_radius, min_wall
    sep = 2.0 * r + m + slack  # min centre-to-centre between two features
    clear = central_radius + m + r  # min centre distance to clear the central disk

    # A lone feature on an otherwise empty plate: centered (or pushed to one end on request).
    if n == 1 and central_radius <= 0.0:
        if pattern is LayoutPattern.END_OFFSET:
            return [(2.0 * (r + m), 0.0)]  # toward +X; the plate body extends in -X
        return [(0.0, 0.0)]

    # Multiple features and/or a central to clear: the single-only patterns don't apply -> default.
    eff = LayoutPattern.CORNER if pattern in _SINGLE_ONLY else pattern
    if eff is LayoutPattern.ROW:
        return _row(n, sep, central_radius, m, r)
    if eff is LayoutPattern.GRID:
        return _grid(n, sep)
    if eff is LayoutPattern.PERIMETER:
        return _perimeter(n, sep, clear)
    if eff is LayoutPattern.BOLT_CIRCLE:
        return _bolt_circle(n, sep, clear)
    # CORNER (the default): a 2x2-style square for <=4, else fall to an even perimeter ring.
    if n <= 4:
        return _corner(n, sep, clear)
    return _perimeter(n, sep, clear)


def _row(n: int, sep: float, central_radius: float, m: float, r: float) -> list[Pos]:
    """A centered line along X. With a central feature the row drops to -Y to clear it."""
    y = 0.0 if central_radius <= 0.0 else -(central_radius + m + r)
    return [((i - (n - 1) / 2.0) * sep, y) for i in range(n)]


def _grid(n: int, sep: float) -> list[Pos]:
    """A centered near-square grid (cols = ceil(sqrt n)), pitch ``sep``."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    out: list[Pos] = []
    for i in range(n):
        rr, cc = divmod(i, cols)
        out.append(((cc - (cols - 1) / 2.0) * sep, (rr - (rows - 1) / 2.0) * sep))
    return out


def _corner(n: int, sep: float, clear: float) -> list[Pos]:
    """The first ``n`` of NW, NE, SW, SE corners of a centered square (matches
    ``rules.centered_rect_corners`` order). ``n`` must be 1..4."""
    c = max(sep / 2.0, clear / math.sqrt(2.0))  # separate adjacent corners AND clear the centre
    corners: list[Pos] = [(-c, c), (c, c), (-c, -c), (c, -c)]
    return corners[:n]


def _bolt_circle(n: int, sep: float, clear: float) -> list[Pos]:
    """``n`` features evenly spaced on a pitch circle (the round-plate analog of the corner pattern).
    The pitch radius is the larger of: enough to clear a central feature, and enough that the chord
    between adjacent holes keeps ``sep`` (so they don't crowd)."""
    if n == 1:
        return [(clear, 0.0)] if clear > 0.0 else [(0.0, 0.0)]
    r_chord = sep / (2.0 * math.sin(math.pi / n))  # chord 2*R*sin(pi/n) >= sep -> min pitch radius
    radius = max(clear, r_chord, sep / 2.0)
    return [
        (radius * math.cos(2.0 * math.pi * i / n), radius * math.sin(2.0 * math.pi * i / n))
        for i in range(n)
    ]


def _perimeter(n: int, sep: float, clear: float) -> list[Pos]:
    """``n`` points evenly spaced by arc-length around a centered square, sized to hold them with
    ``sep`` spacing and to clear the central disk."""
    c = max(n * sep / 8.0, clear / math.sqrt(2.0), sep / 2.0)  # 8c = square perimeter
    per = 8.0 * c
    return [_point_on_square((i / n) * per, c) for i in range(n)]


def _point_on_square(d: float, c: float) -> Pos:
    """Arc-length ``d`` (clockwise from the NW corner) onto a centered square of half-size ``c``."""
    side = 2.0 * c
    d %= 4.0 * side
    if d < side:  # top edge: NW(-c, c) -> NE(c, c)
        return (-c + d, c)
    if d < 2.0 * side:  # right edge: NE -> SE(c, -c)
        return (c, c - (d - side))
    if d < 3.0 * side:  # bottom edge: SE -> SW(-c, -c)
        return (c - (d - 2.0 * side), -c)
    return (-c, -c + (d - 3.0 * side))  # left edge: SW -> NW


__all__ = ["LayoutPattern", "Pos", "derive_positions"]
