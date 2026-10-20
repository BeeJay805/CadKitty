"""Deterministic constraint inference over sketches (ADR-0025, Phase 3)."""

from __future__ import annotations

import math

from forge_mind.intake.constraints import infer_constraints, summarize_constraints
from forge_mind.intake.schema import (
    ConstraintKind,
    InferredConstraint,
    Sketch,
    SketchPoint,
    SketchStroke,
)


def _line(
    sid: str, x0: float, y0: float, x1: float, y1: float, stype: str = "line"
) -> SketchStroke:
    return SketchStroke(
        id=sid, type=stype, points=[SketchPoint(x=x0, y=y0), SketchPoint(x=x1, y=y1)]
    )


def _circle(sid: str, cx: float, cy: float, r: float, n: int = 24) -> SketchStroke:
    pts = [
        SketchPoint(
            x=cx + r * math.cos(2 * math.pi * k / n), y=cy + r * math.sin(2 * math.pi * k / n)
        )
        for k in range(n)
    ]
    return SketchStroke(id=sid, type="circle", points=pts)


def _rect(sid: str, x0: float, y0: float, x1: float, y1: float) -> SketchStroke:
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return SketchStroke(id=sid, type="rect", points=[SketchPoint(x=a, y=b) for a, b in corners])


def _sketch(*strokes: SketchStroke, w: int = 800, h: int = 600) -> Sketch:
    return Sketch(canvas_width=w, canvas_height=h, strokes=list(strokes))


def _kinds(cs: list[InferredConstraint]) -> set[ConstraintKind]:
    return {c.kind for c in cs}


def test_horizontal_vertical_perpendicular_equal() -> None:
    sk = _sketch(_line("h", 100, 100, 300, 100), _line("v", 100, 100, 100, 300))
    kinds = _kinds(infer_constraints(sk))
    assert ConstraintKind.HORIZONTAL in kinds
    assert ConstraintKind.VERTICAL in kinds
    assert ConstraintKind.PERPENDICULAR in kinds
    assert ConstraintKind.EQUAL in kinds  # both 200 long


def test_parallel_equal_lines() -> None:
    sk = _sketch(_line("a", 100, 100, 300, 100), _line("b", 100, 200, 300, 200))
    kinds = _kinds(infer_constraints(sk))
    assert ConstraintKind.PARALLEL in kinds
    assert ConstraintKind.EQUAL in kinds
    assert ConstraintKind.PERPENDICULAR not in kinds


def test_concentric_circles_not_equal_radius() -> None:
    sk = _sketch(_circle("c1", 400, 300, 50), _circle("c2", 400, 300, 110))
    kinds = _kinds(infer_constraints(sk))
    assert ConstraintKind.CONCENTRIC in kinds
    # 50 vs 110 differ > 10% -> not "equal radius"
    assert ConstraintKind.EQUAL not in kinds


def test_equal_radius_distinct_centers_not_concentric() -> None:
    sk = _sketch(_circle("c1", 200, 300, 50), _circle("c2", 600, 300, 50))
    kinds = _kinds(infer_constraints(sk))
    assert ConstraintKind.EQUAL in kinds
    assert ConstraintKind.CONCENTRIC not in kinds


def test_rect_stroke_expands_to_perp_and_equal() -> None:
    sk = _sketch(_rect("box", 100, 100, 300, 200))  # 200 wide x 100 tall
    kinds = _kinds(infer_constraints(sk))
    assert ConstraintKind.PERPENDICULAR in kinds
    assert ConstraintKind.EQUAL in kinds  # top==bottom, left==right
    assert ConstraintKind.HORIZONTAL in kinds
    assert ConstraintKind.VERTICAL in kinds


def test_symmetry_detected_for_mirrored_layout() -> None:
    sk = _sketch(
        _circle("a", 200, 200, 40),
        _circle("b", 600, 200, 40),
        _circle("c", 200, 400, 40),
        _circle("d", 600, 400, 40),
    )
    cs = infer_constraints(sk)
    sym = [c for c in cs if c.kind is ConstraintKind.SYMMETRY]
    assert sym, "a mirrored 4-circle layout should read as symmetric"
    assert any("vertical" in r for c in sym for r in c.refs)


def test_angle_tolerance_band() -> None:
    near = infer_constraints(_sketch(_line("x", 100, 100, 300, 109)))  # ~2.6 deg off horizontal
    far = infer_constraints(_sketch(_line("y", 100, 100, 300, 150)))  # ~14 deg off horizontal
    assert ConstraintKind.HORIZONTAL in _kinds(near)
    assert ConstraintKind.HORIZONTAL not in _kinds(far)


def test_empty_and_degenerate_yield_nothing() -> None:
    assert infer_constraints(_sketch()) == []
    tiny = SketchStroke(
        id="dot", type="line", points=[SketchPoint(x=10, y=10), SketchPoint(x=12, y=11)]
    )
    assert infer_constraints(_sketch(tiny)) == []  # below the min-length noise floor


def test_results_are_deterministic_and_sorted() -> None:
    sk = _sketch(_line("h", 100, 100, 300, 100), _line("v", 100, 100, 100, 300))
    first = infer_constraints(sk)
    second = infer_constraints(sk)
    assert [(c.kind, c.refs) for c in first] == [(c.kind, c.refs) for c in second]


def test_summarize_constraints_reads_human() -> None:
    cs = [
        InferredConstraint(kind=ConstraintKind.SYMMETRY, refs=["axis:vertical"]),
        InferredConstraint(kind=ConstraintKind.CONCENTRIC, refs=["c1", "c2"]),
        InferredConstraint(kind=ConstraintKind.PERPENDICULAR, refs=["a", "b"]),
        InferredConstraint(kind=ConstraintKind.EQUAL, refs=["a", "b"]),
    ]
    lines = summarize_constraints(cs)
    blob = " ".join(lines).lower()
    assert "symmetric about a vertical axis" in blob
    assert "concentric" in blob
    assert "right angles" in blob
    assert "equal" in blob
