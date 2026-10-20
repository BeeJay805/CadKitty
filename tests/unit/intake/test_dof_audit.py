"""DoF auditor tests (AutoConstrain idea: leftover degrees of freedom are the accuracy enemy). The
auditor is ADVISORY and solver-free -- it flags primitives that participate in NO inferred constraint
(they float independently on the next edit). These tests pin that behavior so the heuristic can't
silently drift."""

from __future__ import annotations

import math

from forge_mind.intake.constraints import audit_dof
from forge_mind.intake.schema import ConstrainedState, Sketch, SketchPoint, SketchStroke


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


def test_lone_tilted_line_is_under_pinned() -> None:
    # ~30 deg off horizontal -> no h/v, no pair, no equal -> participates in no constraint.
    audit = audit_dof(_sketch(_line("solo", 100, 100, 300, 215)))
    assert audit.state is ConstrainedState.UNDER
    assert audit.under_pinned == ["solo"]
    assert audit.free_dof == 1


def test_rect_stroke_is_fully_pinned() -> None:
    # A rect expands to 4 axis-aligned edges; each picks up horizontal/vertical/parallel/equal.
    audit = audit_dof(_sketch(_rect("box", 100, 100, 300, 200)))
    assert audit.state is ConstrainedState.FULLY
    assert audit.under_pinned == []
    assert audit.free_dof == 0


def test_equal_parallel_pair_is_pinned() -> None:
    audit = audit_dof(_sketch(_line("a", 100, 100, 300, 100), _line("b", 100, 300, 300, 300)))
    assert audit.state is ConstrainedState.FULLY
    assert audit.under_pinned == []


def test_isolated_circle_floats() -> None:
    audit = audit_dof(_sketch(_circle("hole", 400, 300, 50)))
    assert audit.state is ConstrainedState.UNDER
    assert audit.under_pinned == ["hole"]


def test_mixed_pins_some_and_floats_others() -> None:
    # Two equal horizontals (pinned) + one lonely tilted edge (floats).
    audit = audit_dof(
        _sketch(
            _line("a", 100, 100, 300, 100),
            _line("b", 100, 200, 300, 200),
            _line("free", 400, 400, 470, 520),
        )
    )
    assert audit.state is ConstrainedState.UNDER
    assert audit.under_pinned == ["free"]
    assert audit.free_dof == 1


def test_empty_sketch_is_vacuously_fully() -> None:
    audit = audit_dof(_sketch())
    assert audit.state is ConstrainedState.FULLY
    assert audit.under_pinned == []


def test_audit_is_deterministic() -> None:
    sk = _sketch(_line("a", 100, 100, 300, 100), _circle("c", 400, 300, 50))
    first, second = audit_dof(sk), audit_dof(sk)
    assert first == second
