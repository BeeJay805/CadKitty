"""CPT-style constraint-stability tests (DAVINCI's Constraint-Preserving-Transformation idea, borrowed
not vendored). Two guarantees, both pure Python -- no solver, no dataset download:

  * INVARIANCE: the inferred constraint set is unchanged under sub-threshold stroke jitter.
  * SENSITIVITY: a constraint DROPS exactly when a perturbation crosses the module's tolerance.

This hardens the docstring-claimed Onshape/SolveSpace thresholds in ``intake/constraints.py`` into
regression-guarded behavior -- the first measurable "constraint stability" property cadkitty has.
"""

from __future__ import annotations

import random

from forge_mind.intake.constraints import infer_constraints
from forge_mind.intake.schema import ConstraintKind, Sketch, SketchPoint, SketchStroke


def _line(
    sid: str, x0: float, y0: float, x1: float, y1: float, stype: str = "line"
) -> SketchStroke:
    return SketchStroke(
        id=sid, type=stype, points=[SketchPoint(x=x0, y=y0), SketchPoint(x=x1, y=y1)]
    )


def _sketch(*strokes: SketchStroke, w: int = 800, h: int = 600) -> Sketch:
    return Sketch(canvas_width=w, canvas_height=h, strokes=list(strokes))


def _set(sk: Sketch) -> set[tuple[ConstraintKind, tuple[str, ...]]]:
    return {(c.kind, tuple(c.refs)) for c in infer_constraints(sk)}


def _jitter(stroke: SketchStroke, rng: random.Random, amp: float) -> SketchStroke:
    return SketchStroke(
        id=stroke.id,
        type=stroke.type,
        points=[
            SketchPoint(x=p.x + rng.uniform(-amp, amp), y=p.y + rng.uniform(-amp, amp))
            for p in stroke.points
        ],
    )


# Two clearly-equal horizontal lines: HORIZONTAL(a), HORIZONTAL(b), PARALLEL(a,b), EQUAL(a,b).
def _parallel_pair() -> Sketch:
    return _sketch(_line("a", 100, 100, 300, 100), _line("b", 100, 300, 300, 300))


def test_subthreshold_jitter_preserves_the_constraint_set() -> None:
    """Jitter well under every tolerance (~2 px on 200 px edges => <0.6 deg, <2% length) must not add
    or drop a single constraint, across many seeds."""
    base = _parallel_pair()
    baseline = _set(base)
    assert baseline  # sanity: the fixture really does infer constraints
    for seed in range(30):
        rng = random.Random(seed)
        jittered = _sketch(*[_jitter(s, rng, amp=2.0) for s in base.strokes])
        assert _set(jittered) == baseline, f"seed {seed} changed the inferred set under tiny jitter"


def test_jitter_is_reproducible_per_seed() -> None:
    """Same seed -> identical perturbation -> identical inference (determinism end to end)."""
    base = _parallel_pair()
    a = _sketch(*[_jitter(s, random.Random(7), amp=3.0) for s in base.strokes])
    b = _sketch(*[_jitter(s, random.Random(7), amp=3.0) for s in base.strokes])
    assert _set(a) == _set(b)


def test_crossing_the_angle_tolerance_drops_parallel_and_horizontal() -> None:
    """Tilting one edge past ``_ANGLE_TOL_DEG`` (5 deg) removes its HORIZONTAL and the PARALLEL pair,
    while the untouched edge keeps its HORIZONTAL -- a constraint dropping exactly at the threshold."""
    baseline = _set(_parallel_pair())
    assert (ConstraintKind.PARALLEL, ("a", "b")) in baseline
    assert (ConstraintKind.HORIZONTAL, ("b",)) in baseline

    # Rotate edge "b" to ~14 deg (dy 50 over dx 200): clears the 5 deg band for horizontal + parallel.
    tilted = _sketch(_line("a", 100, 100, 300, 100), _line("b", 100, 300, 300, 350))
    after = _set(tilted)
    assert (ConstraintKind.PARALLEL, ("a", "b")) not in after
    assert (ConstraintKind.HORIZONTAL, ("b",)) not in after
    assert (ConstraintKind.HORIZONTAL, ("a",)) in after  # the unperturbed edge survives


def test_crossing_the_equal_tolerance_drops_equal() -> None:
    """Stretching one edge past ``_EQUAL_TOL`` (8%) removes EQUAL while PARALLEL/HORIZONTAL survive."""
    baseline = _set(_parallel_pair())
    assert (ConstraintKind.EQUAL, ("a", "b")) in baseline

    # "a" stays 200 long; "b" becomes 260 long (+30%, well past 8%).
    stretched = _sketch(_line("a", 100, 100, 300, 100), _line("b", 100, 300, 360, 300))
    after = _set(stretched)
    assert (ConstraintKind.EQUAL, ("a", "b")) not in after
    assert (ConstraintKind.PARALLEL, ("a", "b")) in after  # still parallel + horizontal
    assert (ConstraintKind.HORIZONTAL, ("b",)) in after
