"""Small deterministic nonlinear sketch solver with explicit residual and local DOF."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from forge_mind.workspace.document import Sketch


def solve_sketch(sketch: Sketch) -> dict[str, Any]:
    point_index = {p.id: i * 2 for i, p in enumerate(sketch.points)}
    entities = {e.id: e for e in sketch.entities}
    circles = [e for e in sketch.entities if e.kind == "circle"]
    radius_index = {e.id: len(point_index) * 2 + i for i, e in enumerate(circles)}
    initial = np.array(
        [v for p in sketch.points for v in (p.x, p.y)] + [e.radius for e in circles], dtype=float
    )
    if not len(initial):
        raise ValueError("Sketch has no geometry")

    def residual(x: Any, grouped: bool = False) -> Any:
        groups: list[tuple[str, list[float]]] = []

        def point(name: str) -> Any:
            i = point_index[name]
            return x[i : i + 2]

        def line(name: str) -> tuple[Any, Any]:
            entity = entities[name]
            if entity.kind != "line":
                raise ValueError("Constraint requires a line")
            return point(entity.points[0]), point(entity.points[1])

        def direction(name: str) -> Any:
            a, b = line(name)
            delta = b - a
            length = np.linalg.norm(delta)
            if length < 1e-8:
                raise ValueError("Constraint references a degenerate line")
            return delta / length

        for c in sketch.constraints:
            ps, es = c.points, c.entities
            try:
                if c.kind == "fix":
                    i = point_index[ps[0]]
                    values = list(point(ps[0]) - initial[i : i + 2])
                elif c.kind == "coincident":
                    values = list(point(ps[0]) - point(ps[1]))
                elif c.kind in {"distance", "distance_x", "distance_y"}:
                    delta = point(ps[1]) - point(ps[0])
                    value = (
                        np.linalg.norm(delta)
                        if c.kind == "distance"
                        else delta[0 if c.kind == "distance_x" else 1]
                    )
                    values = [value - c.value]
                elif c.kind in {"horizontal", "vertical"}:
                    a, b = line(es[0])
                    values = [float((b - a)[1 if c.kind == "horizontal" else 0])]
                elif c.kind in {"parallel", "perpendicular"}:
                    a, b = direction(es[0]), direction(es[1])
                    values = [
                        float(a[0] * b[1] - a[1] * b[0])
                        if c.kind == "parallel"
                        else float(np.dot(a, b))
                    ]
                elif c.kind == "equal":
                    a, b = entities[es[0]], entities[es[1]]
                    if a.kind != b.kind:
                        raise ValueError("Equal requires two lines or two circles")
                    if a.kind == "circle":
                        values = [x[radius_index[a.id]] - x[radius_index[b.id]]]
                    else:
                        aa, ab = line(a.id)
                        ba, bb = line(b.id)
                        values = [np.linalg.norm(ab - aa) - np.linalg.norm(bb - ba)]
                elif c.kind == "radius":
                    if c.value <= 0:
                        raise ValueError("Radius must be positive")
                    values = [x[radius_index[es[0]]] - c.value]
                elif c.kind == "symmetry":
                    values = list((point(ps[0]) + point(ps[1])) / 2 - point(ps[2]))
                elif c.kind == "tangent":
                    circle_ids = [e for e in es if entities[e].kind == "circle"]
                    if len(circle_ids) == 2:
                        a, b = circle_ids
                        values = [
                            np.linalg.norm(
                                point(entities[a].points[0]) - point(entities[b].points[0])
                            )
                            - x[radius_index[a]]
                            - x[radius_index[b]]
                        ]
                    elif len(circle_ids) == 1:
                        cid = circle_ids[0]
                        lid = next(e for e in es if entities[e].kind == "line")
                        a, _ = line(lid)
                        d = direction(lid)
                        delta = point(entities[cid].points[0]) - a
                        values = [abs(delta[0] * d[1] - delta[1] * d[0]) - x[radius_index[cid]]]
                    else:
                        raise ValueError("Tangent needs a circle")
                else:
                    raise ValueError("Unsupported constraint")
            except (IndexError, KeyError, StopIteration) as exc:
                raise ValueError(f"Constraint {c.id} has wrong reference arity or type") from exc
            groups.append((c.id, [float(v) for v in values]))
        return groups if grouped else np.array([v for _, values in groups for v in values])

    residual(initial)  # validate even if the solver has no work
    lower = np.full(len(initial), -np.inf)
    upper = np.full(len(initial), np.inf)
    for i in radius_index.values():
        lower[i] = 1e-5
    if sketch.constraints:
        fit = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=300,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        x = fit.x
        rank = int(np.linalg.matrix_rank(fit.jac, tol=1e-7))
    else:
        x, rank = initial, 0
    groups = residual(x, True)
    conflicts = [cid for cid, values in groups if max(map(abs, values), default=0) > 1e-5]
    solved = sketch.model_copy(deep=True)
    for point in solved.points:
        i = point_index[point.id]
        point.x, point.y = float(x[i]), float(x[i + 1])
    for entity in solved.entities:
        if entity.kind == "circle":
            entity.radius = float(x[radius_index[entity.id]])
    return {
        "sketch": solved.model_dump(),
        "status": "conflicting"
        if conflicts
        else "underconstrained"
        if len(x) > rank
        else "constrained",
        "degrees_of_freedom": len(x) - rank,
        "conflicts": conflicts,
        "max_residual": max((abs(v) for _, values in groups for v in values), default=0),
        "method": "Bounded nonlinear least squares; DOF is local Jacobian nullity, not a global uniqueness proof. Tangency of two circles means external tangency; symmetry means midpoint symmetry.",
    }
