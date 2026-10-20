"""Exact selected saved-shape distance; native imports occur only during evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge_mind.workspace.geometry_distance_contract import ResolvedDistanceSelector


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Native distance returned nonfinite/invalid {label}")
    return float(value)


def _distance_operation() -> Any:
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    return BRepExtrema_DistShapeShape()


def _distance_result(operation: Any) -> dict[str, Any]:
    if operation.IsDone() is not True:
        raise ValueError("Native distance did not complete")
    count = operation.NbSolution()
    if type(count) is not int or count < 1:
        raise ValueError("Native distance returned no valid point solution")
    distance = _number(operation.Value(), "distance")
    if distance < 0:
        raise ValueError("Native distance returned a negative distance")
    points = []
    for point in (operation.PointOnShape1(1), operation.PointOnShape2(1)):
        points.append([_number(v, "witness coordinate") for v in (point.X(), point.Y(), point.Z())])
    separation = math.dist(*points)
    if not math.isfinite(separation) or not math.isclose(
        separation, distance, rel_tol=1e-9, abs_tol=1e-6
    ):
        raise ValueError("Native point witnesses disagree with the measured distance")
    inner = operation.InnerSolution()
    if type(inner) is not bool:
        raise ValueError("Native distance returned an invalid inner-solution flag")
    return {
        "minimum_distance_mm": distance,
        "witness_a_mm": points[0],
        "witness_b_mm": points[1],
        "solution_count": count,
        "native_completed": True,
        "native_inner_solution": inner,
    }


def measure_selected_distance(
    step_path: Path,
    topology: dict[str, Any],
    a: ResolvedDistanceSelector,
    b: ResolvedDistanceSelector,
) -> dict[str, Any]:
    """Measure exact saved solids/trimmed faces, without rebuild or pose adjustment.

    Source hashes and ownership belong to the shared worker/service boundary.
    This evaluator independently checks indexed native correspondence and selection.
    """
    import build123d as cad
    from OCP.BRepCheck import BRepCheck_Analyzer

    from forge_mind.workspace.assembly import named_solids, occurrence_label
    from forge_mind.workspace.geometry_distance_contract import ResolvedDistanceSelector
    from forge_mind.workspace.topology import same_exported_solid, signature

    if not isinstance(a, ResolvedDistanceSelector) or not isinstance(b, ResolvedDistanceSelector):
        raise ValueError("Distance requires resolved saved selectors")
    if (a.body_index, a.face_signature) == (b.body_index, b.face_signature):
        raise ValueError("Select two distinct saved entities")
    if not isinstance(topology, dict) or topology.get("frame") != "cad_z_up_mm":
        raise ValueError("Distance topology has no supported saved coordinate frame")
    entities = topology.get("entities")
    if not isinstance(entities, list) or any(not isinstance(row, dict) for row in entities):
        raise ValueError("Distance topology has an invalid entity inventory")
    shape = cad.import_step(step_path)
    if (
        shape is None
        or shape.wrapped is None
        or shape.wrapped.IsNull()
        or not BRepCheck_Analyzer(shape.wrapped).IsValid()
    ):
        raise ValueError("Saved distance STEP is invalid or empty")
    solids = list(shape.solids())
    if not solids:
        raise ValueError("Saved distance STEP contains no solids")
    bodies = [row for row in entities if row.get("kind") == "body"]
    indices = [row.get("body_index") for row in bodies]
    if (
        any(type(index) is not int for index in indices)
        or len(bodies) != len(solids)
        or set(indices) != set(range(len(solids)))
    ):
        raise ValueError("Saved topology body inventory is missing or ambiguous")
    counts = topology.get("topology_counts")
    if isinstance(counts, dict) and (
        type(counts.get("bodies")) is not int or counts["bodies"] != len(solids)
    ):
        raise ValueError("Saved topology native body count disagrees with STEP")
    body_signatures = [signature(solid) for solid in solids]
    for row in bodies:
        index = row["body_index"]
        if (
            row.get("signature") != body_signatures[index]
            or row.get("body_signature") != body_signatures[index]
        ):
            raise ValueError("Saved topology body signature disagrees with actual STEP placement")

    named = (
        named_solids(shape)
        if a.requested.occurrence_path is not None or b.requested.occurrence_path is not None
        else {}
    )

    def selected(selector: ResolvedDistanceSelector) -> Any:
        index = selector.body_index
        if index >= len(solids) or body_signatures[index] != selector.body_signature:
            raise ValueError("Selected body identity disagrees with actual saved STEP")
        body = solids[index]
        if sum(value == selector.body_signature for value in body_signatures) != 1:
            raise ValueError("Selected saved body correspondence is ambiguous")
        if not BRepCheck_Analyzer(body.wrapped).IsValid():
            raise ValueError("Selected saved native body is invalid")
        if selector.requested.occurrence_path is not None:
            occurrence = named.get(occurrence_label(selector.requested.occurrence_path))
            matches = (
                []
                if occurrence is None
                else [i for i, solid in enumerate(solids) if same_exported_solid(occurrence, solid)]
            )
            if matches != [index]:
                raise ValueError("Selected occurrence label disagrees with indexed saved body")
        if selector.face_signature is None:
            return body
        saved = [
            row
            for row in entities
            if row.get("kind") == "face"
            and type(row.get("body_index")) is int
            and row["body_index"] == index
            and row.get("signature") == selector.face_signature
        ]
        if len(saved) != 1 or saved[0].get("body_signature") != selector.body_signature:
            raise ValueError("Selected saved face inventory is missing or ambiguous")
        face_matches = [face for face in body.faces() if signature(face) == selector.face_signature]
        if len(face_matches) != 1 or not BRepCheck_Analyzer(face_matches[0].wrapped).IsValid():
            raise ValueError("Selected actual trimmed face is missing, invalid or ambiguous")
        return face_matches[0]

    left, right = selected(a), selected(b)
    operation = _distance_operation()
    operation.LoadS1(left.wrapped)
    operation.LoadS2(right.wrapped)
    operation.Perform()
    return _distance_result(operation)
