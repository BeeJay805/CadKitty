"""Completed native Boolean operations for frozen nominal proof only."""

from __future__ import annotations

from typing import Any, Literal

RESIDUAL_UNCERTAINTY_MM3 = 1e-6


def checked_boolean(left: Any, right: Any, kind: Literal["cut", "common"]) -> Any:
    import build123d as b
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut

    operation = (BRepAlgoAPI_Cut if kind == "cut" else BRepAlgoAPI_Common)(
        left.wrapped, right.wrapped
    )
    operation.Build()
    if not operation.IsDone() or operation.Shape().IsNull():
        raise ValueError("Nominal Boolean did not return a completed nonnull shape")
    return b.Compound.cast(operation.Shape())


def material_result(shape: Any) -> dict[str, Any]:
    """Only genuinely empty compounds establish no positive material.

    Small positive solids and nonvolumetric debris remain unverified. Invalid
    zero-volume solids cannot bypass validation via a zero-volume fast path.
    """
    from OCP.TopAbs import TopAbs_COMPOUND

    from forge_mind.workspace.geometry_results import boolean_overlap_components, boolean_volume

    if shape is None or shape.wrapped is None or shape.wrapped.IsNull():
        raise ValueError("Nominal Boolean returned no topology")
    solids = list(shape.solids())
    topology_counts = {
        kind: len(getattr(shape, kind)())
        for kind in ("shells", "faces", "wires", "edges", "vertices")
    }
    if (solids or any(topology_counts.values())) and not shape.is_valid:
        return {
            "status": "unverified",
            "volume_mm3": None,
            "components": [],
            "debris": True,
            "invalid_topology": True,
            "topology_counts": topology_counts,
        }
    volume = boolean_volume(shape)
    components = boolean_overlap_components(shape)
    for component in components:
        component["frame"] = "definition_local"
    debris = any(
        topology_counts[kind] != sum(len(getattr(solid, kind)()) for solid in solids)
        for kind in topology_counts
    )
    genuinely_empty = (
        shape.wrapped.ShapeType() == TopAbs_COMPOUND
        and not solids
        and not any(topology_counts.values())
    )
    status = (
        "fail"
        if any(row["volume_mm3"] > RESIDUAL_UNCERTAINTY_MM3 for row in components)
        else "pass"
        if genuinely_empty and not components and not debris and volume == 0
        else "unverified"
    )
    return {
        "status": status,
        "volume_mm3": volume,
        "components": components,
        "debris": debris,
        "topology_counts": topology_counts,
    }
