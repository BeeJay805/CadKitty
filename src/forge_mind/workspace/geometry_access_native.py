"""Saved-body finite cylinder sweeps; CAD imports occur only during evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge_mind.workspace.geometry_access_contract import (
        EvaluatedCylinder,
        EvaluatedMotion,
        ResolvedAccessBody,
    )


def _facts(shape: Any) -> dict[str, Any]:
    from forge_mind.workspace.geometry_distance_native import _number

    volume = _number(shape.volume, "solid volume")
    if volume <= 0:
        raise ValueError("Access native solid has no positive volume")
    bounds = shape.bounding_box()
    lower = [_number(v, "bound") for v in bounds.min]
    upper = [_number(v, "bound") for v in bounds.max]
    if any(a > b for a, b in zip(lower, upper, strict=True)):
        raise ValueError("Access native solid bounds are unordered")
    return {"volume_mm3": volume, "bbox_min_mm": lower, "bbox_max_mm": upper}


def _common_operation(left: Any, right: Any) -> Any:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common

    return BRepAlgoAPI_Common(left.wrapped, right.wrapped)


def _common_result(operation: Any) -> list[dict[str, Any]]:
    """Only a successfully completed native null result can mean empty material."""
    import build123d as cad
    from OCP.BRepCheck import BRepCheck_Analyzer

    from forge_mind.workspace.geometry_access_contract import MAX_OVERLAP_COMPONENTS

    operation.Build()
    if operation.IsDone() is not True:
        raise ValueError("Access native Common did not complete")
    raw = operation.Shape()
    if raw is None:
        raise ValueError("Access native Common returned no result object")
    if raw.IsNull():
        return []
    if not BRepCheck_Analyzer(raw).IsValid():
        raise ValueError("Access native Common result is invalid")
    result = cad.Compound.cast(raw)
    solids = list(result.solids())
    if len(solids) > MAX_OVERLAP_COMPONENTS:
        raise ValueError("Access overlap component limit exceeded")
    components = []
    for solid in solids:
        if (
            solid.wrapped is None
            or solid.wrapped.IsNull()
            or not BRepCheck_Analyzer(solid.wrapped).IsValid()
        ):
            raise ValueError("Access native overlap component is invalid")
        components.append(_facts(solid))
    # Valid lower-dimensional touching results have no positive solid components.
    return components


def _travel(cylinder: EvaluatedCylinder, motion: EvaluatedMotion) -> float:
    from forge_mind.workspace.geometry_access_contract import EvaluatedCylinder, EvaluatedMotion

    if not isinstance(cylinder, EvaluatedCylinder) or not isinstance(motion, EvaluatedMotion):
        raise ValueError("Access requires evaluated frame and motion")
    cylinder = EvaluatedCylinder.model_validate(cylinder.model_dump(mode="json"))
    motion = EvaluatedMotion.model_validate(motion.model_dump(mode="json"))
    if motion.start_center_mm != cylinder.center_mm:
        raise ValueError("Access motion start differs from cylinder center")
    if any(
        not math.isfinite(a + b) or a + b != end
        for a, b, end in zip(
            cylinder.center_mm, motion.translation_mm, motion.end_center_mm, strict=True
        )
    ):
        raise ValueError("Access motion endpoint differs from translation")
    if motion.mode == "static":
        if any(motion.translation_mm):
            raise ValueError("Static access cannot translate")
        return 0.0
    direction = cylinder.frame_axes.x if motion.mode == "transverse" else cylinder.axis
    index = max(range(3), key=lambda i: abs(direction[i]))
    travel = motion.translation_mm[index] / direction[index]
    if (
        not math.isfinite(travel)
        or travel == 0
        or any(
            not math.isclose(value, travel * axis, rel_tol=1e-12, abs_tol=1e-12)
            for value, axis in zip(motion.translation_mm, direction, strict=True)
        )
    ):
        raise ValueError("Access translation differs from declared local motion")
    return travel


def _sweep(cylinder: EvaluatedCylinder, motion: EvaluatedMotion) -> Any:
    import build123d as cad
    from OCP.BRepCheck import BRepCheck_Analyzer

    travel = _travel(cylinder, motion)
    radius, length = cylinder.radius_mm, cylinder.length_mm
    if motion.mode == "transverse":
        profile = cad.Pos(travel / 2, 0, -length / 2) * cad.SlotOverall(
            width=abs(travel) + 2 * radius, height=2 * radius
        )
        local = cad.extrude(profile, amount=length)
    else:
        local = cad.Pos(0, 0, travel / 2) * cad.Cylinder(radius, length + abs(travel))
    plane = cad.Plane(origin=cylinder.center_mm, x_dir=cylinder.frame_axes.x, z_dir=cylinder.axis)
    shape = plane * local
    if not isinstance(shape, cad.Shape):
        raise ValueError("Access native placement did not return a shape")
    if (
        shape.wrapped is None
        or shape.wrapped.IsNull()
        or not BRepCheck_Analyzer(shape.wrapped).IsValid()
    ):
        raise ValueError("Access native swept solid is invalid")
    solids = list(shape.solids())
    if len(solids) != 1 or not BRepCheck_Analyzer(solids[0].wrapped).IsValid():
        raise ValueError("Access sweep must be one connected valid solid")
    _facts(solids[0])
    return solids[0]


def _selected_bodies(
    step_path: Path,
    topology: dict[str, Any],
    obstacles: list[ResolvedAccessBody],
    inventory_body_count: int,
) -> list[Any]:
    import build123d as cad
    from OCP.BRepCheck import BRepCheck_Analyzer

    from forge_mind.workspace.assembly import named_solids, occurrence_label
    from forge_mind.workspace.geometry_access_contract import MAX_OBSTACLES, ResolvedAccessBody
    from forge_mind.workspace.topology import same_exported_solid, signature

    if type(inventory_body_count) is not int or inventory_body_count < 1:
        raise ValueError("Access requires a complete positive saved body inventory")
    if not isinstance(obstacles, list) or not 1 <= len(obstacles) <= MAX_OBSTACLES:
        raise ValueError("Access selected obstacle count is outside limits")
    if any(not isinstance(row, ResolvedAccessBody) for row in obstacles):
        raise ValueError("Access requires resolved saved body selectors")
    obstacles = [
        ResolvedAccessBody.model_validate(row.model_dump(mode="json")) for row in obstacles
    ]
    if len({row.body_index for row in obstacles}) != len(obstacles):
        raise ValueError("Access obstacle aliases resolve to a duplicate body")
    if not isinstance(topology, dict) or topology.get("frame") != "cad_z_up_mm":
        raise ValueError("Access topology has no supported saved coordinate frame")
    entities = topology.get("entities")
    if not isinstance(entities, list) or any(not isinstance(row, dict) for row in entities):
        raise ValueError("Access topology has an invalid entity inventory")
    shape = cad.import_step(step_path)
    if (
        shape is None
        or shape.wrapped is None
        or shape.wrapped.IsNull()
        or not BRepCheck_Analyzer(shape.wrapped).IsValid()
    ):
        raise ValueError("Saved access STEP is invalid or empty")
    solids = list(shape.solids())
    bodies = [row for row in entities if row.get("kind") == "body"]
    indices = [row.get("body_index") for row in bodies]
    if (
        not solids
        or len(solids) != inventory_body_count
        or len(bodies) != len(solids)
        or any(type(index) is not int for index in indices)
        or set(indices) != set(range(len(solids)))
    ):
        raise ValueError("Saved access body inventory is missing or ambiguous")
    counts = topology.get("topology_counts")
    if isinstance(counts, dict) and (
        type(counts.get("bodies")) is not int or counts["bodies"] != len(solids)
    ):
        raise ValueError("Saved access topology body count disagrees with STEP")
    signatures = []
    for solid in solids:
        if (
            solid.wrapped is None
            or solid.wrapped.IsNull()
            or not BRepCheck_Analyzer(solid.wrapped).IsValid()
        ):
            raise ValueError("Saved access native body is invalid")
        _facts(solid)
        signatures.append(signature(solid))
    for row in bodies:
        if (
            row.get("signature") != signatures[row["body_index"]]
            or row.get("body_signature") != signatures[row["body_index"]]
        ):
            raise ValueError("Saved access body signature disagrees with STEP placement")
    named = (
        named_solids(shape)
        if any(row.requested.occurrence_path is not None for row in obstacles)
        else {}
    )
    selected = []
    for row in obstacles:
        index = row.body_index
        if index >= len(solids) or signatures[index] != row.body_signature:
            raise ValueError("Selected access body identity disagrees with STEP")
        if signatures.count(row.body_signature) != 1:
            raise ValueError("Selected access saved body correspondence is ambiguous")
        if row.requested.occurrence_path is not None:
            occurrence = named.get(occurrence_label(row.requested.occurrence_path))
            matches = (
                []
                if occurrence is None
                else [i for i, body in enumerate(solids) if same_exported_solid(occurrence, body)]
            )
            if matches != [index]:
                raise ValueError("Selected access occurrence label disagrees with saved body")
        selected.append(solids[index])
    return selected


def inspect_selected_cylindrical_access(
    step_path: Path,
    topology: dict[str, Any],
    evaluated_cylinder: EvaluatedCylinder,
    evaluated_motion: EvaluatedMotion,
    obstacles: list[ResolvedAccessBody],
    inventory_body_count: int,
) -> dict[str, Any]:
    """Observe complete static/axial/transverse finite-cylinder material and distance."""
    from forge_mind.workspace.geometry_access_contract import (
        SCOPE,
        AccessObservation,
        NumericalGuards,
        OverlapComponent,
        SweepFact,
        classify_observation,
    )
    from forge_mind.workspace.geometry_distance_native import _distance_operation, _distance_result

    selected = _selected_bodies(step_path, topology, obstacles, inventory_body_count)
    sweep = _sweep(evaluated_cylinder, evaluated_motion)
    method = {
        "static": "static_cylinder",
        "axial": "axial_extended_cylinder",
        "transverse": "transverse_capsule_prism",
    }[evaluated_motion.mode]
    fact = SweepFact.model_validate({"method": method, **_facts(sweep)})
    observations = []
    for body, selector in zip(selected, obstacles, strict=True):
        components = _common_result(_common_operation(sweep, body))
        try:
            volume = math.fsum(row["volume_mm3"] for row in components)
        except OverflowError as exc:
            raise ValueError("Access overlap volume sum is nonfinite") from exc
        operation = _distance_operation()
        operation.LoadS1(sweep.wrapped)
        operation.LoadS2(body.wrapped)
        operation.Perform()
        distance = _distance_result(operation)
        row = AccessObservation(
            obstacle=selector,
            overlap_components=[OverlapComponent.model_validate(item) for item in components],
            overlap_volume_mm3=volume,
            minimum_distance_mm=distance["minimum_distance_mm"],
            witness_sweep_mm=distance["witness_a_mm"],
            witness_obstacle_mm=distance["witness_b_mm"],
            solution_count=distance["solution_count"],
            native_inner_solution=distance["native_inner_solution"],
            native_completed=True,
            relation=classify_observation(volume, distance["minimum_distance_mm"]),
        )
        observations.append(row.model_dump(mode="json"))
    return {
        "sweep": fact.model_dump(mode="json"),
        "observations": observations,
        "numerical_guards": NumericalGuards().model_dump(mode="json"),
        "native_completed": True,
        "scope": SCOPE,
    }
