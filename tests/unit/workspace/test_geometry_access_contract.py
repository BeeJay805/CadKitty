"""Request and provenance controls, with independent numerical examples."""

import math
from copy import deepcopy

import pytest

from forge_mind.workspace.geometry_access_contract import (
    MAX_REQUEST_BYTES,
    SCOPE,
    AccessContext,
    AccessQuery,
    AccessResult,
    access_tool_result,
    evaluate_query,
)


def query(mode="static", distance=12):
    return {
        "cylinder": {
            "center_mm": [0, 0, 0],
            "axis": [0, 0, 1],
            "radial_reference": [1, 0, 0],
            "radius_mm": 2,
            "length_mm": 10,
        },
        "motion": {"mode": mode, **({} if mode == "static" else {"distance_mm": distance})},
        "obstacles": [{"body_index": 0}],
    }


def context():
    q = AccessQuery.model_validate(query())
    cylinder, motion = evaluate_query(q)
    return {
        "project_id": "1" * 32,
        "revision_id": "2" * 32,
        "query": q.model_dump(mode="json"),
        "evaluated_cylinder": cylinder.model_dump(mode="json"),
        "evaluated_motion": motion.model_dump(mode="json"),
        "obstacles": [
            {"requested": {"body_index": 0}, "body_index": 0, "body_signature": "a" * 64}
        ],
        "inventory_body_count": 3,
        "omitted_body_indices": [1, 2],
        "source_hashes": dict.fromkeys(("part.step", "topology.json", "document.json"), "b" * 64),
        "saved_engine_fingerprint": "c" * 64,
        "query_engine_fingerprint": "d" * 64,
    }


def result():
    value = context()
    value.update(
        sweep={
            "method": "static_cylinder",
            "volume_mm3": 40 * math.pi,
            "bbox_min_mm": [-2, -2, -5],
            "bbox_max_mm": [2, 2, 5],
        },
        observations=[
            {
                "obstacle": value["obstacles"][0],
                "overlap_components": [],
                "overlap_volume_mm3": 0,
                "minimum_distance_mm": 0.25,
                "witness_sweep_mm": [2, 0, 0],
                "witness_obstacle_mm": [2.25, 0, 0],
                "solution_count": 1,
                "native_inner_solution": False,
                "native_completed": True,
                "relation": "separated",
            }
        ],
        numerical_guards={"distance_mm": 1e-6, "volume_mm3": 1e-6},
        native_completed=True,
        scope=SCOPE,
    )
    return value


@pytest.mark.parametrize(
    "mode,distance,translation",
    [
        ("static", 0, (0, 0, 0)),
        ("axial", -12, (0, 0, -12)),
        ("transverse", 12, (12, 0, 0)),
        ("transverse", -12, (-12, 0, 0)),
    ],
)
def test_signed_local_motion(mode, distance, translation):
    cylinder, motion = evaluate_query(AccessQuery.model_validate(query(mode, distance)))
    assert cylinder.frame_axes.x == (1, 0, 0)
    assert motion.translation_mm == translation
    assert motion.end_center_mm == translation


def test_scaled_extreme_frame_and_proper_rotated_translation():
    value = query("transverse")
    value["cylinder"].update(
        center_mm=[10, -7, 20], axis=[1e308, 0, 0], radial_reference=[0, 0, -1e-308]
    )
    cylinder, motion = evaluate_query(AccessQuery.model_validate(value))
    assert cylinder.frame_axes.x == (0, 0, -1)
    assert cylinder.frame_axes.y == (0, 1, 0)
    assert cylinder.frame_axes.z == (1, 0, 0)
    assert motion.translation_mm == (0, 0, -12)
    assert motion.end_center_mm == (10, -7, 8)


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v["cylinder"].update(axis=[0, 0, 0]),
        lambda v: v["cylinder"].update(radial_reference=[0, 0, 2]),
        lambda v: v["cylinder"].update(radial_reference=[1e-14, 0, 1]),
        lambda v: v["cylinder"].update(radius_mm=True),
        lambda v: v["cylinder"].update(length_mm=0),
        lambda v: v["cylinder"].update(center_mm=[float("nan"), 0, 0]),
        lambda v: v["cylinder"].update(radius_mm=1e308),
        lambda v: v["motion"].update(translation_mm=[12, 0, 1]),
        lambda v: v.update(motion={"mode": "mixed", "distance_mm": 12}),
        lambda v: v.update(motion={"mode": "axial", "distance_mm": 0}),
        lambda v: v.update(obstacles=[{"body_index": 0, "occurrence_path": "root/one"}]),
        lambda v: v.update(obstacles=[{"body_index": 0}, {"body_index": 0}]),
        lambda v: v.update(obstacles=[{"body_index": True}]),
    ],
)
def test_invalid_request_or_derived_geometry_refuses(change):
    value = query()
    change(value)
    with pytest.raises(ValueError):
        evaluate_query(AccessQuery.model_validate(value))


def test_multibyte_paths_exceed_wire_cap_without_exceeding_character_cap():
    value = query()
    value["obstacles"] = [{"occurrence_path": f"{i}/" + "界" * 2040} for i in range(16)]
    assert len(str(value).encode("utf-8")) > MAX_REQUEST_BYTES
    with pytest.raises(ValueError, match="request byte"):
        AccessQuery.model_validate(value)


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v["evaluated_motion"].update(translation_mm=[0, 0, 1]),
        lambda v: v["evaluated_cylinder"]["frame_axes"].update(x=[-1, 0, 0], y=[0, -1, 0]),
        lambda v: v.update(omitted_body_indices=[2]),
        lambda v: v.update(omitted_body_indices=[2, 1]),
        lambda v: v.update(omitted_body_indices=[0, 2]),
        lambda v: v.update(inventory_body_count=10**12),
        lambda v: v["source_hashes"].update(extra="e" * 64),
    ],
)
def test_context_recomputes_geometry_and_complete_omissions(change):
    value = context()
    change(value)
    with pytest.raises(ValueError):
        AccessContext.model_validate(value)


def test_path_index_aliases_are_rejected_after_resolution():
    value = context()
    value["query"]["obstacles"].append({"occurrence_path": "root/one"})
    value["obstacles"].append(
        {"requested": {"occurrence_path": "root/one"}, "body_index": 0, "body_signature": "a" * 64}
    )
    with pytest.raises(ValueError, match="duplicate body"):
        AccessContext.model_validate(value)


@pytest.mark.parametrize(
    "volume,distance,relation",
    [
        (0, 0, "contact_or_unresolved"),
        (0, 0.25, "separated"),
        (2, 0, "positive_volume_interference"),
        (1e-9, 0, "contact_or_unresolved"),
    ],
)
def test_measured_relation_and_small_components_retained(volume, distance, relation):
    value = result()
    row = value["observations"][0]
    row.update(
        overlap_volume_mm3=volume,
        minimum_distance_mm=distance,
        witness_sweep_mm=[0, 0, 0],
        witness_obstacle_mm=[distance, 0, 0],
        relation=relation,
    )
    row["overlap_components"] = (
        []
        if not volume
        else [{"volume_mm3": volume, "bbox_min_mm": [-0.5, -0.5, -1], "bbox_max_mm": [0.5, 0.5, 1]}]
    )
    validated = AccessResult.model_validate(value)
    assert len(validated.observations[0].overlap_components) == bool(volume)


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v["observations"][0].update(relation="positive_volume_interference"),
        lambda v: v["observations"][0].update(native_completed=False),
        lambda v: v["observations"][0].update(solution_count=0),
        lambda v: v["observations"][0].update(witness_obstacle_mm=[99, 0, 0]),
        lambda v: v["observations"][0].update(overlap_volume_mm3=2),
        lambda v: v.update(observations=[]),
        lambda v: v["numerical_guards"].update(distance_mm=0.5),
        lambda v: v.update(scope="clear"),
        lambda v: v["sweep"].update(method="transverse_capsule_prism"),
    ],
)
def test_partial_or_forged_result_refuses(change):
    value = result()
    change(value)
    with pytest.raises(ValueError):
        AccessResult.model_validate(value)


def test_tool_binding_and_input_nonmutation():
    value = result()
    original = deepcopy(value)
    q = AccessQuery.model_validate(query())
    returned = access_tool_result(value, "1" * 32, "2" * 32, q)
    assert returned["query"]["cylinder"]["axis"] == [0, 0, 1]
    assert value == original
    with pytest.raises(ValueError, match="revision or query"):
        access_tool_result(value, "3" * 32, "2" * 32, q)
    changed = query("transverse")
    with pytest.raises(ValueError, match="revision or query"):
        access_tool_result(value, "1" * 32, "2" * 32, AccessQuery.model_validate(changed))
