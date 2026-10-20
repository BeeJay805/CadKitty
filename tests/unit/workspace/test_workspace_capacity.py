"""Large native documents remain editable, measurable and explicitly inspectable."""

from __future__ import annotations

from pathlib import Path

import build123d as b
import numpy as np
import pytest

from forge_mind.workspace.assembly import compile_assembly, expand
from forge_mind.workspace.capacity import enforce_limit, optional_limit
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.inspection import design_graph, query_document, reachable, variable_impact
from forge_mind.workspace.motion_check import MotionCheckSettings, configurations, review_poses
from forge_mind.workspace.motion_rig import MotionRig, group_matrices, resolve_rig
from forge_mind.workspace.topology import topology_index


def large_document() -> DesignDocument:
    return DesignDocument.model_validate(
        {
            "variables": {f"v{i}": 1.0 for i in range(300)},
            "assembly": {
                "parts": [
                    {
                        "id": f"p{i}",
                        "name": f"Part {i}",
                        "features": [
                            {
                                "id": "body",
                                "name": "Body",
                                "kind": "box",
                                "params": {"y": 1, "z": 1},
                                "bindings": {"x": f"v{i}"},
                            }
                        ],
                        "output": "body",
                    }
                    for i in range(65)
                ],
                "roots": [
                    {"id": f"o{i}", "definition": f"p{i % 65}", "position": [i * 4, 0, 0]}
                    for i in range(130)
                ],
            },
        }
    )


def test_large_native_export_cached_edit_and_complete_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from forge_mind.workspace import measure

    # Other agents edit source during this slow CAD test. Freeze this loaded instrument's
    # identity; changing on-disk code legitimately invalidates every cached definition.
    instrument = measure.engine_fingerprint()
    monkeypatch.setattr(measure, "engine_fingerprint", lambda: instrument)
    document = large_document()
    original = tmp_path / "original"
    edited = tmp_path / "edited"
    original.mkdir()
    edited.mkdir()
    shape, first = compile_assembly(document, {}, original)
    assert len(shape.solids()) == 130
    assert first[-1]["built"] == 65
    b.export_step(shape, tmp_path / "large.step")
    imported = b.import_step(tmp_path / "large.step")
    assert len(imported.solids()) == 130
    assert imported.volume == pytest.approx(130)
    topology = topology_index(imported, assembly=True)
    assert topology["topology_counts"]["bodies"] == 130
    assert len(topology["meshes"]) == 130
    raw = document.model_dump()
    raw["variables"]["v64"] = 2
    changed = DesignDocument.model_validate(raw)
    updated, second = compile_assembly(changed, {}, edited, original)
    assert updated.volume == pytest.approx(132)
    assert second[-1]["built"] == 1
    assert second[-1]["reused"] == 64
    paths = []
    cursor = None
    while True:
        page = query_document(changed, "revision", "occurrences", cursor=cursor, limit=17)
        paths.extend(row["path"] for row in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert len(paths) == len(set(paths)) == 130
    assert paths == [row["path"] for row in expand(changed)]


def test_pages_reject_stale_revision_document_and_different_scope() -> None:
    document = large_document()
    first = query_document(document, "a", "variables", limit=20)
    cursor = first["next_cursor"]
    for kwargs in ({"revision_id": "b"}, {"scope": "other"}, {"collection": "parts"}):
        args = {"revision_id": "a", "collection": "variables", "cursor": cursor, **kwargs}
        with pytest.raises(ValueError, match="Cursor"):
            query_document(document, **args)
    changed = document.model_copy(deep=True)
    changed.variables["v0"] = 2
    with pytest.raises(ValueError, match="Cursor"):
        query_document(changed, "a", "variables", cursor=cursor)


def test_direct_graph_and_queried_impact_include_curve_control_variables() -> None:
    raw = large_document().model_dump()
    raw["assembly"]["parts"][0]["features"] = [
        {"id": "profile", "name": "Profile", "kind": "profile"},
        {"id": "path", "name": "Path", "kind": "curve", "points": [[0, 0, 0], ["v299", 0, 1]]},
        {"id": "body", "name": "Body", "kind": "sweep", "inputs": ["profile", "path"]},
    ]
    graph = design_graph(DesignDocument.model_validate(raw))
    assert graph["graph_semantics"] == "direct_v2"
    assert variable_impact(graph, ["v299"])["parts"] == ["p0"]
    assert "variable/v299" in reachable(graph, ["part/p0"], reverse=True)
    assert "impact" not in graph


def test_unrestricted_features_points_and_finite_large_parameters() -> None:
    document = DesignDocument.model_validate(
        {
            "variables": {"wide": 1e6},
            "features": [
                {"id": f"f{i}", "name": "Body", "kind": "box", "params": {"x": 1e6, "y": 2, "z": 3}}
                for i in range(140)
            ],
            "outputs": [f"f{i}" for i in range(140)],
        }
    )
    assert len(document.features) == 140
    for value in (float("inf"), float("nan")):
        with pytest.raises(ValueError):
            DesignDocument(variables={"invalid": value})


def test_motion_many_drivers_uses_linear_extreme_cover() -> None:
    count = 40
    rig = resolve_rig(
        MotionRig.model_validate(
            {
                "drivers": [
                    {"id": f"d{i}", "name": "Driver", "unit": "mm", "minimum": 0, "maximum": 10}
                    for i in range(count)
                ],
                "groups": [
                    {
                        "id": f"g{i}",
                        "name": "Group",
                        "driver": f"d{i}",
                        "kind": "slide",
                        "occurrences": [f"o{i}"],
                    }
                    for i in range(count)
                ],
            }
        ),
        {},
    )
    poses = configurations(rig, MotionCheckSettings())
    assert count < len(poses) <= 9 * count + 3
    assert {f"d{i}": 10 for i in range(count)} in poses
    with pytest.raises(ValueError, match="Operator resource policy"):
        configurations(rig, MotionCheckSettings(max_configurations=5))


def test_sparse_large_motion_proves_disjoint_pairs_without_exact_booleans() -> None:
    solids = [b.Pos(i * 4, 0, 0) * b.Box(1, 1, 1) for i in range(130)]
    poses = [({"x": i}, [np.eye(4)] * 130) for i in range(80)]
    result = review_poses(solids, [str(i) for i in range(130)], [None] * 130, poses)
    assert result["coverage"] == "discrete_only"
    assert result["exact_intersections"] == 0
    assert result["disjoint_bounds"] == result["total_pairs"] == 80 * 130 * 129 // 2
    assert result["completed_pairs"] == result["total_pairs"]


def test_optional_capacity_policy_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_WORKSPACE_BODY_LIMIT", raising=False)
    assert optional_limit("BODY_LIMIT") is None
    enforce_limit(1000000, None, "bodies")
    monkeypatch.setenv("FORGE_WORKSPACE_BODY_LIMIT", "129")
    assert optional_limit("BODY_LIMIT") == 129
    with pytest.raises(ValueError, match="requires 130, limit is 129"):
        enforce_limit(130, optional_limit("BODY_LIMIT"), "bodies")
    monkeypatch.setenv("FORGE_WORKSPACE_BODY_LIMIT", "0")
    with pytest.raises(ValueError, match="positive integer"):
        optional_limit("BODY_LIMIT")


def test_deep_motion_hierarchy_does_not_use_python_recursion() -> None:
    rig = {
        "drivers": [{"id": "d", "minimum": 0, "maximum": 2, "reference": 0}],
        "groups": [
            {
                "id": f"g{i}",
                "parent": f"g{i - 1}" if i else None,
                "driver": "d",
                "ratio": 1,
                "kind": "slide",
                "axis": [1, 0, 0],
            }
            for i in range(1100)
        ],
    }
    matrices = group_matrices(rig, {"d": 1})
    assert matrices["g1099"][0, 3] == pytest.approx(1100)


def test_spatial_sweep_retains_exact_collision_witnesses_for_multiple_pairs() -> None:
    solids = [b.Pos(i * 1.5, 0, 0) * b.Box(2, 2, 2) for i in range(8)]
    matrices = []
    for i in range(8):
        matrix = np.eye(4)
        matrix[1, 3] = (i % 3) * 0.4
        matrices.append(matrix)
    names = [f"body{i}" for i in range(8)]
    poses = [({}, matrices)]
    filtered = review_poses(solids, names, names, poses)
    exact = review_poses(solids, names, names, poses, optimized=False)

    def witnesses(result):
        return sorted(
            (row["a"], row["b"], round(row["overlap_mm3"], 8)) for row in result["collisions"]
        )

    assert witnesses(filtered) == witnesses(exact)
    assert filtered["collision_count"] > 1
    assert filtered["exact_intersections"] < exact["exact_intersections"]
    assert filtered["completed_pairs"] == exact["completed_pairs"] == 28


def test_explicit_sampling_budget_interrupts_large_requested_sampling() -> None:
    rig = {"drivers": [{"id": "d", "reference": 0, "minimum": 0, "maximum": 1}]}
    with pytest.raises(ValueError, match="requires 4, limit is 3"):
        configurations(rig, MotionCheckSettings(samples_per_driver=1000000, max_configurations=3))


def test_deep_assembly_expansion_and_cycles_remain_checked() -> None:
    raw = large_document().model_dump()
    raw["assembly"]["subassemblies"] = [
        {
            "id": f"s{i}",
            "name": "Nested",
            "children": [{"id": "child", "definition": f"s{i + 1}" if i < 30 else "p0"}],
        }
        for i in range(31)
    ]
    raw["assembly"]["roots"] = [{"id": "root", "definition": "s0"}]
    document = DesignDocument.model_validate(raw)
    assert len(expand(document)) == 1
    raw["assembly"]["subassemblies"][-1]["children"][0]["definition"] = "s0"
    with pytest.raises(ValueError, match="cyclic"):
        DesignDocument.model_validate(raw)


def test_curve_control_edit_invalidates_only_its_cached_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from forge_mind.workspace import measure

    instrument = measure.engine_fingerprint()
    monkeypatch.setattr(measure, "engine_fingerprint", lambda: instrument)
    raw = {
        "variables": {"length": 5},
        "assembly": {
            "parts": [
                {
                    "id": "tube",
                    "name": "Tube",
                    "features": [
                        {
                            "id": "section",
                            "name": "Section",
                            "kind": "profile",
                            "params": {"radius_x": 1, "radius_y": 1},
                        },
                        {
                            "id": "path",
                            "name": "Path",
                            "kind": "curve",
                            "points": [[0, 0, 0], [0, 0, "length"]],
                        },
                        {
                            "id": "body",
                            "name": "Body",
                            "kind": "sweep",
                            "inputs": ["section", "path"],
                        },
                    ],
                    "output": "body",
                }
            ],
            "roots": [{"id": "instance", "definition": "tube"}],
        },
    }
    original = tmp_path / "first"
    changed = tmp_path / "second"
    original.mkdir()
    changed.mkdir()
    first, _ = compile_assembly(DesignDocument.model_validate(raw), {}, original)
    raw["variables"]["length"] = 8
    second, history = compile_assembly(DesignDocument.model_validate(raw), {}, changed, original)
    assert history[-1]["built"] == 1
    assert history[-1]["reused"] == 0
    assert second.volume / first.volume == pytest.approx(8 / 5)


def test_opaque_definitions_are_regenerated_and_never_share_captured_breps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import itertools
    import json

    from forge_mind.workspace import kernel, measure

    instrument = measure.engine_fingerprint()
    monkeypatch.setattr(measure, "engine_fingerprint", lambda: instrument)
    native_compile = kernel.compile_document
    external_values = itertools.count(1)
    executions = []

    def changing_author(document, assets):
        if document.features[0].kind != "python":
            return native_compile(document, assets)
        value = next(external_values)
        executions.append(value)
        return b.Box(value, 1, 1), []

    # Simulate opaque source observing an undeclared input that changes per execution.
    monkeypatch.setattr(kernel, "compile_document", changing_author)
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": part,
                        "name": part,
                        "features": [
                            {
                                "id": "body",
                                "name": "Opaque source",
                                "kind": "python",
                                "code": "result = b.Box(1, 1, 1)",
                            }
                        ],
                        "output": "body",
                    }
                    for part in ("left", "right")
                ],
                "roots": [{"id": part, "definition": part} for part in ("left", "right")],
            }
        }
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    a, first_history = compile_assembly(document, {}, first)
    z, second_history = compile_assembly(document, {}, second, first)
    assert executions == [1, 2, 3, 4]
    assert a.volume == pytest.approx(3)
    assert z.volume == pytest.approx(7)
    assert second_history[-1]["built"] == 2
    assert second_history[-1]["reused"] == 0
    assert first_history[0]["signature"] != second_history[0]["signature"]
    assert second_history[0]["cache_policy"] == "disabled_for_python_source"
    manifest = json.loads((second / "parts.json").read_text())
    assert manifest["left"]["file"] != manifest["right"]["file"]
    assert b.import_brep(second / manifest["left"]["file"]).volume == pytest.approx(3)
    assert b.import_brep(second / manifest["right"]["file"]).volume == pytest.approx(4)


@pytest.mark.parametrize("assembly", [False, True])
def test_protected_closure_binds_resolved_curve_points(assembly: bool) -> None:
    from forge_mind.workspace.document_service import _closure

    features = [
        {
            "id": "profile",
            "name": "Profile",
            "kind": "profile",
            "params": {"radius_x": 1, "radius_y": 1},
        },
        {"id": "path", "name": "Path", "kind": "curve", "points": [[0, 0, 0], [0, 0, "end"]]},
        {"id": "body", "name": "Body", "kind": "sweep", "inputs": ["profile", "path"]},
    ]
    data = {
        "variables": {"length": 5, "unrelated": 42},
        "relations": [{"name": "end", "operation": "multiply", "a": "length", "b": 2}],
    }
    if assembly:
        data["assembly"] = {
            "parts": [{"id": "tube", "name": "Tube", "features": features, "output": "body"}],
            "roots": [{"id": "instance", "definition": "tube"}],
        }
    else:
        data.update(features=features, outputs=["body"])
    document = DesignDocument.model_validate(data).model_dump()
    target = "tube" if assembly else "body"
    original = _closure(document, target)
    document["variables"]["unrelated"] = 100
    assert _closure(document, target) == original
    document["variables"]["length"] = 8
    assert _closure(document, target) != original
