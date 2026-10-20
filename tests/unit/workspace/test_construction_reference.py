from __future__ import annotations

import asyncio
from typing import get_args

import build123d as b
import httpx
import pytest

from forge_mind.workspace.authoring import authoring_schema
from forge_mind.workspace.construction_reference import OPERATIONS, RECIPES
from forge_mind.workspace.document import DesignDocument, Feature
from forge_mind.workspace.kernel import compile_document
from forge_mind.workspace.parametric_guidance import PARAMETRIC_DESIGN_GUIDANCE


def test_small_index_discovers_every_operation_without_loading_source() -> None:
    index = authoring_schema("construction")
    assert set(index["operations"]) == set(get_args(Feature.model_fields["kind"].annotation))
    assert {recipe["id"] for recipe in index["recipes"]} == set(RECIPES)
    assert len(str(index)) < 6000
    assert "document" not in str(index["recipes"])
    assert index["build123d_version"] == b.__version__
    assert "python" in index["representations"]
    assert index["parametric_design"] == PARAMETRIC_DESIGN_GUIDANCE


def test_focused_references_keep_their_payloads_without_repeating_index_advice() -> None:
    recipe = authoring_schema("construction:mounting_plate")
    assert "parametric_design" not in recipe
    assert recipe["document"] == RECIPES["mounting_plate"]["document"]
    python = authoring_schema("feature:python")
    assert "parametric_design" not in python
    assert "bindings" in python["parameter_rules"]


def test_references_are_independent_and_unknown_queries_preserve_discovery() -> None:
    recipe = authoring_schema("construction:mounting_plate")
    recipe["document"]["variables"]["width"] = -100
    assert authoring_schema("construction:mounting_plate")["document"]["variables"]["width"] == 60
    assert authoring_schema("construction:unknown")["unknown_topic"] == "unknown"
    assert set(authoring_schema("feature:unknown")["available"]) == set(OPERATIONS)
    assert "construction" in authoring_schema("")["available"]
    for kind in OPERATIONS:
        reference = authoring_schema(f"feature:{kind}")
        assert reference["name"] == kind
        assert reference["inputs"] and reference["meaning"] and reference["parameter_rules"]


@pytest.mark.parametrize(
    ("recipe_id", "variable", "new_value"),
    [
        ("native_python_detail", "slot_length", 12),
        ("rounded_enclosure", "width", 90),
        ("swept_tube", "radius", 8),
        ("twisted_impeller", "height", 18),
        ("ribbed_bracket", "width", 80),
        ("mounting_plate", "thickness", 8),
        ("loft_transition", "waist", 14),
    ],
)
def test_retained_recipe_builds_exports_and_rebuilds_bound_parameters(
    recipe_id, variable, new_value, tmp_path
) -> None:
    doc = DesignDocument.model_validate(authoring_schema(f"construction:{recipe_id}")["document"])
    shape, history = compile_document(doc, {})
    assert shape.is_valid and len(shape.solids()) == 1
    assert all(entry["state"] == "built" for entry in history)
    exported_path = tmp_path / "recipe.step"
    b.export_step(shape, str(exported_path))
    exported = b.import_step(str(exported_path))
    assert exported.is_valid and len(exported.solids()) == 1
    assert exported.volume == pytest.approx(shape.volume, rel=1e-6)
    if recipe_id in {"loft_transition", "swept_tube", "twisted_impeller"}:
        assert any("BSPLINE" in str(face.geom_type) for face in exported.faces())
    doc.variables[variable] = new_value
    changed, changed_history = compile_document(doc, {})
    assert changed.is_valid and len(changed.solids()) == 1
    assert changed.volume > shape.volume
    if doc.features[-1].kind == "python":
        assert history[-1]["source_sha256"] == changed_history[-1]["source_sha256"]
        assert len(history[-1]["source_sha256"]) == 64
    solid = exported.solids()[0]
    if recipe_id == "rounded_enclosure":
        assert solid.is_inside((0, 0, 1))
        assert not solid.is_inside((0, 0, 10))
        assert solid.is_inside((32, 22, 1))
        assert not solid.is_inside((32, 22, 10))
        assert pytest.approx(28) == exported.bounding_box().size.Z
    if recipe_id == "mounting_plate":
        assert not solid.is_inside((0, 0, 0))
        assert solid.is_inside((10, 0, 0))
        assert exported.volume == pytest.approx(60 * 40 * 6 - 3.141592653589793 * 25 * 6)
    if recipe_id == "native_python_detail":
        assert solid.is_inside((0, 0, 0))
        assert not solid.is_inside((-15, 0, 0))
        assert not solid.is_inside((15, 0, 0))
        assert pytest.approx(60) == exported.bounding_box().size.X
    if recipe_id == "ribbed_bracket":
        assert pytest.approx(40) == exported.bounding_box().size.Z
        assert not solid.is_inside((27, 12.5, 2))
        assert solid.is_inside((17.5, -18, 15))
    if recipe_id == "twisted_impeller":
        assert not solid.is_inside((0, 0, 7.5))
        assert solid.is_inside((5, 0, 7.5))
        assert pytest.approx(15, abs=1e-5) == exported.bounding_box().size.Z


def test_invalid_recipe_parameters_keep_the_real_feature_failure() -> None:
    doc = DesignDocument.model_validate(
        authoring_schema("construction:rounded_enclosure")["document"]
    )
    doc.variables["wall"] = 10
    with pytest.raises(ValueError, match=r"Feature body .*wall < corner"):
        compile_document(doc, {})


def test_mcp_discovers_and_forwards_focused_references(tmp_path) -> None:
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    queries = []

    def backend(request):
        assert request.method == "GET"
        assert request.url.path == "/workspace/authoring-schema"
        query = request.url.params["query"]
        queries.append(query)
        return httpx.Response(200, json=authoring_schema(query))

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter)) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            reference = tools["get_construction_reference"]
            assert reference.annotations.read_only_hint
            assert "topic" in reference.input_schema["properties"]
            for topic, expected_name in [
                ("", "construction"),
                ("construction", "construction"),
                ("swept_tube", "swept_tube"),
                (
                    authoring_schema("construction")["recipes"][0]["query"],
                    authoring_schema("construction")["recipes"][0]["id"],
                ),
                ("feature:python", "python"),
                ("feature:sweep", "sweep"),
                ("mechanism", "mechanism"),
                ("construction:mechanism", "mechanism"),
            ]:
                response = await client.call_tool("get_construction_reference", {"topic": topic})
                assert not response.is_error
                assert "unknown_topic" not in response.structured_content
                assert response.structured_content["name"] == expected_name
            for kind in ("part", "feature", "construction"):
                response = await client.call_tool("get_schema", {"kind": kind})
                assert not response.is_error
                assert response.structured_content["schema"]["name"] == kind

    asyncio.run(scenario())
    assert queries == [
        "construction:",
        "construction",
        "construction:swept_tube",
        authoring_schema("construction")["recipes"][0]["query"],
        "feature:python",
        "feature:sweep",
        "construction:mechanism",
        "construction:mechanism",
        "part",
        "feature",
        "construction",
    ]


def test_mechanism_reference_exposes_typed_root_mapping_without_a_geometry_recipe() -> None:
    from forge_mind.workspace.mechanism_models import Mechanism

    index = authoring_schema("construction")
    assert "construction:mechanism" in index["representations"]["mechanism"]
    reference = authoring_schema("construction:mechanism")
    assert reference["name"] == "mechanism"
    assert "unknown_topic" not in reference
    assert "document" not in reference
    assert reference["schema"] == Mechanism.model_json_schema()
    example = reference["mapping_example"]
    graph = Mechanism.model_validate(example["mechanism"])
    assert {component.feature for component in graph.components} == {
        root["id"] for root in example["roots"]
    }
    assert len({root["definition"] for root in example["roots"]}) == 1
    assert all(root["position"] == root["rotation"] == [0, 0, 0] for root in example["roots"])
    assert graph.joints[0].a != graph.joints[0].b
    assert graph.joints[0].pa == graph.joints[0].pb == (0, 0)
    example["mechanism"]["components"][0]["feature"] = "changed"
    assert (
        authoring_schema("construction:mechanism")["mapping_example"]["mechanism"]["components"][0][
            "feature"
        ]
        == "fixed_root"
    )


def test_document_schema_describes_mechanism_assembly_binding() -> None:
    description = DesignDocument.model_json_schema()["$defs"]["Component"]["properties"]["feature"][
        "description"
    ]
    assert "assembly.roots" in description
    assert "part-local" in description
    assert "unique" in description


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_feature", "duplicate_component", "wrong_binding", "root_position", "root_rotation"],
)
def test_mechanism_authoring_errors_report_ids_and_still_refuse_invalid_documents(mutation):
    from pydantic import ValidationError

    example = authoring_schema("construction:mechanism")["mapping_example"]
    data = {
        "name": "Structural mapping validation",
        "assembly": {
            "parts": [
                {
                    "id": "shared_part",
                    "name": "Shared",
                    "features": [
                        {
                            "id": "local_output",
                            "name": "Local",
                            "kind": "box",
                            "params": {"x": 1, "y": 1, "z": 1},
                        }
                    ],
                    "output": "local_output",
                }
            ],
            "roots": example["roots"],
        },
        "mechanism": example["mechanism"],
    }
    assert DesignDocument.model_validate(data).mechanism
    components = data["mechanism"]["components"]
    if mutation == "duplicate_feature":
        components[1]["feature"] = "fixed_root"
        expected = (
            "Components need unique IDs and unique output features",
            "duplicate feature IDs: ['fixed_root']",
        )
    elif mutation == "duplicate_component":
        components[1]["id"] = "fixed"
        expected = (
            "Components need unique IDs and unique output features",
            "duplicate component IDs: ['fixed']",
        )
    elif mutation == "wrong_binding":
        components[1]["feature"] = "local_output"
        expected = (
            "Every mechanism output needs exactly one named component",
            "missing output/root IDs: ['moving_root']",
            "unexpected component feature IDs: ['local_output']",
        )
    else:
        field = "position" if mutation == "root_position" else "rotation"
        data["assembly"]["roots"][1][field] = [0, 0, 1]
        expected = (
            "Mechanism controls assembly root placements; keep root transforms zero",
            "nonzero root IDs: ['moving_root']",
        )
    with pytest.raises(ValidationError) as error:
        DesignDocument.model_validate(data)
    assert all(detail in str(error.value) for detail in expected)
