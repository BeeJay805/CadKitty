"""Shared dimensional capacity without weakening CAD or semantic limits."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from forge_mind.workspace.authoring import AssemblyEdit, apply_assembly_edit
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.inspection import reachable
from forge_mind.workspace.project_access import design_relationships


def controls(count: int) -> dict[str, Any]:
    return {
        "variables": {f"v{i}": 1.0 for i in range(count)},
        "relations": [
            {"name": f"r{i}", "operation": "add", "a": f"r{i - 1}" if i else 0, "b": f"v{i}"}
            for i in range(count)
        ],
    }


@pytest.mark.parametrize("count", [121, 256, 513])
def test_document_keeps_large_shared_controls_and_transitive_dependencies(count: int) -> None:
    document = DesignDocument.model_validate(controls(count))
    assert document.values()[f"r{count - 1}"] == count
    graph = design_relationships(document)
    assert len(graph["edges"]) == count * 2 - 1
    assert (
        len(
            [
                n
                for n in reachable(graph, [f"variable/r{count - 1}"], reverse=True)
                if n.startswith("variable/v")
            ]
        )
        == count
    )


def test_assembly_edit_accepts_controls_beyond_former_merged_ceiling() -> None:
    edit = AssemblyEdit.model_validate(
        {
            **controls(256),
            "parts": [
                {
                    "id": "plate",
                    "name": "Plate",
                    "features": [
                        {
                            "id": "body",
                            "name": "Body",
                            "kind": "box",
                            "params": {"y": 4, "z": 2},
                            "bindings": {"x": "r255"},
                        },
                    ],
                    "output": "body",
                }
            ],
            "roots": [{"id": "plate", "definition": "plate"}],
        }
    )
    document = apply_assembly_edit(None, edit)
    assert document.values()["r255"] == 256
    from forge_mind.workspace.assembly import compile_assembly

    shape, _ = compile_assembly(document, {})
    assert pytest.approx(256) == shape.bounding_box().size.X
    assert shape.volume == pytest.approx(256 * 4 * 2)
    updated = apply_assembly_edit(document, AssemblyEdit(variables={"new_control": 2.0}))
    assert len(updated.variables) == 257
    assert len(document.variables) == 256


@pytest.mark.parametrize("model", [DesignDocument, AssemblyEdit])
@pytest.mark.parametrize("field", ["variables", "relations"])
def test_models_accept_257_controls(model: Any, field: str) -> None:
    raw = controls(257)
    if model is DesignDocument and field == "relations":
        raw["variables"] = {"v0": 1.0}
        for relation in raw["relations"]:
            relation["b"] = "v0"
    else:
        raw = {field: raw[field]}
    assert len(getattr(model.model_validate(raw), field)) == 257


def test_bom_preserves_mixed_production_at_full_dimensional_capacity() -> None:
    from forge_mind.workspace.hardware import bill_of_materials

    document = DesignDocument.model_validate(
        {
            **controls(256),
            "assembly": {
                "parts": [
                    {
                        "id": identifier,
                        "name": identifier,
                        "features": [
                            {
                                "id": "body",
                                "name": "Body",
                                "kind": "box",
                                "params": {"y": 4, "z": 2},
                                "bindings": {"x": "r255"},
                            }
                        ],
                        "output": "body",
                        "production": {"role": role},
                    }
                    for identifier, role in (("bracket", "make"), ("purchased_envelope", "buy"))
                ],
                "roots": [
                    {"id": "mount", "definition": "bracket"},
                    {"id": "left", "definition": "purchased_envelope"},
                    {"id": "right", "definition": "purchased_envelope"},
                ],
            },
        }
    )
    before = document.model_dump()
    bom = bill_of_materials(document)
    assert [(r["part_id"], r["quantity"]) for r in bom["make"]] == [("bracket", 1)]
    assert [(r["part_id"], r["quantity"]) for r in bom["buy"]] == [("purchased_envelope", 2)]
    assert bom["lines"] == bom["buy"]
    assert document.model_dump() == before
    assert document.values()["r255"] == 256


@pytest.mark.parametrize("defect", ["nan", "infinite", "unknown", "overwrite", "zero", "bad_name"])
def test_larger_document_preserves_semantic_guards(defect: str) -> None:
    raw = controls(121)
    if defect == "nan":
        raw["variables"]["v0"] = float("nan")
    elif defect == "infinite":
        raw["variables"]["v0"] = float("inf")
    elif defect == "unknown":
        raw["relations"][0]["a"] = "unknown"
    elif defect == "overwrite":
        raw["relations"][0]["name"] = "v0"
    elif defect == "zero":
        raw["relations"][0].update(operation="divide", b=0)
    else:
        raw["variables"]["invalid/name"] = 1
    with pytest.raises(ValidationError):
        DesignDocument.model_validate(raw)


def test_mcp_edit_capacity_matches_document_and_retains_numeric_guards() -> None:
    pytest.importorskip("mcp")
    from pydantic import TypeAdapter

    from forge_mind.workspace.mcp_server import Variables

    adapter = TypeAdapter(Variables)
    assert len(adapter.validate_python(controls(256)["variables"])) == 256
    assert len(adapter.validate_python(controls(257)["variables"])) == 257
    for invalid in ({"width": float("nan")}, {"width": float("inf")}, {"bad/name": 1}, {}):
        with pytest.raises(ValidationError):
            adapter.validate_python(invalid)


def test_feature_uniqueness_and_large_nested_occurrence_expansion() -> None:
    from forge_mind.workspace.assembly import expand

    feature = {"id": "base", "name": "Base", "kind": "box", "params": {"x": 1, "y": 1, "z": 1}}
    with pytest.raises(ValidationError, match="unique"):
        DesignDocument.model_validate({"features": [feature] * 129})
    group = {
        "id": "group",
        "name": "Group",
        "children": [{"id": f"p{i}", "definition": "part"} for i in range(65)],
    }
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [{"id": "part", "name": "Part", "features": [feature], "output": "base"}],
                "subassemblies": [group],
                "roots": [{"id": "a", "definition": "group"}, {"id": "b", "definition": "group"}],
            }
        }
    )
    assert len(expand(document)) == 130
