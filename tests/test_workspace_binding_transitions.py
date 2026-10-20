"""Explicit dependency changes retain native part definitions and fail atomically."""

import pytest

from forge_mind.workspace.authoring import AssemblyEdit, apply_assembly_edit, authoring_schema
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.inspection import author_revision_context


def relation(name: str, a: str | float, b: str | float, operation: str) -> dict:
    return {"name": name, "a": a, "b": b, "operation": operation}


def sleeve_document() -> DesignDocument:
    production = {
        "role": "make",
        "material": {"name": "PLA"},
        "process": "additive",
        "deliver_stl": True,
    }
    return DesignDocument.model_validate(
        {
            "name": "Editable sleeve and plug",
            "variables": {
                "sleeve_outer_diameter": 30,
                "sleeve_length": 24,
                "plug_diameter": 20,
                "plug_length": 18,
                "radial_clearance": 0.2,
            },
            "relations": [
                relation("diametral_clearance", "radial_clearance", 2, "multiply"),
                relation("sleeve_bore_diameter", "plug_diameter", "diametral_clearance", "add"),
                relation("sleeve_bore_radius", "sleeve_bore_diameter", 2, "divide"),
                relation("sleeve_outer_radius", "sleeve_outer_diameter", 2, "divide"),
                relation("wall_thickness", "sleeve_outer_radius", "sleeve_bore_radius", "subtract"),
                relation("plug_radius", "plug_diameter", 2, "divide"),
            ],
            "assembly": {
                "parts": [
                    {
                        "id": "sleeve",
                        "name": "Sleeve",
                        "production": production,
                        "features": [
                            {
                                "id": "outer",
                                "name": "Outer cylinder",
                                "kind": "cylinder",
                                "params": {"radius": 15, "height": 24},
                                "bindings": {
                                    "radius": "sleeve_outer_radius",
                                    "height": "sleeve_length",
                                },
                            },
                            {
                                "id": "bore",
                                "name": "Through bore",
                                "kind": "cylinder",
                                "params": {"radius": 10.2, "height": 24},
                                "bindings": {
                                    "radius": "sleeve_bore_radius",
                                    "height": "sleeve_length",
                                },
                            },
                            {
                                "id": "body",
                                "name": "Hollow sleeve",
                                "kind": "cut",
                                "inputs": ["outer", "bore"],
                            },
                        ],
                        "output": "body",
                    },
                    {
                        "id": "plug",
                        "name": "Plug",
                        "production": production,
                        "features": [
                            {
                                "id": "body",
                                "name": "Plug cylinder",
                                "kind": "cylinder",
                                "params": {"radius": 10, "height": 18},
                                "bindings": {"radius": "plug_radius", "height": "plug_length"},
                            }
                        ],
                        "output": "body",
                    },
                ],
                "roots": [
                    {"id": "sleeve_occ", "definition": "sleeve"},
                    {"id": "plug_occ", "definition": "plug"},
                ],
            },
        }
    )


def bore_edit() -> AssemblyEdit:
    return AssemblyEdit.model_validate(
        {
            "promote_relations": {"sleeve_bore_diameter": 20.8},
            "derive_variables": [relation("radial_clearance", "diametral_clearance", 2, "divide")],
            "relations": [
                relation("diametral_clearance", "sleeve_bore_diameter", "plug_diameter", "subtract")
            ],
        }
    )


def test_bore_and_clearance_transition_recomputes_bound_sizes_and_preserves_parts() -> None:
    original = sleeve_document()
    before = original.model_dump_json()
    patch = bore_edit()
    patch_before = patch.model_dump_json()
    edited = apply_assembly_edit(original, patch)

    assert edited.variables["sleeve_bore_diameter"] == 20.8
    assert "radial_clearance" not in edited.variables
    values = edited.values()
    assert values["radial_clearance"] == pytest.approx(0.4)
    assert values["sleeve_bore_radius"] == pytest.approx(10.4)
    assert values["wall_thickness"] == pytest.approx(4.6)
    assert values["plug_radius"] == 10
    assert edited.assembly == original.assembly
    assert original.model_dump_json() == before
    assert patch.model_dump_json() == patch_before


def test_binding_category_roundtrip_retains_parts_and_original_values() -> None:
    original = sleeve_document()
    edited = apply_assembly_edit(original, bore_edit())
    restored = apply_assembly_edit(
        edited,
        AssemblyEdit.model_validate(
            {
                "promote_relations": {"radial_clearance": 0.2},
                "derive_variables": [
                    relation("sleeve_bore_diameter", "plug_diameter", "diametral_clearance", "add")
                ],
                "relations": [relation("diametral_clearance", "radial_clearance", 2, "multiply")],
            }
        ),
    )
    assert restored.values() == original.values()
    assert restored.variables == original.variables
    assert restored.assembly == original.assembly
    assert {row.name: row for row in restored.relations} == {
        row.name: row for row in original.relations
    }


@pytest.mark.parametrize(
    "raw",
    [
        {"promote_relations": {"unknown": 1}},
        {"promote_relations": {"plug_diameter": 21}},
        {"derive_variables": [relation("unknown", "plug_diameter", 2, "divide")]},
        {"derive_variables": [relation("sleeve_bore_diameter", "plug_diameter", 1, "add")]},
        {
            "derive_variables": [
                relation("radial_clearance", "plug_diameter", 2, "divide"),
                relation("radial_clearance", "plug_diameter", 3, "divide"),
            ]
        },
        {
            "promote_relations": {"sleeve_bore_diameter": 20.8},
            "derive_variables": [relation("sleeve_bore_diameter", "plug_diameter", 1, "add")],
        },
        {
            "promote_relations": {"sleeve_bore_diameter": 20.8},
            "variables": {"sleeve_bore_diameter": 21},
        },
        {
            "promote_relations": {"sleeve_bore_diameter": 20.8},
            "relations": [relation("sleeve_bore_diameter", "plug_diameter", 1, "add")],
        },
        {
            "derive_variables": [relation("radial_clearance", "plug_diameter", 2, "divide")],
            "variables": {"radial_clearance": 0.4},
        },
        {
            "derive_variables": [relation("radial_clearance", "plug_diameter", 2, "divide")],
            "relations": [relation("radial_clearance", "plug_diameter", 3, "divide")],
        },
        {"derive_variables": [relation("radial_clearance", "diametral_clearance", 2, "divide")]},
        {"derive_variables": [relation("radial_clearance", "missing", 2, "divide")]},
        {"derive_variables": [relation("radial_clearance", "plug_diameter", 0, "divide")]},
        {"derive_variables": [relation("radial_clearance", "radial_clearance", 1, "add")]},
    ],
    ids=[
        "unknown-promotion",
        "wrong-promotion-category",
        "unknown-derivation",
        "wrong-derivation-category",
        "duplicate-derivation",
        "transition-overlap",
        "promotion-variable-conflict",
        "promotion-relation-conflict",
        "derivation-variable-conflict",
        "derivation-relation-conflict",
        "dependency-cycle",
        "missing-dependency",
        "division-by-zero",
        "self-cycle",
    ],
)
def test_invalid_transitions_refuse_without_mutating_source_or_edit(raw: dict) -> None:
    original = sleeve_document()
    before = original.model_dump_json()
    patch = AssemblyEdit.model_validate(raw)
    patch_before = patch.model_dump_json()
    with pytest.raises(ValueError):
        apply_assembly_edit(original, patch)
    assert original.model_dump_json() == before
    assert patch.model_dump_json() == patch_before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_promotion_refuses_even_if_edit_was_mutated(value: float) -> None:
    original = sleeve_document()
    before = original.model_dump_json()
    with pytest.raises(ValueError):
        AssemblyEdit.model_validate({"promote_relations": {"sleeve_bore_diameter": value}})
    patch = AssemblyEdit()
    patch.promote_relations["sleeve_bore_diameter"] = value
    with pytest.raises(ValueError, match="finite"):
        apply_assembly_edit(original, patch)
    assert original.model_dump_json() == before


def test_transitions_cannot_target_a_binding_created_by_the_same_edit() -> None:
    original = sleeve_document()
    before = original.model_dump_json()
    patch = AssemblyEdit.model_validate(
        {
            "variables": {"new_input": 1},
            "derive_variables": [relation("new_input", "plug_diameter", 2, "divide")],
        }
    )
    with pytest.raises(ValueError):
        apply_assembly_edit(original, patch)
    with pytest.raises(ValueError):
        apply_assembly_edit(None, AssemblyEdit(promote_relations={"unknown": 1}))
    assert original.model_dump_json() == before


def test_ordinary_collision_still_requires_an_explicit_transition() -> None:
    original = sleeve_document()
    before = original.model_dump_json()
    with pytest.raises(ValueError, match="cannot overwrite"):
        apply_assembly_edit(original, AssemblyEdit(variables={"sleeve_bore_diameter": 20.8}))
    assert original.model_dump_json() == before
    assert apply_assembly_edit(original, AssemblyEdit()).model_dump_json() == before


def test_binding_transition_schema_and_context_are_bounded_and_queryable() -> None:
    schema = authoring_schema("assembly_edit")
    assert "promote_relations" in schema["schema"]["properties"]
    assert "derive_variables" in schema["schema"]["properties"]
    assert "atomic" in schema["conventions"]
    original = sleeve_document()
    context = author_revision_context({"id": "baseline", "document": original.model_dump()})
    editing = context["binding_editing"]
    assert editing["input_count"] == 5
    assert editing["derived_count"] == 6
    bore = next(row for row in editing["derived_preview"] if row["name"] == "sleeve_bore_diameter")
    assert bore["depends_on"] == ["diametral_clearance", "plug_diameter"]
    assert "read_design" in editing["usage"]
    raw = original.model_dump()
    raw["relations"] = [
        relation(f"derived_{index}", "plug_diameter", 2, "divide") for index in range(30)
    ]
    context = author_revision_context({"id": "large", "document": raw})
    assert context["binding_editing"]["derived_count"] == 30
    assert len(context["binding_editing"]["derived_preview"]) == 12
