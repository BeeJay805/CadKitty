"""Scoped-edit behavior without geometry compilation."""

from copy import deepcopy

import pytest

from forge_mind.workspace.authoring import AssemblyEdit, apply_assembly_edit
from forge_mind.workspace.document import DesignDocument, VariableRelation


def relation(name: str, a: str | float, b: str | float, operation: str = "add") -> dict:
    return {"name": name, "operation": operation, "a": a, "b": b}


def baseline() -> DesignDocument:
    return DesignDocument.model_validate(
        {
            "name": "Retained mounting part",
            "variables": {"base_width": 52, "left_hole_x": -18, "seed": 1},
            "relations": [
                relation("mount_edge_offset", "base_width", -36),
                relation("retained_double", "mount_edge_offset", 2, "multiply"),
                relation("unrelated", "base_width", 3),
            ],
            "delivery_guidance": {
                "print": "Print the made part only; declared settings are untested.",
                "installation": "Preserve the mounting holes.",
                "use_and_removal": "Physical removal has not been tested.",
                "unresolved": "Strength and physical fit are unknown.",
            },
            "assembly": {
                "parts": [
                    {
                        "id": "plate",
                        "name": "Mounting plate",
                        "features": [
                            {
                                "id": "body",
                                "name": "Plate",
                                "kind": "box",
                                "params": {"x": 52, "y": 20, "z": 4},
                            }
                        ],
                        "output": "body",
                        "production": {
                            "role": "make",
                            "deliver_stl": True,
                            "material": {"name": "PLA"},
                            "process": "additive",
                            "appearance": {"color": "#C77025"},
                        },
                        "source": "Retained declaration; not physical evidence",
                    }
                ],
                "roots": [
                    {
                        "id": "mounted",
                        "definition": "plate",
                        "position": [12, -3, 4],
                        "rotation": [0, 0, 90],
                    }
                ],
            },
        }
    )


def test_existing_relation_can_use_new_dependency_and_retains_metadata() -> None:
    document = baseline()
    original = document.model_dump()
    patch = AssemblyEdit.model_validate(
        {
            "relations": [
                relation("half_base_width", "base_width", 2, "divide"),
                relation("mount_edge_offset", "half_base_width", "left_hole_x"),
            ]
        }
    )
    patch_before = patch.model_dump()
    updated = apply_assembly_edit(document, patch)

    assert updated.values()["half_base_width"] == 26
    assert updated.values()["mount_edge_offset"] == 8
    assert updated.values()["retained_double"] == 16
    assert updated.values()["unrelated"] == 55
    expected = deepcopy(original)
    expected["relations"] = [
        original["relations"][2],
        VariableRelation.model_validate(patch_before["relations"][0]).model_dump(),
        VariableRelation.model_validate(patch_before["relations"][1]).model_dump(),
        original["relations"][1],
    ]
    assert updated.model_dump() == expected
    assert document.model_dump() == original
    assert patch.model_dump() == patch_before


def test_new_chain_updates_existing_relation_without_losing_untouched_dependent() -> None:
    document = baseline()
    before = document.model_dump_json()
    patch = AssemblyEdit.model_validate(
        {
            "relations": [
                relation("mount_edge_offset", "chain_tail", "left_hole_x"),
                relation("chain_tail", "chain_head", 2),
                relation("chain_head", "base_width", 2, "divide"),
            ]
        }
    )
    patch_before = patch.model_dump_json()
    updated = apply_assembly_edit(document, patch)

    assert [item.name for item in updated.relations] == [
        "unrelated",
        "chain_head",
        "chain_tail",
        "mount_edge_offset",
        "retained_double",
    ]
    assert updated.values()["chain_head"] == 26
    assert updated.values()["chain_tail"] == 28
    assert updated.values()["mount_edge_offset"] == 10
    assert updated.values()["retained_double"] == 20
    assert updated.assembly == document.assembly
    assert updated.delivery_guidance == document.delivery_guidance
    assert updated.variables == document.variables
    assert document.model_dump_json() == before
    assert patch.model_dump_json() == patch_before


def test_already_valid_order_and_empty_patch_keep_exact_serialization() -> None:
    document = baseline()
    assert (
        apply_assembly_edit(document, AssemblyEdit()).model_dump_json()
        == document.model_dump_json()
    )
    patch = AssemblyEdit.model_validate(
        {"relations": [relation("mount_edge_offset", "base_width", -35)]}
    )
    updated = apply_assembly_edit(document, patch)
    expected = document.model_dump()
    expected["relations"][0] = patch.relations[0].model_dump()
    assert updated.model_dump() == expected
    assert [item.name for item in updated.relations] == [item.name for item in document.relations]


@pytest.mark.parametrize(
    "rows",
    [
        [relation("new", "new", 1)],
        [relation("new_a", "new_b", 1), relation("new_b", "new_a", 1)],
        [relation("new", "absent_value", 1)],
        [relation("base_width", "seed", 1)],
        [relation("new", "seed", 1), relation("new", "seed", 2)],
        [relation("new", "seed", 0, "divide")],
        [relation("new", 1e308, 1e308, "multiply")],
    ],
    ids=["self-cycle", "cycle", "missing", "collision", "duplicate", "zero", "overflow"],
)
def test_invalid_relation_edits_refuse_without_changing_inputs(rows: list[dict]) -> None:
    document = baseline()
    before = document.model_dump_json()
    patch = AssemblyEdit.model_validate({"relations": rows})
    patch_before = patch.model_dump_json()
    with pytest.raises(ValueError):
        apply_assembly_edit(document, patch)
    assert document.model_dump_json() == before
    assert patch.model_dump_json() == patch_before


def test_variable_collision_and_nonfinite_variable_edits_still_refuse() -> None:
    document = baseline()
    before = document.model_dump_json()
    for variables in ({"mount_edge_offset": 1}, {"base_width": float("nan")}):
        with pytest.raises(ValueError):
            apply_assembly_edit(document, AssemblyEdit(variables=variables))
        assert document.model_dump_json() == before


@pytest.mark.parametrize("value", [float("inf"), float("nan")], ids=["infinity", "nan"])
def test_nonfinite_relation_literal_refuses_at_existing_schema_boundary(value: float) -> None:
    with pytest.raises(ValueError):
        AssemblyEdit.model_validate({"relations": [relation("new", value, 1)]})


def test_reverse_chain_above_recursion_limit_has_no_new_capacity_limit() -> None:
    document = baseline()
    before = document.model_dump_json()
    count = 1400
    rows = [relation(f"r{i}", "seed" if i == 0 else f"r{i - 1}", 1) for i in range(count)]
    patch = AssemblyEdit.model_validate({"relations": list(reversed(rows))})
    updated = apply_assembly_edit(document, patch)
    assert len(updated.relations) == len(document.relations) + count
    assert [item.name for item in updated.relations[:3]] == [
        item.name for item in document.relations
    ]
    assert [item.name for item in updated.relations[3:]] == [f"r{i}" for i in range(count)]
    assert updated.values()["r1399"] == 1401
    assert document.model_dump_json() == before


def test_full_document_admission_still_requires_ordered_relations() -> None:
    raw = baseline().model_dump()
    raw["relations"] = [
        relation("dependent", "prerequisite", 1),
        relation("prerequisite", "seed", 1),
    ]
    with pytest.raises(ValueError):
        DesignDocument.model_validate(raw)


def test_repeated_operand_dependency_resolves_once_on_new_assembly() -> None:
    original = baseline()
    patch = AssemblyEdit.model_validate(
        {
            "parts": original.assembly.model_dump()["parts"],
            "roots": original.assembly.model_dump()["roots"],
            "variables": {"width": 12},
            "relations": [
                relation("double_half", "half", "half"),
                relation("half", "width", 2, "divide"),
            ],
        }
    )
    updated = apply_assembly_edit(None, patch)
    assert updated.values() == {"width": 12, "half": 6, "double_half": 12}


def test_missing_dependency_error_identifies_relation_and_operand() -> None:
    patch = AssemblyEdit.model_validate(
        {"relations": [relation("mount_edge_offset", "missing_width", 2)]}
    )
    with pytest.raises(
        ValueError, match="Relation mount_edge_offset references missing value missing_width"
    ):
        apply_assembly_edit(baseline(), patch)


def test_cycle_error_reports_blocked_relations_without_inventing_values() -> None:
    patch = AssemblyEdit.model_validate(
        {
            "relations": [
                relation("a", "b", 1),
                relation("b", "a", 1),
                relation("downstream", "a", 2),
            ]
        }
    )
    with pytest.raises(ValueError, match="Cyclic relation dependencies block: a, b, downstream"):
        apply_assembly_edit(baseline(), patch)
