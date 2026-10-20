from copy import deepcopy

import pytest

from forge_mind.workspace.preview_selection import select_preview_bodies


def fixture():
    document = {
        "assembly": {
            "parts": [
                {"id": "fixture", "production": {"role": "make"}},
                {"id": "context", "production": {"role": "reference"}},
                {"id": "screw", "production": {"role": "buy"}},
            ]
        }
    }
    evidence = {
        "occurrence_count": 4,
        "status": "fail",
        "occurrences": [
            {"path": "nested/context_a", "definition": "context", "body_index": 3},
            {"path": "nested/fixture", "definition": "fixture", "body_index": 1},
            {"path": "moved/context_b", "definition": "context", "body_index": 0},
            {"path": "screw", "definition": "screw", "body_index": 2},
        ],
    }
    return document, evidence


def test_repeated_nested_moved_definitions_select_by_verified_body_identity():
    document, evidence = fixture()
    before = deepcopy((document, evidence))
    selected = select_preview_bodies(document, evidence, 4)
    assert selected.included_body_indices == (1, 2)
    assert selected.omitted_occurrence_ids == ("moved/context_b", "nested/context_a")
    assert selected.omitted_count == 2
    assert selected.status == "reference_excluded"
    assert (document, evidence) == before
    evidence["occurrences"].reverse()
    assert select_preview_bodies(document, evidence, 4) == selected


@pytest.mark.parametrize(
    "role",
    [
        "make",
        "buy",
        "unspecified",
        "REFERENCE",
        " reference",
        "future_role",
        None,
        True,
    ],
)
def test_only_exact_reference_role_omits_bodies(role):
    document, evidence = fixture()
    document["assembly"]["parts"][1]["production"]["role"] = role
    selected = select_preview_bodies(document, evidence, 4)
    assert selected.included_body_indices == (0, 1, 2, 3)
    assert selected.status == "no_reference"


@pytest.mark.parametrize("production", [None, {}, "reference", ["reference"]])
def test_absent_or_unrecognized_role_data_stays_visible(production):
    document, evidence = fixture()
    document["assembly"]["parts"][1]["production"] = production
    assert select_preview_bodies(document, evidence, 4).included_body_indices == (
        0,
        1,
        2,
        3,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("body_index", 1),
        ("body_index", True),
        ("body_index", -1),
        ("body_index", 4),
        ("body_index", 3.0),
        ("body_index", "3"),
        ("body_index", None),
        ("path", "screw"),
        ("path", ""),
        ("path", None),
        ("definition", "missing"),
        ("definition", None),
    ],
)
def test_invalid_mapping_keeps_every_body(field, value):
    document, evidence = fixture()
    evidence["occurrences"][0][field] = value
    selected = select_preview_bodies(document, evidence, 4)
    assert selected.status == "full_context_fallback"
    assert selected.included_body_indices == (0, 1, 2, 3)
    assert selected.omitted_count == 0


@pytest.mark.parametrize(
    "case",
    [
        "missing_row",
        "extra_row",
        "count",
        "bool_count",
        "duplicate_definition",
        "bad_row",
        "missing_evidence",
    ],
)
def test_structural_conflicts_fall_back(case):
    document, evidence = fixture()
    if case == "missing_row":
        evidence["occurrences"].pop()
    elif case == "extra_row":
        evidence["occurrences"].append(evidence["occurrences"][0])
    elif case == "count":
        evidence["occurrence_count"] = 3
    elif case == "bool_count":
        evidence["occurrence_count"] = True
    elif case == "duplicate_definition":
        document["assembly"]["parts"].append({"id": "context", "production": {"role": "make"}})
    elif case == "bad_row":
        evidence["occurrences"][0] = None
    else:
        evidence = None
    selected = select_preview_bodies(document, evidence, 4)
    assert selected.included_body_indices == (0, 1, 2, 3)
    assert selected.status == "full_context_fallback"


@pytest.mark.parametrize(
    "document", [None, {}, {"production": {"role": "reference"}}, {"assembly": None}]
)
def test_legacy_and_nonassembly_never_guess_roles(document):
    selected = select_preview_bodies(document, None, 3)
    assert selected.status == "full_context_fallback"
    assert selected.included_body_indices == (0, 1, 2)


def test_all_reference_has_explicit_empty_inspection_status():
    document, evidence = fixture()
    for part in document["assembly"]["parts"]:
        part["production"]["role"] = "reference"
    selected = select_preview_bodies(document, evidence, 4)
    assert selected.status == "all_reference"
    assert selected.included_body_indices == ()
    assert selected.omitted_count == 4


def test_misleading_name_bbox_and_reference_unused_definition_do_not_select():
    document, evidence = fixture()
    document["assembly"]["parts"][1]["production"]["role"] = "make"
    document["assembly"]["parts"].append({"id": "unused", "production": {"role": "reference"}})
    for row in evidence["occurrences"]:
        row.update(name="reference", bbox_min_mm=[0, 0, 0], bbox_max_mm=[999, 999, 999])
    assert select_preview_bodies(document, evidence, 4).status == "no_reference"


@pytest.mark.parametrize("body_count", [-1, True, 1.0, "1"])
def test_invalid_body_count_is_a_caller_error(body_count):
    with pytest.raises(ValueError, match="nonnegative integer"):
        select_preview_bodies(None, None, body_count)
