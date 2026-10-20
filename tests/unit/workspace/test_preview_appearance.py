from copy import deepcopy

import pytest

from forge_mind.workspace.preview_appearance import ROLE_COLORS, preview_appearance


def fixture():
    document = {
        "assembly": {
            "parts": [
                {
                    "id": "holder",
                    "name": "Holder",
                    "production": {"role": "make", "appearance": {"color": "#D68B3C"}},
                },
                {"id": "device", "name": "Customer object", "production": {"role": "reference"}},
            ]
        }
    }
    assembly = {
        "occurrence_count": 3,
        "occurrences": [
            {"path": "root/nested/holder_a", "definition": "holder", "body_index": 2},
            {"path": "root/device", "definition": "device", "body_index": 0},
            {"path": "root/nested/holder_b", "definition": "holder", "body_index": 1},
        ],
    }
    return document, assembly


def test_nested_shared_definitions_follow_verified_body_indices_not_row_order():
    document, assembly = fixture()
    before = deepcopy((document, assembly))
    result = preview_appearance(document, assembly, 3)
    assert result.body_colors == (ROLE_COLORS["reference"], "#D68B3C", "#D68B3C")
    assert result.legend == (
        ("#D68B3C", "Holder\nmake, qty 2"),
        (ROLE_COLORS["reference"], "Customer object\nreference only, qty 1"),
    )
    assembly["occurrences"].reverse()
    assert preview_appearance(document, assembly, 3) == result
    assembly["occurrences"].reverse()
    assert (document, assembly) == before


def test_same_declared_colors_do_not_merge_different_roles():
    document, assembly = fixture()
    document["assembly"]["parts"][1]["production"]["appearance"] = {"color": "#D68B3C"}
    result = preview_appearance(document, assembly, 3)
    assert set(result.body_colors) == {"#D68B3C"}
    assert len(result.legend) == 2
    assert "make" in result.legend[0][1] and "reference only" in result.legend[1][1]


@pytest.mark.parametrize("role", ["make", "buy", "reference", "unspecified"])
def test_missing_color_uses_explicit_role_default(role):
    document, assembly = fixture()
    document["assembly"]["parts"][0]["production"] = {"role": role}
    assert preview_appearance(document, assembly, 3).body_colors[2] == ROLE_COLORS[role]


@pytest.mark.parametrize(
    "change",
    [
        lambda d, a: a["occurrences"].pop(),
        lambda d, a: a["occurrences"][0].update(body_index=True),
        lambda d, a: a["occurrences"][0].update(body_index=0),
        lambda d, a: a["occurrences"][0].update(definition="unknown"),
        lambda d, a: a["occurrences"][0].update(path="root/device"),
        lambda d, a: d["assembly"]["parts"][0]["production"].update(role="future"),
        lambda d, a: d["assembly"]["parts"][0]["production"].update(appearance={"color": ""}),
        lambda d, a: d["assembly"]["parts"][0]["production"].update(appearance="red"),
        lambda d, a: a.update(occurrence_count=4),
    ],
)
def test_malformed_mapping_or_declarations_are_neutral_without_role_labels(change):
    document, assembly = fixture()
    change(document, assembly)
    result = preview_appearance(document, assembly, 3)
    assert result.body_colors == () and result.legend == ()
    assert result.diagnostic.startswith("Part roles unavailable:")


def test_missing_identity_is_explicit_and_labels_are_bounded_without_omitting_bodies():
    assert preview_appearance(None, None, 2).body_colors == ()
    parts = [
        {"id": f"part{i}", "name": "very long name " * 20, "production": {"role": "make"}}
        for i in range(11)
    ]
    rows = [{"path": p["id"], "definition": p["id"], "body_index": i} for i, p in enumerate(parts)]
    result = preview_appearance({"assembly": {"parts": parts}}, {"occurrences": rows}, 11)
    assert len(result.body_colors) == 11
    assert len(result.legend) == 8 and result.omitted_labels == 3
    assert all(len(label.splitlines()[0]) <= 24 for _, label in result.legend)
