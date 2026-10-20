from __future__ import annotations

import json
from copy import deepcopy

import pytest

from forge_mind.workspace.engineering_report_view import (
    IDENTITY_FIELDS,
    SECTIONS,
    engineering_report_overview,
    engineering_report_section,
    validate_report_view,
)

REPORT_ID = "a" * 32
REVISION_ID = "b" * 32


def report() -> dict:
    return {
        "id": REPORT_ID,
        "revision_id": REVISION_ID,
        "input_version": 3,
        "input_hash": "c" * 64,
        "geometry_hash": "d" * 64,
        "engine_fingerprint": "e" * 64,
        "frame": "cad_z_up_mm",
        "at": 1791357563.6571403,
        "reviews": [
            {
                "category": "assembly",
                "status": "fail",
                "method": "native",
                "extra": {"retained": True},
            },
            {"category": "motion", "status": "unverified", "method": "discrete_only"},
            {"category": "future", "status": "future_status", "method": "future_method"},
        ],
        "open_items": ['Thread and hand access remain unknown.\n"quoted" \\ µ'],
        "geometry": {
            "findings": [{"status": "fail", "exact_int": 900719925474099312345, "float_one": 1.0}],
            "errors": ["native error"],
            "future": {"witness": {"points": [1.0, -0.0, 3], "text": 'line\r\n"quote"\\µ'}},
        },
        "assembly_motion": {
            "status": "unverified",
            "collisions": [
                {
                    "angle": 12.75,
                    "a": "one",
                    "b": "two",
                    "overlap_mm3": 0.0000000001,
                    "unknown_witness_field": {"native": [1, 2, 3]},
                }
            ],
            "errors": ["pose unavailable"],
            "coverage": "discrete_only",
            "requested_angles": [0.0, 90.0, 180.0],
        },
        "mass_kg": None,
        "future_section": {"nested_failure": {"status": "fail", "error": "future error"}},
    }


def test_overview_preserves_identity_reviews_unknown_statuses_and_complete_open_items():
    original = report()
    result = engineering_report_overview(original, REPORT_ID).model_dump(mode="json")
    assert {key: result[key] for key in IDENTITY_FIELDS} == {
        key: original[key] for key in IDENTITY_FIELDS
    }
    assert result["reviews"] == original["reviews"]
    assert result["open_items"] == original["open_items"]
    assert result["nested_failure_inventory_complete"] is False
    assert "geometry" not in result and "assembly_motion" not in result
    assert set(result["omitted_sections"]) == {
        "geometry",
        "assembly_motion",
        "mass_kg",
        "future_section",
    }
    assert set(result["unavailable_sections"]) == set(SECTIONS) - original.keys()


def test_directory_counts_direct_entries_and_exposes_unknown_full_reader_fallback():
    original = report()
    result = engineering_report_overview(original, REPORT_ID)
    directory = {entry.name: entry for entry in result.sections}
    assert set(directory) == original.keys() - set(IDENTITY_FIELDS)
    assert directory["geometry"].direct_count == 3
    assert directory["geometry"].count_scope == "direct_object_keys"
    assert directory["reviews"].direct_count == 3
    assert directory["reviews"].count_scope == "direct_array_items"
    assert directory["reviews"].included_in_overview
    assert directory["mass_kg"].direct_count is None
    assert directory["mass_kg"].json_type == "null"
    assert directory["future_section"].full_reader_required
    assert not directory["future_section"].available_via_section
    assert result.unknown_top_level_fields == ["future_section"]
    for name, entry in directory.items():
        assert entry.value_json_utf8_bytes == len(
            json.dumps(
                original[name], ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        )


@pytest.mark.parametrize(
    "name", ["geometry", "assembly_motion", "mass_kg", "reviews", "open_items"]
)
def test_exact_section_preserves_all_witnesses_errors_coverage_and_unknown_fields(name):
    original = report()
    result = engineering_report_section(original, REPORT_ID, name)
    assert result.value == original[name]
    assert result.section_complete is True
    assert set(result.omitted_sections) == original.keys() - set(IDENTITY_FIELDS) - {name}
    wire = json.loads(result.model_dump_json())
    assert wire["value"] == original[name]


def test_exact_section_preserves_large_integer_float_type_negative_zero_and_escaping():
    result = engineering_report_section(report(), REPORT_ID, "geometry")
    assert type(result.value["findings"][0]["exact_int"]) is int
    assert result.value["findings"][0]["exact_int"] == 900719925474099312345
    assert type(result.value["findings"][0]["float_one"]) is float
    wire = result.model_dump_json()
    assert '"float_one":1.0' in wire
    assert "-0.0" in wire
    assert (
        json.loads(wire)["value"]["future"]["witness"]["text"]
        == report()["geometry"]["future"]["witness"]["text"]
    )


def test_views_do_not_mutate_source_and_nested_output_copies_are_independent():
    original = report()
    saved = deepcopy(original)
    overview = engineering_report_overview(original, REPORT_ID)
    detail = engineering_report_section(original, REPORT_ID, "assembly_motion")
    overview.reviews[0]["extra"]["retained"] = False
    overview.open_items.append("changed")
    detail.value["collisions"][0]["unknown_witness_field"]["native"][0] = 0
    assert original == saved


def test_large_sections_reviews_and_open_items_are_not_clipped():
    original = report()
    original["assembly_motion"]["collisions"] *= 3000
    original["reviews"] *= 1000
    original["open_items"] *= 2000
    overview = engineering_report_overview(original, REPORT_ID)
    detail = engineering_report_section(original, REPORT_ID, "assembly_motion")
    assert overview.reviews == original["reviews"]
    assert overview.open_items == original["open_items"]
    assert detail.value == original["assembly_motion"]
    assert len(detail.value["collisions"]) == 3000


@pytest.mark.parametrize(
    "view,section",
    [
        ("full", "geometry"),
        ("overview", "geometry"),
        ("section", None),
        ("section", "future_section"),
        ("section", "../report.json"),
        ("future", None),
    ],
)
def test_invalid_view_and_section_combinations_refuse(view, section):
    with pytest.raises(ValueError):
        validate_report_view(view, section)


def test_valid_views_and_missing_section_and_wrong_report_identity():
    for view, section in [("full", None), ("overview", None), ("section", "geometry")]:
        validate_report_view(view, section)
    with pytest.raises(ValueError, match="absent"):
        engineering_report_section(report(), REPORT_ID, "production")
    for function in (
        engineering_report_overview,
        lambda r, rid: engineering_report_section(r, rid, "geometry"),
    ):
        with pytest.raises(ValueError, match="requested report ID"):
            function(report(), "f" * 32)


@pytest.mark.parametrize(
    "key,value",
    [
        ("id", "not-an-id"),
        ("revision_id", None),
        ("geometry_hash", "wrong"),
        ("input_hash", 3),
        ("engine_fingerprint", "e" * 63),
        ("input_version", True),
        ("input_version", -1),
        ("input_version", 1.0),
        ("frame", ""),
        ("at", True),
        ("at", "1791357563"),
        ("at", 10**500),
        ("at", float("inf")),
        ("reviews", [{"category": "missing"}]),
        ("open_items", [3]),
    ],
)
def test_malformed_metadata_refuses_without_default_values(key, value):
    original = report()
    original[key] = value
    with pytest.raises(ValueError):
        engineering_report_overview(original, REPORT_ID)
    with pytest.raises(ValueError):
        engineering_report_section(original, REPORT_ID, "geometry")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), {1, 2}])
def test_non_json_or_nonfinite_nested_values_refuse(value):
    original = report()
    original["geometry"]["future"]["invalid"] = value
    with pytest.raises(ValueError):
        engineering_report_overview(original, REPORT_ID)
    with pytest.raises(ValueError):
        engineering_report_section(original, REPORT_ID, "geometry")
