"""Stage projections preserve intent, current evidence, exact reads, and input purity."""

import copy
import json
from typing import Any

import pytest

from forge_mind.workspace.native_stages import construction_context, startup_context


@pytest.fixture
def full_context() -> dict[str, Any]:
    return {
        "request": "Make an adjustable fixture from the supplied reference",
        "user_revision_history": ["Keep the retained pin", "Use an 18 mm bore"],
        "edit_instruction": "Use an 18 mm bore",
        "frozen_contract": {
            "requirements": [
                {"id": "bore", "metric": "hole_diameter", "value": 18, "part_id": "sleeve"},
                {"id": "holding", "metric": "unmeasured", "critical": True},
            ]
        },
        "engineering_inputs": {"version": 4, "material": {"name": "Unknown alloy"}},
        "owned_assets": [{"id": "reference", "sha256": "asset_hash", "kind": "photo"}],
        "source_claims": [{"id": "rated_load", "status": "unverified", "quote": "10 kg"}],
        "source_artifacts": [{"id": "manual", "sha256": "manual_hash", "text": "Exact text"}],
        "edit_basis": {"revision_id": "a" * 32, "plan_version": 2, "input_version": 4},
        "seconds_left": 650,
        "actions_left": 20,
        "per_call_seconds": 90,
        "image_observation": {
            "labels": ["supplied_reference"],
            "scope": "Observed reference only",
            "revision_id": None,
        },
        "failed_submission": {
            "validation_error": "Installed occurrences must be participants",
            "submitted_action": {"kind": "plan", "document_json": "{exact rejected intent}"},
            "revision_created": False,
        },
        "design_plan": {
            "operation_plan": {
                "dimensions": [
                    {"id": "original_bore", "value": 16, "source": "customer"},
                    {
                        "id": "changed_bore",
                        "value": 18,
                        "source": "customer",
                        "supersedes_fact_id": "original_bore",
                    },
                ],
                "steps": [{"id": "sleeve", "part_id": "sleeve", "requirement_ids": ["bore"]}],
            },
            "subsystems": [{"id": "request_scope", "requirement_ids": ["bore", "holding"]}],
            "functional_interfaces": [{"id": "joint", "participant_paths": ["sleeve", "pin"]}],
            "requirement_dispositions": [
                {"requirement_id": "holding", "physical_validation": ["Test retained load"]}
            ],
        },
        "plan_draft": {
            "version": 5,
            "document": {"objective": "Complete saved draft", "unresolved": ["Unknown material"]},
            "errors": [{"field": "steps[0].frame_id", "message": "Wrong owner"}],
        },
        "selected": {
            "id": "a" * 32,
            "document": {
                "assembly": {
                    "parts": [{"id": "sleeve", "features": [{"id": "cut_bore", "inputs": []}]}],
                    "roots": [{"id": "sleeve_occ", "part_id": "sleeve"}],
                    "mates": [{"id": "retained_pin"}],
                }
            },
            "measurement": {
                "geometry_valid": True,
                "requirements_satisfied": False,
                "artifact_hashes": {"part.step": "step_hash"},
                "errors": ["Pin overlap"],
                "facts": {"holes": [{"diameter_mm": 17.8}], "extents_mm": [20, 20, 10]},
            },
        },
        "failed_attempt": {"id": "b" * 32, "errors": ["Non-manifold negative feature"]},
        "early_checks": {"findings": [{"id": "mate", "status": "fail", "clearance_mm": -0.2}]},
        "completion": {
            "status": "unverified",
            "next_target": "retained_pin",
            "external_obligations": [{"physical_validation": ["Test retained load"]}],
        },
        "persistent_tool_observations": [{"source_text": {"text": "Complete captured manual"}}],
        "native_authoring": {
            "plan_schema": {"duplicate": "plan syntax"},
            "document_schema": {"properties": {"assembly": {"type": "object"}}},
            "feature_index": {"subtract": {"schema_query": "feature:subtract"}},
            "discovery": {
                "all_actions": ["read_design", "read_part", "schema", "feature_edit"],
                "plan_draft": "plan query=draft_read|draft_patch|draft_commit",
            },
            "conventions": "Bindings name variables",
        },
        "authoritative_schemas": {
            "plan": {"name": "plan", "schema": {"properties": {"objective": {"type": "string"}}}},
            "feature:subtract": {"kind": "subtract", "inputs": ["tool", "target"]},
        },
        "history": [{"schema_read": "feature:subtract"}],
        "controller_only": {"status": "must not become startup evidence"},
    }


def test_startup_contains_only_supplied_intent_fields(full_context):
    result = startup_context(full_context)
    assert set(result) == {
        "request",
        "user_revision_history",
        "edit_instruction",
        "frozen_contract",
        "engineering_inputs",
        "owned_assets",
        "source_claims",
        "source_artifacts",
        "edit_basis",
        "seconds_left",
        "actions_left",
        "per_call_seconds",
        "image_observation",
        "failed_submission",
        "phase",
        "scope",
    }
    for key in result.keys() - {"phase", "scope"}:
        assert result[key] == full_context[key]
    assert result["phase"] == "startup_intent"
    assert result["scope"] == "Pending authored intent; not geometry or completion evidence"


def test_startup_does_not_invent_missing_optional_facts():
    result = startup_context({"request": "Make a fixture"})
    assert set(result) == {"request", "phase", "scope"}
    assert "failed_submission" not in result


def test_startup_preserves_unlimited_and_zero_declared_allowances(full_context):
    full_context.update(seconds_left=None, actions_left=None, per_call_seconds=0)
    result = startup_context(full_context)
    assert result["seconds_left"] is None
    assert result["actions_left"] is None
    assert result["per_call_seconds"] == 0


def test_startup_is_deeply_pure_and_detached(full_context):
    original = copy.deepcopy(full_context)
    result = startup_context(full_context)
    result["frozen_contract"]["requirements"][0]["value"] = 99
    result["image_observation"]["labels"].append("different reference")
    result["failed_submission"]["submitted_action"]["document_json"] = "Changed in result"
    assert full_context == original


def test_construction_retains_all_saved_scope_and_errors(full_context):
    result = construction_context(full_context)
    for key in full_context.keys() - {"native_authoring", "authoritative_schemas", "history"}:
        assert result[key] == full_context[key]
    assert result["history"] == full_context["history"]
    assert result["design_plan"]["operation_plan"]["dimensions"][0]["value"] == 16
    assert result["design_plan"]["operation_plan"]["dimensions"][1]["value"] == 18
    assert result["completion"]["status"] == "unverified"
    assert result["completion"]["next_target"] == "retained_pin"


def test_construction_suppresses_only_duplicate_bootstrap_plan_schema(full_context):
    result = construction_context(full_context)
    assert result["native_authoring"] == {
        key: value
        for key, value in full_context["native_authoring"].items()
        if key != "plan_schema"
    }
    assert "document_schema" in result["native_authoring"]
    assert result["native_authoring"]["discovery"] == full_context["native_authoring"]["discovery"]


def test_every_cached_schema_is_exact_and_cataloged(full_context):
    exact = full_context["authoritative_schemas"]["feature:subtract"]
    full_context["history"][0]["authoring_schema"] = copy.deepcopy(exact)
    result = construction_context(full_context)
    assert result["authoritative_schemas"]["feature:subtract"] == exact
    assert result["history"] == [{"schema_read": "feature:subtract"}]
    assert json.dumps(result).count('"inputs": ["tool", "target"]') == 1
    assert result["authoritative_schemas"] == full_context["authoritative_schemas"]
    catalog = {row["name"]: row for row in result["schema_catalog"]}
    assert set(catalog) == set(full_context["authoritative_schemas"])
    for name, record in catalog.items():
        assert record["query"] == name
        assert record["retrieve"] == f"schema query={name}"
        assert len(record["sha256"]) == 64
        assert set(record["sha256"]) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    "last_row",
    [
        {"part": {"id": "part", "features": [{"id": "feature"}]}},
        {"revision_id": "a" * 32, "measurement": {"geometry_valid": False}},
        {"tool_error": "Wrong frame owner"},
        {},
        "no schema read",
    ],
    ids=["after_read_part", "after_build", "after_error", "empty", "non_mapping"],
)
def test_every_cached_schema_survives_a_non_schema_history_row(full_context, last_row):
    full_context["history"].append(last_row)
    result = construction_context(full_context)
    assert result["authoritative_schemas"] == full_context["authoritative_schemas"]
    assert result["history"][-1] == last_row


def test_older_matching_schema_duplicates_do_not_erase_other_history_fields(full_context):
    exact = full_context["authoritative_schemas"]["plan"]
    row = {
        "schema_read": "plan",
        "authoring_schema": copy.deepcopy(exact),
        "tool_error": "Part-scoped requirement mismatch",
        "read_design": {"items": [{"id": "negative_feature"}]},
    }
    full_context["history"].insert(0, row)
    result = construction_context(full_context)
    assert result["history"][0] == {
        key: value for key, value in row.items() if key != "authoring_schema"
    }
    assert result["authoritative_schemas"]["plan"] == exact


def test_mismatched_or_uncached_explicit_schema_reads_remain_exact(full_context):
    different = {"properties": {"older_field": {"type": "integer"}}}
    uncached = {"properties": {"external_tool": {"type": "number"}}}
    full_context["history"] = [
        {"schema_read": "plan", "authoring_schema": different},
        {"schema_read": "uncached_tool", "authoring_schema": uncached},
    ]
    result = construction_context(full_context)
    assert result["history"] == full_context["history"]
    assert result["authoritative_schemas"] == full_context["authoritative_schemas"]


def test_exact_part_design_and_source_reads_are_never_capped(full_context):
    rows = [
        {"read_part": {"id": "sleeve", "features": [{"id": f"f{i}"} for i in range(1500)]}},
        {
            "read_design": {
                "items": [{"id": f"p{i}"} for i in range(1500)],
                "cursor": "exact_cursor",
            }
        },
        {"source_text": {"text": "Complete reference text " * 10000}},
        {"source_page": {"text": "Complete page text " * 10000}},
    ]
    full_context["history"] = copy.deepcopy(rows)
    full_context["persistent_tool_observations"] = copy.deepcopy(rows)
    result = construction_context(full_context)
    assert result["history"] == rows
    assert result["persistent_tool_observations"] == rows
    assert len(json.dumps(result)) > 200000


def test_schema_identity_is_content_based_and_independent_of_object_key_order():
    one = construction_context({"authoritative_schemas": {"x": {"b": 2, "a": 1}}})
    same = construction_context({"authoritative_schemas": {"x": {"a": 1, "b": 2}}})
    changed = construction_context({"authoritative_schemas": {"x": {"a": 2, "b": 2}}})
    assert one["schema_catalog"][0]["sha256"] == same["schema_catalog"][0]["sha256"]
    assert one["schema_catalog"][0]["sha256"] != changed["schema_catalog"][0]["sha256"]


@pytest.mark.parametrize("schemas", [None, {}])
def test_absent_or_empty_cache_never_creates_completion_decisions(schemas):
    context = {"authoritative_schemas": schemas, "native_authoring": {"plan_schema": {}}}
    result = construction_context(context)
    assert result["authoritative_schemas"] == schemas
    assert result["native_authoring"] == {}
    assert "completion" not in result
    assert "status" not in result
    assert "phase" not in result


def test_construction_is_deeply_pure_and_detached(full_context):
    original = copy.deepcopy(full_context)
    result = construction_context(full_context)
    assert full_context == original
    result["selected"]["document"]["assembly"]["parts"][0]["features"].append({"id": "new"})
    result["design_plan"]["operation_plan"]["dimensions"][0]["value"] = 99
    result["native_authoring"]["discovery"]["all_actions"].clear()
    result["schema_catalog"][0]["retrieve"] = "Changed only in result"
    result["authoritative_schemas"]["feature:subtract"]["inputs"].append("changed")
    assert full_context == original


def test_entire_upstream_schema_cache_stays_exact_without_another_limit():
    schemas = {f"schema_{index}": {"description": str(index) * 25000} for index in range(6)}
    assert len(json.dumps(schemas)) < 160000
    context = {"authoritative_schemas": schemas, "history": [{"part": {"id": "part"}}]}
    original = copy.deepcopy(context)
    result = construction_context(context)
    assert result["authoritative_schemas"] == schemas
    assert len(json.dumps(result["authoritative_schemas"])) > 150000
    assert len(result["schema_catalog"]) == 6
    assert context == original


@pytest.fixture
def assembly_edit_schema() -> dict[str, Any]:
    return {
        "name": "assembly_edit",
        "schema": {
            "type": "object",
            "properties": {
                "parts": {"type": "array", "items": {"type": "object"}},
                "roots": {"type": "array", "items": {"type": "object"}},
                "mates": {"type": "array", "items": {"type": "object"}},
            },
        },
        "conventions": "Retain omitted parts, occurrence paths, ports, mates, and bindings",
    }


@pytest.mark.parametrize(
    "last_row",
    [
        {"part": {"id": "sleeve", "features": [{"id": "cut_bore", "inputs": []}]}},
        {
            "revision_id": "a" * 32,
            "measurement": {
                "geometry_valid": True,
                "requirements_satisfied": False,
                "errors": ["Pin overlap"],
            },
        },
    ],
    ids=["after_read_part", "after_build"],
)
def test_reusable_assembly_schema_survives_reads_and_builds(
    full_context, assembly_edit_schema, last_row
):
    full_context["authoritative_schemas"]["assembly_edit"] = assembly_edit_schema
    full_context["history"].append(last_row)
    original = copy.deepcopy(full_context)
    result = construction_context(full_context)
    assert result["authoritative_schemas"]["assembly_edit"] == assembly_edit_schema
    assert "is_summary" not in result["authoritative_schemas"]["assembly_edit"]
    for name in ("plan", "feature:subtract"):
        assert result["authoritative_schemas"][name] == original["authoritative_schemas"][name]
    assert {row["name"] for row in result["schema_catalog"]} == set(
        full_context["authoritative_schemas"]
    )
    for key in full_context.keys() - {"native_authoring", "authoritative_schemas"}:
        assert result[key] == original[key]
    assert result["native_authoring"] == {
        key: value for key, value in original["native_authoring"].items() if key != "plan_schema"
    }
    assert result["completion"]["status"] == "unverified"
    assert "status" not in result and "phase" not in result
    assert full_context == original
    result["authoritative_schemas"]["assembly_edit"]["schema"]["properties"].clear()
    assert full_context == original


def test_all_cached_schemas_remain_exact_with_matching_history_copies_removed(
    full_context, assembly_edit_schema
):
    full_context["authoritative_schemas"]["assembly_edit"] = assembly_edit_schema
    full_context["history"].insert(
        0, {"schema_read": "assembly_edit", "authoring_schema": copy.deepcopy(assembly_edit_schema)}
    )
    plan = full_context["authoritative_schemas"]["plan"]
    full_context["history"].insert(
        0, {"schema_read": "plan", "authoring_schema": copy.deepcopy(plan)}
    )
    full_context["history"].insert(
        0, {"schema_read": "plan", "authoring_schema": copy.deepcopy(plan)}
    )
    latest = full_context["authoritative_schemas"]["feature:subtract"]
    full_context["history"][-1]["authoring_schema"] = copy.deepcopy(latest)
    original = copy.deepcopy(full_context)
    result = construction_context(full_context)
    assert result["authoritative_schemas"]["assembly_edit"] == assembly_edit_schema
    assert result["authoritative_schemas"]["feature:subtract"] == latest
    assert result["authoritative_schemas"]["plan"] == original["authoritative_schemas"]["plan"]
    assert result["history"] == [
        {"schema_read": "plan"},
        {"schema_read": "plan"},
        {"schema_read": "assembly_edit"},
        {"schema_read": "feature:subtract"},
    ]
    assert json.dumps(result).count(assembly_edit_schema["conventions"]) == 1
    assert json.dumps(result).count('"inputs": ["tool", "target"]') == 1
    assert json.dumps(result).count('"objective": {"type": "string"}') == 1
    assert full_context == original
