"""Review summaries preserve authority, diagnostics and explicit evidence reads."""

import copy
import json
from typing import get_args

import pytest

from forge_mind.workspace.models import Action
from forge_mind.workspace.native_review_context import compact_review_context


def review_context():
    return {
        "request": "Make the sleeve and deliver review files",
        "edit_instruction": "Change the bore to 18 mm",
        "user_revision_history": ["Use a removable sleeve", "Change the bore to 18 mm"],
        "frozen_contract": {"requirements": [{"id": "bore", "value": 18}]},
        "design_plan": {
            "operation_plan": {
                "dimensions": [
                    {"id": "old", "value": 16, "binding_name": "bore"},
                    {"id": "new", "value": 18, "supersedes_fact_id": "old"},
                ]
            },
            "unresolved": ["Physical fit is unverified"],
        },
        "plan_reconciliation": {"active_facts": [{"id": "new", "value": 18}]},
        "selected": {
            "id": "a" * 32,
            "document_hash": "saved_document_hash",
            "document_is_summary": True,
            "document": {"representation": "summary_only_not_a_design_document"},
            "code": "redundant generated program",
            "measurement": {
                "geometry_valid": True,
                "requirements_satisfied": False,
                "findings": [{"id": "gap", "status": "fail", "measured": 0.1}],
                "errors": ["Saved mate is not aligned"],
                "caveats": ["Material strength is unverified"],
                "artifact_hashes": {"part.step": "exact_hash"},
                "facts": {
                    "bbox_min": [0, 0, 0],
                    "extents_mm": [20, 20, 10],
                    "volume_mm3": 2400,
                    "holes": [{"diameter_mm": 18}],
                    "assembly": {
                        "mates": [{"id": "pin", "status": "fail", "clearance_mm": -0.1}],
                        "ports": {"sleeve.bore": {"status": "fail", "diameter_mm": 18}},
                    },
                },
            },
        },
        "delivery_guidance": {
            "status": "current",
            "basis_hash": "b" * 64,
            "basis": {"revision_id": "a" * 32, "artifact_hashes": {"part.step": "exact_hash"}},
            "facts": {"measured": {"extents_mm": [20, 20, 10]}},
            "record": {
                "basis_hash": "b" * 64,
                "basis": {"revision_id": "a" * 32},
                "guidance": {
                    "print": "Print the sleeve",
                    "installation": "Inspect the bore before installation",
                    "use_and_removal": "Remove the sleeve by hand",
                    "unresolved": "Physical fit and strength remain unverified",
                },
            },
        },
        "authoritative_schemas": {
            "guidance": {"exact": True},
            "document": {"huge": "x" * 90000},
        },
        "native_authoring": {"document_schema": {"huge": "x" * 90000}},
        "history": [{"schema_read": "document", "authoring_schema": {"huge": "x" * 90000}}],
        "completion": {
            "next_target": "mate",
            "external_obligations": [{"physical_validation": ["Fit test"]}],
        },
    }


def test_review_context_is_deeply_pure_and_preserves_current_authority():
    context = review_context()
    original = copy.deepcopy(context)
    result = compact_review_context(context)
    assert context == original
    for key in (
        "request",
        "edit_instruction",
        "user_revision_history",
        "frozen_contract",
        "design_plan",
        "plan_reconciliation",
        "completion",
    ):
        assert result[key] == context[key]
    result["design_plan"]["unresolved"].append("Changed only in the result")
    result["delivery_guidance"]["record"]["guidance"]["print"] = "Changed only in the result"
    assert context == original


def test_review_reduces_duplicate_construction_syntax_but_retains_exact_guidance_schema():
    context = review_context()
    before = copy.deepcopy(context)
    result = compact_review_context(context)
    encoded = json.dumps(result)
    assert len(encoded) - len(json.dumps(result["authoritative_schemas"])) < 60000
    assert encoded.count("x" * 90000) == 1
    assert "document_schema" not in result["native_authoring"]
    assert result["authoritative_schemas"] == context["authoritative_schemas"]
    assert result["history"] == [{"schema_read": "document"}]
    assert context == before
    assert set(result["native_authoring"]["discovery"]["all_actions"]) == set(
        get_args(Action.model_fields["kind"].annotation)
    )
    assert "schema" in result["review_context"]["schema_retrieval"]
    assert "repair" in result["review_context"]["repair"]


def test_guidance_retains_exact_binding_and_all_four_customer_instructions():
    context = review_context()
    result = compact_review_context(context)
    guidance = result["delivery_guidance"]
    assert guidance["status"] == "current"
    assert guidance["basis_hash"] == "b" * 64
    assert guidance["revision_id"] == "a" * 32
    assert guidance["review_basis_available"] is True
    assert "basis" not in guidance and "basis" not in guidance["record"]
    assert guidance["record"]["guidance"] == context["delivery_guidance"]["record"]["guidance"]
    assert guidance["facts"] == context["delivery_guidance"]["facts"]


@pytest.mark.parametrize(
    "guidance",
    [
        {"status": "unavailable", "reason": "Missing artifact"},
        {"status": "current", "basis_hash": "b" * 64},
        {"status": "stale", "basis": {"revision_id": "a" * 32}},
    ],
)
def test_missing_guidance_basis_never_becomes_usable(guidance):
    context = review_context()
    context["delivery_guidance"] = guidance
    result = compact_review_context(context)["delivery_guidance"]
    assert result["review_basis_available"] is False
    assert result["status"] == guidance["status"]
    if "basis_hash" not in guidance:
        assert "basis_hash" not in result
    if "reason" in guidance:
        assert result["reason"] == guidance["reason"]


def test_current_failures_bounds_holes_and_mates_survive_inventory_reduction():
    context = review_context()
    facts = context["selected"]["measurement"]["facts"]
    facts["feature_history"] = [{"id": f"feature{i}", "status": "pass"} for i in range(300)]
    failed = {"id": "last_feature", "status": "fail", "errors": ["Disconnected result"]}
    facts["feature_history"].append(failed)
    context["failed_attempt"] = copy.deepcopy(context["selected"])
    result = compact_review_context(context)
    for key in ("selected", "failed_attempt"):
        before, after = context[key]["measurement"], result[key]["measurement"]
        for field in (
            "geometry_valid",
            "requirements_satisfied",
            "findings",
            "errors",
            "caveats",
            "artifact_hashes",
        ):
            assert after[field] == before[field]
        for field in ("bbox_min", "extents_mm", "volume_mm3", "holes", "assembly"):
            assert after["facts"][field] == before["facts"][field]
        assert after["facts"]["feature_history"]["is_summary"] is True
        assert failed in after["facts"]["feature_history"]["retained_diagnostics"]
        assert result[key]["document_hash"] == "saved_document_hash"


def test_explicit_part_design_and_source_reads_stay_complete():
    context = review_context()
    rows = [
        {"part": {"features": [{"code": "complete program " * 9000}]}},
        {"read_design": {"items": [{"id": f"f{i}"} for i in range(1200)]}},
        {"source_text": {"text": "complete source " * 9000}},
        {"source_page": {"text": "complete page " * 9000}},
    ]
    context["history"] = copy.deepcopy(rows)
    context["persistent_tool_observations"] = copy.deepcopy(rows)
    result = compact_review_context(context)
    assert result["history"] == rows
    assert result["persistent_tool_observations"] == rows
    assert len(json.dumps(result)) > 60000  # Explicit evidence is not silently truncated.


def test_only_identical_operation_plan_snapshot_is_deduplicated():
    context = review_context()
    context["selected"]["operation_record"] = {
        "plan_snapshot": {
            "plan": context["design_plan"],
            "plan_hash": "original_hash",
            "plan_version": 4,
        },
        "dimension_bindings": [{"fact_id": "old", "status": "changed", "current_value": 18}],
    }
    result = compact_review_context(context)
    record = result["selected"]["operation_record"]
    assert record["plan_snapshot"] == {"plan_hash": "original_hash", "plan_version": 4}
    assert (
        record["dimension_bindings"]
        == context["selected"]["operation_record"]["dimension_bindings"]
    )
    context["selected"]["operation_record"]["plan_snapshot"]["plan"] = {"old": True}
    preserved = compact_review_context(context)["selected"]["operation_record"]["plan_snapshot"]
    assert preserved["plan"] == {"old": True}
