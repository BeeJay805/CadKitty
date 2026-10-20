"""Instruction presentation preserves evidence, identity and every repair route."""

from copy import deepcopy
from typing import get_args

import pytest

from forge_mind.workspace.author_guidance import (
    FINAL_INSTRUCTIONS,
    INSTRUCTION_REVIEW,
    build_author_guidance,
)
from forge_mind.workspace.instruction_context import customer_instruction_context
from forge_mind.workspace.models import Action


def instruction_context():
    contract = {"requirements": [{"id": "width", "value": 20, "tolerance": 0}]}
    measurement = {
        "geometry_valid": False,
        "requirements_satisfied": False,
        "contract_hash": "c" * 64,
        "findings": [{"requirement_id": "width", "status": "fail", "measured": 19}],
        "errors": ["Disconnected body"],
        "caveats": ["Adhesion is unverified"],
        "facts": {
            "extents_mm": [19, 10, 5],
            "feature_history": [
                {"id": f"f{index}", "status": "fail", "errors": ["Exact failure"]}
                for index in range(80)
            ],
        },
    }
    basis = {"revision_id": "a" * 32, "plan_hash": "d" * 64, "artifact_hashes": {}}
    return {
        "request": "Make a removable block with STL files",
        "authoring_mode": "native_operations",
        "user_revision_history": ["Keep a 20 mm width"],
        "edit_instruction": "Keep a 20 mm width",
        "frozen_contract": contract,
        "request_attribution": {"proposed_contract": deepcopy(contract), "status": "interpreted"},
        "selected": {
            "id": "a" * 32,
            "document_hash": "e" * 64,
            "document_is_summary": True,
            "document": {"representation": "summary_only_not_a_design_document"},
            "code": "generated source " * 1000,
            "contract": deepcopy(contract),
            "measurement": measurement,
        },
        "design_plan": {"unresolved": ["Physical fit is unverified"]},
        "plan_reconciliation": {"status": "aligned", "active_facts": [{"value": 20}]},
        "completion": {
            "next_target": "body",
            "external_obligations": [{"missing_inputs": ["Printer profile"]}],
            "physical_readiness": "unverified",
        },
        "early_checks": {"status": "check_failed", "reason": "Body is disconnected"},
        "delivery_guidance": {
            "status": "stale",
            "basis": basis,
            "basis_hash": "b" * 64,
            "target_kind": "pending_candidate",
            "design_revision": "a" * 32,
            "selected_revision": "f" * 32,
            "physical_readiness": "unverified",
            "facts": {
                "requirement_findings": deepcopy(measurement["findings"]),
                "measured": {"extents_mm": [19, 10, 5]},
            },
        },
        "native_authoring": {"document_schema": {"huge": "source syntax " * 1000}},
        "authoritative_schemas": {
            "guidance": {"exact": True},
            "assembly_edit": {"huge": "source syntax " * 1000},
        },
        "history": [{"measurement": deepcopy(measurement), "regressions": ["width"]}],
    }


def test_instruction_projection_is_pure_and_keeps_exact_authority_and_all_failures():
    context = instruction_context()
    context["failed_attempt"] = deepcopy(context["selected"])
    before = deepcopy(context)
    result = customer_instruction_context(context)
    assert context == before
    for key in (
        "request",
        "user_revision_history",
        "edit_instruction",
        "frozen_contract",
        "design_plan",
        "plan_reconciliation",
        "completion",
        "early_checks",
    ):
        assert result[key] == context[key]
    for key in ("selected", "failed_attempt"):
        assert result[key]["id"] == context[key]["id"]
        assert result[key]["document_hash"] == context[key]["document_hash"]
        assert result[key]["measurement"] == context[key]["measurement"]
        assert "code" not in result[key]
        assert result[key]["contract_in_context"] == "frozen_contract"
    assert result["history"] == context["history"]
    guidance = result["delivery_guidance"]
    for key in (
        "basis",
        "basis_hash",
        "target_kind",
        "design_revision",
        "selected_revision",
        "facts",
    ):
        assert guidance[key] == context["delivery_guidance"][key]
    result["selected"]["measurement"]["errors"].append("Only the presentation changed")
    result["delivery_guidance"]["basis"]["artifact_hashes"]["new"] = "changed"
    assert context == before


def test_customer_delivery_view_is_carried_as_an_opaque_detached_value(monkeypatch):
    context = instruction_context()
    view = {"future_facet": {"scope": "unchanged opaque contract"}}
    seen = []

    def project(guidance):
        seen.append(guidance)
        return deepcopy(view)

    monkeypatch.setattr("forge_mind.workspace.instruction_context.customer_delivery_view", project)
    result = customer_instruction_context(context)
    assert seen == [context["delivery_guidance"]]
    assert result["customer_delivery"] == view
    assert result["request_attribution"]["proposed_contract_in_context"] == "frozen_contract"


def test_missing_contract_is_not_replaced_by_an_invented_authority_reference():
    context = instruction_context()
    del context["frozen_contract"]
    del context["selected"]["contract"]
    context["request_attribution"]["proposed_contract"] = None
    result = customer_instruction_context(context)
    assert "contract_in_context" not in result["selected"]
    assert result["request_attribution"]["proposed_contract"] is None
    assert "proposed_contract_in_context" not in result["request_attribution"]


def test_different_candidate_contract_and_stale_binding_remain_explicit():
    context = instruction_context()
    context["selected"]["contract"] = {"requirements": [{"id": "width", "value": 18}]}
    context["request_attribution"]["proposed_contract"] = {
        "requirements": [{"id": "width", "value": 22}]
    }
    result = customer_instruction_context(context)
    assert result["selected"]["contract"] == context["selected"]["contract"]
    assert "contract_in_context" not in result["selected"]
    assert result["request_attribution"] == context["request_attribution"]
    assert result["delivery_guidance"]["status"] == "stale"
    assert result["delivery_guidance"]["basis"] == context["delivery_guidance"]["basis"]


def test_instruction_context_keeps_all_actions_and_exact_explicit_reads():
    context = instruction_context()
    explicit = [
        {"part": {"features": [{"code": "exact source " * 1000}]}},
        {"read_design": {"items": [{"id": "f1"}], "next_cursor": "opaque"}},
        {"source_page": {"text": "complete supplier limits " * 1000}},
        {"source_text": {"text": "complete source " * 1000}},
    ]
    context["history"] = deepcopy(explicit)
    context["persistent_tool_observations"] = deepcopy(explicit)
    result = customer_instruction_context(context)
    assert result["history"] == explicit
    assert result["persistent_tool_observations"] == explicit
    assert result["authoritative_schemas"] == context["authoritative_schemas"]
    assert "document_schema" not in result["native_authoring"]
    assert set(result["native_authoring"]["discovery"]["all_actions"]) == set(
        get_args(Action.model_fields["kind"].annotation)
    )
    assert "repair" in result["instruction_presentation"]["retrieval"]


@pytest.mark.parametrize("guidance", [None, {"status": "unavailable"}, {"status": "stale"}])
def test_missing_or_stale_guidance_cannot_gain_a_basis_or_current_status(guidance):
    context = instruction_context()
    context["delivery_guidance"] = guidance
    result = customer_instruction_context(context)
    projected = result["delivery_guidance"]
    if guidance is None:
        assert projected is None
    else:
        assert projected["status"] == guidance["status"]
        assert "basis_hash" not in projected
        assert projected["review_basis_available"] is False


def test_automatic_flat_feature_source_is_not_a_replacement_document():
    context = instruction_context()
    context["selected"]["document"] = {
        "name": "Block",
        "features": [{"id": "body", "kind": "box", "code": "not instruction content"}],
        "outputs": ["body"],
        "production": {"role": "make", "notes": "Physical strength unverified"},
    }
    result = customer_instruction_context(context)
    document = result["selected"]["document"]
    assert "features" not in document
    assert document["feature_ids"] == ["body"]
    assert document["production"] == context["selected"]["document"]["production"]
    assert document["representation"] == "instruction_reference_only_not_a_design_document"
    assert "read_design" in document["retrieve"]


def test_large_automatic_source_keeps_all_non_source_limits_and_metadata():
    context = instruction_context()
    selected = context["selected"]
    selected["document_is_summary"] = False
    selected["document"] = {
        "name": "Block",
        "features": [{"id": "body", "kind": "box", "code": "source " * 5000}],
        "outputs": ["body"],
        "production": {"notes": "Keep every physical limitation " * 1000},
        "delivery_guidance": {"unresolved": "A retained physical limit"},
    }
    result = customer_instruction_context(context)
    document = result["selected"]["document"]
    for key in ("production", "delivery_guidance"):
        assert document[key] == selected["document"][key]
    assert document["feature_ids"] == ["body"]
    assert result["selected"]["measurement"] == selected["measurement"]
    assert result["review_context"]["is_summary"] is True
    assert result["plan_reconciliation"] == context["plan_reconciliation"]


def test_instruction_systems_have_distinct_response_contracts_and_keep_authority():
    assert "Return one NativeAction" in INSTRUCTION_REVIEW
    assert "Every native action remains available" in INSTRUCTION_REVIEW
    assert "guidance query=finish" in INSTRUCTION_REVIEW
    assert "Return the typed GuidanceUpdate only" in FINAL_INSTRUCTIONS
    for system in (INSTRUCTION_REVIEW, FINAL_INSTRUCTIONS):
        assert "customer_delivery" in system
        assert "basis_hash" in system
        assert "physical" in system
    assert "no inline plan field" in INSTRUCTION_REVIEW


def test_native_system_never_advertises_the_unavailable_inline_plan_field():
    system = build_author_guidance(
        {"request": "Make an assembly", "authoring_mode": "native_operations"},
        "trusted_local",
    )
    assert "may attach it to an action" not in system
    assert "Attach the plan to construction" not in system
    assert "the compact response has no inline plan field" in system
    assert "Save the plan separately with kind=plan and document_json" in system
