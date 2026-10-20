"""Protected instruction authority differs from prose freshness and physical proof."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from forge_mind.workspace.instruction_context import ProtectedGuidanceUpdate
from forge_mind.workspace.native_review_context import compact_review_context
from forge_mind.workspace.store import WorkspaceError


def test_protected_response_requires_references_without_adding_prose_authority():
    payload = {
        "revision_id": "a" * 32,
        "basis_hash": "b" * 64,
        "reviewed_current_facts": True,
        "guidance": {
            "print": "Unverified author print note",
            "installation": "Unverified author installation note",
            "use_and_removal": "Unverified author use note",
            "unresolved": "Unverified author limitation",
        },
    }
    with pytest.raises(ValidationError, match="instruction_plan"):
        ProtectedGuidanceUpdate.model_validate(payload)
    payload["instruction_plan"] = {
        "version": 1,
        "fact_refs": ["physical:readiness"],
        "steps": [{"action": "review_physical_limits", "target": "physical:readiness"}],
    }
    update = ProtectedGuidanceUpdate.model_validate(payload)
    assert update.guidance.print == payload["guidance"]["print"]
    assert "instruction_plan" in ProtectedGuidanceUpdate.model_json_schema()["required"]


def test_automatic_history_retains_integrity_metadata_without_repeating_author_claims():
    history = [
        {
            "version": 1,
            "predecessor_hash": None,
            "record_hash": "c" * 64,
            "record": {"guidance": {"print": "unverified old claim " * 1000}},
        }
    ]
    context = {"delivery_guidance": {"history": history, "history_status": {"status": "verified"}}}
    original = deepcopy(context)
    view = compact_review_context(context)
    assert context == original
    summary = view["delivery_guidance"]["history"]
    assert summary["entry_count"] == 1
    assert summary["entries"][0]["record_hash"] == "c" * 64
    assert "record" not in summary["entries"][0]
    assert view["delivery_guidance"]["history_status"] == {"status": "verified"}
    assert "guidance query=read" in summary["retrieve"]


def test_explicit_guidance_history_read_remains_complete_in_next_review_context():
    response = {
        "status": "current",
        "history": [{"version": 1, "record": {"guidance": {"print": "literal old note"}}}],
        "basis": {"revision_id": "a" * 32},
    }
    context = {
        "delivery_guidance": deepcopy(response),
        "history": [{"guidance": deepcopy(response), "explicit_guidance_read": True}],
        "persistent_tool_observations": [
            {"guidance": deepcopy(response), "explicit_guidance_read": True}
        ],
    }
    before = deepcopy(context)
    view = compact_review_context(context)
    assert context == before
    assert view["delivery_guidance"]["history"]["is_summary"] is True
    assert view["history"] == context["history"]
    assert view["persistent_tool_observations"] == context["persistent_tool_observations"]


def closure_case(monkeypatch, *, binding="legacy_unverified", replace_record=False):
    from tests.unit.workspace.test_operation_authoring_flow import contract, document, plan

    from forge_mind.workspace import native_finish

    frozen = contract()
    sid, rid = "b" * 32, "a" * 32
    session = {
        "selected_revision": rid,
        "design_plan": plan().model_dump(),
        "plan_hash": "c" * 64,
        "plan_contract_hash": frozen.fingerprint(),
    }
    revision = {
        "document": document().model_dump(),
        "contract": frozen.model_dump(),
        "operation_record": {"plan_snapshot": {"plan_hash": session["plan_hash"]}},
        "measurement": {"findings": []},
    }
    service = SimpleNamespace(
        store=SimpleNamespace(get=lambda *args: deepcopy(session)),
        _load_revision=lambda *args: deepcopy(revision),
    )
    guide = {
        "status": "current",
        "basis": {
            "contract_hash": frozen.fingerprint(),
            "session_contract_hash": frozen.fingerprint(),
        },
        "basis_hash": "d" * 64,
        "record": {"guidance_hash": "e" * 64},
        "instruction_binding": {"status": binding, "physical_readiness": "unverified"},
    }
    reads = []

    def current(*args):
        result = deepcopy(guide)
        if reads and replace_record:
            result["record"]["guidance_hash"] = "f" * 64
        reads.append(result)
        return result

    monkeypatch.setattr(native_finish, "guidance_context", current)
    monkeypatch.setattr(native_finish, "alignment", lambda *args: {"status": "aligned"})
    monkeypatch.setattr(
        native_finish,
        "coverage",
        lambda *args: {
            "graph_complete": True,
            "references_complete": True,
            "parameters_complete": True,
        },
    )
    monkeypatch.setattr(native_finish, "completion_context", lambda *args: {})
    monkeypatch.setattr(
        native_finish, "reference_collision_context", lambda *args: {"status": "not_applicable"}
    )
    return native_finish, service, sid, rid


def test_legacy_draft_closure_and_protected_bound_closure_remain_separate(monkeypatch):
    module, service, sid, rid = closure_case(monkeypatch)
    legacy = module.finish_context(service, sid, "owner", rid)
    protected = module.finish_context(service, sid, "owner", rid, require_bound_instructions=True)
    assert legacy["status"] == "ready_for_validation"
    assert protected["status"] == "needs_work"
    assert any("instruction plan" in reason for reason in protected["reasons"])
    assert legacy["physical_readiness"] == protected["physical_readiness"] == "unverified"


def test_same_basis_instruction_replacement_cannot_race_final_closure(monkeypatch):
    module, service, sid, rid = closure_case(monkeypatch, binding="bound", replace_record=True)
    with pytest.raises(WorkspaceError, match="inputs changed"):
        module.finish_context(service, sid, "owner", rid, require_bound_instructions=True)
