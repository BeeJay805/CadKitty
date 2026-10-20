"""One explicit preparation dispatch preserves current basis, budget and final authority."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from forge_mind.workspace import native_handoff_prepare as handoff
from forge_mind.workspace.delivery_guidance import GuidanceUpdate
from forge_mind.workspace.document import digest
from forge_mind.workspace.native_authoring import native_bootstrap
from forge_mind.workspace.store import WorkspaceError

RID = "a" * 32


@pytest.fixture
def case(monkeypatch):
    state = {
        "revision_id": RID,
        "plan_hash": "plan",
        "contract_hash": "contract",
        "source_hash": "source",
        "engine_fingerprint": "engine",
        "print_setup_record_hash": digest(None),
    }
    session = {"selected_revision": RID, "job_id": "job", "authoring_mode": "native_operations"}
    c = SimpleNamespace(
        state=state,
        session=session,
        calls=[],
        receipt=None,
        reasons=[],
        required=False,
        supported=True,
        geometry="pass",
        active=True,
        cancelled=False,
        now=0.0,
        after_inspection=lambda: None,
        after_print=lambda: None,
    )

    class Connection:
        def execute(self, query, params):
            assert params == ("job", "project")
            return self

        def fetchone(self):
            return ("running" if c.active else "complete", c.cancelled)

    class Store:
        @contextmanager
        def connect(self):
            yield Connection()

    c.service = SimpleNamespace(store=Store())

    def read(store, sid, owner):
        assert store is c.service.store and sid == "project"
        if owner != "owner":
            raise WorkspaceError("workspace not found")
        return deepcopy(session), {}, deepcopy(state)

    def review(*args, **kwargs):
        c.calls.append("preflight")
        reasons = list(c.reasons)
        if c.required and not c.receipt:
            reasons.append("Canonical missing print obligation")
        return {
            "work_reasons": reasons,
            "print_preparation": {
                "required": c.required,
                "supported": c.supported,
                "status": "current" if c.receipt else "missing",
            },
        }

    def inspect(service, sid, owner, rid, *, deadline, cancelled):
        c.calls.append("inspection")
        assert rid == RID and not cancelled()
        result = {
            "revision_id": rid,
            "basis_hash": digest(state),
            "results": [
                {"method": "nominal_perforated_box_v1", "status": c.geometry, "critical": True},
                {"method": "digital_delivery_v1", "status": "unverified", "critical": True},
            ],
        }
        c.inspection_deadline = deadline
        c.after_inspection()
        return result

    def prepare(service, sid, owner, rid, *, timeout, cancelled):
        c.calls.append("print")
        c.print_timeout = timeout
        c.print_cancelled = cancelled
        assert rid == RID and not cancelled()
        if c.receipt is None:
            c.receipt = {"package_id": "saved", "revision_id": rid, "status": "prepared"}
            state["print_setup_record_hash"] = digest(c.receipt)
        c.after_print()
        return deepcopy(c.receipt)

    def setup(*args):
        c.calls.append("print_readback")
        return {"status": "current", "record": {"package": deepcopy(c.receipt)}}

    def guide(*args):
        c.calls.append("guidance")
        return {
            "status": "missing",
            "basis_hash": "b" * 64,
            "facts": {"physical_readiness": "unverified", "exact": [1, False, "source quote"]},
            "instruction_catalog": {"required_refs": ["physical:readiness"]},
        }

    monkeypatch.setattr(handoff, "_frozen_inputs", read)
    monkeypatch.setattr(handoff, "finish_context", review)
    monkeypatch.setattr(handoff, "inspect_requirements", inspect)
    monkeypatch.setattr(handoff, "prepare_print_files", prepare)
    monkeypatch.setattr(handoff, "print_setup_context", setup)
    monkeypatch.setattr(handoff, "guidance_context", guide)
    monkeypatch.setattr(
        handoff, "expired", lambda deadline: deadline is not None and c.now >= deadline
    )
    monkeypatch.setattr(
        handoff, "time_left", lambda deadline: None if deadline is None else deadline - c.now
    )
    c.guide = guide
    return c


def run(case, *, print_files=False, owner="owner", deadline=10.0):
    return handoff.prepare_handoff(
        case.service,
        "project",
        owner,
        "job",
        handoff.PrepareHandoffRequest(revision_id=RID, prepare_print=print_files),
        deadline=deadline,
        cancelled=lambda: case.cancelled,
    )


def test_explicit_false_never_prepares_and_returns_exact_current_facts(case):
    before = deepcopy(case.state)
    result = run(case)
    assert case.calls == ["preflight", "inspection", "preflight", "guidance"]
    assert result["status"] == "prepared_for_instruction_review"
    assert result["delivery_guidance"] == case.guide()
    assert result["guidance_schema"] == GuidanceUpdate.model_json_schema()
    assert result["requirements_inspection"]["results"][1]["status"] == "unverified"
    assert result["print_setup"] is None and case.receipt is None
    assert case.state == before
    assert case.session == {
        "selected_revision": RID,
        "job_id": "job",
        "authoring_mode": "native_operations",
    }


def test_requested_print_is_only_permitted_prerequisite_and_is_exactly_bound(case):
    case.required = True
    before = deepcopy(case.state)
    result = run(case, print_files=True)
    assert result["status"] == "prepared_for_instruction_review"
    assert case.calls == [
        "preflight",
        "inspection",
        "print",
        "print_readback",
        "preflight",
        "guidance",
    ]
    assert result["print_setup"] == case.receipt
    assert case.state == {**before, "print_setup_record_hash": digest(case.receipt)}
    assert case.inspection_deadline == 10.0 and case.print_timeout == 10.0


def test_already_current_requested_print_retains_its_exact_receipt(case):
    case.receipt = {"package_id": "existing", "revision_id": RID, "status": "prepared"}
    case.state["print_setup_record_hash"] = digest(case.receipt)
    before = deepcopy(case.state)
    result = run(case, print_files=True)
    assert result["print_setup"]["package_id"] == "existing"
    assert case.state == before


@pytest.mark.parametrize("requested", [False, True])
def test_other_work_blocks_even_when_print_is_requested(case, requested):
    case.required = True
    case.reasons = ["Incomplete plan ownership"]
    result = run(case, print_files=requested)
    assert result["status"] == "blocked"
    assert case.calls == ["preflight"] and case.receipt is None


def test_missing_required_print_is_not_implicit(case):
    case.required = True
    assert run(case)["status"] == "blocked"
    assert case.calls == ["preflight"] and case.receipt is None


def test_unsupported_flat_preparation_stops_without_conversion(case):
    case.required, case.supported = True, False
    assert run(case, print_files=True)["status"] == "blocked"
    assert case.calls == ["preflight"] and case.receipt is None


@pytest.mark.parametrize("status", ["fail", "unverified"])
def test_failed_or_unavailable_geometry_stops_before_print_and_guidance(case, status):
    case.geometry = status
    result = run(case, print_files=True)
    assert result["status"] == "blocked"
    assert case.calls == ["preflight", "inspection"]
    assert result["requirements_inspection"]["results"][0]["status"] == status
    assert case.receipt is None


@pytest.mark.parametrize(
    "field", ["plan_hash", "contract_hash", "source_hash", "engine_fingerprint"]
)
def test_inspection_input_change_stops_before_print(case, field):
    case.after_inspection = lambda: case.state.update({field: "changed"})
    with pytest.raises(WorkspaceError, match="inputs changed"):
        run(case, print_files=True)
    assert case.calls == ["preflight", "inspection"] and case.receipt is None


def test_selection_change_and_owner_refusal_stop_without_side_effects(case):
    with pytest.raises(WorkspaceError, match="not found"):
        run(case, owner="other")
    case.session["selected_revision"] = "b" * 32
    with pytest.raises(WorkspaceError, match="selected revision"):
        run(case)
    assert not case.calls


@pytest.mark.parametrize("change", ["cancel", "deadline", "inactive_job"])
def test_original_budget_or_job_stop_blocks_following_workers(case, change):
    if change == "cancel":
        case.after_inspection = lambda: setattr(case, "cancelled", True)
    elif change == "deadline":
        case.after_inspection = lambda: setattr(case, "now", 10.0)
    else:
        case.after_inspection = lambda: setattr(case, "active", False)
    with pytest.raises(WorkspaceError):
        run(case, print_files=True)
    assert case.calls == ["preflight", "inspection"] and case.receipt is None


def test_print_uses_remaining_original_time_and_callback(case):
    case.after_inspection = lambda: setattr(case, "now", 4.0)
    run(case, print_files=True)
    assert case.print_timeout == 6.0
    case.now = 10.0
    assert case.print_cancelled()


def test_later_staleness_retains_truthful_print_receipt_but_never_reports_success(case):
    case.after_print = lambda: case.state.update(plan_hash="changed")
    with pytest.raises(WorkspaceError, match="changed during print"):
        run(case, print_files=True)
    assert case.receipt["status"] == "prepared"
    assert "guidance" not in case.calls


def test_wrong_print_readback_cannot_supply_guidance(case, monkeypatch):
    monkeypatch.setattr(handoff, "print_setup_context", lambda *args: {"status": "stale"})
    assert run(case, print_files=True)["status"] == "blocked"
    assert case.receipt is not None and "guidance" not in case.calls


def test_forged_inspection_basis_refuses_preparation(case, monkeypatch):
    monkeypatch.setattr(
        handoff,
        "inspect_requirements",
        lambda *args, **kwargs: {"revision_id": RID, "basis_hash": "forged", "results": []},
    )
    with pytest.raises(WorkspaceError, match="different saved basis"):
        run(case, print_files=True)
    assert case.receipt is None


@pytest.mark.parametrize(
    "payload",
    [
        {"revision_id": RID},
        {"revision_id": RID, "prepare_print": "true"},
        {"revision_id": RID, "prepare_print": True, "transform": []},
        {"revision_id": "other", "prepare_print": False},
    ],
)
def test_request_is_closed_explicit_and_has_no_authored_pose(payload):
    with pytest.raises(ValidationError):
        handoff.PrepareHandoffRequest.model_validate(payload)


def test_bootstrap_contains_exact_request_schema_and_preserves_old_tools():
    bootstrap = native_bootstrap()
    assert bootstrap["prepare_handoff_schema"] == handoff.PrepareHandoffRequest.model_json_schema()
    for tool in ("requirements", "print_setup", "guidance", "prepare_handoff"):
        assert tool in bootstrap["discovery"]


def test_existing_agent_loop_dispatches_once_then_accepts_reviewed_instructions(
    tmp_path, monkeypatch
):
    from tests.unit.workspace.test_native_handoff_controller import (
        controller_case,
        reviewed_instructions,
        run_script,
    )

    from forge_mind.workspace.delivery_guidance import guidance_context
    from forge_mind.workspace.native_authoring import NativeAction

    controller = controller_case(tmp_path, monkeypatch)
    calls = []

    def prepared(service, sid, owner, jid, request, *, deadline, cancelled):
        assert service is controller.service and owner == "owner"
        assert request == handoff.PrepareHandoffRequest(revision_id=RID, prepare_print=False)
        assert not cancelled() and deadline > controller.clock.now
        assert service.store.get(sid, owner)["job_id"] == jid
        calls.append(request)
        return {
            "status": "prepared_for_instruction_review",
            "revision_id": RID,
            "delivery_guidance": guidance_context(service, sid, owner, RID),
            "guidance_schema": GuidanceUpdate.model_json_schema(),
        }

    monkeypatch.setattr(handoff, "prepare_handoff", prepared)

    def script(context, output_model):
        if len(controller.calls) == 1:
            return NativeAction(
                kind="inspect",
                query="prepare_handoff",
                revision_id=RID,
                explanation="Inspect current requirements and prepare instruction facts",
                document_json=handoff.PrepareHandoffRequest(
                    revision_id=RID, prepare_print=False
                ).model_dump_json(),
            )
        assert len(controller.calls) == 2
        return NativeAction(
            kind="guidance",
            query="finish",
            revision_id=RID,
            explanation="Submit reviewed current instructions for existing final validation",
            document_json=reviewed_instructions(context).model_dump_json(),
        )

    saved = run_script(controller, script)
    assert len(calls) == 1 and len(controller.calls) == 2
    assert "guidance" in saved["schema_cache"]
    assert len(controller.capture) == 1 and len(controller.validation) == 1
    assert saved["native_review"]["status"] == "ready_for_validation"
    assert saved.get("accepted_revision") is None


@pytest.mark.parametrize("phase", ["preflight", "guidance"])
def test_changed_basis_during_read_phase_never_returns_prepared(case, monkeypatch, phase):
    original = handoff.finish_context if phase == "preflight" else handoff.guidance_context

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        case.state["source_hash"] = "changed during read"
        return result

    monkeypatch.setattr(
        handoff, "finish_context" if phase == "preflight" else "guidance_context", changed
    )
    with pytest.raises(WorkspaceError, match="inputs changed"):
        run(case)
    assert case.receipt is None
