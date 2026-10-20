"""Exact approved-command dispatch with real Store state and synthetic CAD output."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest
from tests.unit.workspace.test_operation_plan import contract, document, plan
from tests.unit.workspace.test_revision_bound_edits import saved_revision
from tests.unit.workspace.test_staged_native_author import intent

from forge_mind.llm.cost import LLMUsage
from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.customer_approval import (
    ApprovalRequest,
    CustomerApprovalRequiredError,
    approve_proposal,
    preserve_budget,
    read_proposal,
)
from forge_mind.workspace.customer_construction import (
    construction_finalization_allowed,
    guard_document,
    stage_construction,
)
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Action


def setup_project(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, steps: list[Any], *, admitted: bool = False
):
    from forge_mind.workspace.store import Store

    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    # Synthetic worker bytes carry this declared test engine identity. The
    # separate real source fingerprint used by approval remains unchanged.
    monkeypatch.setattr("forge_mind.workspace.measure.engine_fingerprint", lambda: "f" * 64)
    monkeypatch.setattr(
        "forge_mind.workspace.delivery_guidance.engine_fingerprint", lambda: "f" * 64
    )
    store = Store(tmp_path)
    if admitted:
        # A server fixture with already-admitted requirements isolates author
        # dispatch from the separately tested intake/extraction transaction.
        session = store.create("owner", "Width 20 mm", contract().model_dump(), "native_operations")
        store.update(
            session["id"],
            project_kind="workspace",
            creation_request_id="a" * 32,
            approval_required=True,
            customer_printable=False,
        )
        store.enqueue(session["id"], "owner", {"max_actions": 8, "max_seconds": 100})
    else:
        session = store.create_customer_project(
            "owner",
            {
                "request_id": "a" * 32,
                "prompt": "Width 20 mm",
                "approval_required": True,
                "printable": False,
                "max_seconds": 100,
                "max_actions": 8,
            },
        )
    sid = session["id"]
    calls: list[dict[str, Any]] = []
    responses = iter(steps)

    class Client:
        model = "synthetic-dispatch-test"
        timeout_s = None

        def generate_structured(self, **kwargs: Any):
            calls.append(json.loads(kwargs["user"]))
            value = next(responses)
            return kwargs["output_model"].model_validate(value.model_dump()), LLMUsage(
                self.model, 0, 0
            )

    service = WorkspaceService(store, Executor("trusted_local"), lambda **_: Client())
    monkeypatch.setattr(service, "build", lambda *a, **kw: pytest.fail("No native CAD may run"))
    first = store.claim()
    assert first is not None
    return store, service, sid, first, calls


def approved_command(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    atomic: bool = False,
    feature_edit: bool = False,
):
    store, service, sid, first, calls = setup_project(tmp_path, monkeypatch, [])
    store.set_design_plan_for_job(sid, "owner", first["id"], plan(), 0, contract())
    parent = None
    feature_basis = None
    target_document = document()
    if feature_edit:
        parent = "b" * 32
        saved_revision(
            store, sid, parent, document(), contract(), store.plan_snapshot(sid, "owner")
        )
        store.update(sid, selected_revision=parent)
        feature_basis = store.edit_basis(sid, "owner")
        target_document = target_document.model_copy(update={"name": "Approved edited plate"})
    prepared = None
    if atomic:
        from forge_mind.workspace.construction_transaction import prepare_construction
        from forge_mind.workspace.plan_patch import PlanPatch

        prepared = prepare_construction(
            service,
            sid,
            "owner",
            first["id"],
            target_document,
            PlanPatch(
                basis=store.edit_basis(sid, "owner"), verification=["Inspect the plate edge"]
            ),
            author_intent="Width 20 mm",
        )
    with pytest.raises(CustomerApprovalRequiredError):
        stage_construction(
            service,
            sid,
            "owner",
            first["id"],
            target_document,
            contract(),
            parent,
            "Exact owner-reviewed construction",
            prepared=prepared,
            feature_basis=feature_basis,
            remaining_seconds=75,
            remaining_actions=1,
        )
    preserve_budget(store, sid, "owner", first["id"], 75, 1)
    store.finish(first["id"], sid, "needs_approval")
    proposal = read_proposal(store, sid, "owner")
    approve_proposal(
        store,
        sid,
        "owner",
        ApprovalRequest(proposal_hash=proposal["proposal_hash"], basis=proposal["basis"]),
    )
    resumed = store.claim()
    assert resumed is not None
    assert resumed["max_actions"] == 1
    return store, service, sid, resumed, calls


def test_approved_command_dispatches_exact_document_before_any_model_call(tmp_path, monkeypatch):
    store, service, sid, resumed, calls = approved_command(tmp_path, monkeypatch)
    observed = []

    def build(service, sid, owner, value, frozen, parent, explanation, timeout, cancelled):
        assert calls == []
        guard_document(
            service,
            sid,
            owner,
            value,
            frozen,
            parent,
            explanation,
            store.plan_snapshot(sid, owner),
        )
        observed.append((value.model_dump(), explanation, parent, frozen.model_dump()))
        return saved_revision(
            store, sid, "c" * 32, value, frozen, store.plan_snapshot(sid, owner), parent
        )

    monkeypatch.setattr("forge_mind.workspace.document_service.build_document", build)
    state = service._run_agent_job(resumed)
    assert state != "error"
    assert observed == [
        (
            document().model_dump(),
            "Exact owner-reviewed construction",
            None,
            contract().model_dump(),
        )
    ]
    assert calls == []
    assert store.get(sid, "owner")["selected_revision"] == "c" * 32
    assert construction_finalization_allowed(store, sid, "owner", resumed["id"])
    # These bytes only establish dispatch/publication ordering, never native geometry.
    assert (store.directory(sid, "c" * 32) / "part.step").read_text().startswith("Synthetic")


def test_claimed_failure_stops_before_model_repair_or_second_build(tmp_path, monkeypatch):
    store, service, sid, resumed, calls = approved_command(tmp_path, monkeypatch)
    builds = []

    def fail(*args, **kwargs):
        builds.append(True)
        raise ValueError("Synthetic native worker failure")

    monkeypatch.setattr("forge_mind.workspace.document_service.build_document", fail)
    assert service._run_agent_job(resumed) == "error"
    assert builds == [True]
    assert calls == []
    # A repeated invocation encounters the durable attempted command; it cannot
    # re-enter the dispatcher or obtain a new model-authored substitute.
    assert service._run_agent_job(resumed) == "error"
    assert builds == [True]
    assert calls == []
    assert store.revisions(sid) == []


def test_initial_inventory_does_not_pause_and_every_projection_keeps_workflow_state(
    tmp_path, monkeypatch
):
    startup = intent().model_copy(deep=True)
    for part in startup.parts:
        part.requirement_ids = ["width"]
    store, service, sid, first, calls = setup_project(
        tmp_path,
        monkeypatch,
        [startup, Action(kind="schema", query="construction", explanation="Read construction")],
        admitted=True,
    )
    first["max_actions"] = 2
    state = service._run_agent_job(first)
    assert state not in {"needs_approval", "error"}
    assert len(calls) == 2
    assert calls[0]["phase"] == "startup_intent"
    assert all(call.get("author_workflow_context") for call in calls)
    assert store.revisions(sid) == []
    assert read_proposal(store, sid, "owner")["status"] != "pending"


def test_finish_without_cad_returns_feedback_in_original_action_budget(tmp_path, monkeypatch):
    store, service, sid, first, calls = setup_project(
        tmp_path,
        monkeypatch,
        [
            Action(kind="finish", explanation="Try to finish before CAD"),
            Action(kind="schema", query="construction", explanation="Read construction"),
        ],
        admitted=True,
    )
    store.set_design_plan_for_job(sid, "owner", first["id"], plan(), 0, contract())
    first["max_actions"] = 2
    monkeypatch.setattr(
        "forge_mind.workspace.native_finish.finish_context",
        lambda *a, **kw: pytest.fail("No-revision finish must not enter saved-revision review"),
    )
    state = service._run_agent_job(first)
    assert state != "error"
    assert len(calls) == 2
    assert calls[1]["actions_left"] == 1
    assert any("final_review_correction" in row for row in calls[1]["history"])
    assert store.revisions(sid) == []
    assert store.get(sid, "owner")["native_review"]["status"] == "needs_work"


def test_claimed_failure_suppresses_completion_capture_and_proof(tmp_path, monkeypatch):
    store, service, sid, resumed, calls = approved_command(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("Synthetic native worker failure")

    def forbidden(*args, **kwargs):
        pytest.fail("Uncertain approved work must not finalize, capture or verify")

    monkeypatch.setattr("forge_mind.workspace.document_service.build_document", fail)
    monkeypatch.setattr("forge_mind.workspace.completion.finalize_completion", forbidden)
    monkeypatch.setattr("forge_mind.workspace.native_delivery.capture_native_delivery", forbidden)
    monkeypatch.setattr("forge_mind.workspace.requirement_verification.verify_contract", forbidden)
    service.run_job(resumed)
    session = store.get(sid, "owner")
    assert session["state"] == "error"
    assert session["automatic_delivery"]["status"] == "incomplete"
    assert calls == []


def test_approved_atomic_patch_uses_original_basis_and_existing_publication(tmp_path, monkeypatch):
    store, service, sid, resumed, calls = approved_command(tmp_path, monkeypatch, atomic=True)
    approved = store.session_record(sid, "owner")["customer_proposal"]["command"]
    seen = []
    monkeypatch.setattr("forge_mind.workspace.measure.engine_fingerprint", lambda: "f" * 64)

    def build(
        service,
        sid,
        owner,
        value,
        frozen,
        parent,
        explanation,
        timeout,
        cancelled,
        *,
        plan_snapshot,
    ):
        assert calls == []
        guard_document(service, sid, owner, value, frozen, parent, explanation, plan_snapshot)
        seen.append(plan_snapshot)
        revision = saved_revision(store, sid, "d" * 32, value, frozen, plan_snapshot, parent)
        directory = store.directory(sid, revision["id"])
        for name, content in (
            ("operation-plan.json", plan_snapshot),
            ("operation-record.json", revision["operation_record"]),
        ):
            (directory / name).write_text(json.dumps(content), encoding="utf-8")
            revision["measurement"]["artifact_hashes"][name] = file_sha256(directory / name)
        with store.connect() as con:
            con.execute(
                "UPDATE revisions SET document=? WHERE id=?",
                (json.dumps(revision), revision["id"]),
            )
        return revision

    monkeypatch.setattr("forge_mind.workspace.document_service.build_document", build)
    assert service._run_agent_job(resumed) != "error", store.get(sid, "owner")["events"][-4:]
    assert seen == [approved["target_snapshot"]]
    assert store.get(sid, "owner")["selected_revision"] == "d" * 32
    assert (
        store.get(sid, "owner")["plan_version"]
        == approved["plan_patch"]["basis"]["plan_version"] + 1
    )
    execution = store.session_record(sid, "owner")["customer_proposal"]["execution"]
    assert execution["phase"] == "published"
    assert execution["publication"]["applied"] is True
    assert calls == []


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_claimed_command_checks_cancel_and_original_deadline_before_dispatch(
    tmp_path, monkeypatch, stop
):
    from forge_mind.workspace import customer_construction

    store, service, sid, resumed, calls = approved_command(tmp_path, monkeypatch)
    original_claim = customer_construction.claim_construction

    def claim(*args):
        result = original_claim(*args)
        if stop == "cancel":
            store.cancel(sid, "owner")
        else:
            monkeypatch.setattr("forge_mind.workspace.agent.expired", lambda _: True)
        return result

    monkeypatch.setattr(customer_construction, "claim_construction", claim)
    monkeypatch.setattr(
        "forge_mind.workspace.document_service.build_document",
        lambda *a, **kw: pytest.fail("Stopped approved command cannot enter CAD"),
    )
    assert service._run_agent_job(resumed) == "error"
    assert calls == []
    assert store.revisions(sid) == []


def test_approved_feature_edit_keeps_server_saved_original_basis(tmp_path, monkeypatch):
    store, service, sid, resumed, calls = approved_command(tmp_path, monkeypatch, feature_edit=True)
    approved = store.session_record(sid, "owner")["customer_proposal"]["command"]
    observed = []
    original_publish = store.publish_feature_edit
    monkeypatch.setattr("forge_mind.workspace.measure.engine_fingerprint", lambda: "f" * 64)

    def build(service, sid, owner, value, frozen, parent, explanation, timeout, cancelled):
        assert calls == []
        assert parent == approved["feature_basis"]["revision_id"]
        guard_document(
            service,
            sid,
            owner,
            value,
            frozen,
            parent,
            explanation,
            store.plan_snapshot(sid, owner),
        )
        return saved_revision(
            store, sid, "e" * 32, value, frozen, store.plan_snapshot(sid, owner), parent
        )

    def publish(sid, owner, jid, basis, revision):
        observed.append(basis)
        return original_publish(sid, owner, jid, basis, revision)

    monkeypatch.setattr("forge_mind.workspace.document_service.build_document", build)
    monkeypatch.setattr(store, "publish_feature_edit", publish)
    assert service._run_agent_job(resumed) != "error", store.get(sid, "owner")["events"][-4:]
    assert observed == [approved["feature_basis"]]
    assert store.get(sid, "owner")["selected_revision"] == "e" * 32
    assert calls == []


def test_preview_finish_without_cad_keeps_existing_plan_only_stop(tmp_path, monkeypatch):
    store, service, sid, first, calls = setup_project(
        tmp_path,
        monkeypatch,
        [Action(kind="finish", explanation="Finish plan preview")],
        admitted=True,
    )
    store.set_design_plan_for_job(sid, "owner", first["id"], plan(), 0, contract())
    first.update(max_actions=2, preview_only=True)
    state = service._run_agent_job(first)
    assert state != "error"
    assert len(calls) == 1
    assert not store.get(sid, "owner").get("incomplete_finish_job_id")
    assert store.revisions(sid) == []


@pytest.mark.parametrize("owner_changes_selection", [False, True])
def test_published_continuation_reads_guidance_without_cad_replay_or_new_execution_identity(
    tmp_path, monkeypatch, owner_changes_selection
):
    store, service, sid, executed, initial_calls = approved_command(tmp_path, monkeypatch)

    def build(service, sid, owner, value, frozen, parent, explanation, timeout, cancelled):
        guard_document(
            service,
            sid,
            owner,
            value,
            frozen,
            parent,
            explanation,
            store.plan_snapshot(sid, owner),
        )
        return saved_revision(
            store, sid, "f" * 32, value, frozen, store.plan_snapshot(sid, owner), parent
        )

    monkeypatch.setattr("forge_mind.workspace.document_service.build_document", build)
    initial_state = service._run_agent_job(executed)
    assert initial_state != "error", store.get(sid, "owner")["events"][-4:]
    assert initial_calls == []
    assert construction_finalization_allowed(store, sid, "owner", executed["id"])
    store.finish(executed["id"], sid, initial_state)
    published = deepcopy(store.session_record(sid, "owner")["customer_proposal"]["execution"])
    selected_version = store.session_record(sid, "owner")["selection_version"]
    next_jid = store.enqueue(
        sid, "owner", {"resume": True, "preview_only": False, "max_actions": 1, "max_seconds": 50}
    )
    continued = store.claim()
    assert continued is not None and continued["id"] == next_jid != executed["id"]
    calls = []
    systems = []
    from forge_mind.workspace.delivery_guidance import guidance_context

    guide = guidance_context(service, sid, "owner", published["revision_id"])

    class GuideClient:
        model = "synthetic-published-guide-test"
        timeout_s = None

        def generate_structured(self, **kwargs):
            calls.append(json.loads(kwargs["user"]))
            systems.append(kwargs["system"])
            if owner_changes_selection:
                store.update(sid, selected_revision=None)
            # One remaining action correctly selects the existing protected
            # instruction response, rather than an ordinary NativeAction.
            response = {
                "revision_id": published["revision_id"],
                "basis_hash": guide["basis_hash"],
                "reviewed_current_facts": True,
                "guidance": {
                    "print": "Synthetic lifecycle fixture; no printable files are qualified.",
                    "installation": "Installation remains unverified.",
                    "use_and_removal": "Use and removal remain unverified.",
                    "unresolved": "Physical fit and performance remain unverified.",
                },
                "instruction_plan": {
                    "fact_refs": guide["instruction_catalog"]["required_refs"],
                    "steps": [
                        {"action": "review_check", "target": "checks:summary"},
                        {"action": "review_physical_limits", "target": "physical:readiness"},
                    ],
                },
            }
            return kwargs["output_model"].model_validate(response), LLMUsage(self.model, 0, 0)

    service.client_factory = lambda **kwargs: GuideClient()
    monkeypatch.setattr(
        "forge_mind.workspace.document_service.build_document",
        lambda *a, **kw: pytest.fail("Known published construction must not run CAD again"),
    )
    state = service._run_agent_job(continued)
    assert len(calls) == 1
    assert calls[0]["author_workflow_context"]["status"] == "published"
    assert "When status=published, keep the approved" in systems[0]
    assert "design_plan and operation_plan unchanged" in systems[0]
    assert "Pending wording in the saved plan is historical" in systems[0]
    assert "fresh owner review before construction" in systems[0]
    assert calls[0]["author_workflow_context"]["execution"]["job_id"] == executed["id"]
    latest = store.session_record(sid, "owner")
    assert latest["customer_proposal"]["execution"] == published
    if owner_changes_selection:
        assert state == "error"
        assert latest["selected_revision"] is None
        assert latest["selection_version"] == selected_version + 1
        assert not construction_finalization_allowed(store, sid, "owner", next_jid)
    else:
        assert state != "error", store.get(sid, "owner")["events"][-4:]
        assert latest["selected_revision"] == published["revision_id"]
        assert latest["selection_version"] == selected_version
        assert (
            latest["delivery_guidance_records"][published["revision_id"]]["reviewed_current_facts"]
            is True
        )
        assert construction_finalization_allowed(store, sid, "owner", next_jid)
