"""Direct preview edits retain exact command approval; all worker output is synthetic."""

from copy import deepcopy

import pytest
from tests.unit.workspace.test_customer_approval import approval
from tests.unit.workspace.test_operation_plan import contract, document, plan
from tests.unit.workspace.test_revision_bound_edits import saved_revision

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.customer_approval import approve_proposal
from forge_mind.workspace.customer_construction import (
    construction_finalization_allowed,
    guard_document,
)
from forge_mind.workspace.document_service import run_document_job
from forge_mind.workspace.edit_service import enqueue_feature_edit
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.feature_delta import FeatureChange, FeatureDelta
from forge_mind.workspace.store import Store, WorkspaceError


def direct_edit(tmp_path, monkeypatch, *, requested=True):
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    monkeypatch.setattr("forge_mind.workspace.measure.engine_fingerprint", lambda: "f" * 64)
    store = Store(tmp_path)
    sid = store.create("owner", "Width 20 mm", contract().model_dump(), "native_operations")["id"]
    store.update(
        sid,
        project_kind="workspace",
        creation_request_id="a" * 32,
        approval_required=requested,
        customer_printable=False,
        customer_hosted_limits={"max_seconds": 80, "max_actions": 2},
    )
    target = store.set_design_plan(sid, "owner", plan(), 0, contract=contract())
    parent = "b" * 32
    saved_revision(store, sid, parent, document(), contract(), target)
    store.update(sid, selected_revision=parent)

    class NoModel:
        model = "synthetic-no-inference"
        timeout_s = None

        def generate_structured(self, **kwargs):
            pytest.fail("Approved exact feature replay must dispatch before inference")

    service = WorkspaceService(store, Executor("trusted_local"), lambda **_: NoModel())
    monkeypatch.setattr(service, "start", lambda: None)
    monkeypatch.setattr(
        "forge_mind.workspace.document_service.process_revision",
        lambda *a, **kw: pytest.fail("No native CAD may run"),
    )
    basis = store.edit_basis(sid, "owner")
    changed = document().features[0].model_copy(update={"name": "Reviewed plate feature"})
    delta = FeatureDelta(basis=basis, changes=[FeatureChange(update=[changed])])
    queued = enqueue_feature_edit(service, sid, "owner", delta, 80)
    job = store.claim()
    assert job["id"] == queued["job_id"] and job["preview_only"] is True
    return store, service, sid, parent, basis, job


def assert_no_active(store, sid):
    with store.connect() as con:
        assert (
            con.execute(
                "SELECT count(*) FROM jobs WHERE session=? AND state IN ('queued','running')",
                (sid,),
            ).fetchone()[0]
            == 0
        )


def test_direct_feature_pause_terminalizes_preview_lease_and_preserves_base(tmp_path, monkeypatch):
    store, service, sid, parent, basis, job = direct_edit(tmp_path, monkeypatch)
    run_document_job(service, job)
    state = store.session_record(sid, "owner")
    assert_no_active(store, sid)
    assert state["state"] == "needs_approval" and state["selected_revision"] == parent
    assert "pending_finish" not in state
    command = state["customer_proposal"]["command"]
    assert command["feature_basis"] == basis.model_dump()
    assert command["parent"] == parent and command["document"] == job["document"]
    assert state["customer_proposal"]["remaining_actions"] == 1
    assert 0 < state["customer_proposal"]["remaining_seconds"] <= 80
    assert len(store.revisions(sid)) == 1
    with pytest.raises(WorkspaceError, match="typed or engineering"):
        store.enqueue(sid, "owner", {"resume": True})


def test_direct_feature_approval_replays_through_original_publication_authority(
    tmp_path, monkeypatch
):
    store, service, sid, parent, basis, job = direct_edit(tmp_path, monkeypatch)
    run_document_job(service, job)
    expected = deepcopy(store.session_record(sid, "owner")["customer_proposal"]["command"])
    jid = approve_proposal(store, sid, "owner", approval(store, sid))
    resumed = store.claim()
    assert resumed["id"] == jid and resumed["max_actions"] == 1
    calls = []
    original_publish = store.publish_feature_edit

    def publish(sid, owner, current_job, saved_basis, rid, **kwargs):
        calls.append(saved_basis)
        return original_publish(sid, owner, current_job, saved_basis, rid, **kwargs)

    def build(service, sid, owner, value, frozen, source, explanation, timeout, cancelled):
        assert value.model_dump() == expected["document"] and source == parent
        guard_document(
            service, sid, owner, value, frozen, source, explanation, store.plan_snapshot(sid, owner)
        )
        return saved_revision(
            store, sid, "c" * 32, value, frozen, expected["target_snapshot"], source
        )

    monkeypatch.setattr(store, "publish_feature_edit", publish)
    monkeypatch.setattr("forge_mind.workspace.document_service.build_document", build)
    assert service._run_agent_job(resumed) != "error"
    assert calls == [basis.model_dump()]
    receipt = store.feature_publication(sid, "owner", jid)
    assert receipt["revision_id"] == "c" * 32 and receipt["basis"] == basis.model_dump()
    assert store.session_record(sid, "owner")["last_edit_result"]["base_revision_id"] == parent
    assert construction_finalization_allowed(store, sid, "owner", jid)
    assert len(store.revisions(sid)) == 2


@pytest.mark.parametrize("change", ["source", "selection", "engine", "owner"])
def test_direct_feature_approval_rejects_changed_authority(tmp_path, monkeypatch, change):
    store, service, sid, parent, _basis, job = direct_edit(tmp_path, monkeypatch)
    run_document_job(service, job)
    request = approval(store, sid)
    owner = "owner"
    if change == "source":
        store.update(sid, instructions=["Changed source"])
    elif change == "selection":
        store.update(sid, selected_revision=parent)
    elif change == "engine":
        monkeypatch.setattr(
            "forge_mind.workspace.engine_identity.engine_fingerprint", lambda: "changed"
        )
    else:
        owner = "another-owner"
    with pytest.raises(WorkspaceError):
        approve_proposal(store, sid, owner, request)
    assert_no_active(store, sid)
    assert len(store.revisions(sid)) == 1


def test_cancel_while_pausing_direct_feature_releases_lease_and_denies_approval(
    tmp_path, monkeypatch
):
    store, service, sid, parent, _basis, job = direct_edit(tmp_path, monkeypatch)
    original_finish = store.finish

    def finish(jid, sid, state, **kwargs):
        if state == "needs_approval":
            store.cancel(sid, "owner")
        return original_finish(jid, sid, state, **kwargs)

    monkeypatch.setattr(store, "finish", finish)
    run_document_job(service, job)
    assert_no_active(store, sid)
    assert store.session_record(sid, "owner")["state"] == "cancelled"
    assert store.session_record(sid, "owner")["selected_revision"] == parent
    with pytest.raises(WorkspaceError):
        approve_proposal(store, sid, "owner", approval(store, sid))


def test_unrequested_regular_feature_preview_keeps_its_existing_completion(tmp_path, monkeypatch):
    store, service, sid, parent, _basis, job = direct_edit(tmp_path, monkeypatch, requested=False)

    def build(
        service, sid, owner, value, frozen, source, explanation, timeout, cancelled, **kwargs
    ):
        return saved_revision(store, sid, "c" * 32, value, frozen, kwargs["plan_snapshot"], source)

    monkeypatch.setattr("forge_mind.workspace.document_service.build_document", build)
    run_document_job(service, job)
    assert_no_active(store, sid)
    current = store.session_record(sid, "owner")
    assert current["state"] == "needs_review" and current["selected_revision"] == parent
    assert current["proposal"]["revision_id"] == "c" * 32
    assert current.get("customer_proposal") is None
