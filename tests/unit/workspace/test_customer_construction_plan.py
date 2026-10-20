"""A bound command cannot rewrite its approved target after publication."""

from copy import deepcopy

import pytest
from tests.unit.workspace.test_customer_construction import approved, published

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.customer_construction import (
    claim_construction,
    construction_finalization_allowed,
)
from forge_mind.workspace.edit_service import patch_plan
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, DesignPlan
from forge_mind.workspace.plan_patch import PlanPatch
from forge_mind.workspace.store import WorkspaceError


def current_published(tmp_path, monkeypatch):
    store, sid, _original, _rid = published(tmp_path, monkeypatch)
    jid = store.enqueue(sid, "owner", {"resume": True})
    assert store.claim()["id"] == jid
    assert claim_construction(store, sid, "owner", jid)["mode"] == "continue_published"
    state = store.session_record(sid, "owner")
    target = state["customer_proposal"]["command"]["target_snapshot"]
    return store, sid, jid, state, target


def test_bound_plan_update_rejects_before_any_persistence(tmp_path, monkeypatch):
    store, sid, jid, state, target = current_published(tmp_path, monkeypatch)
    updated = DesignPlan.model_validate(target["plan"]).model_copy(
        update={"verification": ["A newly authored check"]}
    )
    with pytest.raises(WorkspaceError, match="exact approved target plan"):
        store.set_design_plan_for_job(
            sid,
            "owner",
            jid,
            updated,
            state["plan_version"],
            Contract.model_validate(target["contract"]),
        )
    assert store.session_record(sid, "owner") == state
    assert construction_finalization_allowed(store, sid, "owner", jid)


def test_unchanged_bound_plan_is_idempotent_without_new_version(tmp_path, monkeypatch):
    store, sid, jid, state, target = current_published(tmp_path, monkeypatch)
    saved = store.set_design_plan_for_job(
        sid,
        "owner",
        jid,
        DesignPlan.model_validate(target["plan"]),
        state["plan_version"],
        Contract.model_validate(target["contract"]),
    )
    assert saved == target
    assert store.session_record(sid, "owner") == state
    assert construction_finalization_allowed(store, sid, "owner", jid)


def test_bound_patch_uses_same_plan_write_guard(tmp_path, monkeypatch):
    store, sid, jid, state, _target = current_published(tmp_path, monkeypatch)
    service = WorkspaceService(store, Executor("trusted_local"))
    patch = PlanPatch(basis=store.edit_basis(sid, "owner"), verification=["An extra scoped check"])
    with pytest.raises(WorkspaceError, match="exact approved target plan"):
        patch_plan(service, sid, "owner", patch, jid=jid, author_intent="Width 20 mm")
    assert store.session_record(sid, "owner") == state


def test_bound_plan_cannot_change_while_construction_is_claimed(tmp_path):
    store, _service, sid, jid = approved(tmp_path)
    claim_construction(store, sid, "owner", jid)
    state = deepcopy(store.session_record(sid, "owner"))
    target = state["customer_proposal"]["command"]["target_snapshot"]
    with pytest.raises(WorkspaceError, match="must publish"):
        store.set_design_plan_for_job(
            sid,
            "owner",
            jid,
            DesignPlan.model_validate(target["plan"]),
            state["plan_version"],
            Contract.model_validate(target["contract"]),
        )
    assert store.session_record(sid, "owner") == state


def test_new_explicit_owner_job_can_change_plan_without_old_command_authority(
    tmp_path, monkeypatch
):
    store, sid, old_jid, state, target = current_published(tmp_path, monkeypatch)
    store.finish(old_jid, sid, "error")
    jid = store.enqueue(sid, "owner", {"instruction": "Review another plan check"})
    job = store.claim()
    assert job["id"] == jid
    # Follow the existing owner-input application boundary before changing its plan.
    store.apply_input_contract(sid, "owner", jid, job["input_application_id"], target["contract"])
    updated = DesignPlan.model_validate(target["plan"]).model_copy(
        update={"verification": ["A newly requested plan check"]}
    )
    saved = store.set_design_plan_for_job(
        sid,
        "owner",
        jid,
        updated,
        state["plan_version"],
        Contract.model_validate(target["contract"]),
    )
    assert saved["plan"]["verification"] == ["A newly requested plan check"]
    assert saved["plan_version"] == state["plan_version"] + 1
