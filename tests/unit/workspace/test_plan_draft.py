"""Draft recovery never grants admitted intent or geometry authority."""

import hashlib
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError
from tests.unit.workspace.test_operation_authoring_transport import contract, plan

from forge_mind.workspace.models import DesignPlan, OperationFrame
from forge_mind.workspace.plan_draft import (
    PlanDraftCommit,
    PlanDraftPatch,
    PlanDraftRead,
    commit_draft,
    patch_draft,
    read_draft,
    retain_rejected_plan,
    summary,
)
from forge_mind.workspace.store import Store, WorkspaceError


@pytest.fixture
def drafts(tmp_path):
    store = Store(tmp_path)
    sid = store.create(
        "owner", "A block 10 mm wide", contract(), authoring_mode="native_operations"
    )["id"]
    return store, sid


def correction(meta, changes):
    return PlanDraftPatch.model_validate(
        {**{k: meta[k] for k in ("basis", "draft_version", "draft_hash")}, "changes": changes}
    )


def admission(meta):
    return PlanDraftCommit.model_validate(
        {k: meta[k] for k in ("basis", "draft_version", "draft_hash")}
    )


def current(store, sid):
    return {
        "basis": store.edit_basis(sid, "owner").model_dump(),
        "draft_version": 0,
        "draft_hash": None,
    }


def test_large_exact_rejected_plan_survives_restart_and_scoped_repair(drafts):
    store, sid = drafts
    rejected = plan()
    rejected["unresolved"] = [f"External qualification {i}: " + "☃" * 220 for i in range(70)]
    rejected["operation_plan"]["frames"][0]["part_id"] = "other_part"
    raw = json.dumps(rejected, ensure_ascii=False, indent=3)
    assert len(raw.encode()) > 43000
    before = store.session_record(sid, "owner")
    meta = retain_rejected_plan(
        store, sid, "owner", raw, store.edit_basis(sid, "owner"), error="invalid frame"
    )
    assert store.session_record(sid, "owner") == before
    assert not meta["plan_applied"] and not meta["cad_applied"]
    store = Store(store.root)
    assert summary(store, sid, "owner")["source_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    page = read_draft(store, sid, "owner", PlanDraftRead(collection="operation_frames", limit=1))
    frame = page["records"][0]
    frame["part_id"] = "main"
    edited = patch_draft(store, sid, "owner", correction(meta, {"operation_frames": [frame]}))
    assert edited["draft_version"] == meta["draft_version"] + 1
    assert store.session_record(sid, "owner") == before
    with store.connect() as con:
        versions = con.execute(
            "SELECT content,raw_json FROM plan_drafts ORDER BY version"
        ).fetchall()
    assert len(versions) == 2 and all(row["raw_json"] == raw for row in versions)
    expected = deepcopy(rejected)
    expected["operation_plan"]["frames"][0] = OperationFrame.model_validate(frame).model_dump(
        mode="json"
    )
    assert json.loads(versions[-1]["content"]) == expected
    result = commit_draft(store, sid, "owner", admission(edited))
    assert result["plan_applied"] and not result["cad_applied"]
    assert store.get(sid, "owner")["design_plan"] == DesignPlan.model_validate(expected).model_dump(
        mode="json"
    )
    assert not store.revisions(sid) and summary(store, sid, "owner") is None
    assert len(store.session_record(sid, "owner").get("design_plan_history", [])) == 0


def test_failed_commit_retains_exact_draft_and_diagnostics_without_plan(drafts):
    store, sid = drafts
    data = plan()
    data["operation_plan"]["steps"][0]["depends_on"] = ["missing"]
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(data), store.edit_basis(sid, "owner")
    )
    before = store.session_record(sid, "owner")
    with pytest.raises(ValueError, match="unknown dependency"):
        commit_draft(store, sid, "owner", admission(meta))
    after = summary(store, sid, "owner")
    assert after["draft_hash"] == meta["draft_hash"]
    assert "unknown dependency" in after["validation_error"]
    assert store.session_record(sid, "owner") == before
    assert store.revisions(sid) == []


def test_initial_fragments_and_complete_submission_have_same_canonical_plan(drafts):
    store, sid = drafts
    full = plan()
    first = patch_draft(
        store, sid, "owner", correction(current(store, sid), {"objective": full["objective"]})
    )
    with pytest.raises(ValueError, match="operation plan"):
        commit_draft(store, sid, "owner", admission(first))
    second = patch_draft(
        store,
        sid,
        "owner",
        correction(
            first,
            {
                "part_ids": full["operation_plan"]["part_ids"],
                "operation_frames": full["operation_plan"]["frames"],
                "operation_steps": full["operation_plan"]["steps"],
            },
        ),
    )
    result = commit_draft(store, sid, "owner", admission(second))
    assert result["plan_version"] == 1
    assert store.get(sid, "owner")["design_plan"] == DesignPlan.model_validate(full).model_dump(
        mode="json"
    )


@pytest.mark.parametrize("operation", ["read", "patch", "commit"])
def test_foreign_owner_cannot_read_or_change_draft(drafts, operation):
    store, sid = drafts
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(plan()), store.edit_basis(sid, "owner")
    )
    before = summary(store, sid, "owner")
    with pytest.raises(WorkspaceError, match="not found"):
        if operation == "read":
            read_draft(store, sid, "foreign", PlanDraftRead())
        elif operation == "patch":
            patch_draft(store, sid, "foreign", correction(meta, {"objective": "Changed"}))
        else:
            commit_draft(store, sid, "foreign", admission(meta))
    assert summary(store, sid, "owner") == before


@pytest.mark.parametrize("change", ["draft", "plan", "contract"])
@pytest.mark.parametrize("operation", ["patch", "commit"])
def test_stale_identity_refuses_correction_and_publication(drafts, change, operation):
    store, sid = drafts
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(plan()), store.edit_basis(sid, "owner")
    )
    if change == "draft":
        patch_draft(store, sid, "owner", correction(meta, {"objective": "New draft intent"}))
    elif change == "plan":
        store.set_design_plan(sid, "owner", plan(), expected_plan_version=0)
    else:
        changed = contract()
        changed["requirements"][0]["value"] = 11
        store.update(sid, contract=changed)
    before = store.session_record(sid, "owner")
    with pytest.raises(WorkspaceError, match="changed"):
        if operation == "patch":
            patch_draft(
                store, sid, "owner", correction(meta, {"objective": "Forbidden stale change"})
            )
        else:
            commit_draft(store, sid, "owner", admission(meta))
    assert store.session_record(sid, "owner") == before


def test_publication_rechecks_exact_draft_inside_atomic_canonical_save(drafts, monkeypatch):
    store, sid = drafts
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(plan()), store.edit_basis(sid, "owner")
    )
    setter = store._set_design_plan

    def raced(*args, **kwargs):
        patch_draft(store, sid, "owner", correction(meta, {"objective": "Racing draft change"}))
        return setter(*args, **kwargs)

    monkeypatch.setattr(store, "_set_design_plan", raced)
    with pytest.raises(WorkspaceError, match="changed before admission"):
        commit_draft(store, sid, "owner", admission(meta))
    assert not store.get(sid, "owner").get("design_plan")


def test_draft_cannot_remove_admitted_operation_obligations(drafts):
    store, sid = drafts
    store.set_design_plan(sid, "owner", plan(), expected_plan_version=0)
    data = plan()
    data["operation_plan"]["steps"][0]["feature_ids"] = []
    data["operation_plan"]["steps"][0]["kind"] = "inspection"
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(data), store.edit_basis(sid, "owner")
    )
    before = store.get(sid, "owner")["design_plan"]
    with pytest.raises(ValueError, match="cannot change kind"):
        commit_draft(store, sid, "owner", admission(meta))
    assert store.get(sid, "owner")["design_plan"] == before


@pytest.mark.parametrize("value", [0, False, 51, "20"])
def test_draft_read_limits_are_strict(value):
    with pytest.raises(ValidationError):
        PlanDraftRead(limit=value)


def test_ambiguous_rejected_ids_are_never_silently_merged(drafts):
    store, sid = drafts
    data = plan()
    data["operation_plan"]["steps"].append(deepcopy(data["operation_plan"]["steps"][0]))
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(data), store.edit_basis(sid, "owner")
    )
    with pytest.raises(WorkspaceError, match="unique"):
        patch_draft(
            store,
            sid,
            "owner",
            correction(meta, {"operation_steps": [data["operation_plan"]["steps"][0]]}),
        )
    assert summary(store, sid, "owner")["draft_hash"] == meta["draft_hash"]


@pytest.mark.parametrize("invalid", [None, "wrong", [{"missing_id": True}]])
def test_explicit_replacement_repairs_malformed_collections(drafts, invalid):
    store, sid = drafts
    data = plan()
    data["operation_plan"]["steps"] = invalid
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(data), store.edit_basis(sid, "owner")
    )
    request = correction(meta, {"operation_steps": plan()["operation_plan"]["steps"]})
    request.replace_collections = ["operation_steps"]
    fixed = patch_draft(store, sid, "owner", request)
    assert commit_draft(store, sid, "owner", admission(fixed))["plan_applied"]


def test_explicit_whole_operation_replacement_can_recover_malformed_object(drafts):
    store, sid = drafts
    data = plan()
    data["operation_plan"] = "wrong object"
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(data), store.edit_basis(sid, "owner")
    )
    operations = plan()["operation_plan"]
    request = correction(
        meta,
        {
            "part_ids": operations["part_ids"],
            "operation_frames": operations["frames"],
            "operation_steps": operations["steps"],
        },
    )
    request.replace_operation_plan = True
    fixed = patch_draft(store, sid, "owner", request)
    assert commit_draft(store, sid, "owner", admission(fixed))["plan_applied"]


def test_automatic_summary_bounds_invalid_objective_and_nested_diagnostics(drafts):
    store, sid = drafts
    data = plan()
    data["objective"] = "☃" * 50000
    data["functional_interfaces"] = [
        {
            "id": "huge",
            "participant_paths": ["one"],
            "installed_occurrence_paths": [f"missing_{i}" for i in range(10000)],
        }
    ]
    raw = json.dumps(data, ensure_ascii=False)
    meta = retain_rejected_plan(store, sid, "owner", raw, store.edit_basis(sid, "owner"))
    assert len(json.dumps(meta, ensure_ascii=False).encode()) < 32000
    assert meta["omitted_conflicts"] > 0
    explicit = read_draft(store, sid, "owner", PlanDraftRead(collection="diagnostics", limit=1))
    assert explicit["total"] == meta["conflict_count"]
    assert summary(store, sid, "owner")["source_sha256"] == hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize("blocked", ["queued", "cancelled", "foreign_worker"])
@pytest.mark.parametrize("operation", ["patch", "commit"])
def test_running_author_ownership_is_required_for_draft_mutations(drafts, blocked, operation):
    store, sid = drafts
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(plan()), store.edit_basis(sid, "owner")
    )
    jid = store.enqueue(sid, "owner", {"max_actions": 2})
    if blocked != "queued":
        assert store.claim() is not None
    if blocked == "cancelled":
        store.cancel(sid, "owner")
    elif blocked == "foreign_worker":
        with store.connect() as con:
            con.execute("UPDATE jobs SET worker_pid=-1 WHERE id=?", (jid,))
    before = store.session_record(sid, "owner")
    with pytest.raises(WorkspaceError, match="owned running job"):
        if operation == "patch":
            patch_draft(store, sid, "owner", correction(meta, {"objective": "Blocked"}), jid)
        else:
            commit_draft(store, sid, "owner", admission(meta), jid)
    assert store.session_record(sid, "owner") == before


def test_unrelated_correction_retains_exact_admitted_measured_fact(drafts):
    store, sid = drafts
    data = plan()
    data["operation_plan"]["dimensions"] = [
        {
            "id": "measured_width",
            "value": 10,
            "source": "measured_reference",
            "source_quote": "Prior measured reference",
            "part_id": "main",
        }
    ]
    store.set_design_plan(sid, "owner", data, expected_plan_version=0)
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(data), store.edit_basis(sid, "owner")
    )
    jid = store.enqueue(sid, "owner", {"max_actions": 2})
    assert store.claim() is not None
    fixed = patch_draft(
        store, sid, "owner", correction(meta, {"objective": "Corrected prose"}), jid
    )
    result = commit_draft(store, sid, "owner", admission(fixed), jid)
    assert result["plan_version"] == 2
    assert (
        store.get(sid, "owner")["design_plan"]["operation_plan"]["dimensions"][0]["source"]
        == "measured_reference"
    )


def test_unknown_root_field_needs_explicit_removal_before_admission(drafts):
    store, sid = drafts
    data = plan()
    data["datum_conventions"] = "Unrecognized author intent"
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(data), store.edit_basis(sid, "owner")
    )
    with pytest.raises(ValueError, match="Extra inputs"):
        commit_draft(store, sid, "owner", admission(meta))
    request = correction(meta, {"objective": data["objective"]})
    request.remove_unknown_fields = ["datum_conventions"]
    fixed = patch_draft(store, sid, "owner", request)
    assert commit_draft(store, sid, "owner", admission(fixed))["plan_applied"]
    with store.connect() as con:
        original = con.execute(
            "SELECT raw_json FROM plan_drafts WHERE session=? ORDER BY version DESC LIMIT 1", (sid,)
        ).fetchone()[0]
    assert json.loads(original)["datum_conventions"] == data["datum_conventions"]


@pytest.mark.parametrize("field", ["objective", "operation_plan", "functional_interfaces"])
def test_unknown_field_removal_cannot_delete_canonical_intent(drafts, field):
    store, sid = drafts
    with pytest.raises(ValidationError, match="unknown top-level"):
        PlanDraftPatch.model_validate(
            {
                **current(store, sid),
                "changes": {"objective": "Plan"},
                "remove_unknown_fields": [field],
            }
        )


@pytest.mark.parametrize("field", ["missing_inputs", "unsupported_checks", "physical_validation"])
def test_draft_replacement_cannot_erase_admitted_external_obligations(drafts, field):
    store, sid = drafts
    data = plan()
    data["requirement_dispositions"] = [
        {"requirement_id": "width", field: ["Independent external obligation"]}
    ]
    store.set_design_plan(sid, "owner", data, expected_plan_version=0)
    meta = retain_rejected_plan(
        store, sid, "owner", json.dumps(data), store.edit_basis(sid, "owner")
    )
    request = correction(meta, {"requirement_dispositions": [{"requirement_id": "width"}]})
    request.replace_collections = ["requirement_dispositions"]
    edited = patch_draft(store, sid, "owner", request)
    before = store.get(sid, "owner")["design_plan"]
    with pytest.raises(ValueError, match="cannot remove requirement check obligations"):
        commit_draft(store, sid, "owner", admission(edited))
    assert store.get(sid, "owner")["design_plan"] == before
