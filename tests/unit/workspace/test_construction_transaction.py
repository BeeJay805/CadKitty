"""Atomic publication controls with synthetic artifacts, never native CAD proof."""

import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_operation_authoring_transport import contract, document, plan
from tests.unit.workspace.test_revision_bound_edits import saved_revision

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.construction_transaction import build_construction, prepare_construction
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, DesignPlan
from forge_mind.workspace.plan_patch import PlanPatch
from forge_mind.workspace.store import Store, WorkspaceError

BASE = "a" * 32
RESULT = "b" * 32


@pytest.fixture
def construction(tmp_path, monkeypatch):
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    monkeypatch.setattr("forge_mind.workspace.measure.engine_fingerprint", lambda: "f" * 64)
    frozen = Contract.model_validate(contract())
    sid = service.store.create(
        "owner", "10 mm wide", frozen.model_dump(), authoring_mode="native_operations"
    )["id"]
    binding = service.store.set_design_plan(sid, "owner", DesignPlan.model_validate(plan()), 0)
    source = DesignDocument.model_validate(document())
    saved_revision(service.store, sid, BASE, source, frozen, binding)
    service.store.update(sid, selected_revision=BASE)
    service.store.enqueue(sid, "owner", {"preview_only": False})
    job = service.store.claim()
    assert job is not None
    return service, sid, job, source


def prepare(fixture):
    service, sid, job, source = fixture
    edited = source.model_copy(deep=True)
    edited.features[0].params["y"] += 1
    patch = PlanPatch(
        basis=service.store.edit_basis(sid, "owner"), verification=["Measure the edited width"]
    )
    return prepare_construction(
        service, sid, "owner", job["id"], edited, patch, author_intent="10 mm wide"
    )


def candidate(fixture, prepared):
    service, sid, _, _ = fixture
    revision = saved_revision(
        service.store,
        sid,
        RESULT,
        prepared.document,
        Contract.model_validate(prepared.successor_snapshot["contract"]),
        prepared.successor_snapshot,
        BASE,
    )
    directory = service.store.directory(sid, RESULT)
    for name, value in (
        ("operation-plan.json", prepared.successor_snapshot),
        ("operation-record.json", revision["operation_record"]),
    ):
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
        revision["measurement"]["artifact_hashes"][name] = file_sha256(directory / name)
    with service.store.connect() as con:
        con.execute("UPDATE revisions SET document=? WHERE id=?", (json.dumps(revision), RESULT))
    return revision


def canonical(fixture):
    service, sid, _, _ = fixture
    state = service.store.session_record(sid, "owner")
    return deepcopy(
        {
            key: state.get(key)
            for key in (
                "selected_revision",
                "design_plan",
                "plan_version",
                "plan_hash",
                "design_plan_history",
                "accepted_revision",
                "accepted_operation_plan_snapshot",
                "proposal",
            )
        }
    )


def test_prepare_does_not_publish_either_half(construction):
    before = canonical(construction)
    prepared = prepare(construction)
    assert canonical(construction) == before
    assert prepared.successor_snapshot["plan_version"] == before["plan_version"] + 1
    assert prepared.successor_snapshot["plan_hash"] != before["plan_hash"]


def test_single_publication_captures_successor_and_keeps_acceptance(construction):
    service, sid, job, _ = construction
    prepared = prepare(construction)
    candidate(construction, prepared)
    receipt = service.store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
    state = canonical(construction)
    assert receipt["applied"] and receipt["plan_applied"] and receipt["cad_applied"]
    assert state["selected_revision"] == RESULT
    assert state["plan_hash"] == prepared.successor_snapshot["plan_hash"]
    assert state["plan_version"] == 2 and len(state["design_plan_history"]) == 1
    assert state["accepted_revision"] is None
    assert service.store.feature_publication(sid, "owner", job["id"])["revision_id"] == RESULT
    with pytest.raises(WorkspaceError):
        service.store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
    assert canonical(construction) == state


@pytest.mark.parametrize(
    "change",
    ["plan", "selection", "cancel", "artifact", "source", "lease", "same_selection", "acceptance"],
)
def test_stale_or_invalid_candidate_never_publishes_half(construction, change):
    service, sid, job, _ = construction
    prepared = prepare(construction)
    candidate(construction, prepared)
    if change == "plan":
        service.store.update(sid, plan_version=9)
    elif change == "selection":
        service.store.update(sid, selected_revision=None)
    elif change == "cancel":
        service.store.cancel(sid, "owner")
    elif change == "artifact":
        (service.store.directory(sid, RESULT) / "part.step").write_text("tamper")
    elif change == "source":
        (service.store.directory(sid, BASE) / "document.json").write_text("tamper")
    elif change == "same_selection":
        service.store.update(sid, selected_revision=BASE)
    elif change == "acceptance":
        service.store.update(sid, accepted_revision=BASE)
    else:
        with service.store.connect() as con:
            con.execute("UPDATE jobs SET worker_created=-1 WHERE id=?", (job["id"],))
    before = canonical(construction)
    with pytest.raises(WorkspaceError):
        service.store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
    assert canonical(construction) == before


@pytest.mark.parametrize("terminal", ["cancelled", "error", "budget_exhausted"])
def test_committed_successor_survives_terminal_recovery(construction, terminal):
    service, sid, job, _ = construction
    prepared = prepare(construction)
    candidate(construction, prepared)
    service.store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
    service.store.update(sid, last_edit_result={"applied": False})
    service.store.finish(job["id"], sid, terminal)
    state = service.store.get(sid, "owner")
    assert state["selected_revision"] == state["resume_candidate"]["revision_id"] == RESULT
    assert state["resume_candidate"]["plan_snapshot"] == prepared.successor_snapshot
    assert state["plan_hash"] == prepared.successor_snapshot["plan_hash"]


def test_invalid_geometry_is_retained_without_publishing(construction, monkeypatch):
    service, sid, job, _ = construction
    prepared = prepare(construction)
    output = candidate(construction, prepared)
    output["measurement"]["geometry_valid"] = False
    monkeypatch.setattr(
        "forge_mind.workspace.document_service.build_document", lambda *a, **k: output
    )
    before = canonical(construction)
    receipt = build_construction(
        service,
        sid,
        "owner",
        job["id"],
        prepared,
        explanation="Test invalid candidate",
        timeout=1,
        cancelled=lambda: False,
    )
    assert not receipt["applied"] and not receipt["plan_applied"] and not receipt["cad_applied"]
    assert receipt["candidate_revision_id"] == RESULT
    assert canonical(construction) == before


def test_author_cannot_claim_a_measured_fact(construction):
    service, sid, job, source = construction
    patch = PlanPatch.model_validate(
        {
            "basis": service.store.edit_basis(sid, "owner").model_dump(),
            "dimensions": [
                {
                    "id": "fake",
                    "part_id": "main",
                    "binding_name": None,
                    "value": 4,
                    "source": "measured_reference",
                    "source_quote": "fake",
                }
            ],
        }
    )
    before = canonical(construction)
    with pytest.raises(WorkspaceError, match="measured reference"):
        prepare_construction(
            service, sid, "owner", job["id"], source, patch, author_intent="10 mm wide"
        )
    assert canonical(construction) == before


@pytest.mark.parametrize("preview", [False, True])
def test_dead_worker_recovers_committed_plan_pair(construction, preview):
    service, sid, job, _ = construction
    if preview:
        with service.store.connect() as con:
            payload = json.loads(
                con.execute("SELECT payload FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
            )
            payload["preview_only"] = True
            payload["baseline_plan_snapshot"] = service.store.plan_snapshot(sid, "owner")
            con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), job["id"]))
    prepared = prepare(construction)
    candidate(construction, prepared)
    service.store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
    with service.store.connect() as con:
        con.execute("UPDATE jobs SET worker_created=-1 WHERE id=?", (job["id"],))
    assert service.store.claim() is None
    state = service.store.get(sid, "owner")
    recovered = state["proposal"] if preview else state["resume_candidate"]
    assert recovered["revision_id"] == RESULT
    captured = (
        recovered.get("operation_plan_snapshot") if preview else recovered.get("plan_snapshot")
    )
    assert captured == prepared.successor_snapshot
    assert state.get("accepted_revision") is None


def test_explicit_restoration_after_commit_wins_over_old_publication(construction):
    service, sid, job, _ = construction
    prepared = prepare(construction)
    candidate(construction, prepared)
    service.store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
    service.store.update(sid, selected_revision=BASE)
    assert service.store.feature_publication(sid, "owner", job["id"]) is None
    service.store.finish(job["id"], sid, "cancelled")
    assert service.store.get(sid, "owner")["selected_revision"] == BASE


def test_saved_json_assembly_transforms_match_the_prepared_typed_document(construction):
    service, sid, job, source = construction
    assembled = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "main",
                        "name": "Existing block",
                        "features": [
                            feature.model_dump(mode="json") for feature in source.features
                        ],
                        "output": "body",
                    }
                ],
                "roots": [
                    {
                        "id": "block",
                        "definition": "main",
                        "position": [3, 4, 5],
                        "rotation": [0, 0, 90],
                    }
                ],
            },
        }
    )
    patch = PlanPatch(
        basis=service.store.edit_basis(sid, "owner"), verification=["Inspect assembly placement"]
    )
    prepared = prepare_construction(
        service,
        sid,
        "owner",
        job["id"],
        assembled,
        patch,
        author_intent="10 mm wide",
    )
    candidate(construction, prepared)
    receipt = service.store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
    assert receipt["applied"]
    assert service.store.revision(sid, RESULT)["document"] == assembled.model_dump(mode="json")


def test_restored_snapshot_version_is_distinct_from_live_plan_write_counter(construction):
    service, sid, job, _ = construction
    old_binding = service.store.plan_snapshot(sid, "owner")
    assert old_binding["plan_version"] == 1
    # Store._promote_plan intentionally retains a newer plan-write CAS counter
    # when selecting an older saved intent. Reproduce that valid restored state.
    service.store.update(sid, plan_version=3)
    prepared = prepare(construction)
    assert prepared.base_snapshot == old_binding
    assert prepared.basis.plan_version == 3
    assert prepared.successor_snapshot["plan_version"] == 4
    candidate(construction, prepared)
    receipt = service.store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
    assert receipt["plan_version"] == 4
    assert service.store.feature_publication(sid, "owner", job["id"])["revision_id"] == RESULT
