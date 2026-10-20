"""Store lifecycle controls; native geometry, model calls and delivery are stubbed."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest
from tests.unit.workspace.test_candidate_contract import contract as candidate_contract
from tests.unit.workspace.test_operation_authoring_transport import (
    document as source_document,
)
from tests.unit.workspace.test_operation_authoring_transport import plan as source_plan
from tests.unit.workspace.test_scoped_authoring_transport import snapshot

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.document_service import finish_preview, run_document_job
from forge_mind.workspace.edit_service import enqueue_feature_edit, patch_plan
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.feature_delta import FeatureDelta
from forge_mind.workspace.models import Contract, DesignPlan, Measurement
from forge_mind.workspace.operation_plan import snapshot as plan_snapshot
from forge_mind.workspace.plan_patch import PlanPatch
from forge_mind.workspace.store import Store, WorkspaceError

ACCEPTED = "a" * 32
PROPOSED = "b" * 32
BUILT = "c" * 32
PRIOR_JOB = "d" * 32
_AUTHORITY = (
    "selected_revision",
    "accepted_revision",
    "contract",
    "instructions",
    "proposal",
    "resume_candidate",
    "design_plan",
    "plan_version",
    "plan_hash",
    "plan_contract_hash",
    "operation_plan_snapshot",
    "accepted_operation_plan_snapshot",
    "accepted_operation_plan_revision",
)


def authority(store: Store, sid: str) -> dict[str, Any]:
    state = store.session_record(sid, "owner")
    return deepcopy({key: state[key] for key in _AUTHORITY if key in state})


def saved_revision(store, sid, rid, document, contract, binding, parent=None):
    """Create hash-bound synthetic worker output, never a claimed native CAD result."""
    directory = store.directory(sid, rid)
    directory.mkdir(parents=True)
    files = {
        "document.json": document.model_dump_json(),
        "program.py": "# Synthetic lifecycle fixture; never executed.\n",
        "part.step": "Synthetic lifecycle bytes; not STEP geometry.\n",
        "part.stl": "Synthetic lifecycle bytes; not STL geometry.\n",
    }
    for name, contents in files.items():
        (directory / name).write_text(contents, encoding="utf-8")
    measurement = Measurement(
        revision_id=rid,
        contract_hash=contract.fingerprint(),
        engine_fingerprint="f" * 64,
        geometry_valid=True,
        artifact_hashes={name: file_sha256(directory / name) for name in files},
    )
    revision = {
        "id": rid,
        "parent_id": parent,
        "code": files["program.py"],
        "document": document.model_dump(),
        "contract": contract.model_dump(),
        "measurement": measurement.model_dump(),
        "operation_record": {
            "plan_snapshot": binding,
            "plan_hash": binding["plan_hash"],
            "plan_version": binding["plan_version"],
            "contract_hash": binding["contract_hash"],
        },
    }
    store.add_revision(sid, revision)
    return revision


@pytest.fixture
def edits(tmp_path, monkeypatch):
    service = WorkspaceService(
        Store(tmp_path),
        Executor("trusted_local"),
        client_factory=lambda **_: pytest.fail("Lifecycle test attempted a model call"),
    )
    starts = []
    monkeypatch.setattr(service, "start", lambda: starts.append(True))
    monkeypatch.setattr("forge_mind.workspace.measure.engine_fingerprint", lambda: "f" * 64)
    monkeypatch.setattr(
        "forge_mind.workspace.document_service.engine_fingerprint", lambda: "f" * 64
    )
    store = service.store
    baseline = Contract.model_validate(candidate_contract(60, unmeasured=False))
    target = Contract.model_validate(candidate_contract(80))
    sid = store.create(
        "owner",
        "Keep the proposed block editable",
        baseline.model_dump(),
        authoring_mode="native_operations",
    )["id"]
    # Bind already-accepted history when the baseline plan/application is admitted.
    store.update(sid, instructions=["Keep accepted instructions"])
    baseline_plan = DesignPlan.model_validate(source_plan())
    baseline_binding = store.set_design_plan(sid, "owner", baseline_plan, 0)
    document = DesignDocument.model_validate(source_document())
    document.features[0].params["x"] = 60
    saved_revision(store, sid, ACCEPTED, document, baseline, baseline_binding)
    proposed_document = document.model_copy(deep=True)
    proposed_document.features[0].params["x"] = 80
    target_plan = baseline_plan.model_copy(update={"objective": "Continue the proposed block"})
    target_binding = plan_snapshot(target_plan, 2, target)
    saved_revision(store, sid, PROPOSED, proposed_document, target, target_binding, ACCEPTED)
    store.update(
        sid,
        selected_revision=ACCEPTED,
        accepted_revision=ACCEPTED,
        contract=baseline.model_dump(),
        design_plan=baseline_binding["plan"],
        operation_plan_snapshot=baseline_binding,
        plan_hash=baseline_binding["plan_hash"],
        plan_contract_hash=baseline_binding["contract_hash"],
        plan_version=2,
        accepted_operation_plan_snapshot=baseline_binding,
        accepted_operation_plan_revision=ACCEPTED,
        proposal={
            "revision_id": PROPOSED,
            "based_on": ACCEPTED,
            "contract": target.model_dump(),
            "input_version": 0,
            "eligible": False,
            "operation_plan_snapshot": target_binding,
        },
        resume_candidate={
            "revision_id": PROPOSED,
            "based_on": ACCEPTED,
            "scope": "Existing unfinished work must survive admission",
        },
        last_edit_result={"action": "feature_edit", "job_id": PRIOR_JOB, "applied": True},
    )
    return service, sid, starts


def delta_for(service, sid, basis=None):
    basis = basis or service.store.edit_basis(sid, "owner")
    revision = service.store.revision(sid, basis.revision_id)
    replacement = deepcopy(revision["document"]["features"][0])
    replacement["params"]["y"] += 1
    return FeatureDelta.model_validate(
        {"basis": basis.model_dump(), "changes": [{"update": [replacement]}]}
    )


@pytest.mark.parametrize("preview", [False, True])
def test_author_basis_conflict_preserves_newer_authority_through_terminal_cleanup(edits, preview):
    service, sid, _ = edits
    jid = service.store.enqueue(sid, "owner", {"preview_only": preview, "max_seconds": 120})
    job = service.store.claim()
    assert job is not None
    service.store.update(
        sid,
        selected_revision=PROPOSED,
        resume_candidate={"revision_id": PROPOSED},
        instructions=["Newer owner intent"],
    )
    newer = authority(service.store, sid)
    service.store.update(sid, authority_conflict_job_id=jid)
    service.store.finish(jid, sid, "needs_review")
    if preview:
        service.store.complete_preview(sid, jid, job.get("based_on"), None)
    assert authority(service.store, sid) == newer
    assert service.store.session_record(sid, "owner")["state"] == "needs_review"


def admitted_job(edits):
    service, sid, _ = edits
    reply = enqueue_feature_edit(service, sid, "owner", delta_for(service, sid))
    job = service.store.claim()
    assert job is not None and job["id"] == reply["job_id"]
    return job


def built_candidate(edits, job):
    service, sid, _ = edits
    return saved_revision(
        service.store,
        sid,
        BUILT,
        DesignDocument.model_validate(job["document"]),
        Contract.model_validate(job["contract"]),
        job["plan_snapshot"],
        job["revision_id"],
    )


@pytest.mark.parametrize("operation", ["feature", "plan"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("revision_id", ACCEPTED),
        ("document_hash", "1" * 64),
        ("plan_version", 9),
        ("plan_hash", "2" * 64),
        ("contract_hash", "3" * 64),
        ("input_version", 9),
        ("source_fingerprint", "4" * 64),
    ],
)
def test_stale_edit_basis_cannot_change_authority_or_queue(edits, operation, field, value):
    service, sid, starts = edits
    basis = service.store.edit_basis(sid, "owner").model_copy(update={field: value})
    before = snapshot(service.store)
    with pytest.raises(WorkspaceError, match="Edit basis changed"):
        if operation == "feature":
            enqueue_feature_edit(service, sid, "owner", delta_for(service, sid, basis))
        else:
            patch_plan(service, sid, "owner", PlanPatch(basis=basis, verification=["Fresh check"]))
    assert snapshot(service.store) == before
    assert starts == []


def test_sequential_pending_proposal_plan_patches_preserve_the_first_change(edits):
    service, sid, _ = edits
    before = authority(service.store, sid)
    first = patch_plan(
        service,
        sid,
        "owner",
        PlanPatch(
            basis=service.store.edit_basis(sid, "owner"), verification=["First saved update"]
        ),
    )
    assert first["applied"] is True and first["cad_applied"] is False
    second = patch_plan(
        service,
        sid,
        "owner",
        PlanPatch(
            basis=service.store.edit_basis(sid, "owner"), assembly_sequence=["Second saved update"]
        ),
    )
    after = authority(service.store, sid)
    assert after["design_plan"]["verification"] == ["First saved update"]
    assert after["design_plan"]["assembly_sequence"] == ["Second saved update"]
    assert second["plan_version"] == first["plan_version"] + 1
    for key in ("selected_revision", "accepted_revision", "contract", "instructions", "proposal"):
        assert after[key] == before[key]
    assert service.store.edit_basis(sid, "owner").plan_hash == second["plan_hash"]
    assert (
        service.store.edit_basis(sid, "owner").contract_hash
        != Contract.model_validate(after["contract"]).fingerprint()
    )


def test_feature_admission_preserves_proposal_selection_and_resume_without_rollback_copy(edits):
    service, sid, starts = edits
    before = authority(service.store, sid)
    reply = enqueue_feature_edit(service, sid, "owner", delta_for(service, sid))
    assert reply["state"] == "queued" and reply["applied"] is False
    assert authority(service.store, sid) == before
    assert starts == [True]
    with service.store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (reply["job_id"],)).fetchone()[0]
        )
    assert payload["feature_edit_admission"] == service.store._native_author_binding(
        service.store.session_record(sid, "owner")
    )
    assert "feature_edit_original" not in payload
    assert payload["based_on"] == ACCEPTED
    assert payload["revision_id"] == PROPOSED
    assert payload["contract"] == before["proposal"]["contract"]
    assert payload["baseline_contract"] == before["contract"]


@pytest.mark.parametrize("claimed", [False, True])
def test_cancellation_before_publication_preserves_current_authority(edits, claimed):
    service, sid, _ = edits
    before = authority(service.store, sid)
    reply = enqueue_feature_edit(service, sid, "owner", delta_for(service, sid))
    job = service.store.claim() if claimed else None
    service.store.cancel(sid, "owner")
    if job is not None:
        candidate = built_candidate(edits, job)
        with pytest.raises(WorkspaceError, match="owned running job"):
            service.store.publish_feature_edit(
                sid, "owner", job["id"], job["feature_edit_basis"], candidate["id"]
            )
        service.store.finish(job["id"], sid, "cancelled")
        service.store.complete_preview(sid, job["id"], job["based_on"], None)
    assert authority(service.store, sid) == before
    assert Store(service.store.root).session_record(sid, "owner")["proposal"] == before["proposal"]
    with service.store.connect() as con:
        assert (
            con.execute("SELECT state FROM jobs WHERE id=?", (reply["job_id"],)).fetchone()[0]
            == "cancelled"
        )


def test_confirmed_worker_identity_loss_preserves_unpublished_authority(edits):
    service, sid, _ = edits
    before = authority(service.store, sid)
    job = admitted_job(edits)
    with service.store.connect() as con:
        con.execute("UPDATE jobs SET worker_created=-1 WHERE id=?", (job["id"],))
    assert service.store.claim() is None
    assert service.store.session_record(sid, "owner")["state"] == "interrupted"
    assert authority(service.store, sid) == before
    with pytest.raises(WorkspaceError, match="owned running job"):
        service.store.publish_feature_edit(
            sid, "owner", job["id"], job["feature_edit_basis"], BUILT
        )
    assert authority(service.store, sid) == before


def test_stale_publication_preserves_new_authority_through_failed_completion(edits):
    service, sid, _ = edits
    job = admitted_job(edits)
    candidate = built_candidate(edits, job)
    service.store.update(sid, selected_revision=PROPOSED)
    raced = authority(service.store, sid)
    with pytest.raises(WorkspaceError, match="authority changed"):
        service.store.publish_feature_edit(
            sid, "owner", job["id"], job["feature_edit_basis"], candidate["id"]
        )
    service.store.finish(job["id"], sid, "error")
    service.store.complete_preview(sid, job["id"], job["based_on"], None)
    assert authority(service.store, sid) == raced


def test_publication_receipt_names_its_exact_job_and_retains_accepted_baseline(edits):
    service, sid, _ = edits
    before = authority(service.store, sid)
    job = admitted_job(edits)
    candidate = built_candidate(edits, job)
    receipt = service.store.publish_feature_edit(
        sid, "owner", job["id"], job["feature_edit_basis"], candidate["id"]
    )
    assert receipt["job_id"] == job["id"]
    assert receipt["applied"] is True
    assert receipt["base_revision_id"] == PROPOSED and receipt["revision_id"] == BUILT
    state = service.store.session_record(sid, "owner")
    assert state["last_edit_result"] == receipt
    assert service.store._feature_applied(state, job["id"]) is True
    assert service.store._feature_applied(state, PRIOR_JOB) is False
    assert state["accepted_revision"] == before["accepted_revision"]


@pytest.mark.parametrize("artifact", ["program.py", "part.step", "part.stl", "document.json"])
def test_valid_geometry_flag_cannot_publish_changed_artifact_bytes(edits, artifact):
    service, sid, _ = edits
    job = admitted_job(edits)
    candidate = built_candidate(edits, job)
    path = service.store.directory(sid, candidate["id"]) / artifact
    path.write_text("Changed after the synthetic worker receipt", encoding="utf-8")
    before = snapshot(service.store)
    with pytest.raises(WorkspaceError, match=r"(?i)artifact|source|document"):
        service.store.publish_feature_edit(
            sid, "owner", job["id"], job["feature_edit_basis"], candidate["id"]
        )
    assert snapshot(service.store) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision_id", "e" * 32),
        ("engine_fingerprint", "e" * 64),
        ("contract_hash", "e" * 64),
    ],
)
def test_valid_geometry_flag_cannot_publish_foreign_measurement_binding(edits, field, value):
    service, sid, _ = edits
    job = admitted_job(edits)
    candidate = built_candidate(edits, job)
    candidate["measurement"][field] = value
    with service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE id=? AND session=?",
            (json.dumps(candidate), candidate["id"], sid),
        )
    before = snapshot(service.store)
    with pytest.raises(WorkspaceError):
        service.store.publish_feature_edit(
            sid, "owner", job["id"], job["feature_edit_basis"], candidate["id"]
        )
    assert snapshot(service.store) == before


def test_rehashed_source_bytes_must_still_match_the_admitted_document(edits):
    service, sid, _ = edits
    job = admitted_job(edits)
    candidate = built_candidate(edits, job)
    changed = DesignDocument.model_validate(candidate["document"])
    changed.features[0].params["y"] += 7
    path = service.store.directory(sid, candidate["id"]) / "document.json"
    path.write_text(changed.model_dump_json(), encoding="utf-8")
    candidate["measurement"]["artifact_hashes"]["document.json"] = file_sha256(path)
    with service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE id=? AND session=?",
            (json.dumps(candidate), candidate["id"], sid),
        )
    before = snapshot(service.store)
    with pytest.raises(WorkspaceError, match="source differs"):
        service.store.publish_feature_edit(
            sid, "owner", job["id"], job["feature_edit_basis"], candidate["id"]
        )
    assert snapshot(service.store) == before


def test_first_unaccepted_candidate_can_be_edited_without_adopting_it_at_admission(edits):
    service, sid, _ = edits
    proposal = service.store.session_record(sid, "owner")["proposal"]
    target_binding = proposal["operation_plan_snapshot"]
    service.store.update(
        sid,
        selected_revision=None,
        accepted_revision=None,
        contract=None,
        design_plan=target_binding["plan"],
        plan_hash=target_binding["plan_hash"],
        plan_contract_hash=target_binding["contract_hash"],
        operation_plan_snapshot=target_binding,
        accepted_operation_plan_snapshot=None,
        accepted_operation_plan_revision=None,
    )
    before = authority(service.store, sid)
    job = admitted_job(edits)
    assert job["based_on"] is None
    assert authority(service.store, sid) == before
    candidate = built_candidate(edits, job)
    receipt = service.store.publish_feature_edit(
        sid, "owner", job["id"], job["feature_edit_basis"], candidate["id"]
    )
    assert receipt["applied"] is True and receipt["job_id"] == job["id"]
    assert service.store.session_record(sid, "owner")["accepted_revision"] is None


def test_delivery_failure_cannot_relabel_a_committed_cad_edit_unapplied(edits, monkeypatch):
    service, sid, _ = edits
    job = admitted_job(edits)
    calls = []

    def build(_service, _sid, _owner, document, contract, parent, *_args, plan_snapshot=None):
        calls.append("build")
        assert parent == PROPOSED
        assert contract.model_dump() == job["contract"]
        return saved_revision(service.store, sid, BUILT, document, contract, plan_snapshot, parent)

    def fail_delivery(*_args, **_kwargs):
        calls.append("delivery")
        raise WorkspaceError("Owned delivery transport failed")

    monkeypatch.setattr("forge_mind.workspace.document_service.build_document", build)
    monkeypatch.setattr(
        "forge_mind.workspace.native_delivery.capture_native_delivery", fail_delivery
    )
    run_document_job(service, job)
    state = service.store.session_record(sid, "owner")
    receipt = state["last_edit_result"]
    assert calls == ["build", "delivery"]
    assert receipt["applied"] is True and receipt["job_id"] == job["id"]
    assert receipt["revision_id"] == BUILT
    assert "Owned delivery transport failed" in receipt["delivery_error"]
    assert state["state"] == "error"
    assert state["accepted_revision"] == ACCEPTED
    assert (state.get("proposal") or {}).get("revision_id", state["selected_revision"]) == BUILT
    assert service.store.revision(sid, BUILT)["document"] == job["document"]


def inline_publication(edits, preview=True):
    """Publish through an author job, which can execute several independent actions."""
    service, sid, _ = edits
    state = service.store.session_record(sid, "owner")
    binding = state["accepted_operation_plan_snapshot"]
    service.store.update(
        sid,
        proposal=None,
        resume_candidate=None,
        design_plan=binding["plan"],
        plan_hash=binding["plan_hash"],
        plan_contract_hash=binding["contract_hash"],
        operation_plan_snapshot=binding,
    )
    service.store.enqueue(
        sid, "owner", {"preview_only": preview, "revision_id": ACCEPTED, "max_seconds": 120}
    )
    job = service.store.claim()
    assert job is not None
    service.store.update(sid, state="running")
    basis = service.store.edit_basis(sid, "owner")
    document = DesignDocument.model_validate(service.store.revision(sid, ACCEPTED)["document"])
    document.features[0].params["y"] += 1
    candidate = saved_revision(
        service.store,
        sid,
        BUILT,
        document,
        Contract.model_validate(state["contract"]),
        binding,
        ACCEPTED,
    )
    receipt = service.store.publish_feature_edit(sid, "owner", job["id"], basis.model_dump(), BUILT)
    return job, candidate, receipt


def replace_action_receipt(edits, job, replacement):
    service, sid, _ = edits
    if replacement == "plan_patch":
        receipt = patch_plan(
            service,
            sid,
            "owner",
            PlanPatch(
                basis=service.store.edit_basis(sid, "owner"), verification=["Committed intent"]
            ),
            job["id"],
        )
    else:
        receipt = {
            "action": "feature_edit",
            "job_id": job["id"],
            "applied": False,
            "code": "rejected",
        }
        if replacement == "stale_receipt":
            receipt.update(job_id=PRIOR_JOB, applied=True, revision_id=ACCEPTED)
    service.store.update(sid, last_edit_result=receipt)
    return receipt


@pytest.mark.parametrize("replacement", ["plan_patch", "rejected_feature_edit", "stale_receipt"])
@pytest.mark.parametrize("terminal", ["cancelled", "error", "budget_exhausted"])
def test_published_cad_survives_later_action_receipts_and_interrupted_preview(
    edits, replacement, terminal
):
    service, sid, _ = edits
    job, _, _ = inline_publication(edits)
    receipt = replace_action_receipt(edits, job, replacement)
    service.store.finish(job["id"], sid, terminal)
    service.store.complete_preview(sid, job["id"], ACCEPTED, None)
    state = service.store.session_record(sid, "owner")
    assert state["proposal"]["revision_id"] == BUILT
    assert state["proposal"]["eligible"] is False
    assert state["proposal"]["protection_checks"] == []
    assert state["accepted_revision"] == state["selected_revision"] == ACCEPTED
    assert state["last_edit_result"] == receipt
    if replacement == "plan_patch":
        assert state["design_plan"]["verification"] == ["Committed intent"]
    assert (
        Store(service.store.root).feature_publication(sid, "owner", job["id"])["revision_id"]
        == BUILT
    )


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("replacement", ["plan_patch", "rejected_feature_edit"])
def test_worker_death_preserves_publication_and_independent_plan_intent(
    edits, preview, replacement
):
    service, sid, _ = edits
    job, candidate, _ = inline_publication(edits, preview)
    replace_action_receipt(edits, job, replacement)
    later = DesignDocument.model_validate(candidate["document"])
    later.features[0].params["y"] += 2
    saved_revision(
        service.store,
        sid,
        "e" * 32,
        later,
        Contract.model_validate(candidate["contract"]),
        candidate["operation_record"]["plan_snapshot"],
        BUILT,
    )
    service.store.update(sid, selected_revision="e" * 32)
    with service.store.connect() as con:
        con.execute("UPDATE jobs SET worker_created=-1 WHERE id=?", (job["id"],))
    assert service.store.claim() is None
    state = service.store.session_record(sid, "owner")
    assert state["state"] == "interrupted"
    assert (state.get("proposal") or {}).get("revision_id", state["selected_revision"]) == BUILT
    if preview:
        assert state["proposal"]["eligible"] is False
    else:
        assert state["resume_candidate"]["revision_id"] == BUILT
    if replacement == "plan_patch":
        assert state["design_plan"]["verification"] == ["Committed intent"]
        if not preview:
            assert state["resume_candidate"]["plan_snapshot"]["plan"]["verification"] == [
                "Committed intent"
            ]


@pytest.mark.parametrize("terminal", ["cancelled", "error", "budget_exhausted"])
def test_preview_reviews_only_published_candidate_after_a_later_unpublished_build(
    edits, monkeypatch, terminal
):
    service, sid, _ = edits
    job, candidate, _ = inline_publication(edits)
    later = DesignDocument.model_validate(candidate["document"])
    later.features[0].params["y"] += 2
    saved_revision(
        service.store,
        sid,
        "e" * 32,
        later,
        Contract.model_validate(candidate["contract"]),
        candidate["operation_record"]["plan_snapshot"],
        BUILT,
    )
    service.store.update(sid, selected_revision="e" * 32)
    inspected = []
    monkeypatch.setattr(
        "forge_mind.workspace.document_service.protection_results",
        lambda _service, _sid, revision, _inputs: inspected.append(revision["id"]) or [],
    )
    replace_action_receipt(edits, job, "rejected_feature_edit")
    service.store.finish(job["id"], sid, terminal)
    finish_preview(service, job, "e" * 32)
    state = service.store.session_record(sid, "owner")
    assert inspected == [BUILT]
    assert state["proposal"]["revision_id"] == BUILT


def test_cancellation_flag_selects_publication_even_with_stale_pending_state(edits, monkeypatch):
    service, sid, _ = edits
    job, candidate, _ = inline_publication(edits)
    later = DesignDocument.model_validate(candidate["document"])
    later.features[0].params["y"] += 2
    saved_revision(
        service.store,
        sid,
        "e" * 32,
        later,
        Contract.model_validate(candidate["contract"]),
        candidate["operation_record"]["plan_snapshot"],
        BUILT,
    )
    service.store.update(
        sid, selected_revision="e" * 32, state="running", pending_finish="needs_review"
    )
    service.store.cancel(sid, "owner")
    inspected = []
    monkeypatch.setattr(
        "forge_mind.workspace.document_service.protection_results",
        lambda _service, _sid, revision, _inputs: inspected.append(revision["id"]) or [],
    )
    finish_preview(service, job, "e" * 32)
    assert inspected == [BUILT]
    state = service.store.session_record(sid, "owner")
    assert state["state"] == "cancelled"
    assert state["proposal"]["revision_id"] == BUILT


def test_failed_protection_review_retains_publication_without_inventing_eligibility(
    edits, monkeypatch
):
    service, sid, _ = edits
    job, _, _ = inline_publication(edits)

    def unavailable(*_args):
        raise WorkspaceError("Protection review unavailable")

    monkeypatch.setattr("forge_mind.workspace.document_service.protection_results", unavailable)
    service.store.finish(job["id"], sid, "needs_review")
    finish_preview(service, job)
    state = service.store.session_record(sid, "owner")
    assert state["state"] == "error"
    assert state["proposal"]["revision_id"] == BUILT
    assert state["proposal"]["eligible"] is False
    assert state["proposal"]["protection_checks"] == []


def test_later_artifact_damage_does_not_erase_publication_or_authorize_acceptance(
    edits, monkeypatch
):
    service, sid, _ = edits
    job, _, _ = inline_publication(edits)
    (service.store.directory(sid, BUILT) / "part.step").write_text("Changed", encoding="utf-8")
    assert service.store.feature_publication(sid, "owner", job["id"])["revision_id"] == BUILT
    monkeypatch.setattr("forge_mind.workspace.document_service.protection_results", lambda *_: [])
    service.store.finish(job["id"], sid, "error")
    finish_preview(service, job)
    state = service.store.session_record(sid, "owner")
    assert state["proposal"]["revision_id"] == BUILT
    assert state["proposal"]["eligible"] is False


@pytest.mark.parametrize("terminal", ["cancelled", "error", "budget_exhausted"])
def test_nonpreview_terminal_checkpoint_uses_publication_not_later_unpublished_selection(
    edits, terminal
):
    service, sid, _ = edits
    job, candidate, _ = inline_publication(edits, preview=False)
    replace_action_receipt(edits, job, "plan_patch")
    later = DesignDocument.model_validate(candidate["document"])
    later.features[0].params["y"] += 2
    saved_revision(
        service.store,
        sid,
        "e" * 32,
        later,
        Contract.model_validate(candidate["contract"]),
        candidate["operation_record"]["plan_snapshot"],
        BUILT,
    )
    service.store.update(sid, selected_revision="e" * 32)
    service.store.finish(job["id"], sid, terminal)
    state = service.store.session_record(sid, "owner")
    assert state["selected_revision"] == state["resume_candidate"]["revision_id"] == BUILT
    assert state["design_plan"]["verification"] == ["Committed intent"]


def test_publication_records_are_server_owned_before_any_admission_change(edits):
    service, sid, _ = edits
    before = authority(service.store, sid)
    with pytest.raises(WorkspaceError, match="server-owned"):
        service.store.enqueue(sid, "owner", {"feature_edit_publication": {"applied": True}})
    assert authority(service.store, sid) == before


@pytest.mark.parametrize(
    "field",
    [
        "job_id",
        "session_id",
        "revision_id",
        "document_hash",
        "basis",
        "basis_plan",
        "source_revision",
        "contract_hash",
        "engine_fingerprint",
        "applied",
        "artifact_hashes",
    ],
)
def test_invalid_publication_binding_cannot_authorize_cleanup(edits, field):
    service, sid, _ = edits
    job, _, _ = inline_publication(edits)
    with service.store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
        )
        publication = payload["feature_edit_publication"]
        if field == "basis":
            publication[field]["document_hash"] = "0" * 64
        elif field == "basis_plan":
            publication["basis"]["plan_hash"] = "0" * 64
        elif field == "source_revision":
            publication["basis"]["revision_id"] = PROPOSED
        elif field == "applied":
            publication[field] = False
        elif field == "artifact_hashes":
            publication[field] = {}
        else:
            publication[field] = "0" * (
                64 if field in {"document_hash", "contract_hash", "engine_fingerprint"} else 32
            )
        con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), job["id"]))
    assert service.store.feature_publication(sid, "owner", job["id"]) is None
    assert (
        service.store._feature_applied(service.store.session_record(sid, "owner"), job["id"])
        is False
    )
    service.store.finish(job["id"], sid, "cancelled")
    service.store.complete_preview(sid, job["id"], ACCEPTED, None)
    state = service.store.session_record(sid, "owner")
    assert state["proposal"] is None and state["selected_revision"] == ACCEPTED


def test_prior_job_publication_and_exact_current_display_receipt_cannot_authorize_new_job(edits):
    service, sid, _ = edits
    old_job, _, _ = inline_publication(edits)
    with service.store.connect() as con:
        old_payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (old_job["id"],)).fetchone()[0]
        )
    service.store.finish(old_job["id"], sid, "needs_review")
    service.store.complete_preview(sid, old_job["id"], ACCEPTED, None)
    service.store.enqueue(sid, "owner", {"preview_only": True, "revision_id": ACCEPTED})
    job = service.store.claim()
    assert job is not None
    with service.store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
        )
        payload["feature_edit_publication"] = old_payload["feature_edit_publication"]
        con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), job["id"]))
    service.store.update(
        sid,
        last_edit_result={
            "action": "feature_edit",
            "job_id": job["id"],
            "applied": True,
            "revision_id": BUILT,
        },
    )
    assert service.store.feature_publication(sid, "owner", old_job["id"]) is None
    assert service.store.feature_publication(sid, "owner", job["id"]) is None
    service.store.finish(job["id"], sid, "cancelled")
    service.store.complete_preview(sid, job["id"], ACCEPTED, None)
    assert service.store.session_record(sid, "owner")["proposal"] is None


@pytest.mark.parametrize("worker_death", [False, True])
def test_publication_never_overrides_later_authority_conflict(edits, worker_death):
    service, sid, _ = edits
    job, _, _ = inline_publication(edits)
    service.store.update(
        sid,
        selected_revision=PROPOSED,
        instructions=["New authority"],
        resume_candidate={"revision_id": PROPOSED},
        authority_conflict_job_id=job["id"],
    )
    before = authority(service.store, sid)
    if worker_death:
        with service.store.connect() as con:
            con.execute("UPDATE jobs SET worker_created=-1 WHERE id=?", (job["id"],))
        assert service.store.claim() is None
    else:
        service.store.finish(job["id"], sid, "error")
        service.store.complete_preview(sid, job["id"], ACCEPTED, None)
    assert authority(service.store, sid) == before
