"""Real terminal transaction/readback with synthetic bytes, without CAD or a model."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest

from forge_mind.workspace import requirement_verification as verification
from forge_mind.workspace.customer_interpretation import customer_nominal_conformance
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.intent_attribution import IntentAttribution
from forge_mind.workspace.models import Contract, DesignPlan, Finding, Measurement, Requirement
from forge_mind.workspace.store import Store, WorkspaceError
from forge_mind.workspace.verification_admission import criterion_hash

ENGINE = "synthetic-byte-fixture-engine"
MARKERS = ("incomplete_finish_job_id", "authority_conflict_job_id")


def prepare(tmp_path, monkeypatch, marker=None, *, physical=True):
    # Replace only the native measurement import seam. No current_basis,
    # checked_report, final transaction, authority or current reader is mocked.
    measure = ModuleType("forge_mind.workspace.measure")
    measure.engine_fingerprint = lambda: ENGINE

    def verify_bytes(directory, measured):
        return (
            all(
                (directory / name).is_file()
                and hashlib.sha256((directory / name).read_bytes()).hexdigest() == checksum
                for name, checksum in measured.artifact_hashes.items()
            )
            and {"program.py", "part.step", "part.stl"} <= measured.artifact_hashes.keys()
        )

    measure.verify_artifacts = verify_bytes
    monkeypatch.setitem(sys.modules, "forge_mind.workspace.measure", measure)
    target = Contract(
        requirements=[
            Requirement(
                id="files",
                text="STEP",
                source="explicit",
                source_quote="STEP",
                verification={
                    "kind": "digital_delivery",
                    "scope": "digital_artifacts",
                    "deliverable": "project_step",
                    "coverage": "all_made",
                    "source_quote": "STEP",
                },
            ),
            *(
                [
                    Requirement(
                        id="fit", text="Fit the mating part", source_quote="Fit the mating part"
                    )
                ]
                if physical
                else []
            ),
        ]
    )
    store = Store(tmp_path)
    sid = store.create(
        "owner",
        "Give me STEP. Fit the mating part.",
        target.model_dump(),
        authoring_mode="native_operations",
    )["id"]
    jid = store.enqueue(sid, "owner", {"max_seconds": 120, "preview_only": False})
    assert store.claim()["id"] == jid
    rid = "a" * 32
    document = DesignDocument(name="Synthetic saved-byte fixture").model_dump(mode="json")
    directory = store.directory(sid, rid)
    directory.mkdir(parents=True)
    for name in ("program.py", "part.step", "part.stl"):
        (directory / name).write_bytes(("Synthetic non-CAD fixture: " + name).encode())
    measured = Measurement(
        revision_id=rid,
        contract_hash=target.fingerprint(),
        engine_fingerprint=ENGINE,
        geometry_valid=True,
        artifact_hashes={
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("program.py", "part.step", "part.stl")
        },
        findings=[
            Finding(
                requirement_id=req.id,
                status="unverified",
                method="synthetic fixture",
                reason="No geometry claim",
            )
            for req in target.requirements
        ],
    )
    revision = {
        "id": rid,
        "contract": target.model_dump(),
        "measurement": measured.model_dump(),
        "document": document,
    }
    with store.connect() as con:
        con.execute("INSERT INTO revisions VALUES (?,?,?)", (rid, sid, json.dumps(revision)))
    plan = DesignPlan(objective="Exercise terminal publication only").model_dump(mode="json")
    review = {
        "job_id": jid,
        "revision_id": rid,
        "plan_hash": digest(plan),
        "status": "needs_work" if marker == "incomplete_finish_job_id" else "ready_for_validation",
        "reasons": ["Unfinished subsystem"] if marker == "incomplete_finish_job_id" else [],
    }
    store.update(
        sid,
        selected_revision=rid,
        accepted_revision=None,
        design_plan=plan,
        plan_hash=digest(plan),
        plan_version=1,
        plan_contract_hash=target.fingerprint(),
        native_review=review,
        automatic_delivery={
            "status": "delivered",
            "job_id": jid,
            "readback_verified": True,
            "revision_id": rid,
            "receipt": {"sha256": "f" * 64, "bytes": 32},
        },
        **({marker: jid} if marker else {}),
    )
    return SimpleNamespace(
        store=store, sid=sid, jid=jid, rid=rid, target=target, directory=directory
    )


def freeze_report(case):
    """Synthetic typed results exercise lifecycle checks, never geometry qualification."""
    doc = case.store.get(case.sid, "owner")
    with case.store.connect() as con:
        basis = verification.current_basis(case.store, con, case.sid, doc)
    results = [
        {
            "requirement_id": req.id,
            "critical": req.critical,
            "status": "pass" if req.verification else "unverified",
            "criterion_hash": criterion_hash(req.verification.model_dump(mode="json"))
            if req.verification
            else None,
            "method": verification.verification_method(req.verification),
            "reason": "Synthetic result for publication testing only",
            "facts": {},
            "evidence_refs": [],
        }
        for req in case.target.requirements
    ]
    status, satisfied = verification.aggregate(case.target, results)
    rid = "b" * 32
    review = doc["native_review"]
    closure = review["status"] == "ready_for_validation" and all(
        basis[field] != case.jid for field in MARKERS
    )
    report = {
        "version": 1,
        "report_id": rid,
        "project_id": case.sid,
        "job_id": case.jid,
        "revision_id": case.rid,
        "basis": basis,
        "status": status,
        "requirements_satisfied": satisfied,
        "author_closure_ready": closure,
        "results": results,
        "physical_readiness": "unverified",
        "scope": verification.SCOPE,
        "cad_conformance": customer_nominal_conformance(IntentAttribution(), results),
    }
    raw = json.dumps(report, sort_keys=True).encode()
    path = verification._report_path(case.store, case.sid, rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    receipt = {
        "report_id": rid,
        "job_id": case.jid,
        "revision_id": case.rid,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "basis_hash": digest(basis),
    }
    # Prove the SAME actual file is acceptable against the old live basis.
    with case.store.connect() as con:
        assert verification.checked_report(case.store, con, case.sid, doc, receipt) == report
    return receipt, report, raw, path


@pytest.mark.parametrize("marker", MARKERS)
def test_meaningful_negative_marker_proof_survives_real_finish_and_current_read(
    tmp_path, monkeypatch, marker
):
    case = prepare(tmp_path, monkeypatch, marker)
    receipt, report, raw, path = freeze_report(case)
    if marker == "authority_conflict_job_id":

        def forbidden(*args, **kwargs):
            pytest.fail("Authority conflict cannot checkpoint or publish a different baseline")

        monkeypatch.setattr(case.store, "_checkpoint", forbidden)
        monkeypatch.setattr(case.store, "_feature_publication", forbidden)
    case.store.finish(case.jid, case.sid, "needs_review", verification_report=receipt)
    saved = case.store.get(case.sid, "owner")
    current = verification.read_verification(case.store, case.sid, "owner")
    assert current == {**report, "receipt": receipt}
    assert saved[marker] == case.jid and saved["state"] == "needs_review"
    assert saved["accepted_revision"] is None and saved["selected_revision"] == case.rid
    assert current["author_closure_ready"] is False and current["requirements_satisfied"] is False
    assert current["physical_readiness"] == "unverified"
    assert current["results"][1]["status"] == "unverified"
    assert path.read_bytes() == raw
    if marker == "incomplete_finish_job_id":
        assert saved["native_review"]["status"] == "needs_work"
        assert saved["resume_candidate"]["revision_id"] == case.rid
        assert saved["resume_candidate"]["native_author_binding"]


def test_complete_finish_keeps_same_readable_proof_without_accepting_geometry(
    tmp_path, monkeypatch
):
    case = prepare(tmp_path, monkeypatch, physical=False)
    receipt, report, raw, path = freeze_report(case)
    case.store.finish(case.jid, case.sid, "needs_review", verification_report=receipt)
    assert verification.read_verification(case.store, case.sid, "owner") == {
        **report,
        "receipt": receipt,
    }
    saved = case.store.get(case.sid, "owner")
    assert saved["state"] == "contract_satisfied" and saved["accepted_revision"] is None
    assert path.read_bytes() == raw


@pytest.mark.parametrize("marker", MARKERS)
@pytest.mark.parametrize(
    "proof",
    [
        "absent",
        "rejected",
        "cancelled",
        "deadline",
        "deadline_crossing",
        "wrong_job",
        "missing_closure",
    ],
)
def test_unauthenticated_receipt_keeps_legacy_cleanup_and_never_installs_proof(
    tmp_path, monkeypatch, marker, proof
):
    case = prepare(tmp_path, monkeypatch, marker, physical=proof != "missing_closure")
    receipt, _, raw, path = freeze_report(case)
    checked_calls = []
    if proof == "absent":
        receipt = None
    elif proof == "rejected":
        receipt = {**receipt, "sha256": "0" * 64}
    elif proof == "wrong_job":
        receipt = {**receipt, "job_id": "other-job"}
    elif proof == "missing_closure":
        malformed = json.loads(raw)
        del malformed["author_closure_ready"]
        raw = json.dumps(malformed, sort_keys=True).encode()
        path.write_bytes(raw)
        receipt = {**receipt, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        # The existing reader admits this shape. Publication must still reject its
        # missing closure field without retaining markers as authenticated proof.
        with case.store.connect() as con:
            verification.checked_report(
                case.store, con, case.sid, case.store.session_record(case.sid, "owner"), receipt
            )
    elif proof == "deadline_crossing":
        actual_checked_report = verification.checked_report

        def read_then_cross_deadline(store, con, sid, document, proof_receipt):
            result = actual_checked_report(store, con, sid, document, proof_receipt)
            checked_calls.append(result)
            deadline = con.execute("SELECT deadline FROM jobs WHERE id=?", (case.jid,)).fetchone()[
                0
            ]
            monkeypatch.setattr(time, "time", lambda: deadline + 1)
            return result

        monkeypatch.setattr(verification, "checked_report", read_then_cross_deadline)
    else:
        with case.store.connect() as con:
            if proof == "cancelled":
                con.execute("UPDATE jobs SET cancel=1 WHERE id=?", (case.jid,))
            else:
                con.execute("UPDATE jobs SET deadline=? WHERE id=?", (time.time() - 1, case.jid))
    case.store.finish(case.jid, case.sid, "needs_review", verification_report=receipt)
    saved = case.store.get(case.sid, "owner")
    assert saved.get(marker) is None and saved.get("contract_verification_report") is None
    assert saved["state"] != "contract_satisfied" and saved["accepted_revision"] is None
    assert verification.read_verification(case.store, case.sid, "owner")["status"] == "unavailable"
    assert path.read_bytes() == raw
    if proof == "deadline_crossing":
        assert len(checked_calls) == 1
        assert saved["contract_verification_error"] == (
            "Original job deadline expired during final proof readback"
        )


@pytest.mark.parametrize("field", MARKERS)
def test_unrelated_job_finish_never_changes_current_job_markers(tmp_path, monkeypatch, field):
    case = prepare(tmp_path, monkeypatch, field)
    receipt, _, _, _ = freeze_report(case)
    case.store.update(case.sid, job_id="other-job", **{field: "other-job"})
    before = case.store.session_record(case.sid, "owner")
    case.store.finish(case.jid, case.sid, "needs_review", verification_report=receipt)
    assert case.store.session_record(case.sid, "owner") == before


@pytest.mark.parametrize("marker", MARKERS)
def test_new_job_invalidates_old_proof_and_treats_retained_marker_as_historical(
    tmp_path, monkeypatch, marker
):
    case = prepare(tmp_path, monkeypatch, marker, physical=False)
    receipt, _, _, _ = freeze_report(case)
    case.store.finish(case.jid, case.sid, "needs_review", verification_report=receipt)
    old_jid = case.jid
    payload = (
        {"resume": True, "max_seconds": 120}
        if marker == "incomplete_finish_job_id"
        else {
            "instruction": "Review the saved candidate.",
            "revision_id": case.rid,
            "max_seconds": 120,
        }
    )
    case.jid = case.store.enqueue(case.sid, "owner", payload)
    assert case.store.claim()["id"] == case.jid
    saved = case.store.get(case.sid, "owner")
    assert saved[marker] == old_jid
    with pytest.raises(WorkspaceError):
        verification.read_verification(case.store, case.sid, "owner")
    review = {
        **saved["native_review"],
        "job_id": case.jid,
        "status": "ready_for_validation",
        "reasons": [],
    }
    case.store.update(case.sid, native_review=review)
    fresh, report, _, _ = freeze_report(case)
    assert report["author_closure_ready"] is True
    case.store.finish(case.jid, case.sid, "needs_review", verification_report=fresh)
    assert verification.read_verification(case.store, case.sid, "owner")["status"] == "pass"
    assert case.store.get(case.sid, "owner")[marker] == old_jid


@pytest.mark.parametrize(
    "mutation",
    [
        "contract",
        "plan",
        "source",
        "input",
        "selection",
        "guide",
        "artifact",
        "review",
        "delivery",
        *MARKERS,
    ],
)
def test_genuine_postproof_mutation_still_refuses_current_read(tmp_path, monkeypatch, mutation):
    case = prepare(tmp_path, monkeypatch, "incomplete_finish_job_id")
    receipt, _, _, _ = freeze_report(case)
    case.store.finish(case.jid, case.sid, "needs_review", verification_report=receipt)
    # This successful read is essential: reject a new mutation, not the old finish bug.
    assert verification.read_verification(case.store, case.sid, "owner")["status"] == "unverified"
    saved = case.store.get(case.sid, "owner")
    if mutation == "contract":
        changed = deepcopy(saved["contract"])
        changed["requirements"][0]["text"] += " changed"
        case.store.update(case.sid, contract=changed)
    elif mutation == "plan":
        case.store.update(case.sid, plan_version=saved["plan_version"] + 1)
    elif mutation == "source":
        case.store.update(case.sid, prompt=saved["prompt"] + " Changed request.")
    elif mutation == "input":
        with case.store.connect() as con:
            con.execute("INSERT INTO engineering_inputs VALUES (?,?,?)", (case.sid, 1, "{}"))
    elif mutation == "selection":
        # Even restoring the same ID advances the server-owned selection version.
        case.store.update(case.sid, selected_revision=case.rid)
    elif mutation == "guide":
        case.store.update(case.sid, delivery_guidance_record={"changed": True})
    elif mutation == "artifact":
        (case.directory / "part.step").write_bytes(b"Changed bytes")
    elif mutation == "review":
        case.store.update(
            case.sid, native_review={**saved["native_review"], "reasons": ["Changed"]}
        )
    elif mutation == "delivery":
        case.store.update(case.sid, automatic_delivery={"status": "unavailable"})
    else:
        case.store.update(case.sid, **{mutation: "different-job"})
    with pytest.raises(WorkspaceError):
        verification.read_verification(case.store, case.sid, "owner")
