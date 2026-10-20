"""Final digital proof and publication guards use saved bytes, never native CAD."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import time
from copy import deepcopy
from zipfile import ZipFile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.unit.workspace.test_delivery_guidance_sdk import connection
from tests.unit.workspace.test_project_delivery import RID, saved_project
from tests.unit.workspace.test_verification_admission import document as nominal_document

from forge_mind.workspace import completion, delivery_guidance, measure, native_finish
from forge_mind.workspace import requirement_verification as verification
from forge_mind.workspace.api import router
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.models import Contract, DesignPlan, Finding, Measurement, Requirement
from forge_mind.workspace.native_delivery import current_delivery_binding
from forge_mind.workspace.project_access import project_access_router, project_overview
from forge_mind.workspace.project_delivery import create_project_export, read_project_export
from forge_mind.workspace.store import WorkspaceError


def digital(deliverable="project_stl", *, identifier="files"):
    quote = {
        "project_stl": "STL",
        "project_step": "STEP",
        "editable_native": "editable native CAD",
        "prepared_print_package": "prepared print package",
        "bound_instruction_plan": "bound instructions",
    }[deliverable]
    return Requirement(
        id=identifier,
        text=quote,
        source_quote=quote,
        source="explicit",
        verification={
            "kind": "digital_delivery",
            "scope": "digital_artifacts",
            "deliverable": deliverable,
            "coverage": "all_made",
            "source_quote": quote,
        },
    )


def prepared(tmp_path, monkeypatch, *, physical=False, preview=False, target=None):
    service, sid, directory = saved_project(tmp_path, monkeypatch)
    target = target or Contract(requirements=[digital()])
    if physical:
        target.requirements.append(
            Requirement(id="holding", text="Hold one kilogram", source_quote="Hold one kilogram")
        )
    revision = service.store.revision(sid, RID)
    revision["contract"] = target.model_dump()
    revision["measurement"].update(
        contract_hash=target.fingerprint(),
        engine_fingerprint="fixture-export-engine",
        findings=[
            Finding(
                requirement_id=r.id,
                status="unverified",
                method="unsupported",
                reason="Old evidence remains unverified",
            ).model_dump()
            for r in target.requirements
        ],
    )
    with service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(revision), sid, RID),
        )
    bindings = [
        {"subject_id": r.verification.subject_id, "part_id": "base", "frame": "definition_local"}
        for r in target.requirements
        if hasattr(r.verification, "subject_id")
    ]
    plan = DesignPlan(
        objective="Exact saved digital files", verification_bindings=bindings
    ).model_dump()
    service.store.update(
        sid, contract=target.model_dump(), design_plan=plan, plan_hash=digest(plan), plan_version=1
    )
    monkeypatch.setattr(measure, "engine_fingerprint", lambda: "fixture-export-engine")
    monkeypatch.setattr(completion, "engine_fingerprint", lambda: "fixture-export-engine")
    jid = service.store.enqueue(sid, "owner", {"preview_only": preview, "max_seconds": 120})
    assert service.store.claim()["id"] == jid
    review = {
        "status": "ready_for_validation",
        "job_id": jid,
        "revision_id": RID,
        "plan_hash": digest(plan),
        "basis_hash": "b" * 64,
        "reasons": [],
    }
    service.store.update(sid, native_review=review)
    receipt = create_project_export(service, sid, "owner", RID, include_stl=True)
    content, readback = read_project_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert readback == receipt
    delivery = {
        **current_delivery_binding(service, sid, "owner"),
        "job_id": jid,
        "status": "delivered",
        "readback_verified": True,
        "receipt": receipt,
    }
    service.store.update(sid, automatic_delivery=delivery)
    # Author closure is independently tested by native_finish tests. These tests isolate
    # exact digital readback, immutable report bytes and the final publication transaction.
    guide = {"status": "current", "basis_hash": review["basis_hash"], "record": {}}
    monkeypatch.setattr(delivery_guidance, "guidance_context", lambda *a, **k: deepcopy(guide))
    monkeypatch.setattr(native_finish, "finish_context", lambda *a, **k: deepcopy(review))
    return service, sid, jid, target, delivery, content, directory


def prove(service, sid, jid):
    return verification.verify_contract(
        service, sid, "owner", jid, deadline=time.monotonic() + 30, cancelled=lambda: False
    )


def unsupported_contract():
    return Contract(
        requirements=[
            Requirement(id="material", text="Make the part from PLA", source_quote="PLA"),
            Requirement(id="fit", text="Fit my organiser", source_quote="Fit my organiser"),
        ]
    )


def test_all_unmeasured_status_report_is_complete_current_and_never_satisfied(
    tmp_path, monkeypatch
):
    service, sid, jid, target, _, _, _ = prepared(
        tmp_path, monkeypatch, target=unsupported_contract()
    )
    before = target.model_dump(mode="json")
    assert verification.has_verification(before) is False
    receipt = prove(service, sid, jid)
    service.store.finish(jid, sid, "contract_satisfied", verification_report=receipt)
    result = verification.read_verification(service.store, sid, "owner")
    assert result["status"] == "unverified" and result["requirements_satisfied"] is False
    assert result["physical_readiness"] == "unverified"
    assert [
        (row["requirement_id"], row["status"], row["critical"]) for row in result["results"]
    ] == [("material", "unverified", True), ("fit", "unverified", True)]
    assert all(row["criterion_hash"] is None for row in result["results"])
    saved = service.store.get(sid, "owner")
    assert saved["state"] == "needs_review" and saved["contract"] == before
    with pytest.raises(WorkspaceError, match="not found"):
        verification.read_verification(service.store, sid, "other-owner")
    service.store.update(sid, plan_version=saved["plan_version"] + 1)
    with pytest.raises(WorkspaceError, match="basis changed"):
        verification.read_verification(service.store, sid, "owner")


def test_all_unmeasured_report_rejects_rehashed_false_pass(tmp_path, monkeypatch):
    service, sid, jid, _, _, _, _ = prepared(tmp_path, monkeypatch, target=unsupported_contract())
    receipt = prove(service, sid, jid)
    path = verification._report_path(service.store, sid, receipt["report_id"])
    report = json.loads(path.read_text())
    report.update(status="pass", requirements_satisfied=True)
    for row in report["results"]:
        row["status"] = "pass"
    data = json.dumps(report).encode()
    path.write_bytes(data)
    receipt.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    service.store.finish(jid, sid, "contract_satisfied", verification_report=receipt)
    saved = service.store.get(sid, "owner")
    assert saved["state"] == "needs_review"
    assert saved.get("contract_verification_report") is None
    assert (
        "Unsupported requirement cannot acquire a final pass"
        in saved["contract_verification_error"]
    )


def test_run_job_records_all_unmeasured_requirement_status_without_typed_worker(
    tmp_path, monkeypatch
):
    from forge_mind.workspace import native_delivery

    service, sid, jid, _, delivery, _, _ = prepared(
        tmp_path, monkeypatch, target=unsupported_contract()
    )
    monkeypatch.setattr(service.store, "authoring_mode", lambda project: "native_operations")
    monkeypatch.setattr(service, "_run_agent_job", lambda job: "needs_review")
    monkeypatch.setattr(completion, "finalize_completion", lambda *args: None)
    monkeypatch.setattr(
        native_delivery,
        "capture_native_delivery",
        lambda *args: service.store.update(sid, automatic_delivery=deepcopy(delivery)),
    )
    monkeypatch.setattr(
        service.executor,
        "process_observation",
        lambda *args, **kwargs: pytest.fail("Unsupported duties cannot run a proof worker"),
    )
    service.run_job({"id": jid, "session": sid, "owner": "owner", "max_seconds": 120})
    report = verification.read_verification(service.store, sid, "owner")
    assert report["status"] == "unverified" and len(report["results"]) == 2
    assert report["requirements_satisfied"] is False


def test_exact_digital_proof_publishes_after_readback_without_rewriting_old_evidence(
    tmp_path, monkeypatch
):
    service, sid, jid, _, delivery, _, _ = prepared(tmp_path, monkeypatch)
    old = deepcopy(service.store.revision(sid, RID))
    captured = deepcopy(service.store.get(sid, "owner")["automatic_delivery"])
    receipt = prove(service, sid, jid)
    assert service.store.get(sid, "owner").get("contract_verification_report") is None
    service.store.finish(jid, sid, "needs_review", verification_report=receipt)
    session = service.store.get(sid, "owner")
    report = verification.read_contract_verification(service, sid, "owner")
    assert session["state"] == "contract_satisfied"
    assert session["accepted_revision"] is None
    assert report["status"] == "pass" and report["requirements_satisfied"] is True
    assert report["physical_readiness"] == "unverified"
    assert report["results"][0]["facts"]["made_quantities"] == {"base": 1, "cap": 2}
    assert service.store.revision(sid, RID) == old
    assert session["automatic_delivery"] == captured
    assert delivery["receipt"] == captured["receipt"]
    overview = project_overview(service.store, sid, "owner")
    assert overview["contract_verification"]["status"] == "pass"
    assert overview["contract_verification"]["requirements_satisfied"] is True
    assert overview["contract_verification"]["finding_counts"] == {
        "pass": 1,
        "fail": 0,
        "unverified": 0,
    }
    assert overview["selected"]["requirements_satisfied"] is False
    assert overview["selected"]["evidence_stage"] == "build_time_before_final_verification"
    package, _ = read_project_export(
        service, sid, "owner", delivery["receipt"]["package_id"], delivery["receipt"]["sha256"]
    )
    with ZipFile(io.BytesIO(package)) as archive:
        assert verification.REPORT_FILE not in archive.namelist()
        readme = archive.read("README.md").decode()
        assert "do not report the later final contract proof" in readme
        assert "no final pass or failure is claimed in this ZIP" in readme
        assert f"GET /workspace/sessions/{sid}/contract-verification" in readme
        assert "contract_satisfied" not in readme


def test_physical_conjunct_stays_unknown_and_cannot_publish_success(tmp_path, monkeypatch):
    service, sid, jid, _, _, _, _ = prepared(tmp_path, monkeypatch, physical=True)
    receipt = prove(service, sid, jid)
    service.store.finish(jid, sid, "contract_satisfied", verification_report=receipt)
    report = verification.read_contract_verification(service, sid, "owner")
    assert service.store.get(sid, "owner")["state"] == "needs_review"
    assert [r["status"] for r in report["results"]] == ["pass", "unverified"]
    assert report["requirements_satisfied"] is False


@pytest.mark.parametrize("field", ["authority_conflict_job_id", "incomplete_finish_job_id"])
def test_old_other_job_markers_remain_historical_and_do_not_stale_current_proof(
    tmp_path, monkeypatch, field
):
    service, sid, jid, _, _, _, _ = prepared(tmp_path, monkeypatch)
    service.store.update(sid, **{field: "old-job"})
    # Isolate the existing native finish cleanup without running a native builder.
    monkeypatch.setattr(service.store, "_native_author_job", lambda *args: True)
    receipt = prove(service, sid, jid)
    service.store.finish(jid, sid, "needs_review", verification_report=receipt)
    assert service.store.get(sid, "owner")["state"] == "contract_satisfied"
    assert service.store.get(sid, "owner")[field] == "old-job"
    assert verification.read_contract_verification(service, sid, "owner")["status"] == "pass"


def test_report_revision_must_equal_the_evaluated_current_basis_even_with_rehashed_metadata(
    tmp_path, monkeypatch
):
    service, sid, jid, _, _, _, _ = prepared(tmp_path, monkeypatch)
    receipt = prove(service, sid, jid)
    path = verification._report_path(service.store, sid, receipt["report_id"])
    report = json.loads(path.read_bytes())
    report["revision_id"] = "c" * 32
    data = json.dumps(report).encode()
    path.write_bytes(data)
    receipt.update(revision_id="c" * 32, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    service.store.finish(jid, sid, "contract_satisfied", verification_report=receipt)
    assert service.store.get(sid, "owner")["state"] == "needs_review"
    assert service.store.get(sid, "owner").get("contract_verification_report") is None


@pytest.mark.parametrize(
    "mutation",
    [
        "plan_version",
        "selection_version",
        "accepted",
        "source",
        "guide",
        "print",
        "engine",
        "artifact",
        "report",
        "receipt",
        "review",
        "cancel",
        "deadline",
    ],
)
def test_finish_atomic_guard_rejects_stale_tampered_cancelled_or_expired_proof(
    tmp_path, monkeypatch, mutation
):
    service, sid, jid, _, _, _, directory = prepared(tmp_path, monkeypatch)
    receipt = prove(service, sid, jid)
    if mutation == "plan_version":
        service.store.update(sid, plan_version=2)
    elif mutation == "selection_version":
        service.store.update(sid, selected_revision=RID)  # Restoring the same ID is still a write.
    elif mutation == "accepted":
        service.store.update(sid, accepted_revision=RID)
    elif mutation == "source":
        service.store.update(sid, instructions=["Changed owner source"])
    elif mutation == "guide":
        service.store.update(sid, delivery_guidance_record={"changed": True})
    elif mutation == "print":
        service.store.update(sid, print_setup_records={RID: {"changed": True}})
    elif mutation == "engine":
        monkeypatch.setattr(measure, "engine_fingerprint", lambda: "changed-engine")
    elif mutation == "artifact":
        (directory / "definition_base.step").write_bytes(b"changed")
    elif mutation == "report":
        verification._report_path(service.store, sid, receipt["report_id"]).write_bytes(b"changed")
    elif mutation == "receipt":
        receipt["sha256"] = "0" * 64
    elif mutation == "review":
        service.store.update(sid, native_review={"status": "ready_for_validation", "job_id": "old"})
    elif mutation == "cancel":
        service.store.cancel(sid, "owner")
    else:
        with service.store.connect() as con:
            con.execute("UPDATE jobs SET deadline=? WHERE id=?", (time.time() - 1, jid))
    service.store.finish(jid, sid, "contract_satisfied", verification_report=receipt)
    assert service.store.get(sid, "owner")["state"] != "contract_satisfied"
    assert service.store.get(sid, "owner").get("contract_verification_report") is None


@pytest.mark.parametrize("mutation", ["preview", "incomplete", "old_job", "missing_proof"])
def test_explicit_current_author_closure_and_nonpreview_proof_are_required(
    tmp_path, monkeypatch, mutation
):
    service, sid, jid, _, _, _, _ = prepared(tmp_path, monkeypatch, preview=mutation == "preview")
    if mutation == "incomplete":
        service.store.update(sid, incomplete_finish_job_id=jid)
    if mutation == "old_job":
        value = service.store.get(sid, "owner")["native_review"]
        service.store.update(sid, native_review={**value, "job_id": "old"})
    receipt = None if mutation == "missing_proof" else prove(service, sid, jid)
    service.store.finish(jid, sid, "contract_satisfied", verification_report=receipt)
    session = service.store.get(sid, "owner")
    assert session["state"] != "contract_satisfied"
    assert session.get("pending_finish") != "contract_satisfied"


@pytest.mark.parametrize("state", ["error", "budget_exhausted", "needs_input"])
def test_even_valid_proof_does_not_override_prior_terminal_failure(tmp_path, monkeypatch, state):
    service, sid, jid, _, _, _, _ = prepared(tmp_path, monkeypatch)
    receipt = prove(service, sid, jid)
    service.store.finish(jid, sid, state, verification_report=receipt)
    assert service.store.get(sid, "owner")["state"] == state
    overview = project_overview(service.store, sid, "owner")
    assert overview["state"] == state
    assert overview["current_completion_status"] == "unverified"


def test_current_reader_owner_isolation_corruption_and_staleness(tmp_path, monkeypatch):
    service, sid, jid, _, _, _, _ = prepared(tmp_path, monkeypatch)
    receipt = prove(service, sid, jid)
    service.store.finish(jid, sid, "needs_review", verification_report=receipt)
    app = FastAPI()
    app.include_router(router(service.store, service, lambda: "owner"))
    app.include_router(project_access_router(service.store, service, lambda: "owner"))
    with TestClient(app) as client:
        assert (
            client.get(f"/workspace/sessions/{sid}/contract-verification").json()["status"]
            == "pass"
        )
        overview = client.get(f"/workspace/sessions/{sid}/overview").json()
        assert overview["contract_verification"]["requirements_satisfied"] is True
        assert overview["selected"]["evidence_stage"] == "build_time_before_final_verification"
        other = service.store.create("someone-else", "Private", None)["id"]
        assert client.get(f"/workspace/sessions/{other}/contract-verification").status_code == 404
        service.store.update(sid, plan_version=2)
        assert client.get(f"/workspace/sessions/{sid}/contract-verification").status_code == 409
        assert (
            verification.verification_summary(service.store, sid, "owner")["status"] == "unverified"
        )
        assert (
            client.get(f"/workspace/sessions/{sid}/overview").json()["contract_verification"][
                "status"
            ]
            == "unverified"
        )
        stale = client.get(f"/workspace/sessions/{sid}/overview").json()
        assert stale["state"] == "needs_review"
        assert stale["job_outcome_state"] == "contract_satisfied"
        assert stale["current_completion_status"] == "unverified"
        assert service.store.session_record(sid, "owner")["state"] == "contract_satisfied"
        service.store.update(sid, plan_version=1)
        path = verification._report_path(service.store, sid, receipt["report_id"])
        path.write_bytes(b"changed")
        assert client.get(f"/workspace/sessions/{sid}/contract-verification").status_code == 409
    with pytest.raises(ValueError):
        verification.read_contract_verification(service, sid, "someone-else")


def test_digital_scope_does_not_substitute_missing_stl_step_native_or_print_package(
    tmp_path, monkeypatch
):
    service, sid, _, _, captured, content, _ = prepared(tmp_path, monkeypatch)
    document = DesignDocument.model_validate(service.store.revision(sid, RID)["document"])
    for kind in ("project_stl", "project_step", "editable_native"):
        result = verification.evaluate_digital_delivery(
            digital(kind),
            captured_delivery=captured,
            guidance_context={},
            document=document,
            package_content=content,
        )
        assert result["status"] == "pass"
    for kind in ("prepared_print_package", "bound_instruction_plan"):
        result = verification.evaluate_digital_delivery(
            digital(kind),
            captured_delivery=captured,
            guidance_context={},
            document=document,
            package_content=content,
        )
        assert result["status"] == "unverified"
    corrupted = {**captured, "readback_verified": False}
    assert (
        verification.evaluate_digital_delivery(
            digital(),
            captured_delivery=corrupted,
            guidance_context={},
            document=document,
            package_content=content,
        )["status"]
        == "unverified"
    )


def test_aggregate_rejects_duplicate_or_missing_results_and_requires_every_critical_clause():
    contract = Contract(
        requirements=[
            digital(),
            Requirement(id="physical", text="Physical use", source_quote="Physical use"),
        ]
    )
    results = [
        {"requirement_id": "files", "status": "pass"},
        {"requirement_id": "physical", "status": "unverified"},
    ]
    assert verification.aggregate(contract, results) == ("unverified", False)
    with pytest.raises(ValueError):
        verification.aggregate(contract, results[:1])
    with pytest.raises(ValueError):
        verification.aggregate(contract, [results[0], results[0]])
    assert verification.aggregate(contract, [results[0], {**results[1], "status": "fail"}]) == (
        "fail",
        False,
    )


def test_current_completion_projection_preserves_legacy_meaning_and_historical_job_outcome():
    legacy = {"state": "contract_satisfied", "contract": Contract().model_dump()}
    assert verification.completion_projection(legacy, {"status": "unavailable"}) == {
        "state": "contract_satisfied"
    }
    typed = {
        **legacy,
        "contract": Contract(requirements=[digital()]).model_dump(),
        "job_id": "job",
        "selected_revision": RID,
    }
    stale = verification.completion_projection(typed, {"status": "unverified"})
    assert stale["state"] == "needs_review" and stale["job_outcome_state"] == "contract_satisfied"
    assert typed["state"] == "contract_satisfied"
    valid = {"status": "pass", "requirements_satisfied": True, "job_id": "job", "revision_id": RID}
    assert verification.completion_projection(typed, valid)["state"] == "contract_satisfied"
    assert (
        verification.completion_projection(typed, {**valid, "job_id": "old"})["state"]
        == "needs_review"
    )


def test_original_deadline_and_cancellation_prevent_any_new_proof(tmp_path, monkeypatch):
    service, sid, jid, _, _, _, _ = prepared(tmp_path, monkeypatch)
    for deadline, cancelled in ((time.monotonic() - 1, False), (None, True)):
        with pytest.raises(ValueError, match=r"deadline|cancelled"):
            verification.verify_contract(
                service,
                sid,
                "owner",
                jid,
                deadline=deadline,
                cancelled=lambda value=cancelled: value,
            )


def test_generated_criteria_require_current_verified_source_attribution(tmp_path, monkeypatch):
    service, sid, _, contract, _, _, _ = prepared(tmp_path, monkeypatch)
    contract.requirements[0].source = "request"
    with pytest.raises(ValueError, match="attribution"):
        verification._intent_authority(service.store.get(sid, "owner"), contract)


def test_report_digest_does_not_hash_itself_or_enter_guidance_basis(tmp_path, monkeypatch):
    service, sid, jid, _, _, _, _ = prepared(tmp_path, monkeypatch)
    before = service.store.get(sid, "owner").get("delivery_guidance_record")
    receipt = prove(service, sid, jid)
    data = verification._report_path(service.store, sid, receipt["report_id"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == receipt["sha256"]
    assert "sha256" not in json.loads(data)
    assert service.store.get(sid, "owner").get("delivery_guidance_record") == before


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_sdk_current_proof_reader_is_owned_and_reports_stale_or_tampered_followup(
    tmp_path, monkeypatch, surface
):
    service, sid, jid, _, _, _, _ = prepared(tmp_path, monkeypatch)
    receipt = prove(service, sid, jid)
    service.store.finish(jid, sid, "needs_review", verification_report=receipt)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            actual = await sdk.call_tool("get_contract_verification", {"project_id": sid})
            assert (
                not actual.is_error and actual.structured_content["requirements_satisfied"] is True
            )
            overview = await sdk.call_tool("get_project", {"project_id": sid})
            assert overview.structured_content["contract_verification"]["status"] == "pass"
            service.store.update(sid, plan_version=2)
            stale = await sdk.call_tool("get_contract_verification", {"project_id": sid})
            assert stale.is_error
            overview = await sdk.call_tool("get_project", {"project_id": sid})
            assert overview.structured_content["contract_verification"]["status"] == "unverified"
            assert overview.structured_content["state"] == "needs_review"
            assert overview.structured_content["job_outcome_state"] == "contract_satisfied"
            service.store.update(sid, plan_version=1)
            verification._report_path(service.store, sid, receipt["report_id"]).write_bytes(
                b"changed"
            )
            corrupt = await sdk.call_tool("get_contract_verification", {"project_id": sid})
            assert corrupt.is_error
            overview = await sdk.call_tool("get_project", {"project_id": sid})
            assert overview.structured_content["state"] == "needs_review"
            assert overview.structured_content["job_outcome_state"] == "contract_satisfied"
        async with connection(service, tmp_path, surface, foreign=True) as sdk:
            foreign = await sdk.call_tool("get_contract_verification", {"project_id": sid})
            assert foreign.is_error

    asyncio.run(scenario())


def geometry_target(kind):
    criterion = {
        "version": 1,
        "kind": kind,
        "scope": "nominal_cad",
        "subject_id": "plate",
        "subject_role": "designed",
        "frame": "definition_local",
        "axis": "z",
        "count_policy": "exact",
        "holes": [
            {
                "center_mm": [0.0, 0.0],
                "diameter_mm": 4.0,
                "diameter_tolerance_mm": 0.1,
                "position_tolerance_mm": 0.1,
            }
        ],
    }
    if kind == "nominal_perforated_box":
        criterion.update(size_mm=[10.0, 10.0, 10.0], center_mm=[0.0, 0.0, 0.0])
    quote = json.dumps(criterion)
    criterion["source_quote"] = quote
    return Requirement(
        id="nominal",
        text="Nominal CAD geometry",
        source_quote=quote,
        source="explicit",
        verification=criterion,
    )


@pytest.mark.parametrize("kind", ["nominal_through_hole_pattern", "nominal_perforated_box"])
@pytest.mark.parametrize("status", ["pass", "fail", "unverified"])
def test_author_and_final_use_the_same_frozen_geometry_worker_without_authority_mutation(
    tmp_path, monkeypatch, kind, status
):
    target = Contract(requirements=[geometry_target(kind), digital()])
    service, sid, jid, _, _, _, directory = prepared(tmp_path, monkeypatch, target=target)
    monkeypatch.setattr(service.executor, "mode", "trusted_local")
    seen = []
    expected_hash = hashlib.sha256((directory / "definition_base.step").read_bytes()).hexdigest()
    assert expected_hash != hashlib.sha256((directory / "part.step").read_bytes()).hexdigest()

    def worker(scratch, payload, timeout, cancelled):
        request = payload["request"]
        assert (scratch / "part.step").read_bytes() == (
            directory / "definition_base.step"
        ).read_bytes()
        assert timeout is not None and 0 < timeout <= 30 and cancelled() is False
        assert (
            request["holes"] == target.requirements[0].verification.model_dump(mode="json")["holes"]
        )
        assert request["part_id"] == "base" and request["frame"] == "definition_local"
        expected = (
            "nominal_perforated_box"
            if kind == "nominal_perforated_box"
            else "nominal_through_pattern"
        )
        assert payload["operation"] == expected
        seen.append(payload)
        return {
            "status": status,
            "reason": "Reliably wrong saved geometry (fixture result)",
            "method": request["protocol"],
            "criterion_hash": request["criterion_hash"],
            "facts": {
                "step_sha256": hashlib.sha256((scratch / "part.step").read_bytes()).hexdigest()
            },
            "evidence_refs": [{"artifact": "part.step", "sha256": expected_hash}],
        }

    monkeypatch.setattr(verification, "process_revision", worker)
    before = deepcopy(service.store.session_record(sid, "owner"))
    inspected = verification.inspect_requirements(
        service, sid, "owner", RID, deadline=time.monotonic() + 30, cancelled=lambda: False
    )
    assert [r["status"] for r in inspected["results"]] == [status, "unverified"]
    assert service.store.session_record(sid, "owner") == before
    receipt = prove(service, sid, jid)
    service.store.finish(jid, sid, "needs_review", verification_report=receipt)
    final = verification.read_contract_verification(service, sid, "owner")
    assert [r["status"] for r in final["results"]] == [status, "pass"]
    for current in (inspected, final):
        assert current["results"][0]["evidence_refs"] == [
            {
                "kind": "saved_step",
                "revision_id": RID,
                "artifact": "definition_base.step",
                "sha256": expected_hash,
                "engine_fingerprint": "fixture-export-engine",
            }
        ]
    assert seen[0] == seen[1]
    assert final["requirements_satisfied"] is (status == "pass")


@pytest.mark.parametrize("kind", ["nominal_through_hole_pattern", "nominal_perforated_box"])
@pytest.mark.parametrize("status", ["pass", "fail", "unverified"])
def test_flat_geometry_caller_retains_canonical_part_step_identity(
    tmp_path, monkeypatch, kind, status
):
    target = Contract(requirements=[geometry_target(kind)])
    service, sid, _, _, _, _, directory = prepared(tmp_path, monkeypatch, target=target)
    monkeypatch.setattr(service.executor, "mode", "trusted_local")
    measured = Measurement.model_validate(service.store.revision(sid, RID)["measurement"])
    expected_hash = measured.artifact_hashes["part.step"]
    plan = DesignPlan(
        objective="Canonical flat artifact identity",
        verification_bindings=[
            {"subject_id": "plate", "part_id": "part", "frame": "definition_local"}
        ],
    )

    def worker(scratch, payload, timeout, cancelled):
        assert (scratch / "part.step").read_bytes() == (directory / "part.step").read_bytes()
        assert payload["request"]["part_id"] == "part"
        return {
            "status": status,
            "reason": "Projection fixture only",
            "method": payload["request"]["protocol"],
            "criterion_hash": payload["request"]["criterion_hash"],
            "facts": {"step_sha256": expected_hash},
            "evidence_refs": [{"artifact": "part.step", "sha256": expected_hash}],
        }

    monkeypatch.setattr(verification, "process_revision", worker)
    before = deepcopy(service.store.session_record(sid, "owner"))
    result = verification.evaluate_frozen_geometry(
        service,
        sid,
        RID,
        target.requirements[0],
        plan,
        nominal_document(assembly=False),
        measured,
        tmp_path / "flat-worker",
        deadline=time.monotonic() + 30,
        cancelled=lambda: False,
    )
    assert result["status"] == status and result["facts"]["step_sha256"] == expected_hash
    assert result["evidence_refs"] == [
        {
            "kind": "saved_step",
            "revision_id": RID,
            "artifact": "part.step",
            "sha256": expected_hash,
            "engine_fingerprint": measured.engine_fingerprint,
        }
    ]
    assert service.store.session_record(sid, "owner") == before


@pytest.mark.parametrize("kind", ["nominal_through_hole_pattern", "nominal_perforated_box"])
@pytest.mark.parametrize(
    "fault", ["name", "hash", "extra_fields", "non_list", "non_object", "duplicate"]
)
def test_geometry_caller_refuses_spoofed_worker_references_without_authority_mutation(
    tmp_path, monkeypatch, kind, fault
):
    target = Contract(requirements=[geometry_target(kind)])
    service, sid, _, _, _, _, directory = prepared(tmp_path, monkeypatch, target=target)
    monkeypatch.setattr(service.executor, "mode", "trusted_local")
    measured = Measurement.model_validate(service.store.revision(sid, RID)["measurement"])
    expected_hash = measured.artifact_hashes["definition_base.step"]
    reference = {"artifact": "part.step", "sha256": expected_hash}
    references = [reference]
    if fault == "name":
        reference["artifact"] = "definition_base.step"
    elif fault == "hash":
        reference["sha256"] = measured.artifact_hashes["part.step"]
        assert reference["sha256"] != expected_hash
    elif fault == "extra_fields":
        reference["kind"] = "saved_step"
    elif fault == "non_object":
        references = ["part.step"]
    elif fault == "duplicate":
        references = [reference, deepcopy(reference)]
    else:
        references = reference

    def worker(scratch, payload, timeout, cancelled):
        return {
            "status": "pass",
            "reason": "Projection fixture only",
            "method": payload["request"]["protocol"],
            "criterion_hash": payload["request"]["criterion_hash"],
            "facts": {"step_sha256": expected_hash},
            "evidence_refs": references,
        }

    monkeypatch.setattr(verification, "process_revision", worker)
    before = deepcopy(service.store.session_record(sid, "owner"))
    with pytest.raises(ValueError, match="worker evidence references"):
        verification.evaluate_frozen_geometry(
            service,
            sid,
            RID,
            target.requirements[0],
            DesignPlan.model_validate(before["design_plan"]),
            DesignDocument.model_validate_json((directory / "document.json").read_bytes()),
            measured,
            tmp_path / "spoofed-worker",
            deadline=time.monotonic() + 30,
            cancelled=lambda: False,
        )
    assert service.store.session_record(sid, "owner") == before


@pytest.mark.parametrize("fault", ["criterion", "step", "stale", "cancel"])
def test_author_requirement_inspection_refuses_mismatched_stale_or_cancelled_worker(
    tmp_path, monkeypatch, fault
):
    target = Contract(requirements=[geometry_target("nominal_through_hole_pattern")])
    service, sid, _, _, _, _, _ = prepared(tmp_path, monkeypatch, target=target)
    monkeypatch.setattr(service.executor, "mode", "trusted_local")
    stopped = False

    def worker(scratch, payload, timeout, cancelled):
        nonlocal stopped
        request = payload["request"]
        if fault == "stale":
            service.store.update(sid, plan_version=2)
        if fault == "cancel":
            stopped = True
        return {
            "status": "pass",
            "reason": "Fixture only",
            "method": request["protocol"],
            "criterion_hash": "0" * 64 if fault == "criterion" else request["criterion_hash"],
            "facts": {
                "step_sha256": "0" * 64
                if fault == "step"
                else hashlib.sha256((scratch / "part.step").read_bytes()).hexdigest()
            },
            "evidence_refs": [],
        }

    monkeypatch.setattr(verification, "process_revision", worker)
    contract = deepcopy(service.store.get(sid, "owner")["contract"])
    with pytest.raises(ValueError):
        verification.inspect_requirements(
            service, sid, "owner", RID, deadline=time.monotonic() + 30, cancelled=lambda: stopped
        )
    assert service.store.get(sid, "owner")["contract"] == contract
