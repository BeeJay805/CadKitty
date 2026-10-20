"""Physical pilot software controls; temporary reports are not real observations."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError
from tests.unit.workspace.test_project_delivery import RID, saved_project

from forge_mind.workspace import delivery_guidance
from forge_mind.workspace.document import digest
from forge_mind.workspace.physical_pilot_models import PhysicalPilotOutcomeRequest
from forge_mind.workspace.physical_pilot_service import (
    prepare_physical_pilot,
    read_outcomes,
    record_outcome,
)
from forge_mind.workspace.store import WorkspaceError

TEXT = {
    "print": "Use the saved proposed face and the lab's approved material profile.",
    "installation": "Use the declared hardware and a matching driver and wrench. Check access.",
    "use_and_removal": "Support the object and follow the saved removal sequence.",
    "unresolved": "Physical fit, printer profile, wall anchors and strength remain unverified.",
}
CHECK_IDS = (
    "print_completion",
    "part_inspection",
    "hardware_fit",
    "assembly_fit",
    "removal",
    "wall_installation",
)


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    service, sid, source = saved_project(tmp_path, monkeypatch)
    service.store.update(sid, contract=service.store.revision(sid, RID)["contract"])
    monkeypatch.setattr(delivery_guidance, "engine_fingerprint", lambda: "fixture-engine")
    guide = delivery_guidance.guidance_context(service, sid, "owner", RID)
    delivery_guidance.update_guidance(
        service,
        sid,
        "owner",
        delivery_guidance.GuidanceUpdate(
            revision_id=RID,
            basis_hash=guide["basis_hash"],
            reviewed_current_facts=True,
            guidance=TEXT,
        ),
    )
    return service, sid, source


def snapshot(service, sid):
    with service.store.connect() as con:
        tables = {
            name: [tuple(row) for row in con.execute(f"SELECT * FROM {name}")]
            for name in ("sessions", "revisions", "jobs")
        }
    files = {path.name: path.read_bytes() for path in service.store.directory(sid, RID).iterdir()}
    return tables, files


def report(packet, **changes):
    raw = {
        "revision_id": RID,
        "protocol_hash": packet["protocol_hash"],
        "specimen_id": "software-test-specimen; not a real printed sample",
        "provenance": "physical_reported",
        "printer": {"name": "Fixture lab printer; model unconfirmed"},
        "material": {"name": "PLA; fixture owner declaration"},
        "hardware": [
            {
                "part_id": "bolt",
                "name": "Fixture bolt",
                "quantity": 1,
                "note": "Temporary software-control declaration, not acquired hardware",
            }
        ],
        "checks": [
            {
                "check_id": identifier,
                "status": "not_run",
                "observation": "Printing and physical checks are deferred",
            }
            for identifier in CHECK_IDS
        ],
        "acquisition_note": "Temporary test data only. No physical specimen or instrument acquisition.",
    }
    raw.update(changes)
    return PhysicalPilotOutcomeRequest.model_validate(raw)


def test_prepare_is_read_only_deferred_and_uses_actual_quantities(pilot):
    service, sid, _ = pilot
    before = snapshot(service, sid)
    packet = prepare_physical_pilot(service, sid, "owner", RID)
    assert packet["physical_status"] == "deferred"
    assert packet["physical_qualification"] == "unverified"
    assert all(row["status"] == "not_run" for row in packet["pending_observations"])
    assert {row["part_id"]: row["quantity"] for row in packet["protocol"]["made_parts"]} == {
        "base": 1,
        "cap": 2,
    }
    assert [(row["part_id"], row["quantity"]) for row in packet["protocol"]["hardware"]] == [
        ("bolt", 1)
    ]
    assert not any(row["part_id"] == "guide" for row in packet["protocol"]["made_parts"])
    assert packet["protocol"]["instructions"] == TEXT
    assert packet["protocol"]["binding"]["print_setup_hash"] is None
    assert packet["protocol"]["binding"]["guidance_record_hash"]
    assert "lab's approved printer setup" in packet["packet_markdown"]
    assert "No saved automatic print setup" in packet["packet_markdown"]
    assert "make/definition_base.stl" in packet["packet_markdown"]
    assert snapshot(service, sid) == before
    assert read_outcomes(service, sid, "owner", RID)["records"] == []
    assert service.worker is None


def test_unknown_with_measurement_remains_reported_and_unqualified(pilot):
    service, sid, _ = pilot
    packet = prepare_physical_pilot(service, sid, "owner", RID)
    raw = report(packet).model_dump()
    raw["checks"][1].update(
        status="unknown",
        observation="Fixture inconclusive reading; no real specimen exists",
        measurements=[
            {"name": "Fixture opening", "value": 10.0, "unit": "mm", "method": "Software fixture"}
        ],
    )
    saved = record_outcome(service, sid, "owner", PhysicalPilotOutcomeRequest.model_validate(raw))
    assert saved["physical_status"] == "reported"
    assert saved["physical_qualification"] == "unverified"
    assert saved["report"]["checks"][1]["status"] == "unknown"
    assert saved["report"]["checks"][1]["measurements"][0]["value"] == 10.0


def test_reported_failure_is_append_only_without_acceptance_or_physical_pass(pilot):
    service, sid, _ = pilot
    before = service.store.session_record(sid, "owner")
    before_snapshot = snapshot(service, sid)
    packet = prepare_physical_pilot(service, sid, "owner", RID)
    checks = report(packet).model_dump()["checks"]
    checks[1] = {
        "check_id": "part_inspection",
        "status": "observed_fail",
        "observation": "Fixture hole is blocked; do not force hardware",
        "measurements": [
            {
                "name": "hole diameter",
                "value": 3.7,
                "unit": "mm",
                "method": "Fixture caliper declaration",
            }
        ],
    }
    first = record_outcome(service, sid, "owner", report(packet, checks=checks))
    second = record_outcome(service, sid, "owner", report(packet))
    assert first["report"]["checks"][1]["status"] == "observed_fail"
    assert first["physical_status"] == "reported"
    assert second["physical_status"] == "not_run"
    assert first["physical_qualification"] == second["physical_qualification"] == "unverified"
    assert second["sequence"] == 2
    assert second["previous_receipt_sha256"] == first["receipt_sha256"]
    assert "no authenticated instrument acquisition" in first["acquisition_scope"]
    assert read_outcomes(service, sid, "owner", RID)["records"] == [first, second]
    after = service.store.session_record(sid, "owner")
    after.pop("physical_pilot_records")
    assert after == before
    after_snapshot = snapshot(service, sid)
    assert after_snapshot[0]["revisions"] == before_snapshot[0]["revisions"]
    assert after_snapshot[0]["jobs"] == before_snapshot[0]["jobs"] == []
    assert after_snapshot[1] == before_snapshot[1]


def test_all_owner_passes_remain_unqualified(pilot):
    service, sid, _ = pilot
    packet = prepare_physical_pilot(service, sid, "owner", RID)
    checks = [
        {
            "check_id": identifier,
            "status": "observed_pass",
            "observation": "Owner-reported fixture result, not instrument attestation",
        }
        for identifier in CHECK_IDS
    ]
    result = record_outcome(service, sid, "owner", report(packet, checks=checks))
    assert result["physical_qualification"] == "unverified"
    assert result["physical_status"] == "reported"
    assert "load rating" in result["scope"]
    assert service.store.session_record(sid, "owner")["accepted_revision"] is None


@pytest.mark.parametrize(
    "defect", ["foreign", "protocol", "guide", "engine", "artifact", "head", "hardware"]
)
def test_refuse_stale_unowned_or_unknown_reports_without_writes(pilot, monkeypatch, defect):
    service, sid, source = pilot
    packet = prepare_physical_pilot(service, sid, "owner", RID)
    request = report(packet)
    owner = "foreign" if defect == "foreign" else "owner"
    if defect == "protocol":
        request = request.model_copy(update={"protocol_hash": "0" * 64})
    elif defect == "guide":
        context = delivery_guidance.guidance_context(service, sid, "owner", RID)
        delivery_guidance.update_guidance(
            service,
            sid,
            "owner",
            delivery_guidance.GuidanceUpdate(
                revision_id=RID,
                basis_hash=context["basis_hash"],
                reviewed_current_facts=True,
                guidance={**TEXT, "installation": "New owner-reviewed instruction"},
            ),
        )
    elif defect == "engine":
        monkeypatch.setattr(delivery_guidance, "engine_fingerprint", lambda: "other-engine")
    elif defect == "artifact":
        (source / "definition_cap.stl").write_bytes(b"Changed after measurement")
    elif defect == "head":
        service.store.update(sid, proposal={"revision_id": "c" * 32})
    elif defect == "hardware":
        request = report(
            packet,
            hardware=[
                {"name": "Unknown", "quantity": 1, "part_id": "foreign_bolt", "note": "Unmapped"}
            ],
        )
    before = snapshot(service, sid)
    with pytest.raises(WorkspaceError):
        record_outcome(service, sid, owner, request)
    assert snapshot(service, sid) == before
    assert "physical_pilot_records" not in service.store.session_record(sid, "owner")


def test_past_reports_remain_readable_after_supersession(pilot):
    service, sid, _ = pilot
    packet = prepare_physical_pilot(service, sid, "owner", RID)
    saved = record_outcome(service, sid, "owner", report(packet))
    service.store.update(sid, proposal={"revision_id": "c" * 32})
    result = read_outcomes(service, sid, "owner", RID)
    assert result["historical"] is True
    assert result["records"] == [saved]
    assert result["record_context"][saved["id"]]["matches_current_protocol"] is False
    with pytest.raises(WorkspaceError):
        read_outcomes(service, sid, "foreign", RID)


def test_changed_guide_does_not_relabel_past_reports_current(pilot):
    service, sid, _ = pilot
    packet = prepare_physical_pilot(service, sid, "owner", RID)
    saved = record_outcome(service, sid, "owner", report(packet))
    context = delivery_guidance.guidance_context(service, sid, "owner", RID)
    delivery_guidance.update_guidance(
        service,
        sid,
        "owner",
        delivery_guidance.GuidanceUpdate(
            revision_id=RID,
            basis_hash=context["basis_hash"],
            reviewed_current_facts=True,
            guidance={**TEXT, "installation": "New owner-reviewed setup"},
        ),
    )
    result = read_outcomes(service, sid, "owner", RID)
    assert result["records"] == [saved]
    assert result["current_protocol_hash"] != saved["protocol_hash"]
    assert saved["protocol"] == packet["protocol"]
    assert saved["protocol"]["instructions"]["installation"] == TEXT["installation"]
    assert digest(saved["protocol"]) == saved["protocol_hash"]
    assert result["record_context"][saved["id"]]["matches_current_protocol"] is False


def test_tampered_history_refuses_reads_and_future_appends(pilot):
    service, sid, _ = pilot
    packet = prepare_physical_pilot(service, sid, "owner", RID)
    record_outcome(service, sid, "owner", report(packet))
    session = service.store.session_record(sid, "owner")
    rows = copy.deepcopy(session["physical_pilot_records"])
    rows[RID][0]["report"]["checks"][0]["status"] = "observed_pass"
    service.store.update(sid, physical_pilot_records=rows)
    with pytest.raises(WorkspaceError, match="integrity"):
        read_outcomes(service, sid, "owner", RID)
    with pytest.raises(WorkspaceError, match="integrity"):
        record_outcome(service, sid, "owner", report(packet))


def test_pending_candidate_packet_does_not_require_acceptance(pilot):
    service, sid, _ = pilot
    candidate = service.store.revision(sid, RID)
    service.store.update(
        sid,
        selected_revision=None,
        accepted_revision=None,
        proposal={"revision_id": RID, "contract": candidate["contract"]},
    )
    context = delivery_guidance.guidance_context(service, sid, "owner", RID)
    delivery_guidance.update_guidance(
        service,
        sid,
        "owner",
        delivery_guidance.GuidanceUpdate(
            revision_id=RID,
            basis_hash=context["basis_hash"],
            reviewed_current_facts=True,
            guidance=TEXT,
        ),
    )
    packet = prepare_physical_pilot(service, sid, "owner", RID)
    assert packet["physical_status"] == "deferred"
    record_outcome(service, sid, "owner", report(packet))
    session = service.store.session_record(sid, "owner")
    assert session["selected_revision"] is session["accepted_revision"] is None


def test_optional_print_setup_uses_validated_receipt_and_invalidates_old_protocol(
    pilot, monkeypatch
):
    from forge_mind.workspace import print_setup_service

    service, sid, _ = pilot
    source_packet = prepare_physical_pilot(service, sid, "owner", RID)
    files = {
        "print/definition_" + part["part_id"] + ".stl": {
            "part_id": part["part_id"],
            "quantity": part["quantity"],
            "sha256": "e" * 64,
            "bytes": 100,
            "source_stl_sha256": part["source_sha256"],
            "candidate_hash": "f" * 64,
            "frame": "plate_z_up",
            "units": "mm",
        }
        for part in source_packet["protocol"]["made_parts"]
    }
    saved = {
        "basis": {"revision_id": RID},
        "basis_hash": "c" * 64,
        "selected_candidates": {"base": "f" * 64, "cap": "f" * 64},
        "files": files,
        "package": {"revision_id": RID, "package_id": "a" * 32, "sha256": "d" * 64},
        "result_sha256": "b" * 64,
        "result": {
            "parts": {
                name: {
                    "recommended_candidate_hash": "f" * 64,
                    "candidates": [{"candidate_hash": "f" * 64}],
                }
                for name in ("base", "cap")
            }
        },
    }
    service.store.update(sid, print_setup_records={RID: {"raw_untrusted_record": True}})

    def verified(service_arg, con, sid_arg, owner, rid_arg):
        assert service_arg is service and sid_arg == sid and owner == "owner" and rid_arg == RID
        assert con.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone()
        return {"status": "current", "record": saved, "record_hash": digest(saved)}

    monkeypatch.setattr(print_setup_service, "_locked_print_setup", verified)
    context = delivery_guidance.guidance_context(service, sid, "owner", RID)
    assert context["status"] == "stale"
    delivery_guidance.update_guidance(
        service,
        sid,
        "owner",
        delivery_guidance.GuidanceUpdate(
            revision_id=RID,
            basis_hash=context["basis_hash"],
            reviewed_current_facts=True,
            guidance=TEXT,
        ),
    )
    packet = prepare_physical_pilot(service, sid, "owner", RID)
    assert packet["protocol"]["print_setup"]["record_hash"] == digest(saved)
    assert "raw_untrusted_record" not in str(packet["protocol"]["print_setup"])
    assert packet["protocol_hash"] != source_packet["protocol_hash"]
    assert "print/definition_cap.stl" in packet["packet_markdown"]
    request = report(packet)
    saved["files"]["print/definition_cap.stl"]["sha256"] = "a" * 64
    context = delivery_guidance.guidance_context(service, sid, "owner", RID)
    delivery_guidance.update_guidance(
        service,
        sid,
        "owner",
        delivery_guidance.GuidanceUpdate(
            revision_id=RID,
            basis_hash=context["basis_hash"],
            reviewed_current_facts=True,
            guidance=TEXT,
        ),
    )
    with pytest.raises(WorkspaceError, match="protocol changed"):
        record_outcome(service, sid, "owner", request)
    assert read_outcomes(service, sid, "owner", RID)["records"] == []


def test_stale_print_setup_is_not_bound_as_prepared(pilot, monkeypatch):
    from forge_mind.workspace import print_setup_service

    service, sid, _ = pilot
    service.store.update(sid, print_setup_records={RID: {"raw_untrusted_record": True}})
    monkeypatch.setattr(
        print_setup_service,
        "_locked_print_setup",
        lambda *args: {
            "status": "stale",
            "record": {
                "files": {},
                "selected_candidates": {},
                "result": {"parts": {}},
                "package": {},
            },
            "record_hash": "a" * 64,
        },
    )
    with pytest.raises(WorkspaceError, match="print setup is stale"):
        prepare_physical_pilot(service, sid, "owner", RID)


@pytest.mark.parametrize(
    "field", ["specimen_id", "acquisition_note", "printer", "material", "protocol_hash"]
)
def test_report_requires_identities_and_protocol(field, pilot):
    service, sid, _ = pilot
    raw = report(prepare_physical_pilot(service, sid, "owner", RID)).model_dump()
    raw.pop(field)
    with pytest.raises(ValidationError):
        PhysicalPilotOutcomeRequest.model_validate(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "4.0"])
def test_measurements_refuse_nonfinite_or_coerced_values(value, pilot):
    service, sid, _ = pilot
    raw = report(prepare_physical_pilot(service, sid, "owner", RID)).model_dump()
    raw["checks"][1].update(
        status="observed_pass",
        measurements=[
            {
                "name": "diameter",
                "value": value,
                "unit": "mm",
                "method": "Caliper declaration",
            }
        ],
    )
    with pytest.raises(ValidationError):
        PhysicalPilotOutcomeRequest.model_validate(raw)


@pytest.mark.parametrize(
    "defect", ["unit", "method", "not_run_measurement", "duplicate_check", "synthetic"]
)
def test_measurements_and_provenance_cannot_imply_unperformed_checks(defect, pilot):
    service, sid, _ = pilot
    raw = report(prepare_physical_pilot(service, sid, "owner", RID)).model_dump()
    raw["checks"][1].update(
        status="observed_pass",
        measurements=[
            {
                "name": "diameter",
                "value": 4.0,
                "unit": "mm",
                "method": "Caliper declaration",
            }
        ],
    )
    if defect in {"unit", "method"}:
        raw["checks"][1]["measurements"][0].pop(defect)
    elif defect == "not_run_measurement":
        raw["checks"][1]["status"] = "not_run"
    elif defect == "duplicate_check":
        raw["checks"][1]["check_id"] = raw["checks"][0]["check_id"]
    else:
        raw["provenance"] = "synthetic"
    with pytest.raises(ValidationError):
        PhysicalPilotOutcomeRequest.model_validate(raw)
