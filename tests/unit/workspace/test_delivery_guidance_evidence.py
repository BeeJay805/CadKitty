"""Current receipt projections use saved fixtures; no model or CAD execution."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy

import pytest
from tests.unit.workspace.test_completion_obligations import example as interface_fixture
from tests.unit.workspace.test_functional_guidance import pending as pending_fixture
from tests.unit.workspace.test_guidance_transport import instructions
from tests.unit.workspace.test_mesh_handoff import RID

from forge_mind.workspace import completion, delivery_guidance
from forge_mind.workspace.completion_readonly import (
    motion_summary,
    prepared_binding,
    project_interfaces,
)
from forge_mind.workspace.delivery_guidance import GuidanceUpdate, guidance_context, update_guidance
from forge_mind.workspace.edit_service import patch_plan
from forge_mind.workspace.motion_check import MotionCheckSettings, configurations
from forge_mind.workspace.plan_patch import PlanPatch
from forge_mind.workspace.product_tools import _SCOPES
from forge_mind.workspace.store import WorkspaceError

BASIS_KEYS = (
    "revision_id",
    "contract_hash",
    "session_contract_hash",
    "engine_fingerprint",
    "source_fingerprint",
    "input_version",
    "plan_hash",
    "artifact_hashes",
)
CHECK_ID = "space"
RECEIPT_ID = "d" * 32


def basis(context):
    return {key: context["basis"][key] for key in BASIS_KEYS}


def projection(context):
    return context["facts"]["functional_interfaces"]


def write_receipt(service, sid, receipt, *, corrupt_digest=False):
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    with service.store.connect() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS product_checks (id TEXT PRIMARY KEY, session TEXT, document TEXT, digest TEXT)"
        )
        con.execute(
            "INSERT OR REPLACE INTO product_checks VALUES (?,?,?,?)",
            (receipt["id"], sid, encoded, "tampered" if corrupt_digest else digest),
        )


@pytest.fixture(params=["selected", "pending"])
def evidence(request, tmp_path, monkeypatch):
    service, sid, _, _, _, _, _ = pending_fixture.__wrapped__(tmp_path, monkeypatch)
    if request.param == "selected":
        service.store.update(sid, selected_revision=RID, proposal=None)
    patch_plan(
        service,
        sid,
        "owner",
        PlanPatch.model_validate(
            {
                "basis": service.store.edit_basis(sid, "owner").model_dump(),
                "subsystems": [
                    {
                        "id": "clearance",
                        "name": "Saved clearance",
                        "part_ids": ["base", "bolt"],
                        "requirement_ids": ["support"],
                        "checks": [
                            {
                                "id": CHECK_ID,
                                "request": {
                                    "tool": "clearance_volume",
                                    "units": "mm",
                                    "reservation": {"occurrence_path": "base"},
                                    "obstacles": [{"occurrence_path": "bolt"}],
                                },
                            }
                        ],
                    }
                ],
                "functional_interfaces": [
                    {
                        "id": "joint",
                        "requirement_ids": ["support"],
                        "participant_paths": ["base", "bolt"],
                        "required_features": [{"occurrence_path": "base", "feature_id": "body"}],
                        "duties": [
                            {
                                "id": "clear",
                                "kind": "clearance",
                                "proof_scope": "static_clearance",
                                "check_ids": [CHECK_ID],
                            },
                            {
                                "id": "hold",
                                "kind": "holding",
                                "proof_scope": "physical_function",
                                "physical_validation": ["Test actual support."],
                            },
                        ],
                    }
                ],
            }
        ),
    )
    context = guidance_context(service, sid, "owner", RID)
    session = service.store.get(sid, "owner")
    definition = next(
        check
        for subsystem in session["design_plan"]["subsystems"]
        for check in subsystem["checks"]
        if check["id"] == CHECK_ID
    )
    frozen = basis(context)
    receipt = {
        "id": RECEIPT_ID,
        "revision_id": RID,
        "source_fingerprint": frozen["source_fingerprint"],
        "engine_fingerprint": frozen["engine_fingerprint"],
        "artifact_hashes": frozen["artifact_hashes"],
        "contract": service.store.revision(sid, RID)["contract"],
        "request": definition["request"],
        "result": {
            "tool": "clearance_volume",
            "status": "pass",
            "algorithm_scope": _SCOPES["clearance_volume"],
            "qualification_status": "unverified",
        },
    }
    write_receipt(service, sid, receipt)
    service.store.update(
        sid,
        prepared_check_cache={
            CHECK_ID: {
                "binding": prepared_binding(frozen, definition),
                "receipt_id": RECEIPT_ID,
            }
        },
    )
    return service, sid, receipt


def test_current_receipt_reaches_guidance_without_closing_physical_obligations(evidence):
    service, sid, _ = evidence
    before = deepcopy(service.store.session_record(sid, "owner"))
    context = guidance_context(service, sid, "owner", RID)
    row = projection(context)["rows"][0]
    assert row["duties"][0]["status"] == "nominal_checked"
    assert row["duties"][1]["status"] == "unknown"
    assert projection(context)["external_obligations"][0]["physical_validation"] == [
        "Test actual support."
    ]
    assert context["physical_readiness"] == "unverified"
    assert service.store.session_record(sid, "owner") == before
    assert service.store.get(sid, "owner")["accepted_revision"] is None


@pytest.mark.parametrize("status", ["pass", "fail"])
def test_guidance_and_completion_share_actual_receipt_verdict(evidence, status):
    service, sid, receipt = evidence
    receipt["result"]["status"] = status
    write_receipt(service, sid, receipt)
    context = guidance_context(service, sid, "owner", RID)
    session = service.store.get(sid, "owner")
    plan = session["design_plan"]
    revision = service.store.revision(sid, RID)
    rows = [
        {
            "id": "clearance",
            "checks": [{"id": CHECK_ID, "status": status, "receipt_id": RECEIPT_ID}],
        }
    ]
    expected = completion._interface_projection(
        service, sid, "owner", plan, revision, rows, basis(context)
    )
    assert projection(context)["rows"] == expected["rows"]
    assert expected["rows"][0]["duties"][0]["status"] == (
        "check_failed" if status == "fail" else "nominal_checked"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "digest",
        "receipt",
        "revision",
        "foreign",
        "request",
        "source",
        "engine",
        "input",
        "contract",
    ],
)
def test_missing_tampered_or_changed_basis_cannot_supply_nominal_proof(
    evidence, monkeypatch, fault
):
    service, sid, receipt = evidence
    context = guidance_context(service, sid, "owner", RID)
    old_update = GuidanceUpdate.model_validate(instructions(context))
    update_guidance(service, sid, "owner", old_update)
    if fault == "missing":
        with service.store.connect() as con:
            con.execute("DELETE FROM product_checks WHERE id=?", (RECEIPT_ID,))
    elif fault == "digest":
        write_receipt(service, sid, receipt, corrupt_digest=True)
    elif fault == "receipt":
        receipt["artifact_hashes"] = {"part.step": "forged"}
        write_receipt(service, sid, receipt)
    elif fault == "revision":
        receipt["revision_id"] = "c" * 32
        write_receipt(service, sid, receipt)
    elif fault == "foreign":
        with service.store.connect() as con:
            con.execute("UPDATE product_checks SET session=? WHERE id=?", ("foreign", RECEIPT_ID))
    elif fault == "request":
        receipt["request"] = {**receipt["request"], "obstacles": []}
        write_receipt(service, sid, receipt)
    elif fault == "source":
        monkeypatch.setattr(delivery_guidance, "source_fingerprint", lambda *_: "changed")
    elif fault == "engine":
        monkeypatch.setattr(delivery_guidance, "engine_fingerprint", lambda: "changed")
    elif fault == "input":
        service.store.save_engineering(sid, "owner", 0, {})
    else:
        receipt["contract"] = {"requirements": []}
        write_receipt(service, sid, receipt)
    current = guidance_context(service, sid, "owner", RID)
    assert projection(current)["rows"][0]["duties"][0]["status"] == "check_pending"
    assert current["status"] == "stale"
    with pytest.raises(WorkspaceError):
        update_guidance(service, sid, "owner", old_update)


def test_locked_guidance_uses_only_supplied_connection_and_performs_no_writes(
    evidence, monkeypatch
):
    service, sid, _ = evidence

    def forbidden(*args, **kwargs):
        pytest.fail("Locked evidence reads cannot fetch independent state or execute work")

    with service.store.connect() as con:
        con.execute("BEGIN")
        before = con.total_changes
        con.set_authorizer(
            lambda action, *_: (
                sqlite3.SQLITE_OK
                if action
                in {
                    sqlite3.SQLITE_SELECT,
                    sqlite3.SQLITE_READ,
                    sqlite3.SQLITE_FUNCTION,
                    sqlite3.SQLITE_TRANSACTION,
                }
                else sqlite3.SQLITE_DENY
            )
        )
        monkeypatch.setattr(service.store, "connect", forbidden)
        monkeypatch.setattr(service, "_load_revision", forbidden)
        monkeypatch.setattr(completion, "completion_context", forbidden)
        monkeypatch.setattr("forge_mind.workspace.product_service.run_tool", forbidden)
        monkeypatch.setattr("forge_mind.workspace.motion_service.run_motion_job", forbidden)
        monkeypatch.setattr("forge_mind.workspace.processing.process_revision", forbidden)
        _, context = delivery_guidance._locked_context(service, con, sid, "owner", RID)
        assert projection(context)["rows"][0]["duties"][0]["status"] == "nominal_checked"
        assert con.total_changes == before


def test_unrelated_plan_update_retains_receipt_but_stales_old_guide(evidence):
    service, sid, _ = evidence
    context = guidance_context(service, sid, "owner", RID)
    update_guidance(service, sid, "owner", GuidanceUpdate.model_validate(instructions(context)))
    patch_plan(
        service,
        sid,
        "owner",
        PlanPatch.model_validate(
            {
                "basis": service.store.edit_basis(sid, "owner").model_dump(),
                "objective": "Retain the same measured geometry and explicit support obligations",
            }
        ),
    )
    current = guidance_context(service, sid, "owner", RID)
    assert current["basis"]["plan_hash"] != context["basis"]["plan_hash"]
    assert projection(current)["rows"][0]["duties"][0]["status"] == "nominal_checked"
    assert current["status"] == "stale"


def test_changed_saved_artifact_is_refused_before_any_receipt_projection(evidence):
    service, sid, _ = evidence
    (service.store.directory(sid, RID) / "part.step").write_bytes(b"Changed saved geometry")
    with pytest.raises(WorkspaceError, match="artifacts changed"):
        guidance_context(service, sid, "owner", RID)


def test_equivalent_receipt_and_unchanged_finalization_preserve_current_guide(
    evidence, monkeypatch
):
    service, sid, receipt = evidence
    # Finalization acts on selected candidates; the pending-preview controls remain above.
    service.store.update(sid, selected_revision=RID, proposal=None)
    context = guidance_context(service, sid, "owner", RID)
    update_guidance(service, sid, "owner", GuidanceUpdate.model_validate(instructions(context)))
    changed_id = {**deepcopy(receipt), "id": "e" * 32, "job_id": "later-job", "at": 999}
    write_receipt(service, sid, changed_id)
    cache = service.store.get(sid, "owner")["prepared_check_cache"]
    cache[CHECK_ID]["receipt_id"] = changed_id["id"]
    service.store.update(sid, prepared_check_cache=cache)
    equivalent = guidance_context(service, sid, "owner", RID)
    assert equivalent["status"] == "current"
    assert equivalent["basis_hash"] == context["basis_hash"]

    def forbidden(*args, **kwargs):
        pytest.fail("An unchanged prepared receipt must not run CAD or another probe")

    monkeypatch.setattr(completion, "engine_fingerprint", lambda: "f" * 64)
    monkeypatch.setattr("forge_mind.workspace.product_service.run_tool", forbidden)
    monkeypatch.setattr("forge_mind.workspace.processing.process_revision", forbidden)
    monkeypatch.setattr(
        "forge_mind.workspace.product_interfaces.validate_interfaces", lambda *_: []
    )
    monkeypatch.setattr("forge_mind.workspace.engineering_service.run_report_job", lambda *_: None)
    report = completion.finalize_completion(
        service, {"session": sid, "owner": "owner"}, None, lambda: False, "needs_review"
    )
    check = next(row for row in report["subsystems"] if row["id"] == "clearance")["checks"][0]
    assert check["status"] == "pass" and check["reused"] is True
    current = guidance_context(service, sid, "owner", RID)
    assert current["status"] == "current"
    assert current["basis_hash"] == context["basis_hash"]


def test_changed_physical_result_stales_guidance_even_when_verdict_stays_pass(evidence):
    service, sid, receipt = evidence
    receipt["result"]["results"] = [{"overlap_mm3": 0.0}]
    write_receipt(service, sid, receipt)
    context = guidance_context(service, sid, "owner", RID)
    update_guidance(service, sid, "owner", GuidanceUpdate.model_validate(instructions(context)))
    receipt["result"]["results"] = [{"overlap_mm3": 0.0000001}]
    write_receipt(service, sid, receipt)
    current = guidance_context(service, sid, "owner", RID)
    assert projection(current)["rows"] == projection(context)["rows"]
    assert current["basis"]["facts_hash"] == context["basis"]["facts_hash"]
    assert (
        current["basis"]["completion_evidence_hash"] != context["basis"]["completion_evidence_hash"]
    )
    assert current["status"] == "stale"


@pytest.mark.parametrize("collided", [False, True])
def test_registered_motion_is_available_before_finalization_and_latest_tamper_cannot_fallback(
    tmp_path, collided
):
    plan, revision, rows = interface_fixture.__wrapped__()
    rig = {"drivers": [{"id": "lift", "minimum": 0, "maximum": 80, "reference": 0}], "groups": []}
    revision["measurement"]["engine_fingerprint"] = "engine"
    revision["document"]["motion_rig"] = {"kind": "prescribed_rigid_v1"}
    revision["measurement"]["facts"]["motion_rig"] = {
        "rig": rig,
        "bindings": [{"path": "member", "group": "moving"}, {"path": "pin", "group": None}],
    }
    interface = plan["functional_interfaces"][0]
    interface["states"] = [{"id": "lifted", "kind": "motion", "driver_values": {"lift": 80.0}}]
    interface["duties"] = [
        {
            "id": "remove",
            "kind": "travel",
            "proof_scope": "sampled_clearance",
            "state_ids": ["lifted"],
        }
    ]
    frozen = {"revision_id": revision["id"], **revision["completion_basis"]}
    settings = MotionCheckSettings()
    poses = configurations(rig, settings)
    result = {
        "revision_id": revision["id"],
        "engine_fingerprint": "engine",
        "step_sha256": "saved-step",
        "document_sha256": "saved-document",
        "settings": settings.model_dump(),
        "status": "unverified",
        "coverage": "discrete_only",
        "errors": [],
        "collisions": [],
        "collision_count": 0,
        "completed_pairs": len(poses),
        "total_pairs": len(poses),
        "exact_intersections": 0,
        "disjoint_bounds": len(poses),
        "rigid_pair_reuses": 0,
        "requested_configurations": poses,
        "completed_configurations": poses,
    }
    if collided:
        result.update(
            status="fail",
            collisions=[{"a": "member", "b": "pin", "values": {"lift": 80.0}, "overlap_mm3": 1.0}],
            collision_count=1,
            exact_intersections=1,
            disjoint_bounds=len(poses) - 1,
        )
    session = {"motion_checks": []}
    original_rows = deepcopy(rows)

    def save(identifier, *, wall_seconds=1):
        path = tmp_path / "motion-checks" / "s" / identifier / "check.json"
        path.parent.mkdir(parents=True)
        content = json.dumps({**result, "wall_seconds": wall_seconds}).encode()
        path.write_bytes(content)
        session["motion_checks"].append(
            {
                "id": identifier,
                "revision_id": revision["id"],
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        return path

    with sqlite3.connect(":memory:") as con:
        absent, absent_hash = project_interfaces(
            con, "s", session, plan, revision, frozen, rows=rows, root=tmp_path
        )
        assert absent["rows"][0]["duties"][0]["status"] == "check_pending"
        save("old")
        first, first_hash = project_interfaces(
            con, "s", session, plan, revision, frozen, root=tmp_path
        )
        assert first["rows"][0]["duties"][0]["status"] == (
            "check_failed" if collided else "nominal_checked"
        )
        assert first["physical_readiness"] == "unverified"
        if collided:
            assert first["next_work_category"] == "repair"
        else:
            assert first["next_target"] is None
        with_rows, with_rows_hash = project_interfaces(
            con, "s", session, plan, revision, frozen, rows=rows, root=tmp_path
        )
        assert with_rows == first and with_rows_hash == first_hash
        latest = save("new", wall_seconds=2)
        same, same_hash = project_interfaces(
            con, "s", session, plan, revision, frozen, root=tmp_path
        )
        assert same == first and same_hash == first_hash
        same_with_rows, same_with_rows_hash = project_interfaces(
            con, "s", session, plan, revision, frozen, rows=rows, root=tmp_path
        )
        assert same_with_rows == same and same_with_rows_hash == same_hash

        supplied_motion = motion_summary(result, revision, frozen, "new")
        explicit, explicit_hash = project_interfaces(
            con,
            "s",
            session,
            plan,
            revision,
            frozen,
            rows=rows,
            motion=supplied_motion,
            root=tmp_path,
        )
        assert explicit == same and explicit_hash == same_hash
        for supplied in ({}, {**supplied_motion, "reason": "Unverified replacement summary"}):
            refused, refused_hash = project_interfaces(
                con,
                "s",
                session,
                plan,
                revision,
                frozen,
                rows=rows,
                motion=supplied,
                root=tmp_path,
            )
            assert refused == absent and refused_hash == absent_hash

        latest.write_text("{}", encoding="utf-8")
        invalid, invalid_hash = project_interfaces(
            con, "s", session, plan, revision, frozen, root=tmp_path
        )
        assert invalid["rows"][0]["duties"][0]["status"] == "check_pending"
        assert invalid_hash != first_hash
        invalid_with_rows, invalid_with_rows_hash = project_interfaces(
            con, "s", session, plan, revision, frozen, rows=rows, root=tmp_path
        )
        assert invalid_with_rows == invalid and invalid_with_rows_hash == invalid_hash
        assert invalid == absent and invalid_hash == absent_hash
        assert con.total_changes == 0
    assert rows == original_rows


def test_saved_motion_summary_preserves_exact_coverage_and_failure_rules():
    rig = {"drivers": [{"id": "lift", "minimum": 0, "maximum": 80, "reference": 0}], "groups": []}
    settings = MotionCheckSettings()
    poses = configurations(rig, settings)
    revision = {"measurement": {"facts": {"motion_rig": {"rig": rig, "bindings": [{}, {}]}}}}
    frozen = {
        "revision_id": RID,
        "engine_fingerprint": "engine",
        "artifact_hashes": {"part.step": "step", "document.json": "doc"},
    }
    result = {
        "revision_id": RID,
        "engine_fingerprint": "engine",
        "step_sha256": "step",
        "document_sha256": "doc",
        "settings": settings.model_dump(),
        "status": "unverified",
        "coverage": "discrete_only",
        "errors": [],
        "collisions": [],
        "collision_count": 0,
        "completed_pairs": len(poses),
        "total_pairs": len(poses),
        "exact_intersections": 0,
        "disjoint_bounds": len(poses),
        "rigid_pair_reuses": 0,
        "requested_configurations": poses,
        "completed_configurations": poses,
    }
    assert motion_summary(result, revision, frozen, "check")["status"] == "checked"
    incomplete = {**result, "completed_configurations": poses[:-1]}
    assert motion_summary(incomplete, revision, frozen, "check")["status"] == "unverified"
    failed = {**result, "collision_count": 1, "collisions": [{"overlap_mm3": 1.0}]}
    assert motion_summary(failed, revision, frozen, "check")["status"] == "checked_fail"
    with pytest.raises(WorkspaceError):
        motion_summary({**result, "step_sha256": "changed"}, revision, frozen, "check")
