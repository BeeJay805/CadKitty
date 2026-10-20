"""Global saved-motion disclosure does not depend on mapped interface duties."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy

import pytest
from tests.unit.workspace.test_functional_guidance import pending as pending_fixture
from tests.unit.workspace.test_guidance_transport import instructions
from tests.unit.workspace.test_mesh_handoff import RID

from forge_mind.workspace import completion, delivery_guidance
from forge_mind.workspace.completion_readonly import motion_summary, project_motion
from forge_mind.workspace.delivery_guidance import GuidanceUpdate, guidance_context, update_guidance
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.edit_service import patch_plan
from forge_mind.workspace.instruction_plan import instruction_catalog
from forge_mind.workspace.motion_check import MotionCheckSettings, configurations
from forge_mind.workspace.plan_patch import PlanPatch
from forge_mind.workspace.store import WorkspaceError


@pytest.fixture(params=[False, True], ids=["no_interface", "nominal_only"])
def motion_fixture(request, tmp_path, monkeypatch):
    service, sid, *_ = pending_fixture.__wrapped__(tmp_path, monkeypatch)
    if request.param:
        patch_plan(
            service,
            sid,
            "owner",
            PlanPatch.model_validate(
                {
                    "basis": service.store.edit_basis(sid, "owner").model_dump(),
                    "functional_interfaces": [
                        {
                            "id": "nominal_joint",
                            "requirement_ids": ["support"],
                            "participant_paths": ["base", "bolt"],
                            "mate_ids": ["nominal_fit"],
                            "states": [{"id": "seated", "kind": "reference"}],
                            "duties": [
                                {
                                    "id": "fit",
                                    "kind": "connection",
                                    "proof_scope": "nominal_fit",
                                    "state_ids": ["seated"],
                                }
                            ],
                        }
                    ],
                }
            ),
        )
    revision = service.store.revision(sid, RID)
    raw = deepcopy(revision["document"])
    raw["motion_rig"] = {
        "drivers": [{"id": "lift", "name": "Lift", "unit": "mm", "minimum": 0, "maximum": 80}],
        "groups": [
            {
                "id": "moving",
                "name": "Moving part",
                "driver": "lift",
                "kind": "slide",
                "occurrences": ["bolt"],
            }
        ],
    }
    document = DesignDocument.model_validate(raw)
    directory = service.store.directory(sid, RID)
    content = document.model_dump_json().encode()
    (directory / "document.json").write_bytes(content)
    revision["document"] = document.model_dump(mode="json")
    revision["measurement"]["artifact_hashes"]["document.json"] = hashlib.sha256(
        content
    ).hexdigest()
    mesh_path = directory / "definition-meshes.json"
    mesh = json.loads(mesh_path.read_bytes())
    mesh["document_sha256"] = hashlib.sha256(content).hexdigest()
    mesh_content = json.dumps(mesh, sort_keys=True).encode()
    mesh_path.write_bytes(mesh_content)
    revision["measurement"]["artifact_hashes"][mesh_path.name] = hashlib.sha256(
        mesh_content
    ).hexdigest()
    assembly = revision["measurement"]["facts"]["assembly"]
    # Saved nominal fixture rows are not a native or physical qualification.
    assembly["mates"] = [
        {"id": "nominal_fit", "a": "base.axis", "b": "bolt.axis", "status": "pass"}
    ]
    rig = document.motion_rig.model_dump(mode="json")
    revision["measurement"]["facts"]["motion_rig"] = {
        "rig": rig,
        "bindings": [
            {"path": row["path"], "group": "moving" if row["path"] == "bolt" else None}
            for row in assembly["occurrences"]
        ],
    }
    with service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(revision), sid, RID),
        )
    settings = MotionCheckSettings()
    poses = configurations(rig, settings)
    pairs = len(poses) * len(assembly["occurrences"]) * (len(assembly["occurrences"]) - 1) // 2
    result = {
        "revision_id": RID,
        "engine_fingerprint": "f" * 64,
        "step_sha256": revision["measurement"]["artifact_hashes"]["part.step"],
        "document_sha256": revision["measurement"]["artifact_hashes"]["document.json"],
        "settings": settings.model_dump(),
        "status": "unverified",
        "coverage": "discrete_only",
        "errors": [],
        "collisions": [],
        "collision_count": 0,
        "completed_pairs": pairs,
        "total_pairs": pairs,
        "exact_intersections": 0,
        "disjoint_bounds": pairs,
        "rigid_pair_reuses": 0,
        "requested_configurations": poses,
        "completed_configurations": poses,
        "scope": "All bound exported bodies at the listed poses; no continuous or physical approval.",
    }
    return service, sid, revision, result, request.param


def save_motion(fixture, identifier, result):
    service, sid, *_ = fixture
    content = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    path = service.store.root / "motion-checks" / sid / identifier / "check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    registered = service.store.session_record(sid, "owner").get("motion_checks", [])
    service.store.update(
        sid,
        motion_checks=[
            *[row for row in registered if row["id"] != identifier],
            {"id": identifier, "revision_id": RID, "sha256": hashlib.sha256(content).hexdigest()},
        ],
    )
    return path


def failed_result(result):
    return {
        **deepcopy(result),
        "status": "fail",
        "collision_count": 1,
        "collisions": [{"a": "base", "b": "bolt", "values": {"lift": 0.0}, "overlap_mm3": 1.25}],
        "exact_intersections": 1,
        "disjoint_bounds": result["total_pairs"] - 1,
    }


def save_guide(service, sid, context):
    request = instructions(context)
    request["instruction_plan"] = {
        "version": 1,
        "fact_refs": context["instruction_catalog"]["required_refs"],
        "steps": [{"action": "review_check", "target": "check:motion:current"}],
    }
    return update_guidance(service, sid, "owner", GuidanceUpdate.model_validate(request))


def test_global_failed_motion_is_required_even_with_only_nominal_duties(motion_fixture):
    service, sid, _, result, nominal = motion_fixture
    failed = failed_result(result)
    save_motion(motion_fixture, "current", failed)
    context = guidance_context(service, sid, "owner", RID)
    motion = context["facts"]["motion_check"]
    assert motion["status"] == "checked_fail"
    assert motion["collisions"] == failed["collisions"]
    assert motion["scope"] == failed["scope"]
    assert motion["verified_evidence"] == failed
    assert context["basis"]["motion_evidence_hash"]
    if nominal:
        duty = context["facts"]["functional_interfaces"]["rows"][0]["duties"][0]
        assert duty["proof_scope"] == "nominal_fit" and duty["status"] == "nominal_checked"
    else:
        assert "functional_interfaces" not in context["facts"]
    # Aggregation must preserve a failure beside any number of unknown requirements.
    facts = {
        **context["facts"],
        "requirement_findings": [
            {"requirement_id": f"r{index}", "status": "unverified"} for index in range(13)
        ],
    }
    catalog = instruction_catalog({**context, "facts": facts})
    assert catalog["facts"]["check:motion:current"]["kind"] == "check"
    assert catalog["facts"]["check:motion:current"]["state"] == "checked_fail"
    assert "check:motion:current" in catalog["required_refs"]
    assert catalog["facts"]["checks:summary"]["state"] == "fail"
    saved = save_guide(service, sid, context)
    assert saved["instruction_binding"]["status"] == "bound"
    assert saved["physical_readiness"] == "unverified"


def test_missing_motion_receipt_remains_unverified(motion_fixture):
    service, sid, *_ = motion_fixture
    context = guidance_context(service, sid, "owner", RID)
    assert context["facts"]["motion_check"]["status"] == "unverified"
    assert "motion_evidence_hash" not in context["basis"]
    assert context["instruction_catalog"]["facts"]["check:motion:current"]["state"] == "unverified"
    assert context["instruction_catalog"]["facts"]["checks:summary"]["state"] == "unverified"


@pytest.mark.parametrize(
    "change", ["missing", "tamper", "revision", "step", "document", "settings", "engine"]
)
def test_latest_invalid_motion_never_falls_back_to_older_checked_receipt(motion_fixture, change):
    service, sid, _, result, _ = motion_fixture
    save_motion(motion_fixture, "old", result)
    changed = failed_result(result)
    if change in {"revision", "engine"}:
        changed["revision_id" if change == "revision" else "engine_fingerprint"] = "changed"
    elif change in {"step", "document"}:
        changed[change + "_sha256"] = "changed"
    elif change == "settings":
        changed["settings"]["samples_per_driver"] += 1
    latest = save_motion(motion_fixture, "latest", changed)
    if change == "missing":
        latest.unlink()
    elif change == "tamper":
        latest.write_text("{}", encoding="utf-8")
    context = guidance_context(service, sid, "owner", RID)
    assert context["facts"]["motion_check"]["status"] == "unverified"
    assert "verified_evidence" not in context["facts"]["motion_check"]
    assert "motion_evidence_hash" not in context["basis"]
    assert context["instruction_catalog"]["facts"]["check:motion:current"]["state"] == "unverified"
    assert context["physical_readiness"] == "unverified"


def test_all_saved_failure_witnesses_errors_and_scope_remain_literal(motion_fixture):
    service, sid, _, result, _ = motion_fixture
    changed = failed_result(result)
    changed["collisions"] = [
        {"a": "base", "b": "bolt", "values": pose, "overlap_mm3": index + 0.25}
        for index, pose in enumerate(result["requested_configurations"])
    ]
    changed["collision_count"] = len(changed["collisions"])
    changed["errors"] = [str(index) + "x" * 350 for index in range(8)]
    changed["scope"] = "Complete saved scope. " * 70
    save_motion(motion_fixture, "all_failures", changed)
    context = guidance_context(service, sid, "owner", RID)
    motion = context["facts"]["motion_check"]
    assert len(changed["collisions"]) > 5
    assert motion["status"] == "checked_fail"
    for key in ("collisions", "errors", "scope"):
        assert motion[key] == changed[key]
        assert motion["verified_evidence"][key] == changed[key]
    assert context["instruction_catalog"]["facts"]["checks:summary"]["state"] == "fail"


def test_incomplete_or_error_motion_has_no_positive_state(motion_fixture):
    service, sid, _, result, _ = motion_fixture
    changed = {
        **deepcopy(result),
        "errors": ["Saved incomplete intersection"],
        "completed_configurations": [],
    }
    save_motion(motion_fixture, "incomplete", changed)
    context = guidance_context(service, sid, "owner", RID)
    assert context["facts"]["motion_check"]["status"] == "unverified"
    assert context["facts"]["motion_check"]["errors"] == changed["errors"]
    assert context["instruction_catalog"]["facts"]["checks:summary"]["state"] == "unverified"


def test_equivalent_reissued_motion_keeps_current_guide_but_physical_change_stales_it(
    motion_fixture,
):
    service, sid, _, result, _ = motion_fixture
    first = {
        **failed_result(result),
        "wall_seconds": 1,
        "cpu_seconds": 0.5,
        "worker_cpu_affinity": [0],
    }
    save_motion(motion_fixture, "first", first)
    context = guidance_context(service, sid, "owner", RID)
    saved = save_guide(service, sid, context)
    equivalent = {
        **deepcopy(first),
        "wall_seconds": 7,
        "cpu_seconds": 4,
        "worker_cpu_affinity": [2],
    }
    save_motion(motion_fixture, "reissued", equivalent)
    current = guidance_context(service, sid, "owner", RID)
    assert current["status"] == "current"
    assert current["basis_hash"] == context["basis_hash"]
    assert current["facts"]["motion_check"] == context["facts"]["motion_check"]
    changed = deepcopy(equivalent)
    changed["collisions"][0]["overlap_mm3"] = 2.5
    save_motion(motion_fixture, "physical_change", changed)
    stale = guidance_context(service, sid, "owner", RID)
    assert stale["facts"]["motion_check"]["status"] == "checked_fail"
    assert stale["basis"]["motion_evidence_hash"] != current["basis"]["motion_evidence_hash"]
    assert stale["basis_hash"] != current["basis_hash"]
    assert stale["status"] == "stale" and stale["history"] == saved["history"]
    with pytest.raises(WorkspaceError, match="changed"):
        update_guidance(service, sid, "owner", GuidanceUpdate.model_validate(instructions(current)))


def test_shared_motion_reader_verifies_explicit_summary_and_current_engine(motion_fixture):
    service, sid, revision, result, _ = motion_fixture
    save_motion(motion_fixture, "current", failed_result(result))
    context = guidance_context(service, sid, "owner", RID)
    session = service.store.session_record(sid, "owner")
    summary, proof = project_motion(session, revision, context["basis"], root=service.store.root)
    explicit = motion_summary(failed_result(result), revision, context["basis"], "current")
    assert summary["verified_evidence"] == proof == failed_result(result)
    assert project_motion(
        session, revision, context["basis"], motion=explicit, root=service.store.root
    ) == (summary, proof)
    assert project_motion(
        session, revision, context["basis"], motion={}, root=service.store.root
    ) == ({}, None)
    refused, missing = project_motion(
        session,
        revision,
        context["basis"],
        motion={**explicit, "reason": "Changed summary"},
        root=service.store.root,
    )
    assert refused["status"] == "unverified" and missing is None
    old = deepcopy(revision)
    old["measurement"]["engine_fingerprint"] = "old-engine"
    refused, missing = project_motion(session, old, context["basis"], root=service.store.root)
    assert (
        refused["status"] == "unverified" and "verified_evidence" not in refused and missing is None
    )


def test_motion_projection_uses_caller_transaction_without_new_checks(motion_fixture, monkeypatch):
    service, sid, _, result, _ = motion_fixture
    save_motion(motion_fixture, "current", failed_result(result))

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only motion projection cannot open another connection or run work")

    with service.store.connect() as con:
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
        monkeypatch.setattr("forge_mind.workspace.motion_service.run_motion_job", forbidden)
        monkeypatch.setattr("forge_mind.workspace.processing.process_revision", forbidden)
        _, context = delivery_guidance._locked_context(service, con, sid, "owner", RID)
        assert context["facts"]["motion_check"]["status"] == "checked_fail"
        assert con.total_changes == before
