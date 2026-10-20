"""Exact final nominal/digital proof; legacy evidence and physical unknowns stay intact."""

from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import BadZipFile, ZipFile

from forge_mind.workspace.budget import expired, time_left
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.models import Contract, DesignPlan, Measurement
from forge_mind.workspace.processing import copy_artifact, process_revision, regular_file
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.store import Store

REPORT_FILE = "contract-verification.json"
MAX_REPORT_BYTES = 4 * 1024 * 1024
SCOPE = (
    "Frozen, explicitly admitted nominal CAD and digital predicates only. "
    "Physical fit, printing, accuracy, strength and safe use remain unverified. "
    "Earlier revision evidence and authored proposals retain their original scope."
)


def has_verification(contract: dict[str, Any] | None) -> bool:
    return any(r.get("verification") is not None for r in (contract or {}).get("requirements", []))


def has_requirement_status(contract: dict[str, Any] | None) -> bool:
    """Include unsupported duties, including mixed contracts; retain scalar-only behavior."""
    requirements = (contract or {}).get("requirements", [])
    return has_verification(contract) or (
        bool(requirements)
        and any(row.get("metric", "unmeasured") == "unmeasured" for row in requirements)
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _selected_revision(session: dict[str, Any]) -> str:
    rid = session.get("selected_revision")
    if not isinstance(rid, str) or re.fullmatch(r"[a-f0-9]{32}", rid) is None:
        raise WorkspaceError("Verification requires an exact selected revision identity")
    return rid


def completion_projection(session: dict[str, Any], proof: dict[str, Any]) -> dict[str, Any]:
    """Keep the recorded job outcome separate from current typed proof validity."""
    raw = session["state"]
    if not has_verification(session.get("contract")) and not (
        session.get("authoring_mode") == "native_operations"
        and has_requirement_status(session.get("contract"))
    ):
        return {"state": raw}
    verified = (
        raw == "contract_satisfied"
        and proof.get("status") == "pass"
        and proof.get("requirements_satisfied") is True
        and proof.get("job_id") == session.get("job_id")
        and proof.get("revision_id") == session.get("selected_revision")
    )
    return {
        "state": "needs_review" if raw == "contract_satisfied" and not verified else raw,
        "job_outcome_state": raw,
        "current_completion_status": "verified" if verified else "unverified",
    }


def _report_path(store: Store, sid: str, report_id: str) -> Path:
    if any(re.fullmatch(r"[a-f0-9]{32}", value) is None for value in (sid, report_id)):
        raise WorkspaceError("Invalid verification artifact identity")
    path = store.root / "reports" / sid / report_id / REPORT_FILE
    for parent in (path, *path.parents):
        if parent.is_symlink() or (
            parent.exists() and getattr(parent.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise WorkspaceError("Verification artifact contains a link or reparse point")
    return path


def current_basis(
    store: Store, con: sqlite3.Connection, sid: str, session: dict[str, Any]
) -> dict[str, Any]:
    """Read live DB identities through the caller's transaction, including write versions."""
    from forge_mind.workspace.intent_attribution_service import request_text
    from forge_mind.workspace.measure import engine_fingerprint, verify_artifacts

    rid = _selected_revision(session)
    row = con.execute(
        "SELECT document FROM revisions WHERE session=? AND id=?", (sid, rid)
    ).fetchone()
    if row is None:
        raise WorkspaceError("Verification requires a selected saved revision")
    revision = json.loads(row[0])
    contract = Contract.model_validate(revision["contract"])
    measured = Measurement.model_validate(revision["measurement"])
    if measured.contract_hash != contract.fingerprint():
        raise WorkspaceError("Saved measurement belongs to a different frozen contract")
    if [f.requirement_id for f in measured.findings] != [r.id for r in contract.requirements]:
        raise WorkspaceError("Saved findings do not exactly cover the frozen requirements")
    if not measured.geometry_valid or not verify_artifacts(store.directory(sid, rid), measured):
        raise WorkspaceError("Verification source geometry is invalid or its artifacts changed")
    inputs = con.execute(
        "SELECT version,document FROM engineering_inputs WHERE session=?", (sid,)
    ).fetchone()
    return {
        "edit_basis": store._edit_basis(con, sid, session, rid).model_dump(mode="json"),
        "job_id": session.get("job_id"),
        "revision_id": rid,
        "selection_version": session.get("selection_version", 0),
        "accepted_revision": session.get("accepted_revision"),
        "revision_hash": digest(revision),
        "contract_hash": Contract.model_validate(session.get("contract")).fingerprint(),
        "revision_contract_hash": Contract.model_validate(revision["contract"]).fingerprint(),
        "plan_hash": session.get("plan_hash"),
        "plan_document_hash": digest(session.get("design_plan")),
        "plan_version": session.get("plan_version", 0),
        "plan_contract_hash": session.get("plan_contract_hash"),
        "operation_snapshot_hash": digest(session.get("operation_plan_snapshot")),
        "input_version": inputs[0] if inputs else 0,
        "engineering_inputs_hash": digest(json.loads(inputs[1]) if inputs else None),
        "document_hash": digest(revision.get("document")),
        "artifact_hashes": measured.artifact_hashes,
        "engine_fingerprint": engine_fingerprint(),
        "measured_engine_fingerprint": measured.engine_fingerprint,
        "guidance_record_hash": digest(session.get("delivery_guidance_record")),
        "print_setup_record_hash": digest((session.get("print_setup_records") or {}).get(rid)),
        "intent_record_hash": digest(session.get("intent_attribution")),
        "request_hash": digest(request_text(session)),
        "delivery_hash": digest(session.get("automatic_delivery")),
        "native_review_hash": digest(session.get("native_review")),
        "incomplete_finish_job_id": session.get("incomplete_finish_job_id"),
        "authority_conflict_job_id": session.get("authority_conflict_job_id"),
        **_photo_basis(con, sid, session),
    }


def _photo_basis(con: sqlite3.Connection, sid: str, session: dict[str, Any]) -> dict[str, Any]:
    """The transactional proof basis includes mutable workflow receipt identity."""
    from forge_mind.workspace.verification_models import NominalPhotoSolid

    contract = Contract.model_validate(session.get("contract") or {})
    if not any(isinstance(row.verification, NominalPhotoSolid) for row in contract.requirements):
        return {}
    if not con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='photo_generation'"
    ).fetchone():
        raise WorkspaceError("Photo nominal proof lacks an authenticated workflow record")
    row = con.execute(
        "SELECT document,digest FROM photo_generation WHERE session=?", (sid,)
    ).fetchone()
    if row is None:
        raise WorkspaceError("Photo nominal proof lacks an authenticated workflow record")
    return {"photo_workflow_hash": digest({"document": row[0], "digest": row[1]})}


def _result(
    status: str, reason: str, method: str, criterion_hash: str | None = None
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "method": method,
        "criterion_hash": criterion_hash,
        "facts": {},
        "evidence_refs": [],
    }


def verification_method(criterion: Any) -> str:
    if criterion is None:
        return "legacy_measurement"
    methods = {
        "digital_delivery": "digital_delivery_v1",
        "nominal_perforated_box": "nominal_perforated_box_v1",
        "nominal_through_hole_pattern": "nominal_through_hole_pattern_v1",
        "nominal_photo_solid": "nominal_photo_solid_v1",
        "nominal_subject_inventory": "nominal_subject_inventory_v1",
        "nominal_delivery_scope": "nominal_delivery_scope_v1",
    }
    if criterion.kind not in methods:
        raise WorkspaceError("Unsupported frozen verification kind")
    return methods[criterion.kind]


def evaluate_frozen_policy(
    requirement: Any,
    *,
    plan: DesignPlan,
    document: DesignDocument,
    captured_delivery: dict[str, Any] | None = None,
    package_content: bytes = b"",
    print_context: dict[str, Any] | None = None,
    final: bool = False,
) -> dict[str, Any]:
    """Same inventory/scope evaluators for author inspection and final capture proof."""
    from forge_mind.workspace.nominal_delivery_scope import evaluate_delivery_scope
    from forge_mind.workspace.nominal_inventory import evaluate_inventory

    if requirement.verification.kind == "nominal_subject_inventory":
        return evaluate_inventory(
            requirement,
            plan=plan,
            document=document,
            captured_delivery=captured_delivery,
            package_content=package_content,
            final=final,
        )
    if requirement.verification.kind == "nominal_delivery_scope":
        return evaluate_delivery_scope(
            requirement,
            document=document,
            captured_delivery=captured_delivery,
            package_content=package_content,
            print_context=print_context,
            final=final,
        )
    raise WorkspaceError("Requirement has no supported inventory or software policy predicate")


def _frozen_inputs(
    store: Store, sid: str, owner: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read the objects to evaluate and their basis from one owner-scoped DB snapshot."""
    with store.connect() as con:
        con.execute("BEGIN")
        row = con.execute(
            "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        if row is None:
            raise WorkspaceError("workspace not found")
        session = json.loads(row[0])
        frozen = current_basis(store, con, sid, session)
        row = con.execute(
            "SELECT document FROM revisions WHERE session=? AND id=?", (sid, frozen["revision_id"])
        ).fetchone()
        revision = json.loads(row[0])
        if digest(revision) != frozen["revision_hash"]:
            raise WorkspaceError("Evaluated revision differs from its frozen snapshot")
        return session, revision, frozen


def aggregate(contract: Contract, results: list[dict[str, Any]]) -> tuple[str, bool]:
    """Require exactly one result per requirement; no vacuous or duplicate proof."""
    expected = [r.id for r in contract.requirements]
    if [r.get("requirement_id") for r in results] != expected:
        raise WorkspaceError("Verification results do not exactly cover the frozen contract")
    if any(r.get("status") not in {"pass", "fail", "unverified"} for r in results):
        raise WorkspaceError("Invalid verification result status")
    critical = [
        result for req, result in zip(contract.requirements, results, strict=True) if req.critical
    ]
    status = (
        "fail"
        if any(r["status"] == "fail" for r in critical)
        else "unverified"
        if not critical or any(r["status"] != "pass" for r in critical)
        else "pass"
    )
    return status, status == "pass"


def evaluate_digital_delivery(
    requirement: Any,
    *,
    captured_delivery: dict[str, Any],
    guidance_context: dict[str, Any],
    document: DesignDocument,
    package_content: bytes,
    print_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Consume existing byte-verified readers, then check the exact admitted artifact scope."""
    from forge_mind.workspace.handoff import referenced_import_asset_ids
    from forge_mind.workspace.instruction_plan import resolve_instruction_plan
    from forge_mind.workspace.production import production_report
    from forge_mind.workspace.verification_admission import criterion_hash

    criterion = requirement.verification.model_dump(mode="json")
    checksum = criterion_hash(criterion)
    result = _result(
        "unverified", "No exact current captured package", "digital_delivery_v1", checksum
    )
    receipt = captured_delivery.get("receipt") or {}
    if (
        captured_delivery.get("status") != "delivered"
        or captured_delivery.get("readback_verified") is not True
        or receipt.get("sha256") != _sha(package_content)
        or receipt.get("bytes") != len(package_content)
    ):
        return result
    with ZipFile(io.BytesIO(package_content)) as archive:
        members = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        saved_document = json.loads(archive.read("document.json"))
        if DesignDocument.model_validate(saved_document) != document:
            return {
                **result,
                "reason": "Captured native document differs from the selected document",
            }
        made = production_report(document)["make"]
        if not made:
            return {**result, "reason": "No made definitions establish all-made delivery coverage"}
        deliverable = criterion["deliverable"]
        if deliverable in {"project_stl", "project_step"}:
            extension = "stl" if deliverable == "project_stl" else "step"
            production = json.loads(archive.read("production.json"))
            quantities = {p["part_id"]: p["quantity"] for p in production["make"]}
            expected = {p["part_id"]: p["quantity"] for p in made}
            paths = {
                f"make/definition_{p['part_id']}.{extension}"
                if document.assembly
                else f"make/part.{extension}"
                for p in made
            }
            actual = {
                name
                for name in members
                if name.startswith("make/") and name.endswith("." + extension)
            }
            if (
                quantities != expected
                or actual != paths
                or (extension == "stl" and manifest.get("include_stl") is not True)
                or any(
                    _sha(archive.read(name))
                    != captured_delivery.get("artifact_hashes", {}).get(name.removeprefix("make/"))
                    for name in paths
                )
            ):
                return {
                    **result,
                    "reason": "Captured made file bytes, membership or quantities are incomplete",
                }
            result["facts"] = {"made_quantities": expected, "files": sorted(paths)}
        elif deliverable == "editable_native":
            assets = referenced_import_asset_ids(document.model_dump(mode="json"))
            required = {f"inputs/input_{asset_id}.step" for asset_id in assets}
            if assets:
                required.update({"inputs/assets.json", "import-assets.json"})
            if required - members:
                return {
                    **result,
                    "reason": "Captured editable declarations lack referenced native inputs",
                }
            result["facts"] = {"document": "document.json", "imported_inputs": sorted(required)}
        elif deliverable == "prepared_print_package":
            context = print_context or {}
            record = context.get("record") or {}
            package = record.get("package") or {}
            files = record.get("files") or {}
            actual_quantities = {entry["part_id"]: entry["quantity"] for entry in files.values()}
            expected = {p["part_id"]: p["quantity"] for p in made}
            if (
                context.get("status") != "current"
                or context.get("readback_verified") is not True
                or actual_quantities != expected
                or len(files) != len(expected)
                or package.get("revision_id") != receipt.get("revision_id")
            ):
                return {**result, "reason": "Exact all-made prepared print readback is not current"}
            result["facts"] = {"made_quantities": expected, "files": sorted(files)}
            result["evidence_refs"].append(
                {"kind": "print_package", "receipt": package, "sha256": digest(package)}
            )
        elif deliverable == "bound_instruction_plan":
            binding = resolve_instruction_plan(guidance_context, members=members)
            if "customer-guidance.json" not in members:
                return {**result, "reason": "Captured package lacks typed instruction records"}
            packaged = json.loads(archive.read("customer-guidance.json"))
            # The exact handoff reader verifies this context. Require the current source record too.
            current = guidance_context.get("record") or {}
            if (
                guidance_context.get("status") != "current"
                or binding.get("status") != "bound"
                or not current.get("instruction_plan")
                or packaged.get("record") != current
            ):
                return {
                    **result,
                    "reason": "Captured typed instructions are missing, stale or unbound",
                }
            result["facts"] = {
                "instruction_plan_hash": current.get("instruction_plan_hash"),
                "basis_hash": guidance_context.get("basis_hash"),
            }
        else:
            return {**result, "reason": "Unsupported digital verification predicate"}
    result.update(
        status="pass", reason="Exact current all-made " + deliverable + " captured and read back"
    )
    result["evidence_refs"].append(
        {"kind": "project_package", "receipt": receipt, "sha256": digest(receipt)}
    )
    return result


def _intent_authority(session: dict[str, Any], contract: Contract) -> None:
    from forge_mind.workspace.intent_attribution import IntentAttribution
    from forge_mind.workspace.intent_attribution_service import _record, request_text
    from forge_mind.workspace.verification_admission import validate_verification_contract

    record = _record(session)
    from forge_mind.workspace.photo_nominal import validate_photo_source_replay

    validate_photo_source_replay(
        session,
        session.get("job_id", ""),
        request_text(session),
        contract,
        IntentAttribution.model_validate((record or {}).get("attribution") or {}),
    )
    explicit = all(
        r.source == "explicit" for r in contract.requirements if r.verification is not None
    )
    if explicit:
        validate_verification_contract(contract, request_text(session), None, allow_explicit=True)
        return
    if (
        not record
        or record.get("status") != "interpreted"
        or record.get("contract_hash") != contract.fingerprint()
        or record.get("source_hash") != digest(request_text(session))
        or record.get("source_text") != request_text(session)
    ):
        raise WorkspaceError("Frozen typed criteria lack current source-bound intent attribution")
    validate_verification_contract(
        contract, request_text(session), IntentAttribution.model_validate(record["attribution"])
    )


def evaluate_frozen_geometry(
    service: WorkspaceService,
    sid: str,
    rid: str,
    requirement: Any,
    plan: DesignPlan,
    document: DesignDocument,
    measured: Measurement,
    directory: Path,
    *,
    deadline: float | None,
    cancelled: Callable[[], bool],
    photo_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One saved-STEP evaluator shared by author inspection and final proof."""
    from forge_mind.workspace.verification_admission import (
        criterion_hash,
        resolve_verification_subject,
    )

    if cancelled() or expired(deadline):
        raise WorkspaceError("Frozen geometry inspection cancelled or deadline exhausted")
    if service.executor.mode != "trusted_local" and not service.executor.typed_container_verified:
        raise WorkspaceError("Nominal proof worker is private-local until container qualification")
    criterion = requirement.verification
    if criterion.kind == "nominal_photo_solid":
        return _evaluate_photo_geometry(
            service,
            sid,
            rid,
            requirement,
            plan,
            document,
            measured,
            directory,
            deadline=deadline,
            cancelled=cancelled,
            cache=photo_cache,
        )
    if criterion.kind not in {"nominal_perforated_box", "nominal_through_hole_pattern"}:
        raise WorkspaceError("Saved geometry worker requires a geometric criterion")
    resolved = resolve_verification_subject(requirement, plan, document)
    artifact = resolved["step_artifact"]
    checksum_step = measured.artifact_hashes.get(artifact)
    if not checksum_step:
        raise WorkspaceError("The explicitly bound definition has no saved STEP artifact")
    directory.mkdir(parents=True, exist_ok=False)
    copy_artifact(service.store.directory(sid, rid) / artifact, directory / "part.step")
    request = {
        key: value
        for key, value in criterion.model_dump(mode="json").items()
        if key not in {"version", "kind", "scope", "source_quote", "subject_role"}
    }
    box = criterion.kind == "nominal_perforated_box"
    protocol = "nominal_perforated_box_v1" if box else "nominal_through_hole_pattern_v1"
    operation = "nominal_perforated_box" if box else "nominal_through_pattern"
    checksum = criterion_hash(criterion.model_dump(mode="json"))
    request.update(protocol=protocol, criterion_hash=checksum, part_id=resolved["part_id"])
    process = (
        process_revision
        if service.executor.mode == "trusted_local"
        else service.executor.process_observation
    )
    result = process(
        directory, {"operation": operation, "request": request}, time_left(deadline), cancelled
    )
    if cancelled() or expired(deadline):
        raise WorkspaceError("Frozen geometry inspection cancelled or deadline exhausted")
    if (
        result.get("criterion_hash") != checksum
        or (result.get("facts") or {}).get("step_sha256") != checksum_step
    ):
        raise WorkspaceError(
            "Nominal worker output does not bind its frozen criterion and exact STEP"
        )
    if (
        result.get("status") not in {"pass", "fail", "unverified"}
        or result.get("method") != protocol
    ):
        raise WorkspaceError("Malformed nominal proof result")
    # Worker basenames belong to its staging directory, not the revision artifact namespace.
    references = result.get("evidence_refs", [])
    if (
        not isinstance(references, list)
        or len(references) > 1
        or any(
            not isinstance(reference, dict)
            or set(reference) != {"artifact", "sha256"}
            or reference.get("artifact") != "part.step"
            or reference.get("sha256") != checksum_step
            for reference in references
        )
    ):
        raise WorkspaceError("Nominal worker evidence references differ from the staged STEP")
    result["evidence_refs"] = [
        {
            "kind": "saved_step",
            "revision_id": rid,
            "artifact": artifact,
            "sha256": checksum_step,
            "engine_fingerprint": measured.engine_fingerprint,
        }
    ]
    return result


def _evaluate_photo_geometry(
    service: WorkspaceService,
    sid: str,
    rid: str,
    requirement: Any,
    plan: DesignPlan,
    document: DesignDocument,
    measured: Measurement,
    directory: Path,
    *,
    deadline: float | None,
    cancelled: Callable[[], bool],
    cache: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """One immutable whole-body worker per exact target/STEP within this call."""
    from copy import deepcopy

    from forge_mind.workspace.photo_generation import photo_closure, read_photo_target
    from forge_mind.workspace.photo_nominal import validate_photo_target_constraints
    from forge_mind.workspace.verification_admission import (
        criterion_hash,
        resolve_verification_subject,
    )

    criterion = requirement.verification
    # The enclosing inspector/finalizer already authenticated this workspace.
    # Never accept the owner or target from a model-controlled worker request.
    with service.store.connect() as con:
        row = con.execute("SELECT owner FROM sessions WHERE id=?", (sid,)).fetchone()
    if row is None:
        raise WorkspaceError("Photo proof workspace disappeared")
    owner = row[0]
    resolved = resolve_verification_subject(requirement, plan, document)
    artifact = resolved["step_artifact"]
    step_hash = measured.artifact_hashes.get(artifact)
    target = read_photo_target(service, sid, owner, rid, criterion.subject_id)
    validate_photo_target_constraints(criterion, target["source_constraints"])
    closure = photo_closure(service, sid, owner, rid)
    expected_basis = target["current_basis"]
    if (
        target["part_id"] != resolved["part_id"]
        or not step_hash
        or expected_basis.get("step_sha256") != step_hash
        or expected_basis.get("revision_id") != rid
        or expected_basis.get("engine_fingerprint") != measured.engine_fingerprint
        or closure.get("status") != "checked"
        or closure.get("basis") != expected_basis
        or closure.get("target_sha256") != target["target_sha256"]
    ):
        raise WorkspaceError(
            "Photo nominal proof lacks exact source constraints/current checked target"
        )
    key = digest(
        {
            "target": target,
            "closure": closure,
            "step_sha256": step_hash,
            "source_constraints": target["source_constraints"],
            "engine_fingerprint": measured.engine_fingerprint,
        }
    )
    checksum = criterion_hash(criterion.model_dump(mode="json"))
    if cache is not None and key in cache:
        result = deepcopy(cache[key])
    else:
        directory.mkdir(parents=True, exist_ok=False)
        copy_artifact(service.store.directory(sid, rid) / artifact, directory / "part.step")
        request = {
            "protocol": "nominal_photo_solid_v1",
            "criterion_hash": key,
            "subject_id": criterion.subject_id,
            "part_id": resolved["part_id"],
            "frame": "definition_local",
            "target_sha256": target["target_sha256"],
            "expected_document": target["proposal_document"],
            "source_constraints": target["source_constraints"],
        }
        process = (
            process_revision
            if service.executor.mode == "trusted_local"
            else service.executor.process_observation
        )
        result = process(
            directory,
            {"operation": "nominal_photo_solid", "request": request},
            time_left(deadline),
            cancelled,
        )
        facts = result.get("facts") or {}
        if (
            result.get("criterion_hash") != key
            or result.get("method") != "nominal_photo_solid_v1"
            or result.get("status") not in {"pass", "fail", "unverified"}
            or facts.get("step_sha256") != step_hash
            or facts.get("target_sha256") != target["target_sha256"]
            or facts.get("expected_document_sha256") != digest(target["proposal_document"])
            or facts.get("source_constraints") != target["source_constraints"]
            or result.get("evidence_refs") != [{"artifact": "part.step", "sha256": step_hash}]
        ):
            raise WorkspaceError("Photo whole-body worker result differs from exact staged inputs")
        result["evidence_refs"] = [
            {
                "kind": "saved_step",
                "revision_id": rid,
                "artifact": artifact,
                "sha256": step_hash,
                "engine_fingerprint": measured.engine_fingerprint,
            }
        ]
        if cache is not None:
            cache[key] = deepcopy(result)
    if cancelled() or expired(deadline):
        raise WorkspaceError("Photo nominal proof cancelled or original deadline exhausted")
    if (
        read_photo_target(service, sid, owner, rid, criterion.subject_id) != target
        or photo_closure(service, sid, owner, rid) != closure
    ):
        raise WorkspaceError("Photo target changed during final whole-body proof")
    result["criterion_hash"] = checksum
    result["facts"] = {
        **result["facts"],
        "shared_evaluation_hash": key,
        "proved_claims": criterion.claims,
        "photo_comparison_receipt_sha256": closure["receipt_sha256"],
    }
    return result


def inspect_requirements(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str,
    *,
    deadline: float | None,
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    """Explicit author feedback, without target writes, delivery claims or state promotion."""
    from forge_mind.workspace.verification_admission import (
        criterion_hash,
        validate_verification_bindings,
    )

    if cancelled() or expired(deadline):
        raise WorkspaceError("Frozen requirement inspection cancelled or deadline exhausted")
    session, revision, frozen = _frozen_inputs(service.store, sid, owner)
    if session.get("selected_revision") != rid:
        raise WorkspaceError("Inspect requirements on the current selected revision only")
    contract = Contract.model_validate(session["contract"])
    if has_verification(session["contract"]):
        _intent_authority(session, contract)
    document = DesignDocument.model_validate(revision["document"])
    plan = DesignPlan.model_validate(session["design_plan"])
    validate_verification_bindings(plan, contract, document)
    if (
        frozen["contract_hash"] != frozen["revision_contract_hash"]
        or frozen["engine_fingerprint"] != frozen["measured_engine_fingerprint"]
    ):
        raise WorkspaceError("Frozen requirement inspection has stale source or engine authority")
    measured = Measurement.model_validate(revision["measurement"])
    findings = {finding.requirement_id: finding for finding in measured.findings}
    results: list[dict[str, Any]] = []
    photo_cache: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="forge-frozen-requirements-") as scratch:
        for requirement in contract.requirements:
            if cancelled() or expired(deadline):
                raise WorkspaceError(
                    "Frozen requirement inspection cancelled or deadline exhausted"
                )
            criterion = requirement.verification
            checksum = criterion_hash(criterion.model_dump(mode="json")) if criterion else None
            if criterion is None:
                finding = findings[requirement.id]
                result = _result(
                    "unverified" if requirement.metric == "unmeasured" else finding.status,
                    finding.reason,
                    "legacy_measurement",
                )
            elif criterion.kind == "digital_delivery":
                result = _result(
                    "unverified",
                    "Author inspection does not establish final captured digital delivery",
                    "digital_delivery_v1",
                    checksum,
                )
            elif criterion.kind in {"nominal_subject_inventory", "nominal_delivery_scope"}:
                result = evaluate_frozen_policy(requirement, plan=plan, document=document)
            else:
                result = evaluate_frozen_geometry(
                    service,
                    sid,
                    rid,
                    requirement,
                    plan,
                    document,
                    measured,
                    Path(scratch) / str(len(results)),
                    deadline=deadline,
                    cancelled=cancelled,
                    photo_cache=photo_cache,
                )
            results.append(
                {**result, "requirement_id": requirement.id, "critical": requirement.critical}
            )
    with service.store.connect() as con:
        con.execute("BEGIN")
        row = con.execute(
            "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        if current_basis(service.store, con, sid, json.loads(row[0])) != frozen:
            raise WorkspaceError(
                "Frozen requirement inspection inputs changed; read the current basis"
            )
    return {
        "status": "scoped_results",
        "revision_id": rid,
        "basis_hash": digest(frozen),
        "results": results,
        "physical_readiness": "unverified",
        "scope": "Current frozen predicates only; inventory and software-boundary obligations await final capture. No targets, plan, selection, delivery or acceptance state changed. Final proof still runs after capture/readback.",
    }


def verify_contract(
    service: WorkspaceService,
    sid: str,
    owner: str,
    jid: str,
    *,
    deadline: float | None,
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    """Build an immutable proof after actual delivery, within the original job deadline."""
    from forge_mind.workspace.delivery_guidance import guidance_context
    from forge_mind.workspace.native_delivery import current_delivery_binding
    from forge_mind.workspace.native_finish import finish_context
    from forge_mind.workspace.print_setup_service import print_setup_context, read_print_export
    from forge_mind.workspace.project_delivery import read_project_export
    from forge_mind.workspace.verification_admission import (
        criterion_hash,
        validate_verification_bindings,
    )

    def stopped() -> None:
        if cancelled() or expired(deadline):
            raise WorkspaceError("Final verification cancelled or original deadline exhausted")

    stopped()
    session, revision, frozen = _frozen_inputs(service.store, sid, owner)
    if session.get("job_id") != jid:
        raise WorkspaceError("Verification job no longer owns the session")
    rid = _selected_revision(session)
    contract = Contract.model_validate(session["contract"])
    if not has_requirement_status(session["contract"]):
        raise WorkspaceError("Empty or legacy scalar contracts retain their existing evidence path")
    _intent_authority(session, contract)
    document = DesignDocument.model_validate(revision["document"])
    plan = DesignPlan.model_validate(session["design_plan"])
    validate_verification_bindings(plan, contract, document)
    if (
        frozen["contract_hash"] != frozen["revision_contract_hash"]
        or frozen["engine_fingerprint"] != frozen["measured_engine_fingerprint"]
    ):
        raise WorkspaceError("Selected revision has a stale frozen contract or engine")
    review = session.get("native_review") or {}
    guide = guidance_context(service, sid, owner, rid)
    closure = finish_context(service, sid, owner, rid, require_bound_instructions=True)
    closure_ready = (
        review.get("job_id") == jid
        and review.get("status") == closure["status"] == "ready_for_validation"
        and review.get("revision_id") == rid
        and review.get("basis_hash") == closure["basis_hash"]
        and review.get("plan_hash") == closure["plan_hash"]
        and frozen["incomplete_finish_job_id"] != jid
        and frozen["authority_conflict_job_id"] != jid
    )
    delivery = session.get("automatic_delivery") or {}
    content = b""
    if delivery.get("status") == "delivered" and delivery.get("job_id") == jid:
        if any(
            delivery.get(key) != value
            for key, value in current_delivery_binding(service, sid, owner).items()
        ):
            raise WorkspaceError("Captured delivery basis differs from current final proof inputs")
        receipt = delivery.get("receipt") or {}
        package_id, package_hash = receipt.get("package_id"), receipt.get("sha256")
        if (
            not isinstance(package_id, str)
            or re.fullmatch(r"[a-f0-9]{32}", package_id) is None
            or not isinstance(package_hash, str)
            or re.fullmatch(r"[a-f0-9]{64}", package_hash) is None
        ):
            raise WorkspaceError("Captured delivery has no exact package identity and digest")
        content, readback = read_project_export(service, sid, owner, package_id, package_hash)
        if readback != receipt or receipt.get("revision_id") != rid:
            raise WorkspaceError("Final delivery does not bind the selected revision")
    print_context: dict[str, Any] = {}
    if any(
        getattr(r.verification, "deliverable", None) == "prepared_print_package"
        for r in contract.requirements
    ) or (
        rid in (session.get("print_setup_records") or {})
        and any(
            getattr(r.verification, "kind", None) == "nominal_delivery_scope"
            for r in contract.requirements
        )
    ):
        print_context = print_setup_context(service, sid, owner, rid)
        if print_context.get("status") == "current":
            package = print_context["record"]["package"]
            _, readback = read_print_export(
                service, sid, owner, package["package_id"], package["sha256"]
            )
            print_context["readback_verified"] = readback == package
    report_id = uuid.uuid4().hex
    path = _report_path(service.store, sid, report_id)
    path.parent.mkdir(parents=True, exist_ok=False)
    measured = Measurement.model_validate(revision["measurement"])
    old = {f.requirement_id: f for f in measured.findings}
    results: list[dict[str, Any]] = []
    photo_cache: dict[str, dict[str, Any]] = {}
    for requirement in contract.requirements:
        stopped()
        criterion = requirement.verification
        checksum = criterion_hash(criterion.model_dump(mode="json")) if criterion else None
        try:
            if criterion is None:
                finding = old.get(requirement.id)
                if requirement.metric == "unmeasured" or finding is None:
                    result = _result(
                        "unverified",
                        "No supported frozen predicate proves this requirement",
                        "legacy_measurement",
                    )
                else:
                    result = _result(finding.status, finding.reason, "legacy_measurement")
                    result["facts"] = {"finding": finding.model_dump(mode="json")}
            elif criterion.kind == "digital_delivery":
                result = evaluate_digital_delivery(
                    requirement,
                    captured_delivery=delivery,
                    guidance_context=guide,
                    document=document,
                    package_content=content,
                    print_context=print_context,
                )
            elif criterion.kind in {"nominal_subject_inventory", "nominal_delivery_scope"}:
                result = evaluate_frozen_policy(
                    requirement,
                    plan=plan,
                    document=document,
                    captured_delivery=delivery,
                    package_content=content,
                    print_context=print_context,
                    final=True,
                )
            else:
                result = evaluate_frozen_geometry(
                    service,
                    sid,
                    rid,
                    requirement,
                    plan,
                    document,
                    measured,
                    path.parent / ("geometry-" + str(len(results))),
                    deadline=deadline,
                    cancelled=cancelled,
                    photo_cache=photo_cache,
                )
        except (ValueError, OSError, KeyError, TypeError, BadZipFile, WorkspaceError) as exc:
            method = verification_method(criterion)
            result = _result("unverified", str(exc)[:1000], method, checksum)
        results.append(
            {**result, "requirement_id": requirement.id, "critical": requirement.critical}
        )
    stopped()
    status, satisfied = aggregate(contract, results)
    from forge_mind.workspace.customer_interpretation import customer_nominal_conformance
    from forge_mind.workspace.intent_attribution import IntentAttribution

    conformance = customer_nominal_conformance(
        IntentAttribution.model_validate(
            (session.get("intent_attribution") or {}).get("attribution") or {}
        ),
        results,
    )
    report = {
        "version": 1,
        "report_id": report_id,
        "project_id": sid,
        "job_id": jid,
        "revision_id": rid,
        "basis": frozen,
        "at": time.time(),
        "results": results,
        "status": status,
        "requirements_satisfied": satisfied,
        "author_closure_ready": closure_ready,
        "physical_readiness": "unverified",
        "scope": SCOPE,
        **({"cad_conformance": conformance} if conformance is not None else {}),
    }
    data = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(data) > MAX_REPORT_BYTES:
        raise WorkspaceError("Verification report exceeds its transfer bound")
    with path.open("xb") as stream:
        stream.write(data)
    receipt = {
        "report_id": report_id,
        "job_id": jid,
        "revision_id": rid,
        "bytes": len(data),
        "sha256": _sha(data),
        "basis_hash": digest(frozen),
    }
    with service.store.connect() as con:
        con.execute("BEGIN")
        row = con.execute(
            "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        if current_basis(service.store, con, sid, json.loads(row[0])) != frozen:
            raise WorkspaceError(
                "Final verification inputs changed; immutable report remains historical"
            )
    stopped()
    return receipt


def checked_report(
    store: Store,
    con: sqlite3.Connection,
    sid: str,
    session: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Verify bytes and live basis inside finish's transaction or a current read transaction."""
    path = _report_path(store, sid, receipt["report_id"])
    if not regular_file(path, MAX_REPORT_BYTES):
        raise WorkspaceError("Verification report is missing, linked or oversized")
    data = path.read_bytes()
    if len(data) != receipt.get("bytes") or _sha(data) != receipt.get("sha256"):
        raise WorkspaceError("Verification report bytes differ from their receipt")
    raw_report = json.loads(data)
    if not isinstance(raw_report, dict) or any(not isinstance(key, str) for key in raw_report):
        raise WorkspaceError("Verification report must be an object with named fields")
    report: dict[str, Any] = raw_report
    basis = current_basis(store, con, sid, session)
    if (
        report.get("version") != 1
        or report.get("report_id") != receipt.get("report_id")
        or report.get("project_id") != sid
        or report.get("job_id") != receipt.get("job_id")
        or report.get("revision_id") != receipt.get("revision_id")
        or report.get("revision_id") != basis["revision_id"]
        or receipt.get("job_id") != session.get("job_id")
        or report.get("basis") != basis
        or digest(basis) != receipt.get("basis_hash")
        or report.get("physical_readiness") != "unverified"
        or report.get("scope") != SCOPE
    ):
        raise WorkspaceError("Verification report identity or live basis changed")
    contract = Contract.model_validate(session["contract"])
    _intent_authority(session, contract)
    status, satisfied = aggregate(contract, report.get("results", []))
    if report.get("status") != status or report.get("requirements_satisfied") is not satisfied:
        raise WorkspaceError("Verification report aggregate is inconsistent")
    from forge_mind.workspace.customer_interpretation import customer_nominal_conformance
    from forge_mind.workspace.intent_attribution import IntentAttribution

    conformance = customer_nominal_conformance(
        IntentAttribution.model_validate(
            (session.get("intent_attribution") or {}).get("attribution") or {}
        ),
        report["results"],
    )
    if report.get("cad_conformance") != conformance:
        raise WorkspaceError("CAD conformance display differs from its current interpreted targets")
    from forge_mind.workspace.verification_admission import criterion_hash

    for req, result in zip(contract.requirements, report["results"], strict=True):
        expected = (
            criterion_hash(req.verification.model_dump(mode="json")) if req.verification else None
        )
        if result.get("criterion_hash") != expected or result.get("critical") is not req.critical:
            raise WorkspaceError(
                "Verification result criterion differs from the frozen requirement"
            )
        if (
            req.metric == "unmeasured"
            and req.verification is None
            and result["status"] != "unverified"
        ):
            raise WorkspaceError("Unsupported requirement cannot acquire a final pass")
        method = verification_method(req.verification)
        if result.get("method") != method:
            raise WorkspaceError("Verification method differs from the admitted predicate")
    review = session.get("native_review") or {}
    if report.get("author_closure_ready") is True and (
        review.get("job_id") != session.get("job_id")
        or review.get("status") != "ready_for_validation"
        or review.get("revision_id") != basis["revision_id"]
        or review.get("plan_hash") != basis["plan_hash"]
        or review.get("reasons")
    ):
        raise WorkspaceError("Final proof lacks this job's completed explicit native review")
    return report


def read_contract_verification(service: WorkspaceService, sid: str, owner: str) -> dict[str, Any]:
    """Current proof view stays outside pre-capture guidance and package hash bases."""
    return read_verification(service.store, sid, owner)


def read_verification(store: Store, sid: str, owner: str) -> dict[str, Any]:
    """Owner-scoped exact sidecar reader, shared by HTTP/MCP and compact overview."""
    with store.connect() as con:
        con.execute("BEGIN")
        row = con.execute(
            "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        if row is None:
            raise WorkspaceError("workspace not found")
        session = json.loads(row[0])
        receipt = session.get("contract_verification_report")
        if not receipt:
            return {"status": "unavailable", "physical_readiness": "unverified", "scope": SCOPE}
        return {**checked_report(store, con, sid, session, receipt), "receipt": receipt}


def verification_summary(store: Store, sid: str, owner: str) -> dict[str, Any]:
    """Derived display only; never part of guide facts or an immutable package basis."""
    try:
        report = read_verification(store, sid, owner)
        return {
            "evidence_stage": "postcapture_final_verification",
            "finding_counts": {
                state: sum(result["status"] == state for result in report.get("results", []))
                for state in ("pass", "fail", "unverified")
            },
            **{
                key: report[key]
                for key in (
                    "status",
                    "report_id",
                    "job_id",
                    "revision_id",
                    "requirements_satisfied",
                    "receipt",
                    "physical_readiness",
                    "scope",
                    "cad_conformance",
                )
                if key in report
            },
        }
    except (ValueError, OSError, KeyError, TypeError, WorkspaceError) as exc:
        return {
            "status": "unverified",
            "reason": str(exc)[:1000],
            "physical_readiness": "unverified",
            "scope": SCOPE,
        }
