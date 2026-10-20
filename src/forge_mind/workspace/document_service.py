"""Immutable CAD candidates, protection checks, and bounded document jobs."""

from __future__ import annotations

import difflib
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.budget import expired, time_left
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.measure import engine_fingerprint, verify_artifacts
from forge_mind.workspace.models import Contract, Finding, Measurement
from forge_mind.workspace.processing import copy_artifact, process_revision, regular_file
from forge_mind.workspace.store import WorkspaceError
from forge_mind.workspace.topology import resolve
from forge_mind.workspace.typed_cache import cacheable_document, parent_cache_input

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.store import Store


def read_saved_record(service: WorkspaceService, sid: str, rid: str, name: str) -> dict[str, Any]:
    """Read one literal saved record, preserving the HTTP artifact integrity boundary.

    Callers establish owner authority. This reader checks only this requested file;
    an agent may separately require all revision artifacts through _load_revision.
    """
    if name not in {"evidence.json", "contract.json"}:
        raise ValueError("Only saved evidence and contract records can be read here")
    revision = service.store.revision(sid, rid)
    directory = service.store.directory(sid, rid)
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
        raise FileNotFoundError("Saved record is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Saved record is not valid JSON") from exc
    expected = revision["measurement" if name == "evidence.json" else "contract"]
    if not isinstance(value, dict) or json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) != json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False):
        raise ValueError("Artifact no longer matches its revision record")
    return value


def read_topology(service: WorkspaceService, sid: str, rid: str) -> dict[str, Any]:
    revision = service._load_revision(sid, rid)
    directory = service.store.directory(sid, rid)
    if "topology.json" not in revision["measurement"]["artifact_hashes"]:
        raise WorkspaceError(
            "Revision predates selectable topology; rebuild or import it as a base"
        )
    result: dict[str, Any] = json.loads((directory / "topology.json").read_text(encoding="utf-8"))
    return result


def asset_paths(service: WorkspaceService, sid: str, owner: str) -> dict[str, Path]:
    return owned_asset_paths(service.store, sid, owner)


def owned_asset_paths(store: Store, sid: str, owner: str) -> dict[str, Path]:
    session = store.get(sid, owner)
    result = {}
    for item in session.get("assets", []):
        path = store.root / "assets" / sid / (item["id"] + ".step")
        if not regular_file(path) or file_sha256(path) != item["sha256"]:
            raise WorkspaceError("Imported base changed or is unavailable")
        result[item["id"]] = path
    return result


def preflight_document(
    service: WorkspaceService,
    sid: str,
    owner: str,
    document: DesignDocument,
    contract: Contract,
    parent: str | None,
    plan_snapshot: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, dict[str, Path]]:
    """Validate existing non-CAD authority and assets before approval or allocation."""
    from forge_mind.workspace.authoring_policy import enforce_document_policy, source_classification

    mode = service.store.authoring_mode(sid)
    enforce_document_policy(document, mode)
    operation_record = None
    captured = None
    if mode == "native_operations":
        from forge_mind.workspace.models import DesignPlan
        from forge_mind.workspace.operation_plan import coverage, validate_plan

        captured = plan_snapshot or service.store.plan_snapshot(sid, owner)
        if captured is None:
            raise WorkspaceError(
                "Save an operation design plan and requirements before construction"
            )
        if captured["contract_hash"] != contract.fingerprint():
            raise WorkspaceError("Operation plan contract differs from the build contract")
        validate_plan(DesignPlan.model_validate(captured["plan"]), contract, document=document)
        operation_record = {
            **coverage(captured, document),
            "source": source_classification(document),
        }
    if service.executor.mode != "trusted_local" and not service.executor.typed_container_verified:
        raise WorkspaceError(
            "Typed CAD worker is private-local only until its Linux container is verified"
        )
    available = asset_paths(service, sid, owner)
    all_features = (
        [f for p in document.assembly.parts for f in p.features]
        if document.assembly
        else document.features
    )
    if any(feature.asset_id and feature.asset_id not in available for feature in all_features):
        raise WorkspaceError("Feature references an unowned imported asset")
    if parent:
        service._load_revision(sid, parent)
    return mode, captured, operation_record, available


def build_document(
    service: WorkspaceService,
    sid: str,
    owner: str,
    document: DesignDocument,
    contract: Contract,
    parent: str | None,
    explanation: str,
    timeout: float | None,
    cancelled: Callable[[], bool],
    plan_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from forge_mind.workspace.photo_generation import require_grounded

    require_grounded(service, sid, owner)
    _mode, captured, operation_record, available = preflight_document(
        service, sid, owner, document, contract, parent, plan_snapshot
    )
    from forge_mind.workspace.customer_construction import guard_document

    guard_document(service, sid, owner, document, contract, parent, explanation, captured)
    rid = uuid.uuid4().hex
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    (directory / "document.json").write_text(document.model_dump_json(indent=2), encoding="utf-8")
    assets = {}
    all_features = (
        [f for p in document.assembly.parts for f in p.features]
        if document.assembly
        else document.features
    )
    for feature in all_features:
        if feature.asset_id and feature.asset_id not in assets:
            if feature.asset_id not in available:
                raise WorkspaceError("Feature references an unowned imported asset")
            target = directory / ("input_" + feature.asset_id + ".step")
            copy_artifact(available[feature.asset_id], target)
            assets[feature.asset_id] = str(target)
    code = "from pathlib import Path\nfrom build123d import export_step\nfrom forge_mind.workspace.document import DesignDocument\nfrom forge_mind.workspace.kernel import compile_document\n"
    code += f"document = DesignDocument.model_validate({document.model_dump()!r})\n"
    code += f"assets = {{key: Path(__file__).parent / ('input_' + key + '.step') for key in {list(assets)!r}}}\n"
    code += (
        "result, history = compile_document(document, assets)\nexport_step(result, 'part.step')\n"
    )
    (directory / "program.py").write_text(code, encoding="utf-8")
    service.store.event(sid, "build", explanation, revision_id=rid)
    started = time.monotonic()
    measured = Measurement(
        revision_id=rid,
        contract_hash=contract.fingerprint(),
        engine_fingerprint=engine_fingerprint(),
        geometry_valid=False,
        artifact_hashes={
            name: file_sha256(directory / name) for name in ("program.py", "document.json")
        },
    )
    origin: dict[str, Any] = {}
    try:
        cache_source = None
        cache_input = None
        if document.assembly and parent:
            previous_revision = service._load_revision(sid, parent)
            if service.executor.mode == "docker" and cacheable_document(document.model_dump()):
                cache_input = parent_cache_input(
                    previous_revision,
                    service.store.directory(sid, parent),
                    directory,
                    service.executor.typed_container_image,
                )
            elif (
                service.executor.mode == "trusted_local"
                and "parts.json" in previous_revision["measurement"]["artifact_hashes"]
            ):
                cache_source = str(service.store.directory(sid, parent))
        payload = {
            "revision_id": rid,
            "contract": contract.model_dump(),
            "document": document.model_dump(),
            "assets": assets,
            "cache_source": cache_source,
        }
        if service.executor.mode == "trusted_local":
            result = process_revision(directory, payload, timeout, cancelled)
        else:
            result = service.executor.process_document(
                directory,
                payload,
                min(timeout, 600.0) if timeout is not None else timeout,
                cancelled,
                cache_input=cache_input,
                origin=origin,
            )
        measured = Measurement.model_validate(result)
    except (ValueError, OSError) as exc:
        measured.errors.append(str(exc))
        measured.findings = [
            Finding(
                requirement_id=r.id,
                status="unverified",
                expected=r.value,
                method="Feature regeneration failed",
                reason=str(exc),
            )
            for r in contract.requirements
        ]
    for filename in assets:
        name = "input_" + filename + ".step"
        measured.artifact_hashes[name] = file_sha256(directory / name)
    asset_records = [
        a for a in service.store.get(sid, owner).get("assets", []) if a["id"] in assets
    ]
    (directory / "assets.json").write_text(json.dumps(asset_records, indent=2), encoding="utf-8")
    measured.artifact_hashes["assets.json"] = file_sha256(directory / "assets.json")
    previous = service.store.revision(sid, parent)["code"] if parent else ""
    revision: dict[str, Any] = {
        "id": rid,
        "parent_id": parent,
        "at": time.time(),
        "explanation": explanation,
        "code": code,
        "document": document.model_dump(),
        "assets": asset_records,
        "parameters": document.variables,
        "editable_parameters": [],
        "diff": "".join(
            difflib.unified_diff(
                previous.splitlines(True),
                code.splitlines(True),
                fromfile=parent or "empty",
                tofile=rid,
            )
        ),
        "contract": contract.model_dump(),
        "measurement": measured.model_dump(),
        "execution_seconds": time.monotonic() - started,
        "execution_mode": "bounded_typed_document",
    }
    if origin:
        revision["typed_cache_origin"] = origin
    if operation_record is not None:
        revision["operation_record"] = {
            **operation_record,
            "plan_snapshot": captured,
            "revision_id": rid,
            "source_hash": document.fingerprint(),
            "geometry_valid": measured.geometry_valid,
        }
        for name, data in (
            ("operation-plan.json", captured),
            ("operation-record.json", revision["operation_record"]),
        ):
            (directory / name).write_text(
                json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
            )
            measured.artifact_hashes[name] = file_sha256(directory / name)
        revision["measurement"] = measured.model_dump()
    (directory / "evidence.json").write_text(measured.model_dump_json(indent=2), encoding="utf-8")
    (directory / "contract.json").write_text(contract.model_dump_json(indent=2), encoding="utf-8")
    service.store.add_revision(sid, revision)
    service.store.event(
        sid,
        "measurement",
        "Measured document candidate",
        revision_id=rid,
        geometry_valid=measured.geometry_valid,
    )
    return revision


def _closure(document: dict[str, Any], feature_id: str) -> dict[str, Any]:
    if document.get("assembly"):
        part = next((p for p in document["assembly"]["parts"] if p["id"] == feature_id), None)
        if part is None:
            raise WorkspaceError("Protected part definition is missing")
        values = DesignDocument.model_validate(document).values()
        bindings = {v for f in part["features"] for v in f.get("bindings", {}).values()}
        bindings.update(
            v
            for feature in part["features"]
            for point in feature.get("points", [])
            for v in point
            if isinstance(v, str)
        )
        bindings.update(
            v
            for port in part.get("ports", [])
            for v in [port["diameter"], *port["origin"]]
            if isinstance(v, str)
        )
        return {"part": part, "bound_values": {v: values[v] for v in sorted(bindings)}}
    features = {f["id"]: f for f in document.get("features", [])}
    visited: dict[str, Any] = {}
    values = DesignDocument.model_validate(document).values()

    def visit(fid: str) -> None:
        if fid in visited:
            return
        if fid not in features:
            raise WorkspaceError("Protected feature is missing")
        feature = features[fid]
        visited[fid] = {
            **feature,
            "bound_values": {k: values[v] for k, v in feature.get("bindings", {}).items()},
            "point_values": [
                [values[v] if isinstance(v, str) else v for v in point]
                for point in feature.get("points", [])
            ],
        }
        for dependency in feature.get("inputs", []):
            visit(dependency)

    visit(feature_id)
    return visited


def protection_results(
    service: WorkspaceService, sid: str, revision: dict[str, Any], inputs: EngineeringInputs
) -> list[dict[str, Any]]:
    results = []
    for protection in inputs.protections:
        ok, reason = False, "Protected reference is unresolved"
        try:
            baseline = service._load_revision(sid, protection.revision_id)
            if protection.reference:
                topology = read_topology(service, sid, revision["id"])
                resolution = resolve(protection.reference.model_dump(), topology)
                ok = resolution["status"] == "resolved"
                reason = (
                    "Exact complete B-rep record is uniquely preserved"
                    if ok
                    else "Protected geometry changed or is ambiguous"
                )
            elif protection.feature_id:
                fid = protection.feature_id
                a, b = baseline.get("document"), revision.get("document")
                ok = bool(a and b and _closure(a, fid) == _closure(b, fid))
                histories = [
                    r["measurement"].get("facts", {}).get("feature_history", [])
                    for r in (baseline, revision)
                ]
                signatures = [
                    next(
                        (
                            f.get("signature")
                            or (f.get("key") if a and a.get("assembly") else None)
                            for f in history
                            if f["id"] == fid
                        ),
                        None,
                    )
                    for history in histories
                ]
                ok = ok and bool(signatures[0]) and signatures[0] == signatures[1]
                reason = (
                    "Feature and dependency closure regenerated identically; downstream geometry is protected separately by face/body locks"
                    if ok
                    else "Protected feature/dependency/evaluated stage changed"
                )
        except (ValueError, OSError, KeyError) as exc:
            reason = str(exc)
        results.append({"id": protection.id, "status": "pass" if ok else "fail", "reason": reason})
    return results


def finish_preview(
    service: WorkspaceService, job: dict[str, Any], candidate: str | None = None
) -> None:
    try:
        _finish_preview(service, job, candidate)
    except Exception as exc:
        service.store.event(job["session"], "error", f"Candidate review failed: {exc}")
        service.store.finish(job["id"], job["session"], "error")
        service.store.complete_preview(job["session"], job["id"], job.get("based_on"), None)


def _finish_preview(
    service: WorkspaceService, job: dict[str, Any], candidate: str | None = None
) -> None:
    sid, owner = job["session"], job["owner"]
    session = service.store.get(sid, owner)
    if session.get("pending_finish", session.get("state")) in {"needs_input", "needs_approval"} or (
        (session.get("intent_attribution") or {}).get("status") == "needs_input"
    ):
        # Paused intent has no authority to trigger fresh checks or acceptance.
        service.store.complete_preview(sid, job["id"], job.get("based_on"), None)
        return
    publication = service.store.feature_publication(sid, owner, job["id"])
    interrupted = service.store.cancelled(job["id"]) or session.get(
        "pending_finish", session["state"]
    ) in {"cancelled", "error", "budget_exhausted", "interrupted"}
    rid = (
        publication["revision_id"]
        if publication is not None and interrupted
        else candidate or session.get("selected_revision")
    )
    based_on = job.get("based_on")
    proposal = None
    if (
        rid
        and rid != based_on
        and (
            session["state"] not in {"cancelled", "error", "budget_exhausted"}
            or publication is not None
        )
    ):
        revision = service.store.revision(sid, rid)
        inputs = EngineeringInputs.model_validate(session["engineering"]["inputs"])
        checks = protection_results(service, sid, revision, inputs)
        # Acceptance baseline and selected editing source can differ: protect both,
        # including a first candidate that has never been accepted.
        for source_id in dict.fromkeys([based_on, job.get("revision_id")]):
            baseline = service._load_revision(sid, source_id) if source_id else None
            if baseline and (baseline.get("document") or {}).get("assembly"):
                from forge_mind.workspace.assembly import preserved_assembly

                preserved, reason = preserved_assembly(
                    baseline["document"], revision.get("document") or {}
                )
                checks.append(
                    {
                        "id": "assembly_intent:" + str(source_id),
                        "status": "pass" if preserved else "fail",
                        "reason": reason,
                    }
                )
            if source_id and baseline and (baseline.get("document") or {}).get("mechanism"):
                from forge_mind.workspace.mechanism import preserved_intent

                preserved, reason = preserved_intent(
                    baseline["document"], revision.get("document") or {}
                )
                checks.append(
                    {
                        "id": "mechanism_intent:" + source_id,
                        "status": "pass" if preserved else "fail",
                        "reason": reason
                        + ". Create a separate project to replace the mechanism family.",
                    }
                )
        measurement = Measurement.model_validate(revision["measurement"])
        eligible = (
            measurement.geometry_valid
            and verify_artifacts(service.store.directory(sid, rid), measurement)
            and all(c["status"] == "pass" for c in checks)
        )
        proposal = {
            "revision_id": rid,
            "based_on": based_on,
            "input_version": job.get("input_version", 0),
            "eligible": eligible,
            "protection_checks": checks,
            "engine_fingerprint": engine_fingerprint(),
            "contract": revision["contract"],
            "instructions": session.get("instructions", []),
        }
        from forge_mind.workspace.product_sources import read_sources

        proposal["source_fingerprint"] = read_sources(service.store, sid, owner)["fingerprint"]
    service.store.complete_preview(sid, job["id"], based_on, proposal)


def run_document_job(service: WorkspaceService, job: dict[str, Any]) -> None:
    from forge_mind.workspace.budget import end_time

    sid, owner = job["session"], job["owner"]
    candidate = None
    feature_applied = False
    deadline = end_time(job.get("max_seconds"))

    def cancelled() -> bool:
        return service.stop.is_set() or service.store.cancelled(job["id"])

    try:
        from forge_mind.workspace.operation_recovery import guard_job

        guard_job(service, job)
        session = service.store.get(sid, owner)
        if job.get("lab_binding") or job.get("lab_realization"):
            from forge_mind.workspace.lab_service import check_lab_job

            check_lab_job(service, job)
        service.store.update(sid, state="running")
        if job.get("engineering_operation") == "motion_check":
            from forge_mind.workspace.motion_service import run_motion_job

            run_motion_job(service, job, cancelled)
        elif job.get("engineering_operation") == "product_tool":
            from forge_mind.workspace.product_service import run_tool_job

            run_tool_job(service, job, cancelled)
        elif job.get("engineering_operation") == "design_study":
            from forge_mind.workspace.design_study_service import run_study_job

            run_study_job(service, job, cancelled)
        elif job.get("engineering_operation") == "structural_study":
            from forge_mind.workspace.structural_service import run_structural_job

            run_structural_job(service, job, cancelled)
        elif job.get("engineering_operation"):
            from forge_mind.workspace.engineering_service import run_report_job

            run_report_job(service, job, cancelled)
        else:
            document = DesignDocument.model_validate(job["document"])
            contract = Contract.model_validate(job.get("contract") or session.get("contract") or {})
            if job.get("feature_edit_basis") is None:
                service.store.update(sid, contract=contract.model_dump())
            from forge_mind.workspace.customer_construction import stage_construction
            from forge_mind.workspace.edit_contracts import EditBasis

            stage_construction(
                service,
                sid,
                owner,
                job["id"],
                document,
                contract,
                job.get("revision_id") or job.get("based_on"),
                job.get("explanation", "Regenerate feature document"),
                feature_basis=EditBasis.model_validate(job["feature_edit_basis"])
                if job.get("feature_edit_basis") is not None
                else None,
                remaining_seconds=time_left(deadline),
                remaining_actions=None
                if job.get("max_actions") is None
                else max(0, job["max_actions"] - 1),
            )
            revision = build_document(
                service,
                sid,
                owner,
                document,
                contract,
                job.get("revision_id") or job.get("based_on"),
                job.get("explanation", "Regenerate feature document"),
                job.get("max_seconds"),
                cancelled,
                **({"plan_snapshot": job["plan_snapshot"]} if job.get("plan_snapshot") else {}),
            )
            candidate = revision["id"]
            if job.get("feature_edit_basis") is not None:
                service.store.publish_feature_edit(
                    sid, owner, job["id"], job["feature_edit_basis"], candidate
                )
                feature_applied = True
            else:
                service.store.update(sid, selected_revision=candidate)
            if session.get("authoring_mode") == "native_operations":
                from forge_mind.workspace.photo_generation import (
                    compare_candidate,
                    photo_closure,
                    record_failure,
                )

                pending_photo = photo_closure(service, sid, owner, candidate)
                if pending_photo["status"] == "comparison_needed" and (
                    job.get("max_actions") is None or job["max_actions"] >= 2
                ):
                    left = time_left(deadline)
                    if not cancelled() and (left is None or left > 0):
                        try:
                            compare_candidate(
                                service,
                                sid,
                                owner,
                                job["id"],
                                candidate,
                                timeout=30 if left is None else min(30, left),
                                cancelled=cancelled,
                            )
                        except (ValueError, OSError) as exc:
                            if not cancelled() and not expired(deadline):
                                record_failure(
                                    service,
                                    sid,
                                    owner,
                                    job["id"],
                                    "compare",
                                    candidate,
                                    str(exc)[:2000],
                                )
                from forge_mind.workspace.native_delivery import capture_native_delivery

                capture_native_delivery(service, job, deadline, cancelled, "needs_review")
        service.store.finish(job["id"], sid, "needs_review")
    except Exception as exc:
        from forge_mind.workspace.customer_approval import (
            CustomerApprovalRequiredError,
            preserve_budget,
        )

        if isinstance(exc, CustomerApprovalRequiredError):
            record = service.store.session_record(sid, owner)["customer_proposal"]
            preserve_budget(
                service.store,
                sid,
                owner,
                job["id"],
                time_left(deadline),
                record.get("remaining_actions"),
            )
            service.store.finish(job["id"], sid, "needs_approval")
            if job.get("preview_only"):
                # Cancellation can win while pausing. Complete that preview lease
                # without checking or publishing the unbuilt candidate.
                service.store.complete_preview(sid, job["id"], job.get("based_on"), None)
            return
        if job.get("feature_edit_basis") is not None and not feature_applied:
            service.store.update(
                sid,
                last_edit_result={
                    "action": "feature_edit",
                    "job_id": job["id"],
                    "applied": False,
                    "base_revision_id": job["feature_edit_basis"]["revision_id"],
                    "candidate_revision_id": candidate,
                    "code": "rejected",
                    "message": str(exc)[:1000],
                },
            )
        elif feature_applied:
            receipt = service.store.get(sid, owner)["last_edit_result"]
            service.store.update(
                sid, last_edit_result={**receipt, "delivery_error": str(exc)[:1000]}
            )
        service.store.event(sid, "error", f"{type(exc).__name__}: {exc}")
        service.store.finish(job["id"], sid, "cancelled" if cancelled() else "error")
    if job.get("preview_only"):
        finish_preview(service, job, candidate)
