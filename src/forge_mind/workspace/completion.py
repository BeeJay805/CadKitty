"""Deterministic, bounded validation of persisted subsystem author intent.

Checked probes never close frozen acceptance requirements or confirm source claims.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.budget import expired, share_time, time_left
from forge_mind.workspace.capacity import completion_policy, enforce_limit, optional_limit
from forge_mind.workspace.completion_models import (
    FunctionalInterface,
    RequirementDisposition,
    Subsystem,
)
from forge_mind.workspace.inspection_plan import validate_inspection_plan
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.product_sources import read_sources
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def _document(value: Any) -> dict[str, Any]:
    return value.model_dump() if hasattr(value, "model_dump") else (value or {})


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_plan(plan: Any, contract: Any, previous: Any = None) -> None:
    """Validate the graph and retain previously declared work within one contract."""
    document = _document(plan)
    if not isinstance(document, dict):
        raise ValueError("Completion plan must be an object")
    enforce_limit(
        len(json.dumps(document, ensure_ascii=False, allow_nan=False).encode("utf-8")),
        optional_limit("COMPLETION_PLAN_BYTES"),
        "completion plan JSON bytes",
    )
    raw_subsystems = document.get("subsystems", [])
    if not isinstance(raw_subsystems, list):
        raise ValueError("Completion subsystems must be a list")
    enforce_limit(
        len(raw_subsystems), optional_limit("COMPLETION_SUBSYSTEM_LIMIT"), "completion subsystems"
    )
    check_count = 0
    for raw_subsystem in raw_subsystems:
        if not isinstance(raw_subsystem, dict) or not isinstance(
            raw_subsystem.get("checks", []), list
        ):
            raise ValueError("Completion subsystems must be objects with check lists")
        check_count += len(raw_subsystem.get("checks", []))
    enforce_limit(
        check_count,
        optional_limit("COMPLETION_CHECK_LIMIT"),
        "prepared completion checks",
    )
    records = [Subsystem.model_validate(s) for s in raw_subsystems]
    ids = {s.id for s in records}
    if len(ids) != len(records):
        raise ValueError("Subsystem IDs must be unique")
    known = {r["id"] for r in _document(contract).get("requirements", [])}
    checks: set[str] = set()
    for subsystem in records:
        if set(subsystem.requirement_ids) - known:
            raise ValueError("Subsystem references an unknown frozen requirement")
        if set(subsystem.dependencies) - ids or subsystem.id in subsystem.dependencies:
            raise ValueError("Subsystem has an unknown or self dependency")
        for values in (
            subsystem.requirement_ids,
            subsystem.dependencies,
            subsystem.part_ids,
            subsystem.occurrence_paths,
        ):
            if len(values) != len(set(values)) or any(not v or len(v) > 256 for v in values):
                raise ValueError("Subsystem references must be bounded, nonempty and unique")
        if any(not value or len(value) > 2000 for value in subsystem.unresolved):
            raise ValueError("Subsystem unresolved items must contain 1 to 2000 characters")
        for check in subsystem.checks:
            if check.id in checks:
                raise ValueError("Prepared check IDs must be globally unique")
            checks.add(check.id)
    interfaces = [
        FunctionalInterface.model_validate(value)
        for value in document.get("functional_interfaces", [])
    ]
    interface_ids = {interface.id for interface in interfaces}
    if len(interface_ids) != len(interfaces):
        raise ValueError("Functional interface IDs must be unique")
    for interface in interfaces:
        if set(interface.requirement_ids) - known:
            raise ValueError("Functional interface references an unknown frozen requirement")
        for duty in interface.duties:
            if set(duty.check_ids) - checks:
                raise ValueError("Interface duty references an unknown prepared check")
    validate_inspection_plan(document, _document(contract))
    dependency_counts = {s.id: len(s.dependencies) for s in records}
    dispositions = [
        RequirementDisposition.model_validate(d)
        for d in document.get("requirement_dispositions", [])
    ]
    if len({d.requirement_id for d in dispositions}) != len(dispositions):
        raise ValueError("Requirement dispositions must be unique")
    for disposition in dispositions:
        if disposition.requirement_id not in known or set(disposition.check_ids) - checks:
            raise ValueError("Disposition references an unknown requirement or prepared check")
        if len(disposition.check_ids) != len(set(disposition.check_ids)):
            raise ValueError("Disposition check references must be unique")
        for value in [
            *disposition.missing_inputs,
            *disposition.unsupported_checks,
            *disposition.physical_validation,
        ]:
            if not value.strip() or len(value) > 2000:
                raise ValueError("Disposition obligations must contain 1 to 2000 characters")
    dependents: dict[str, list[str]] = {s.id: [] for s in records}
    for subsystem in records:
        for dependency in subsystem.dependencies:
            dependents[dependency].append(subsystem.id)
    ready = deque(key for key, count in dependency_counts.items() if not count)
    visited = 0
    while ready:
        visited += 1
        for dependent in dependents[ready.popleft()]:
            dependency_counts[dependent] -= 1
            if not dependency_counts[dependent]:
                ready.append(dependent)
    if visited != len(records):
        raise ValueError("Subsystem dependency cycle")
    if previous is not None:
        from forge_mind.workspace.check_amendment import validate_check_amendments

        validate_check_amendments(document, _document(previous))
        current_interfaces = {interface.id: interface for interface in interfaces}
        for value in _document(previous).get("functional_interfaces", []):
            prior_interface = FunctionalInterface.model_validate(value)
            current_interface = current_interfaces.get(prior_interface.id)
            if current_interface is None:
                raise ValueError("Previously declared functional interfaces cannot be removed")
            for key in (
                "requirement_ids",
                "participant_paths",
                "installed_occurrence_paths",
                "port_ids",
                "mate_ids",
            ):
                if set(getattr(prior_interface, key)) - set(getattr(current_interface, key)):
                    raise ValueError("Previously declared interface obligations cannot be removed")
            current_features = {
                (feature.occurrence_path, feature.feature_id)
                for feature in current_interface.required_features
            }
            if any(
                (feature.occurrence_path, feature.feature_id) not in current_features
                for feature in prior_interface.required_features
            ):
                raise ValueError("Previously required interface Features cannot be removed")
            current_duties = {duty.id: duty for duty in current_interface.duties}
            for duty in prior_interface.duties:
                successor = current_duties.get(duty.id)
                if successor is None or (successor.kind, successor.proof_scope) != (
                    duty.kind,
                    duty.proof_scope,
                ):
                    raise ValueError(
                        "Previously declared duty identity and proof scope must remain"
                    )
                for key in (
                    "state_ids",
                    "check_ids",
                    "missing_inputs",
                    "unsupported_checks",
                    "physical_validation",
                ):
                    if set(getattr(duty, key)) - set(getattr(successor, key)):
                        raise ValueError("Previously declared duty obligations cannot be removed")
            states = {state.id: state for state in current_interface.states}
            if any(states.get(state.id) != state for state in prior_interface.states):
                raise ValueError("Retain original interface states; append new states explicitly")
        old_dispositions = _document(previous).get("requirement_dispositions", [])
        if {d["requirement_id"] for d in old_dispositions} - {
            d.requirement_id for d in dispositions
        }:
            raise ValueError("Previously declared requirement dispositions cannot be removed")
        old = _document(previous).get("subsystems", [])
        if {s["id"] for s in old} - ids:
            raise ValueError("Previously declared subsystems cannot be removed within a contract")
        by_id = {s.id: s for s in records}
        for prior in old:
            current = by_id[prior["id"]]
            if {c["id"] for c in prior.get("checks", [])} - {c.id for c in current.checks}:
                raise ValueError("Previously declared checks cannot be removed within a contract")
            if set(prior.get("requirement_ids", [])) - set(current.requirement_ids):
                raise ValueError("Previously assigned frozen requirements cannot be removed")


def _snapshot(service: WorkspaceService, sid: str, owner: str, rid: str | None) -> dict[str, Any]:
    session = service.store.get(sid, owner)
    revision = service._load_revision(sid, rid) if rid else {}
    effective_contract = session.get("contract")
    for name in ("proposal", "resume_candidate"):
        candidate = session.get(name) or {}
        if rid and candidate.get("revision_id") == rid:
            effective_contract = candidate.get("contract", effective_contract)
    return {
        "revision_id": rid,
        "contract_hash": _hash(revision.get("contract", session.get("contract"))),
        "session_contract_hash": _hash(effective_contract),
        "engine_fingerprint": engine_fingerprint(),
        "source_fingerprint": read_sources(service.store, sid, owner)["fingerprint"],
        "input_version": session.get("engineering", {}).get("version", 0),
        "plan_hash": _hash(session.get("design_plan")),
        "artifact_hashes": revision.get("measurement", {}).get("artifact_hashes", {}),
    }


def _structure(plan: dict[str, Any], revision: dict[str, Any]) -> list[dict[str, Any]]:
    document = revision.get("document") or {}
    assembly = document.get("assembly") or {}
    parts = {p["id"] for p in assembly.get("parts", [])}
    if document and not assembly:
        parts.add("part")
    facts = revision.get("measurement", {}).get("facts", {}).get("assembly") or {}
    paths = {o["path"] for o in facts.get("occurrences", [])}
    if assembly:
        parts &= {o["definition"] for o in facts.get("occurrences", [])}
    rows = []
    for subsystem in plan.get("subsystems", []):
        missing_parts = sorted(set(subsystem.get("part_ids", [])) - parts)
        missing_paths = sorted(set(subsystem.get("occurrence_paths", [])) - paths)
        rows.append(
            {
                "id": subsystem["id"],
                "name": subsystem["name"],
                "requirement_ids": subsystem.get("requirement_ids", []),
                "dependencies": subsystem.get("dependencies", []),
                "missing_part_ids": missing_parts,
                "missing_occurrence_paths": missing_paths,
                "unresolved": subsystem.get("unresolved", []),
                "geometry_declared": bool(
                    subsystem.get("part_ids") or subsystem.get("occurrence_paths")
                ),
                "status": "missing" if not revision or missing_parts or missing_paths else "ready",
                "checks": [
                    {"id": check["id"], "status": "unverified", "reason": "Not run"}
                    for check in subsystem.get("checks", [])
                ],
            }
        )
    return rows


def _motion_applicable(revision: dict[str, Any]) -> bool:
    return (revision.get("document") or {}).get("motion_rig") is not None or (
        revision.get("measurement", {}).get("facts") or {}
    ).get("motion_rig") is not None


def _motion_receipt(
    service: WorkspaceService, sid: str, owner: str, snapshot: dict[str, Any], check_id: str
) -> dict[str, Any]:
    from forge_mind.workspace.completion_readonly import motion_summary
    from forge_mind.workspace.motion_service import check_path

    result = json.loads(check_path(service, sid, owner, check_id).read_text(encoding="utf-8"))
    revision = service._load_revision(sid, snapshot["revision_id"])
    return motion_summary(result, revision, snapshot, check_id)


def _verify_receipts(
    service: WorkspaceService, sid: str, owner: str, report: dict[str, Any]
) -> bool:
    """Read exact receipts, including integrity, rather than trusting a cached green row."""
    try:
        plan = service.store.get(sid, owner).get("design_plan") or {}
        revision = service._load_revision(sid, report["revision_id"])
        expected_rows = _structure(plan, revision)
        actual_rows = report.get("subsystems", [])
        if [r["id"] for r in expected_rows] != [r["id"] for r in actual_rows]:
            return False
        definitions = {c["id"]: c for s in plan.get("subsystems", []) for c in s.get("checks", [])}
        for expected, actual in zip(expected_rows, actual_rows, strict=True):
            if any(
                actual.get(k) != v for k, v in expected.items() if k not in {"status", "checks"}
            ):
                return False
            if [c["id"] for c in expected["checks"]] != [c["id"] for c in actual["checks"]]:
                return False
            statuses = [c["status"] for c in actual["checks"]]
            if actual["status"] == "checked_pass" and (
                not statuses
                or any(s != "pass" for s in statuses)
                or actual["unresolved"]
                or not actual["geometry_declared"]
                or expected["status"] == "missing"
            ):
                return False
            if actual["status"] == "checked_fail" and "fail" not in statuses:
                return False
        with service.store.connect() as con:
            for subsystem in actual_rows:
                for check in subsystem.get("checks", []):
                    if check.get("status") not in {"pass", "fail"}:
                        continue
                    row = con.execute(
                        "SELECT document,digest FROM product_checks WHERE session=? AND id=?",
                        (sid, check.get("receipt_id")),
                    ).fetchone()
                    if row is None or hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
                        return False
                    receipt = json.loads(row[0])
                    if (
                        receipt.get("id") != check.get("receipt_id")
                        or receipt.get("revision_id") != report.get("revision_id")
                        or receipt.get("source_fingerprint") != report.get("source_fingerprint")
                        or receipt.get("engine_fingerprint") != report.get("engine_fingerprint")
                        or receipt.get("artifact_hashes") != report.get("artifact_hashes")
                        or _hash(receipt.get("contract")) != report.get("contract_hash")
                        or _hash(receipt.get("request")) != check.get("request_hash")
                        or _hash(definitions[check["id"]]["request"]) != check.get("request_hash")
                        or receipt.get("result", {}).get("status") != check["status"]
                    ):
                        return False
        motion = report.get("motion") or {}
        if report.get("status") == "checked_pass" and (
            not actual_rows
            or any(r["status"] != "checked_pass" for r in actual_rows)
            or {r["id"] for r in (revision.get("contract") or {}).get("requirements", [])}
            - {rid for row in actual_rows for rid in row["requirement_ids"]}
            or motion.get("status") not in {"checked", "not_applicable"}
        ):
            return False
        return not (
            motion.get("status") in {"checked", "checked_fail"}
            and _motion_receipt(service, sid, owner, report, motion["check_id"]) != motion
        )
    except Exception:
        return False


def _current_attribution(session: dict[str, Any], contract: dict[str, Any]) -> Any:
    """Only an existing current server-admitted record supplies nominal context."""
    from forge_mind.workspace.intent_attribution import IntentAttribution
    from forge_mind.workspace.models import Contract
    from forge_mind.workspace.requirement_verification import _intent_authority

    try:
        _intent_authority(session, Contract.model_validate(contract))
        record = session.get("intent_attribution") or {}
        return IntentAttribution.model_validate(record["attribution"]) if record else None
    except (ValueError, TypeError, KeyError, WorkspaceError):
        return None


def _pending_final_requirements(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str | None,
    plan: dict[str, Any],
    intent_contract: dict[str, Any],
    mapping: dict[str, Any],
) -> set[str]:
    """Route supported frozen duties to final evaluation; assign no new pass."""
    typed = mapping.get("typed_requirements") or []
    if not rid or (
        not typed
        and not any(
            row.get("metric", "unmeasured") != "unmeasured"
            for row in intent_contract.get("requirements", [])
        )
    ):
        return set()
    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.document import digest as document_digest
    from forge_mind.workspace.input_application import source as input_source
    from forge_mind.workspace.intent_attribution_service import request_text
    from forge_mind.workspace.measure import _finding
    from forge_mind.workspace.models import Contract, DesignPlan, Measurement
    from forge_mind.workspace.requirement_verification import (
        _frozen_inputs,
        _intent_authority,
        verification_method,
    )
    from forge_mind.workspace.verification_admission import validate_verification_bindings

    try:
        session, revision, frozen = _frozen_inputs(service.store, sid, owner)
        contract = Contract.model_validate(session["contract"])
        measured = Measurement.model_validate(revision["measurement"])
        captured = (revision.get("operation_record") or {}).get("plan_snapshot") or {}
        application = session.get("contract_application")
        if application is not None:
            applied = application.get("applied") or {}
            if (
                application.get("pending") is not None
                or applied.get("contract_hash") != contract.fingerprint()
                or any(applied.get(key) != value for key, value in input_source(session).items())
            ):
                return set()
        if (
            session.get("selected_revision") != rid
            or measured.revision_id != rid
            or session.get("design_plan") != plan
            or contract.fingerprint() != Contract.model_validate(intent_contract).fingerprint()
            or frozen["contract_hash"] != frozen["revision_contract_hash"]
            or frozen["engine_fingerprint"] != frozen["measured_engine_fingerprint"]
            or measured.engine_fingerprint != frozen["engine_fingerprint"]
            or measured.contract_hash != contract.fingerprint()
            or frozen.get("request_hash") != document_digest(request_text(session))
            or captured.get("plan") != plan
            or captured.get("plan_hash") != session.get("plan_hash")
            or captured.get("plan_version") != session.get("plan_version")
            or captured.get("contract_hash") != contract.fingerprint()
        ):
            return set()
        _intent_authority(session, contract)
        validate_verification_bindings(
            DesignPlan.model_validate(plan),
            contract,
            DesignDocument.model_validate(revision["document"]),
        )
        declared = {
            row["requirement_id"]
            for row in typed
            if row.get("mapping_status")
            in {
                "subject_binding_declared",
                "subject_bindings_declared",
                "delivery_obligation",
                "policy_obligation",
            }
        }
        findings = {finding.requirement_id: finding for finding in measured.findings}
        pending: set[str] = set()
        for requirement in contract.requirements:
            if requirement.verification is not None:
                if requirement.id in declared:
                    verification_method(requirement.verification)
                    pending.add(requirement.id)
            elif requirement.metric != "unmeasured":
                finding = findings.get(requirement.id)
                if (
                    finding
                    and finding.status == "pass"
                    and finding.expected == requirement.value
                    and _finding(
                        requirement, finding.measured, finding.method, finding.reason
                    ).status
                    == "pass"
                ):
                    pending.add(requirement.id)
        return pending
    except (ValueError, TypeError, KeyError, OSError, WorkspaceError, sqlite3.Error):
        return set()


def completion_context(
    service: WorkspaceService, sid: str, owner: str, rid: str | None
) -> dict[str, Any]:
    """Return derived work status, marking persisted results stale on any context change."""
    session = service.store.get(sid, owner)
    plan = session.get("design_plan") or {}
    prior = session.get("completion_report") or {}
    try:
        snapshot = _snapshot(service, sid, owner, rid)
        revision = service._load_revision(sid, rid) if rid else {}
        intent_contract = revision.get("contract") or session.get("contract") or {}
        if session.get("authoring_mode") == "native_operations":
            active_binding = session.get("operation_plan_snapshot") or {}
            intent_contract = (
                active_binding.get("contract")
                if active_binding.get("plan") == plan
                else session.get("contract") or {}
            )
        validate_plan(plan, intent_contract)
    except (ValueError, OSError, WorkspaceError) as exc:
        return {
            "status": "unverified",
            "stale": True,
            "reason": str(exc)[:500],
            "subsystems": [],
            "next_target": None,
        }
    stale = bool(prior) and any(prior.get(key) != value for key, value in snapshot.items())
    if prior and _motion_applicable(revision) and not prior.get("motion"):
        stale = True
    if prior and not stale and not _verify_receipts(service, sid, owner, prior):
        stale = True
    rows = _structure(plan, revision)
    if prior and not stale:
        rows = prior.get("subsystems", rows)
    else:
        from forge_mind.workspace.completion_inline import _read_cached
        from forge_mind.workspace.completion_readonly import prepared_binding as _binding

        early = session.get("author_checkpoint") or {}
        if early and not early.get("stale") and all(early.get(k) == v for k, v in snapshot.items()):
            definitions = {
                c["id"]: c for s in plan.get("subsystems", []) for c in s.get("checks", [])
            }
            cache = session.get("prepared_check_cache") or {}
            # Checkpoints are mutable guidance, not evidence. Reconstruct the
            # declared coverage and verdicts from the current plan and receipts.
            for row in rows:
                for check in row["checks"]:
                    receipt = _read_cached(
                        service, sid, snapshot, definitions[check["id"]], cache.get(check["id"], {})
                    )
                    if receipt is not None:
                        check.update(
                            status=receipt["result"]["status"],
                            receipt_id=receipt["id"],
                            reason="Current exploratory receipt",
                            result=receipt["result"],
                        )
                    else:
                        attempt = early.get("attempts", {}).get(check["id"], {})
                        if attempt.get("binding") == _binding(snapshot, definitions[check["id"]]):
                            # Display-only diagnosis never supplies a measured verdict.
                            check["reason"] = str(attempt.get("reason", "Probe unavailable"))[:500]
                statuses = [c["status"] for c in row["checks"]]
                if row["status"] != "missing":
                    row["status"] = (
                        "checked_fail"
                        if "fail" in statuses
                        else "checked_pass"
                        if statuses
                        and all(s == "pass" for s in statuses)
                        and row["geometry_declared"]
                        and not row["unresolved"]
                        else "unverified"
                    )
    known = {r["id"] for r in intent_contract.get("requirements", [])}
    covered = {r for s in rows for r in s["requirement_ids"]}
    uncovered = sorted(known - covered)
    by_id = {s["id"]: s for s in rows}
    # Repair known failures before unrelated detail. Unavailable physical/source
    # obligations remain visible without blocking independent construction.
    actionable = sorted(
        rows,
        key=lambda s: (s["status"] != "checked_fail", s["status"] != "missing", bool(s["checks"])),
    )
    next_target = next(
        (
            s["id"]
            for s in actionable
            if (
                s["status"] in {"missing", "checked_fail"}
                or not s["checks"]
                or not s["geometry_declared"]
                or any(c.get("reason") == "Not run" for c in s["checks"])
            )
            and all(by_id[d]["status"] != "missing" for d in s["dependencies"])
        ),
        None,
    )
    work = {}
    if session.get("authoring_mode") == "native_operations":
        from forge_mind.workspace.completion_coverage import functional_coverage
        from forge_mind.workspace.completion_work import remaining_work
        from forge_mind.workspace.intent_attribution_service import request_text
        from forge_mind.workspace.operation_plan import coverage
        from forge_mind.workspace.operation_plan import snapshot as plan_snapshot

        operation_error = None
        functional_mapping = functional_coverage(
            plan,
            intent_contract,
            source_text=request_text(session) if isinstance(session.get("prompt"), str) else None,
            attribution=_current_attribution(session, intent_contract),
        )
        pending_final = _pending_final_requirements(
            service, sid, owner, rid, plan, intent_contract, functional_mapping
        )
        operation_work: dict[str, Any] = {"steps": []}
        if plan:
            try:
                binding = session.get("operation_plan_snapshot")
                if not binding or binding.get("plan") != plan:
                    binding = plan_snapshot(
                        plan, session.get("plan_version", 0), session.get("contract") or {}
                    )
                operation_work = coverage(binding, revision.get("document") or {})
            except (ValueError, KeyError, TypeError) as exc:
                operation_error = str(exc)[:1000]

        functional = _interface_projection(
            service,
            sid,
            owner,
            plan,
            revision,
            rows,
            snapshot,
            prior.get("motion") if prior and not stale else None,
        )
        work = remaining_work(
            rows,
            requirement_dispositions(plan, intent_contract, rows),
            operation_work["steps"],
            functional,
            pending_final_requirement_ids=pending_final,
            executable_requirement_ids={
                requirement["id"]
                for requirement in intent_contract.get("requirements", [])
                if requirement.get("metric", "unmeasured") != "unmeasured"
                or requirement.get("verification") is not None
            },
        )
        if pending_final:
            work["pending_final_requirement_ids"] = sorted(pending_final)
            work["pending_final_verification_status"] = "unverified"
        if plan.get("functional_interfaces"):
            work["functional_interfaces"] = functional
        next_target = work["next_target"]
        if operation_error:
            work.update(
                operation_plan_error=operation_error,
                next_work_category="review",
                next_work_reason="Repair operation plan coverage",
            )
            next_target = "operation_plan"
    elif plan.get("functional_interfaces"):
        from forge_mind.workspace.completion_work import remaining_work

        functional = _interface_projection(
            service,
            sid,
            owner,
            plan,
            revision,
            rows,
            snapshot,
            prior.get("motion") if prior and not stale else None,
        )
        work = remaining_work(
            rows,
            requirement_dispositions(plan, intent_contract, rows),
            interface_work=functional,
        )
        work["functional_interfaces"] = functional
        next_target = work["next_target"]
    context_status = prior.get("status", "unverified") if prior and not stale else "unverified"
    if session.get("authoring_mode") == "native_operations":
        work["functional_coverage"] = functional_mapping
        if functional_mapping["status"] != "not_applicable" and context_status == "checked_pass":
            context_status = "unverified"
        if functional_mapping.get("next_target") and next_target is None:
            next_target = functional_mapping["next_target"]
            work.update(
                next_work_category="review", next_work_reason=functional_mapping["next_work_reason"]
            )
    if context_status == "checked_pass" and (
        next_target is not None
        or (work.get("functional_interfaces") or {}).get("status", "not_declared")
        not in {"not_declared", "nominal_checked"}
    ):
        context_status = "unverified"
    return {
        **snapshot,
        **{key: value for key, value in work.items() if key != "scope"},
        "subsystems": rows,
        "uncovered_requirement_ids": uncovered,
        "next_target": next_target,
        "resource_policy": completion_policy(),
        "requirement_dispositions": requirement_dispositions(plan, intent_contract, rows),
        "status": context_status,
        "stale": stale,
        "motion": prior.get("motion")
        if prior and not stale
        else {
            "status": "unverified" if _motion_applicable(revision) else "not_applicable",
            "reason": "No current completion motion evidence"
            if _motion_applicable(revision)
            else "Revision has no motion rig",
        },
        "scope": "Subsystem construction and exploratory checks; frozen acceptance is unchanged.",
    }


def _interface_projection(
    service: WorkspaceService,
    sid: str,
    owner: str,
    plan: dict[str, Any],
    revision: dict[str, Any],
    rows: list[dict[str, Any]],
    snapshot: dict[str, Any],
    motion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct exact evidence before passing rows into the pure projector."""
    from forge_mind.workspace.completion_obligations import interface_obligations
    from forge_mind.workspace.completion_readonly import project_interfaces

    if not plan.get("functional_interfaces"):
        return interface_obligations(plan, revision, [])
    session = service.store.get(sid, owner)
    with service.store.connect() as con:
        projected, _ = project_interfaces(
            con,
            sid,
            session,
            plan,
            revision,
            snapshot,
            rows=rows,
            motion=motion,
            root=getattr(service.store, "root", None),
        )
    return projected


def requirement_dispositions(
    plan: dict[str, Any], contract: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    declared = {d["requirement_id"]: d for d in plan.get("requirement_dispositions", [])}
    checks = {c["id"]: c for row in rows for c in row.get("checks", [])}
    return [
        {
            "requirement_id": requirement["id"],
            "status": "unverified",
            "declared": requirement["id"] in declared,
            "checks": [
                {
                    "id": cid,
                    "status": checks.get(cid, {}).get("status", "unverified"),
                    "receipt_id": checks.get(cid, {}).get("receipt_id"),
                }
                for cid in declared.get(requirement["id"], {}).get("check_ids", [])
            ],
            **{
                key: declared.get(requirement["id"], {}).get(key, [])
                for key in ("missing_inputs", "unsupported_checks", "physical_validation")
            },
            "scope": "Author-proposed coverage. Scoped checks do not establish equivalence to or satisfaction of this frozen requirement.",
        }
        for requirement in contract.get("requirements", [])
    ]


def _numeric_selector(value: Any) -> bool:
    if isinstance(value, dict):
        return ("body_index" in value and value["body_index"] is not None) or any(
            _numeric_selector(v) for v in value.values()
        )
    return isinstance(value, list) and any(_numeric_selector(v) for v in value)


def finalize_completion(
    service: WorkspaceService,
    job: dict[str, Any],
    deadline: float | None,
    cancelled: Callable[[], bool],
    author_state: str,
) -> dict[str, Any]:
    """Run saved strict checks, respecting cancellation and any explicit deadline."""
    from forge_mind.workspace.completion_inline import execute_prepared
    from forge_mind.workspace.engineering_service import run_report_job
    from forge_mind.workspace.motion_check import MotionCheckSettings
    from forge_mind.workspace.motion_service import run_motion_job
    from forge_mind.workspace.product_interfaces import validate_interfaces

    sid, owner = job["session"], job["owner"]
    session = service.store.get(sid, owner)
    rid = session.get("selected_revision")
    revision: dict[str, Any] = {}
    report: dict[str, Any] = {
        "phase": "validating",
        "author_state": author_state,
        "revision_id": rid,
        "status": "unverified",
        "subsystems": [],
        "receipt_ids": [],
        "interfaces": {"status": "unverified", "reason": "Not run"},
        "motion": {"status": "unverified", "reason": "Not run"},
        "engineering_report": {"status": "unverified", "reason": "Not run"},
        "physical_readiness": "unverified",
        "scope": "Exploratory checks do not establish frozen acceptance or physical readiness.",
    }

    def persist() -> None:
        if "contract_hash" in report:
            if session.get("authoring_mode") == "native_operations":
                from forge_mind.workspace.completion_coverage import functional_coverage
                from forge_mind.workspace.intent_attribution_service import request_text

                report["functional_coverage"] = functional_coverage(
                    session.get("design_plan") or {},
                    revision.get("contract") or session.get("contract") or {},
                    source_text=request_text(session)
                    if isinstance(session.get("prompt"), str)
                    else None,
                    attribution=_current_attribution(
                        session, revision.get("contract") or session.get("contract") or {}
                    ),
                )
            report["requirement_dispositions"] = requirement_dispositions(
                session.get("design_plan") or {},
                revision.get("contract") or session.get("contract") or {},
                report["subsystems"],
            )
            if (session.get("design_plan") or {}).get("functional_interfaces"):
                basis = {
                    key: report[key]
                    for key in (
                        "revision_id",
                        "contract_hash",
                        "session_contract_hash",
                        "engine_fingerprint",
                        "source_fingerprint",
                        "input_version",
                        "plan_hash",
                        "artifact_hashes",
                    )
                }
                report["functional_interfaces"] = _interface_projection(
                    service,
                    sid,
                    owner,
                    session["design_plan"],
                    revision,
                    report["subsystems"],
                    basis,
                    report.get("motion"),
                )
        progress = service.store.get(sid, owner).get("completion_progress") or {}
        service.store.update(
            sid,
            completion_report=report,
            completion_progress={
                **progress,
                "phase": "validation" if report["phase"] == "validating" else "finished",
                "message": "Checking saved subsystem geometry and prepared probes",
                "remaining_seconds": time_left(deadline),
            },
        )

    def unavailable() -> str | None:
        if cancelled():
            return "Cancelled"
        if expired(deadline):
            return "Validation time budget exhausted"
        return None

    def worker_cancelled() -> bool:
        return cancelled() or expired(deadline)

    try:
        snapshot = _snapshot(service, sid, owner, rid)
        report.update(snapshot)
        plan = session.get("design_plan") or {}
        revision = service._load_revision(sid, rid) if rid else {}
        motion_applicable = _motion_applicable(revision)
        if not motion_applicable:
            report["motion"] = {"status": "not_applicable", "reason": "Revision has no motion rig"}
        validate_plan(plan, revision.get("contract") or session.get("contract"))
        rows = _structure(plan, revision)
        report["subsystems"] = rows
        if not rows:
            report["reason"] = "No typed subsystem plan is saved"
        report["uncovered_requirement_ids"] = sorted(
            {
                r["id"]
                for r in (revision.get("contract") or session.get("contract") or {}).get(
                    "requirements", []
                )
            }
            - {r for s in rows for r in s["requirement_ids"]}
        )
        persist()
        if not rid or not revision.get("measurement", {}).get("geometry_valid"):
            report["reason"] = "No valid saved geometry is available"
            for row in rows:
                for check in row["checks"]:
                    check["reason"] = report["reason"]
            report["interfaces"]["reason"] = report["reason"]
            if motion_applicable:
                report["motion"]["reason"] = report["reason"]
            report["engineering_report"]["reason"] = report["reason"]
            return report

        def unchanged() -> bool:
            try:
                return _snapshot(service, sid, owner, rid) == snapshot
            except (ValueError, OSError, WorkspaceError):
                return False

        by_id = {r["id"]: r for r in rows}
        prepared = {
            check["id"]: check
            for subsystem in plan.get("subsystems", [])
            for check in subsystem.get("checks", [])
        }
        # These are construction dependencies, not proof dependencies. Every
        # request is self-contained; unresolved torque or a failed upstream
        # probe must not prevent measuring available downstream geometry.
        pending = {r["id"] for r in rows}
        while pending:
            progressed = False
            for row in rows:
                if row["id"] not in pending:
                    continue
                if row["status"] == "missing":
                    for check in row["checks"]:
                        check["reason"] = "Expected subsystem geometry is missing"
                    pending.remove(row["id"])
                    progressed = True
                    continue
                dependencies = [by_id[d] for d in row["dependencies"]]
                if any(d["id"] in pending for d in dependencies):
                    continue
                if any(
                    d["status"] in {"missing", "blocked"} or not d["geometry_declared"]
                    for d in dependencies
                ):
                    row["status"] = "blocked"
                    for check in row["checks"]:
                        check["reason"] = "Required dependency geometry is missing or undeclared"
                    pending.remove(row["id"])
                    progressed = True
                    continue
                check = next((c for c in row["checks"] if c["reason"] == "Not run"), None)
                if check is None:
                    statuses = [c["status"] for c in row["checks"]]
                    row["status"] = (
                        "checked_fail"
                        if "fail" in statuses
                        else "checked_pass"
                        if statuses
                        and all(s == "pass" for s in statuses)
                        and not row["unresolved"]
                        and row["geometry_declared"]
                        else "unverified"
                    )
                    pending.remove(row["id"])
                    progressed = True
                    continue
                definition = prepared[check["id"]]
                reason = unavailable()
                if (
                    not reason
                    and _numeric_selector(definition["request"])
                    and definition.get("reference_revision_id") != rid
                ):
                    reason = "Numeric body selectors require the exact final reference revision"
                if not reason and not unchanged():
                    reason = "Validation inputs changed"
                if reason:
                    check["reason"] = reason
                else:
                    remaining_checks = sum(
                        c["reason"] == "Not run" for r in rows for c in r["checks"]
                    )
                    share = share_time(deadline, remaining_checks + (3 if motion_applicable else 2))
                    try:
                        receipt, reused = execute_prepared(
                            service, sid, owner, snapshot, definition, share, worker_cancelled
                        )
                        check.update(
                            status=receipt.get("result", {}).get("status", "unverified"),
                            reason="Exploratory probe completed",
                            receipt_id=receipt["id"],
                            request_hash=_hash(definition["request"]),
                            reused=reused,
                        )
                        report["receipt_ids"].append(receipt["id"])
                        if not unchanged() or worker_cancelled():
                            check.update(
                                status="unverified", reason="Validation inputs changed or cancelled"
                            )
                    except Exception as exc:
                        check.update(
                            status="unverified", reason=f"{type(exc).__name__}: {str(exc)[:300]}"
                        )
                progressed = True
                persist()
            if not progressed:
                raise WorkspaceError("Completion dependency scheduler made no progress")

        phases = ["interfaces", *(["motion"] if motion_applicable else []), "engineering_report"]
        for phase_index, phase in enumerate(phases):
            reason = unavailable()
            if not reason and not unchanged():
                reason = "Validation inputs changed"
            if reason:
                report[phase]["reason"] = reason
                continue
            share = share_time(deadline, len(phases) - phase_index)
            try:
                if phase == "interfaces":
                    receipts = validate_interfaces(
                        service, sid, owner, rid, share, worker_cancelled
                    )
                    report[phase] = {
                        "status": "checked" if receipts else "unverified",
                        "reason": "Confirmed interfaces checked"
                        if receipts
                        else "No confirmed interfaces",
                        "receipt_ids": [r["id"] for r in receipts if r.get("id")],
                    }
                elif phase == "motion":
                    measured_motion = (
                        revision.get("measurement", {}).get("facts", {}).get("motion_rig")
                    )
                    if not isinstance(measured_motion, dict) or not measured_motion:
                        raise WorkspaceError(
                            "Declared motion rig has no measured exported bindings; rebuild this revision"
                        )
                    check_id = run_motion_job(
                        service,
                        {
                            "session": sid,
                            "owner": owner,
                            "revision_id": rid,
                            "settings": MotionCheckSettings().model_dump(),
                            "max_seconds": share,
                        },
                        worker_cancelled,
                    )
                    report[phase] = _motion_receipt(service, sid, owner, snapshot, check_id)
                    if not unchanged() or worker_cancelled():
                        report[phase].update(
                            status="unverified", reason="Validation inputs changed or cancelled"
                        )
                else:
                    run_report_job(
                        service,
                        {"session": sid, "owner": owner, "revision_id": rid, "max_seconds": share},
                        worker_cancelled,
                    )
                    reports = service.store.get(sid, owner).get("reports", [])
                    report[phase] = {
                        "status": "checked",
                        "reason": "Engineering report generated; inspect its findings",
                        "receipt_id": reports[-1]["id"] if reports else None,
                    }
            except Exception as exc:
                report[phase] = {
                    "status": "unverified",
                    "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            persist()
        fresh = unchanged()
        report["stale"] = not fresh
        if not fresh or worker_cancelled():
            if motion_applicable:
                report["motion"].update(
                    status="unverified", reason="Validation inputs changed or cancelled"
                )
            for row in rows:
                if row["status"] == "checked_pass":
                    row["status"] = "unverified"
                for check in row["checks"]:
                    if check["status"] == "pass":
                        check.update(
                            status="unverified", reason="Validation inputs changed or cancelled"
                        )
        functional = _interface_projection(
            service, sid, owner, plan, revision, rows, snapshot, report["motion"]
        )
        if plan.get("functional_interfaces"):
            report["functional_interfaces"] = functional
        report["status"] = (
            "checked_fail"
            if any(r["status"] == "checked_fail" for r in rows)
            or report["motion"]["status"] == "checked_fail"
            or any(row["status"] == "check_failed" for row in functional["rows"])
            else "checked_pass"
            if rows
            and all(r["status"] == "checked_pass" for r in rows)
            and not report["uncovered_requirement_ids"]
            and fresh
            and not worker_cancelled()
            and report["motion"]["status"] in {"checked", "not_applicable"}
            and functional["status"] in {"not_declared", "nominal_checked"}
            and (report.get("functional_coverage") or {}).get("status", "not_applicable")
            == "not_applicable"
            else "unverified"
        )
        return report
    except Exception as exc:
        report.update(status="unverified", reason=f"{type(exc).__name__}: {str(exc)[:500]}")
        return report
    finally:
        report["phase"] = "cancelled" if cancelled() else "complete"
        persist()
