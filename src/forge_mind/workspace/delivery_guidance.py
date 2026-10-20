"""Revision-bound customer declarations, without geometry or physical approval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, StrictStr, field_validator, model_validator

from forge_mind.workspace.completion import _hash, _snapshot
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.guidance_history import (
    guidance_history_status as guidance_history_status,
)
from forge_mind.workspace.guidance_history import (
    guidance_record_provenance,
    guidance_record_revision,
)
from forge_mind.workspace.handoff import referenced_import_asset_ids
from forge_mind.workspace.instruction_plan import (
    InstructionPlan,
    instruction_catalog,
    resolve_instruction_plan,
    validate_instruction_plan,
)
from forge_mind.workspace.measure import engine_fingerprint, verify_artifacts
from forge_mind.workspace.models import Contract, Measurement, Record
from forge_mind.workspace.product_models import DeliveryGuidance
from forge_mind.workspace.product_sources import source_fingerprint
from forge_mind.workspace.production import package_file_inventory
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    import sqlite3

    from forge_mind.workspace.agent import WorkspaceService

SCOPE = (
    "Current means these author declarations were reviewed against the captured saved inputs. "
    "Freshness does not establish textual correctness, printability, fit, strength or physical approval."
)
VARIABLE_SCOPE = (
    "Declared CAD inputs and evaluated relations, not independently measured dimensions. "
    "Displayed variables and parts are bounded summaries; design complexity is not limited."
)


class ReviewedDeliveryGuidance(DeliveryGuidance):
    print: StrictStr = Field(min_length=1, max_length=4000)
    installation: StrictStr = Field(min_length=1, max_length=4000)
    use_and_removal: StrictStr = Field(min_length=1, max_length=4000)
    unresolved: StrictStr = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def supplied_fields(self) -> ReviewedDeliveryGuidance:
        if any(not value.strip() for value in self.model_dump().values()):
            raise ValueError("Supply nonempty print, installation, use and unresolved guidance")
        return self


class GuidanceUpdate(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    basis_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewed_current_facts: Literal[True]
    guidance: ReviewedDeliveryGuidance
    instruction_plan: InstructionPlan | None = None

    @field_validator("reviewed_current_facts", mode="before")
    @classmethod
    def explicit_review(cls, value: Any) -> bool:
        if value is not True:
            raise ValueError("Explicitly review the current saved facts before updating guidance")
        return True


def _history_entry(revision_id: str, record: Any, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": len(entries) + 1,
        "predecessor_hash": digest(entries[-1]) if entries else None,
        "revision_id": revision_id,
        "record": deepcopy(record),
        "record_hash": digest(record),
    }


def _history_for_write(
    session: dict[str, Any], records: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    stored = session.get("delivery_guidance_history", {})
    if not isinstance(stored, dict):
        raise WorkspaceError("Guidance history is invalid; preserve it for review")
    for revision_id, entries in stored.items():
        if (
            guidance_history_status(entries, revision_id, records.get(revision_id))["status"]
            != "verified"
        ):
            raise WorkspaceError("Guidance history integrity changed; no authored update was saved")
    history = deepcopy(stored)
    for revision_id, record in records.items():
        if revision_id not in history:
            history[revision_id] = [_history_entry(revision_id, record, [])]
    return history


def _input_authority(session: dict[str, Any], target: str) -> dict[str, Any]:
    """Describe current owner-input identity without refusing an authorized active job."""
    from forge_mind.workspace.input_application import FIELD, contract_hash, source
    from forge_mind.workspace.intent_attribution_service import _record

    reasons = []
    sources: dict[str, Any] = {}
    applications: dict[str, Any] = {}
    snapshots: dict[str, Any] = {}
    pending_identities: set[str] = set()
    target_location = "current"
    for location, value in (
        ("current", session),
        ("proposal", session.get("proposal")),
        ("checkpoint", session.get("resume_candidate")),
    ):
        if value is None:
            sources[location] = applications[location] = None
            continue
        if not isinstance(value, dict):
            sources[location] = applications[location] = None
            reasons.append(f"{location} owner-input snapshot is invalid")
            continue
        if location != "current" and value.get("revision_id") == target:
            target_location = location
        snapshot = {
            "id": session.get("id"),
            "prompt": session["prompt"],
            "instructions": value.get("instructions", session.get("instructions", [])),
            "contract": value.get("contract", session.get("contract")),
        }
        application = value.get(FIELD)
        snapshots[location] = {**snapshot, FIELD: application}
        applications[location] = None
        try:
            applications[location] = digest(application) if application is not None else None
            if not isinstance(snapshot["instructions"], list) or any(
                not isinstance(item, str) for item in snapshot["instructions"]
            ):
                raise ValueError("instruction history is invalid")
            sources[location] = source(snapshot)
            if application is None:
                continue  # Historical snapshots did not track consumption explicitly.
            if not isinstance(application, dict):
                raise ValueError("input application is invalid")
            if application.get("pending") is not None:
                reasons.append(f"{location} owner input is pending")
                pending_identities.add(digest(application["pending"]))
            applied = application.get("applied")
            if (
                not isinstance(applied, dict)
                or any(applied.get(key) != expected for key, expected in sources[location].items())
                or applied.get("contract_hash") != contract_hash(snapshot["contract"])
            ):
                reasons.append(
                    f"{location} owner input is not applied to its exact source and Contract"
                )
        except (KeyError, TypeError, ValueError) as exc:
            sources[location] = None
            reasons.append(f"{location} owner-input identity is invalid: {exc}")
    intent_hash = None
    try:
        intent_hash = digest(session.get("intent_attribution"))
        intent = _record(session)
        if intent is not None:
            from forge_mind.workspace.input_lineage import FIELD as LINEAGE
            from forge_mind.workspace.input_lineage import validated

            if intent.get("status") == "needs_input":
                reasons.append("Owner clarification remains unresolved")
            elif intent.get("status") not in {"interpreted", "answered"}:
                reasons.append("Owner interpretation status is invalid")
            scoped = snapshots.get(target_location)
            target_application = scoped.get(FIELD) if scoped is not None else None
            historical = (
                validated(scoped, target_application, intent)
                if scoped is not None
                and isinstance(target_application, dict)
                and LINEAGE in target_application
                else None
            )
            target_source = sources.get(target_location)
            if historical is None and (
                target_source is None or target_source["source_hash"] != intent.get("source_hash")
            ):
                reasons.append("Owner interpretation source changed")
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append(f"Owner interpretation identity is invalid: {exc}")
    return {
        "status": "blocked" if reasons else "clear",
        "identity": {
            "revision_id": target,
            "source": sources[target_location],
            "application_hash": applications[target_location],
            "pending": sorted(pending_identities),
            "intent_hash": intent_hash,
        },
        "reasons": reasons,
    }


def input_authority_status(value: Any) -> str:
    """Derive captured authority state from typed identity and explicit refusal reasons."""
    if not isinstance(value, dict) or not isinstance(value.get("identity"), dict):
        return "blocked"
    identity = value["identity"]
    reasons = value.get("reasons")
    source = identity.get("source")
    pending = identity.get("pending")

    def checksum(item: Any) -> bool:
        return isinstance(item, str) and re.fullmatch(r"[a-f0-9]{64}", item) is not None

    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or not isinstance(source, dict)
        or type(source.get("source_count")) is not int
        or source["source_count"] < 0
        or not checksum(source.get("source_hash"))
        or not isinstance(pending, list)
        or any(not checksum(item) for item in pending)
        or not checksum(identity.get("intent_hash"))
        or (
            identity.get("application_hash") is not None
            and not checksum(identity["application_hash"])
        )
    ):
        return "blocked"
    return "blocked" if reasons or pending else "clear"


def _instruction_context(context: dict[str, Any]) -> dict[str, Any]:
    context["instruction_catalog"] = instruction_catalog(context)
    context["instruction_binding"] = resolve_instruction_plan(context)
    return context


def _facts(revision: dict[str, Any], *, session: dict[str, Any] | None = None) -> dict[str, Any]:
    from forge_mind.workspace.customer_delivery_policy import customer_delivery_context
    from forge_mind.workspace.native_delivery import requested_mesh

    document = DesignDocument.model_validate(revision["document"])
    policy = customer_delivery_context(session) if session is not None else None
    include_stl = (
        requested_mesh(revision["document"], session)
        if session is not None and policy is not None
        else True
    )
    measured = revision.get("measurement", {}).get("facts") or {}
    summary: dict[str, Any] = {}
    for key in ("extents_mm", "bbox_min", "bbox_max"):
        value = measured.get(key)
        if (
            isinstance(value, list)
            and len(value) == 3
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in value
            )
        ):
            summary[key] = value
    volume = measured.get("volume_mm3")
    if (
        isinstance(volume, (int, float))
        and not isinstance(volume, bool)
        and math.isfinite(volume)
        and volume >= 0
    ):
        summary["volume_mm3"] = volume
    count = measured.get("solid_count")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        summary["solid_count"] = count
    if isinstance(measured.get("step_valid"), bool):
        summary["step_valid"] = measured["step_valid"]
    if isinstance(measured.get("holes"), list):
        summary["hole_count"] = len(measured["holes"])
    values = document.values()
    inputs = dict(list(document.variables.items())[:64])
    evaluated = dict(list(values.items())[:64])
    parts = document.assembly.parts if document.assembly else []
    quantities = Counter(
        row["definition"] for row in (measured.get("assembly") or {}).get("occurrences", [])
    )
    measured_occurrences = (measured.get("assembly") or {}).get("occurrences")
    instruction_parts = [
        {
            "part_id": part.id,
            "declared_name": part.name,
            "role": part.production.role,
            "material": part.production.material.model_dump(mode="json")
            if part.production.material is not None
            else None,
            "exported_occurrence_quantity": quantities[part.id]
            if isinstance(measured_occurrences, list)
            else None,
        }
        for part in parts
    ]
    if document.assembly is None:
        production = document.production
        instruction_parts = [
            {
                "part_id": "part",
                "declared_name": document.name,
                "role": production.role if production is not None else "unspecified",
                "material": production.material.model_dump(mode="json")
                if production is not None and production.material is not None
                else None,
                "exported_occurrence_quantity": 1
                if type(measured.get("solid_count")) is int and measured["solid_count"] == 1
                else None,
            }
        ]
    result = {
        "measured": summary,
        "package_files": package_file_inventory(
            document,
            revision.get("measurement", {}).get("artifact_hashes", {}),
            include_stl=include_stl,
            include_import_assets=bool(referenced_import_asset_ids(document.model_dump())),
        ),
        "requirement_findings": revision.get("measurement", {}).get("findings", []),
        "measured_scope": "Stored CAD measurements for this revision; physical performance is unverified.",
        "declared_variables": {
            "inputs": inputs,
            "evaluated": evaluated,
            "inputs_total": len(document.variables),
            "evaluated_total": len(values),
            "inputs_truncated": len(inputs) < len(document.variables),
            "evaluated_truncated": len(evaluated) < len(values),
            "scope": VARIABLE_SCOPE,
        },
        "parts": [
            {
                "part_id": part.id,
                "name": part.name,
                "output": part.output,
                "production": part.production.model_dump(),
                "exported_occurrence_quantity": quantities[part.id],
            }
            for part in parts[:32]
        ],
        "parts_total": len(parts),
        "parts_truncated": len(parts) > 32,
        "instruction_parts": instruction_parts,
    }
    raw_contract = revision.get("contract") or {}
    if any(r.get("verification") is not None for r in raw_contract.get("requirements", [])):
        contract = Contract.model_validate(raw_contract)
        if any(requirement.verification is not None for requirement in contract.requirements):
            result["requirement_findings_stage"] = "build_time_before_final_contract_verification"
    return result


def _locked_context(
    service: WorkspaceService,
    con: sqlite3.Connection,
    sid: str,
    owner: str,
    rid: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = con.execute(
        "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
    ).fetchone()
    if row is None:
        raise WorkspaceError("workspace not found")
    session: dict[str, Any] = json.loads(row[0])
    selected = session.get("selected_revision")
    proposal = session.get("proposal") or {}
    design_head = proposal.get("revision_id") or selected
    target = rid or selected
    if not isinstance(target, str):
        raise WorkspaceError("Customer guidance requires a saved structured revision")
    revision_row = con.execute(
        "SELECT document FROM revisions WHERE session=? AND id=?", (sid, target)
    ).fetchone()
    if revision_row is None:
        raise WorkspaceError("Customer guidance requires a saved structured revision")
    revision: dict[str, Any] = json.loads(revision_row[0])
    if not revision.get("document"):
        raise WorkspaceError("Customer guidance requires a saved structured revision")
    measurement = Measurement.model_validate(revision["measurement"])
    if not measurement.geometry_valid:
        raise WorkspaceError("Customer guidance requires valid saved geometry")
    if not verify_artifacts(service.store.directory(sid, target), measurement):
        raise WorkspaceError("Revision artifacts changed or are missing")
    engineering_row = con.execute(
        "SELECT version,document FROM engineering_inputs WHERE session=?", (sid,)
    ).fetchone()
    engineering = (
        {"version": engineering_row[0], "inputs": json.loads(engineering_row[1])}
        if engineering_row
        else {"version": 0, "inputs": EngineeringInputs().model_dump()}
    )
    effective_contract = session.get("contract")
    for name in ("proposal", "resume_candidate"):
        candidate = session.get(name) or {}
        if candidate.get("revision_id") == target:
            effective_contract = candidate.get("contract", effective_contract)
    plan = session.get("design_plan")
    captured_plan = (revision.get("operation_record") or {}).get("plan_snapshot")
    pending_binding = None
    if target == proposal.get("revision_id") and target != selected:
        pending_binding = proposal.get("operation_plan_snapshot")
        if session.get("authoring_mode") == "native_operations" and pending_binding is None:
            raise WorkspaceError("Pending native candidate requires its captured operation plan")
        if pending_binding is not None:
            if not isinstance(pending_binding, dict) or pending_binding != captured_plan:
                raise WorkspaceError("Pending candidate plan differs from its saved revision")
            active = service.store._edit_target_snapshot(session, pending_binding["contract_hash"])
            plan = (active or pending_binding).get("plan")
    # Keep the exact completion binding; extra guidance fields must not alter cache keys.
    completion_basis = {
        "revision_id": target,
        "contract_hash": _hash(revision.get("contract", session.get("contract"))),
        "session_contract_hash": _hash(effective_contract),
        "engine_fingerprint": engine_fingerprint(),
        "source_fingerprint": source_fingerprint(con, sid),
        "input_version": engineering["version"],
        "plan_hash": _hash(plan),
        "artifact_hashes": measurement.artifact_hashes,
    }
    facts = _facts(revision, session=session)
    from forge_mind.workspace.plan_history import historical_check_context

    historical_checks = historical_check_context(con, sid, session, plan)
    if historical_checks:
        facts["historical_check_amendments"] = historical_checks
    if session.get("authoring_mode") == "native_operations":
        from forge_mind.workspace.completion_coverage import functional_coverage
        from forge_mind.workspace.intent_attribution_service import request_text

        facts["functional_coverage"] = functional_coverage(
            plan or {},
            effective_contract or {},
            source_text=request_text(session) if isinstance(session.get("prompt"), str) else None,
        )
    from forge_mind.workspace.completion_readonly import project_motion

    motion, motion_proof = project_motion(
        session, revision, completion_basis, root=service.store.root
    )
    if motion or measurement.facts.get("motion_rig") or revision["document"].get("motion_rig"):
        facts["motion_check"] = {
            key: deepcopy(value)
            for key, value in motion.items()
            if key not in {"check_id", "verified_evidence"}
        } or {
            "status": "unverified",
            "reason": "No current saved motion receipt is available",
            "scope": "Discrete motion evidence only; physical readiness remains unverified",
            "qualification_status": "unverified",
        }
        if motion_proof is not None:
            facts["motion_check"]["verified_evidence"] = deepcopy(motion_proof)
            for key in ("collisions", "errors", "scope"):
                if key in motion_proof:
                    facts["motion_check"][key] = deepcopy(motion_proof[key])
    completion_proof_hash = None
    if (plan or {}).get("functional_interfaces"):
        from forge_mind.workspace.completion_readonly import project_interfaces

        projected, completion_proof_hash = project_interfaces(
            con,
            sid,
            session,
            plan or {},
            revision,
            completion_basis,
            root=service.store.root,
        )
        facts["functional_interfaces"] = {
            **projected,
            "scope": "Current saved construction and verified receipt evidence only. Nominal checks do not establish physical function; missing or stale receipts remain unverified.",
        }
    setup_records = session.get("print_setup_records", {})
    if not isinstance(setup_records, dict):
        raise WorkspaceError("Saved print setup records are invalid")
    setup = None
    if target in setup_records:
        from forge_mind.workspace.print_setup_service import _locked_print_setup

        setup = _locked_print_setup(service, con, sid, owner, target)
        record = setup.get("record")
        if not isinstance(record, dict):
            raise WorkspaceError("Saved print setup has no validated record")
        facts["print_setup"] = {
            "status": setup["status"],
            "record_hash": setup["record_hash"],
            "files": record["files"],
            "selected_candidates": record["selected_candidates"],
            "selected_poses": {
                part_id: next(
                    candidate
                    for candidate in part["candidates"]
                    if candidate["candidate_hash"] == part["recommended_candidate_hash"]
                )
                for part_id, part in record["result"]["parts"].items()
            },
            "package": record["package"],
            "scope": "Saved rigid plate placement; printer settings, supports and physical success remain unverified.",
        }
    # Match completion._snapshot without its source-table initialization writes.
    basis = {
        "session_id": sid,
        **completion_basis,
        "document_hash": digest(revision["document"]),
        "engineering_inputs_hash": digest(engineering),
        "measured_engine_fingerprint": measurement.engine_fingerprint,
        "measured_facts_hash": digest(measurement.facts),
        "facts_hash": digest(facts),
    }
    if completion_proof_hash is not None:
        basis["completion_evidence_hash"] = completion_proof_hash
    if motion_proof is not None:
        basis["motion_evidence_hash"] = _hash(motion_proof)
    if setup is not None:
        basis["print_setup_record_hash"] = setup["record_hash"]
    input_authority = _input_authority(session, target)
    basis["input_authority_hash"] = digest(input_authority["identity"])
    basis_hash = digest(basis)
    records = _guidance_records(session)
    target_record = records.get(target)
    record = target_record
    record_source = "target"
    if record is None and target == selected:
        # Preserve earlier selected instructions as stale after a numeric edit.
        # Pending and historical targets must still use their own exact record.
        legacy = session.get("delivery_guidance_record")
        if isinstance(legacy, dict):
            record = legacy
            record_source = "selected_fallback"
    status = "unbound"
    if record is not None:
        status = "stale"
        if isinstance(record, dict):
            try:
                guidance = ReviewedDeliveryGuidance.model_validate(record.get("guidance"))
                current = (
                    target in {selected, design_head}
                    and record.get("basis") == basis
                    and record.get("basis_hash") == basis_hash
                    and record.get("guidance_hash") == digest(guidance.model_dump(mode="json"))
                    and record.get("reviewed_current_facts") is True
                    and record.get("scope") == SCOPE
                    and measurement.engine_fingerprint == basis["engine_fingerprint"]
                    and input_authority["status"] == "clear"
                    and (
                        "input_authority_hash" not in record
                        or (
                            input_authority["status"] == "clear"
                            and record["input_authority_hash"] == basis["input_authority_hash"]
                        )
                    )
                )
                if current:
                    status = "current"
            except ValueError:
                pass  # Retain invalid historical declarations, explicitly stale.
    stored_history = session.get("delivery_guidance_history", {})
    history_present = not isinstance(stored_history, dict) or target in stored_history
    history = (
        deepcopy(stored_history.get(target, []))
        if isinstance(stored_history, dict)
        else deepcopy(stored_history)
    )
    context = {
        "basis": basis,
        "basis_hash": basis_hash,
        "facts": facts,
        "status": status,
        "record": record,
        "input_authority": input_authority,
        "history": history,
        "history_status": guidance_history_status(
            history, target, target_record, present=history_present
        ),
        "target_kind": "pending_candidate"
        if target == design_head and target != selected
        else "selected_candidate"
        if target == selected
        else "historical",
        "selected_revision": selected,
        "accepted_revision": session.get("accepted_revision"),
        "design_revision": design_head,
        "captured_plan_hash": captured_plan.get("plan_hash")
        if isinstance(captured_plan, dict)
        else None,
        "legacy_guidance": revision["document"].get("delivery_guidance"),
        "legacy_guidance_status": "unbound",
        "scope": SCOPE,
        "physical_readiness": "unverified",
    }
    origin = guidance_record_revision(record)
    context["record_provenance"] = guidance_record_provenance(
        context,
        source=record_source,
        origin_history=(
            deepcopy(stored_history.get(origin, []))
            if isinstance(stored_history, dict)
            else deepcopy(stored_history)
        ),
        origin_history_present=not isinstance(stored_history, dict) or origin in stored_history,
    )
    provenance = context["record_provenance"]
    if origin is not None and provenance is not None:
        try:
            source_matches = origin in records and digest(records[origin]) == digest(record)
        except (TypeError, ValueError, OverflowError):
            source_matches = False
        if not source_matches:
            origin_status = provenance["history_status"]
            provenance["history_status"] = {
                **origin_status,
                "status": "unverified",
                "errors": [
                    *origin_status["errors"],
                    "Displayed author record differs from its saved source-map head",
                ],
            }
    return session, _instruction_context(context)


def _guidance_records(session: dict[str, Any]) -> dict[str, Any]:
    """Read revision records and lazily retain a matching legacy selected record."""
    stored = session.get("delivery_guidance_records", {})
    if not isinstance(stored, dict):
        raise WorkspaceError("Revision guidance records are invalid")
    records = dict(stored)
    legacy = session.get("delivery_guidance_record")
    if isinstance(legacy, dict) and isinstance(legacy.get("basis"), dict):
        rid = legacy["basis"].get("revision_id")
        if isinstance(rid, str) and rid not in records:
            records[rid] = legacy
    return records


def guidance_context(
    service: WorkspaceService, sid: str, owner: str, rid: str | None = None
) -> dict[str, Any]:
    """Read selected facts by default, or an exact owned revision when explicit."""
    session = service.store.get(sid, owner)
    target = rid or session.get("selected_revision")
    _snapshot(service, sid, owner, target)  # Initialize sources before the read transaction.
    with service.store.connect() as con:
        con.execute("BEGIN")
        _, context = _locked_context(service, con, sid, owner, rid)
    return context


def update_guidance(
    service: WorkspaceService, sid: str, owner: str, request: GuidanceUpdate
) -> dict[str, Any]:
    """Compare and save declarations atomically; geometry and job state remain unchanged."""
    request = GuidanceUpdate.model_validate(request.model_dump())
    _snapshot(service, sid, owner, request.revision_id)
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        session, context = _locked_context(service, con, sid, owner, request.revision_id)
        if context["design_revision"] != request.revision_id:
            raise WorkspaceError("Current design revision changed; refresh customer guidance")
        if context["basis_hash"] != request.basis_hash:
            raise WorkspaceError("Guidance inputs changed; refresh the saved facts before updating")
        if context["input_authority"]["status"] != "clear":
            raise WorkspaceError(
                "Owner input must be applied and clarified before publishing customer guidance: "
                + "; ".join(context["input_authority"]["reasons"])
            )
        setup = context["facts"].get("print_setup")
        if setup is not None and setup["status"] != "current":
            raise WorkspaceError(
                "Saved print setup is stale; prepare current files before instructions"
            )
        if (
            context["basis"]["measured_engine_fingerprint"]
            != context["basis"]["engine_fingerprint"]
        ):
            raise WorkspaceError("Saved geometry was measured by a different engine")
        if (
            context["target_kind"] == "pending_candidate"
            and context["basis"]["contract_hash"] != context["basis"]["session_contract_hash"]
        ):
            raise WorkspaceError("Pending candidate differs from its active requirements")
        guidance = request.guidance.model_dump(mode="json")
        record = {
            "basis": context["basis"],
            "basis_hash": context["basis_hash"],
            "guidance": guidance,
            "guidance_hash": digest(guidance),
            "reviewed_current_facts": True,
            "scope": SCOPE,
            "input_authority_hash": context["basis"]["input_authority_hash"],
        }
        if request.instruction_plan is not None:
            try:
                plan = validate_instruction_plan(request.instruction_plan, context)
            except ValueError as exc:
                raise WorkspaceError(str(exc)) from exc
            record["instruction_plan"] = plan.model_dump(mode="json")
            record["instruction_plan_hash"] = digest(record["instruction_plan"])
        records = _guidance_records(session)
        history = _history_for_write(session, records)
        entries = history.setdefault(request.revision_id, [])
        entries.append(_history_entry(request.revision_id, record, entries))
        records[request.revision_id] = record
        session["delivery_guidance_records"] = records
        session["delivery_guidance_history"] = history
        if request.revision_id == session.get("selected_revision"):
            session["delivery_guidance_record"] = record
        con.execute(
            "UPDATE sessions SET document=? WHERE id=? AND owner=?",
            (json.dumps(session, allow_nan=False), sid, owner),
        )
        context.update(
            status="current",
            record=record,
            history=deepcopy(entries),
            history_status=guidance_history_status(entries, request.revision_id, record),
        )
        context["record_provenance"] = guidance_record_provenance(context, source="target")
    return _instruction_context(context)
