"""Fresh native review closure checks; never acceptance or physical approval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from forge_mind.workspace.completion import completion_context
from forge_mind.workspace.customer_delivery_policy import (
    customer_delivery_context,
    customer_print_document_gaps,
    customer_print_requested,
)
from forge_mind.workspace.delivery_guidance import guidance_context
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.models import Contract, DesignPlan
from forge_mind.workspace.native_collision_closure import reference_collision_context
from forge_mind.workspace.native_preflight import print_preparation
from forge_mind.workspace.operation_plan import coverage
from forge_mind.workspace.plan_reconciliation import alignment
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def finish_context(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str | None,
    *,
    require_bound_instructions: bool = False,
) -> dict[str, Any]:
    """Re-read exact saved state before an explicit terminal review request."""
    session = service.store.get(sid, owner)
    if not rid or session.get("selected_revision") != rid:
        raise WorkspaceError("Selected revision changed; refresh the final review")
    revision = service._load_revision(sid, rid)
    guide = guidance_context(service, sid, owner, rid)
    plan = DesignPlan.model_validate(session.get("design_plan"))
    document = DesignDocument.model_validate(revision.get("document"))
    status = alignment(plan, document)
    captured = (revision.get("operation_record") or {}).get("plan_snapshot")
    reasons = []
    if guide["status"] != "current":
        reasons.append("Refresh all four instructions against current saved facts")
    instruction_binding = guide.get("instruction_binding") or {}
    if (
        require_bound_instructions or customer_delivery_context(session) is not None
    ) and instruction_binding.get("status") != "bound":
        reasons.append("Supply an instruction plan bound to the current supported fact references")
    instruction_reason_count = len(reasons)
    from forge_mind.workspace.photo_generation import photo_closure

    photo_review = photo_closure(service, sid, owner, rid)
    if photo_review["status"] not in {"checked", "not_applicable"}:
        reasons.append(photo_review["reason"])
    if status["status"] != "aligned":
        reasons.append("Reconcile active dimension facts with current named bindings")
    if not captured or captured.get("plan_hash") != session.get("plan_hash"):
        reasons.append("Build the current document with the updated operation plan before review")
    else:
        graph = coverage(captured, document)
        if not graph["graph_complete"] or not graph["references_complete"]:
            reasons.append("Complete the planned construction and references before final review")
        if not graph["parameters_complete"]:
            missing_steps = [
                step["step_id"]
                for step in graph["steps"]
                if step["kind"] == "parameters" and not step["parameters_present"]
            ]
            reasons.append(
                f"Complete the planned parameter steps before final review: {missing_steps}"
            )
    contract = Contract.model_validate(revision["contract"])
    if session.get("plan_contract_hash") != contract.fingerprint():
        reasons.append("The current plan and selected candidate have different requirements")
    basis = guide["basis"]
    if (
        not basis.get("contract_hash")
        or basis.get("session_contract_hash") != basis["contract_hash"]
    ):
        reasons.append(
            "Bind the current plan and rebuilt CAD to the active requirements before review"
        )
    completion = completion_context(service, sid, owner, rid)
    if completion.get("next_target"):
        reasons.append("Complete or resolve actionable subsystem work before final review")
    reference_clearance = reference_collision_context(service, sid, owner, revision, document)
    if reference_clearance["status"] not in {"checked", "not_applicable"}:
        reasons.append(reference_clearance["reason"])
    preparation = print_preparation(
        document,
        guide,
        explicitly_requested=customer_print_requested(session)
        or any(
            getattr(requirement.verification, "deliverable", None) == "prepared_print_package"
            for requirement in contract.requirements
        ),
    )
    if customer_print_requested(session):
        reasons.extend(customer_print_document_gaps(document))
    if preparation["required"] and not preparation.get("supported", True):
        reasons.append(
            "Convert flat additive CAD to an explicit single-part assembly with preserved Features and bindings, then prepare current plate-aligned print files"
        )
    elif preparation["required"] and preparation["status"] != "current":
        reasons.append(
            "Prepare current plate-aligned print files for declared additive parts: "
            + ", ".join(preparation["part_ids"])
        )
    # Recheck after reads. Later validation/capture retain their own immutable bindings.
    latest = guidance_context(service, sid, owner, rid)
    if (
        latest["basis_hash"] != guide["basis_hash"]
        or latest["status"] != guide["status"]
        or latest.get("record") != guide.get("record")
        or latest.get("instruction_binding") != guide.get("instruction_binding")
    ):
        raise WorkspaceError("Review inputs changed while checking closure")
    return {
        "status": "needs_work" if reasons else "ready_for_validation",
        "revision_id": rid,
        "basis_hash": guide["basis_hash"],
        "instruction_binding": instruction_binding,
        "plan_hash": session.get("plan_hash"),
        "reasons": reasons,
        "work_reasons": reasons[instruction_reason_count:],
        "print_preparation": preparation,
        "dimension_alignment": status,
        "reference_clearance": reference_clearance,
        "photo_generation": photo_review,
        "requirement_findings": revision["measurement"].get("findings", []),
        "functional_interfaces": completion.get("functional_interfaces"),
        "functional_coverage": completion.get("functional_coverage"),
        "external_obligations": completion.get("external_obligations", []),
        "physical_readiness": "unverified",
        "scope": "Explicit author review closure only; saved validation and exact delivery still run. Failed and unverified requirements remain open.",
    }
