"""One explicit bounded preparation action; final instructions and proof stay separate."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from forge_mind.workspace.budget import expired, time_left
from forge_mind.workspace.delivery_guidance import GuidanceUpdate, guidance_context
from forge_mind.workspace.document import digest
from forge_mind.workspace.models import Record
from forge_mind.workspace.native_finish import finish_context
from forge_mind.workspace.print_setup_service import (
    WORKER_TIMEOUT,
    prepare_print_files,
    print_setup_context,
)
from forge_mind.workspace.requirement_verification import _frozen_inputs, inspect_requirements
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


class PrepareHandoffRequest(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    prepare_print: bool = Field(
        strict=True,
        description="Explicitly request existing all-made unsliced print preparation; false performs no print write.",
    )


def prepare_handoff(
    service: WorkspaceService,
    sid: str,
    owner: str,
    jid: str,
    request: PrepareHandoffRequest,
    *,
    deadline: float | None,
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    """Inspect, optionally prepare, then return exact current instruction inputs.

    No whole-action atomicity is promised: an existing print receipt can survive
    a later stale/cancelled read. This function never writes instructions, plan,
    CAD, selection, acceptance, final capture or a completion verdict.
    """
    request = PrepareHandoffRequest.model_validate(request.model_dump())

    def stopped() -> None:
        if cancelled() or expired(deadline):
            raise WorkspaceError("Handoff preparation cancelled or deadline exhausted")

    def basis() -> dict[str, Any]:
        stopped()
        session, _, current = _frozen_inputs(service.store, sid, owner)
        if (
            session.get("authoring_mode") != "native_operations"
            or session.get("selected_revision") != request.revision_id
            or session.get("job_id") != jid
        ):
            raise WorkspaceError(
                "Handoff preparation requires this job's current selected revision"
            )
        with service.store.connect() as con:
            job = con.execute(
                "SELECT state,cancel FROM jobs WHERE id=? AND session=?", (jid, sid)
            ).fetchone()
        if job is None or job[0] != "running" or job[1]:
            raise WorkspaceError("Handoff preparation job is no longer active")
        return current

    expected = basis()

    def unchanged() -> dict[str, Any]:
        current = basis()
        if current != expected:
            raise WorkspaceError("Handoff preparation inputs changed; refresh the current basis")
        return current

    review = finish_context(service, sid, owner, request.revision_id)
    unchanged()
    reasons = review["work_reasons"]
    preparation = review["print_preparation"]
    # finish_context always includes a print work reason for this exact structured
    # missing-preparation state. Permit that sole prerequisite only when explicitly
    # requested; never suppress another blocker or interpret its prose.
    only_requested_print = (
        request.prepare_print
        and preparation["required"]
        and preparation.get("supported", True)
        and preparation["status"] != "current"
        and len(reasons) == 1
    )
    if reasons and not only_requested_print:
        return {
            "status": "blocked",
            "revision_id": request.revision_id,
            "basis_hash": digest(expected),
            "work_reasons": reasons,
            "print_preparation": preparation,
            "physical_readiness": "unverified",
        }
    inspection = inspect_requirements(
        service, sid, owner, request.revision_id, deadline=deadline, cancelled=cancelled
    )
    unchanged()
    if inspection.get("revision_id") != request.revision_id or inspection.get(
        "basis_hash"
    ) != digest(expected):
        raise WorkspaceError("Handoff inspection returned a different saved basis")
    geometric_methods = {"nominal_through_hole_pattern_v1", "nominal_perforated_box_v1"}
    if any(
        row.get("status") == "fail"
        or (row.get("method") in geometric_methods and row.get("status") != "pass")
        for row in inspection["results"]
    ):
        return {
            "status": "blocked",
            "revision_id": request.revision_id,
            "basis_hash": digest(expected),
            "requirements_inspection": inspection,
            "work_reasons": ["Repair the failed or unverified saved requirement inspection"],
            "physical_readiness": "unverified",
        }
    print_receipt = None
    if request.prepare_print:
        unchanged()
        remaining = time_left(deadline)
        print_receipt = prepare_print_files(
            service,
            sid,
            owner,
            request.revision_id,
            timeout=WORKER_TIMEOUT if remaining is None else min(WORKER_TIMEOUT, remaining),
            cancelled=lambda: cancelled() or expired(deadline),
        )
        current = basis()
        next_expected = {**expected, "print_setup_record_hash": current["print_setup_record_hash"]}
        if current != next_expected:
            raise WorkspaceError("Handoff inputs changed during print preparation")
        expected = next_expected
        setup = print_setup_context(service, sid, owner, request.revision_id)
        unchanged()
        if (
            setup.get("status") != "current"
            or (setup.get("record") or {}).get("package") != print_receipt
        ):
            return {
                "status": "blocked",
                "revision_id": request.revision_id,
                "requirements_inspection": inspection,
                "print_setup": print_receipt,
                "work_reasons": ["Print preparation did not produce a current verified package"],
                "physical_readiness": "unverified",
            }
    review = finish_context(service, sid, owner, request.revision_id)
    unchanged()
    if review["work_reasons"]:
        return {
            "status": "blocked",
            "revision_id": request.revision_id,
            "requirements_inspection": inspection,
            "print_setup": print_receipt,
            "work_reasons": review["work_reasons"],
            "physical_readiness": "unverified",
        }
    guidance = guidance_context(service, sid, owner, request.revision_id)
    unchanged()
    return {
        "status": "prepared_for_instruction_review",
        "revision_id": request.revision_id,
        "basis_hash": digest(expected),
        "requirements_inspection": inspection,
        "print_setup": print_receipt,
        "delivery_guidance": guidance,
        "guidance_schema": GuidanceUpdate.model_json_schema(),
        "physical_readiness": "unverified",
        "scope": "Preparation only. Submit current bound instructions; final capture/readback and every frozen requirement proof still run. Existing prepared files may remain if a later step fails.",
    }
