"""Requested native saved-file capture, separate from author and approval state."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.budget import expired, time_left
from forge_mind.workspace.completion import _snapshot
from forge_mind.workspace.customer_delivery_policy import (
    customer_delivery_context,
    customer_print_document_gaps,
    customer_print_requested,
)
from forge_mind.workspace.document import digest
from forge_mind.workspace.handoff import referenced_import_asset_ids
from forge_mind.workspace.project_delivery import create_project_export, read_project_export
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def _declared_mesh(document: dict[str, Any]) -> bool:
    productions = [document.get("production") or {}]
    productions.extend(
        part.get("production") or {} for part in (document.get("assembly") or {}).get("parts", [])
    )
    return any(p.get("role") == "make" and p.get("deliver_stl") is True for p in productions)


def requested_mesh(document: dict[str, Any], session: dict[str, Any]) -> bool:
    """Use the same retained mesh choice for automatic capture and prospective facts."""
    return _declared_mesh(document) or customer_print_requested(session)


def current_delivery_binding(service: WorkspaceService, sid: str, owner: str) -> dict[str, Any]:
    """Share the existing exact saved-file basis with the postcapture verifier."""
    current = service.store.get(sid, owner)
    selected = current.get("selected_revision")
    value = service._load_revision(sid, selected) if selected else {}
    measurement = value.get("measurement") or {}
    return {
        **_snapshot(service, sid, owner, selected),
        "measurement_engine_fingerprint": measurement.get("engine_fingerprint"),
        "document_hash": digest(value.get("document") or {}),
        "engineering_inputs_hash": digest(current.get("engineering") or {}),
        "guidance_record_hash": digest(current.get("delivery_guidance_record")),
    }


def _customer_policy_binding(
    service: WorkspaceService, sid: str, owner: str, rid: str, document: dict[str, Any]
) -> dict[str, Any] | None:
    """Reuse current guidance and immutable print readers; add no geometry or physical verdict."""
    from forge_mind.workspace.delivery_guidance import guidance_context
    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.print_setup_service import print_setup_context
    from forge_mind.workspace.production import production_report

    session = service.store.get(sid, owner)
    policy = customer_delivery_context(session)
    if policy is None:
        return None
    parsed = DesignDocument.model_validate(document)
    if customer_print_requested(session):
        gaps = customer_print_document_gaps(parsed)
        if gaps:
            raise WorkspaceError("; ".join(gaps))
    guide = guidance_context(service, sid, owner, rid)
    if (
        guide.get("status") != "current"
        or guide.get("target_kind") != "selected_candidate"
        or guide.get("selected_revision") != rid
        or (guide.get("instruction_binding") or {}).get("status") != "bound"
    ):
        raise WorkspaceError(
            "Customer delivery requires current guidance and bound instruction steps"
        )
    result = {
        "policy": policy,
        "guidance_basis_hash": guide["basis_hash"],
        "guidance_record_hash": digest(guide["record"]),
    }
    if customer_print_requested(session):
        # This existing reader verifies ZIP/member bytes and the exact live setup basis.
        setup = print_setup_context(service, sid, owner, rid)
        record = setup.get("record") or {}
        files = record.get("files") or {}
        expected = {row["part_id"]: row["quantity"] for row in production_report(parsed)["make"]}
        actual = {entry["part_id"]: entry["quantity"] for entry in files.values()}
        if (
            setup.get("status") != "current"
            or len(files) != len(expected)
            or actual != expected
            or (record.get("package") or {}).get("revision_id") != rid
        ):
            raise WorkspaceError(
                "Customer print delivery requires exact current all-made prepared files"
            )
        result.update(print_setup_record_hash=setup["record_hash"], print_package=record["package"])
    return result


def capture_native_delivery(
    service: WorkspaceService,
    job: dict[str, Any],
    deadline: float | None,
    cancelled: Callable[[], bool],
    author_state: str,
) -> dict[str, Any]:
    """Capture requested current files after finalization; never accept or rebuild CAD.

    The caller clears the previous current delivery when a new job starts. A returned
    package remains recorded if cancellation or budget exhaustion occurs afterward.
    Capture/readback are synchronous existing services, so guards cannot interrupt IO.
    """
    sid, owner = job["session"], job["owner"]
    session = service.store.get(sid, owner)
    report: dict[str, Any] = {
        "job_id": job["id"],
        "revision_id": session.get("selected_revision"),
        "author_state": author_state,
        "status": "not_requested",
        "receipt": None,
        "scope": "Saved private review files only; no acceptance or physical approval.",
        "limits": {
            "deadline_supplied": deadline is not None,
            "synchronous_capture": True,
            "scope": "Cancellation and deadline checked around capture/readback; IO is not preemptible.",
        },
    }

    def persist(status: str, reason: str) -> dict[str, Any]:
        report.update(status=status, reason=reason, remaining_seconds=time_left(deadline))
        service.store.update(sid, automatic_delivery=report)
        return report

    def stopped() -> str | None:
        if cancelled():
            return "cancelled"
        if expired(deadline):
            return "budget_exhausted"
        return None

    try:
        if session.get("authoring_mode") != "native_operations":
            return persist("not_requested", "Automatic native delivery does not apply")
        if author_state == "needs_approval":
            return persist("incomplete", "Owner plan approval remains; no package may be captured")
        rid = report["revision_id"]
        revision = service._load_revision(sid, rid) if rid else {}
        document = revision.get("document") or {}
        mesh_requested = requested_mesh(document, session)
        if not mesh_requested and job.get("delivery_requested") is not True:
            return persist("not_requested", "No explicit saved-file delivery request")
        status = stopped()
        if status:
            return persist(status, "Delivery stopped before capture")
        if author_state in {"error", "cancelled"}:
            return persist("incomplete", "Author did not complete this phase; no package captured")
        if job.get("preview_only"):
            return persist("incomplete", "Preview phase does not create automatic delivery")
        if job.get("parameters") and rid == (
            job.get("native_delivery_baseline") or job.get("based_on") or job.get("revision_id")
        ):
            return persist("incomplete", "Parameter edit produced no new selected candidate")
        if not rid or not document or not revision.get("measurement", {}).get("geometry_valid"):
            return persist("incomplete", "No valid structured selected revision is available")
        from forge_mind.workspace.photo_generation import photo_closure

        photo_review = photo_closure(service, sid, owner, rid)
        if photo_review["status"] not in {"checked", "not_applicable"}:
            return persist("incomplete", photo_review["reason"])

        def binding() -> dict[str, Any]:
            return current_delivery_binding(service, sid, owner)

        frozen = binding()
        report.update(frozen)
        status = stopped()
        if status:
            return persist(status, "Delivery stopped before capture")
        if frozen["revision_id"] != rid:
            return persist("incomplete", "Selected revision changed before capture")
        if frozen["measurement_engine_fingerprint"] != frozen["engine_fingerprint"]:
            return persist("incomplete", "Saved geometry was measured by a different engine")
        try:
            policy_binding = _customer_policy_binding(service, sid, owner, rid, document)
        except WorkspaceError as exc:
            return persist("incomplete", str(exc))
        receipt = create_project_export(
            service,
            sid,
            owner,
            rid,
            include_stl=mesh_requested,
            include_import_assets=bool(referenced_import_asset_ids(document)),
        )
        report["receipt"] = receipt
        report["guidance_status"] = receipt.get("guidance_status", "unbound")
        persist("captured", "Exact package captured; readback is pending")
        status = stopped()
        if status:
            return persist(status, "Package retained; delivery stopped after capture")
        if binding() != frozen:
            return persist("incomplete", "Package retained; selected inputs changed after capture")
        content, readback = read_project_export(
            service, sid, owner, receipt["package_id"], receipt["sha256"]
        )
        status = stopped()
        if status:
            return persist(status, "Package retained; delivery stopped after readback")
        if binding() != frozen:
            return persist(
                "incomplete", "Package retained; selected inputs changed during readback"
            )
        if (
            readback != receipt
            or receipt.get("project_id") != sid
            or receipt.get("revision_id") != rid
            or receipt.get("bytes") != len(content)
            or receipt.get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            return persist("failed", "Package readback does not match the captured identity")
        if customer_print_requested(session):
            from forge_mind.workspace.document import DesignDocument
            from forge_mind.workspace.production import production_report

            made = production_report(DesignDocument.model_validate(document))["make"]
            expected = {part["part_id"]: part["quantity"] for part in made}
            captured_parts = receipt.get("made_parts") or []
            if (
                receipt.get("include_stl") is not True
                or len(captured_parts) != len(expected)
                or {part["part_id"]: part["quantity"] for part in captured_parts} != expected
                or any(
                    set(part.get("files") or [])
                    != {
                        f"make/definition_{part['part_id']}.step",
                        f"make/definition_{part['part_id']}.stl",
                    }
                    for part in captured_parts
                )
            ):
                return persist(
                    "incomplete", "Captured customer files lack exact all-made STEP/STL coverage"
                )
        try:
            if _customer_policy_binding(service, sid, owner, rid, document) != policy_binding:
                return persist("incomplete", "Customer delivery inputs changed during readback")
        except WorkspaceError as exc:
            return persist("incomplete", str(exc))
        report["readback_verified"] = True
        return persist("delivered", "Exact saved package captured and read back")
    except Exception as exc:
        return persist("failed", f"{type(exc).__name__}: {str(exc)[:500]}")
