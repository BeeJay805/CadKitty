"""Durable customer file intent over the existing production and handoff readers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from forge_mind.workspace.customer_projects import customer_project
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.document import DesignDocument


def customer_print_requested(session: dict[str, Any]) -> bool:
    """An absent policy preserves historical behavior; only a strict true requests print files."""
    return customer_project(session) and session.get("customer_printable") is True


def customer_delivery_context(session: dict[str, Any]) -> dict[str, Any] | None:
    if not customer_project(session) or session.get("customer_request_version") != 2:
        return None
    printable = session.get("customer_printable")
    if type(printable) is not bool:
        raise WorkspaceError("Customer delivery policy must retain its strict printable choice")
    required = ["project_step", "editable_native", "bound_instruction_plan"]
    if printable:
        required.extend(["project_stl", "prepared_print_package"])
    return {
        "version": 2,
        "printable": printable,
        "required_deliverables": required,
        "scope": (
            "Server-retained software delivery choice, separate from owner source quotes and "
            "frozen geometry requirements. Print files are unsliced and plate aligned. "
            "No printer profile, material, supports, G-code, physical fit or print success is implied. "
            "Explicit owner print requirements and declared additive prerequisites remain active "
            "when printable is false."
        ),
    }


def customer_print_document_gaps(document: DesignDocument) -> list[str]:
    """Expose declaration gaps; actual meshes and print placement keep their existing readers."""
    from forge_mind.workspace.production import production_report

    report = production_report(document)
    gaps = []
    if document.assembly is None:
        gaps.append(
            "Use an explicit assembly with separately declared made parts for print delivery"
        )
    if report["unresolved"]:
        gaps.append("Declare production roles for every occurring part before print delivery")
    if not report["make"]:
        gaps.append("Print delivery requires at least one occurring made part")
    for part in report["make"]:
        if part["production"].get("deliver_stl") is not True:
            gaps.append(f"Declare deliver_stl=true for made part {part['part_id']} before building")
    return gaps
