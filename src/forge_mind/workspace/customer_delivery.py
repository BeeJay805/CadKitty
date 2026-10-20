"""Pure presentation of server-verified handoff facts, never another proof reader."""

from __future__ import annotations

from collections.abc import Collection
from copy import deepcopy
from typing import Any

from forge_mind.workspace.instruction_plan import resolve_instruction_plan


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _procedure_proposal(context: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    """Preserve an eligible author's pre-capture proposal as literal source history."""
    from forge_mind.workspace.delivery_guidance import (
        guidance_history_status,
        input_authority_status,
    )

    result: dict[str, Any] = {
        "status": "unavailable",
        "procedures": {},
        "scope": (
            "Saved author proposal from review before final package capture. "
            "These literal words can contain old workflow tasks or status statements. "
            "They are source history, not current lifecycle status or bound customer steps. "
            "Current server facts take priority. Procedure correctness and physical use remain unverified."
        ),
    }
    record = _mapping(context.get("record"))
    basis = _mapping(context.get("basis"))
    revision_id = basis.get("revision_id")
    if (
        context.get("status") != "current"
        or binding.get("status") != "bound"
        or not isinstance(revision_id, str)
        or input_authority_status(context.get("input_authority")) != "clear"
        or _mapping(context.get("history_status")).get("status") != "verified"
    ):
        return result
    history = guidance_history_status(context.get("history"), revision_id, record)
    if history["status"] != "verified":
        return result
    guidance = _mapping(record.get("guidance"))
    fields = ("print", "installation", "use_and_removal", "unresolved")
    if not all(
        isinstance(guidance.get(field), str) and guidance[field].strip() for field in fields
    ):
        return result
    result.update(
        status="available_unverified",
        procedures={field: guidance[field] for field in fields},
        source={
            "review_stage": "before_final_capture",
            "revision_id": revision_id,
            "basis_hash": context.get("basis_hash"),
            "guidance_hash": record.get("guidance_hash"),
            "history_head_hash": history["head_hash"],
        },
    )
    return result


def customer_delivery_view(
    guidance_context: dict[str, Any] | None,
    *,
    members: Collection[str] | None = None,
    package_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project an existing trusted context without IO, mutation or inferred fulfillment.

    ``members`` describes actual assembled project members, not a delivery receipt.
    Print files retain the separately verified print-package status and coordinates.
    A package inventory override changes file presentation only, never the saved
    instruction basis or the facts supplied to its resolver.
    """
    context = _mapping(guidance_context)
    facts = _mapping(context.get("facts"))
    inventory = (
        _mapping(package_inventory)
        if package_inventory is not None
        else _mapping(facts.get("package_files"))
    )
    actual = frozenset(members) if members is not None else None
    files: list[dict[str, Any]] = []
    for path, value in sorted(_mapping(inventory.get("geometry")).items()):
        entry = _mapping(value)
        files.append(
            {
                "path": path,
                "part_id": entry.get("part_id"),
                "role": entry.get("role"),
                "quantity": entry.get("quantity"),
                "source_present": entry.get("source_present") is True,
                "included": path in actual if actual is not None else None,
            }
        )
    omitted = []
    if actual is not None:
        expected = {entry["path"] for entry in files}
        expected.update(inventory.get("metadata_files") or [])
        omitted = sorted(expected - actual - {"README.md", "manifest.json"})
    if not inventory:
        project_status = "unavailable"
    elif actual is not None:
        project_status = "partial" if omitted else "included"
    else:
        project_status = (
            "available"
            if files and all(entry["source_present"] for entry in files)
            else "incomplete"
        )
    setup = _mapping(facts.get("print_setup"))
    print_status = setup.get("status", "unavailable")
    if print_status not in {"unavailable", "current", "stale", "missing"}:
        print_status = "unverified"
    guidance_status = context.get("status", "unavailable")
    if guidance_status not in {"unavailable", "current", "stale", "unbound"}:
        guidance_status = "unbound"
    findings = facts.get("requirement_findings") or []
    basis = _mapping(context.get("basis"))
    binding = resolve_instruction_plan(context, members=members)
    return {
        "project_files": {
            "status": project_status,
            "files": files,
            "omitted": omitted,
            "excluded": deepcopy(inventory.get("excluded_saved_artifacts") or []),
            "scope": (
                "Project members keep their original CAD frames. "
                "Inclusion is not captured delivery or physical approval."
            ),
        },
        "print_files": {
            "status": print_status,
            "files": deepcopy(_mapping(setup.get("files"))),
            "package": deepcopy(setup["package"])
            if isinstance(setup.get("package"), dict)
            else None,
            "scope": setup.get("scope")
            or (
                "Separate plate-aligned print files. "
                "Preparation does not establish printing success or physical fit."
            ),
        },
        "guidance": {
            "status": guidance_status,
            "revision_id": basis.get("revision_id"),
            "basis_hash": context.get("basis_hash"),
            "scope": context.get("scope")
            or (
                "Current means reviewed against saved facts, "
                "not verified instructions or physical approval."
            ),
        },
        "instruction_binding": binding,
        "procedure_proposal": _procedure_proposal(context, binding),
        "checks": {
            "failed": deepcopy([row for row in findings if row.get("status") == "fail"]),
            "unverified": deepcopy([row for row in findings if row.get("status") == "unverified"]),
            "interfaces": deepcopy(_mapping(facts.get("functional_interfaces"))),
            "coverage": deepcopy(_mapping(facts.get("functional_coverage"))),
            "motion": deepcopy(_mapping(facts.get("motion_check"))),
            "scope": (
                "Scoped checks and author-declared obligations retain their existing meaning; "
                "file receipts do not satisfy frozen requirements."
            ),
        },
        "physical_readiness": "unverified",
        "scope": (
            "Read-only customer presentation of supplied server facts. "
            "No delivered or readback status is inferred."
        ),
    }
