"""Expose existing plan-patch syntax at actual blockers; never author intent."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from forge_mind.workspace.edit_contracts import EditBasis


def completion_amendment(
    context: dict[str, Any], workflow: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Offer an ordinary amendment without changing admission or any saved state."""
    if context.get("authoring_mode") != "native_operations":
        return None
    if workflow is not None and (
        workflow.get("approval_required") is not False or workflow.get("status") != "not_requested"
    ):
        # Approved commands retain their exact target, including downstream checks.
        return None
    plan = context.get("design_plan") or {}
    completion = context.get("completion") or {}
    if not plan.get("operation_plan") or completion.get("next_work_category") != "review":
        return None
    try:
        basis = EditBasis.model_validate(context.get("edit_basis"))
    except ValidationError:
        return None
    if basis.plan_hash is None:
        return None
    target = completion.get("next_target")
    requirements = {
        row["id"]
        for row in (context.get("frozen_contract") or {}).get("requirements", [])
        if row.get("metric", "unmeasured") == "unmeasured" and row.get("verification") is None
    }
    if target == "functional_coverage":
        candidates = (completion.get("functional_coverage") or {}).get(
            "missing_requirement_ids", []
        )
    else:
        subsystem = next(
            (row for row in completion.get("subsystems", []) if row.get("id") == target), None
        )
        if (
            subsystem is None
            or not subsystem.get("geometry_declared")
            or subsystem.get("missing_part_ids")
            or subsystem.get("missing_occurrence_paths")
        ):
            return None
        undeclared = {
            row["requirement_id"]
            for row in completion.get("requirement_dispositions", [])
            if row.get("declared") is False
        }
        candidates = [rid for rid in subsystem.get("requirement_ids", []) if rid in undeclared]
    missing = list(dict.fromkeys(rid for rid in candidates if rid in requirements))
    if not missing:
        return None

    from forge_mind.workspace.native_authoring import _core_schema
    from forge_mind.workspace.plan_patch import PlanPatch

    return {
        "target": target,
        "requirement_ids": missing,
        "basis": basis.model_dump(mode="json"),
        "action": {"kind": "plan", "query": "patch", "payload_field": "document_json"},
        "schema": _core_schema(
            PlanPatch, {"basis", "requirement_dispositions", "functional_interfaces"}
        ),
        "retained_requirement_dispositions": deepcopy(
            [
                row
                for row in plan.get("requirement_dispositions", [])
                if row["requirement_id"] in missing
            ]
        ),
        "retained_functional_interfaces": deepcopy(
            [
                row
                for row in plan.get("functional_interfaces", [])
                if set(row["requirement_ids"]) & set(missing)
            ]
        ),
        "scope": (
            "Optional existing PlanPatch syntax for these current intent gaps. Author the actual "
            "interface or meaningful external obligations; no obligation text is supplied. Retain "
            "existing obligations. Normal full merged-plan, owner and basis validation still applies. "
            "Refresh edit_basis after a save. Plan changes apply no CAD or rejected edit and grant "
            "no source authority, measurement, requirement pass or acceptance. Missing construction "
            "and final current guide/files remain required. Other plan fields use schema query=plan_patch."
        ),
    }
