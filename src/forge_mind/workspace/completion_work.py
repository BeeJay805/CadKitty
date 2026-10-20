"""Classify native author work without closing acceptance or external obligations."""

from __future__ import annotations

from typing import Any


def remaining_work(
    rows: list[dict[str, Any]],
    dispositions: list[dict[str, Any]],
    operation_steps: list[dict[str, Any]] | None = None,
    interface_work: dict[str, Any] | None = None,
    *,
    pending_final_requirement_ids: set[str] | None = None,
    executable_requirement_ids: set[str] | None = None,
) -> dict[str, Any]:
    pending_final = pending_final_requirement_ids or set()
    executable = executable_requirement_ids or set()
    explicit_coverage = executable_requirement_ids is not None
    declared = {d["requirement_id"]: d for d in dispositions if d.get("declared")}
    by_id = {row["id"]: row for row in rows}
    actions = []
    external = []
    for row in rows:
        obligations = [
            {"requirement_id": rid, key: value}
            for rid in row["requirement_ids"]
            for key in ("missing_inputs", "unsupported_checks", "physical_validation")
            if (value := declared.get(rid, {}).get(key))
        ]
        external_ids = {
            rid
            for rid in row["requirement_ids"]
            if rid not in executable
            and any(
                declared.get(rid, {}).get(key)
                for key in ("unsupported_checks", "physical_validation")
            )
        }
        externally_declared = (
            bool(row["requirement_ids"]) and set(row["requirement_ids"]) <= external_ids
        )
        missing_geometry = bool(row.get("missing_part_ids") or row.get("missing_occurrence_paths"))
        if obligations:
            external.append({"subsystem_id": row["id"], "obligations": obligations})
        if row["status"] == "checked_fail" or any(
            check.get("status") == "fail" for check in row["checks"]
        ):
            category, reason, priority = "repair", "A current prepared check failed", 0
        elif missing_geometry or (row["status"] == "missing" and not externally_declared):
            category, reason, priority = "construction", "Requested geometry is missing", 1
        elif any(check.get("reason") == "Not run" for check in row["checks"]):
            category, reason, priority = "check", "A prepared CAD check remains executable", 2
        elif any(check.get("status") != "pass" for check in row["checks"]):
            category, reason, priority = (
                "review",
                "A prepared CAD check lacks current verified evidence",
                3,
            )
        elif any(declared.get(rid, {}).get("missing_inputs") for rid in row["requirement_ids"]):
            category, reason, priority = "review", "Resolve the declared missing source inputs", 3
        elif set(row["requirement_ids"]) & executable - pending_final:
            category, reason, priority = (
                "review",
                "Recheck or repair the current frozen requirement measurement",
                3,
            )
        elif any(item["requirement_id"] in executable for item in obligations):
            category, reason, priority = (
                "review",
                "Resolve the additional obligation on the executable requirement",
                3,
            )
        elif externally_declared and not row["checks"]:
            continue
        elif (
            not row["checks"]
            and row["geometry_declared"]
            and "unresolved" in row
            and (not row["unresolved"] or (explicit_coverage and external_ids))
            and (
                not obligations
                or (
                    explicit_coverage
                    and all(item["requirement_id"] in external_ids for item in obligations)
                )
            )
            and row["requirement_ids"]
            and set(row["requirement_ids"])
            <= pending_final | (external_ids if explicit_coverage else set())
        ):
            # Scheduling only. Final frozen evaluation still owns every verdict.
            continue
        elif not row["checks"] or not row["geometry_declared"]:
            category, reason, priority = "review", "Declare construction and a scoped check", 3
        else:
            continue
        if all(by_id[dependency]["status"] != "missing" for dependency in row["dependencies"]):
            actions.append((priority, row["id"], category, reason))
    covered = {rid for row in rows for rid in row["requirement_ids"]}
    for requirement_id in sorted(executable - pending_final - covered):
        actions.append(
            (
                3,
                requirement_id,
                "review",
                "Recheck or repair the current frozen requirement measurement",
            )
        )
    for step in operation_steps or []:
        if step.get("kind", "construction") == "construction" and not step["graph_present"]:
            actions.append(
                (1, step["step_id"], "construction", "A planned construction step is missing")
            )
        elif step.get("kind") == "parameters" and not step["parameters_present"]:
            actions.append(
                (1, step["step_id"], "parameters", "A planned parameter step is incomplete")
            )
    if interface_work:
        external.extend(interface_work.get("external_obligations", []))
        if interface_work.get("next_target"):
            category = interface_work["next_work_category"]
            priority = {"repair": 0, "construction": 1, "check": 2, "review": 3}[category]
            actions.append(
                (
                    priority,
                    interface_work["next_target"],
                    category,
                    interface_work["next_work_reason"],
                )
            )
    selected = min(actions, default=None, key=lambda item: item[0])
    result = {
        "next_target": selected[1] if selected else None,
        "next_work_category": selected[2] if selected else None,
        "next_work_reason": selected[3] if selected else None,
        "external_obligations": external,
        "scope": "Work guidance only; external obligations and frozen acceptance remain open",
    }
    if (
        selected
        and interface_work
        and selected[1] == interface_work.get("next_target")
        and interface_work.get("suggested_action")
    ):
        result["suggested_action"] = interface_work["suggested_action"]
    return result
