"""Pure reference consistency for customer instructions, never another evidence reader."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Collection
from copy import deepcopy
from typing import Any, Literal, TypeGuard

from pydantic import Field, StrictStr, field_validator

from forge_mind.workspace.guidance_history import verify_guidance_history_context
from forge_mind.workspace.models import Record

Action = Literal[
    "open_project_file",
    "load_print_file",
    "review_part",
    "review_check",
    "review_physical_limits",
    "prepare_print_files",
]


class InstructionStep(Record):
    action: Action
    target: StrictStr = Field(min_length=1)


class InstructionPlan(Record):
    version: Literal[1] = 1
    fact_refs: list[StrictStr] = Field(min_length=1)
    steps: list[InstructionStep] = Field(min_length=1)

    @field_validator("version", mode="before")
    @classmethod
    def strict_version(cls, value: Any) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("Instruction plan version must be integer 1")
        return value


SCOPE = (
    "Bound means supported references match this saved basis, not verified adjacent prose, "
    "frozen requirement satisfaction, manufacturing readiness or physical approval. "
    "Project membership is a rendering overlay; capture and readback remain separate."
)
_ACTIONS = {
    "open_project_file": {"kinds": ["project_file"], "actor": "customer"},
    "load_print_file": {"kinds": ["prepared_file"], "actor": "customer"},
    "review_part": {"kinds": ["part"], "actor": "customer"},
    "review_check": {"kinds": ["check", "checks_summary"], "actor": "reviewer"},
    "review_physical_limits": {"kinds": ["physical_readiness"], "actor": "customer"},
    "prepare_print_files": {"kinds": ["print_preparation"], "actor": "workspace_owner"},
}
FAILED_CHECK_STATES = frozenset(
    {"fail", "failed", "check_failed", "checked_fail", "missing_construction"}
)
_PASS = {"pass", "checked", "nominal_checked"}
_STATES = (
    FAILED_CHECK_STATES
    | _PASS
    | {
        "unverified",
        "unknown",
        "check_pending",
        "pending",
        "stale",
        "unsupported",
        "unbound",
    }
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _quantity(value: Any) -> bool:
    return type(value) is int and value >= 0


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _path(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not any(char in value for char in ("\\", ":", "\x00", "\n", "\r"))
        and all(segment not in {"", ".", ".."} for segment in value.split("/"))
    )


def _identifier(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value) is not None


def _hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _state(row: dict[str, Any]) -> str:
    status = row.get("status")
    return status if isinstance(status, str) and status in _STATES else "unverified"


def instruction_catalog(context: dict[str, Any]) -> dict[str, Any]:
    """Project complete supplied server facts. No IO, verdict repair or prose classification."""
    facts = _mapping(context.get("facts"))
    basis = _mapping(context.get("basis"))
    catalog: dict[str, dict[str, Any]] = {}
    required = {"checks:summary", "physical:readiness"}

    def add(
        ref: str, kind: str, subject: str, value: Any, state: str, scope: str, text: str, field: str
    ) -> None:
        row = {
            "id": ref,
            "kind": kind,
            "subject": subject,
            "value": deepcopy(value),
            "state": state,
            "scope": scope,
            "text": text,
            "source": {
                "revision_id": basis.get("revision_id"),
                "basis_hash": context.get("basis_hash"),
                "field": field,
            },
        }
        if ref in catalog:
            row.update(
                state="unverified",
                text=f"Reference {ref} is ambiguous.",
                value={"conflicting_records": [catalog[ref]["value"], row["value"]]},
            )
        catalog[ref] = row

    inventory = _mapping(facts.get("package_files"))
    geometry = _mapping(inventory.get("geometry"))
    paths = {path for path in geometry if _path(path)}
    metadata = inventory.get("metadata_files")
    if isinstance(metadata, list):
        paths.update(path for path in metadata if _path(path))
    for path in sorted(paths):
        entry = _mapping(geometry.get(path))
        state = (
            "available"
            if entry.get("source_present") is True
            else "unavailable"
            if path in geometry
            else "prospective"
        )
        add(
            f"project:{path}",
            "project_file",
            path,
            {"path": path, **entry},
            state,
            "Prospective project-relative file; original CAD frame. No capture or readback is inferred.",
            f"Project file {path}: {state}; inclusion depends on the selected download.",
            f"facts.package_files/{path}",
        )

    for part in _rows(facts.get("instruction_parts")):
        part_id = part.get("part_id")
        if not _identifier(part_id):
            continue
        role, quantity = part.get("role"), part.get("exported_occurrence_quantity")
        known = (
            isinstance(role, str) and role in {"make", "buy", "reference"} and _quantity(quantity)
        )
        state = (
            "available"
            if known and isinstance(quantity, int) and quantity > 0
            else "missing"
            if known
            else "unverified"
        )
        text = (
            f"Part {part_id}: declared role {role}; exported quantity {quantity}."
            if known
            else f"Part {part_id}: exported quantity or role is unverified."
        )
        material_name = _mapping(part.get("material")).get("name")
        if isinstance(material_name, str) and material_name.strip():
            text += " Material declaration (unverified literal): " + json.dumps(material_name) + "."
        add(
            f"part:{part_id}",
            "part",
            part_id,
            {
                **part,
                "declared_name_scope": "Literal authored label, not a verified claim.",
                "material_scope": "Literal material declaration, not material qualification.",
            },
            state,
            "Exported occurrence quantity and declared production role only; presence does not prove function.",
            text,
            f"facts.instruction_parts/{part_id}",
        )
        if isinstance(role, str) and role in {"make", "buy"} and _quantity(quantity):
            required.add(f"part:{part_id}")

    measured = _mapping(facts.get("measured"))
    for key, units in {
        "extents_mm": "mm",
        "bbox_min": "mm",
        "bbox_max": "mm",
        "volume_mm3": "mm3",
        "solid_count": "count",
        "hole_count": "count",
        "step_valid": "boolean",
    }.items():
        if key not in measured:
            continue
        value = measured[key]
        valid = (
            (
                isinstance(value, list)
                and len(value) == 3
                and all(_finite(number) for number in value)
            )
            if key in {"extents_mm", "bbox_min", "bbox_max"}
            else (
                type(value) is bool
                if key == "step_valid"
                else _quantity(value)
                if units == "count"
                else _finite(value) and value >= 0
            )
        )
        if valid and key == "extents_mm":
            valid = all(number >= 0 for number in value)
        state = (
            "fail"
            if valid and key == "step_valid" and value is False
            else "measured"
            if valid
            else "unverified"
        )
        add(
            f"measurement:{key}",
            "measurement",
            key,
            {"value": value if valid else None, "units": units},
            state,
            "Saved nominal CAD measurement only; no physical qualification.",
            f"Saved CAD {key}: {json.dumps(value, allow_nan=False)} {units}."
            if valid
            else f"Saved CAD {key} is unverified.",
            f"facts.measured/{key}",
        )

    setup = _mapping(facts.get("print_setup"))
    setup_status = setup.get("status") if isinstance(setup.get("status"), str) else "unverified"
    prepared = _mapping(setup.get("files"))
    for path in sorted(path for path in prepared if _path(path) and path.startswith("print/")):
        raw = prepared[path]
        entry = _mapping(raw)
        valid = (
            setup_status == "current"
            and _quantity(entry.get("quantity"))
            and entry["quantity"] > 0
            and (
                entry.get("frame") == "plate_z_up"
                and entry.get("units") == "mm"
                and _hash(entry.get("sha256"))
                and _identifier(entry.get("part_id"))
                and _hash(entry.get("candidate_hash"))
            )
        )
        state = "current" if valid else "stale" if setup_status == "stale" else "unverified"
        add(
            f"prepared:{path}",
            "prepared_file",
            path,
            {"path": path, **entry},
            state,
            "Separate prepared print package in plate_z_up coordinates; no support, printer or physical approval.",
            f"Prepared print file {path}: quantity {entry['quantity']}, plate_z_up frame; print success is unverified."
            if valid
            else f"Prepared print file {path}: {state}.",
            f"facts.print_setup.files/{path}",
        )
        if setup_status == "current":
            required.add(f"prepared:{path}")

    checks: list[dict[str, Any]] = []
    motion = _mapping(facts.get("motion_check"))
    if motion:
        checks.append(
            {
                "ref": "check:motion:current",
                "subject": "current_motion",
                "row": motion,
                "field": "motion_check",
            }
        )
        required.add("check:motion:current")
    for row in _rows(facts.get("requirement_findings")):
        identifier = row.get("requirement_id", row.get("id"))
        if isinstance(identifier, str) and identifier:
            checks.append(
                {
                    "ref": f"check:requirement:{identifier}",
                    "subject": identifier,
                    "row": row,
                    "field": "requirement_findings",
                }
            )
    for interface in _rows(_mapping(facts.get("functional_interfaces")).get("rows")):
        identifier = interface.get("id")
        if not _identifier(identifier):
            continue
        checks.append(
            {
                "ref": f"check:interface:{identifier}",
                "subject": identifier,
                "row": interface,
                "field": "functional_interfaces",
            }
        )
        for duty in _rows(interface.get("duties")):
            duty_id = duty.get("id")
            if _identifier(duty_id):
                checks.append(
                    {
                        "ref": f"check:duty:{identifier}/{duty_id}",
                        "subject": f"{identifier}/{duty_id}",
                        "row": duty,
                        "field": "functional_interfaces",
                    }
                )
    for check in checks:
        state = _state(check["row"])
        add(
            check["ref"],
            "check",
            check["subject"],
            check["row"],
            state,
            "Exact saved global sampled-motion state only; no continuous clearance or physical approval."
            if check["field"] == "motion_check"
            else "Exact saved requirement/interface/duty state only. Raw declarations remain unverified notes; no broader pass is inferred.",
            f"Saved check {check['ref']}: {state}.",
            f"facts.{check['field']}/{check['subject']}",
        )
    states = [_state(check["row"]) for check in checks]
    summary = (
        "fail"
        if any(state in FAILED_CHECK_STATES for state in states)
        else "pass"
        if states and all(state in _PASS for state in states)
        else "unverified"
    )
    add(
        "checks:summary",
        "checks_summary",
        "saved_checks",
        [{"ref": check["ref"], "state": _state(check["row"])} for check in checks],
        summary,
        "Summary of supplied saved check states only, not whole-requirement or physical approval.",
        f"Saved checks: {summary}. Review each check's exact scope; unknown and failed requirements remain open.",
        "facts.requirement_findings+functional_interfaces+motion_check"
        if motion
        else "facts.requirement_findings+functional_interfaces",
    )
    add(
        "physical:readiness",
        "physical_readiness",
        "physical_readiness",
        "unverified",
        "unverified",
        "No supplied nominal CAD, file or check receipt grants physical approval.",
        "Physical fit, printing, strength and safe operation remain unverified.",
        "physical_readiness",
    )
    print_state = (
        "complete"
        if setup_status == "current"
        and prepared
        and all(catalog.get(f"prepared:{path}", {}).get("state") == "current" for path in prepared)
        else "pending"
        if not setup or setup_status in {"unbound", "stale", "missing"}
        else "unverified"
    )
    add(
        "workflow:print_preparation",
        "print_preparation",
        "print_preparation",
        {"setup_status": setup.get("status", "unbound")},
        print_state,
        "Preparation state only; separate prepared files do not prove a successful print.",
        "Separate print files are already prepared for this revision."
        if print_state == "complete"
        else f"Print preparation: {print_state}.",
        "facts.print_setup",
    )
    return {
        "version": 1,
        "facts": catalog,
        "required_refs": sorted(required),
        "actions": deepcopy(_ACTIONS),
    }


def validate_instruction_plan(
    plan: InstructionPlan | dict[str, Any], context: dict[str, Any]
) -> InstructionPlan:
    """Validate authored references only; context freshness is checked by the resolver/caller."""
    value = InstructionPlan.model_validate(
        plan.model_dump() if isinstance(plan, InstructionPlan) else plan
    )
    catalog = instruction_catalog(context)
    refs, known = value.fact_refs, catalog["facts"]
    errors = []
    duplicates = sorted(ref for ref, count in Counter(refs).items() if count > 1)
    if duplicates:
        errors.append("Duplicate fact_refs: " + ", ".join(duplicates))
    unknown = sorted(set(refs) - known.keys())
    if unknown:
        errors.append("Unknown fact_refs: " + ", ".join(unknown))
    missing = sorted(set(catalog["required_refs"]) - set(refs))
    if missing:
        errors.append("Missing required fact_refs: " + ", ".join(missing))
    seen = set()
    for step in value.steps:
        if step.target not in refs:
            errors.append("Undisclosed step target: " + step.target)
        row = known.get(step.target)
        if row is None:
            errors.append("Unknown step target: " + step.target)
        elif row["kind"] not in _ACTIONS[step.action]["kinds"]:
            errors.append(f"Wrong kind for {step.action}: {step.target}")
        key = step.action, step.target
        if key in seen:
            errors.append(f"Duplicate step: {step.action} {step.target}")
        seen.add(key)
    if errors:
        raise ValueError("; ".join(errors))
    return value


def resolve_instruction_plan(
    context: dict[str, Any], *, members: Collection[str] | None = None
) -> dict[str, Any]:
    """Resolve one stored plan and optionally overlay actual project membership, without writes."""
    catalog = instruction_catalog(context)
    facts = deepcopy(catalog["facts"])
    if members is not None:
        for row in facts.values():
            if row["kind"] == "project_file":
                row["state"] = "included" if row["subject"] in members else "omitted"
                row["text"] = (
                    f"Project file {row['subject']}: {row['state']} in this download; no capture or readback is inferred."
                )
    result: dict[str, Any] = {
        "version": 1,
        "status": "legacy_unverified",
        "facts": list(facts.values()),
        "steps": [],
        "errors": [],
        "physical_readiness": "unverified",
        "scope": SCOPE,
    }

    def refused(status: str, message: str) -> dict[str, Any]:
        result.update(status=status, errors=[message])
        return result

    authority = context.get("input_authority")
    if "input_authority" in context and (
        not isinstance(authority, dict)
        or authority.get("status") != "clear"
        or not isinstance(authority.get("identity"), dict)
        or not isinstance(authority.get("reasons"), list)
        or bool(authority["reasons"])
    ):
        return refused(
            "blocked_input",
            "input_authority requires clear status, a bound identity and empty refusal reasons",
        )
    record = _mapping(context.get("record"))
    raw_plan = record.get("instruction_plan")
    if raw_plan is None:
        return result
    history_status = context.get("history_status")
    if "record_provenance" in context:
        try:
            history_status = verify_guidance_history_context(context)
        except (KeyError, TypeError, ValueError) as exc:
            return refused("invalid", str(exc))
    if "history_status" in context and _mapping(history_status).get("status") != "verified":
        refused("invalid", "history_status does not verify the authored record history")
        history_errors = _mapping(history_status).get("errors")
        if isinstance(history_errors, list):
            result["errors"].extend(error for error in history_errors if isinstance(error, str))
        return result
    basis = _mapping(context.get("basis"))
    if (
        context.get("status") != "current"
        or record.get("basis") != basis
        or record.get("basis_hash") != context.get("basis_hash")
    ):
        return refused("stale", "Stored instruction basis differs from the current guidance basis")
    try:
        if not _hash(context.get("basis_hash")) or _digest(basis) != context["basis_hash"]:
            return refused("invalid", "basis_hash does not bind basis")
        if (
            not _hash(basis.get("facts_hash"))
            or _digest(context.get("facts")) != basis["facts_hash"]
        ):
            return refused("invalid", "basis.facts_hash does not bind facts")
        guidance = _mapping(record.get("guidance"))
        if (
            record.get("reviewed_current_facts") is not True
            or set(guidance) != {"print", "installation", "use_and_removal", "unresolved"}
            or not all(isinstance(value, str) and value.strip() for value in guidance.values())
            or not _hash(record.get("guidance_hash"))
            or _digest(guidance) != record["guidance_hash"]
        ):
            return refused(
                "invalid",
                "record.guidance_hash or reviewed_current_facts is invalid; prose remains unverified",
            )
        if authority is not None and (
            record.get("input_authority_hash") != _digest(authority["identity"])
            or basis.get("input_authority_hash") != _digest(authority["identity"])
        ):
            return refused(
                "stale", "record/basis input_authority_hash differs from input_authority.identity"
            )
        if (
            not _hash(record.get("instruction_plan_hash"))
            or _digest(raw_plan) != record["instruction_plan_hash"]
        ):
            return refused("invalid", "record.instruction_plan_hash does not bind instruction_plan")
        plan = validate_instruction_plan(raw_plan, context)
        if plan.model_dump() != raw_plan:
            return refused("invalid", "Stored instruction_plan is not normalized")
    except (ValueError, TypeError, OverflowError) as exc:
        return refused("invalid", str(exc))
    result.update(status="bound", facts=[facts[ref] for ref in plan.fact_refs])
    for step in plan.steps:
        row = facts[step.target]
        state, text = "review", row["text"]
        if step.action == "open_project_file":
            state = (
                "ready" if row["state"] in {"available", "prospective", "included"} else "blocked"
            )
            text = (
                f"Open {row['subject']} from this project download."
                if row["state"] == "included"
                else (
                    f"Use {row['subject']} only if it is included in your selected project download."
                    if state == "ready"
                    else f"Do not use {row['subject']}: it is {row['state']} in this download."
                )
            )
        elif step.action == "load_print_file":
            state = "ready" if row["state"] == "current" else "blocked"
            text = (
                f"Load {row['subject']} from the separate prepared print package; keep its plate_z_up placement and use millimeters without scaling. Printer settings and print success remain unverified."
                if state == "ready"
                else f"Do not load {row['subject']}: its print setup is {row['state']}."
            )
        elif step.action == "review_part":
            state = "review" if row["state"] == "available" else "blocked"
        elif step.action in {"review_check", "review_physical_limits"}:
            state = row["state"]
        elif step.action == "prepare_print_files":
            state = row["state"] if row["state"] in {"complete", "pending"} else "blocked"
            text = (
                "Print files are already prepared; use the current separate prepared package."
                if state == "complete"
                else (
                    "Prepare separate print files for the current revision before loading them."
                    if state == "pending"
                    else "Print preparation is unverified; review its saved diagnostics."
                )
            )
        result["steps"].append(
            {
                "action": step.action,
                "target": step.target,
                "actor": _ACTIONS[step.action]["actor"],
                "state": state,
                "text": text,
            }
        )
    return result
