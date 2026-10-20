"""Explicit coverage of unmeasured intent, never proof of whole functional prose."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from forge_mind.workspace.completion_models import FunctionalInterface, RequirementDisposition

if TYPE_CHECKING:
    from forge_mind.workspace.intent_attribution import IntentAttribution

_EXTERNAL_FIELDS = ("missing_inputs", "unsupported_checks", "physical_validation")


def _mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        raise ValueError("Functional coverage requires an object")
    return value


def _external_notes(value: Any) -> dict[str, list[str]]:
    return {
        field: [text for text in getattr(value, field) if text.strip()]
        for field in _EXTERNAL_FIELDS
    }


def _typed_mappings(
    plan: dict[str, Any],
    contract: dict[str, Any],
    source_text: str | None,
    attribution: IntentAttribution | None = None,
) -> list[dict[str, Any]]:
    """Classify admitted closed predicates without assigning a verification verdict."""
    if not any(
        _mapping(r).get("verification") is not None for r in contract.get("requirements", [])
    ):
        return []
    from forge_mind.workspace.models import Contract, DesignPlan
    from forge_mind.workspace.verification_admission import (
        validate_verification_bindings,
        validate_verification_contract,
    )
    from forge_mind.workspace.verification_models import (
        DigitalDelivery,
        NominalDeliveryScope,
        NominalPattern,
        NominalPhotoSolid,
        NominalSubjectInventory,
    )

    try:
        frozen = Contract.model_validate(contract)
        design = DesignPlan.model_validate(plan)
        validate_verification_contract(
            frozen,
            source_text
            if source_text is not None
            else "\n".join(r.source_quote for r in frozen.requirements),
            attribution,
            allow_explicit=True,
        )
        validate_verification_bindings(design, frozen)
    except (ValueError, TypeError):
        # Malformed or unsupported criteria keep the existing strict mapping duty.
        return []
    bindings = {binding.subject_id: binding for binding in design.verification_bindings}
    rows: list[dict[str, Any]] = []
    for requirement in frozen.requirements:
        criterion = requirement.verification
        if isinstance(criterion, (NominalPattern, NominalPhotoSolid)):
            binding = bindings.get(criterion.subject_id)
            rows.append(
                {
                    "requirement_id": requirement.id,
                    "kind": criterion.kind,
                    "status": "unverified",
                    "mapping_status": "subject_binding_declared"
                    if binding
                    else "missing_subject_binding",
                    "subject_id": criterion.subject_id,
                    "binding": binding.model_dump(mode="json") if binding else None,
                    "scope": "Explicit subject selection only. The saved made definition and frozen geometry still require trusted current proof.",
                }
            )
        elif isinstance(criterion, DigitalDelivery):
            rows.append(
                {
                    "requirement_id": requirement.id,
                    "kind": criterion.kind,
                    "status": "unverified",
                    "mapping_status": "delivery_obligation",
                    "deliverable": criterion.deliverable,
                    "coverage": criterion.coverage,
                    "scope": "Frozen all-made file obligation only. Actual final capture and exact readback are not established by this mapping.",
                }
            )
        elif isinstance(criterion, NominalSubjectInventory):
            missing = [
                item.subject_id for item in criterion.items if item.subject_id not in bindings
            ]
            rows.append(
                {
                    "requirement_id": requirement.id,
                    "kind": criterion.kind,
                    "status": "unverified",
                    "mapping_status": "missing_subject_binding"
                    if missing
                    else "subject_bindings_declared",
                    "coverage": criterion.coverage,
                    "subject_ids": [item.subject_id for item in criterion.items],
                    "missing_subject_ids": missing,
                    "bindings": [
                        bindings[item.subject_id].model_dump(mode="json")
                        for item in criterion.items
                        if item.subject_id in bindings
                    ],
                    "scope": "Declared identity and occurrence obligations only. Exact saved names, all-made inventory and final delivered quantities remain unverified.",
                }
            )
        elif isinstance(criterion, NominalDeliveryScope):
            rows.append(
                {
                    "requirement_id": requirement.id,
                    "kind": criterion.kind,
                    "status": "unverified",
                    "mapping_status": "policy_obligation",
                    "policy": criterion.policy,
                    "scope": "Frozen software boundary obligation only. Actual final digital delivery protocols and physical qualification boundary remain unverified.",
                }
            )
    return rows


def functional_coverage(
    plan: Any,
    contract: Any,
    *,
    source_text: str | None = None,
    attribution: IntentAttribution | None = None,
) -> dict[str, Any]:
    """Project mapping duties; callers apply the rule to fresh native completion."""
    plan = _mapping(plan)
    contract = _mapping(contract)
    typed = _typed_mappings(plan, contract, source_text, attribution)
    typed_ids = {row["requirement_id"] for row in typed}
    requirement_ids = list(
        dict.fromkeys(
            requirement["id"]
            for value in contract.get("requirements", [])
            if (requirement := _mapping(value)).get("metric", "unmeasured") == "unmeasured"
            and requirement["id"] not in typed_ids
        )
    )
    result: dict[str, Any] = {
        "status": "not_applicable",
        "missing_requirement_ids": [],
        "interface_requirement_ids": [],
        "external_requirements": [],
        "scope": "Explicit intent mapping only; mappings and scoped checks do not prove whole unmeasured requirements",
    }
    missing_bindings = [
        row["requirement_id"] for row in typed if row["mapping_status"] == "missing_subject_binding"
    ]
    if typed:
        result.update(status="unverified", typed_requirements=typed)
    if not requirement_ids:
        if missing_bindings:
            result.update(
                status="needs_review",
                next_target="verification_bindings",
                next_work_category="review",
                next_work_reason="Declare exact subject-to-made-definition bindings for frozen requirements: "
                + ", ".join(missing_bindings),
            )
        return result
    needed = set(requirement_ids)
    mapped_interfaces: set[str] = set()
    mapped_external: set[str] = set()
    notes: dict[str, dict[str, list[str]]] = {}

    def add_notes(requirement_id: str, additions: dict[str, list[str]]) -> None:
        if not any(additions.values()):
            return
        current = notes.setdefault(requirement_id, {field: [] for field in _EXTERNAL_FIELDS})
        for field in _EXTERNAL_FIELDS:
            current[field].extend(additions[field])

    for value in plan.get("requirement_dispositions", []):
        disposition = RequirementDisposition.model_validate(_mapping(value))
        if disposition.requirement_id not in needed:
            continue
        additions = _external_notes(disposition)
        if any(additions.values()):
            mapped_external.add(disposition.requirement_id)
            add_notes(disposition.requirement_id, additions)
    for value in plan.get("functional_interfaces", []):
        interface = FunctionalInterface.model_validate(_mapping(value))
        referenced = needed.intersection(interface.requirement_ids)
        mapped_interfaces.update(referenced)
        for duty in interface.duties:
            additions = _external_notes(duty)
            for requirement_id in referenced:
                add_notes(requirement_id, additions)
    missing = [
        requirement_id
        for requirement_id in requirement_ids
        if requirement_id not in mapped_interfaces | mapped_external
    ]
    result.update(
        status="needs_review" if missing else "unverified",
        missing_requirement_ids=missing,
        interface_requirement_ids=[
            requirement_id
            for requirement_id in requirement_ids
            if requirement_id in mapped_interfaces
        ],
        external_requirements=[
            {"requirement_id": requirement_id, **notes[requirement_id]}
            for requirement_id in requirement_ids
            if requirement_id in notes
        ],
    )
    if missing:
        result.update(
            next_target="functional_coverage",
            next_work_category="review",
            next_work_reason=(
                "Map each unmeasured requirement to a functional interface or meaningful external "
                f"disposition: {', '.join(missing)}"
            ),
        )
    elif missing_bindings:
        result.update(
            status="needs_review",
            next_target="verification_bindings",
            next_work_category="review",
            next_work_reason="Declare exact subject-to-made-definition bindings for frozen requirements: "
            + ", ".join(missing_bindings),
        )
    return result
