"""Constrained provisional verification repair, never source or proof authority."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, get_args

from pydantic import BaseModel, create_model

from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution import ContractDraft
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.models import Record, Requirement
from forge_mind.workspace.verification_admission import (
    criterion_hash,
    verification_repair_exclusions,
)
from forge_mind.workspace.verification_models import VerificationRepairCriterionV3

PROTOCOL = "verification_repair_v3"


@dataclass(frozen=True)
class PreparedVerificationRepair:
    response_model: type[BaseModel]
    basis: dict[str, Any]
    original_schema: dict[str, Any]

    def request_record(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "route": "verification_only",
            "original_schema": deepcopy(self.original_schema),
            "schema_hash": digest(self.original_schema),
            "basis": deepcopy(self.basis),
        }

    def response_record(self, response: BaseModel) -> dict[str, Any]:
        return {**self.request_record(), "raw_response": response.model_dump(mode="json")}


def prepare_verification_repair(
    draft: ContractDraft, source_text: str, compilation: dict[str, Any]
) -> PreparedVerificationRepair | None:
    """Protect valid peers during the one eligible verification-only correction."""
    diagnostics = compilation.get("diagnostics")
    if compilation.get("status") != "invalid_interpretation" or not diagnostics:
        return None
    requirements = {requirement.id: requirement for requirement in draft.contract.requirements}
    invalid_ids: set[str] = set()
    for error in diagnostics:
        identifier, field = error.get("requirement_id"), error.get("field", "")
        if (
            identifier not in requirements
            or not isinstance(field, str)
            or not field.startswith(f"contract.requirements.{identifier}.verification")
        ):
            return None
        invalid_ids.add(identifier)
    models = get_args(VerificationRepairCriterionV3)
    slots: list[dict[str, Any]] = []
    fields: dict[str, Any] = {}
    for requirement in draft.contract.requirements:
        if requirement.id not in invalid_ids:
            continue
        if (
            requirement.verification is None
            or not isinstance(requirement.verification, models)
            or requirement.source == "assumption"
            or not requirement.source_quote.strip()
            or requirement.source_quote not in source_text
            or requirement.metric != "unmeasured"
            or requirement.value is not None
            or requirement.part_id is not None
        ):
            return None
        certificates = verification_repair_exclusions(requirement, source_text, draft.attribution)
        excluded = sorted({certificate["kind"] for certificate in certificates})
        index = len(slots)
        quote_annotation: Any = Literal.__getitem__(requirement.source_quote)
        identifier_annotation: Any = Literal.__getitem__(requirement.id)
        slot_fields: dict[str, Any] = {
            "requirement_id": (identifier_annotation, ...),
            "unverified_reason": (str | None, ...),
        }
        for model_index, model in enumerate(models):
            kind = model.model_json_schema()["properties"]["kind"]["const"]
            if kind in excluded:
                slot_fields[kind] = (type(None), ...)
                continue
            candidate = create_model(
                f"VerificationRepairCandidate{index}_{model_index}",
                __base__=model,
                source_quote=(quote_annotation, ...),
            )
            slot_fields[kind] = (candidate | None, ...)
        slot_model = create_model(
            f"VerificationRepairSlot{index}",
            __base__=Record,
            **slot_fields,
        )
        slot = f"repair_{index}"
        fields[slot] = (slot_model, ...)
        slots.append(
            {
                "slot": slot,
                "requirement_id": requirement.id,
                "parent_quote_sha256": hashlib.sha256(
                    requirement.source_quote.encode()
                ).hexdigest(),
                "original_criterion_hash": criterion_hash(
                    requirement.verification.model_dump(mode="json")
                ),
                "excluded_kinds": excluded,
                "exclusion_certificates": certificates,
            }
        )
    response_model = create_model("VerificationRepair", __base__=Record, **fields)
    return PreparedVerificationRepair(
        response_model=response_model,
        basis={
            "version": 1,
            "original_draft_hash": digest(draft.model_dump(mode="json")),
            "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            "slots": slots,
        },
        original_schema=response_model.model_json_schema(),
    )


def apply_verification_repair(
    draft: ContractDraft,
    source_text: str,
    prepared: PreparedVerificationRepair,
    response: BaseModel,
) -> ContractDraft:
    """Copy only invalid verification fields after fresh source/basis certification."""
    fresh = prepare_verification_repair(
        draft, source_text, compile_intent_draft(draft, source_text)
    )
    if (
        fresh is None
        or fresh.basis != prepared.basis
        or fresh.original_schema != prepared.original_schema
    ):
        raise ValueError("Verification repair basis or shared exclusion certificate changed")
    validated = fresh.response_model.model_validate(response.model_dump(mode="json"))
    kinds = [
        model.model_json_schema()["properties"]["kind"]["const"]
        for model in get_args(VerificationRepairCriterionV3)
    ]
    replacements = {}
    for slot in fresh.basis["slots"]:
        item = getattr(validated, slot["slot"])
        selected = [getattr(item, kind) for kind in kinds if getattr(item, kind) is not None]
        reason = item.unverified_reason
        if len(selected) == 1 and reason is None:
            replacements[slot["requirement_id"]] = selected[0].model_dump(mode="json")
        elif not selected and isinstance(reason, str) and reason.strip():
            replacements[slot["requirement_id"]] = None
        else:
            raise ValueError(
                "Verification repair requires exactly one family with a null reason, "
                "or no families with a nonblank unverified reason"
            )
    result = draft.model_copy(deep=True)
    result.contract.requirements = [
        Requirement.model_validate(
            {**requirement.model_dump(mode="json"), "verification": replacements[requirement.id]}
        )
        if requirement.id in replacements
        else requirement
        for requirement in result.contract.requirements
    ]
    return result
