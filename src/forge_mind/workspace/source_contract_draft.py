"""Prospective digital source rows; canonical contracts and old drafts stay unchanged.

The finite source-clause check detects uncovered words. It is not a general
language interpreter or evidence that a requested result has been achieved.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field, create_model

from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution import ContractDraft
from forge_mind.workspace.intent_capabilities import validate_preparation_source_scope
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.models import Contract, Record, Requirement
from forge_mind.workspace.verification_admission import validate_verification_contract
from forge_mind.workspace.verification_models import DigitalDelivery

PROTOCOL = "source_contract_draft_v1"


def _field(model: type[BaseModel], name: str, annotation: Any = None) -> tuple[Any, Any]:
    original = model.model_fields[name]
    return original.annotation if annotation is None else annotation, deepcopy(original)


# Reuse every canonical field, constraint and criterion definition. Only the
# prospective digital row omits free prose. Lowering invokes canonical validators.
_digital_fields: dict[str, Any] = {
    name: _field(Requirement, name)
    for name in Requirement.model_fields
    if name not in {"text", "verification"}
}
_digital_fields["verification"] = (
    DigitalDelivery,
    Field(description="Exact requested digital artifact criterion"),
)
SourceDigitalRequirement = create_model(
    "SourceDigitalRequirement", __base__=Record, **_digital_fields
)
_requirements_type = list.__class_getitem__(SourceDigitalRequirement | Requirement)
SourceContract = create_model(
    "SourceContract",
    __base__=Contract,
    requirements=_field(Contract, "requirements", _requirements_type),
)
SourceContractDraft = create_model(
    "SourceContractDraft",
    __base__=ContractDraft,
    contract=_field(ContractDraft, "contract", SourceContract),
)

SOURCE_DRAFT_INSTRUCTIONS = """For this initial extraction return SourceContractDraft.
For complete closed digital file-list clauses, use SourceDigitalRequirement:
omit text and emit the exact source_quote. The host copies only that quote into
canonical text. For mixed geometry/file sentences or other syntax, use the full
canonical Requirement branch with text, retaining every actual source duty.
All non-digital requirements retain normal text and all canonical fields. These
are explicit input branches; the host never silently falls back between them.
Retain every actual duty. A fragment inside a file-list sentence is permitted only
when all other source words belong to independently complete digital clauses or
separate critical unmeasured requirements without verification. Scalar targets,
subject labels, facts, noncritical rows and coverage_notes cannot cover extra
machine, material, physical or procedural words. Unknown source duties stay
critical and unverified. Do not shorten or rewrite the immutable source.
"""


def lower_source_contract_draft(proposal: BaseModel) -> ContractDraft:
    """Copy the supplied digital quote, without selecting kinds or source spans."""
    if not isinstance(proposal, SourceContractDraft):
        raise ValueError("Initial source extraction requires SourceContractDraft")
    value = proposal.model_dump(mode="json")
    for row in value["contract"]["requirements"]:
        if "text" not in row:
            row["text"] = row["source_quote"]
    return ContractDraft.model_validate(value)


def source_proposal_record(
    proposal: BaseModel, lowered: ContractDraft | None = None
) -> dict[str, Any]:
    """Retain original wire fields separately from the existing canonical record."""
    if not isinstance(proposal, SourceContractDraft):
        raise ValueError("Initial source extraction requires SourceContractDraft")
    raw = proposal.model_dump(mode="json")
    record: dict[str, Any] = {
        "protocol": PROTOCOL,
        "raw_response": raw,
        "raw_response_hash": digest(raw),
        "source_only_requirement_ids": [
            row["id"] for row in raw["contract"]["requirements"] if "text" not in row
        ],
    }
    if lowered is not None:
        canonical = lowered.model_dump(mode="json")
        if lower_source_contract_draft(proposal).model_dump(mode="json") != canonical:
            raise ValueError("Source proposal does not lower to this canonical draft")
        record["lowered_draft_hash"] = digest(canonical)
    return record


_REQUEST = (
    r"(?:(?:please\s+)?(?:give\s+me|deliver|provide|export|prepare|include)|"
    r"I\s+(?:need|want|request))"
)
_ARTIFACT = {
    "editable_native": r"editable\s+native\s+CAD(?:\s+(?:file|files|project))?",
    "project_step": r"STEP(?:\s+(?:file|files|export))?",
    "project_stl": r"STL(?:\s+(?:file|files|export))?",
    "bound_instruction_plan": r"bound\s+(?:instruction\s+plan|instructions)",
}
_COVERAGE = r"(?:\s+(?:covering|for)\s+(?:everything\s+made|all\s+made(?:\s+(?:parts|definitions|items))?))?"


def _closed_digital_quote(criterion: DigitalDelivery, quote: str) -> bool:
    """Whole artifact syntax only; legacy keyword admission is insufficient here."""
    try:
        structured = json.loads(quote)
    except (ValueError, TypeError):
        structured = None
    if isinstance(structured, dict) and "source_quote" not in structured:
        try:
            decoded = DigitalDelivery.model_validate({**structured, "source_quote": quote})
        except ValueError:
            return False
        return decoded == criterion
    if criterion.deliverable == "prepared_print_package":
        try:
            validate_preparation_source_scope(quote)
        except ValueError:
            return False
        return True
    artifact = _ARTIFACT[criterion.deliverable]
    return (
        re.fullmatch(
            rf"(?:{_REQUEST}\s+)?(?:(?:an?|the|one)\s+)?{artifact}{_COVERAGE}[.!]?",
            quote.strip(),
            re.I,
        )
        is not None
    )


def _unique_span(source: str, quote: str) -> tuple[int, int] | None:
    if not quote.strip():
        return None
    start = source.find(quote)
    if start < 0 or source.find(quote, start + 1) >= 0:
        return None
    return start, start + len(quote)


def _enclosing_clause(source: str, span: tuple[int, int]) -> tuple[int, int]:
    # Only immutable delimiters determine boundaries, never generated prose.
    boundaries = [0]
    for index, char in enumerate(source):
        decimal = (
            char == "."
            and index > 0
            and index + 1 < len(source)
            and source[index - 1].isdigit()
            and source[index + 1].isdigit()
        )
        if char in ".!?" and not decimal:
            boundaries.append(index + 1)
    boundaries.append(len(source))
    return (
        max(boundary for boundary in boundaries if boundary <= span[0]),
        min(boundary for boundary in boundaries if boundary >= span[1]),
    )


def source_scope_diagnostics(
    draft: ContractDraft,
    source: str,
    *,
    source_only_requirement_ids: Collection[str],
) -> list[dict[str, Any]]:
    """Refuse uncovered digital source clauses; never repair or grant a pass."""
    diagnostics: list[dict[str, Any]] = []
    eligible: list[tuple[int, int]] = []
    for row in draft.contract.requirements:
        if row.source != "request" or not row.critical:
            continue
        span = _unique_span(source, row.source_quote)
        if span is None:
            continue
        if row.metric == "unmeasured" and row.verification is None:
            eligible.append(span)
        elif isinstance(row.verification, DigitalDelivery) and _closed_digital_quote(
            row.verification, row.source_quote
        ):
            try:
                validate_verification_contract(
                    Contract(requirements=[row]),
                    source,
                    draft.attribution.model_copy(update={"customer_nominal_interpretation": None}),
                )
            except ValueError:
                continue
            eligible.append(span)
    for row in draft.contract.requirements:
        if row.id not in source_only_requirement_ids or not isinstance(
            row.verification, DigitalDelivery
        ):
            continue
        reason = ""
        span = _unique_span(source, row.source_quote)
        if row.source != "request" or not row.critical:
            reason = "Source-only digital rows require a critical request-backed duty."
        elif row.text != row.source_quote:
            reason = "Prospective digital text must remain its exact source quote."
        elif span is None:
            reason = "Digital source quote must have exactly one immutable source occurrence."
        elif not _closed_digital_quote(row.verification, row.source_quote):
            reason = "Digital quote must contain only a complete supported artifact clause."
        else:
            low, high = _enclosing_clause(source, span)
            covered = [False] * (high - low)
            for start, end in [span, *eligible]:
                if low <= start and end <= high:
                    covered[start - low : end - low] = [True] * (end - start)
            residual = "".join(
                " " if covered[index] else char for index, char in enumerate(source[low:high])
            ).strip()
            residual = re.sub(rf"^(?:{_REQUEST})\b", "", residual, count=1, flags=re.I)
            residual = re.sub(r"\b(?:and|also|plus)\b", "", residual, flags=re.I)
            if not re.fullmatch(r"[\s,.;:!()?\[\]{}\-/]*", residual):
                reason = (
                    "The enclosing source clause retains uncovered words. Preserve each actual "
                    "extra duty as a critical unmeasured requirement without verification; "
                    "a scalar or keyword-only artifact peer cannot cover it."
                )
        if reason:
            diagnostics.append(
                {
                    "requirement_id": row.id,
                    "field": f"contract.requirements.{row.id}.source_quote",
                    "code": "incomplete_digital_source_scope",
                    "message": reason,
                }
            )
    return diagnostics


def compile_source_contract_draft(
    draft: ContractDraft,
    source: str,
    *,
    source_only_requirement_ids: Collection[str],
    previous: ContractDraft | None = None,
) -> dict[str, Any]:
    """Run the unchanged compiler plus prospective source coverage on every attempt."""
    result = compile_intent_draft(draft, source, previous=previous)
    extra = source_scope_diagnostics(
        draft, source, source_only_requirement_ids=source_only_requirement_ids
    )
    if extra:
        return {
            **result,
            "status": "invalid_interpretation",
            "diagnostics": [*result["diagnostics"], *extra],
        }
    return result
