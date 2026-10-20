"""Admit ordinary nominal interpretations without rewriting source obligations.

The host checks exact source identities, fact links, units and arithmetic. Semantic
association remains a fallible recorded interpretation, never deterministic proof.
Saved-STEP conformance is evaluated later by the existing frozen observers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.customer_interpretation_models import (
    CustomerNominalInterpretation,
    CustomerNominalTarget,
)
from forge_mind.workspace.document import digest
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.verification_models import NominalPerforatedBox

if TYPE_CHECKING:
    from forge_mind.workspace.intent_attribution import ContractDraft, IntentAttribution, IntentFact

# A unit table is dimensional arithmetic, not an English-admission phrase grammar.
_LENGTH_MM = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "millimetre": 1.0,
    "millimetres": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "centimetre": 10.0,
    "centimetres": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "metre": 1000.0,
    "metres": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    '"': 25.4,
}
_COUNT_UNITS = {"hole", "holes", "bore", "bores"}
_SCALES = {"value": 1.0, "negative": -1.0, "half": 0.5, "negative_half": -0.5}


def attribution_identity(attribution: IntentAttribution) -> str:
    return digest(attribution.model_dump(mode="json", exclude={"customer_nominal_interpretation"}))


def _target_requirement(target: CustomerNominalTarget) -> Requirement:
    return Requirement(
        id=target.requirement_id,
        text="Saved CAD conformance to recorded dimensional interpretation",
        source_quote=target.criterion.source_quote,
        source="request",
        critical=False,
        verification=target.criterion,
    )


def _numeric_fields(target: CustomerNominalTarget) -> dict[str, float]:
    criterion = target.criterion
    fields: dict[str, float] = {}
    if isinstance(criterion, NominalPerforatedBox):
        fields.update({f"size_mm.{i}": v for i, v in enumerate(criterion.size_mm)})
        fields.update({f"center_mm.{i}": v for i, v in enumerate(criterion.center_mm)})
    for index, hole in enumerate(criterion.holes):
        fields[f"holes.{index}.diameter_mm"] = hole.diameter_mm
        fields.update({f"holes.{index}.center_mm.{i}": v for i, v in enumerate(hole.center_mm)})
        if hole.diameter_tolerance_mm != 0 or hole.position_tolerance_mm != 0:
            raise ValueError(
                "Ordinary nominal targets use fixed observer error, not authored allowances"
            )
    return fields


def _validate_target(
    target: CustomerNominalTarget,
    originals: dict[str, Requirement],
    attribution: IntentAttribution,
) -> None:
    criterion = target.criterion
    parents = set(target.source_requirement_ids)
    if not parents <= originals.keys():
        raise ValueError("Nominal interpretation references an unknown original source duty")
    for identifier in parents:
        row = originals[identifier]
        if (
            row.source != "request"
            or not row.critical
            or row.metric != "unmeasured"
            or row.verification
        ):
            raise ValueError(
                "Retain original physical duties as critical unverified source requirements"
            )
        if attribution.requirement_subjects.get(identifier) != [criterion.subject_id]:
            raise ValueError(
                "Ordinary nominal targets need one uniquely attributed designed subject"
            )
    if criterion.source_quote not in {originals[key].source_quote for key in parents}:
        raise ValueError("Nominal target quote must retain an exact original source duty quote")
    subjects = {subject.id: subject for subject in attribution.subjects}
    if criterion.subject_id not in subjects or subjects[criterion.subject_id].role != "designed":
        raise ValueError("Nominal interpretation cannot select a reference or unknown subject")
    facts = {fact.id: fact for fact in attribution.facts}
    used: set[str] = set()

    def fact(identifier: str) -> IntentFact:
        if identifier not in facts:
            raise ValueError("Nominal field references an unknown source fact")
        current = facts[identifier]
        if current.subject_id != criterion.subject_id or current.candidate_subject_ids:
            raise ValueError("Nominal source fact has an unresolved or different subject")
        if not parents.intersection(current.requirement_ids):
            raise ValueError("Nominal source fact is not linked to its original source duty")
        if not math.isfinite(current.value):
            raise ValueError("Nominal source facts must be finite")
        used.add(identifier)
        return current

    count = fact(target.hole_count_fact_id)
    if count.unit.casefold() not in _COUNT_UNITS or count.value != len(criterion.holes):
        raise ValueError("Exact complete hole count must equal its source count fact")
    fields = _numeric_fields(target)
    if {binding.field for binding in target.field_bindings} != fields.keys():
        raise ValueError("Every nominal numeric target needs exactly one field binding")
    if target.frame_policy == "centered_body" and (
        not isinstance(criterion, NominalPerforatedBox) or criterion.center_mm != (0.0, 0.0, 0.0)
    ):
        raise ValueError("Centered design coordinates require a centered body predicate")
    for binding in target.field_bindings:
        if binding.basis == "design_origin":
            if target.frame_policy != "centered_body" or binding.field not in {
                "center_mm.0",
                "center_mm.1",
                "center_mm.2",
            }:
                raise ValueError("Only an explicit centered-body origin may use a design zero")
            expected = 0.0
        else:
            operands = []
            for term in binding.terms:
                operand = fact(term.fact_id)
                factor = _LENGTH_MM.get(operand.unit.casefold())
                if factor is None:
                    raise ValueError("Nominal length target needs a supported source length unit")
                operands.append(operand.value * factor * _SCALES[term.operation])
            expected = math.fsum(operands)
        # This guards arithmetic roundoff, never saved geometry or manufacturing.
        if not math.isfinite(expected) or not math.isclose(
            fields[binding.field], expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                "Nominal target differs from its exact source operands and unit conversion"
            )
    for issue in attribution.issues:
        if parents.intersection(issue.requirement_ids) or used.intersection(issue.fact_ids):
            raise ValueError("An unresolved source issue blocks dependent nominal interpretation")


def attach_customer_interpretation(
    draft: ContractDraft, source_text: str, targets: list[CustomerNominalTarget]
) -> ContractDraft:
    """Host-only lowering of explicit proposals; original rows remain byte-for-byte."""
    from forge_mind.workspace.intent_attribution import ContractDraft, validate_attribution

    if not targets:
        return draft
    if draft.attribution.customer_nominal_interpretation is not None:
        raise ValueError("A caller cannot replace a server-created nominal interpretation")
    validate_attribution(draft.attribution, draft.contract, source_text, validate_predicates=False)
    targets = [CustomerNominalTarget.model_validate(target.model_dump()) for target in targets]
    originals = {row.id: row for row in draft.contract.requirements}
    target_ids = [target.requirement_id for target in targets]
    if len(set(target_ids)) != len(target_ids) or originals.keys() & set(target_ids):
        raise ValueError("Nominal subcheck IDs must be new and unique")
    for target in targets:
        _validate_target(target, originals, draft.attribution)
    result = ContractDraft.model_validate(draft.model_dump())
    result.contract = Contract(
        requirements=[*result.contract.requirements, *map(_target_requirement, targets)],
        coverage_notes=result.contract.coverage_notes,
    )
    for target in targets:
        result.attribution.requirement_subjects[target.requirement_id] = [
            target.criterion.subject_id
        ]
    result.attribution.customer_nominal_interpretation = CustomerNominalInterpretation(
        source_sha256=digest(source_text),
        contract_sha256=result.contract.fingerprint(),
        attribution_sha256=attribution_identity(result.attribution),
        original_requirement_ids=list(originals),
        original_requirements_sha256=digest(
            [row.model_dump(mode="json") for row in originals.values()]
        ),
        targets=targets,
    )
    validate_customer_interpretation(result.contract, source_text, result.attribution)
    return result


def validate_customer_interpretation(
    contract: Contract, source_text: str, attribution: IntentAttribution
) -> frozenset[str]:
    """Check the persisted host record against the exact current source and targets."""
    from forge_mind.workspace.intent_attribution import validate_attribution

    marker = attribution.customer_nominal_interpretation
    if marker is None:
        return frozenset()
    marker = CustomerNominalInterpretation.model_validate(marker.model_dump())
    validate_attribution(attribution, contract, source_text, validate_predicates=False)
    if (
        marker.source_sha256 != digest(source_text)
        or marker.contract_sha256 != contract.fingerprint()
    ):
        raise ValueError("Nominal interpretation source or contract identity changed")
    if marker.attribution_sha256 != attribution_identity(attribution):
        raise ValueError("Nominal interpretation attribution identity changed")
    target_ids = [target.requirement_id for target in marker.targets]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Nominal interpretation target IDs must be unique")
    originals = {row.id: row for row in contract.requirements if row.id not in target_ids}
    if list(
        originals
    ) != marker.original_requirement_ids or marker.original_requirements_sha256 != digest(
        [row.model_dump(mode="json") for row in originals.values()]
    ):
        raise ValueError("Nominal interpretation changed or discarded an original source duty")
    rows = {row.id: row for row in contract.requirements}
    for target in marker.targets:
        _validate_target(target, originals, attribution)
        if rows.get(target.requirement_id) != _target_requirement(target):
            raise ValueError("Nominal subcheck differs from its frozen interpreted target")
        if attribution.requirement_subjects.get(target.requirement_id) != [
            target.criterion.subject_id
        ]:
            raise ValueError("Nominal subcheck subject binding changed")
    return frozenset(target_ids)


def customer_interpretation_summary(attribution: IntentAttribution) -> dict[str, Any] | None:
    marker = attribution.customer_nominal_interpretation
    if marker is None:
        return None
    return {
        "policy": marker.policy,
        "source_sha256": marker.source_sha256,
        "contract_sha256": marker.contract_sha256,
        "interpretation_sha256": digest(marker.model_dump(mode="json")),
        "scope": "Recorded dimensional interpretation; saved CAD conformance is separate proof. Physical source duties remain unverified.",
        "targets": [target.model_dump(mode="json") for target in marker.targets],
        "original_requirement_ids": marker.original_requirement_ids,
    }


def validate_server_interpretation_proposal(
    document: dict[str, Any],
    job_id: str,
    source_text: str,
    contract: Contract,
    attribution: IntentAttribution,
) -> None:
    """A client-authored hash cannot mint the persisted server policy marker.

    Only the internal job's retained initial extraction can authorize host lowering.
    All hashes are replayed against that proposal, not accepted as declarations.
    """
    if attribution.customer_nominal_interpretation is None:
        return
    from forge_mind.workspace.source_delivery_contract_draft import (
        lower_source_delivery_draft,
        source_delivery_draft_from_record,
    )

    compilation = document.get("intent_compilation") or {}
    proposal = compilation.get("source_proposal") or {}
    raw = proposal.get("raw_response")
    if (
        compilation.get("job_id") != job_id
        or compilation.get("source_hash") != digest(source_text)
        or compilation.get("status") not in {"ready", "needs_customer_input"}
        or proposal.get("protocol")
        not in {"source_delivery_contract_draft_v1", "photo_source_contract_draft_v2"}
        or proposal.get("source_hash") != digest(source_text)
        or not isinstance(raw, dict)
        or proposal.get("raw_response_hash") != digest(raw)
    ):
        raise ValueError(
            "Nominal policy requires the current internal job's retained source proposal"
        )
    lowered = lower_source_delivery_draft(source_delivery_draft_from_record(proposal), source_text)
    if (
        lowered.contract != contract
        or lowered.attribution != attribution
        or compilation.get("status")
        != ("needs_customer_input" if lowered.attribution.issues else "ready")
        or proposal.get("lowered_draft_hash") != digest(lowered.model_dump(mode="json"))
    ):
        raise ValueError("Nominal policy differs from the server-lowered current source proposal")


def customer_nominal_conformance(
    attribution: IntentAttribution, results: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Display current saved-STEP subproof separately from original duty status."""
    summary = customer_interpretation_summary(attribution)
    if summary is None:
        return None
    marker = attribution.customer_nominal_interpretation
    assert marker is not None
    identifiers = {target.requirement_id for target in marker.targets}
    selected = [row for row in results if row.get("requirement_id") in identifiers]
    states = [row.get("status") for row in selected]
    complete = (
        len(selected) == len(identifiers)
        and {row.get("requirement_id") for row in selected} == identifiers
    )
    return {
        **summary,
        "status": "unverified"
        if not complete
        else "fail"
        if "fail" in states
        else "pass"
        if all(state == "pass" for state in states)
        else "unverified",
        "results": selected,
        "interpretation_status": "recorded_model_interpretation",
        "physical_readiness": "unverified",
    }
