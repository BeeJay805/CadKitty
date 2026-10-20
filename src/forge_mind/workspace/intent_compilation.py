"""Pure provisional admission with bounded repair diagnostics, never target edits."""

from __future__ import annotations

import re
from typing import Any

from forge_mind.workspace.intent_attribution import (
    AttributionValidationError,
    ContractDraft,
    IntentAttribution,
    IntentIssue,
    _literal_value,
    valid_unit_quote_expansion,
    validate_attribution,
    validate_fact,
)
from forge_mind.workspace.intent_capabilities import (
    preparation_capability,
    validate_preparation_source_scope,
)
from forge_mind.workspace.models import Contract
from forge_mind.workspace.verification_admission import (
    validate_verification_contract,
    verification_diagnostics,
)
from forge_mind.workspace.verification_models import DigitalDelivery


def validate_issue_dependency(
    issue: IntentIssue, attribution: IntentAttribution, contract: Contract
) -> None:
    """Reject invented operation prerequisites; retain ordinary source ambiguity."""
    dependency = issue.input_dependency
    requirements = {requirement.id: requirement for requirement in contract.requirements}
    linked = [requirements[key] for key in issue.requirement_ids if key in requirements]
    facts = {fact.id: fact for fact in attribution.facts}
    ambiguous = any(
        key in facts and facts[key].subject_id is None and facts[key].candidate_subject_ids
        for key in issue.fact_ids
    )
    only_preparation = bool(linked) and all(
        isinstance(requirement.verification, DigitalDelivery)
        and requirement.verification.deliverable == "prepared_print_package"
        for requirement in linked
    )
    compound_preparation = False
    if only_preparation:
        try:
            for requirement in linked:
                validate_preparation_source_scope(requirement.source_quote)
                validate_preparation_source_scope(requirement.text)
        except ValueError:
            # A compound source duty is real uncertainty, never a false prerequisite.
            only_preparation = False
            compound_preparation = True
    if dependency is not None and dependency.kind == "operation_input":
        capability = preparation_capability()
        if compound_preparation:
            return
        if not only_preparation or dependency.operation != capability["operation"]:
            raise ValueError(
                "Operation dependency must reference its requested delivery obligation"
            )
        if dependency.input_name not in capability["required_customer_inputs"]:
            raise ValueError(
                "Unsliced prepared print placement requires saved CAD inputs, not a customer "
                "printer, material or slicer. Preserve explicit machine/physical duties separately."
            )
    elif dependency is not None and dependency.kind == "subject_assignment":
        if not ambiguous:
            raise ValueError("A subject-assignment question needs a linked unresolved source fact")
    if only_preparation and not ambiguous:
        raise ValueError(
            "A prepared print obligation alone supplies no missing customer input. "
            "Link any actual source ambiguity or explicit physical/machine duty separately."
        )


def _diagnostic(field: str, code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"field": field, "code": code, "message": message, **context}


def _retention_diagnostics(
    previous: ContractDraft, corrected: ContractDraft, source_text: str
) -> list[dict[str, Any]]:
    """A correction may supply proof, but cannot discard the first source duties."""
    diagnostics = []
    requirements = {requirement.id: requirement for requirement in corrected.contract.requirements}
    for original in previous.contract.requirements:
        current = requirements.get(original.id)
        field = f"contract.requirements.{original.id}"
        if current is None:
            diagnostics.append(_diagnostic(field, "dropped_requirement", "Retain this source duty"))
            continue
        before = original.model_dump(exclude={"verification", "source_quote"})
        after = current.model_dump(exclude={"verification", "source_quote"})
        if before != after or (
            original.source_quote in source_text and original.source_quote != current.source_quote
        ):
            diagnostics.append(
                _diagnostic(
                    field,
                    "changed_source_duty",
                    "Retain text, critical/source flags, scalar targets and selectors, and every "
                    "previous exact source quote. A correction only supplies missing valid proof.",
                )
            )
        if original.verification is not None:
            try:
                validate_verification_contract(
                    Contract(requirements=[original]), source_text, previous.attribution
                )
            except ValueError:
                pass
            else:
                if original.verification != current.verification:
                    diagnostics.append(
                        _diagnostic(
                            field + ".verification",
                            "changed_valid_predicate",
                            "Retain the valid frozen predicate",
                        )
                    )
    subjects = {item.id: item for item in previous.attribution.subjects}
    updated_subjects = {item.id: item for item in corrected.attribution.subjects}
    for subject in subjects.values():
        updated = updated_subjects.get(subject.id)
        if updated is None or (subject.source_quote in source_text and updated != subject):
            diagnostics.append(
                _diagnostic(
                    f"attribution.subjects.{subject.id}",
                    "changed_source_attribution",
                    "Retain valid attributed subjects; an invalid source quote can be corrected explicitly",
                )
            )
    original_requirement_ids = {item.id for item in previous.contract.requirements}
    updated_facts = {item.id: item for item in corrected.attribution.facts}
    for fact in previous.attribution.facts:
        updated_fact = updated_facts.get(fact.id)
        if updated_fact is None:
            try:
                validate_fact(fact, previous.attribution, previous.contract, source_text)
            except AttributionValidationError:
                # An inadmissible provisional fact is not frozen source authority.
                # Independent requirements, questions and valid facts remain protected.
                continue
        protected: set[str] = set()
        if fact.source_quote in source_text:
            if updated_fact is None or not valid_unit_quote_expansion(
                fact, updated_fact, previous.attribution, previous.contract, source_text
            ):
                protected.add("source_quote")
            if _literal_value(fact.value, fact.source_quote, unit=fact.unit):
                protected.add("value")
            if re.search(
                r"(?<![A-Za-z])" + re.escape(fact.unit) + r"(?![A-Za-z])", fact.source_quote
            ):
                protected.add("unit")
            if {"value", "unit"} <= protected:
                protected.add("property")
        if (
            fact.subject_id is not None
            and fact.subject_id in subjects
            and subjects[fact.subject_id].source_quote in source_text
        ):
            protected.add("subject_id")
        valid_requirement_ids = set(fact.requirement_ids) & original_requirement_ids
        valid_candidates = set(fact.candidate_subject_ids) & subjects.keys()
        if (
            updated_fact is None
            or any(getattr(fact, field) != getattr(updated_fact, field) for field in protected)
            or not valid_requirement_ids <= set(updated_fact.requirement_ids)
            or not valid_candidates <= set(updated_fact.candidate_subject_ids)
        ):
            diagnostics.append(
                _diagnostic(
                    f"attribution.facts.{fact.id}",
                    "changed_source_attribution",
                    "Retain valid quoted values, units, properties and references. Only invalid "
                    "attribution fields can be corrected from the immutable source.",
                )
            )
    after_issues = {issue.id: issue for issue in corrected.attribution.issues}
    for issue in previous.attribution.issues:
        if not set(issue.requirement_ids) <= original_requirement_ids or not set(
            issue.fact_ids
        ) <= {fact.id for fact in previous.attribution.facts}:
            continue
        try:
            validate_issue_dependency(issue, previous.attribution, previous.contract)
        except ValueError:
            continue
        updated_issue = after_issues.get(issue.id)
        if updated_issue is None or updated_issue.model_dump(
            exclude={"input_dependency"}
        ) != issue.model_dump(exclude={"input_dependency"}):
            diagnostics.append(
                _diagnostic(
                    f"attribution.issues.{issue.id}",
                    "removed_customer_question",
                    "Retain genuine source uncertainty until the customer answers it",
                )
            )
    return diagnostics


def compile_intent_draft(
    draft: ContractDraft, source_text: str, *, previous: ContractDraft | None = None
) -> dict[str, Any]:
    """Classify a draft without demoting, changing or admitting any requirement."""
    diagnostics: list[dict[str, Any]] = []
    blockers = []
    for issue in draft.attribution.issues:
        try:
            validate_issue_dependency(issue, draft.attribution, draft.contract)
        except ValueError as exc:
            diagnostics.append(
                _diagnostic(
                    f"attribution.issues.{issue.id}.input_dependency",
                    "invalid_dependency",
                    str(exc),
                    issue_id=issue.id,
                )
            )
        else:
            blockers.append(issue.id)
    # Check provenance first, then collect predicate diagnostics separately below.
    try:
        validate_attribution(
            draft.attribution, draft.contract, source_text, validate_predicates=False
        )
    except AttributionValidationError as exc:
        diagnostics.append(_diagnostic(exc.field, "invalid_attribution", str(exc)))
    except ValueError as exc:
        diagnostics.append(_diagnostic("attribution", "invalid_attribution", str(exc)))
    # Real source questions stay blockers; diagnostics never answer them. The shared
    # admission rules collect independent errors and full-contract conflicts at once.
    diagnostics.extend(
        verification_diagnostics(
            draft.contract,
            source_text,
            draft.attribution
            if draft.attribution.customer_nominal_interpretation is not None
            else draft.attribution.model_copy(update={"issues": []}),
        )
    )
    if previous is not None:
        diagnostics.extend(_retention_diagnostics(previous, draft, source_text))
    return {
        "status": "invalid_interpretation"
        if diagnostics
        else "needs_customer_input"
        if blockers
        else "ready",
        "diagnostics": diagnostics,
        "blocking_issue_ids": blockers,
    }
