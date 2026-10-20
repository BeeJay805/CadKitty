"""Source-bound intent interpretations, never owner confirmation or measured proof."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Annotated, Any, Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator
from pydantic.json_schema import SkipJsonSchema

from forge_mind.workspace.customer_interpretation_models import CustomerNominalInterpretation
from forge_mind.workspace.models import Contract, PlanName, Record
from forge_mind.workspace.source_numbers import source_word_values

Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
RequirementId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class IntentSubject(Record):
    id: PlanName
    label: Text = Field(
        description=(
            "Source-backed subject designation or product-category context, not a claim of "
            "achieved physical operation. Retain the original designation through this label "
            "and source_quote. Explicit requested properties, performance, interfaces, appearance "
            "and procedures still require their own contract obligations."
        )
    )
    role: Literal["designed", "reference", "environment"]
    source_quote: Text


class IntentFact(Record):
    id: PlanName
    subject_id: PlanName | None = None
    candidate_subject_ids: list[PlanName] = Field(default_factory=list)
    property: Text
    value: float = Field(strict=True)
    unit: Text
    source_quote: Text
    requirement_ids: list[RequirementId] = Field(default_factory=list)


class IntentInputDependency(Record):
    """A proposed missing input, never an answer or permission to change scope."""

    kind: Literal["source_fact", "subject_assignment", "operation_input"]
    input_name: Text
    operation: Literal["prepared_print_package"] | None = None

    @model_validator(mode="after")
    def operation_reference(self) -> IntentInputDependency:
        if (self.kind == "operation_input") != (self.operation is not None):
            raise ValueError("Only an operation input requires its operation reference")
        return self


class IntentIssue(Record):
    id: PlanName
    question: Text
    reason: Text
    fact_ids: list[PlanName] = Field(default_factory=list)
    requirement_ids: list[RequirementId] = Field(default_factory=list)
    input_dependency: IntentInputDependency | None = None

    @model_serializer(mode="wrap")
    def legacy_dependency_absence(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result: dict[str, Any] = handler(self)
        if self.input_dependency is None:
            result.pop("input_dependency", None)
        return result


class IntentAttribution(Record):
    subjects: list[IntentSubject] = Field(default_factory=list)
    facts: list[IntentFact] = Field(default_factory=list)
    requirement_subjects: dict[RequirementId, list[PlanName]] = Field(default_factory=dict)
    issues: list[IntentIssue] = Field(default_factory=list)
    # Persistence only. The service recomputes this from retained raw proposals.
    customer_nominal_interpretation: SkipJsonSchema[CustomerNominalInterpretation | None] = None

    @model_serializer(mode="wrap")
    def legacy_interpretation_absence(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        result: dict[str, Any] = handler(self)
        if self.customer_nominal_interpretation is None:
            result.pop("customer_nominal_interpretation", None)
        return result


class IntentConflict(Record):
    issues: list[IntentIssue] = Field(min_length=1)


class ContractDraft(Record):
    contract: Contract = Field(
        description=(
            "Proposed acceptance obligations for actually requested properties, behavior, "
            "interfaces, appearance, procedures and files. Keep subject designation/context "
            "in attribution and contract.coverage_notes without inventing an unspecified "
            "functional result. coverage_notes cannot hide a real unsupported duty or ambiguity. "
            "Nominal-only language does not waive an explicit physical or procedural conjunct."
        )
    )
    attribution: IntentAttribution


class IntentAnswer(Record):
    issue_id: PlanName
    answer: Text


class IntentResolution(Record):
    expected_version: int = Field(ge=1, strict=True)
    draft_hash: Sha256
    answers: list[IntentAnswer] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_answers(self) -> IntentResolution:
        _unique((answer.issue_id for answer in self.answers), "answer issue IDs")
        return self


_NUMBER = re.compile(r"(?<![\w.])[+\-\u2212]?(?:\d+(?:\.\d+)?|\.\d+)(?![\d.]|[eE][+\-]?\d)")


def _unique(values: Iterable[str], label: str) -> set[str]:
    items = list(values)
    if len(items) != len(set(items)):
        raise ValueError(f"Intent {label} must be unique")
    return set(items)


def _references(values: list[str], known: set[str], label: str) -> None:
    identifiers = _unique(values, label)
    if identifiers - known:
        raise ValueError(f"Intent {label} contain unknown references")


def _literal_value(value: float, quote: str, *, unit: str) -> bool:
    numbers = [float(match.group().replace("\u2212", "-")) for match in _NUMBER.finditer(quote)]
    numbers.extend(source_word_values(quote, unit=unit))
    return value in numbers


class AttributionValidationError(ValueError):
    """Retain the exact invalid field without changing validation authority."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field


def validate_fact(
    fact: IntentFact, attribution: IntentAttribution, contract: Contract, source_text: str
) -> None:
    """Apply the same literal provenance and reference checks to one provisional fact."""
    subjects = {subject.id for subject in attribution.subjects}
    requirements = {requirement.id for requirement in contract.requirements}

    def reject(field: str, message: str) -> None:
        raise AttributionValidationError(f"attribution.facts.{fact.id}.{field}", message)

    def references(values: list[str], known: set[str], label: str, field: str) -> None:
        try:
            _references(values, known, label)
        except ValueError as exc:
            reject(field, str(exc))

    if fact.source_quote not in source_text:
        reject("source_quote", "Intent fact quote must occur exactly in source text")
    if not _literal_value(fact.value, fact.source_quote, unit=fact.unit):
        reject("value", "Intent fact value must occur literally in its quote")
    if not re.search(r"(?<![A-Za-z])" + re.escape(fact.unit) + r"(?![A-Za-z])", fact.source_quote):
        reject("unit", "Intent fact unit must occur literally in its quote")
    references(fact.requirement_ids, requirements, "fact requirement IDs", "requirement_ids")
    references(
        fact.candidate_subject_ids, subjects, "candidate subject IDs", "candidate_subject_ids"
    )
    if fact.subject_id is not None:
        references([fact.subject_id], subjects, "fact subject ID", "subject_id")
        if fact.candidate_subject_ids:
            reject("subject_id", "Intent fact cannot have a subject and unresolved candidates")
    elif not fact.candidate_subject_ids or not any(
        fact.id in issue.fact_ids for issue in attribution.issues
    ):
        reject("subject_id", "Intent unresolved fact needs candidate subjects and a linked issue")


def valid_unit_quote_expansion(
    original: IntentFact,
    corrected: IntentFact,
    attribution: IntentAttribution,
    contract: Contract,
    source_text: str,
) -> bool:
    """Complete missing unit provenance without moving an established numeric anchor.

    Literal membership does not prove linguistic applicability. A missing value
    cannot gain authority by collecting a different number from broader context.
    """
    if (
        original.source_quote == corrected.source_quote
        or original.source_quote not in corrected.source_quote
        or original.model_dump(exclude={"source_quote"})
        != corrected.model_dump(exclude={"source_quote"})
    ):
        return False
    try:
        validate_fact(original, attribution, contract, source_text)
    except AttributionValidationError as exc:
        # This check follows exact source and literal value validation.
        if exc.field != f"attribution.facts.{original.id}.unit":
            return False
    else:
        return False
    try:
        validate_fact(corrected, attribution, contract, source_text)
    except AttributionValidationError:
        return False
    return True


def validate_attribution(
    attribution: IntentAttribution,
    contract: Contract,
    source_text: str,
    *,
    validate_predicates: bool = True,
) -> None:
    """Check provenance and references without proving linguistic applicability.

    An issue is valid retained uncertainty. Callers must handle it before dependent
    admission; successful validation never means that the issue was resolved.
    """
    attribution = IntentAttribution.model_validate(attribution.model_dump())
    contract = Contract.model_validate(contract.model_dump())
    if not isinstance(source_text, str):
        raise ValueError("Intent source text must be a string")
    subjects = _unique((subject.id for subject in attribution.subjects), "subject IDs")
    facts = _unique((fact.id for fact in attribution.facts), "fact IDs")
    _unique((issue.id for issue in attribution.issues), "issue IDs")
    requirements = {requirement.id for requirement in contract.requirements}
    for subject in attribution.subjects:
        if subject.source_quote not in source_text:
            raise ValueError("Intent subject quote must occur exactly in source text")
    for requirement_id, subject_ids in attribution.requirement_subjects.items():
        if requirement_id not in requirements:
            raise ValueError("Intent requirement subjects contain an unknown requirement")
        if not subject_ids:
            raise ValueError("Intent requirement subjects need at least one subject")
        _references(subject_ids, subjects, "requirement subject IDs")
    for issue in attribution.issues:
        if not issue.fact_ids and not issue.requirement_ids:
            raise ValueError("Intent issue must reference a fact or requirement")
        _references(issue.fact_ids, facts, "issue fact IDs")
        _references(issue.requirement_ids, requirements, "issue requirement IDs")
        from forge_mind.workspace.intent_compilation import validate_issue_dependency

        validate_issue_dependency(issue, attribution, contract)
    for fact in attribution.facts:
        validate_fact(fact, attribution, contract, source_text)
    if validate_predicates:
        from forge_mind.workspace.verification_admission import validate_verification_contract

        validate_verification_contract(contract, source_text, attribution)
