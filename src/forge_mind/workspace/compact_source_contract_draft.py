"""Prospective compact references, never source interpretation or proof authority."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, Field, create_model
from pydantic_core import PydanticUndefined

from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution import (
    ContractDraft,
    IntentFact,
    IntentIssue,
    IntentSubject,
)
from forge_mind.workspace.models import Contract, PlanName, Record, Requirement
from forge_mind.workspace.verification_models import VerificationCriterion

PROTOCOL = "compact_source_contract_draft_v1"


def _field(model: type[BaseModel], name: str, *, required: bool = False) -> tuple[Any, Any]:
    field = deepcopy(model.model_fields[name])
    if required:
        field.default = PydanticUndefined
        field.default_factory = None
    return field.annotation, field


def _project(
    name: str,
    model: type[BaseModel],
    fields: set[str],
    **extra: Any,
) -> type[BaseModel]:
    definitions: dict[str, Any] = {
        key: _field(model, key) for key in model.model_fields if key in fields
    }
    definitions.update(extra)
    return create_model(name, __base__=Record, **definitions)


def _constants(model: type[BaseModel]) -> dict[str, Any]:
    """Only an explicit family's singleton Literal fields have mechanical values."""
    return {
        name: get_args(field.annotation)[0]
        for name, field in model.model_fields.items()
        if get_origin(field.annotation) is Literal and len(get_args(field.annotation)) == 1
    }


_CRITERIA: dict[str, type[BaseModel]] = {
    _constants(model)["kind"]: model for model in get_args(VerificationCriterion)
}
_WIRE_CRITERIA = {
    kind: _project(
        f"Compact{model.__name__}",
        model,
        set(model.model_fields) - {"source_quote", *_constants(model)},
        id=(PlanName, ...),
        quote_ref=(PlanName, ...),
    )
    for kind, model in _CRITERIA.items()
}
_SCALAR_FIELDS = set(Requirement.model_fields) - {
    "id",
    "text",
    "source_quote",
    "source",
    "critical",
    "verification",
}
_SCALAR_DEFINITIONS: dict[str, Any] = {
    name: _field(Requirement, name, required=True)
    for name in Requirement.model_fields
    if name in _SCALAR_FIELDS
}
CompactScalar = create_model("CompactSourceScalar", __base__=Record, **_SCALAR_DEFINITIONS)
CompactRequirement = _project(
    "CompactSourceRequirement",
    Requirement,
    {"id", "source", "critical"},
    text=(str | None, Field(default=None, min_length=1, max_length=2000)),
    quote_ref=(PlanName, ...),
    subject_ids=(list[PlanName], Field(default_factory=list)),
    verification_id=(PlanName | None, None),
    scalar=(CompactScalar | None, None),
)
CanonicalRequirement = create_model(
    "CompactCanonicalRequirement",
    __base__=Record,
    requirement=(Requirement, ...),
    subject_ids=(list[PlanName], Field(default_factory=list)),
)
SourceQuote = create_model(
    "CompactSourceQuote",
    __base__=Record,
    id=(PlanName, ...),
    text=_field(IntentSubject, "source_quote"),
)
CompactSubject = _project(
    "CompactSourceSubject",
    IntentSubject,
    {"id", "label", "role"},
    quote_ref=(PlanName, ...),
)
CompactFactValue = _project("CompactSourceFactValue", IntentFact, {"id", "property", "value"})
CompactFactGroup = _project(
    "CompactSourceFactGroup",
    IntentFact,
    {"subject_id", "candidate_subject_ids", "unit", "requirement_ids"},
    quote_ref=(PlanName, ...),
    values=(list.__class_getitem__(CompactFactValue), Field(min_length=1)),
)
_DRAFT_DEFINITIONS: dict[str, Any] = {
    "quotes": (list.__class_getitem__(SourceQuote), Field(default_factory=list)),
    "requirements": (list.__class_getitem__(CompactRequirement), Field(default_factory=list)),
    "canonical_requirements": (
        list.__class_getitem__(CanonicalRequirement),
        Field(default_factory=list),
    ),
    "requirement_order": (list[PlanName], Field(default_factory=list)),
    "subjects": (list.__class_getitem__(CompactSubject), Field(default_factory=list)),
    "fact_groups": (list.__class_getitem__(CompactFactGroup), Field(default_factory=list)),
    "fact_order": (list[PlanName], Field(default_factory=list)),
    "issues": (list[IntentIssue], Field(default_factory=list)),
    "coverage_notes": _field(Contract, "coverage_notes"),
    **{
        kind: (list.__class_getitem__(model), Field(default_factory=list))
        for kind, model in _WIRE_CRITERIA.items()
    },
}
CompactSourceContractDraft = create_model(
    "CompactSourceContractDraft", __base__=Record, **_DRAFT_DEFINITIONS
)

COMPACT_SOURCE_DRAFT_INSTRUCTIONS = """Return CompactSourceContractDraft for initial extraction.
Share exact immutable source strings in quotes [{id,text}]. All quote_ref fields
select those exact strings. References do not prove source meaning or completeness.
Retain every requested duty, source fact, subject, genuine question and coverage note.
requirements contains separate rows with authored text, source, critical, subject_ids,
quote_ref, verification_id and scalar. Select exactly one verification_id or scalar,
or null for both to retain an unverified duty. Never merge separate requirements.
Use canonical_requirements [{requirement,subject_ids}] for full canonical fields,
unsupported syntax, assumptions or semantic exceptions. requirement_order lists every
requirement ID once across both branches, preserving their intended order.
Explicitly place criteria in the five named canonical family lists. Each has its
own id and quote_ref; references can share the same authored criterion. Retain all
variable targets, axes, selectors, roles and tolerances. Only singleton constants
from that selected family's canonical schema are supplied mechanically. Inventory
checks names/counts, never body shape. Unsupported duties stay critical/unverified.
For a complete closed digital file-list clause ONLY, a compact digital requirement
may set text=null. Its exact quote becomes canonical text under the existing complete
source-clause guard. Geometry, physical and mixed duties retain normal full text.
Subjects use explicit labels, roles and quote_ref. fact_groups has explicit common
subject_id, candidate_subject_ids, unit, quote_ref and requirement_ids; values lists
every {id,property,value}. Split groups when any common field differs. Keep original
literal values and units. fact_order lists every fact ID exactly once, even when
groups interleave. Do not convert units, infer coordinates or omit repeated facts.
All criterion IDs are unique across families and every criterion is referenced.
Each requirement and its criterion must resolve to the same exact source quote.
Never use notes, scalar facts or file words to cover an extra physical/procedural duty.
Keep authored prose concise, precise and complete. State each requested duty without
extraction rationale or explanations of validators. Use short, specific fact property
labels while retaining every literal fact. Use coverage_notes for source-backed
subject designation/context, genuine ambiguity and genuine omissions. Keep those notes
brief; omit repeated encoded targets, predicate mechanics, lists of unstated constraints
and declarations that no issue exists. Preserve every distinct duty, fact, source flag,
critical flag, qualifier, tolerance, subject link and genuine question. Preserve exact
source quotes and all required fields. Concision never permits scope reduction or
weaker checks. There is no row, fact, issue or note count limit.
Use short, stable opaque local IDs for newly created quote, criterion, requirement
and fact records. IDs identify records; duty text and fact properties retain meaning.
Reuse each ID exactly in references and order lists. Preserve every customer name
and part/subject identifier. Never rename an existing retained or admitted ID.
"""


def _raw(proposal: BaseModel) -> dict[str, Any]:
    if not isinstance(proposal, CompactSourceContractDraft):
        raise ValueError("Initial compact extraction requires CompactSourceContractDraft")
    raw: dict[str, Any] = proposal.model_dump(mode="json")
    return raw


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["id"] in indexed:
            raise ValueError(f"Compact {label} IDs must be unique")
        indexed[row["id"]] = row
    return indexed


def _ordered(rows: dict[str, dict[str, Any]], order: list[str], label: str) -> list[dict[str, Any]]:
    if len(order) != len(set(order)) or set(order) != rows.keys():
        raise ValueError(f"Compact {label} order must cover every ID exactly once")
    return [rows[identifier] for identifier in order]


def lower_compact_source_draft(proposal: BaseModel, source_text: str) -> ContractDraft:
    """Expand explicit references and constants; canonical admission remains separate."""
    raw = _raw(proposal)
    quotes = _index(raw["quotes"], "quote")
    for row in quotes.values():
        if row["text"] not in source_text:
            raise ValueError("Compact quote text must occur exactly in immutable source")

    def quote(reference: str) -> str:
        if reference not in quotes:
            raise ValueError("Compact quote reference is unknown")
        return str(quotes[reference]["text"])

    criteria: dict[str, dict[str, Any]] = {}
    for kind, model in _CRITERIA.items():
        for row in raw[kind]:
            if row["id"] in criteria:
                raise ValueError("Compact criterion IDs must be unique across families")
            criteria[row["id"]] = {
                **{key: value for key, value in row.items() if key not in {"id", "quote_ref"}},
                **_constants(model),
                "source_quote": quote(row["quote_ref"]),
            }

    rows: list[dict[str, Any]] = []
    subject_links: dict[str, list[str]] = {}
    used: set[str] = set()
    for row in raw["requirements"]:
        resolved = quote(row["quote_ref"])
        identifier, scalar = row["verification_id"], row["scalar"]
        if identifier is not None and scalar is not None:
            raise ValueError("Compact requirement cannot combine scalar and verification")
        verification = None
        if identifier is not None:
            if identifier not in criteria:
                raise ValueError("Compact verification reference is unknown")
            verification = criteria[identifier]
            if verification["source_quote"] != resolved:
                raise ValueError("Compact parent and criterion quotes must match exactly")
            used.add(identifier)
        text = row["text"]
        if text is None:
            if verification is None or verification["kind"] != "digital_delivery":
                raise ValueError("Compact null text requires explicit digital delivery")
            text = resolved
        rows.append(
            {
                "id": row["id"],
                "text": text,
                "source_quote": resolved,
                "source": row["source"],
                "critical": row["critical"],
                **(scalar or {}),
                "verification": verification,
            }
        )
        if row["subject_ids"]:
            subject_links[row["id"]] = row["subject_ids"]
    if used != criteria.keys():
        raise ValueError("Every compact criterion must be referenced")
    for row in raw["canonical_requirements"]:
        rows.append(row["requirement"])
        if row["subject_ids"]:
            subject_links[row["requirement"]["id"]] = row["subject_ids"]
    requirements = _ordered(_index(rows, "requirement"), raw["requirement_order"], "requirement")

    subjects = [
        {
            **{key: value for key, value in row.items() if key != "quote_ref"},
            "source_quote": quote(row["quote_ref"]),
        }
        for row in raw["subjects"]
    ]
    subject_ids = set(_index(subjects, "subject"))
    facts = []
    for group in raw["fact_groups"]:
        header = {key: value for key, value in group.items() if key not in {"values", "quote_ref"}}
        for value in group["values"]:
            facts.append({**header, **value, "source_quote": quote(group["quote_ref"])})
    ordered_facts = _ordered(_index(facts, "fact"), raw["fact_order"], "fact")
    requirement_ids = {row["id"] for row in requirements}
    fact_ids = {row["id"] for row in ordered_facts}
    for linked_subjects in subject_links.values():
        if set(linked_subjects) - subject_ids:
            raise ValueError("Compact requirement subject reference is unknown")
    for fact in ordered_facts:
        fact_subjects = set(fact["candidate_subject_ids"])
        if fact["subject_id"] is not None:
            fact_subjects.add(fact["subject_id"])
        if fact_subjects - subject_ids or set(fact["requirement_ids"]) - requirement_ids:
            raise ValueError("Compact fact subject or requirement reference is unknown")
    for issue in raw["issues"]:
        if set(issue["fact_ids"]) - fact_ids or set(issue["requirement_ids"]) - requirement_ids:
            raise ValueError("Compact issue fact or requirement reference is unknown")
    return ContractDraft.model_validate(
        {
            "contract": {"requirements": requirements, "coverage_notes": raw["coverage_notes"]},
            "attribution": {
                "subjects": subjects,
                "facts": ordered_facts,
                "requirement_subjects": subject_links,
                "issues": raw["issues"],
            },
        }
    )


def compact_source_proposal_record(
    proposal: BaseModel, source_text: str, lowered: ContractDraft | None = None
) -> dict[str, Any]:
    """Retain wire choices before expansion, including invalid cross-references."""
    raw = _raw(proposal)
    record: dict[str, Any] = {
        "protocol": PROTOCOL,
        "raw_response": raw,
        "raw_response_hash": digest(raw),
        "source_hash": digest(source_text),
        "source_only_requirement_ids": [
            row["id"] for row in raw["requirements"] if row["text"] is None
        ],
    }
    if lowered is not None:
        canonical = lowered.model_dump(mode="json")
        if lower_compact_source_draft(proposal, source_text).model_dump(mode="json") != canonical:
            raise ValueError("Compact source proposal does not lower to this canonical draft")
        record["lowered_draft_hash"] = digest(canonical)
    return record
