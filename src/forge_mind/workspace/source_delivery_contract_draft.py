"""Prospective source-backed digital duties; legacy wire and authority stay intact."""

from __future__ import annotations

from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, create_model

from forge_mind.workspace.compact_source_contract_draft import (
    CanonicalRequirement,
    CompactRequirement,
    CompactSourceContractDraft,
    lower_compact_source_draft,
)
from forge_mind.workspace.customer_interpretation_models import CustomerNominalTarget
from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution import ContractDraft
from forge_mind.workspace.models import Record, Requirement
from forge_mind.workspace.photo_nominal import PhotoNominalClaim, PhotoReferenceDisposition
from forge_mind.workspace.photo_source_contract_v2 import (
    PROTOCOL as PHOTO_PROTOCOL_V2,
)
from forge_mind.workspace.photo_source_contract_v2 import (
    SourcePhotoRequirement,
)
from forge_mind.workspace.verification_models import (
    DigitalDelivery,
    NominalDeliveryScope,
    NominalPerforatedBox,
    NominalSubjectInventory,
    NominalThroughHolePattern,
)

PROTOCOL = "source_delivery_contract_draft_v1"
_NONDIGITAL_POOLS = (
    "nominal_through_hole_pattern",
    "nominal_perforated_box",
    "nominal_subject_inventory",
    "nominal_delivery_scope",
)
_NondigitalCriterion = (
    NominalThroughHolePattern
    | NominalPerforatedBox
    | NominalSubjectInventory
    | NominalDeliveryScope
)


def _field(model: type[BaseModel], name: str, annotation: Any = None) -> tuple[Any, Any]:
    field = deepcopy(model.model_fields[name])
    return field.annotation if annotation is None else annotation, field


_ordinary_fields: dict[str, Any] = {
    name: _field(CompactRequirement, name) for name in CompactRequirement.model_fields
}
_ordinary_fields["text"] = _field(Requirement, "text")
_OrdinaryRequirement = create_model(
    "SourceDeliveryOrdinaryRequirement", __base__=Record, **_ordinary_fields
)
_canonical_fields: dict[str, Any] = {
    name: _field(Requirement, name) for name in Requirement.model_fields
}
_canonical_fields["verification"] = _field(Requirement, "verification", _NondigitalCriterion | None)
_NondigitalRequirement = create_model(
    "SourceDeliveryNondigitalRequirement", __base__=Record, **_canonical_fields
)
_canonical_row_fields: dict[str, Any] = {
    name: _field(CanonicalRequirement, name) for name in CanonicalRequirement.model_fields
}
_canonical_row_fields["requirement"] = _field(
    CanonicalRequirement, "requirement", _NondigitalRequirement
)
_CanonicalRow = create_model("SourceDeliveryCanonicalRow", __base__=Record, **_canonical_row_fields)
_digital_fields: dict[str, Any] = {
    name: _field(CompactRequirement, name)
    for name in ("id", "source", "critical", "quote_ref", "subject_ids")
}
_digital_fields["deliverable"] = _field(DigitalDelivery, "deliverable")
_DigitalRequirement = create_model(
    "SourceDeliveryDigitalRequirement", __base__=Record, **_digital_fields
)
_draft_fields: dict[str, Any] = {
    name: _field(CompactSourceContractDraft, name)
    for name in CompactSourceContractDraft.model_fields
    if name not in {"digital_delivery", "nominal_photo_solid"}
}
_draft_fields["requirements"] = _field(
    CompactSourceContractDraft, "requirements", list.__class_getitem__(_OrdinaryRequirement)
)
_draft_fields["canonical_requirements"] = _field(
    CompactSourceContractDraft, "canonical_requirements", list.__class_getitem__(_CanonicalRow)
)
_draft_fields["digital_requirements"] = (
    list.__class_getitem__(_DigitalRequirement),
    Field(default_factory=list),
)
SourceDeliveryContractDraft = create_model(
    "SourceDeliveryContractDraft",
    __base__=Record,
    nominal_targets=(list[CustomerNominalTarget], Field(default_factory=list)),
    photo_nominal_claims=(list[PhotoNominalClaim], Field(default_factory=list)),
    photo_reference=(PhotoReferenceDisposition | None, None),
    **_draft_fields,
)
PhotoSourceDeliveryContractDraft = create_model(
    "PhotoSourceDeliveryContractDraft",
    __base__=SourceDeliveryContractDraft,
    photo_nominal_claims=(list[PhotoNominalClaim], ...),
    photo_reference=(PhotoReferenceDisposition, ...),
)
PhotoSourceContractDraftV2 = create_model(
    "PhotoSourceContractDraftV2",
    __base__=Record,
    nominal_targets=(list[CustomerNominalTarget], Field(default_factory=list)),
    photo_requirements=(list[SourcePhotoRequirement], ...),
    photo_reference=(PhotoReferenceDisposition, ...),
    **_draft_fields,
)

PHOTO_SOURCE_DRAFT_INSTRUCTIONS_V2 = """Return PhotoSourceContractDraftV2 for this fresh photo intake.
Retain every actual source duty, exact quote, ID, critical flag, subject, fact and genuine issue.
Use photo_requirements for supported photo CAD duties. Each row has id, source=request,
critical=true, quote_ref, subject_ids=[one designed subject], facets, size_fact_ids=[plane_u,
plane_v,thickness], optional hole_count_fact_id, frame_policy=image_plane_xy. There is NO text,
scalar or separate photo criterion pool. The host creates canonical display text from these fields.
facets are visible_profile, rectangular_outline, extent_u, extent_v, thickness, opening_count,
straight_through. One original compound ID may list several facets. Do not split or merge retained IDs.
Each photo quote must retain its complete enclosing owner clause and have one unique occurrence.
Every recognized facet in that clause must have a supported same-quote, same-subject photo row.
All such rows share exact source size/count facts and the same declared image-plane frame.
Keep unknown modifiers, physical duties, negation and unsupported conjuncts critical/unverified in
ordinary requirements or canonical_requirements. An unsupported label cannot excuse truncating a
photo clause. If the complete clause is unsupported, retain it as an ordinary unverified duty.
For copying requests EVERY original owner clause must have a retained critical duty. Retain an
unsupported complete clause verbatim as both text and source quote, critical and unmeasured.
Facts, labels, notes, noncritical rows and generic residual labels cannot cover an omitted clause.
Never invent owner hole sizes, centers or image-derived facts. The immutable target supplies those.
Literal dimensions/counts stay in fact_groups, with exact quotes, units and original requirement links.
Missing count may remain derived image inventory when the owner states no count; explicit zero stays zero.
photo_reference={role:copy|context|ambiguous,source_quote:exact source clause}. Context requires a
positive contextual-only instruction. Genuine ambiguity needs a linked source_fact/reference_role issue.
Use ordinary requirements/canonical_requirements for other duties and existing digital_requirements
for closed file clauses. Existing nominal_targets remain available for ordinary/context CAD subchecks;
they cannot overlap photo IDs or replace source obligations. Do not emit photo_nominal_claims.
requirement_order lists every ID exactly once across all pools. quotes, subjects, fact_groups,
fact_order, issues and coverage_notes retain the existing compact source shapes. Never hide duties
in notes. Correction retains IDs, source quotes, valid facts, flags and valid admitted bindings.
"""

SOURCE_DELIVERY_DRAFT_INSTRUCTIONS = """Return SourceDeliveryContractDraft for initial extraction.
Use compact JSON syntax without insignificant whitespace outside strings; preserve every string, escape and value exactly.
Retain every actual source duty, fact, subject, question, flag and qualifier.
Use quotes [{id,text}] for exact immutable source strings; quote_ref selects one.
For typed digital file duties use digital_requirements ONLY. Each row has id,
source, critical, quote_ref, subject_ids and an explicit canonical deliverable.
There is no digital text, scalar payload or separate digital criterion reference.
The host copies that exact quote into canonical text and both source quotes.
Each digital row remains subject to the complete enclosing source-clause guard.
An extra machine, material, physical, geometry or procedural duty cannot disappear
into a file clause. Keep unsupported duties critical and unverified or questioned.
Ordinary requirements retain full authored text, source, critical, subject_ids,
quote_ref, verification_id and scalar. Select a nondigital criterion or scalar,
or null for both for an unverified duty. All four nondigital families remain available.
canonical_requirements preserves full canonical fields for nondigital/scalar/mixed/
assumption/unsupported duties. Digital verification is unavailable in that escape.
Never merge separate requirements. requirement_order lists every ID exactly once
across ordinary, digital and canonical rows. Source-only digital IDs stay fixed
through correction. Legacy records and admitted duties are never reinterpreted.
Subjects retain explicit labels, roles and quote_ref. fact_groups shares explicit
subject_id, candidate_subject_ids, unit, quote_ref and requirement_ids; values lists
every {id,property,value}. Split groups when any common field differs. fact_order
lists every fact ID exactly once, including interleaved groups and repeated facts.
Keep literal values and units; never infer coordinates, change allowances or omit facts.
Keep authored prose concise, precise and complete. Omit extraction rationale and
validator explanations. Coverage notes retain source-backed designation/context,
genuine ambiguity and genuine omissions, without repeating already encoded targets.
Never hide source duties in notes, labels, facts or noncritical rows. Exact source
quotes, every semantic field, critical flag, target, tolerance, link and question remain.
There is no row, fact, issue or note count limit. Short stable opaque IDs may name
new quote/criterion/requirement/fact records. Preserve customer names, all part/subject
identifiers and every retained/admitted ID. Reuse IDs exactly in links and order lists.
"""

SOURCE_DELIVERY_DRAFT_INSTRUCTIONS += """
For a customer reference classify photo_reference={role:copy|context|ambiguous,
source_quote:exact source clause}. A copying clause cannot be context. A photo of
an object to hold/mate can be context; do not force image-copy fidelity onto it.
Retain genuine role ambiguity as an issue. Never use role=context to skip an
actual source copying instruction.
For role=ambiguous, retain a genuine original-duty IntentIssue with
input_dependency={kind:source_fact,input_name:reference_role}. The existing owner
clarification path handles that question before construction. Do not invent an
answer or turn unresolved role into a successful skipped measurement stage.
For a customer photo, photo_nominal_claims MUST disposition EVERY original
requirement ID exactly once. Supported rows use requirement_id, subject_id,
claims, size_fact_ids=[plane_u,plane_v,thickness fact IDs], hole_count_fact_id,
frame_policy=image_plane_xy. Claims are visible_profile, rectangular_outline,
extent_u, extent_v, thickness, opening_count, straight_through. These record a
conditional visible-planar CAD interpretation, never physical performance.
Use only literal owner size/count facts. NEVER create hole size/center facts from
pixels. The host obtains derived hole geometry from its immutable photo target.
Keep original IDs, text, quotes and critical flags. For a physical, mixed,
unsupported or genuinely ambiguous duty use {requirement_id,unsupported_reason}
with a specific retained reason and no supported fields. Do not omit a duty or
invent a physical requirement merely because this is a digital copy request.
Photo predicates require a rectangular source envelope, thickness and exact
opening count somewhere in the same attributed source. Missing source facts
stay genuine questions or unsupported dispositions, never fabricated values.

For ordinary dimensional wording, nominal_targets may propose additional CAD
subchecks using nominal_perforated_box or nominal_through_hole_pattern. Keep ALL
original source duties as critical, unmeasured requirements without verification.
Each target has a NEW requirement_id, source_requirement_ids, criterion,
frame_policy, field_bindings and hole_count_fact_id. The host appends these
noncritical subchecks; do not put their IDs in requirement_order. Do not claim
source meaning or physical properties are proved. Never supply policy markers.
criterion.source_quote must equal the exact source_quote of one selected original
source requirement. Do not concatenate quotes or substitute the whole request.
Keep literal source values/units in existing facts. Bind EVERY size_mm.i,
center_mm.i, holes.i.center_mm.j and holes.i.diameter_mm field exactly once.
Each binding has field, basis=source_relation and terms [{fact_id,operation}],
where operation is value, negative, half or negative_half. Terms sum after exact
source-unit conversion to mm. For a centered body, choose frame_policy=centered_body
and center_mm=(0,0,0); those three fields alone may use basis=design_origin with
empty terms. This is an explicit CAD frame choice, never an owner-stated origin.
For a source-defined local frame use source_defined_frame and grounded operands.
For edge offsets, retain operands: e.g. a center coordinate can be negative_half
of body width plus value of the edge offset. Do not invent a coordinate fact.
hole_count_fact_id selects an existing count fact with source unit holes or bores.
Every selected fact must have this one designed subject and link to an original
source_requirement_id. Ambiguous subjects, frames, relations, units or conflicting
dimensions need an issue; do not propose affected targets before clarification.
Hole diameter_tolerance_mm and position_tolerance_mm MUST be 0.0. The saved STEP
observer supplies fixed computational error; no manufacturing allowance is inferred.
Explicit physical tolerance remains an original critical unverified source duty.
Ordinary wording needs no special nominal phrase or blanket new approval.
"""


def _raw(proposal: BaseModel) -> dict[str, Any]:
    if not isinstance(proposal, (SourceDeliveryContractDraft, PhotoSourceContractDraftV2)):
        raise ValueError("Initial source delivery requires SourceDeliveryContractDraft")
    raw: dict[str, Any] = proposal.model_dump(mode="json")
    return raw


def _source_delivery_base(proposal: BaseModel, source_text: str) -> ContractDraft:
    """One shared expansion; an unadmitted base can also retain rejected source duties."""
    raw = _raw(proposal)
    if isinstance(proposal, PhotoSourceContractDraftV2):
        from forge_mind.workspace.photo_source_contract_v2 import expand_photo_rows

        checked = PhotoSourceContractDraftV2.model_validate(raw).model_dump(mode="json")
        return _source_delivery_base(
            SourceDeliveryContractDraft.model_validate(expand_photo_rows(checked)), source_text
        )
    if raw["photo_nominal_claims"] and raw["photo_reference"] is None:
        raise ValueError("Prospective photo claims need an explicit source reference role")
    # Revalidate the public schema even if a caller used model_construct or model_copy.
    raw = SourceDeliveryContractDraft.model_validate(raw).model_dump(mode="json")
    original_ids = {row["id"] for kind in _NONDIGITAL_POOLS for row in raw[kind]}
    for row in raw["requirements"]:
        if row["verification_id"] is not None and row["verification_id"] not in original_ids:
            raise ValueError(
                "Source delivery ordinary references must select a nondigital criterion"
            )
    intermediate = {
        name: value
        for name, value in raw.items()
        if name
        not in {
            "digital_requirements",
            "nominal_targets",
            "photo_nominal_claims",
            "photo_reference",
        }
    }
    intermediate["digital_delivery"] = []
    next_id = 0
    for row in raw["digital_requirements"]:
        # Internal references carry no meaning and cannot capture authored references.
        while f"source_delivery_{next_id}" in original_ids:
            next_id += 1
        identifier = f"source_delivery_{next_id}"
        original_ids.add(identifier)
        next_id += 1
        intermediate["digital_delivery"].append(
            {"id": identifier, "quote_ref": row["quote_ref"], "deliverable": row["deliverable"]}
        )
        intermediate["requirements"].append(
            {
                **{name: value for name, value in row.items() if name != "deliverable"},
                "text": None,
                "verification_id": identifier,
                "scalar": None,
            }
        )
    return lower_compact_source_draft(
        CompactSourceContractDraft.model_validate(intermediate), source_text
    )


def lower_source_delivery_draft(proposal: BaseModel, source_text: str) -> ContractDraft:
    """Reuse canonical expansion and the strict host interpretation factory."""
    lowered = _source_delivery_base(proposal, source_text)
    from forge_mind.workspace.customer_interpretation import attach_customer_interpretation

    result = attach_customer_interpretation(
        lowered,
        source_text,
        [
            CustomerNominalTarget.model_validate(target)
            for target in _raw(proposal)["nominal_targets"]
        ],
    )
    from forge_mind.workspace.photo_nominal import (
        attach_photo_nominal_claims,
        validate_photo_reference,
    )

    raw = _raw(proposal)
    is_v2 = isinstance(proposal, PhotoSourceContractDraftV2)
    photo_rows = raw["photo_requirements"] if is_v2 else raw["photo_nominal_claims"]
    if raw["photo_reference"] is not None:
        reference = PhotoReferenceDisposition.model_validate(raw["photo_reference"])
        validate_photo_reference(reference, source_text)
        if reference.role == "ambiguous" and not any(
            issue.requirement_ids
            and issue.input_dependency is not None
            and issue.input_dependency.kind == "source_fact"
            and issue.input_dependency.input_name == "reference_role"
            for issue in result.attribution.issues
        ):
            raise ValueError(
                "An ambiguous reference role needs a genuine linked owner clarification issue"
            )
        if (
            reference.role == "copy"
            and not photo_rows
            and (not is_v2 or not result.contract.requirements)
        ):
            raise ValueError(
                "Every copying source duty needs an explicit photo nominal disposition"
            )
        if reference.role != "copy" and (
            bool(photo_rows) if is_v2 else any(row["claims"] for row in photo_rows)
        ):
            raise ValueError("Context/ambiguous references cannot admit photo copying proof")

    if is_v2:
        from forge_mind.workspace.photo_source_contract_v2 import (
            attach_source_photo_rows,
            validate_original_copy_clause_coverage,
        )

        if photo_rows:
            result = attach_source_photo_rows(result, source_text, raw)
        if raw["photo_reference"]["role"] == "copy":
            validate_original_copy_clause_coverage(result, source_text)
        return result
    return attach_photo_nominal_claims(
        result,
        source_text,
        [PhotoNominalClaim.model_validate(value) for value in raw["photo_nominal_claims"]],
    )


def source_delivery_proposal_record(
    proposal: BaseModel, source_text: str, lowered: ContractDraft | None = None
) -> dict[str, Any]:
    """Retain raw direct digital choices before any reference or source rejection."""
    raw = _raw(proposal)
    record: dict[str, Any] = {
        "protocol": PHOTO_PROTOCOL_V2
        if isinstance(proposal, PhotoSourceContractDraftV2)
        else PROTOCOL,
        "raw_response": raw,
        "raw_response_hash": digest(raw),
        "source_hash": digest(source_text),
        "source_only_requirement_ids": [row["id"] for row in raw["digital_requirements"]],
    }
    if isinstance(proposal, PhotoSourceContractDraftV2):
        record["source_only_photo_requirement_ids"] = [
            row["id"] for row in raw["photo_requirements"]
        ]
    if lowered is not None:
        canonical = lowered.model_dump(mode="json")
        expanded = lower_source_delivery_draft(proposal, source_text).model_dump(mode="json")
        if digest(expanded) != digest(canonical):
            raise ValueError("Source delivery proposal does not lower to this canonical draft")
        record["lowered_draft_hash"] = digest(canonical)
    return record


def source_delivery_draft_from_record(record: dict[str, Any]) -> BaseModel:
    """Protocol dispatch is shared by admission, role lookup and final replay."""
    models = {PROTOCOL: SourceDeliveryContractDraft, PHOTO_PROTOCOL_V2: PhotoSourceContractDraftV2}
    model = models.get(str(record.get("protocol")))
    if model is None:
        raise ValueError("Unsupported retained source delivery protocol")
    return model.model_validate(record["raw_response"])


@dataclass(frozen=True)
class SourceDeliveryPreflight:
    """Provisional diagnostics and retention context, never admission authority."""

    source_proposal: dict[str, Any]
    draft: ContractDraft | None
    retention_draft: ContractDraft | None
    compilation: dict[str, Any]


def _wire_retention_diagnostics(
    previous: dict[str, Any],
    current: dict[str, Any],
    source_text: str,
    *,
    retain_raw: bool,
    valid_photo_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Retain targets and, if expansion failed, the still-provisional source records."""
    diagnostics: list[dict[str, Any]] = []

    def reject(field: str) -> None:
        diagnostics.append(
            {
                "field": field,
                "code": "changed_source_proposal",
                "message": "Retain the original source duties, valid records and proposed target IDs; correct invalid fields without dropping obligations.",
            }
        )

    updated_targets = {row["requirement_id"]: row for row in current.get("nominal_targets", [])}
    if {row["id"] for row in previous.get("digital_requirements", [])} != {
        row["id"] for row in current.get("digital_requirements", [])
    }:
        reject("digital_requirements")
    for target in previous.get("nominal_targets", []):
        updated = updated_targets.get(target["requirement_id"])
        if updated is None or not set(target["source_requirement_ids"]) <= set(
            updated["source_requirement_ids"]
        ):
            reject(f"nominal_targets.{target['requirement_id']}")
    before_photo = {row["requirement_id"] for row in previous.get("photo_nominal_claims", [])}
    after_photo = {row["requirement_id"] for row in current.get("photo_nominal_claims", [])}
    if not before_photo <= after_photo:
        reject("photo_nominal_claims")
    updated_photo = {row["id"]: row for row in current.get("photo_requirements", [])}
    for row in previous.get("photo_requirements", []):
        updated = updated_photo.get(row["id"])
        if (
            updated is None
            or any(
                row[name] != updated[name]
                for name in ("source", "critical", "quote_ref", "subject_ids")
            )
            or (row["id"] in (valid_photo_ids or set()) and row != updated)
        ):
            reject(f"photo_requirements.{row['id']}")
    if retain_raw:
        # With no canonical base, remain conservative. Added quote/criterion records
        # may fix references, but cannot erase or rewrite these source-bearing rows.
        for field in (
            "requirements",
            "canonical_requirements",
            "digital_requirements",
            "subjects",
            "fact_groups",
            "issues",
        ):
            if any(row not in current.get(field, []) for row in previous.get(field, [])):
                reject(field)
        current_quotes = {row["id"]: row["text"] for row in current.get("quotes", [])}
        for quote in previous.get("quotes", []):
            if quote["text"] in source_text and current_quotes.get(quote["id"]) != quote["text"]:
                reject(f"quotes.{quote['id']}")
    return diagnostics


def preflight_source_delivery_draft(
    proposal: BaseModel, source_text: str, *, previous: SourceDeliveryPreflight | None = None
) -> SourceDeliveryPreflight:
    """Run the same deterministic factory inside the existing bounded repair boundary."""
    from forge_mind.workspace.source_contract_draft import compile_source_contract_draft

    record = source_delivery_proposal_record(proposal, source_text)
    draft = None
    retention_draft = None
    try:
        draft = lower_source_delivery_draft(proposal, source_text)
        record = source_delivery_proposal_record(proposal, source_text, draft)
    except ValueError as exc:
        if record["protocol"] not in {PROTOCOL, PHOTO_PROTOCOL_V2}:
            # Explicit historical compact-wire replay retains its original behavior.
            raise
        draft = None
        with suppress(ValueError):
            retention_draft = _source_delivery_base(proposal, source_text)
        compilation = {
            "status": "invalid_interpretation",
            "blocking_issue_ids": [],
            "diagnostics": [
                {
                    "field": getattr(exc, "field", "photo_requirements")
                    if record["protocol"] == PHOTO_PROTOCOL_V2
                    else "nominal_targets"
                    if retention_draft is not None
                    else "source_delivery",
                    "code": "invalid_source_lowering",
                    "message": str(exc),
                }
            ],
        }
    else:
        retention_draft = draft
        previous_draft = previous.retention_draft if previous else None
        if (
            previous_draft is not None
            and record["protocol"] == PHOTO_PROTOCOL_V2
            and previous is not None
            and previous.compilation["status"] == "invalid_interpretation"
        ):
            # Provisional display is not original prose. Source-bearing wire fields
            # remain protected below; valid predicates retain the ordinary guard.
            previous_draft = ContractDraft.model_validate(previous_draft.model_dump())
            generated_ids = {
                row["id"]
                for row in previous.source_proposal["raw_response"].get("photo_requirements", [])
            }
            current_rows = {row.id: row for row in draft.contract.requirements}
            for row in previous_draft.contract.requirements:
                if row.id in generated_ids and row.id in current_rows:
                    row.text = current_rows[row.id].text
        compilation = compile_source_contract_draft(
            draft,
            source_text,
            source_only_requirement_ids=(
                previous.source_proposal["source_only_requirement_ids"]
                if previous
                else record["source_only_requirement_ids"]
            ),
            previous=previous_draft,
        )
    if previous is not None:
        valid_photo_ids: set[str] = set()
        if (
            previous.retention_draft is not None
            and previous.source_proposal.get("protocol") == PHOTO_PROTOCOL_V2
        ):
            from forge_mind.workspace.photo_source_contract_v2 import valid_source_photo_rows

            valid_photo_ids = valid_source_photo_rows(
                previous.retention_draft, source_text, previous.source_proposal["raw_response"]
            )
        diagnostics = _wire_retention_diagnostics(
            previous.source_proposal["raw_response"],
            record["raw_response"],
            source_text,
            retain_raw=previous.retention_draft is None,
            valid_photo_ids=valid_photo_ids,
        )
        if diagnostics:
            compilation = {
                **compilation,
                "status": "invalid_interpretation",
                "diagnostics": [*compilation["diagnostics"], *diagnostics],
            }
    return SourceDeliveryPreflight(record, draft, retention_draft, compilation)
