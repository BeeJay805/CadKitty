"""Prospective source-only photo rows; canonical prose is a host projection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from forge_mind.workspace.models import Record
from forge_mind.workspace.verification_models import Name, PhotoClaim

PROTOCOL = "photo_source_contract_draft_v2"

if TYPE_CHECKING:
    from forge_mind.workspace.intent_attribution import ContractDraft


class SourcePhotoRequirement(Record):
    id: Name
    source: Literal["request"] = "request"
    critical: Literal[True] = True
    quote_ref: Name
    subject_ids: list[Name] = Field(min_length=1, max_length=1)
    facets: list[PhotoClaim] = Field(min_length=1)
    size_fact_ids: tuple[Name, Name, Name]
    hole_count_fact_id: Name | None = None
    frame_policy: Literal["image_plane_xy"] = "image_plane_xy"


class PhotoSourceLoweringError(ValueError):
    def __init__(self, identifier: str, field: str, message: str):
        self.field = f"photo_requirements.{identifier}.{field}"
        super().__init__(message)


def photo_requirement_text(
    facets: Sequence[str], sizes: tuple[float, ...], count: int | None
) -> str:
    """Fixed ordered templates, never authored acceptance prose or image facts."""
    templates = {
        "visible_profile": "Copy the visible profile.",
        "rectangular_outline": "The plate must have a rectangular outline.",
        "extent_u": f"The plate must have a {sizes[0]:g} mm dimension.",
        "extent_v": f"The plate must have a {sizes[1]:g} mm dimension.",
        "thickness": f"The plate must be {sizes[2]:g} mm thick.",
        "opening_count": f"The plate must have {count} holes.",
        "straight_through": "The holes must go straight through.",
    }
    return " ".join(text for facet, text in templates.items() if facet in facets)


def expand_photo_rows(raw: dict[str, Any]) -> dict[str, Any]:
    """Expand only the source row shape, before canonical reference validation."""
    from copy import deepcopy

    value = deepcopy(raw)
    rows = value.pop("photo_requirements")
    value["photo_nominal_claims"] = []
    for row in rows:
        value["requirements"].append(
            {
                **{
                    key: row[key]
                    for key in ("id", "source", "critical", "quote_ref", "subject_ids")
                },
                "text": "Source photo duty.",
                "verification_id": None,
                "scalar": None,
            }
        )
    return value


def _lower_photo_row(draft: ContractDraft, source: str, authored: dict[str, Any]) -> None:
    """Shared per-row lowering/validation for admission and provisional retention."""
    from forge_mind.workspace.customer_interpretation import _LENGTH_MM
    from forge_mind.workspace.photo_nominal import _clause_facets, validate_photo_nominal_source
    from forge_mind.workspace.source_contract_draft import _enclosing_clause, _unique_span
    from forge_mind.workspace.verification_models import NominalPhotoSolid

    identifier = authored["id"]
    row = next(row for row in draft.contract.requirements if row.id == identifier)
    facts = {fact.id: fact for fact in draft.attribution.facts}
    try:
        sizes = tuple(
            facts[key].value * _LENGTH_MM[facts[key].unit.casefold()]
            for key in authored["size_fact_ids"]
        )
        count_value = (
            facts[authored["hole_count_fact_id"]].value if authored["hole_count_fact_id"] else None
        )
        if count_value is not None and not count_value.is_integer():
            raise ValueError("Source opening count must be an exact integer")
        row.verification = NominalPhotoSolid(
            subject_id=authored["subject_ids"][0],
            claims=authored["facets"],
            size_mm=(sizes[0], sizes[1], sizes[2]),
            hole_count=int(count_value) if count_value is not None else None,
            size_fact_ids=authored["size_fact_ids"],
            hole_count_fact_id=authored["hole_count_fact_id"],
            source_quote=row.source_quote,
            source_text_policy="source_only_v2",
        )
    except (KeyError, ValueError) as exc:
        raise PhotoSourceLoweringError(identifier, "size_fact_ids", str(exc)) from exc
    row.text = photo_requirement_text(authored["facets"], sizes, row.verification.hole_count)
    span = _unique_span(source, row.source_quote)
    if span is None:
        raise PhotoSourceLoweringError(
            identifier, "quote_ref", "Photo source quote needs one unique immutable occurrence"
        )
    start, end = _enclosing_clause(source, span)
    if source[start:end].strip() != row.source_quote.strip():
        raise PhotoSourceLoweringError(
            identifier,
            "quote_ref",
            "Retain the complete enclosing owner clause, including every conjunct",
        )
    try:
        validate_photo_nominal_source(row, draft.attribution, source, draft.contract)
        facets = _clause_facets(row.source_quote, row.verification)
    except ValueError as exc:
        raise PhotoSourceLoweringError(identifier, "facets", str(exc)) from exc
    for facet, fact_id in zip(
        ("extent_u", "extent_v", "thickness"), row.verification.size_fact_ids, strict=True
    ):
        if facet in facets and facts[fact_id].source_quote not in row.source_quote:
            raise PhotoSourceLoweringError(
                identifier,
                "size_fact_ids",
                "Source dimension facts must belong to this exact owner clause",
            )
    if "opening_count" in facets and (
        row.verification.hole_count_fact_id is None
        or facts[row.verification.hole_count_fact_id].source_quote not in row.source_quote
    ):
        raise PhotoSourceLoweringError(
            identifier,
            "hole_count_fact_id",
            "The explicit source count must belong to this exact owner clause",
        )


def valid_source_photo_rows(draft: ContractDraft, source: str, raw: dict[str, Any]) -> set[str]:
    """An invalid sibling cannot erase an independently valid original typed duty."""
    from forge_mind.workspace.intent_attribution import ContractDraft

    valid = set()
    for authored in raw.get("photo_requirements", []):
        candidate = ContractDraft.model_validate(draft.model_dump())
        try:
            _lower_photo_row(candidate, source, authored)
        except ValueError:
            continue
        valid.add(authored["id"])
    return valid


def attach_source_photo_rows(
    draft: ContractDraft, source: str, raw: dict[str, Any]
) -> ContractDraft:
    """One lowered criterion path, with quote-local complete source coverage."""
    from forge_mind.workspace.intent_attribution import ContractDraft, validate_attribution
    from forge_mind.workspace.photo_nominal import _clause_facets
    from forge_mind.workspace.verification_models import NominalPhotoSolid

    validate_attribution(draft.attribution, draft.contract, source, validate_predicates=False)
    result = ContractDraft.model_validate(draft.model_dump())
    typed = {row["id"]: row for row in raw["photo_requirements"]}
    if len(typed) != len(raw["photo_requirements"]):
        raise PhotoSourceLoweringError("*", "id", "Photo source IDs must be unique")
    for authored in typed.values():
        _lower_photo_row(result, source, authored)
    for row in result.contract.requirements:
        criterion = row.verification
        if not isinstance(criterion, NominalPhotoSolid):
            continue
        facets = _clause_facets(row.source_quote, criterion)
        covered: set[str] = set()
        for other in result.contract.requirements:
            sibling = other.verification
            if (
                not isinstance(sibling, NominalPhotoSolid)
                or other.source_quote != row.source_quote
                or sibling.subject_id != criterion.subject_id
            ):
                continue
            if (sibling.size_fact_ids, sibling.hole_count_fact_id, sibling.frame, sibling.axis) != (
                criterion.size_fact_ids,
                criterion.hole_count_fact_id,
                criterion.frame,
                criterion.axis,
            ):
                raise PhotoSourceLoweringError(
                    row.id,
                    "size_fact_ids",
                    "Shared source-clause facets need the same exact fact/frame bindings",
                )
            covered.update(sibling.claims)
        if not facets <= covered:
            raise PhotoSourceLoweringError(
                row.id,
                "facets",
                "Every source-clause facet needs a same-quote, same-subject original duty",
            )
    return result


def validate_original_copy_clause_coverage(draft: ContractDraft, source: str) -> None:
    """Every original copy-source clause needs a retained critical source duty.

    This is immutable span coverage, not an English meaning classifier. Unknown
    clauses must remain verbatim, critical and unverified. Facts and notes cannot
    stand in for a duty or manufacture acceptance authority.
    """
    from forge_mind.workspace.source_contract_draft import _enclosing_clause, _unique_span
    from forge_mind.workspace.verification_models import DigitalDelivery, NominalPhotoSolid

    covered = []
    for row in draft.contract.requirements:
        if row.source != "request" or not row.critical:
            continue
        if not isinstance(row.verification, (NominalPhotoSolid, DigitalDelivery)) and not (
            row.verification is None and row.metric == "unmeasured" and row.text == row.source_quote
        ):
            continue
        span = _unique_span(source, row.source_quote)
        if span is not None:
            covered.append(span)
    cursor = 0
    while cursor < len(source):
        if source[cursor].isspace():
            cursor += 1
            continue
        start, end = _enclosing_clause(source, (cursor, cursor + 1))
        while start < end and source[start].isspace():
            start += 1
        while end > start and source[end - 1].isspace():
            end -= 1
        if not any(left <= start and end <= right for left, right in covered):
            raise PhotoSourceLoweringError(
                "*",
                "source_coverage",
                f"Original owner clause at source[{start}:{end}] lacks a retained critical complete duty: {source[start:end]!r}. Retain unsupported clauses verbatim and unverified.",
            )
        cursor = end
