"""Prospective source nominal claims; raw replay, not a caller policy label, grants use."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from forge_mind.workspace.customer_interpretation import _LENGTH_MM
from forge_mind.workspace.document import digest
from forge_mind.workspace.models import Record
from forge_mind.workspace.source_numbers import COUNT_WORDS
from forge_mind.workspace.verification_models import Name, NominalPhotoSolid, PhotoClaim

if TYPE_CHECKING:
    from forge_mind.workspace.intent_attribution import ContractDraft, IntentAttribution
    from forge_mind.workspace.models import Contract, Requirement


class PhotoNominalClaim(Record):
    requirement_id: Name
    subject_id: Name | None = None
    claims: list[PhotoClaim] = Field(default_factory=list)
    size_fact_ids: tuple[Name, Name, Name] | None = None
    hole_count_fact_id: Name | None = None
    frame_policy: Literal["image_plane_xy"] | None = None
    unsupported_reason: str | None = Field(default=None, min_length=1)


class PhotoReferenceDisposition(Record):
    role: Literal["copy", "context", "ambiguous"]
    source_quote: str = Field(min_length=1)


def validate_photo_reference(value: PhotoReferenceDisposition, source_text: str) -> None:
    if value.source_quote not in source_text:
        raise ValueError("Reference role must retain an exact source clause")
    copy = re.search(r"\b(?:copy|reproduce|match)\b", source_text, re.I)
    if value.role == "copy" and not re.search(
        r"\b(?:copy|reproduce|match)\b", value.source_quote, re.I
    ):
        raise ValueError("Reference copying needs a source copy clause")
    if value.role == "context" and copy:
        raise ValueError("A source copy clause cannot be skipped as reference context")
    if value.role == "context" and not re.fullmatch(
        r"\s*(?:use|treat)\s+(?:this|the)\s+(?:photo|image|reference)\s+"
        r"(?:(?:only\s+as|as)\s+context\s+only|only\s+as\s+context)\s*[.!]?\s*",
        value.source_quote,
        re.I,
    ):
        raise ValueError(
            "Context exemption needs an explicit contextual-only source directive; otherwise retain ambiguous role"
        )
    if value.role == "copy" and re.search(
        r"\b(?:not|never|without|don't|do not)\b", value.source_quote, re.I
    ):
        raise ValueError(
            "Negated or conditional copying needs an explicit unresolved interpretation"
        )


def photo_reference_role(session: dict[str, Any]) -> str:
    """Only an authenticated retained intake can exempt contextual reference use."""
    from forge_mind.workspace.intent_attribution_service import request_text

    source = request_text(session)
    compilation = session.get("intent_compilation") or {}
    proposal = compilation.get("source_proposal") or {}
    raw = proposal.get("raw_response")
    if (
        not isinstance(raw, dict)
        or proposal.get("raw_response_hash") != digest(raw)
        or proposal.get("source_hash") != digest(source)
        or compilation.get("source_hash") != digest(source)
        or proposal.get("protocol")
        not in {"source_delivery_contract_draft_v1", "photo_source_contract_draft_v2"}
        or compilation.get("status") not in {"ready", "needs_customer_input"}
        or not compilation.get("job_id")
        or compilation.get("job_id") != session.get("job_id")
    ):
        return "ambiguous"
    value = raw.get("photo_reference")
    if value is None:
        return "ambiguous"
    disposition = PhotoReferenceDisposition.model_validate(value)
    validate_photo_reference(disposition, source)
    from forge_mind.workspace.source_delivery_contract_draft import (
        lower_source_delivery_draft,
        source_delivery_draft_from_record,
    )

    lowered = lower_source_delivery_draft(source_delivery_draft_from_record(proposal), source)
    record = session.get("intent_attribution") or {}
    if disposition.role == "context":
        from forge_mind.workspace.models import Contract

        proposed = Contract.model_validate(record.get("proposed_contract") or {})
        if (
            proposal.get("lowered_draft_hash") != digest(lowered.model_dump(mode="json"))
            or record.get("source_hash") != digest(source)
            or record.get("contract_hash") != proposed.fingerprint()
            or record.get("status") not in {"interpreted", "needs_input"}
            or any(
                isinstance(row.verification, NominalPhotoSolid)
                for row in [*lowered.contract.requirements, *proposed.requirements]
            )
        ):
            return "ambiguous"
        return "context"
    if (
        proposal.get("lowered_draft_hash") != digest(lowered.model_dump(mode="json"))
        or record.get("contract_hash") != lowered.contract.fingerprint()
        or record.get("source_hash") != digest(source)
        or record.get("attribution") != lowered.attribution.model_dump()
    ):
        return "ambiguous"
    return disposition.role


# Known unsupported additions withhold this closed family. This is not a
# universal English classifier or a whitelist for proving arbitrary prose.
_UNSUPPORTED = re.compile(
    r"\b(?:withstand|force|newton|torque|seal|water|engrave|emboss|label|text|"
    r"rounded|chamfer|fillet|taper|slope|hinge|snap|nonrectangular|non-rectangular|"
    r"at least|at most|minimum|maximum|tolerance|except|unless|optional)\b|±|\+/-",
    re.I,
)


def _clause_facets(text: str, criterion: NominalPhotoSolid) -> set[str]:
    """Closed nominal vocabulary with complete residue coverage, not entailment.

    Unknown modifiers/operators retain the entire duty as unsupported. Quotes
    may contain several facets shared across atomic original requirement rows.
    """
    remaining = text.casefold()
    facets: set[str] = set()
    number = r"\d+(?:\.\d+)?"
    unit = r"(?:mm|cm|inch(?:es)?|in|millimet(?:er|re)s?)"
    pair = rf"({number})\s*(?:by|x|\u00d7)\s*({number})\s*({unit})"

    def body_quantity(start: int, end: int) -> None:
        prefix = remaining[:start].rsplit(".", 1)[-1]
        referents = re.findall(r"\b(plate|part|body|panel|sheet|holes?|bores?)\b", prefix)
        if referents and referents[-1] in {"hole", "holes", "bore", "bores"}:
            raise ValueError("An opening quantity cannot become a source body dimension")
        if re.match(
            r"\s*(?:(?:rectangular|straight|through)\s+)?(?:holes?|bores?)\b", remaining[end:]
        ):
            raise ValueError("A following opening noun cannot acquire body-dimension authority")

    for match in re.finditer(pair, remaining):
        body_quantity(match.start(), match.end())
        factor = _LENGTH_MM.get(match.group(3), 1.0)
        if any(
            not math.isclose(
                float(match.group(i + 1)) * factor, criterion.size_mm[i], rel_tol=0, abs_tol=1e-9
            )
            for i in range(2)
        ):
            raise ValueError("A source plane-dimension pair differs from the bound source facts")
    if re.search(pair, remaining):
        facets.update(("extent_u", "extent_v"))
        remaining = re.sub(pair, " ", remaining)
    thickness = rf"(?:({number})\s*({unit})\s*(?:thick|thickness)|(?:thickness|thick)\s*(?:of|is|:|=)?\s*({number})\s*({unit}))"
    for match in re.finditer(thickness, remaining):
        body_quantity(match.start(), match.end())
        amount, symbol = (
            (match.group(1), match.group(2))
            if match.group(1) is not None
            else (match.group(3), match.group(4))
        )
        if not math.isclose(
            float(amount) * _LENGTH_MM.get(symbol, 1.0),
            criterion.size_mm[2],
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("A source thickness differs from its bound literal fact")
    if re.search(thickness, remaining):
        facets.add("thickness")
        remaining = re.sub(thickness, " ", remaining)
    for match in list(re.finditer(rf"({number})\s*({unit})", remaining)):
        body_quantity(match.start(), match.end())
        prefix = remaining[: match.start()].rsplit(".", 1)[-1]
        referents = re.findall(r"\b(plate|part|body|panel|sheet|holes?|bores?)\b", prefix)
        suffix = remaining[match.end() :].split(".", 1)[0]
        if re.search(r"\b(?:length|width|height)\b", prefix + suffix):
            raise ValueError(
                "Named in-plane dimension roles need an explicit supported source-frame mapping"
            )
        if (referents and referents[-1] in {"hole", "holes", "bore", "bores"}) or re.match(
            r"\s*(?:diameter|wide|deep|radius|hole|bore)\b", suffix
        ):
            raise ValueError("An opening-size duty cannot become an envelope dimension")
        value = float(match.group(1)) * _LENGTH_MM.get(match.group(2), 1.0)
        matches = [
            name
            for name, expected in zip(
                ("extent_u", "extent_v", "thickness"), criterion.size_mm, strict=True
            )
            if math.isclose(value, expected, rel_tol=0, abs_tol=1e-9)
        ]
        if not matches:
            raise ValueError("An additional numeric source duty has no photo nominal predicate")
        facets.update(matches)
    remaining = re.sub(rf"{number}\s*{unit}", " ", remaining)
    counts = "|".join(COUNT_WORDS)
    count = rf"\b(?:{number}|{counts})\s+(?:visible\s+)?(?:holes?|bores?)\b"
    for match in re.finditer(count, remaining):
        token = match.group().split()[0]
        count_word = COUNT_WORDS.get(token)
        actual_count = float(token) if count_word is None else float(count_word)
        if actual_count != criterion.hole_count:
            raise ValueError("An additional opening count differs from its bound literal fact")
    if re.search(count, remaining):
        facets.add("opening_count")
        remaining = re.sub(count, " holes ", remaining)
    if re.search(r"\bthrough\b", remaining):
        facets.add("straight_through")
    if re.search(r"\brectangular\b", remaining):
        facets.add("rectangular_outline")
    if re.search(r"\b(?:copy|reproduce|match)\b", remaining):
        facets.add("visible_profile")
    # Only nouns/operators of the closed visible planar family are consumed.
    # This rejects unrepresented conjuncts instead of chasing their keywords.
    vocabulary = {
        "copy",
        "reproduce",
        "match",
        "this",
        "the",
        "a",
        "an",
        "its",
        "it",
        "of",
        "in",
        "from",
        "as",
        "shown",
        "supplied",
        "provided",
        "reference",
        "visual",
        "image",
        "photo",
        "photograph",
        "including",
        "include",
        "visible",
        "hole",
        "holes",
        "bore",
        "bores",
        "shape",
        "shapes",
        "profile",
        "profiles",
        "relative",
        "size",
        "sizes",
        "position",
        "positions",
        "placement",
        "rectangular",
        "outline",
        "plate",
        "part",
        "body",
        "panel",
        "sheet",
        "must",
        "have",
        "has",
        "be",
        "is",
        "are",
        "go",
        "goes",
        "straight",
        "through",
        "and",
        "with",
        "both",
        "side",
        "dimension",
        "dimensions",
        "length",
        "width",
        "height",
        "thickness",
        "thick",
        "stated",
        "first",
        "second",
        "exactly",
        "all",
        "into",
        "cad",
    }
    residue = [
        token
        for token in re.findall(r"[\w]+|[^\w\s.,;:()\-]", remaining)
        if token not in vocabulary
    ]
    if residue or not facets:
        raise ValueError(
            "Unrepresented source clause/operator remains outside the closed photo nominal scope"
        )
    return facets


def photo_source_constraints(criterion: NominalPhotoSolid) -> dict[str, Any]:
    return criterion.model_dump(
        mode="json",
        include={
            "subject_id",
            "frame",
            "axis",
            "size_mm",
            "hole_count",
            "size_fact_ids",
            "hole_count_fact_id",
        },
    )


def validate_photo_target_constraints(criterion: NominalPhotoSolid, actual: dict[str, Any]) -> None:
    expected = photo_source_constraints(criterion)
    if criterion.hole_count is None:
        if type(actual.get("hole_count")) is not int or actual["hole_count"] < 0:
            raise ValueError("Derived opening count needs complete nonnegative observed inventory")
        expected["hole_count"] = actual["hole_count"]
    if actual != expected:
        raise ValueError("Photo target differs from its exact source/derived constraint provenance")


def validate_photo_nominal_source(
    requirement: Requirement,
    attribution: IntentAttribution,
    source_text: str,
    contract: Contract,
) -> None:
    from forge_mind.workspace.intent_attribution import validate_fact
    from forge_mind.workspace.verification_admission import _source_scope

    criterion = requirement.verification
    if not isinstance(criterion, NominalPhotoSolid):
        raise ValueError("Expected a prospective photo nominal criterion")
    _source_scope(requirement)
    source_only = criterion.source_text_policy == "source_only_v2"
    combined = (
        requirement.source_quote
        if source_only
        else requirement.source_quote + " " + requirement.text
    )
    if _UNSUPPORTED.search(combined):
        raise ValueError("An unsupported or compound duty cannot hide in photo nominal proof")
    subjects = {subject.id: subject for subject in attribution.subjects}
    if (
        criterion.subject_id not in subjects
        or subjects[criterion.subject_id].role != "designed"
        or attribution.requirement_subjects.get(requirement.id) != [criterion.subject_id]
    ):
        raise ValueError("Photo nominal proof needs one exact designed subject")
    facts = {fact.id: fact for fact in attribution.facts}
    used = set(criterion.size_fact_ids)
    if criterion.hole_count_fact_id is not None:
        used.add(criterion.hole_count_fact_id)
    for identifier in used:
        if identifier not in facts:
            raise ValueError("Photo nominal proof references a missing source fact")
        fact = facts[identifier]
        validate_fact(fact, attribution, contract, source_text)
        if (
            fact.subject_id != criterion.subject_id
            or fact.candidate_subject_ids
            or not fact.requirement_ids
        ):
            raise ValueError("Photo nominal fact has unresolved subject or source-duty binding")
    for identifier, target in zip(criterion.size_fact_ids, criterion.size_mm, strict=True):
        fact = facts[identifier]
        factor = _LENGTH_MM.get(fact.unit.casefold())
        if factor is None or not math.isclose(
            fact.value * factor, target, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("Photo nominal dimension differs from literal source units")
    count = facts.get(criterion.hole_count_fact_id) if criterion.hole_count_fact_id else None
    if count is not None and (
        count.unit.casefold() not in {"hole", "holes", "bore", "bores"}
        or count.value != criterion.hole_count
    ):
        raise ValueError("Photo nominal opening count differs from the literal hole count")
    if count is None and any(
        fact.subject_id == criterion.subject_id
        and fact.unit.casefold() in {"hole", "holes", "bore", "bores"}
        for fact in facts.values()
    ):
        raise ValueError("An explicit source opening count cannot be replaced by image inventory")
    for issue in attribution.issues:
        if requirement.id in issue.requirement_ids or used.intersection(issue.fact_ids):
            raise ValueError("A genuine unanswered input blocks dependent photo nominal proof")
    if (
        criterion.source_quote != requirement.source_quote
        or requirement.source_quote not in source_text
    ):
        raise ValueError("Photo nominal criterion must retain the complete source duty quote")
    source_facets = _clause_facets(requirement.source_quote, criterion)
    if source_only:
        from forge_mind.workspace.photo_source_contract_v2 import photo_requirement_text

        if requirement.text != photo_requirement_text(
            criterion.claims, criterion.size_mm, criterion.hole_count
        ):
            raise ValueError("V2 display must equal its deterministic typed source projection")
        if not set(criterion.claims) <= source_facets:
            raise ValueError("Each typed photo facet must belong to its own owner clause")
    else:
        text_facets = _clause_facets(requirement.text, criterion)
        if not text_facets <= set(criterion.claims):
            raise ValueError("The original atomic duty contains an unproved nominal facet")
    if "visible_profile" in criterion.claims and not re.search(
        r"\b(?:copy|reproduce|match)\b", combined, re.I
    ):
        raise ValueError("Visible-profile interpretation needs an actual copy duty")
    if "rectangular_outline" in criterion.claims and not re.search(
        r"\brectangular\b", combined, re.I
    ):
        raise ValueError("Rectangular-outline proof needs its source duty")
    if "straight_through" in criterion.claims and not re.search(r"\bthrough\b", combined, re.I):
        raise ValueError("Throughness proof needs explicit source throughness")
    for name, identifier in zip(
        ("extent_u", "extent_v", "thickness"), criterion.size_fact_ids, strict=True
    ):
        if name in criterion.claims and requirement.id not in facts[identifier].requirement_ids:
            raise ValueError("A dimension claim must link its own original source duty")
    if "opening_count" in criterion.claims and (
        count is None or requirement.id not in count.requirement_ids
    ):
        raise ValueError("An opening-count claim must link its own original source duty")


def attach_photo_nominal_claims(
    draft: ContractDraft,
    source_text: str,
    proposals: list[PhotoNominalClaim],
) -> ContractDraft:
    if not proposals:
        return draft
    from forge_mind.workspace.intent_attribution import ContractDraft, validate_attribution

    validate_attribution(draft.attribution, draft.contract, source_text, validate_predicates=False)
    result = ContractDraft.model_validate(draft.model_dump())
    rows = {row.id: row for row in result.contract.requirements}
    facts = {fact.id: fact for fact in result.attribution.facts}
    seen: set[str] = set()
    if {proposal.requirement_id for proposal in proposals} != rows.keys():
        raise ValueError(
            "Every original photo duty needs an explicit supported or unsupported disposition"
        )
    for proposal in proposals:
        if proposal.requirement_id in seen or proposal.requirement_id not in rows:
            raise ValueError("Photo nominal claims need unique existing original duty IDs")
        seen.add(proposal.requirement_id)
        row = rows[proposal.requirement_id]
        if proposal.unsupported_reason is not None:
            if proposal.claims or any(
                value is not None
                for value in (
                    proposal.subject_id,
                    proposal.size_fact_ids,
                    proposal.hole_count_fact_id,
                    proposal.frame_policy,
                )
            ):
                raise ValueError("An unsupported duty cannot also supply a supported photo claim")
            continue
        if (
            not proposal.claims
            or proposal.subject_id is None
            or proposal.size_fact_ids is None
            or proposal.frame_policy is None
        ):
            raise ValueError("A supported photo duty needs complete source/frame bindings")
        if (
            row.source != "request"
            or not row.critical
            or row.metric != "unmeasured"
            or row.verification
        ):
            raise ValueError("Prospective photo claims cannot replace an existing frozen criterion")
        try:
            sizes = tuple(
                facts[key].value * _LENGTH_MM[facts[key].unit.casefold()]
                for key in proposal.size_fact_ids
            )
            count = (
                facts[proposal.hole_count_fact_id].value if proposal.hole_count_fact_id else None
            )
        except KeyError as exc:
            raise ValueError("Photo nominal claim has a missing fact or length unit") from exc
        if count is not None and not count.is_integer():
            raise ValueError("Photo opening count must be an exact integer")
        row.verification = NominalPhotoSolid(
            subject_id=proposal.subject_id,
            claims=proposal.claims,
            size_mm=(sizes[0], sizes[1], sizes[2]),
            hole_count=int(count) if count is not None else None,
            size_fact_ids=proposal.size_fact_ids,
            hole_count_fact_id=proposal.hole_count_fact_id,
            source_quote=row.source_quote,
        )
        try:
            validate_photo_nominal_source(row, result.attribution, source_text, result.contract)
        except ValueError as exc:
            from forge_mind.workspace.photo_source_contract_v2 import PhotoSourceLoweringError

            raise PhotoSourceLoweringError(row.id, "facets", str(exc)) from exc
    # Shared full quotes may distribute their recognized facets across original
    # atomic rows. Every facet still needs a supported row, never just a note.
    for row in result.contract.requirements:
        criterion = row.verification
        if not isinstance(criterion, NominalPhotoSolid):
            continue
        facets = _clause_facets(row.source_quote, criterion)
        covered = {
            claim
            for other in result.contract.requirements
            if isinstance(other.verification, NominalPhotoSolid)
            and other.verification.subject_id == criterion.subject_id
            for claim in other.verification.claims
        }
        if not facets <= covered:
            raise ValueError(
                "A recognized complete source-quote facet lacks an admitted original duty"
            )
    return result


def validate_photo_source_replay(
    document: dict[str, Any],
    job_id: str,
    source_text: str,
    contract: Contract,
    attribution: IntentAttribution,
    *,
    require_photo: bool = False,
) -> None:
    if not require_photo and not any(
        isinstance(row.verification, NominalPhotoSolid) for row in contract.requirements
    ):
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
        or (require_photo and not raw.get("photo_reference"))
    ):
        raise ValueError(
            "Photo nominal proof requires the current internal retained source proposal"
        )
    lowered = lower_source_delivery_draft(source_delivery_draft_from_record(proposal), source_text)
    if require_photo and raw.get("photo_reference") is not None:
        reference = PhotoReferenceDisposition.model_validate(raw["photo_reference"])
        validate_photo_reference(reference, source_text)
        if (
            reference.role == "context"
            and not any(
                isinstance(row.verification, NominalPhotoSolid)
                for row in [*lowered.contract.requirements, *contract.requirements]
            )
            and proposal.get("lowered_draft_hash") == digest(lowered.model_dump(mode="json"))
        ):
            # Only reference-role authority is reused. The existing canonical
            # correction/admission still validates every current source duty.
            # No geometric photo criterion can enter through this exception.
            return
    if (
        lowered.contract != contract
        or lowered.attribution != attribution
        or proposal.get("lowered_draft_hash") != digest(lowered.model_dump(mode="json"))
        or compilation.get("status")
        != ("needs_customer_input" if lowered.attribution.issues else "ready")
    ):
        raise ValueError("Photo nominal proof differs from exact current source-proposal replay")
