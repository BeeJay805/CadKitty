"""Conservative source admission and exact subject resolution, never CAD proof.

These checks establish supported structure and source links. They do not prove
arbitrary English completeness. Unrecognized claims retain legacy uncertainty.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator
from functools import partial
from typing import TYPE_CHECKING, Any, get_args

from forge_mind.workspace.source_numbers import POSITIVE_COUNT_WORDS as _COUNT_WORDS
from forge_mind.workspace.verification_models import (
    DigitalDelivery,
    NominalDeliveryScope,
    NominalInventoryItem,
    NominalPattern,
    NominalPerforatedBox,
    NominalPhotoSolid,
    NominalSubjectInventory,
    VerificationCriterion,
)

if TYPE_CHECKING:
    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.intent_attribution import IntentAttribution
    from forge_mind.workspace.models import Contract, DesignPlan, Requirement

_NUMBER = r"[+\-\u2212]?(?:\d+(?:\.\d+)?|\.\d+)"
# A known extra duty must not disappear inside a convenient nominal predicate.
# This conservative refusal is not a universal natural-language classifier.
_EXTRA_DUTY = re.compile(
    r"\b(?:load|support(?:s|ing)?|hold(?:s|ing)?|strength|safe(?:ty)?|accurac(?:y|ies)|"
    r"accurate|precis(?:e|ion)|fit(?:s|ting)?|friction|thread(?:s|ed)?|"
    r"simple|easy|novice|durab(?:le|ility)|travel|"
    r"clearance|threading|PLA|PETG|ABS)\b",
    re.I,
)
_GEOMETRY_DUTY = re.compile(
    r"\b(?:mm|cm|inches?|diameter|holes?|bores?|through|thickness|width|height|"
    r"length|center(?:ed)?|centre(?:d)?|rectangular|circular|dimensions?|solid_count)\b",
    re.I,
)
# This only withholds repair exclusions. It adds no accepted English grammar or
# global admission rule and does not claim exhaustive ambiguity recognition.
_INVENTORY_REPAIR_AMBIGUITY = re.compile(
    r"\b(?:if|unless|when|whether|or|either|neither|optional(?:ly)?|maybe|perhaps|"
    r"provided|depending|conditional(?:ly)?|alternatively|except)\b",
    re.I,
)


def _body_source(criterion: NominalPerforatedBox, quote: str) -> None:
    if not re.search(r"\brectangular\b", quote, re.I):
        raise ValueError("The rectangular material domain must occur in the source")
    if re.search(r"\b(?:digital|stl|step|print|printing|editable)\b", quote, re.I):
        raise ValueError("Keep digital delivery duties separate from nominal body geometry")
    for axis, size in zip("xyz", criterion.size_mm, strict=True):
        values = [
            float(match.replace("\u2212", "-"))
            for pattern in (
                rf"({_NUMBER})\s*mm\s+in\s+local\s+{axis}\b",
                rf"\blocal\s+{axis}\s*[:=]?\s*({_NUMBER})\s*mm\b",
            )
            for match in re.findall(pattern, quote, re.I)
        ]
        if not values or any(
            not math.isclose(size, value, rel_tol=0, abs_tol=1e-9) for value in values
        ):
            raise ValueError("Each local prism dimension must match its source axis")
    if re.search(r"cent(?:er|re)ed\s+(?:on|at)\s+(?:the\s+)?local\s+origin\b", quote, re.I):
        if criterion.center_mm != (0.0, 0.0, 0.0):
            raise ValueError("Centered on the local origin requires zero outer prism center")
        return
    centers = [
        tuple(float(value.replace("\u2212", "-")) for value in match)
        for match in re.findall(
            rf"\bcent(?:er|re)(?:ed)?\s*(?:at|:|=)?\s*(?:local\s*)?"
            rf"[\[(]\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*[\])]",
            quote,
            re.I,
        )
    ]
    if centers != [criterion.center_mm]:
        raise ValueError("An explicit complete local outer prism center is required")


def criterion_hash(criterion: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(criterion, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _structured_source(criterion: VerificationCriterion, quote: str) -> bool:
    """Explicit machine-readable nominal input is optional, not a hidden phrase."""
    try:
        value = json.loads(quote)
        if not isinstance(value, dict) or "source_quote" in value:
            return False
        validated = type(criterion).model_validate({**value, "source_quote": quote})
    except (ValueError, TypeError):
        return False
    return validated == criterion


def _inventory_source_item(item: NominalInventoryItem, quote: str) -> tuple[int, int]:
    """Associate a literal named subject with its own explicit occurrence count."""
    count = r"\d+|" + "|".join(_COUNT_WORDS)
    name = re.escape(item.required_name)
    naming = (
        rf'(?P<literal>"{name}"|\'{name}\'|`{name}`|{name})(?=$|[.;,!?]|\s+(?:and|plus|also)\b)'
    )
    pattern = (
        rf"(?=\b(?P<count>{count})\s+"
        rf"(?P<label>(?:(?!\b(?:named|called)\b)[^.;!?\n])+?)"
        rf"\b(?:named|called)\s+{naming})"
    )
    quantities = []
    spans = []
    for match in re.finditer(pattern, quote, re.I):
        literal = match.group("literal")
        if literal[:1] in {'"', "'", "`"} and literal[-1:] == literal[:1]:
            literal = literal[1:-1]
        if literal != item.required_name:
            continue
        label = match.group("label")
        # Another quantity or a dimensional unit cannot silently become this
        # subject's occurrence count. Unsupported wording remains unverified.
        if re.search(rf"\b(?:{count}|mm|cm|inches?|solids?|holes?|bores?)\b", label, re.I):
            continue
        if re.search(
            r"\b(?:pairs?|sets?|packs?|dozens?|couples?|scores?|hundreds?|thousands?|"
            r"millions?|billions?|bundles?|batches?|groups?|assemblies|subassemblies)\b|"
            r"\b(?:box|boxes|kit|kits|carton|cartons|tray|trays)\s+of\b",
            label,
            re.I,
        ):
            continue
        raw = match.group("count").lower()
        quantities.append(_COUNT_WORDS[raw] if raw in _COUNT_WORDS else int(raw))
        spans.append((match.start("count"), match.end("literal")))
    if quantities != [item.quantity]:
        raise ValueError("Each inventory name and occurrence quantity must trace to its source")
    if re.search(r"\b(?:at least|at most|minimum|maximum|about|approximately)\b", quote, re.I):
        raise ValueError("Inventory acceptance requires an exact source quantity")
    return spans[0]


def _scope_source(quote: str) -> None:
    prefix = (
        r"(?:(?:I\s+(?:need|want)|(?:please\s+)?(?:provide|give\s+me|deliver)|"
        r"scope\s*:|(?:limit\s+(?:delivery|scope)\s+to))\s*)?"
    )
    domain = r"(?:only\s+)?nominal\s+(?:CAD|geometry)\s+(?:and|plus|\+)\s+digital\s+(?:files|deliverables|artifacts)(?:\s+only)?"
    if not re.fullmatch(prefix + domain + r"[.!]?", quote.strip(), re.I) or not re.search(
        r"\bonly\b", quote, re.I
    ):
        raise ValueError("Nominal delivery policy requires an explicit scope-only source clause")


def _inventory_source_complete(criterion: NominalSubjectInventory, quote: str) -> None:
    naming = re.compile(
        r"\b(?:named|called)\s+(?:\"(?P<double>[^\"\n]+)\"|'(?P<single>[^'\n]+)'|"
        r"`(?P<backtick>[^`\n]+)`|(?P<bare>[^\W\d_][\w -]*?))"
        r"(?=$|[.;,!?]|\s+(?:and|plus|also)\b)",
        re.I,
    )
    names = []
    residual = list(quote)
    for match in naming.finditer(quote):
        names.append(next(value for value in match.groups() if value is not None))
        residual[match.start() : match.end()] = " " * (match.end() - match.start())
    if sorted(names) != sorted(item.required_name for item in criterion.items) or re.search(
        r"\b(?:named|called)\b", "".join(residual), re.I
    ):
        raise ValueError("The complete explicit named inventory must match its frozen items")


def _inventory_residue(text: str) -> str:
    return re.sub(
        r"\b(?:make|create|provide|deliver|I|need|want|please|exactly|only|and|plus|also)\b",
        "",
        text,
        flags=re.I,
    )


def _has_nominal_cad_scope(source_text: str) -> bool:
    return bool(
        re.search(r"\bnominal\b", source_text, re.I)
        and re.search(r"\b(?:cad|geometry)\b", source_text, re.I)
    )


def _source_identity_checks(
    requirement: Requirement, source_text: str, allow_explicit: bool
) -> Iterator[tuple[bool, str, str, str]]:
    criterion = requirement.verification
    assert criterion is not None
    yield (
        requirement.metric == "unmeasured"
        and requirement.value is None
        and not requirement.part_id,
        "verification",
        "mixed_scalar_targets",
        "Typed verification cannot mix scalar targets or part selectors",
    )
    yield (
        requirement.source != "assumption",
        "source",
        "unconfirmed_assumption",
        "A verification assumption requires explicit owner confirmation",
    )
    yield (
        criterion.source_quote == requirement.source_quote,
        "verification.source_quote",
        "source_quote_mismatch",
        "Typed verification must retain its complete requirement source quote",
    )
    if not (allow_explicit and requirement.source == "explicit"):
        yield (
            criterion.source_quote in source_text,
            "verification.source_quote",
            "source_quote_not_in_request",
            "Verification source quote must occur exactly in the request",
        )
        yield (
            requirement.source_quote in source_text,
            "source_quote",
            "requirement_quote_not_in_request",
            "Complete requirement source quote must occur exactly in the request",
        )


def _source_scope(requirement: Requirement) -> None:
    quote = requirement.source_quote
    if _EXTRA_DUTY.search(quote) or _EXTRA_DUTY.search(requirement.text):
        raise ValueError("A mixed physical or subjective claim is not a closed nominal predicate")


def _inventory_geometry_text(requirement: Requirement, criterion: NominalSubjectInventory) -> str:
    text = requirement.text
    for item in criterion.items:
        text = text.replace(item.required_name, "")
    return text


def _digital_geometry_conflict(requirement: Requirement) -> bool:
    return bool(
        _GEOMETRY_DUTY.search(requirement.source_quote) or _GEOMETRY_DUTY.search(requirement.text)
    )


def _source_family(requirement: Requirement) -> None:
    criterion = requirement.verification
    assert criterion is not None
    if isinstance(criterion, (NominalSubjectInventory, NominalDeliveryScope)):
        text = requirement.text
        if isinstance(criterion, NominalSubjectInventory):
            text = _inventory_geometry_text(requirement, criterion)
        if _GEOMETRY_DUTY.search(text):
            raise ValueError(
                "Identity or software policy cannot replace a geometry requirement. "
                "nominal_subject_inventory proves names and occurrence counts only; "
                "nominal_perforated_box covers an explicitly specified rectangular body and bores; "
                "nominal_through_hole_pattern covers only its explicit bore pattern. "
                "Retain the complete source duty and supply source-backed fields, not inferred targets."
            )
    if isinstance(criterion, DigitalDelivery) and _digital_geometry_conflict(requirement):
        raise ValueError(
            "Digital artifact proof cannot replace a geometry requirement. "
            "nominal_perforated_box covers an explicitly specified rectangular body and bores; "
            "nominal_through_hole_pattern covers only its explicit bore pattern. "
            "Keep file delivery as a separate source-backed obligation; do not infer targets."
        )
    if isinstance(criterion, DigitalDelivery) and criterion.deliverable == "prepared_print_package":
        from forge_mind.workspace.intent_capabilities import validate_preparation_source_scope

        validate_preparation_source_scope(requirement.text)


def _source_targets(requirement: Requirement, source_text: str) -> None:
    criterion = requirement.verification
    assert criterion is not None
    if isinstance(criterion, NominalPhotoSolid):
        raise ValueError("Photo nominal source admission requires attributed prospective lowering")
    # Inspect the actual complete source duty even when predicate provenance is
    # wrong. This supplies diagnostics, never a copied or corrected criterion.
    quote = requirement.source_quote
    if _structured_source(criterion, quote):
        return
    if isinstance(criterion, DigitalDelivery) and criterion.deliverable == "prepared_print_package":
        from forge_mind.workspace.intent_capabilities import validate_preparation_source_scope

        validate_preparation_source_scope(quote)
    if isinstance(criterion, NominalDeliveryScope):
        _scope_source(quote)
        return
    if isinstance(criterion, NominalSubjectInventory):
        if not _has_nominal_cad_scope(source_text):
            raise ValueError("Nominal inventory requires explicit nominal CAD scope")
        _inventory_source_complete(criterion, quote)
        unquoted_names = quote
        remaining = list(quote)
        for item in criterion.items:
            start, end = _inventory_source_item(item, quote)
            remaining[start:end] = " " * (end - start)
            unquoted_names = unquoted_names.replace(item.required_name, "")
        if re.search(r"\b(?:mm|cm|inches?|diameter|holes?|bores?|through)\b", unquoted_names, re.I):
            raise ValueError("Keep dimensional and bore duties separate from inventory proof")
        residue = _inventory_residue("".join(remaining))
        if re.search(r"\w", residue):
            raise ValueError("Additional source duties cannot hide inside inventory proof")
        return
    if isinstance(criterion, DigitalDelivery):
        supported = {
            "project_stl": r"\bstl\b",
            "project_step": r"\bstep\b",
            "editable_native": r"\beditable\b.*\b(?:native|cad)\b|\bnative\b.*\beditable\b",
            "prepared_print_package": r"\bprepared\b.*\bprint\b|\bprint\b.*\bprepared\b",
            "bound_instruction_plan": r"\bbound\b.*\binstructions?\b|\binstructions?\b.*\bbound\b",
        }
        present = {
            kind for kind, pattern in supported.items() if re.search(pattern, quote, re.I | re.S)
        }
        if present != {criterion.deliverable}:
            raise ValueError("Requested digital deliverable cannot be traced to its source clause")
        if re.search(r"\b(?:and|also|plus)\b", quote, re.I):
            raise ValueError("Keep additional requested clauses separate from typed file proof")
        return
    if not _has_nominal_cad_scope(source_text):
        raise ValueError("Through-pattern acceptance requires explicit nominal CAD scope")
    if not re.search(r"\bthrough\b", quote, re.I) or not re.search(
        r"\b(?:holes?|bores?)\b", quote, re.I
    ):
        raise ValueError("Through-hole scope must occur explicitly in the source")
    if not re.search(r"\blocal\b", quote, re.I) or not re.search(
        rf"\b{criterion.axis}\b", quote, re.I
    ):
        raise ValueError("The local axis must occur explicitly in the source")
    if not re.search(r"\bmm\b", quote, re.I):
        raise ValueError("Nominal pattern v1 requires explicit millimetre source units")
    if isinstance(criterion, NominalPerforatedBox):
        _body_source(criterion, quote)
    elif re.search(r"\b(?:width|length|thickness|rectangular|centered|centred)\b", quote, re.I):
        raise ValueError("Keep additional body geometry separate from a through-pattern predicate")
    count_words = "|".join(_COUNT_WORDS)
    counts = [
        _COUNT_WORDS[value.lower()] if value.lower() in _COUNT_WORDS else int(value)
        for value in re.findall(
            rf"\b(\d+|{count_words})\s+(?:(?:uniform|straight|cylindrical|through)[ -]+)*"
            r"(?:holes?|bores?)\b",
            quote,
            re.I,
        )
    ]
    if counts != [len(criterion.holes)] or re.search(
        r"\b(?:at least|at most|minimum|maximum)\b", quote, re.I
    ):
        raise ValueError("Exact pattern count cannot be traced to the source")
    center_matches = list(re.finditer(rf"[\[(]\s*({_NUMBER})\s*,\s*({_NUMBER})\s*[\])]", quote))
    pairs = [
        tuple(float(value.replace("\u2212", "-")) for value in match.groups())
        for match in center_matches
    ]
    if len(pairs) != len(criterion.holes) or set(pairs) != {
        hole.center_mm for hole in criterion.holes
    }:
        raise ValueError("The complete source center pattern must match the frozen pattern")
    # Bind a source center to its preceding explicit diameter declaration. This
    # supports ordinary grouped patterns without treating any number as a target.
    diameters = sorted(
        [
            (match.start(), float(match.group(1).replace("\u2212", "-")))
            for pattern in (
                rf"\bdiameter\s*(?:of|is|:|=)?\s*({_NUMBER})",
                rf"({_NUMBER})\s*mm\s+(?:in\s+)?diameter\b",
            )
            for match in re.finditer(pattern, quote, re.I)
        ]
    )
    source_diameters: dict[tuple[float, ...], float] = {}
    for match, pair in zip(center_matches, pairs, strict=True):
        preceding = [diameter for position, diameter in diameters if position < match.start()]
        if not preceding:
            raise ValueError("Each source center needs an explicit associated diameter")
        source_diameters[pair] = preceding[-1]
    allowances = [
        float(item)
        for match in re.findall(
            r"(?:±|\+\s*/\s*-)\s*(\d+(?:\.\d+)?)|"
            r"diameter\s+tolerance\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
            quote,
            re.I,
        )
        for item in match
        if item
    ]
    positions = [
        float(value)
        for value in re.findall(
            r"position(?:al)?\s+tolerance\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)", quote, re.I
        )
    ]
    if (
        len({(hole.diameter_tolerance_mm, hole.position_tolerance_mm) for hole in criterion.holes})
        != 1
    ):
        raise ValueError(
            "Different hole allowances require explicit structured nominal source data"
        )
    for hole in criterion.holes:
        if hole.center_mm not in pairs:
            raise ValueError("Each complete transverse center pair must occur in the source")
        if not math.isclose(
            hole.diameter_mm, source_diameters[hole.center_mm], rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("Hole diameter cannot be traced to the source")
        if (hole.diameter_tolerance_mm and not allowances) or any(
            not math.isclose(hole.diameter_tolerance_mm, float(value), rel_tol=0, abs_tol=1e-9)
            for value in allowances
        ):
            raise ValueError("Diameter tolerance cannot be traced to the source")
        if (hole.position_tolerance_mm and not positions) or any(
            not math.isclose(hole.position_tolerance_mm, float(value), rel_tol=0, abs_tol=1e-9)
            for value in positions
        ):
            raise ValueError("Position tolerance cannot be traced to the source")


def _requirement_attribution(requirement: Requirement, attribution: IntentAttribution) -> None:
    criterion = requirement.verification
    if isinstance(criterion, NominalPattern):
        subjects = {subject.id: subject for subject in attribution.subjects}
        subject = subjects.get(criterion.subject_id)
        if subject is None or subject.role != "designed":
            raise ValueError("Nominal verification requires an attributed designed subject")
        if attribution.requirement_subjects.get(requirement.id) != [criterion.subject_id]:
            raise ValueError("Nominal verification requires one exact requirement subject")
        if any(requirement.id in issue.requirement_ids for issue in attribution.issues):
            raise ValueError("Resolve subject or target uncertainty before typed acceptance")
    if isinstance(criterion, NominalSubjectInventory):
        subjects = {subject.id: subject for subject in attribution.subjects}
        expected = {item.subject_id for item in criterion.items}
        linked = attribution.requirement_subjects.get(requirement.id, [])
        if len(linked) != len(expected) or set(linked) != expected:
            raise ValueError("Inventory requires its exact set of attributed subjects")
        structured = _structured_source(criterion, criterion.source_quote)
        for item in criterion.items:
            subject = subjects.get(item.subject_id)
            if subject is None or subject.role != "designed":
                raise ValueError("Inventory requires attributed designed subjects")
            if subject.source_quote not in criterion.source_quote:
                raise ValueError("Inventory subject quote must belong to its frozen source")
            if not structured:
                _inventory_source_item(item, subject.source_quote)
        if any(requirement.id in issue.requirement_ids for issue in attribution.issues):
            raise ValueError("Resolve inventory subject uncertainty before typed acceptance")


def _supported_kinds() -> dict[str, Any]:
    """Project canonical schema metadata, without a second capability catalog."""
    result = {}
    for model in get_args(VerificationCriterion):
        schema = model.model_json_schema()
        fields = schema["properties"]
        result[fields["kind"]["const"]] = {
            "scope": fields["scope"]["const"],
            "required_fields": schema["required"],
            "description": schema.get("description", ""),
        }
    return result


def verification_repair_exclusions(
    requirement: Requirement,
    source_text: str,
    attribution: IntentAttribution | None,
) -> list[dict[str, str]]:
    """Certify a negative kind only when changing its payload cannot repair it.

    This is relative to existing canonical admission, not universal English
    interpretation. No preferred replacement, source edit or target is supplied.
    Callers must recompute this evidence against the immutable initial draft and
    source when applying a repair; generic diagnostic codes are not certificates.
    """
    criterion = requirement.verification
    if (
        criterion is None
        or requirement.source == "assumption"
        or requirement.source_quote not in source_text
        or requirement.metric != "unmeasured"
        or requirement.value is not None
        or requirement.part_id is not None
    ):
        return []
    if attribution is not None:
        linked_facts = {
            fact.id for fact in attribution.facts if requirement.id in fact.requirement_ids
        }
        if any(
            requirement.id in issue.requirement_ids or linked_facts.intersection(issue.fact_ids)
            for issue in attribution.issues
        ):
            return []
    try:
        _source_scope(requirement)
    except ValueError:
        return []
    rule = ""
    reason = ""
    if isinstance(criterion, DigitalDelivery) and _digital_geometry_conflict(requirement):
        rule = "digital_fixed_geometry_conflict_v1"
        reason = (
            "The unchanged complete parent source/text conflicts with digital artifact scope "
            "for every deliverable. A prepared-package-only error is not this certificate."
        )
    elif isinstance(criterion, NominalSubjectInventory):
        quote = requirement.source_quote
        # Customer source fixes identity alternatives. Generated requirement text
        # remains unchanged/hash-bound and supplies the family contradiction;
        # it cannot introduce a new customer naming/count ambiguity.
        if not _has_nominal_cad_scope(source_text) or _INVENTORY_REPAIR_AMBIGUITY.search(quote):
            return []
        try:
            _inventory_source_complete(criterion, quote)
            spans = [_inventory_source_item(item, quote) for item in criterion.items]
            if attribution is not None:
                _requirement_attribution(requirement, attribution)
        except ValueError:
            return []
        # Existing name/count grammar must cover the complete naming prefix.
        # The later geometric remainder is deliberately not treated as inventory.
        naming_end = max(end for _, end in spans)
        prefix = list(quote[:naming_end])
        for start, end in spans:
            prefix[start:end] = " " * (end - start)
        if re.search(r"\w", _inventory_residue("".join(prefix))):
            return []
        if not _GEOMETRY_DUTY.search(_inventory_geometry_text(requirement, criterion)):
            return []
        rule = "inventory_fixed_names_geometry_conflict_v1"
        reason = (
            "Every canonically admissible inventory for the unchanged parent quote must use "
            "these same complete recognized names/counts. Removing those fixed names leaves "
            "the same geometry-family contradiction. This certifies no positive target."
        )
    if not rule:
        return []
    return [
        {
            "kind": criterion.kind,
            "rule": rule,
            "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            "parent_quote_sha256": hashlib.sha256(requirement.source_quote.encode()).hexdigest(),
            "requirement_text_sha256": hashlib.sha256(requirement.text.encode()).hexdigest(),
            "requirement_sha256": criterion_hash(requirement.model_dump(mode="json")),
            "criterion_sha256": criterion_hash(criterion.model_dump(mode="json")),
            "attribution_sha256": criterion_hash(
                {"attribution": attribution.model_dump(mode="json") if attribution else None}
            ),
            "reason": reason,
        }
    ]


def verification_diagnostics(
    contract: Contract,
    source_text: str,
    attribution: IntentAttribution | None,
    *,
    allow_explicit: bool = False,
) -> list[dict[str, Any]]:
    """Collect independent admission errors without editing or proving any duty.

    Provenance, scope, predicate family/targets and attribution run separately.
    Source semantics always inspect the complete containing requirement. A bad
    predicate quote therefore cannot hide a second family error until a retry.
    Dependent parsing within one target rule remains conservative and may stop
    when its prerequisites fail; no missing values are inferred or supplied.
    """
    diagnostics: list[dict[str, Any]] = []
    interpreted_ids: frozenset[str] = frozenset()
    if attribution is not None and attribution.customer_nominal_interpretation is not None:
        from forge_mind.workspace.customer_interpretation import validate_customer_interpretation

        try:
            interpreted_ids = validate_customer_interpretation(contract, source_text, attribution)
        except ValueError as exc:
            diagnostics.append(
                {
                    "requirement_id": None,
                    "field": "attribution.customer_nominal_interpretation",
                    "code": "invalid_customer_nominal_interpretation",
                    "message": str(exc),
                }
            )
    for requirement in contract.requirements:
        if requirement.verification is None:
            continue
        path = f"contract.requirements.{requirement.id}"
        for valid, field, code, message in _source_identity_checks(
            requirement, source_text, allow_explicit
        ):
            if not valid:
                diagnostics.append(
                    {
                        "requirement_id": requirement.id,
                        "field": path + "." + field,
                        "code": code,
                        "message": message,
                    }
                )
        # Only the exact host-bound ordinary rows use the alternate admission path.
        if requirement.id in interpreted_ids:
            continue
        if isinstance(requirement.verification, NominalPhotoSolid):
            try:
                if attribution is None:
                    raise ValueError("Photo nominal proof needs current source attribution")
                from forge_mind.workspace.photo_nominal import validate_photo_nominal_source

                validate_photo_nominal_source(requirement, attribution, source_text, contract)
            except ValueError as exc:
                diagnostics.append(
                    {
                        "requirement_id": requirement.id,
                        "field": path + ".verification",
                        "code": "invalid_photo_nominal_source",
                        "message": str(exc),
                    }
                )
            continue
        checks: list[tuple[str, str, Callable[[], None]]] = [
            ("verification.scope", "unsupported_scope", partial(_source_scope, requirement)),
            (
                "verification.kind",
                "incompatible_predicate_family",
                partial(_source_family, requirement),
            ),
            (
                "verification",
                "unsupported_source_targets",
                partial(_source_targets, requirement, source_text),
            ),
        ]
        if attribution is not None:
            checks.append(
                (
                    "verification.subject_id",
                    "invalid_subject_attribution",
                    partial(_requirement_attribution, requirement, attribution),
                )
            )
        for field, code, check in checks:
            try:
                check()
            except ValueError as exc:
                diagnostics.append(
                    {
                        "requirement_id": requirement.id,
                        "field": path + "." + field,
                        "code": code,
                        "message": str(exc),
                        "supported_kinds": _supported_kinds(),
                    }
                )
    try:
        _overlapping_targets(contract)
    except ValueError as exc:
        diagnostics.append(
            {
                "requirement_id": None,
                "field": "contract.requirements",
                "code": "conflicting_targets",
                "message": str(exc),
            }
        )
    return diagnostics


def validate_verification_contract(
    contract: Contract,
    source_text: str,
    attribution: IntentAttribution | None,
    *,
    allow_explicit: bool = False,
) -> None:
    diagnostics = verification_diagnostics(
        contract, source_text, attribution, allow_explicit=allow_explicit
    )
    if diagnostics:
        raise ValueError(diagnostics[0]["message"])


def _overlapping_targets(contract: Contract) -> None:
    """Reject reliably contradictory closed specifications of the same material."""
    patterns: dict[str, list[NominalPattern]] = {}
    boxes: dict[str, tuple[Any, ...]] = {}
    inventory: tuple[tuple[str, str, int], ...] | None = None
    for requirement in contract.requirements:
        criterion = requirement.verification
        if isinstance(criterion, NominalSubjectInventory):
            current = tuple(
                sorted(
                    (item.subject_id, item.required_name, item.quantity) for item in criterion.items
                )
            )
            if inventory is not None and inventory != current:
                raise ValueError("Conflicting frozen all-made subject inventories")
            inventory = current
        if not isinstance(criterion, NominalPattern):
            continue
        previous = patterns.setdefault(criterion.subject_id, [])
        for prior in previous:
            if prior.axis != criterion.axis or prior.frame != criterion.frame:
                continue  # Do not infer equivalence across distinct measurement domains.
            if len(prior.holes) != len(criterion.holes):
                raise ValueError("Conflicting exact hole counts for the same designed subject")
            for hole in criterion.holes:
                if all(
                    math.dist(hole.center_mm, other.center_mm)
                    > hole.position_tolerance_mm + other.position_tolerance_mm + 1e-9
                    or abs(hole.diameter_mm - other.diameter_mm)
                    > hole.diameter_tolerance_mm + other.diameter_tolerance_mm + 1e-9
                    for other in prior.holes
                ):
                    raise ValueError("Conflicting frozen patterns for the same designed subject")
        previous.append(criterion)
        if isinstance(criterion, NominalPerforatedBox):
            box = (criterion.size_mm, criterion.center_mm)
            if boxes.setdefault(criterion.subject_id, box) != box:
                raise ValueError(
                    "Conflicting frozen rectangular bodies for the same designed subject"
                )
    if inventory is not None and not set(patterns).issubset({row[0] for row in inventory}):
        raise ValueError("All-made inventory omits another designed verification subject")


def _part_artifact(document: DesignDocument, part_id: str) -> str:
    if document.assembly is None:
        if part_id != "part" or document.production is None or document.production.role != "make":
            raise ValueError("Flat verification must explicitly bind the canonical made part")
        if not document.features or len(document.outputs) != 1:
            raise ValueError("Flat verification requires one explicit output")
        features = document.features
        output = document.outputs[0]
        artifact = "part.step"
    else:
        parts = [part for part in document.assembly.parts if part.id == part_id]
        if len(parts) != 1 or parts[0].production.role != "make":
            raise ValueError("Verification binding needs one exact named make definition")
        # Membership needs no CAD import, placement calculation or instance expansion.
        groups = {group.id: group for group in document.assembly.subassemblies}
        definition_ids = {part.id for part in document.assembly.parts}
        pending = [item.definition for item in document.assembly.roots]
        visited: set[str] = set()
        installed: set[str] = set()
        while pending:
            identifier = pending.pop()
            if identifier in definition_ids:
                installed.add(identifier)
            elif identifier in groups and identifier not in visited:
                visited.add(identifier)
                pending.extend(item.definition for item in groups[identifier].children)
            elif identifier not in groups:
                raise ValueError("Verification assembly contains an unknown definition")
        if part_id not in installed:
            raise ValueError("Verification definition has no installed occurrence")
        features = parts[0].features
        output = parts[0].output
        artifact = f"definition_{part_id}.step"
    # Only output ancestry establishes authored material. Unused custom Features
    # cannot turn a transformed import/catalog output into a designed part.
    lookup = {feature.id: feature for feature in features}
    seen: set[str] = set()
    feature = lookup.get(output)
    while feature is not None and not feature.suppressed and feature.kind == "transform":
        if feature.id in seen or len(feature.inputs) != 1:
            raise ValueError("Verification output ancestry is unavailable or ambiguous")
        seen.add(feature.id)
        feature = lookup.get(feature.inputs[0])
    if feature is None or feature.suppressed or feature.kind in {"import", "hardware"}:
        raise ValueError(
            "An import-only or reference definition cannot satisfy a made-subject binding"
        )
    return artifact


def verification_binding_subject_ids(contract: Contract) -> set[str]:
    """Return existing typed binding identities; attribution alone adds no eligibility."""
    subjects: set[str] = set()
    for requirement in contract.requirements:
        criterion = requirement.verification
        if isinstance(criterion, (NominalPattern, NominalPhotoSolid)):
            subjects.add(criterion.subject_id)
        elif isinstance(criterion, NominalSubjectInventory):
            subjects.update(item.subject_id for item in criterion.items)
    return subjects


def validate_verification_bindings(
    plan: DesignPlan,
    contract: Contract,
    document: DesignDocument | None = None,
) -> None:
    if document is not None:
        from forge_mind.workspace.document import DesignDocument

        document = DesignDocument.model_validate(document)
    subjects = verification_binding_subject_ids(contract)
    seen_subjects: set[str] = set()
    seen_parts: set[str] = set()
    for binding in plan.verification_bindings:
        if binding.subject_id not in subjects:
            raise ValueError("Verification binding references an unknown frozen subject")
        if binding.subject_id in seen_subjects or binding.part_id in seen_parts:
            raise ValueError("Verification subject bindings must be one to one")
        seen_subjects.add(binding.subject_id)
        seen_parts.add(binding.part_id)
        if document is not None:
            _part_artifact(document, binding.part_id)
        for requirement in contract.requirements:
            criterion = requirement.verification
            if (
                not isinstance(criterion, NominalPerforatedBox)
                or criterion.subject_id != binding.subject_id
            ):
                continue
            for scalar in contract.requirements:
                if scalar.part_id != binding.part_id or scalar.value is None:
                    continue
                if scalar.metric in {"extent_x", "extent_y", "extent_z"}:
                    expected = criterion.size_mm["xyz".index(scalar.metric[-1])]
                elif scalar.metric == "solid_count":
                    expected = 1.0
                else:
                    continue
                allowed = {
                    "eq": abs(expected - scalar.value) <= scalar.tolerance + 1e-6,
                    "ge": expected >= scalar.value - scalar.tolerance - 1e-6,
                    "le": expected <= scalar.value + scalar.tolerance + 1e-6,
                }[scalar.comparator]
                if not allowed:
                    raise ValueError(
                        "Frozen named scalar target conflicts with its rectangular body"
                    )


def resolve_verification_subject(
    requirement: Requirement,
    plan: DesignPlan,
    document: DesignDocument,
) -> dict[str, Any]:
    from forge_mind.workspace.document import DesignDocument

    document = DesignDocument.model_validate(document)
    criterion = requirement.verification
    if not isinstance(criterion, (NominalPattern, NominalPhotoSolid)):
        raise ValueError("Requirement has no nominal geometry subject")
    return _resolve_subject(criterion.subject_id, plan, document)


def _resolve_subject(subject_id: str, plan: DesignPlan, document: DesignDocument) -> dict[str, Any]:
    bindings = [
        binding for binding in plan.verification_bindings if binding.subject_id == subject_id
    ]
    if len(bindings) != 1:
        raise ValueError("One explicit verification subject binding is required")
    binding = bindings[0]
    if sum(row.part_id == binding.part_id for row in plan.verification_bindings) != 1:
        raise ValueError("Different source subjects cannot alias one made definition")
    return {
        **binding.model_dump(mode="json"),
        "step_artifact": _part_artifact(document, binding.part_id),
    }


def resolve_verification_inventory(
    requirement: Requirement, plan: DesignPlan, document: DesignDocument
) -> list[dict[str, Any]]:
    """Resolve target-free make bindings; this does not prove names or quantities."""
    from forge_mind.workspace.document import DesignDocument

    document = DesignDocument.model_validate(document)
    criterion = requirement.verification
    if not isinstance(criterion, NominalSubjectInventory):
        raise ValueError("Requirement has no nominal subject inventory")
    return [_resolve_subject(item.subject_id, plan, document) for item in criterion.items]
