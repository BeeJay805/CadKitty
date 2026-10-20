"""Closed owner-clause authority for prospective labelled assembly envelope bounds.

This is not an English quantity parser. Unknown qualifiers stay unmeasured;
role labels and a quote-wide bag of numbers cannot establish an envelope target.
Supported source begins with one affirmative designed-assembly introduction,
then the envelope command. Unknown preceding prose stays unsupported regardless
of wording or paragraph breaks; independent later duties remain intact.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from forge_mind.workspace.intent_attribution import IntentAttribution
from forge_mind.workspace.models import Requirement

_UNIT = r"mm|cm|inches|inch|in"
_SCALES = {"mm": 1.0, "cm": 10.0, "in": 25.4, "inch": 25.4, "inches": 25.4}
_ENTRY = r"(?P<a{n}>[xyz])\s*(?:[:=]\s*)?(?P<v{n}>\d+(?:\.\d+)?)\s*(?P<u{n}>" + _UNIT + r")?"
_BOUND = re.compile(
    r"Keep\s+the\s+(?:assembled|overall|assembly)\s+envelope\s+"
    r"(?:within|at\s+most|no\s+more\s+than|to\s+a\s+maximum\s+of)\s+"
    + _ENTRY.replace("{n}", "1")
    + r"\s*,\s*"
    + _ENTRY.replace("{n}", "2")
    + r"\s*,\s*(?:and\s+)?"
    + _ENTRY.replace("{n}", "3"),
    re.I,
)
_BREAK = re.compile(r"(?<!\d)[.!?;\n]|[.!?;\n](?!\d)")
_NOUN = r"[a-z]+(?:[ -]+[a-z]+)*"
_DESIGN = re.compile(
    r"(?:Design|Make|Build|Create)\s+(?:a|an|the)\s+(?:"
    + _NOUN
    + r"\s+)?assembly(?:\s+with\s+"
    + _NOUN
    + r"(?:,\s*"
    + _NOUN
    + r")*(?:\s+and\s+"
    + _NOUN
    + r")?)?",
    re.I,
)
_QUALIFIER = re.compile(
    r"\b(?:no|not|never|unless|except|reference|environment|existing|supplier|example|quoted)\b"
    r"|\b[a-z]+n['\u2019]t\b",
    re.I,
)
_INSTRUCTION_SCOPE = re.compile(
    r"\b(?:next|following|instructions?|clauses?|sentences?|commands?|requests?|quotes?|quotation)\b",
    re.I,
)
_PREDICATION = re.compile(
    r"\b(?:is|are|was|were|be|being|if|then|because|since|instead|as|must|may|will|shall|"
    r"should|could|would|can)\b",
    re.I,
)
_ORIENTATION = re.compile(
    r"[a-z]+(?:[ -]+[a-z]+){0,8}\s+"
    r"(?:in\s+a\s+row\s+along|(?:axis|axes)\s+along|parallel\s+to|perpendicular\s+to|"
    r"aligned\s+(?:with|along)|facing|along)\s+(?:(?:positive|negative)\s+)?[xyz]",
    re.I,
)
_BOUND_MODIFIER = re.compile(
    r"\b(?:envelope|bound|bounds|limit|limits|maximum|minimum|within|beyond|exceed|above|below|"
    r"outside|larger|smaller|tolerance|allowance)\b",
    re.I,
)


@dataclass(frozen=True)
class EnvelopeGrounding:
    state: Literal["not_applicable", "supported", "unsupported"]
    code: str
    source_span: tuple[int, int] | None = None
    bounds_mm: tuple[tuple[str, float], ...] = ()


def _clauses(source: str) -> list[tuple[int, int, str]]:
    spans = []
    start = 0
    for boundary in [*_BREAK.finditer(source), None]:
        end = boundary.start() if boundary is not None else len(source)
        raw = source[start:end]
        if raw.strip():
            left = start + len(raw) - len(raw.lstrip())
            right = end - len(raw) + len(raw.rstrip())
            spans.append((left, right, source[left:right]))
        start = boundary.end() if boundary is not None else len(source)
    return spans


def _family(text: str) -> bool:
    return bool(re.search(r"\benvelope\b", text, re.I) and re.search(r"\b[xyz]\b", text, re.I))


def _qualified_prefix(source: str, start: int) -> bool:
    prefix = source[:start].rstrip()
    # Closed prelude, not English scope inference: unknown extra clauses may
    # qualify the next command. No sentence or paragraph break clears them.
    if (
        _QUALIFIER.search(prefix)
        or _INSTRUCTION_SCOPE.search(prefix)
        or prefix.endswith(":")
        or prefix.count('"') % 2
    ):
        return True
    if len(re.findall(r"(?<!\w)'|'(?!\w)", prefix)) % 2:
        return True
    if any(
        prefix.rfind(left) > prefix.rfind(right)
        for left, right in (("\u201c", "\u201d"), ("\u2018", "\u2019"))
    ):
        return True
    if not prefix:
        return False
    clauses = _clauses(prefix)
    return not (
        len(clauses) == 1
        and _DESIGN.fullmatch(clauses[0][2])
        and not _PREDICATION.search(clauses[0][2])
    )


def _orientation_suffix(suffix: str) -> bool:
    """Separate nonnumeric orientation syntax without granting orientation proof."""
    suffix = suffix.strip()
    if not suffix:
        return True
    if not re.match(r",\s*with\s+", suffix, re.I):
        return False
    if _QUALIFIER.search(suffix) or _BOUND_MODIFIER.search(suffix):
        return False
    text = re.sub(r"^,\s*with\s+", "", suffix, flags=re.I)
    return all(_ORIENTATION.fullmatch(part) for part in re.split(r"\s+and\s+", text, flags=re.I))


def _designed_subject(
    requirement: Requirement,
    source: str,
    clauses: list[tuple[int, int, str]],
    attribution: IntentAttribution | None,
) -> str | None:
    if not isinstance(attribution, IntentAttribution):
        return None
    ids = attribution.requirement_subjects.get(requirement.id, [])
    subjects = [subject for subject in attribution.subjects if subject.id in ids]
    if len(ids) != 1 or len(subjects) != 1:
        return None
    subject = subjects[0]
    if (
        subject.role != "designed"
        or not re.search(r"\bassembly\b", subject.label, re.I)
        or _QUALIFIER.search(subject.label)
        or subject.source_quote not in source
    ):
        return None
    if any(requirement.id in issue.requirement_ids for issue in attribution.issues):
        return None
    # The complete original clause, not a stripped affirmative substring, must
    # corroborate design intent. An authored role label alone is insufficient.
    if not any(
        text in subject.source_quote
        and _DESIGN.fullmatch(text)
        and not _PREDICATION.search(text)
        and not _QUALIFIER.search(text)
        and not _qualified_prefix(source, start)
        for start, _, text in clauses
    ):
        return None
    return subject.id


def ground_source_envelope(
    requirement: Requirement, source: str, attribution: IntentAttribution | None
) -> EnvelopeGrounding:
    """Keep unrecognized scalar families unchanged; refuse unsafe envelope fallback."""
    if requirement.metric == "unmeasured" or requirement.value is None:
        return EnvelopeGrounding("not_applicable", "not_numeric_envelope")
    clauses = _clauses(source)
    governing = [
        (start, end, text)
        for start, end, text in clauses
        if _family(text) and (requirement.source_quote in text or text in requirement.source_quote)
    ]
    if not governing and not _family(requirement.source_quote):
        return EnvelopeGrounding("not_applicable", "not_envelope_family")

    def refuse(code: str) -> EnvelopeGrounding:
        return EnvelopeGrounding("unsupported", code)

    if requirement.source_quote not in source:
        return refuse("envelope_quote_not_in_source")
    family = [(start, end, text) for start, end, text in clauses if _family(text)]
    if len(governing) != 1 or len(family) != 1:
        return refuse("envelope_ambiguous_clauses")
    start, _, text = governing[0]
    if _qualified_prefix(source, start):
        return refuse("envelope_qualified_source")
    match = _BOUND.match(text)
    if match is None:
        return refuse("envelope_unsupported_clause")
    if not _orientation_suffix(text[match.end() :]):
        return refuse("envelope_unsupported_suffix")
    units = [match.group(f"u{index}") for index in (1, 2, 3)]
    if units[-1] is None:
        return refuse("envelope_explicit_unit_required")
    scale = _SCALES[units[-1].lower()]
    if any(unit is not None and _SCALES[unit.lower()] != scale for unit in units):
        return refuse("envelope_mixed_units")
    axes = [match.group(f"a{index}").lower() for index in (1, 2, 3)]
    if set(axes) != {"x", "y", "z"}:
        return refuse("envelope_duplicate_or_missing_axis")
    values = [float(match.group(f"v{index}")) * scale for index in (1, 2, 3)]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        return refuse("envelope_invalid_quantity")
    bounds = dict(zip(axes, values, strict=True))
    core = text[: match.end("u3")]
    if core not in requirement.source_quote:
        return refuse("envelope_incomplete_quote")
    subject = _designed_subject(requirement, source, clauses, attribution)
    if subject is None:
        return refuse("envelope_designed_assembly_required")
    if (
        requirement.source != "request"
        or requirement.metric not in {"extent_x", "extent_y", "extent_z"}
        or requirement.comparator != "le"
        or requirement.part_id is not None
        or requirement.center is not None
        or requirement.tolerance != 0
    ):
        return refuse("envelope_target_scope")
    expected = bounds[requirement.metric[-1]]
    if not math.isclose(requirement.value, expected, rel_tol=0, abs_tol=1e-6):
        return refuse("envelope_axis_value_mismatch")
    assert attribution is not None
    facts = [fact for fact in attribution.facts if requirement.id in fact.requirement_ids]
    if not facts or any(
        fact.subject_id != subject
        or fact.candidate_subject_ids
        or core not in fact.source_quote
        or fact.source_quote not in source
        or fact.unit.lower() not in _SCALES
        or _SCALES.get(fact.unit.lower()) != scale
        or not math.isclose(
            fact.value * _SCALES.get(fact.unit.lower(), 0), expected, rel_tol=0, abs_tol=1e-6
        )
        for fact in facts
    ):
        return refuse("envelope_fact_corroboration")
    return EnvelopeGrounding(
        "supported",
        "envelope_bound",
        (start, start + len(core)),
        tuple((axis, bounds[axis]) for axis in "xyz"),
    )
