"""Feature-presence gate v1 (ADR-0117): deterministic checks that the AUTHORED geometry
plausibly contains the brief's declared features -- validation only, never generation.

Scoped to the three MEASURED omission mechanisms from the 2026-07-18/19 autopsies:

  G1  volume-vs-declared -- the slab class: a "L-bracket" that fills 94% of its envelope
      (74,960 mm^3) when the brief's own feature dims imply ~21,000. Parses the additive
      features' approx_dims_mm; when the measured solid volume exceeds the declared-material
      estimate by a wide, calibrated factor, that is measured excess massing.
  G2  additive-declared-but-SUBTRACT-only -- the wall_hook class: additive features (hook,
      boss, rib) authored exclusively as Mode.SUBTRACT pockets cut into one primitive.
  G3  declared-count-vs-authored-ops -- the fan_shroud class: a count>=2 additive feature
      (4 mounting bosses) with neither a location pattern nor enough additive ops in code.

FAIL-OPEN BY CONSTRUCTION: any feature this module cannot parse, any measurement it cannot
make, returns UNKNOWN and contributes NOTHING to the verdict (no-false-FAIL discipline; the
no-false-PASS rule is satisfied because the gate only ever blocks MORE, never less). The
2026-07-19 offline replay over every archived shipped run is the calibration evidence; the
thresholds here must keep that replay at ZERO false-FAILs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge_mind.agentic.brief import BriefFeature, DesignBrief

# G1: measured volume must exceed the declared-material estimate by MORE than this factor
# before it flags (the slab L-bracket sits at ~3.5x; calibration must keep every legitimately
# shipped archived part below the line).
_VOLUME_EXCESS_FACTOR = 3.0
# G1 only speaks when the parseable additive features cover a meaningful share of the brief
# (else the estimate is too partial to accuse anyone with).
_MIN_PARSED_FEATURES = 2
# Advisory (non-gating) threshold on the SCALED estimate for partial parses: 1.75x puts the
# measured slab (74,960 vs ~40,000 scaled) over the line while staying advisory-only.
_ADVISORY_EXCESS_FACTOR = 1.75

_NEGATIVE_WORDS = ("hole", "bore", "slot", "pocket", "vent", "cutout", "opening", "recess")
_ADDITIVE_METHODS = ("box", "revolve", "sweep", "loft", "extrude")
_PATTERN_OPS = ("PolarLocations", "GridLocations", "Locations(")
_ADDITIVE_CODE_OPS = (
    "extrude(",
    "revolve(",
    "sweep(",
    "loft(",
    "Cylinder(",
    "Box(",
    "Sphere(",
    "Cone(",
)


def feature_presence_enabled() -> bool:
    """OPT-IN via FORGE_FEATURE_PRESENCE=1 (the repo's lever pattern: default OFF, measured
    at n>=8 on the frozen corpora, promoted only on the ADR-0117 thresholds). Flag off =
    validate is byte-identical."""
    return os.environ.get("FORGE_FEATURE_PRESENCE") == "1"


@dataclass(frozen=True)
class PresenceFinding:
    """One deterministic feature-presence finding. ``gates`` -> counts against ``passed``."""

    check: str  # "volume_excess" | "additive_as_subtract" | "count_pattern"
    message: str
    gates: bool


@dataclass(frozen=True)
class PresenceReport:
    findings: list[PresenceFinding] = field(default_factory=list)
    # Advisory context (scorecard): the parsed expectation, for transparency in autopsies.
    expected_volume_mm3: float | None = None
    measured_volume_mm3: float | None = None

    @property
    def gating(self) -> list[PresenceFinding]:
        return [f for f in self.findings if f.gates]


_NEGATIVE_METHODS = ("pocket", "drill", "bore", "bolt_holes", "thread")

# v2 (ADR-0118): name-positive features are additive BY NAME, whatever the method text says.
# Measured lid class: the local planner declares corner standoffs with method="pocket" and
# the dims of the bore they RECEIVE, so v1 marked all four negative, the additive set
# collapsed to one instance, and G2 never saw the slab-with-carved-holes lid (three archived
# acceptances; only the 32B critic refused it). A name that ALSO carries a negative word
# ("boss_bore") stays negative -- the override never rescues ambiguous names. Offline replay
# 2026-07-21 over 450 archived runs: the only newly-gated accepted runs are those three
# lid-class false-PASSes; zero new gates elsewhere, zero lost findings.
_NAME_POSITIVE = ("boss", "standoff")
_GENERIC_FEATURE_TOKENS = frozenset(
    {
        "diameter",
        "thickness",
        "height",
        "width",
        "length",
        "depth",
        "radius",
        "outer",
        "inner",
        "upper",
        "lower",
        "vertical",
        "horizontal",
    }
)


def feature_name_tokens(name: str) -> list[str]:
    """Informative feature-name tokens shared by evidence consumers."""
    return [
        token
        for token in re.split(r"[\s_/-]+", name.lower())
        if len(token) > 3 and token not in _GENERIC_FEATURE_TOKENS
    ]


def _is_name_positive(f: BriefFeature) -> bool:
    name = f.name.lower()
    return any(w in name for w in _NAME_POSITIVE) and not any(w in name for w in _NEGATIVE_WORDS)


def _is_negative(f: BriefFeature) -> bool:
    """Negative (material-removing) features are judged by NAME and METHOD only. The purpose
    text is deliberately excluded: a boss's purpose legitimately mentions the screw/hole it
    RECEIVES (measured: fan_shroud's mounting_bosses were mis-marked negative through their
    purpose text and escaped every check). Name-positive features (v2) never go negative
    through their method text."""
    if _is_name_positive(f):
        return False
    name = f.name.lower()
    method = (f.method or "").lower()
    return any(w in name for w in _NEGATIVE_WORDS) or any(m in method for m in _NEGATIVE_METHODS)


def _is_additive(f: BriefFeature) -> bool:
    if _is_name_positive(f):
        return True
    method = (f.method or "").lower()
    return not _is_negative(f) and any(m in method for m in _ADDITIVE_METHODS)


_DIMS3 = re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)")  # noqa: RUF001 -- briefs legitimately use the multiplication sign
_DISC = re.compile(r"d\s*(\d+(?:\.\d+)?)\s*(?:[x×]\s*(\d+(?:\.\d+)?))?", re.IGNORECASE)  # noqa: RUF001 -- briefs legitimately use the multiplication sign
_DISC_WORDY = re.compile(
    r"(\d+(?:\.\d+)?)\s*mm\s*diam\w*(?:\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm\s*(?:height|tall|deep|long)\w*)?",  # noqa: RUF001
    re.IGNORECASE,
)


def _feature_volume_mm3(f: BriefFeature) -> float | None:
    """Rough declared volume for ONE instance of an additive feature, from approx_dims_mm.
    None (UNKNOWN) whenever the text does not parse cleanly -- never guess."""
    text = f.approx_dims_mm or ""
    m = _DIMS3.search(text)
    if m:
        a, b, c = (float(g) for g in m.groups())
        return a * b * c
    for pattern in (_DISC, _DISC_WORDY):
        m = pattern.search(text)
        if m and m.group(2):
            d, h = float(m.group(1)), float(m.group(2))
            return 3.14159 * (d / 2.0) ** 2 * h
    return None


def feature_was_measured(feature: BriefFeature) -> bool:
    """Whether an existing deterministic audit observed this declared feature.

    Through-hole features participate in the measured bore/count gates. Additive
    features participate only when their declared volume parsed. Negative features
    such as slots, pockets, vents, and recesses currently have no presence instrument,
    so they must never be described as measurement-refuted.
    """
    name = str(getattr(feature, "name", "")).lower()
    if "hole" in name or "bore" in name:
        return True
    if any(word in name for word in _NEGATIVE_WORDS):
        return False
    return _is_additive(feature) and _feature_volume_mm3(feature) is not None


def _declared_count(f: BriefFeature) -> int:
    c = getattr(f, "count", None)
    return int(c) if isinstance(c, int) and c >= 1 else 1


def check_volume_excess(brief: DesignBrief, measured_volume_mm3: float) -> PresenceReport:
    """G1: is the measured solid FAR heavier than the brief's own additive features imply?"""
    additive = [f for f in brief.features if _is_additive(f)]
    vols = [(_feature_volume_mm3(f), _declared_count(f)) for f in additive]
    parsed = [(v, n) for v, n in vols if v is not None]
    if len(parsed) < _MIN_PARSED_FEATURES or len(parsed) < len(additive):
        # Partial parse -> the estimate cannot GATE (fail-open). But when even the scaled
        # estimate (assume unparsed features are similar scale) is dwarfed, say so as an
        # ADVISORY: the slab class (74,960 mm^3 vs ~21,000 declared) has no clean
        # deterministic separator at v1 (calibration 2026-07-19: legit flat parts fill
        # 98% of bbox), so the author gets the NUMBER and the critic stays the judge.
        if parsed:
            scaled = sum(v * n for v, n in parsed) * (len(additive) / len(parsed))
            if scaled > 0 and measured_volume_mm3 > _ADVISORY_EXCESS_FACTOR * scaled:
                msg = (
                    f"MASSING ADVISORY (does not block): the solid measures "
                    f"{measured_volume_mm3:.0f} mm^3 while the brief's parseable features "
                    f"suggest roughly ~{scaled:.0f} mm^3 of declared material. If the part "
                    "reads as a filled block, rebuild the declared features as their own "
                    "solids instead of filling the envelope."
                )
                return PresenceReport(
                    findings=[PresenceFinding("volume_excess_advisory", msg, gates=False)],
                    expected_volume_mm3=scaled,
                    measured_volume_mm3=measured_volume_mm3,
                )
        return PresenceReport(measured_volume_mm3=measured_volume_mm3)
    expected = sum(v * n for v, n in parsed)
    if expected <= 0:
        return PresenceReport(measured_volume_mm3=measured_volume_mm3)
    if measured_volume_mm3 > _VOLUME_EXCESS_FACTOR * expected:
        msg = (
            f"MEASURED EXCESS MASSING: the solid is {measured_volume_mm3:.0f} mm^3 but the "
            f"brief's own declared features add to ~{expected:.0f} mm^3 -- the part is "
            f"{measured_volume_mm3 / expected:.1f}x the declared material. The silhouette is "
            "almost certainly a filled block instead of the declared shapes: build each "
            "declared feature as its OWN solid with the stated dims, fuse, and REMOVE the "
            "filler mass."
        )
        return PresenceReport(
            findings=[PresenceFinding("volume_excess", msg, gates=True)],
            expected_volume_mm3=expected,
            measured_volume_mm3=measured_volume_mm3,
        )
    return PresenceReport(expected_volume_mm3=expected, measured_volume_mm3=measured_volume_mm3)


def _additive_op_lines(code: str) -> int:
    """Count code LINES that perform an additive solid operation. Line-level on purpose: the
    measured wall_hook authored its 'hook' as `extrude(amount=-15, mode=Mode.SUBTRACT)` --
    an extrude call that REMOVES material must not count as building a feature."""
    n = 0
    for line in code.splitlines():
        if "SUBTRACT" in line:
            continue
        if any(op in line for op in _ADDITIVE_CODE_OPS):
            n += 1
    return n


def check_additive_as_subtract(brief: DesignBrief, code: str) -> PresenceReport:
    """G2: additive features declared, but after the FIRST primitive the code only ever cuts.

    Fires ONLY when the brief declares >= 2 additive feature INSTANCES: a part whose sole
    additive feature is its own body (a plate) is CORRECTLY authored as one primitive plus
    cuts (calibration 2026-07-19: the shipped nema17_plate false-FAILed the first draft of
    this check exactly that way)."""
    additive = [f for f in brief.features if _is_additive(f)]
    declared_instances = sum(_declared_count(f) for f in additive)
    if declared_instances < 2 or not code:
        return PresenceReport()
    add_ops = _additive_op_lines(code)
    # one additive op = the base primitive; everything else is Mode.SUBTRACT carving
    if add_ops <= 1 and "Mode.SUBTRACT" in code:
        names = ", ".join(f.name for f in additive)
        msg = (
            f"ADDITIVE FEATURES AUTHORED AS CUTS: the brief declares added material "
            f"({names}) but the script builds ONE primitive and only ever subtracts from "
            "it. A hook / boss / rib is ADDED solid -- construct each as its own solid "
            "(extrude/revolve/sweep) and fuse it on; do not carve dents and call them "
            "features."
        )
        return PresenceReport(findings=[PresenceFinding("additive_as_subtract", msg, gates=True)])
    return PresenceReport()


def check_count_pattern(brief: DesignBrief, code: str) -> PresenceReport:
    """G3: a count>=2 additive feature with no location pattern and too few additive ops."""
    if not code:
        return PresenceReport()
    findings: list[PresenceFinding] = []
    for f in brief.features:
        n = _declared_count(f)
        if n < 2 or not _is_additive(f):
            continue
        has_pattern = any(op in code for op in _PATTERN_OPS)
        add_ops = _additive_op_lines(code)
        # base primitive + at least one op per declared instance, OR any pattern op
        if not has_pattern and add_ops < 1 + n:
            msg = (
                f"DECLARED COUNT NOT AUTHORED: the brief declares {n}x '{f.name}' but the "
                f"script has no location pattern (GridLocations/PolarLocations) and too few "
                f"additive operations to build {n} instances. Replicate the feature with a "
                "pattern at the declared positions."
            )
            from forge_mind.agentic.idioms import idiom_pointer

            idiom = idiom_pointer(f.name)
            if idiom:
                msg = f"{msg}\n{idiom}"
            findings.append(PresenceFinding("count_pattern", msg, gates=True))
    return PresenceReport(findings=findings)


def feature_presence_report(
    brief: DesignBrief, measured_volume_mm3: float | None, code: str | None
) -> PresenceReport:
    """All three checks, merged. Every UNKNOWN contributes nothing (fail-open)."""
    findings: list[PresenceFinding] = []
    expected = None
    if measured_volume_mm3 is not None and measured_volume_mm3 > 0:
        r1 = check_volume_excess(brief, measured_volume_mm3)
        findings.extend(r1.findings)
        expected = r1.expected_volume_mm3
    if code:
        findings.extend(check_additive_as_subtract(brief, code).findings)
        findings.extend(check_count_pattern(brief, code).findings)
    return PresenceReport(
        findings=findings,
        expected_volume_mm3=expected,
        measured_volume_mm3=measured_volume_mm3,
    )


__all__ = [
    "PresenceFinding",
    "PresenceReport",
    "check_additive_as_subtract",
    "check_count_pattern",
    "check_volume_excess",
    "feature_name_tokens",
    "feature_presence_report",
    "feature_was_measured",
]
