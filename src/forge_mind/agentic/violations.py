"""Structured violations (ADR-0119): the validator's prose issues, parsed into typed
objects so search harnesses, the reward endpoint, and training-data miners consume
FIELDS instead of re-regexing prose downstream.

READ-SIDE ONLY by design: the issue strings stay the author-facing contract (prompts
byte-identical), and this module structures them after the fact. The formats parsed
here are our OWN emitters' (validate.py, mapped_dims.py, feature_presence.py, the
material wall gate); any line that matches nothing degrades to ``kind="other"`` with
the raw message preserved -- parsing never drops a violation (fail-open).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Violation:
    """One structured validator complaint. ``message`` is always the verbatim prose."""

    kind: str  # min_wall | overhang | envelope | mapped_position | presence | count_pattern | other
    message: str
    measured: float | None = None  # the offending number, in ``unit``
    required: float | None = None  # the limit it broke
    unit: str | None = None
    location: tuple[float, float, float] | None = None  # centroid when the emitter located it
    feature: str | None = None  # the brief feature named, when one was


_MIN_WALL = re.compile(
    r"[Tt]hinnest wall (\d+(?:\.\d+)?)\s*mm is below (?:the required|PLA's|\w+'s) (\d+(?:\.\d+)?)"
)
_THIN_LOC = re.compile(
    r"THIN-WALL location:.*centroid (-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)"
)
_OVERHANG = re.compile(r"(\d+) unsupported overhang region\(s\), total (\d+(?:\.\d+)?) mm\^2")
_ENVELOPE = re.compile(r"ENVELOPE: built ([\d.]+) x ([\d.]+) x ([\d.]+) mm vs the brief's")
_MAPPED = re.compile(
    r"MAPPED POSITION.*?'([^']+)'.*?(\d+(?:\.\d+)?)\s*mm vs the stated (\d+(?:\.\d+)?)"
)
_MISSING = re.compile(r"still MISSING (\d+) x [^(]*\('([^']+)'\)")
_COUNT = re.compile(r"DECLARED COUNT NOT AUTHORED: the brief declares (\d+)x '([^']+)'")
_WALL_GATE = re.compile(r"WALL GATE: (\d+(?:\.\d+)?)% .* thinner than the (\d+(?:\.\d+)?) mm floor")


def _parse_one(line: str) -> Violation:
    m = _MIN_WALL.search(line)
    if m:
        return Violation(
            "min_wall", line, measured=float(m.group(1)), required=float(m.group(2)), unit="mm"
        )
    m = _THIN_LOC.search(line)
    if m:
        return Violation(
            "min_wall", line, location=(float(m.group(1)), float(m.group(2)), float(m.group(3)))
        )
    m = _WALL_GATE.search(line)
    if m:
        return Violation(
            "min_wall", line, measured=float(m.group(1)), required=float(m.group(2)), unit="%area"
        )
    m = _OVERHANG.search(line)
    if m:
        return Violation("overhang", line, measured=float(m.group(2)), unit="mm^2")
    m = _ENVELOPE.search(line)
    if m:
        return Violation("envelope", line)
    m = _MAPPED.search(line)
    if m:
        return Violation(
            "mapped_position",
            line,
            measured=float(m.group(2)),
            required=float(m.group(3)),
            unit="mm",
            feature=m.group(1),
        )
    m = _MISSING.search(line)
    if m:
        return Violation("presence", line, measured=float(m.group(1)), feature=m.group(2))
    m = _COUNT.search(line)
    if m:
        return Violation("count_pattern", line, required=float(m.group(1)), feature=m.group(2))
    if "ADDITIVE FEATURES AUTHORED AS CUTS" in line:
        return Violation("presence", line)
    if "MEASURED EXCESS MASSING" in line or "MASSING ADVISORY" in line:
        return Violation("presence", line)
    # Run-1 mining surfaced these emitters unclassified (54 'other' entries): the
    # hole-count and bore-size audits, and the coherence core.
    if "MISSING FEATURES" in line or "MEASURED BORES vs PLAN" in line:
        return Violation("presence", line)
    if "SIZE MISMATCH" in line:
        return Violation("presence", line)
    if "watertight" in line.lower() or "single connected solid" in line.lower():
        return Violation("coherence", line)
    return Violation("other", line)


def parse_violations(issues: Sequence[str]) -> list[Violation]:
    """One Violation per issue line, verbatim prose preserved, unknowns kept as 'other'."""
    return [_parse_one(line) for line in issues if line and line.strip()]


__all__ = ["Violation", "parse_violations"]
