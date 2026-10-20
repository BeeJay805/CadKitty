"""The requirements compiler's fact layer (ADR-0087 S1) — provenance from geometry to KNOWLEDGE.

A problem ("I need one more SSD in this case") becomes a build only through a chain of FACTS: the
part's own dimensions, the payload's standard envelope, the environment's researched clearances, the
user's stated constraints. Every one of those numbers must carry WHERE it came from — the same
no-false-PASS discipline the validator applies to geometry, applied to knowledge.

This module is the ledger those facts live in. It does NOT research, ask, or compile a spec yet
(S3/S4); it defines:

  * :class:`Fact` — one number with its provenance. ``source`` is the honest origin
    (``user`` / ``registry`` / ``researched`` / ``derived``); a non-user fact must cite. Values are
    kept verbatim AND exposed normalized to mm/g by DETERMINISTIC code (never a model conversion —
    the inches->mm class of error, ADR-0087 P4).
  * :class:`FactLedger` — the collected facts, with conflict detection (two facts, one name,
    different measured value — ADR-0087 P10) and serialization into the EXISTING stated-facts grammar
    (``- dimension: <name> = <value> <unit> (source: <source>)``, ADR-0040) so the engine's brief
    research binds to them through the proven ``produce_part(extra_context=...)`` seam. A researched
    fact round-trips to the SOURCED tier and can NEVER surface as STATED (the anchor invariant).

Derived facts (arithmetic on other facts) are held on the ledger with their working shown, but are
NOT emitted into the anchor block — they are computed targets, surfaced in the compiled prompt (S4),
not claims about what the user or the world stated.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

FactSource = Literal["user", "registry", "researched", "derived"]

# The Fact source -> the ADR-0040 grammar `source:` token (which anchors a tier via
# provenance._INTAKE_SOURCE_TIER). `derived` has no grammar token: it never enters the anchor block.
_SOURCE_TO_GRAMMAR: dict[FactSource, str] = {
    "user": "user",  # -> STATED
    "registry": "standard",  # a registry row is a cited standard -> SOURCED
    "researched": "researched",  # a cited web finding -> SOURCED (verify-required)
}

# Deterministic unit normalization to the engine's mm/g domain. Only KNOWN units convert; an unknown
# unit (deg, count, ...) passes through unchanged -- we never guess a conversion (P4 is a code
# guarantee, not a model behaviour).
_LENGTH_TO_MM: dict[str, float] = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    '"': 25.4,
}
_MASS_TO_G: dict[str, float] = {
    "g": 1.0,
    "kg": 1000.0,
    "oz": 28.349523125,
    "lb": 453.59237,
    "lbs": 453.59237,
}


def normalize_measure(value: float, unit: str) -> tuple[float, str]:
    """(value, unit) -> (value, canonical unit) in mm/g. Deterministic; unknown units pass through.

    Rounded to 4 decimals so 2.5 in -> 63.5 mm is exact, not 63.50000000001."""
    u = unit.strip().lower()
    if u in _LENGTH_TO_MM:
        return round(value * _LENGTH_TO_MM[u], 4), "mm"
    if u in _MASS_TO_G:
        return round(value * _MASS_TO_G[u], 4), "g"
    return value, unit.strip()


def _fmt(value: float) -> str:
    """Grammar-safe number: no trailing .0, no scientific notation (the regex wants \\d+(\\.\\d+)?)."""
    return f"{value:.4f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))


class Fact(BaseModel):
    """One measured requirement with its provenance. ``value``/``unit`` are kept VERBATIM (the card
    shows '4 in -> 101.6 mm'); ``as_mm_or_g`` is the deterministic normalization the engine reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: float
    unit: str
    source: FactSource
    # A non-user fact must cite its anchor: a registry key, a URL, or (derived) the arithmetic. The
    # fail-closed discipline at the type level -- a researched number with no citation cannot exist.
    citation: str | None = None
    # For researched facts: the fetched span the value was read from. S3 verifies value ∈ quote
    # before the fact is admitted; carried here for the audit trail and the card.
    quote: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _must_cite(self) -> Fact:
        if (
            self.source in ("registry", "researched", "derived")
            and not (self.citation or "").strip()
        ):
            raise ValueError(f"a {self.source} fact must carry a citation (fact: {self.name!r})")
        return self

    @property
    def as_mm_or_g(self) -> tuple[float, str]:
        return normalize_measure(self.value, self.unit)

    def to_grammar_line(self) -> str | None:
        """The ADR-0040 stated-facts line, or None for a ``derived`` fact (not an anchor)."""
        token = _SOURCE_TO_GRAMMAR.get(self.source)
        if token is None:
            return None
        v, u = self.as_mm_or_g
        return f"- dimension: {self.name} = {_fmt(v)} {u} (source: {token})"


class FactConflict(BaseModel):
    """Two facts naming the same requirement with different measured values (ADR-0087 P10). Never
    silently averaged -- surfaced to become a clarifying question or an open assumption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    a: Fact
    b: Fact


class FactLedger(BaseModel):
    """The collected facts for one problem, provenance intact. Frozen; build the whole list then
    construct (the compiler assembles registry + research + clarify facts, then reasons over them)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    facts: tuple[Fact, ...] = ()

    def anchors(self) -> list[Fact]:
        """Facts that enter the stated-facts block (user/registry/researched) — not derived."""
        return [f for f in self.facts if f.source != "derived"]

    def derived(self) -> list[Fact]:
        return [f for f in self.facts if f.source == "derived"]

    def conflicts(self) -> list[FactConflict]:
        """Same normalized name, different normalized value = a conflict. Compared on the mm/g value
        so '4 in' and '101.6 mm' for one name do NOT read as a conflict."""
        out: list[FactConflict] = []
        for i, a in enumerate(self.facts):
            for b in self.facts[i + 1 :]:
                if a.name.strip().lower() != b.name.strip().lower():
                    continue
                av, au = a.as_mm_or_g
                bv, bu = b.as_mm_or_g
                if au == bu and abs(av - bv) > 1e-6:
                    out.append(FactConflict(name=a.name, a=a, b=b))
        return out

    def to_stated_facts_block(self) -> str:
        """The anchor facts as an ADR-0040 stated-facts block, ready for ``produce_part``'s
        ``extra_context``. ``parse_stated_dimensions`` reads it back at the honest tier per source."""
        lines = [ln for f in self.facts if (ln := f.to_grammar_line()) is not None]
        return "\n".join(lines)


__all__ = ["Fact", "FactConflict", "FactLedger", "FactSource", "normalize_measure"]
