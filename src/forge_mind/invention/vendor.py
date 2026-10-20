"""The vendor-research gate (ADR-0048) — real-parts BOM with a fail-closed `listed` tier.

The single highest false-PASS hazard the pivot opens: a buyable BOM line carries a BUY BUTTON, so a
hallucinated-but-plausible SKU is a no-false-PASS breach on a surface users trust MORE than geometry.
This gate is built GATE-FIRST (the council's "gates before generators"): a SKU enters the BOM ONLY
when an exact, fetched listing confirms it; anything unconfirmed is FLAGGED, never guessed into a line
— exactly how ``kit/bom.py`` refuses to invent a screw length and ``provenance.py`` refuses to
promote a tier the model authored.

The discipline mirrors ADR-0040's anchor invariant, applied to a NEW axis:

  - The model NOMINATES a :class:`PartRequirement` (a designation it needs bought). It never
    authors the SKU/price/URL — those come only from a fetched :class:`VendorListing`, an artifact
    the model did not write this run.
  - Admission is EXACT (normalized designation equality, no fuzzy/substring matching — fuzz is how
    laundering starts). A listing for a DIFFERENT part is not a match; an empty result is not a
    match. Either way the requirement is FLAGGED.
  - The admitted line's provenance is :data:`VendorProvenance.LISTED` — a SEPARATE axis from the
    dimensional ``Tier``. A vendor offer is volatile (price/stock change), so it can NEVER equal the
    registry's ``verified`` tier. There is structurally no code path from LISTED to verified.

The ``fetcher`` is injectable (a :class:`VendorFetcher`); the default :class:`NullFetcher` confirms
nothing, so with no real catalog wired EVERY requirement FLAGs (fail closed — no fake parts). The
real networked fetcher (a sibling of ``agentic/step_parts.py``) is the deferred live piece; the GATE
that keeps it honest is built and tested now.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class VendorProvenance(StrEnum):
    """A vendor line's provenance — a SEPARATE axis from the dimensional ``Tier`` (ADR-0040). There
    is no ``verified`` here: a fetched offer is volatile and can never be promoted to the registry's
    verified tier. Only two states exist: an exact-confirmed offer, or an unconfirmed requirement."""

    LISTED = "listed"  # an exact fetch-confirmed real offer (SKU + price + URL). NEVER == verified.
    UNCONFIRMED = "unconfirmed"  # searched but no exact match -> FLAGGED, never a line.


class PartRequirement(BaseModel):
    """A part the design NEEDS bought (not printed). The model nominates the ``designation``; the
    gate confirms it against a real listing. The ``query`` is what the fetcher searches."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    designation: str = Field(..., description="The canonical spec needed, e.g. 'M3x12 SHCS'.")
    category: str = Field(..., description="Display role: screw / bearing / insert / ... .")
    query: str = Field(..., description="The vendor search string (the model may author this).")


class VendorListing(BaseModel):
    """One real search result from a vendor catalog — the artifact the model did not author. The
    fetcher parses each result into its canonical ``designation`` so the gate can exact-match it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    designation: str = Field(..., description="The canonical spec this listing IS (for matching).")
    sku: str
    title: str
    url: str
    vendor: str
    price_usd: float | None = None


class VendorFetcher(Protocol):
    """The injectable catalog seam. A real implementation hits a vendor API (deferred live work,
    a sibling of ``step_parts.py``); tests pass a stub; the default confirms nothing."""

    def search(self, query: str) -> list[VendorListing]: ...


class NullFetcher:
    """The fail-closed default: confirms nothing, so every requirement FLAGs. With no real catalog
    wired, the gate can never emit a fake part — absence of a fetcher means absence of a SKU."""

    def search(self, query: str) -> list[VendorListing]:
        return []


class VendorLine(BaseModel):
    """An ADMITTED buyable line: a fetched listing that exactly matched a requirement. Always
    :data:`VendorProvenance.LISTED` — never the dimensional ``verified`` tier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    designation: str
    category: str
    listing: VendorListing
    provenance: VendorProvenance = VendorProvenance.LISTED


class VendorBom(BaseModel):
    """The vendor BOM: confirmed buyable lines + the requirements the gate could not confirm
    (FLAGGED, never guessed into a line — the no-false-PASS record)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lines: list[VendorLine] = Field(default_factory=list)
    flagged: list[str] = Field(
        default_factory=list,
        description="Requirements with no exact confirmed listing — FLAGGED, never a line.",
    )

    @property
    def all_confirmed(self) -> bool:
        """True iff every requirement resolved to a real listing (nothing flagged)."""
        return not self.flagged


def _normalize(designation: str) -> str:
    """Normalize a designation for EXACT matching: lowercase, collapse whitespace, strip. No fuzzy
    or substring logic — an exact normalized equality is the only admission test (conservative: a
    real match whose strings differ is FLAGGED, but a wrong part is NEVER admitted)."""
    return re.sub(r"\s+", " ", designation.strip().lower())


def _confirm(requirement: PartRequirement, listings: list[VendorListing]) -> VendorListing | None:
    """Return the FIRST listing whose canonical designation EXACTLY matches the requirement, or None.
    The match is on the fetcher-parsed ``designation``, never the free-text title — the gate does not
    trust the fetcher to have searched correctly; it CHECKS what came back."""
    want = _normalize(requirement.designation)
    for listing in listings:
        if _normalize(listing.designation) == want:
            return listing
    return None


def research_vendor_parts(
    requirements: list[PartRequirement],
    fetcher: VendorFetcher | None = None,
) -> VendorBom:
    """Resolve each required bought part against a real catalog, confirm-or-FLAG.

    A requirement becomes a :class:`VendorLine` (provenance LISTED) ONLY when the fetcher returns a
    listing whose canonical designation exactly matches it; otherwise it is FLAGGED. Deterministic,
    REPORTED, never gates. The default :class:`NullFetcher` confirms nothing (fail closed)."""
    fetch = fetcher or NullFetcher()
    lines: list[VendorLine] = []
    flagged: list[str] = []
    for requirement in requirements:
        listing = _confirm(requirement, fetch.search(requirement.query))
        if listing is None:
            flagged.append(requirement.designation)
            continue
        lines.append(
            VendorLine(
                designation=requirement.designation,
                category=requirement.category,
                listing=listing,
            )
        )
    return VendorBom(lines=lines, flagged=flagged)


__all__ = [
    "NullFetcher",
    "PartRequirement",
    "VendorBom",
    "VendorFetcher",
    "VendorLine",
    "VendorListing",
    "VendorProvenance",
    "research_vendor_parts",
]
