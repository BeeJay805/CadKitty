"""Per-dimension provenance tiers — earned in code, never self-claimed (ADR-0040).

The reframed moat: the engine may ATTEMPT anything; no-false-PASS gates the CLAIM. Every
agentic part's dimensions therefore carry a provenance tier the user can trust:

    verified  > sourced > stated > estimated

- ``verified``  — code resolved the feature's ``standard`` designation against a
  NON-provisional row of the cited standards registry; the citation is the registry's
  own ``source``. Nothing else earns verified (the council's anchor invariant).
- ``sourced``   — a provisional registry row, or an intake STATED-FACTS dimension whose
  intake source was itself standards/image-derived (a real but weaker anchor).
- ``stated``    — the user's own confirmed number (intake ``source: user``). Authoritative
  for INTENT, validating nothing about the world: display copy is "you told us",
  never "we checked" (the registry-keeper seat's fourth tier).
- ``estimated`` — everything else: the model's engineering reasoning. Carries the
  self-reported ``estimate_basis`` (method + anchor) so an estimate is auditable
  reasoning, not vibes — but the basis NEVER raises the tier.

THE ANCHOR INVARIANT (the law this module exists to enforce): promotion above
``estimated`` requires exact code resolution against an artifact the model did not author
in this run — a git-reviewed registry row or the user-confirmed intake block. The LLM
nominates designations; it never confirms tiers.

The ledger (``FORGE_PROVENANCE_LEDGER`` env, append-only JSONL) records DEMAND (a
designation the registry lacked — the curation priority signal) and CANDIDATE facts
(intake-sourced dims with citations). It is runtime-write-only and runtime-read-NEVER;
the only path into the registry is a human git review. Best-effort: a ledger miss never
affects a build.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.agentic.brief import DesignBrief
from forge_mind.dsl.standards import ALL_REGISTRIES


class Tier(StrEnum):
    """Dimension provenance, strongest first. Order matters: labels take the WEAKEST present."""

    VERIFIED = "verified"
    SOURCED = "sourced"
    STATED = "stated"
    ESTIMATED = "estimated"


# How a record EARNED its tier — replayable in an audit (the registry-keeper's mechanism enum).
Mechanism = Literal[
    "registry_resolution",  # non-provisional registry row -> verified
    "provisional_registry",  # provisional registry row -> sourced
    "intake_sourced",  # intake dimension, source standard/image -> sourced
    "intake_researched",  # intake dimension, source researched (cited web finding) -> sourced
    "intake_stated",  # intake dimension, source user -> stated
    "default_estimated",  # no out-of-model anchor -> estimated
]


class EstimateBasis(BaseModel):
    """The model's SELF-REPORTED reasoning for an estimated dimension (methodology seat).
    Auditable engineering reasoning vs vibes; informational only — never raises the tier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["anchor_scaled", "functional_requirement", "proportion_rule", "default_guess"]
    anchor: str | None = Field(
        default=None, description="The named reference/constraint the estimate hangs on."
    )


class DimensionProvenance(BaseModel):
    """One feature's dimension-provenance record (auditable: tier + mechanism + citation)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature: str
    dims_text: str  # the brief's approx_dims_mm verbatim (drift audit)
    tier: Tier
    mechanism: Mechanism
    citation: str | None = None  # registry source / the intake fact line; None for estimated
    estimate_basis: EstimateBasis | None = None


class TierCounts(BaseModel):
    """The tier mix the confidence label is derived from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verified: int = 0
    sourced: int = 0
    stated: int = 0
    estimated: int = 0


# --- registry resolution (the verified/sourced anchor) -----------------------------------------

_FASTENER_RE = re.compile(r"[mM]\d+(?:\.\d+)?")
_NUMERIC_RE = re.compile(r"\d+")
_MAGNET_RE = re.compile(r"\d+x\d+", re.IGNORECASE)


def resolve_standard(designation: str) -> tuple[str, object] | None:
    """Resolve a designation against the cited registries. Returns (registry_name, spec) or None.

    Exact normalized key match only (no fuzzy matching — fuzz is how tier laundering starts):
    'm3'/'M3' -> 'M3'; '608' -> '608'; '6X3' -> '6x3'. The FIRST whitespace token is judged.
    """
    token = designation.strip().split()[0] if designation.strip() else ""
    if not token:
        return None
    if _FASTENER_RE.fullmatch(token):
        key = "M" + token[1:]
    elif _MAGNET_RE.fullmatch(token):
        key = token.lower()
    elif _NUMERIC_RE.fullmatch(token):
        key = token
    else:
        return None
    for name, reg in ALL_REGISTRIES.items():
        if reg.has(key):
            return name, reg.get(key)
    return None


# --- intake STATED-FACTS parsing (the sourced/stated anchors) -----------------------------------

# `- dimension: mount width = 120 mm (source: user)` — emitted by intake.brief.brief_to_context.
_STATED_DIM_RE = re.compile(
    r"^- dimension:\s*(?P<name>.+?)\s*=\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>\S+)"
    r"\s*\(source:\s*(?P<source>[a-z_]+)\)\s*$",
    re.MULTILINE,
)
# Intake sources that anchor a tier (DEFAULT/UNKNOWN never promote: a default is not a fact).
# ``researched`` (ADR-0087): a fact pulled from a CITED web finding (the requirements compiler's
# research subsystem). It is externally-sourced like a standard, so it earns SOURCED -- but NEVER
# stated: a researched dimension is "we looked it up, verify it against your hardware", never "you
# told us". Its distinct mechanism (``intake_researched``) is what the UI keys the verify-required
# label off. The anchor invariant holds: the model may propose a researched value, but a downstream
# citation+quote verification (ADR-0087 S3, the vendor-gate discipline) is what lets it enter here.
_INTAKE_SOURCE_TIER: dict[str, tuple[Tier, Mechanism]] = {
    "standard": (Tier.SOURCED, "intake_sourced"),
    "image": (Tier.SOURCED, "intake_sourced"),
    "researched": (Tier.SOURCED, "intake_researched"),
    "user": (Tier.STATED, "intake_stated"),
}

_NUM_IN_TEXT_RE = re.compile(r"\d+(?:\.\d+)?")


class _StatedDim(BaseModel):
    model_config = ConfigDict(frozen=True)

    line: str
    value: float
    tier: Tier
    mechanism: Mechanism


def parse_stated_dimensions(extra_context: str | None) -> list[_StatedDim]:
    """The intake block's typed dimension lines, with their tier mapping. Pure + deterministic."""
    if not extra_context:
        return []
    out: list[_StatedDim] = []
    for m in _STATED_DIM_RE.finditer(extra_context):
        mapping = _INTAKE_SOURCE_TIER.get(m.group("source"))
        if mapping is None:
            continue
        out.append(
            _StatedDim(
                line=m.group(0).lstrip("- ").strip(),
                value=float(m.group("value")),
                tier=mapping[0],
                mechanism=mapping[1],
            )
        )
    return out


def _match_stated(dims_text: str, stated: list[_StatedDim]) -> _StatedDim | None:
    """An exact numeric match between a feature's dims and an intake fact (the brief is told to
    copy stated facts verbatim, so equality — not tolerance — is the honest test). Returns the
    STRONGEST matching fact. A coincidental equality is possible; the cited line rides the record
    so it is auditable (flagged in ADR-0040)."""
    nums = {float(n) for n in _NUM_IN_TEXT_RE.findall(dims_text or "")}
    if not nums:
        return None
    hits = [s for s in stated if s.value in nums]
    if not hits:
        return None
    order = {Tier.SOURCED: 0, Tier.STATED: 1}
    return sorted(hits, key=lambda s: order.get(s.tier, 9))[0]


# --- the resolver -------------------------------------------------------------------------------


def derive_provenance(
    brief: DesignBrief, extra_context: str | None = None
) -> list[DimensionProvenance]:
    """Earn each feature's tier deterministically (the anchor invariant lives here).

    Order of strength per feature: registry resolution of its ``standard`` designation
    (verified / sourced) > intake fact match (sourced / stated) > estimated. The model's
    own ``estimate_basis`` is recorded for estimates but never changes a tier.
    """
    stated = parse_stated_dimensions(extra_context)
    records: list[DimensionProvenance] = []
    for f in brief.features:
        basis = (
            EstimateBasis(method=f.estimate_basis, anchor=f.estimate_anchor)
            if f.estimate_basis is not None
            else None
        )
        if f.standard:
            hit = resolve_standard(f.standard)
            if hit is not None:
                name, spec = hit
                provisional = bool(getattr(spec, "provisional", False))
                records.append(
                    DimensionProvenance(
                        feature=f.name,
                        dims_text=f.approx_dims_mm,
                        tier=Tier.SOURCED if provisional else Tier.VERIFIED,
                        mechanism="provisional_registry" if provisional else "registry_resolution",
                        citation=f"{name} {f.standard.strip()}: {getattr(spec, 'source', '')}",
                    )
                )
                continue
            # The model cited a designation the registry lacks: DEMAND signal for curation,
            # and the dimension stays estimated (a model-authored citation earns nothing).
            record_demand(f.standard, feature=f.name)
        fact = _match_stated(f.approx_dims_mm, stated)
        if fact is not None:
            records.append(
                DimensionProvenance(
                    feature=f.name,
                    dims_text=f.approx_dims_mm,
                    tier=fact.tier,
                    mechanism=fact.mechanism,
                    citation=f"intake: {fact.line}",
                )
            )
            if fact.tier is Tier.SOURCED:
                record_candidate(fact.line, feature=f.name)
            continue
        records.append(
            DimensionProvenance(
                feature=f.name,
                dims_text=f.approx_dims_mm,
                tier=Tier.ESTIMATED,
                mechanism="default_estimated",
                estimate_basis=basis or EstimateBasis(method="default_guess", anchor=None),
            )
        )
    return records


def tier_counts(records: list[DimensionProvenance]) -> TierCounts:
    counts = dict.fromkeys(Tier, 0)
    for r in records:
        counts[r.tier] += 1
    return TierCounts(
        verified=counts[Tier.VERIFIED],
        sourced=counts[Tier.SOURCED],
        stated=counts[Tier.STATED],
        estimated=counts[Tier.ESTIMATED],
    )


# --- the ledger (runtime-write-only; never read at generation time) -----------------------------


def _ledger_path() -> Path | None:
    p = os.environ.get("FORGE_PROVENANCE_LEDGER", "").strip()
    return Path(p) if p else None


def _append_ledger(record: dict[str, object]) -> None:
    """Best-effort append; a ledger miss never affects a build (and is never read back here)."""
    path = _ledger_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return


def record_demand(designation: str, *, feature: str | None = None) -> None:
    """A generation wanted a designation the registry lacks — the curation priority queue."""
    _append_ledger(
        {
            "kind": "demand",
            "designation": designation.strip(),
            "feature": feature,
            "at": datetime.now(UTC).isoformat(),
        }
    )


def record_candidate(cited_fact: str, *, feature: str | None = None) -> None:
    """An intake fact WITH a source marker — evidence for human curation, never authority."""
    _append_ledger(
        {
            "kind": "candidate",
            "fact": cited_fact,
            "feature": feature,
            "at": datetime.now(UTC).isoformat(),
        }
    )


__all__ = [
    "DimensionProvenance",
    "EstimateBasis",
    "Tier",
    "TierCounts",
    "derive_provenance",
    "parse_stated_dimensions",
    "record_candidate",
    "record_demand",
    "resolve_standard",
    "tier_counts",
]
