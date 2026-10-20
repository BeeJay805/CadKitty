"""The invention-lab lifecycle layer (ADR-0047 / ADR-0048) — the pivot's frame around the engine.

This package wraps the proven CAD engine in an invention lifecycle without touching it.

  - ``labels``    — the moat-gate label types (the "NOT PHYSICALLY VALIDATED" banner; advisory
    feasibility that can never read as a pass).
  - ``package``   — the ``InventionPackage`` assembler: composes a finished part's existing outputs
    into one honest document. REPORTED, never gated.
  - ``concept``   — the ``ConceptBrief`` / ``ConceptPlan`` contracts (closed mechanism vocabulary +
    attributed design-proposal prose; no fitness claim).
  - ``frontdoor`` — ``run_front_door``: ONE Fable brain pass, problem -> ConceptPlan. The injectable
    ``client`` seam makes it fully testable offline (no spend); the live measurement waits for Fable.

The validator stays the sole verdict-issuer throughout.
"""

from __future__ import annotations

from forge_mind.invention.concept import ConceptBrief, ConceptPlan, Mechanism
from forge_mind.invention.frontdoor import run_front_door
from forge_mind.invention.labels import (
    EvidenceTier,
    FeasibilityAssessment,
    FeasibilityVerdict,
    MaturityLabel,
    PhysicalValidation,
    PhysicalValidationStatus,
    derive_maturity,
    derive_package_evidence,
    evidence_rank,
)
from forge_mind.invention.lifecycle import (
    GeneratedPart,
    MissionPersister,
    build_mission_log,
    concept_to_prompt,
    invention_enabled,
    run_and_persist_lifecycle,
    run_lifecycle,
    run_lifecycle_with_mission,
)
from forge_mind.invention.package import InventionPackage, build_invention_package
from forge_mind.invention.vendor import (
    NullFetcher,
    PartRequirement,
    VendorBom,
    VendorFetcher,
    VendorLine,
    VendorListing,
    VendorProvenance,
    research_vendor_parts,
)

__all__ = [
    "ConceptBrief",
    "ConceptPlan",
    "EvidenceTier",
    "FeasibilityAssessment",
    "FeasibilityVerdict",
    "GeneratedPart",
    "InventionPackage",
    "MaturityLabel",
    "Mechanism",
    "MissionPersister",
    "NullFetcher",
    "PartRequirement",
    "PhysicalValidation",
    "PhysicalValidationStatus",
    "VendorBom",
    "VendorFetcher",
    "VendorLine",
    "VendorListing",
    "VendorProvenance",
    "build_invention_package",
    "build_mission_log",
    "concept_to_prompt",
    "derive_maturity",
    "derive_package_evidence",
    "evidence_rank",
    "invention_enabled",
    "research_vendor_parts",
    "run_and_persist_lifecycle",
    "run_front_door",
    "run_lifecycle",
    "run_lifecycle_with_mission",
]
