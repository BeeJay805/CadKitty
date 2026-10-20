"""Support Copilot KB v0 (ADR-0048, P5) — the per-product, GROUNDED support knowledge base.

Every invention ships a structured support KB built DETERMINISTICALLY from its own data (the package,
the kit, the mission). This is NOT an autonomous chatbot: it is (1) a typed :class:`SupportCopilotKB`,
(2) a deterministic assembler, (3) a grounded retrieval interface, and (4) a READ-ONLY question/answer
seam with an INJECTABLE model client. The model never invents an answer: it phrases ONLY from
retrieved KB snippets, an ungrounded question is labelled ``unknown``, and any load/safety question is
REFUSED (the copilot knows product limits and refuses unsafe advice). With no client wired it still
answers — deterministically, by returning the grounded snippets.

The eventual chatbot answers, all from this KB: how to assemble it, why a part exists, what failed
previously, what substitutions are allowed, what remains unverified, and how to troubleshoot.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.invention.labels import EvidenceTier, derive_maturity, derive_package_evidence
from forge_mind.invention.mission import MissionEvent
from forge_mind.invention.package import InventionPackage
from forge_mind.invention.reports import build_revision_summary

# ---------------------------------------------------------------------------------------------
# The typed KB
# ---------------------------------------------------------------------------------------------


class KBPart(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: str  # printed | fastener | bearing | insert | ...
    description: str


class KBArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    ref: str


class KBBomItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component: str
    quantity: int
    source: str
    provisional: bool


class KBTroubleshoot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symptom: str
    checks: list[str] = Field(default_factory=list)


class KBValidationStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    disposition: str
    maturity: str
    evidence_tier: str
    banner: str  # the mandatory NOT-PHYSICALLY-VALIDATED disclosure


class SupportCopilotKB(BaseModel):
    """The grounded support knowledge for ONE invention. Every field is derived from the product's
    own typed outputs -- no free-text claims about fitness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    product_purpose: str
    parts: list[KBPart] = Field(default_factory=list)
    artifacts: list[KBArtifact] = Field(default_factory=list)
    bom: list[KBBomItem] = Field(default_factory=list)
    assembly_steps: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    validation_status: KBValidationStatus
    troubleshooting: list[KBTroubleshoot] = Field(default_factory=list)
    substitute_constraints: list[str] = Field(default_factory=list)
    safety_warnings: list[str] = Field(default_factory=list)
    revision_history: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# The deterministic assembler
# ---------------------------------------------------------------------------------------------

_LIMIT_PHYSICAL = "Not physically built, assembled, or tested -- this is a digital model."
_LIMIT_FEASIBILITY = "Feasibility is our engineering assessment, not a guarantee."
_LIMIT_ESTIMATED = "Some dimensions are estimated -- verify them against the real parts before use."

_SAFETY_BASE = (
    "This is a conceptual digital model, not a rated or certified component. Do not rely on it for "
    "any load-bearing, safety-critical, or injury-risk use."
)

# Generic troubleshooting for a printed part (closed table; enriched by the BOM below).
_GENERIC_TROUBLESHOOT: list[KBTroubleshoot] = [
    KBTroubleshoot(
        symptom="The part does not fit the mating hardware.",
        checks=[
            "Verify any estimated dimensions against the real part.",
            "Confirm you printed at 100% scale.",
            "Check the print orientation matches the make-instructions.",
        ],
    ),
    KBTroubleshoot(
        symptom="The part is weak or breaks.",
        checks=[
            "Increase wall count / infill to realize the recommended minimum wall.",
            "Re-orient so layer lines do not run across the load path.",
        ],
    ),
]


def _parts(package: InventionPackage) -> list[KBPart]:
    parts = [KBPart(id="printed_part", kind="printed", description=package.part_name)]
    for line in package.kit.bom.lines:
        parts.append(KBPart(id=line.key, kind=line.kind, description=line.component))
    return parts


def _bom(package: InventionPackage) -> list[KBBomItem]:
    return [
        KBBomItem(
            component=line.component,
            quantity=line.quantity,
            source=line.source,
            provisional=line.provisional,
        )
        for line in package.kit.bom.lines
    ]


def _known_limitations(package: InventionPackage) -> list[str]:
    limits = [_LIMIT_PHYSICAL, _LIMIT_FEASIBILITY]
    if package.confidence is not None and package.confidence.dimension_tiers.estimated > 0:
        limits.append(_LIMIT_ESTIMATED)
    if package.confidence is not None:
        limits.extend(f"Not fit for: {nf}" for nf in package.confidence.not_fit_for)
    return limits


def _troubleshooting(package: InventionPackage) -> list[KBTroubleshoot]:
    entries = list(_GENERIC_TROUBLESHOOT)
    if any(line.kind == "insert" for line in package.kit.bom.lines):
        entries.append(
            KBTroubleshoot(
                symptom="A heat-set insert will not seat squarely.",
                checks=[
                    "Heat the insert with the iron, press slowly, keep it perpendicular.",
                    "Confirm the boss bore matches the insert's outer diameter.",
                ],
            )
        )
    return entries


def _substitute_constraints(package: InventionPackage) -> list[str]:
    out: list[str] = []
    for line in package.kit.bom.lines:
        note = f"{line.component}: a substitute must match the spec"
        if line.provisional:
            note += " (this line is PROVISIONAL -- confirm size and length before substituting)"
        else:
            note += f" -- cited: {line.source}"
        out.append(note)
    if not out:
        out.append("No purchased parts -- nothing to substitute.")
    return out


def _safety_warnings(package: InventionPackage) -> list[str]:
    warnings = [_SAFETY_BASE]
    if package.confidence is not None:
        warnings.extend(f"Not fit for: {nf}" for nf in package.confidence.not_fit_for)
    return warnings


def build_support_kb(
    package: InventionPackage, *, events: list[MissionEvent] | None = None
) -> SupportCopilotKB:
    """Assemble the grounded KB deterministically from the product's own outputs. ``events`` (the
    mission timeline, optional) supplies the revision history; everything else comes from the
    package."""
    basis = package.confidence.basis if package.confidence is not None else None
    revisions: list[str] = []
    if events is not None:
        revisions = [
            f"#{entry.seq} {entry.event_type}: {entry.summary}"
            for entry in build_revision_summary(events).timeline
        ]
    return SupportCopilotKB(
        product_purpose=package.problem_statement or package.part_name,
        parts=_parts(package),
        artifacts=[KBArtifact(kind=k, ref=v) for k, v in sorted(package.artifacts.items())],
        bom=_bom(package),
        assembly_steps=list(package.kit.instructions.steps),
        known_limitations=_known_limitations(package),
        validation_status=KBValidationStatus(
            disposition=package.disposition,
            maturity=derive_maturity(basis, package.physical_validation.status).value,
            evidence_tier=derive_package_evidence(basis).value,
            banner=package.physical_validation.banner,
        ),
        troubleshooting=_troubleshooting(package),
        substitute_constraints=_substitute_constraints(package),
        safety_warnings=_safety_warnings(package),
        revision_history=revisions,
    )


# ---------------------------------------------------------------------------------------------
# Grounded retrieval + the read-only QA seam
# ---------------------------------------------------------------------------------------------


class KBSnippet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    section: str  # which KB section grounds this snippet
    text: str


_SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "assembly": ("assemble", "assembly", "build", "install", "put together", "step"),
    "parts": ("part", "component", "bom", "buy", "purchase", "why does", "why is"),
    "troubleshooting": (
        "troubleshoot",
        "fix",
        "problem",
        "fit",
        "loose",
        "tight",
        "break",
        "wrong",
    ),
    "substitutes": ("substitute", "replace", "alternative", "swap", "instead"),
    "limitations": ("limit", "unverified", "unproven", "proven", "validated", "limitation"),
    "revisions": ("revision", "history", "change", "previous", "failed before"),
}


def retrieve(kb: SupportCopilotKB, query: str) -> list[KBSnippet]:
    """Deterministic grounded retrieval: match the query against the KB sections and return the
    grounding snippets. No match -> empty (the QA seam then answers 'unknown')."""
    q = query.lower()
    sections = {s for s, kws in _SECTION_KEYWORDS.items() if any(k in q for k in kws)}
    snippets: list[KBSnippet] = []
    if "assembly" in sections and kb.assembly_steps:
        snippets.append(KBSnippet(section="assembly", text="; ".join(kb.assembly_steps)))
    if "parts" in sections:
        snippets.append(
            KBSnippet(
                section="parts", text="; ".join(f"{p.description} ({p.kind})" for p in kb.parts)
            )
        )
    if "troubleshooting" in sections:
        for entry in kb.troubleshooting:
            snippets.append(
                KBSnippet(
                    section="troubleshooting", text=f"{entry.symptom} -> {'; '.join(entry.checks)}"
                )
            )
    if "substitutes" in sections and kb.substitute_constraints:
        snippets.append(KBSnippet(section="substitutes", text="; ".join(kb.substitute_constraints)))
    if "limitations" in sections and kb.known_limitations:
        snippets.append(KBSnippet(section="limitations", text="; ".join(kb.known_limitations)))
    if "revisions" in sections and kb.revision_history:
        snippets.append(KBSnippet(section="revisions", text="; ".join(kb.revision_history)))
    return snippets


_SAFETY_KEYWORDS = (
    "safe",
    "safety",
    "load",
    "kg",
    "pound",
    "lbs",
    "weight",
    "hold up",
    "rated",
    "rating",
    "strength",
    "strong enough",
    "carry",
    "support a",
    "withstand",
)


class KBAnswerer(Protocol):
    """The injectable model seam: phrase an answer ONLY from the retrieved grounding snippets. The
    real implementation wraps an LLM; tests pass a stub; without one the QA seam answers
    deterministically from the snippets."""

    def answer(self, query: str, snippets: list[KBSnippet]) -> str: ...


class SupportAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["answered", "unknown", "refused"]
    answer: str | None
    citations: list[str] = Field(default_factory=list)
    evidence_tier: str = EvidenceTier.AI_HYPOTHESIS.value


def _is_safety_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _SAFETY_KEYWORDS)


def answer_question(
    kb: SupportCopilotKB, query: str, *, answerer: KBAnswerer | None = None
) -> SupportAnswer:
    """Answer a support question, GROUNDED and read-only. A load/safety question is REFUSED; an
    ungrounded question is labelled ``unknown`` (never fabricated); a grounded question is phrased by
    the injected model from the snippets only, or returned verbatim when no model is wired."""
    if _is_safety_query(query):
        return SupportAnswer(
            status="refused",
            answer="; ".join(kb.safety_warnings),
            citations=["safety_warnings"],
        )
    snippets = retrieve(kb, query)
    if not snippets:
        return SupportAnswer(status="unknown", answer=None, citations=[])
    citations = sorted({s.section for s in snippets})
    if answerer is None:
        text = " | ".join(f"[{s.section}] {s.text}" for s in snippets)
    else:
        text = answerer.answer(query, snippets)
    return SupportAnswer(
        status="answered",
        answer=text,
        citations=citations,
        evidence_tier=kb.validation_status.evidence_tier,
    )


__all__ = [
    "KBAnswerer",
    "KBArtifact",
    "KBBomItem",
    "KBPart",
    "KBSnippet",
    "KBTroubleshoot",
    "KBValidationStatus",
    "SupportAnswer",
    "SupportCopilotKB",
    "answer_question",
    "build_support_kb",
    "retrieve",
]
