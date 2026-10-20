"""Expert engineering review (founder-directed 2026-07-12): before parts are authored,
a structured LLM call acts as a PANEL -- mechanical, materials, thermal/fluids, chemical,
and aerospace engineers -- and reviews what the object must satisfy BEYOND geometry
(a rocket engine needs cooling; a gearbox needs lubrication paths; PLA cannot live at
300 C). A second call reviews the finished, validated assembly.

THE HONESTY ARCHITECTURE (constitution: models classify, they never decide):
  * the panel SURFACES requirements -- it is a requirements engine, never a certifier;
  * requirements carrying NUMBERS that deterministic code can compare (operating
    temperature vs the material's service ceiling) are ENFORCED by code -- the LLM only
    pointed at them;
  * everything else ships as NAMED, severity-tagged UNVERIFIED requirements in the
    report, and any blocker-severity finding CAPS the disposition at caveats -- the
    mechanism never ships 'clean' with unaddressed blocker-class physics. We never
    pretend to certify combustion, fatigue, or fluid dynamics from geometry.

FORGE_EXPERT_REVIEW=0 disables (default ON). Fail-open: a review call that errors
changes nothing about the run (the deterministic validators still stand)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.config import BRAIN_MODEL
from forge_mind.llm.provider import LLMClientLike, make_client

if TYPE_CHECKING:
    from forge_mind.materials import Material

# Continuous-service temperature ceilings (C), conservative published values. Local to
# the review (migrate onto Material once the table grows more properties). Deterministic
# enforcement data -- the LLM never supplies these numbers, only the environment.
_SERVICE_TEMP_C: dict[str, float] = {
    "PLA": 55.0,
    "PETG": 70.0,
    "ABS": 90.0,
    "TPU": 80.0,
    "Nylon": 100.0,
    "Resin": 60.0,
    "Aluminum": 400.0,
    "Steel": 500.0,
    "Brass": 200.0,
}


def expert_review_enabled() -> bool:
    return os.environ.get("FORGE_EXPERT_REVIEW", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


class ExpertRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discipline: str = Field(
        ..., description="mechanical | materials | thermal | chemical | aerospace | other"
    )
    requirement: str = Field(..., description="What must be satisfied, one sentence.")
    why: str = Field(..., description="The physical reason, one sentence.")
    quantity: str | None = Field(
        default=None,
        description="A quantitative limit when one exists, e.g. 'chamber wall < 350 C' "
        "or 'burst pressure > 2x operating'. None if inherently qualitative.",
    )
    severity: str = Field(..., description="blocker | major | minor")
    cad_verifiable: bool = Field(
        ...,
        description="True ONLY if pure geometry/kinematics can verify it (a clearance, a "
        "wall thickness, a passage existing). False for combustion, fatigue, chemistry, "
        "fluid/thermal behavior.",
    )


class ExpertEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operating_temp_c: float | None = Field(
        default=None, description="Peak sustained operating temperature the parts see (C)."
    )
    pressure_bar: float | None = Field(default=None, description="Peak pressure if any (bar).")
    medium: str | None = Field(
        default=None, description="Working medium touching the parts (air/water/fuel/none)."
    )
    duty: str = Field(
        ...,
        description="'demonstration' (hand-powered desk model) or 'functional' (does real "
        "work in its environment). A hand-crank DEMO of an engine is demonstration duty.",
    )


class ExpertReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: ExpertEnvironment
    requirements: list[ExpertRequirement] = Field(default_factory=list, max_length=16)
    material_concern: str | None = Field(
        default=None,
        description="One sentence if the stated material is unsuitable for the "
        "environment; None if acceptable.",
    )


_PLAN_SYSTEM = """You are an ENGINEERING REVIEW PANEL: a mechanical engineer, a materials \
engineer, a thermal/fluids engineer, a chemical engineer, and an aerospace engineer reviewing a \
design brief BEFORE anything is built.

First, read the request carefully and decide the DUTY: a desk-scale hand-cranked DEMONSTRATION \
model only needs to move, look right, and survive handling -- do NOT invent combustion/flight \
requirements for it. A FUNCTIONAL device that does real work in a real environment gets the full \
panel treatment.

Extract the operating ENVIRONMENT (temperature, pressure, medium) from the request -- numbers \
only when the request implies them (a combustion chamber implies >2000 C gas; a water pump \
implies wet, ~ambient; a desk demo implies room temperature).

Then list the requirements the design must satisfy BEYOND geometry, each tagged with discipline, \
severity, a quantitative limit when one exists, and whether pure CAD geometry could verify it. \
BE SELECTIVE: 3-10 requirements that would actually change the design, not a textbook dump. \
Severity 'blocker' is reserved for physics that makes the design not work AT ALL if unaddressed \
(no cooling on a regeneratively-cooled chamber; PLA at flame temperature)."""

_FINAL_SYSTEM = """You are the same ENGINEERING REVIEW PANEL, now reviewing the FINISHED, \
geometry-validated assembly summary. The geometric/kinematic verdicts are settled by \
deterministic code -- do NOT relitigate them. Your job: flag anything the finished configuration \
newly implies (a requirement now clearly unmet, a material/environment mismatch visible in the \
final report), same output schema, same selectivity. If nothing new, return zero requirements."""


def _deterministic_material_findings(
    review: ExpertReview, mat: Material | None
) -> list[dict[str, Any]]:
    """The enforcement half: number-vs-number comparisons CODE can make from what the
    panel extracted. The LLM surfaced the environment; these verdicts are deterministic."""
    findings: list[dict[str, Any]] = []
    if mat is None:
        return findings
    ceiling = _SERVICE_TEMP_C.get(mat.name)
    temp = review.environment.operating_temp_c
    if ceiling is not None and temp is not None and temp > ceiling:
        findings.append(
            {
                "discipline": "materials",
                "finding": (
                    f"operating temperature {temp:.0f} C exceeds {mat.name}'s continuous "
                    f"service ceiling ~{ceiling:.0f} C"
                ),
                "severity": "blocker",
                "deterministic": True,
            }
        )
    return findings


def run_expert_review(
    *,
    stage: str,  # "plan" | "final"
    prompt: str,
    summary: str,
    mat: Material | None,
    client: LLMClientLike | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """One panel call -> a structured review dict for the report:
    {environment, requirements[], material_concern, deterministic_findings[], blockers}.
    ``blockers`` counts blocker-severity items (LLM-flagged + deterministic); the caller
    caps the disposition at caveats when > 0. Fail-open: any error -> None."""
    if not expert_review_enabled():
        return None
    try:
        client = client or make_client()
        system = _PLAN_SYSTEM if stage == "plan" else _FINAL_SYSTEM
        user = (
            f"REQUEST:\n{prompt}\n\n"
            f"MATERIAL: {mat.name if mat else 'unspecified'}\n\n"
            f"{'PLAN SUMMARY' if stage == 'plan' else 'VALIDATED-RESULT SUMMARY'}:\n{summary}\n\n"
            "Return the panel review."
        )
        review, _u = client.generate_structured(
            system=system,
            user=user,
            output_model=ExpertReview,
            model=model or BRAIN_MODEL,
            max_tokens=4000,
        )
    except Exception:
        return None
    det = _deterministic_material_findings(review, mat)
    blockers = len(det) + sum(1 for r in review.requirements if r.severity == "blocker")
    return {
        "stage": stage,
        "environment": review.environment.model_dump(),
        "requirements": [r.model_dump() for r in review.requirements],
        "material_concern": review.material_concern,
        "deterministic_findings": det,
        "blockers": blockers,
    }


__all__ = ["ExpertReview", "expert_review_enabled", "run_expert_review"]
