"""``compose_physics_report`` — assemble a contract ``PhysicsReport`` from an already-finished
``IntrinsicReport``. PURE and READ-ONLY: it takes a finished report DTO (not raw geometry) and reprojects
it; it is structurally incapable of running a check or gating anything.

THE LOAD-BEARING INVARIANT (ADR-0056, council-ruled): ``overall.verdict`` is a PURE RELABEL of the
validator's aggregate verdict (``IntrinsicReport.printable`` == ``aggregate_status(checks)``), NEVER a
re-aggregation of the projected findings. So even if a future suite projects a finding the validator never
ran, it can move neither the verdict nor (because only Check-backed findings carry a blocking severity)
the blocking count. This projection can never become a second verdict authority.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge_mind.physics.fea import LoadResponse

from forge_mind.physics.report.holes import DetectedHole
from forge_mind.physics.report.projectors import (
    project_boltbrain,
    project_fitproof,
    project_loadproof,
    project_printproof,
    project_step_integrity,
    project_vitals,
)
from forge_mind.physics.report.schema import (
    Finding,
    Intent,
    MotionReport,
    Overall,
    OverallConfidence,
    PartView,
    PhysicsReport,
    PhysicsVerdict,
    Scope,
    Suite,
)
from forge_mind.types import Vec3
from forge_mind.validator.intrinsic import IntrinsicReport
from forge_mind.validator.report import CheckStatus

# IntrinsicReport.printable is a CheckStatus (the validator's sole aggregate). Pure relabel — no re-roll.
# aggregate_status only ever returns PASS/WARN/FAIL (INFO never moves the aggregate), so those three are
# the only reachable keys; `.get(..., "warn")` is a defensive fallback that never green-washes.
_VERDICT_MAP: dict[CheckStatus, PhysicsVerdict] = {
    CheckStatus.PASS: "pass",
    CheckStatus.WARN: "warn",
    CheckStatus.FAIL: "fail",
}

# The non-removable honesty sentences (ADR-0041 advisory disclaimer + the solid-PLA assumption + the
# no-intent ceiling). Asserted present by a test.
_DISCLAIMERS = [
    "Physics results are an engineering assessment, not a guarantee. Verify with a real build before "
    "relying on a load-bearing part.",
    "Estimates assume solid (100% infill) PLA at room temperature unless stated otherwise.",
    "Fit and function are NOT certified: no part intent (load, material, mating parts) was provided, so "
    "those suites did not run.",
]

_SUITE_TITLES = {
    "printproof": "PrintProof — will it print?",
    "fitproof": "FitProof — will it fit & assemble?",
    "boltbrain": "BoltBrain — fasteners",
    "loadproof": "LoadProof — will it hold up?",
    "motion": "MotionProof — what really moves?",
}


def _suite_verdict(findings: list[Finding]) -> PhysicsVerdict:
    """A per-suite SUMMARY of its own findings (descriptive, not the report verdict authority). All
    not_run -> not_run; any fail -> fail; any warn -> warn; else pass."""
    statuses = {f.status for f in findings}
    if statuses <= {"not_run"}:
        return "not_run"
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    if {"pass", "info"} & statuses:
        return "pass"
    return "not_run"


def _suite_summary(suite_id: str, findings: list[Finding]) -> str:
    ran = [f for f in findings if f.status != "not_run"]
    fails = [f for f in ran if f.status == "fail"]
    warns = [f for f in ran if f.status == "warn"]
    if not ran:
        reason = (
            findings[0].reason_not_run if findings and findings[0].reason_not_run else "not run"
        )
        return f"Not run — {reason}"
    if fails:
        return f"{len(fails)} blocking issue(s): {fails[0].explanation}"
    if warns:
        return f"{len(warns)} caution(s): {warns[0].explanation}"
    return "All checks passed."


def _suite(suite_id: str, findings: list[Finding]) -> Suite:
    return Suite(
        id=suite_id,
        title=_SUITE_TITLES[suite_id],
        verdict=_suite_verdict(findings),
        summary=_suite_summary(suite_id, findings),
        findings=findings,
    )


def _basis(verdict: PhysicsVerdict) -> str:
    head = {
        "pass": "Prints as a solid: all print-readiness checks pass.",
        "warn": "Prints with cautions (see PrintProof).",
        "fail": "Not printable as-is (see PrintProof).",
    }.get(verdict, "Print-readiness assessed.")
    return (
        f"{head} Verdict mirrors the pre-print validator. Fit, fastener and load analysis need part "
        "intent and were not run."
    )


def compose_physics_report(
    report: IntrinsicReport,
    *,
    intent: Intent,
    bbox_mm: Vec3,
    surface_area_mm2: float | None,
    scope: Scope = "single_part",
    part_id: str = "partstl",
    part_name: str = "Uploaded part",
    holes: list[DetectedHole] | None = None,
    load_response: LoadResponse | None = None,
    motion: MotionReport | None = None,
    motion_findings: list[Finding] | None = None,
) -> PhysicsReport:
    """Project a finished IntrinsicReport into the frozen PhysicsReport contract. Read-only. ``holes`` are
    the measured B-rep bores the seam detected on a STEP upload (empty/None for STL); ``load_response`` is
    the validated mesh-FEA result under ``intent.load`` (None when no load was stated or the solve was
    refused/failed); ``motion``/``motion_findings`` are the ADR-0086 MotionProof projection (advisory —
    findings never carry a blocking severity, so they cannot move the verdict or counts)."""
    printproof = project_printproof(report.checks)
    # A STEP upload carries a B-rep: lead PrintProof with the editable-CAD integrity finding, and run
    # BoltBrain over the detected holes (the hole geometry now exists).
    has_brep = report.step_proof is not None and report.step_proof.is_brep is True
    if report.step_proof is not None:
        printproof = [project_step_integrity(report.step_proof), *printproof]
    loadproof = project_loadproof(
        report.stability, load_response, load_requested=intent.load is not None
    )
    fitproof = project_fitproof(intent.mates_with or [], holes or [])
    boltbrain = project_boltbrain(holes or [], has_brep=has_brep)

    suites = [
        _suite("printproof", printproof),
        _suite("fitproof", fitproof),
        _suite("boltbrain", boltbrain),
        _suite("loadproof", loadproof),
    ]
    if motion_findings:
        suites.append(_suite("motion", motion_findings))
    # The verdict is the PURE MAP — never recomputed from findings.
    verdict = _VERDICT_MAP.get(report.printable, "warn")
    # Display counts are PAIRED with the verdict, which is the PRINTABILITY aggregate only (ADR-0056:
    # fit/bolt/load are separate projections, each shown in its own suite). Counting fails across ALL
    # suites let a failing fitproof/loadproof finding push blocking >= 1 while verdict == "pass" -- a
    # self-contradictory header (audit M5). Count only the printability suite the verdict maps from;
    # the other suites' findings stay visible in their own sections.
    blocking = sum(1 for f in printproof if f.status == "fail")
    warnings = sum(1 for f in printproof if f.status == "warn")

    overall = Overall(
        verdict=verdict,
        blocking=blocking,
        warnings=warnings,
        confidence=OverallConfidence(tier="computed", basis=_basis(verdict)),
    )

    return PhysicsReport(
        schema_version=2,
        scope=scope,
        parts=[PartView(id=part_id, name=part_name, bbox_mm=bbox_mm)],
        intent=intent,
        vitals=project_vitals(report.metrics, surface_area_mm2),
        suites=suites,
        overall=overall,
        disclaimers=list(_DISCLAIMERS),
        motion=motion,
    )


__all__ = ["compose_physics_report"]
