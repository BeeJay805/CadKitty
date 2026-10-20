"""Deterministic lifecycle reports (ADR-0048, P4): structured projections over typed inputs, no
model calls. Feasibility/concept reviews are ai_hypothesis; the validation summary is a
deterministic_check; the package summary names PROVEN vs UNPROVEN explicitly."""

from __future__ import annotations

from typing import Any

from forge_mind.agentic.confidence import standard_lane_label
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.invention import (
    ConceptBrief,
    ConceptPlan,
    FeasibilityVerdict,
    Mechanism,
    build_invention_package,
)
from forge_mind.invention.mission import (
    EventType,
    MissionLog,
    RejectedPath,
    ValidationRecord,
)
from forge_mind.invention.reports import (
    build_concept_review,
    build_feasibility_review,
    build_package_ready_summary,
    build_revision_summary,
    build_validation_summary,
)
from forge_mind.kit import build_kit
from forge_mind.types import Vec3

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)


def _plan() -> ConceptPlan:
    return ConceptPlan(
        problem_restated="route flexible conductors along an edge without fasteners",
        feasibility_verdict=FeasibilityVerdict.LIKELY_FEASIBLE,
        feasibility_note="routine FDM print",
        concepts=[
            ConceptBrief(
                concept_id="c1",
                name="C-clip",
                primary_mechanism=Mechanism.SNAP_FIT,
                approach="flex over the edge",
                obvious_baseline="bracket",
                difference_from_obvious="no screws",
            ),
            ConceptBrief(
                concept_id="c2",
                name="saddle",
                primary_mechanism=Mechanism.FRICTION_FIT,
                approach="weighted saddle",
                obvious_baseline="adhesive",
                difference_from_obvious="removable",
            ),
        ],
        recommended_concept_id="c1",
        recommendation_reason="simplest, no hardware",
    )


def _package() -> Any:
    program = PartProgram(
        metadata=Metadata(name="cable-clip", print_orientation=_UP_Z),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return build_invention_package(
        part_name="cable-clip",
        disposition="shipped",
        kit=build_kit(program),
        confidence=standard_lane_label(1),  # basis = "validated"
        problem_statement="route two cables",
    )


def test_feasibility_review_is_advisory_ai_hypothesis() -> None:
    review = build_feasibility_review("route two cables", _plan().feasibility())
    assert review.verdict == "likely_feasible"
    assert "not a guarantee" in review.disclaimer
    assert review.evidence_tier == "ai_hypothesis"  # no deterministic checker behind feasibility


def test_concept_review_lists_considered_selected_and_rejected() -> None:
    review = build_concept_review(
        _plan(),
        selected_concept_id="c1",
        rejected=[
            RejectedPath(
                concept_id="c2",
                idea="friction saddle",
                reason="slides under cable tension",
                reopen_conditions=["add a lip"],
            )
        ],
    )
    assert [o.concept_id for o in review.considered] == ["c1", "c2"]
    assert review.selected is not None and review.selected.concept_id == "c1"
    assert review.selection_reason == "simplest, no hardware"
    assert review.rejected[0].concept_id == "c2"
    assert review.rejected[0].reopen_conditions == ["add a lip"]


def test_validation_summary_counts_and_reports_latest_failure() -> None:
    log = MissionLog("m1", "p")
    log.append(
        EventType.VALIDATION_FAILED,
        summary="overhang",
        validation=ValidationRecord(passed=False, failed_checks=["printability.overhang"]),
    )
    log.append(EventType.VALIDATION_PASSED, summary="ok", validation=ValidationRecord(passed=True))
    summary = build_validation_summary(log.state())
    assert summary.total == 2
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.latest_passed is True
    assert summary.failed_checks == []  # latest passed, so no failed checks surfaced
    assert summary.evidence_tier == "deterministic_check"


def test_revision_summary_builds_the_timeline() -> None:
    log = MissionLog("m1", "p")
    log.append(
        EventType.VALIDATION_FAILED, summary="collision", validation=ValidationRecord(passed=False)
    )
    log.append(EventType.REVISION_CREATED, summary="moved the jaw offset", stage="cad")
    log.append(
        EventType.VALIDATION_PASSED, summary="clean", validation=ValidationRecord(passed=True)
    )
    summary = build_revision_summary(log.events())
    assert summary.revisions == 1
    assert [e.event_type for e in summary.timeline] == [
        "validation_failed",
        "revision_created",
        "validation_passed",
    ]


def test_package_ready_summary_names_proven_and_unproven() -> None:
    summary = build_package_ready_summary(_package())
    assert summary.maturity == "digitally_validated"  # basis validated, but never physical
    assert summary.evidence_tier == "deterministic_check"
    assert any("validated" in p for p in summary.proven)
    # unproven ALWAYS names the physical + feasibility gaps -- digital is never physical proof
    assert any("not physically" in u for u in summary.unproven)
    assert any("feasibility" in u for u in summary.unproven)
    assert "NOT PHYSICALLY VALIDATED" in summary.banner
    assert summary.next_steps  # a digitally-validated part has concrete next steps
    assert any("physical" in s for s in summary.next_steps)
