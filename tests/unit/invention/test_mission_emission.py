"""The mission-spine PRODUCER (ADR-0050): a finished lifecycle run composes into a real append-only
MissionLog -- the chosen concept selected, the runners-up RETAINED as memory, an honest validation
mapped from the disposition -- which replays to a MissionState, feeds the dashboard, and round-trips
through the mission_events adapter. Deterministic, no model call."""

from __future__ import annotations

from forge_mind.agentic.confidence import standard_lane_label
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.invention.concept import ConceptBrief, ConceptPlan, Mechanism
from forge_mind.invention.dashboard import build_dashboard
from forge_mind.invention.labels import EvidenceTier, FeasibilityVerdict
from forge_mind.invention.lifecycle import build_mission_log
from forge_mind.invention.mission import EventType
from forge_mind.invention.package import InventionPackage, build_invention_package
from forge_mind.kit import build_kit
from forge_mind.learning.mission_records import log_from_rows, rows_for_log
from forge_mind.types import Vec3

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)


def _plan() -> ConceptPlan:
    return ConceptPlan(
        problem_restated="constrain two cables along an edge",
        feasibility_verdict=FeasibilityVerdict.LIKELY_FEASIBLE,
        feasibility_note="a routine FDM build",
        concepts=[
            ConceptBrief(
                concept_id="c1",
                name="C-clip",
                primary_mechanism=Mechanism.SNAP_FIT,
                approach="a flexure that snaps over the edge",
                obvious_baseline="a screwed bracket",
                difference_from_obvious="no fasteners",
            ),
            ConceptBrief(
                concept_id="c2",
                name="friction saddle",
                primary_mechanism=Mechanism.FRICTION_FIT,
                approach="a weighted body held by friction",
                obvious_baseline="adhesive",
                difference_from_obvious="removable",
            ),
        ],
        recommended_concept_id="c1",
        recommendation_reason="simplest and needs no hardware",
    )


def _package(
    disposition: str = "shipped", artifacts: dict[str, str] | None = None
) -> InventionPackage:
    program = PartProgram(
        metadata=Metadata(name="cable-clip", print_orientation=_UP_Z),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return build_invention_package(
        part_name="cable-clip",
        disposition=disposition,
        kit=build_kit(program),
        confidence=standard_lane_label(1),
        problem_statement="route two cables",
        artifacts=artifacts if artifacts is not None else {"stl": "part.stl", "glb": "part.glb"},
    )


def test_producer_composes_the_full_event_timeline() -> None:
    log = build_mission_log("route two cables", _plan(), _package(), mission_id="m1")
    types = [e.event_type for e in log.events()]
    assert types[0] is EventType.MISSION_CREATED
    assert EventType.CONCEPT_SELECTED in types
    assert types.count(EventType.CONCEPT_REJECTED) == 1  # the one runner-up, retained as memory
    assert EventType.DECISION_RECORDED in types
    assert types.count(EventType.ARTIFACT_CREATED) == 2  # stl + glb
    assert EventType.VALIDATION_PASSED in types
    assert types[-1] is EventType.PACKAGE_ASSEMBLED


def test_replayed_state_is_honest() -> None:
    state = build_mission_log("route two cables", _plan(), _package(), mission_id="m1").state()
    assert state.selected_concept_id == "c1"
    assert state.current_stage == "package"
    # the runner-up is RETAINED with a reopen condition (a failure becomes memory, never erased)
    assert len(state.rejected_paths) == 1
    assert state.rejected_paths[0].concept_id == "c2"
    assert state.rejected_paths[0].reopen_conditions
    # the recommendation is a model judgment -- it never claims authority it lacks
    assert state.decisions[0].evidence_tier is EvidenceTier.AI_HYPOTHESIS
    assert state.validations[0].passed is True


def test_failed_disposition_emits_a_failed_validation() -> None:
    state = build_mission_log(
        "p", _plan(), _package(disposition="rejected"), mission_id="m1"
    ).state()
    assert state.validations[0].passed is False
    assert any("rejected" in c for c in state.validations[0].failed_checks)


def test_producer_feeds_the_dashboard() -> None:
    log = build_mission_log("route two cables", _plan(), _package(), mission_id="m1")
    dash = build_dashboard(log.state(), log.events(), package=_package())
    assert dash.mission_id == "m1"
    assert dash.current_stage == "package"
    assert dash.latest_finding is not None
    assert dash.latest_finding.event_type == "package_assembled"
    assert dash.validation_state.passed >= 1
    assert dash.support_kb_ready is True
    assert any("Not physically validated" in r for r in dash.unresolved_risks)


def test_mission_round_trips_through_the_adapter() -> None:
    log = build_mission_log("route two cables", _plan(), _package(), mission_id="m1")
    rebuilt = log_from_rows(rows_for_log(log))
    assert rebuilt.state() == log.state()
