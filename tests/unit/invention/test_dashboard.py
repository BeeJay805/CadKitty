"""Minimal dashboard data API (ADR-0048, P8): a backend read-model composed deterministically from
the mission timeline + the package. Backend-only, no model call."""

from __future__ import annotations

from forge_mind.agentic.confidence import standard_lane_label
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.invention import build_invention_package
from forge_mind.invention.dashboard import MissionDashboard, build_dashboard
from forge_mind.invention.mission import (
    ArtifactRecord,
    EventType,
    MissionLog,
    RejectedPath,
    ValidationRecord,
)
from forge_mind.invention.package import InventionPackage
from forge_mind.kit import build_kit
from forge_mind.types import Vec3

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)


def _log() -> MissionLog:
    log = MissionLog("m1", "route two cables")
    log.append(EventType.CONCEPT_SELECTED, summary="pick C-clip", concept_id="c1", stage="concept")
    log.append(
        EventType.CONCEPT_REJECTED,
        summary="reject magnet clip",
        rejected_path=RejectedPath(
            concept_id="c2",
            idea="magnet clip",
            reason="too weak",
            reopen_conditions=["source an N52 magnet"],
        ),
    )
    log.append(
        EventType.ARTIFACT_CREATED,
        summary="stl written",
        artifact=ArtifactRecord(artifact_id="a1", kind="stl", ref="part.stl"),
    )
    log.append(
        EventType.VALIDATION_FAILED,
        summary="overhang",
        validation=ValidationRecord(passed=False, failed_checks=["printability.overhang"]),
        stage="validation",
    )
    log.append(EventType.REVISION_CREATED, summary="re-oriented", stage="cad")
    log.append(
        EventType.VALIDATION_PASSED, summary="clean", validation=ValidationRecord(passed=True)
    )
    return log


def _package() -> InventionPackage:
    program = PartProgram(
        metadata=Metadata(name="cable-clip", units="mm", print_orientation=_UP_Z),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return build_invention_package(
        part_name="cable-clip",
        disposition="shipped",
        kit=build_kit(program),
        confidence=standard_lane_label(1),
        problem_statement="route two cables",
        artifacts={"glb": "part.glb"},
    )


def test_dashboard_folds_the_mission() -> None:
    log = _log()
    dash = build_dashboard(log.state(), log.events())
    assert isinstance(dash, MissionDashboard)
    assert dash.mission_id == "m1"
    assert dash.problem == "route two cables"
    assert dash.current_stage == "cad"  # the latest stage-advancing event
    assert dash.latest_finding is not None
    assert dash.latest_finding.event_type == "validation_passed"
    assert dash.validation_state.passed == 1
    assert dash.validation_state.failed == 1
    assert [e.event_type for e in dash.revision_timeline] == [
        "validation_failed",
        "revision_created",
        "validation_passed",
    ]


def test_dashboard_without_a_package_has_no_maturity() -> None:
    log = _log()
    dash = build_dashboard(log.state(), log.events())
    assert dash.maturity is None
    assert dash.banner is None
    assert dash.support_kb_ready is False


def test_dashboard_with_a_package_surfaces_maturity_and_banner() -> None:
    log = _log()
    dash = build_dashboard(log.state(), log.events(), package=_package())
    assert dash.maturity == "digitally_validated"
    assert dash.evidence_tier == "deterministic_check"
    assert dash.banner is not None and "NOT PHYSICALLY VALIDATED" in dash.banner
    assert dash.support_kb_ready is True


def test_unresolved_risks_always_name_the_physical_gap_and_open_items() -> None:
    log = _log()
    dash = build_dashboard(log.state(), log.events())
    assert any("Not physically validated" in r for r in dash.unresolved_risks)
    # the latest failed validation was healed by a later pass, so it is NOT an open risk anymore
    assert not any("printability.overhang" in r for r in dash.unresolved_risks)
    # a rejected path's reopen condition is an open item to track
    assert any("Reopen c2" in r and "N52" in r for r in dash.unresolved_risks)


def test_artifact_previews_merge_mission_and_package() -> None:
    log = _log()
    dash = build_dashboard(log.state(), log.events(), package=_package())
    refs = {p.ref for p in dash.artifact_previews}
    assert "part.stl" in refs  # from the mission timeline
    assert "part.glb" in refs  # from the package
