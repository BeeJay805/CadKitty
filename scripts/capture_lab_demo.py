"""Capture the canonical OFFLINE demo artifacts for the invention lab (ADR-0048).

Runs the full lifecycle OFFLINE (problem -> concept -> part -> package) with the benchmark's mocked
model seam + the REAL deterministic CAD engine, then composes the dashboard read-model. Produces:

  - the full InventionPackage JSON + MissionDashboard JSON under benchmarks/results/lab_demo/
    (gitignored, regenerable), and
  - a trimmed, committed reference at benchmarks/invention_demo/offline_demo.reference.json.

HONEST about what is real: the CONCEPT is the front-door MOCK (a fixed 2-concept plan -- no model is
called); the PART geometry / kit is the REAL engine output (deterministic). It is NOT a live result and
NOT physically validated. The offline lifecycle does not itself persist a mission log (that is the live
API binding's job, ADR-0048 P3), so the mission TIMELINE here is composed from the real run output by
this script -- every event is a true statement about the run, recorded the way the live binding would.

Deterministic (no timestamps, fixed mission id), so the committed reference is stable across reruns.

Run: uv run --no-sync python scripts/capture_lab_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from forge_mind.invention.bench import _offline_produce, _OfflineFrontDoor, default_problems
from forge_mind.invention.concept import ConceptPlan
from forge_mind.invention.dashboard import MissionDashboard, build_dashboard
from forge_mind.invention.lifecycle import run_lifecycle
from forge_mind.invention.mission import EventType, MissionLog, RejectedPath, ValidationRecord
from forge_mind.invention.package import InventionPackage

_MISSION_ID = "demo-cable-clip-offline"
_RESULTS_DIR = Path("benchmarks/results/lab_demo")
_REFERENCE = Path("benchmarks/invention_demo/offline_demo.reference.json")
_PASSED_DISPOSITIONS = {"shipped", "shipped_with_caveats"}


def _compose_mission(package: InventionPackage, plan: ConceptPlan) -> MissionDashboard:
    """Record the REAL run as an append-only MissionLog (the way the live binding would), then fold it
    into the dashboard. The offline lifecycle has no recorder, so the demo composes this from the run;
    every event below is a true statement about the run that just executed."""
    log = MissionLog(_MISSION_ID, package.problem_statement or "")
    chosen = plan.recommended()
    # rejected alternatives are RETAINED as memory (a failure / also-ran becomes a reopen condition).
    for concept in plan.concepts:
        if concept.concept_id == chosen.concept_id:
            continue
        log.append(
            EventType.CONCEPT_REJECTED,
            summary=f"not recommended: {concept.name} ({concept.primary_mechanism.value})",
            stage="concept",
            rejected_path=RejectedPath(
                concept_id=concept.concept_id,
                idea=f"{concept.name}: {concept.approach}",
                reason="the front door recommended a different concept",
                reopen_conditions=["if the recommended concept fails its gates"],
            ),
        )
    log.append(
        EventType.CONCEPT_SELECTED,
        summary=f"selected {chosen.name} ({chosen.primary_mechanism.value})",
        stage="concept",
        concept_id=chosen.concept_id,
    )
    passed = package.disposition in _PASSED_DISPOSITIONS
    log.append(
        EventType.VALIDATION_PASSED if passed else EventType.VALIDATION_FAILED,
        summary=f"engine disposition: {package.disposition}",
        stage="validation",
        validation=ValidationRecord(passed=passed),
    )
    log.append(
        EventType.PACKAGE_ASSEMBLED,
        summary="invention package assembled (REPORTED -- not physically validated)",
        stage="package",
    )
    return build_dashboard(log.state(), log.events(), package=package)


def _trimmed_package(package: InventionPackage) -> dict[str, object]:
    """The honesty-relevant package fields, with the bulky kit replaced by a pointer (readability)."""
    data = package.model_dump(mode="json")
    data["kit"] = "<omitted; full kit in benchmarks/results/lab_demo/invention_package.json>"
    return data


def main() -> int:
    problem = default_problems()[0]  # the v0 cable clip
    package, plan = run_lifecycle(
        problem.problem, produce=_offline_produce, client=_OfflineFrontDoor()
    )
    dashboard = _compose_mission(package, plan)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (_RESULTS_DIR / "invention_package.json").write_text(
        package.model_dump_json(indent=2), encoding="utf-8"
    )
    (_RESULTS_DIR / "mission_dashboard.json").write_text(
        dashboard.model_dump_json(indent=2), encoding="utf-8"
    )

    reference = {
        "_about": (
            "Offline demo reference for the invention lab (trimmed for readability). MOCKED concept "
            "(the front-door stub, no model call) + REAL deterministic engine geometry. NOT a live "
            "result and NOT physically validated. Regenerate the full artifacts with "
            "`uv run python scripts/capture_lab_demo.py`."
        ),
        "problem_id": problem.id,
        "invention_package": _trimmed_package(package),
        "mission_dashboard": dashboard.model_dump(mode="json"),
    }
    _REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    _REFERENCE.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {_RESULTS_DIR / 'invention_package.json'}")
    print(f"wrote {_RESULTS_DIR / 'mission_dashboard.json'}")
    print(f"wrote {_REFERENCE} (committed reference)")
    print(f"\npart: {package.part_name}  disposition: {package.disposition}")
    print(f"concept: {plan.recommended().name} ({plan.recommended().primary_mechanism.value})")
    print(f"maturity: {dashboard.maturity}  evidence: {dashboard.evidence_tier}")
    print(f"banner: {dashboard.banner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
