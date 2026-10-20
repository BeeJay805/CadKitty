"""The invention lifecycle orchestrator (ADR-0048) — problem -> concept -> part -> package.

Wires the front door to the proven CAD engine WITHOUT duplicating the API's per-lane composition:
the caller injects a ``produce`` callable that turns a prompt into a :class:`GeneratedPart`
(part_name, disposition, kit, confidence, artifacts). Both seams — the front door's ``client`` and
``produce`` — are injectable, so the whole lifecycle is testable offline with no spend; the live
measurement waits for Fable.

``FORGE_INVENTION`` gates whether the lifecycle is OFFERED. With it off the existing /generate path
is byte-identical (this module is never reached). REPORTED throughout: the validator inside
``produce`` is the sole verdict-issuer; nothing here gates.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from forge_mind.agentic.confidence import ConfidenceLabel
from forge_mind.invention.concept import ConceptBrief, ConceptPlan
from forge_mind.invention.frontdoor import run_front_door
from forge_mind.invention.labels import EvidenceTier
from forge_mind.invention.mission import (
    ArtifactRecord,
    DecisionRecord,
    EventType,
    MissionLog,
    RejectedPath,
    ValidationRecord,
)
from forge_mind.invention.package import InventionPackage, build_invention_package
from forge_mind.kit import Kit
from forge_mind.llm.provider import LLMClientLike

_INVENTION_ENV = "FORGE_INVENTION"


def invention_enabled() -> bool:
    """True iff ``FORGE_INVENTION=1`` — the live lifecycle is OFFERED. Default off → the existing
    /generate path is byte-identical (the lifecycle is never reached)."""
    return os.environ.get(_INVENTION_ENV, "").strip() == "1"


@dataclass(frozen=True)
class GeneratedPart:
    """The composable outputs of one engine run, as the lifecycle needs them. A real ``produce``
    adapter fills this from ``produce_part`` + ``build_kit`` + the confidence label, reusing the
    engine's own composition (never re-derived here); tests fill it with a stub."""

    part_name: str
    disposition: str
    kit: Kit
    confidence: ConfidenceLabel | None = None
    artifacts: dict[str, str] = field(default_factory=dict)


def concept_to_prompt(problem: str, concept: ConceptBrief) -> str:
    """Turn the chosen concept into a part-generation prompt for the proven engine. Deterministic:
    the concept's name + approach become the build request; the original problem rides as context."""
    return f"{concept.name}: {concept.approach.strip()} (solving: {problem.strip()})"


def run_lifecycle(
    problem: str,
    *,
    produce: Callable[[str], GeneratedPart],
    client: LLMClientLike | None = None,
    model: str | None = None,
) -> tuple[InventionPackage, ConceptPlan]:
    """Run problem -> concept -> part -> package.

    ONE front-door brain pass proposes 2-3 concepts + a recommendation; the recommended concept's
    prompt feeds ``produce`` (the engine); the result is composed into a REPORTED InventionPackage
    carrying the concept + the advisory feasibility + the always-present banner. Both seams are
    injectable, so this is fully offline-testable. Returns the package and the full plan (so the
    caller can surface the alternatives, not just the chosen one).
    """
    plan, _ = run_front_door(problem, client=client, model=model)
    chosen = plan.recommended()
    part = produce(concept_to_prompt(problem, chosen))
    package = build_invention_package(
        part_name=part.part_name,
        disposition=part.disposition,
        kit=part.kit,
        confidence=part.confidence,
        artifacts=part.artifacts,
        problem_statement=problem,
        feasibility=plan.feasibility(),
        concept=chosen,
    )
    return package, plan


def build_mission_log(
    problem: str,
    plan: ConceptPlan,
    package: InventionPackage,
    *,
    mission_id: str,
    at: str = "",
) -> MissionLog:
    """Compose the append-only :class:`MissionLog` for one finished lifecycle run from its typed
    outputs -- the mission spine's PRODUCER (deterministic, no model call). Without this, the spine
    (the ``mission_events`` store + the dashboard) has nothing to record; with it, a front-door +
    engine run becomes a real, replayable ``MissionState``.

    Event order: MISSION_CREATED (the locked problem) -> CONCEPT_SELECTED (the recommended concept) ->
    CONCEPT_REJECTED for each runner-up (RETAINED with a reopen condition, never erased) ->
    DECISION_RECORDED (the recommendation -- ``ai_hypothesis``, a model judgment, not a checked fact) ->
    ARTIFACT_CREATED per produced artifact -> VALIDATION_PASSED/FAILED (mapped from the honest
    disposition) -> PACKAGE_ASSEMBLED. The decision never claims authority it lacks, and a runner-up
    is kept as memory with what would make it worth reopening.
    """
    chosen_id = plan.recommended_concept_id
    log = MissionLog(mission_id, problem, at=at)
    log.append(
        EventType.CONCEPT_SELECTED,
        summary=f"selected concept {chosen_id}",
        at=at,
        stage="concept",
        concept_id=chosen_id,
    )
    for concept in plan.concepts:
        if concept.concept_id == chosen_id:
            continue
        log.append(
            EventType.CONCEPT_REJECTED,
            summary=f"runner-up: {concept.name}",
            at=at,
            rejected_path=RejectedPath(
                concept_id=concept.concept_id,
                idea=f"{concept.name}: {concept.approach}",
                reason=f"not recommended first; {plan.recommendation_reason}",
                reopen_conditions=["the recommended concept fails validation or a physical test"],
            ),
        )
    log.append(
        EventType.DECISION_RECORDED,
        summary=f"build {chosen_id} first",
        at=at,
        stage="concept",
        decision=DecisionRecord(
            decision=f"build concept {chosen_id} first",
            rationale=plan.recommendation_reason,
            evidence_refs=[chosen_id],
            evidence_tier=EvidenceTier.AI_HYPOTHESIS,
        ),
    )
    for kind, ref in sorted(package.artifacts.items()):
        log.append(
            EventType.ARTIFACT_CREATED,
            summary=f"{kind} artifact",
            at=at,
            stage="cad",
            artifact=ArtifactRecord(artifact_id=kind, kind=kind, ref=ref),
        )
    passed = package.disposition in ("shipped", "shipped_with_caveats")
    log.append(
        EventType.VALIDATION_PASSED if passed else EventType.VALIDATION_FAILED,
        summary=f"validator disposition: {package.disposition}",
        at=at,
        stage="validation",
        validation=ValidationRecord(
            passed=passed,
            failed_checks=[] if passed else [f"disposition: {package.disposition}"],
        ),
    )
    log.append(
        EventType.PACKAGE_ASSEMBLED,
        summary=f"package assembled for {package.part_name}",
        at=at,
        stage="package",
    )
    return log


def run_lifecycle_with_mission(
    problem: str,
    *,
    produce: Callable[[str], GeneratedPart],
    mission_id: str,
    client: LLMClientLike | None = None,
    model: str | None = None,
    at: str = "",
) -> tuple[InventionPackage, ConceptPlan, MissionLog]:
    """Run the lifecycle and ALSO emit its append-only :class:`MissionLog` (the spine's producer). A
    thin wrapper over :func:`run_lifecycle`: the package/plan are unchanged; the log composes from them
    deterministically. Persist it with ``learning.MissionEventStore`` and read it with
    ``invention.dashboard.build_dashboard``."""
    package, plan = run_lifecycle(problem, produce=produce, client=client, model=model)
    log = build_mission_log(problem, plan, package, mission_id=mission_id, at=at)
    return package, plan, log


class MissionPersister(Protocol):
    """The persistence seam the lifecycle needs: append a whole :class:`MissionLog`.
    ``learning.MissionEventStore`` satisfies it in prod; a fake store satisfies it in tests. Injected,
    so this module never imports the supabase-py-bound store -- which would not even import in the
    deployed server (ADR-0052), and which the offline lifecycle must not require."""

    def append_log(self, log: MissionLog) -> None: ...


def run_and_persist_lifecycle(
    problem: str,
    *,
    produce: Callable[[str], GeneratedPart],
    mission_id: str,
    store: MissionPersister,
    client: LLMClientLike | None = None,
    model: str | None = None,
    at: str = "",
) -> tuple[InventionPackage, ConceptPlan, MissionLog]:
    """Run the lifecycle, emit its :class:`MissionLog`, and PERSIST it via the injected ``store`` --
    the spine's write path, the inverse of the dashboard route's read path. Returns
    ``(package, plan, log)`` like :func:`run_lifecycle_with_mission`, after the mission is durably
    appended, so a later :func:`~forge_mind.invention.dashboard.build_dashboard` over the reloaded log
    reflects this run. ``store`` is injected (``MissionEventStore`` in prod, a fake in tests), keeping
    this module decoupled from the concrete supabase-py binding. Flag-gating is at the offer site
    (:func:`invention_enabled`), exactly like :func:`run_lifecycle` -- this orchestrator is only reached
    when the lifecycle is OFFERED.

    NOTE (ADR-0052/0053 v0 gap): the mission TIMELINE is persisted, not the assembled
    :class:`~forge_mind.invention.package.InventionPackage`, so a dashboard rebuilt from the reloaded
    mission shows the timeline (stage / validation / artifacts / the always-on NOT-PHYSICALLY-VALIDATED
    risk) but not the package-derived maturity / evidence_tier / banner. Persisting the package is a
    separate data-model decision (the next backlog item)."""
    package, plan, log = run_lifecycle_with_mission(
        problem, produce=produce, mission_id=mission_id, client=client, model=model, at=at
    )
    store.append_log(log)
    return package, plan, log


__all__ = [
    "GeneratedPart",
    "MissionPersister",
    "build_mission_log",
    "concept_to_prompt",
    "invention_enabled",
    "run_and_persist_lifecycle",
    "run_lifecycle",
    "run_lifecycle_with_mission",
]
