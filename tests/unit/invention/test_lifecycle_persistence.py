"""Wire mission persistence into the lifecycle + the end-to-end lab smoke (ADR-0053, ladder items 2-3).

``run_and_persist_lifecycle`` runs the offline lifecycle (front-door client + engine ``produce`` both
mocked), persists the resulting ``MissionLog`` via an injected store, and the mission reloads by
``mission_id`` and feeds the dashboard -- the whole lab spine (problem -> package -> mission -> dashboard)
proven end-to-end with NO model call and NO spend. The store round-trips through the real
``mission_events`` records, so the offline fake exercises the same serialization the prod store uses."""

from __future__ import annotations

from typing import Any

from forge_mind.agentic.confidence import standard_lane_label
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.invention import (
    ConceptBrief,
    ConceptPlan,
    FeasibilityVerdict,
    GeneratedPart,
    Mechanism,
    run_and_persist_lifecycle,
)
from forge_mind.invention.dashboard import build_dashboard
from forge_mind.invention.mission import MissionLog
from forge_mind.kit import build_kit
from forge_mind.learning.mission_records import log_from_rows, rows_for_log
from forge_mind.llm.cost import LLMUsage
from forge_mind.types import Vec3

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)


def _plan() -> ConceptPlan:
    return ConceptPlan(
        problem_restated="route two cables along a desk edge without fasteners",
        feasibility_verdict=FeasibilityVerdict.LIKELY_FEASIBLE,
        feasibility_note="a snap-fit clip is a routine FDM print",
        concepts=[
            ConceptBrief(
                concept_id="c1",
                name="cantilever C-clip",
                primary_mechanism=Mechanism.SNAP_FIT,
                approach="a C-shaped jaw that flexes over the desk edge",
                obvious_baseline="a screwed-on bracket",
                difference_from_obvious="no fasteners; the jaw flexes to grip",
            ),
            ConceptBrief(
                concept_id="c2",
                name="friction saddle",
                primary_mechanism=Mechanism.FRICTION_FIT,
                approach="a weighted saddle resting on the edge",
                obvious_baseline="an adhesive clip",
                difference_from_obvious="removable, leaves no residue",
            ),
        ],
        recommended_concept_id="c1",
        recommendation_reason="simplest, needs no hardware",
    )


class _StubClient:
    """Returns a fixed ConceptPlan offline (no model, no spend) -- the front-door seam."""

    def __init__(self, plan: ConceptPlan) -> None:
        self.plan = plan

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[Any],
        model: str,
        max_tokens: int,
        cache_system: bool = True,
    ) -> tuple[Any, LLMUsage]:
        return self.plan, LLMUsage(model=model, input_tokens=1, output_tokens=1)

    def generate_structured_with_image(self, **_: Any) -> tuple[Any, LLMUsage]:
        raise AssertionError("the lifecycle does not use images")


def _produce(_prompt: str) -> GeneratedPart:
    program = PartProgram(
        metadata=Metadata(name="cable-clip", print_orientation=_UP_Z),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return GeneratedPart(
        part_name="cable-clip",
        disposition="shipped",
        kit=build_kit(program),
        confidence=standard_lane_label(0),
        artifacts={"stl": "/tmp/part.stl", "glb": "/tmp/part.glb"},
    )


class _FakeStore:
    """A ``MissionPersister`` + loader that round-trips through the real ``mission_events`` records
    (no supabase-py): ``append_log`` upserts idempotently by ``(mission_id, seq)``; ``load`` folds back."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[int, dict[str, Any]]] = {}

    def append_log(self, log: MissionLog) -> None:
        rows = self._rows.setdefault(log.mission_id, {})
        for row in rows_for_log(log):
            rows[int(row["seq"])] = row

    def load(self, mission_id: str) -> MissionLog | None:
        rows = self._rows.get(mission_id)
        if not rows:
            return None
        return log_from_rows([rows[seq] for seq in sorted(rows)])


def test_run_and_persist_then_reload_round_trips() -> None:
    store = _FakeStore()
    _package, _plan_out, log = run_and_persist_lifecycle(
        "cables fall off my desk edge",
        produce=_produce,
        mission_id="m-1",
        store=store,
        client=_StubClient(_plan()),
    )
    reloaded = store.load("m-1")
    assert reloaded is not None
    # the persisted mission replays to exactly the produced log
    assert reloaded.state() == log.state()
    assert reloaded.state().selected_concept_id == "c1"
    assert reloaded.state().current_stage == "package"


def test_persisted_mission_feeds_the_dashboard_honestly() -> None:
    store = _FakeStore()
    run_and_persist_lifecycle(
        "cables fall off my desk edge",
        produce=_produce,
        mission_id="m-1",
        store=store,
        client=_StubClient(_plan()),
    )
    reloaded = store.load("m-1")
    assert reloaded is not None
    dash = build_dashboard(reloaded.state(), reloaded.events())
    assert dash.mission_id == "m-1"
    assert dash.current_stage == "package"
    assert dash.validation_state.passed >= 1
    # the timeline carries both produced artifacts
    assert {p.kind for p in dash.artifact_previews} >= {"stl", "glb"}
    # honest labels: the NOT-PHYSICALLY-VALIDATED risk is always present...
    assert any("Not physically validated" in risk for risk in dash.unresolved_risks)
    # ...and maturity / banner stay absent until the package itself is persisted (ADR-0052/0053 gap).
    assert dash.maturity is None
    assert dash.banner is None


def test_load_unknown_mission_is_none() -> None:
    assert _FakeStore().load("nope") is None
