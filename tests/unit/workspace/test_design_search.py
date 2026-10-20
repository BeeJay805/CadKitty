"""Numerical search controls; these fixtures do not supply CAD or physical evidence."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from forge_mind.workspace import design_study_service as service
from forge_mind.workspace.design_search import choose_search_anchor
from forge_mind.workspace.design_study_models import DesignStudySpec, StudyEvaluator


def spec():
    return DesignStudySpec.model_validate(
        {
            "name": "Two independent clearance repairs",
            "objectives": [{"metric": "geometry.volume_mm3", "direction": "minimize"}],
            "constraints": [
                {"metric": "geometry.extent_x_mm", "operator": "ge", "value": 1},
                {"metric": "geometry.extent_y_mm", "operator": "ge", "value": 1},
            ],
            "variables": {"x": [0, 2], "y": [0, 2]},
            "evaluators": [{"kind": "geometry"}],
        }
    )


def candidate(identifier="baseline", *, x=0, y=0, volume=10, feasible=False):
    return {
        "id": identifier,
        "status": "complete",
        "eligible": True,
        "feasible": feasible,
        "parameters": {"x": x, "y": y},
        "metrics": {
            "geometry.extent_x_mm": x,
            "geometry.extent_y_mm": y,
            "geometry.volume_mm3": volume,
        },
        "original_requirement_failures": [],
    }


def test_verified_feasible_best_wins_without_mutating_the_ledger():
    baseline = candidate(x=2, y=2, feasible=True)
    first = candidate("first", x=1, y=1, volume=1, feasible=True)
    ledger = [first, baseline]
    before = deepcopy(ledger)
    selected = choose_search_anchor(spec(), ledger, {"best_candidate_id": "baseline"})
    assert selected["candidate_id"] == "baseline"
    assert selected["status"] == "feasible"
    assert [row["residual"] for row in selected["violations"]] == [0, 0]
    assert ledger == before


def test_constraint_residuals_precede_objectives_and_preserve_frozen_order():
    lighter = candidate("lighter", x=0, y=1, volume=1)
    progressing = candidate("progressing", x=1, y=0, volume=1000)
    selected = choose_search_anchor(spec(), [lighter, progressing], {})
    assert selected["candidate_id"] == "progressing"
    assert selected["status"] == "search_only"
    assert selected["violations"] == [
        {
            "metric": "geometry.extent_x_mm",
            "operator": "ge",
            "value": 1,
            "measured": 1,
            "residual": 0,
        },
        {
            "metric": "geometry.extent_y_mm",
            "operator": "ge",
            "value": 1,
            "measured": 0,
            "residual": 1,
        },
    ]
    assert progressing["feasible"] is False


def test_feasible_candidate_beats_lighter_constraint_violation_without_promotion():
    good = candidate("good", x=1, y=1, volume=100)
    light = candidate("light", x=0, y=0, volume=1)
    selected = choose_search_anchor(spec(), [light, good], {})
    assert selected["candidate_id"] == "good"
    assert selected["status"] == "search_only"
    assert good["feasible"] is False


def test_upper_bound_residual_and_objective_ties_use_stable_ledger_order():
    protocol = spec()
    protocol.constraints[0].operator = "le"
    first = candidate("first", x=2, y=1, volume=8)
    second = candidate("second", x=2, y=1, volume=8)
    worse = candidate("worse", x=3, y=1, volume=1)
    selected = choose_search_anchor(protocol, [first, second, worse], {})
    assert selected["candidate_id"] == "first"
    assert selected["violations"][0]["residual"] == 1
    protocol.objectives[0].direction = "maximize"
    second["metrics"]["geometry.volume_mm3"] = 9
    assert choose_search_anchor(protocol, [first, second], {})["candidate_id"] == "second"


@pytest.mark.parametrize("metric", ["geometry.volume_mm3", "geometry.extent_x_mm"])
@pytest.mark.parametrize(
    "value", [None, True, False, "1", float("nan"), float("inf"), -float("inf"), 10**1000]
)
def test_unmeasured_or_invalid_metric_never_becomes_an_anchor(metric, value):
    entry = candidate()
    if value is None:
        entry["metrics"].pop(metric)
    else:
        entry["metrics"][metric] = value
    result = choose_search_anchor(spec(), [entry], {"best_candidate_id": entry["id"]})
    assert result["candidate_id"] is None
    assert result["status"] == "unavailable"
    assert result["violations"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"status": "incomplete"},
        {"status": "failed"},
        {"eligible": False},
        {"eligible": 1},
        {"metrics": None},
        {"id": None},
        {"original_requirement_failures": None},
        {"original_requirement_failures": [{"requirement_id": "r1", "status": "fail"}]},
    ],
)
def test_ineligible_or_owner_violating_candidate_is_not_a_repair_tradeoff(change):
    entry = candidate(x=1, y=1, feasible=True)
    entry.update(change)
    result = choose_search_anchor(spec(), [entry], {"best_candidate_id": "baseline"})
    assert result["candidate_id"] is None


def test_missing_owner_failure_evaluation_is_not_assumed_clear():
    entry = candidate()
    entry.pop("original_requirement_failures")
    assert choose_search_anchor(spec(), [entry], {})["candidate_id"] is None


def test_overflowed_constraint_residual_cannot_be_ranked():
    protocol = spec()
    protocol.constraints[0].value = 1e308
    entry = candidate(x=-1e308)
    assert choose_search_anchor(protocol, [entry], {})["candidate_id"] is None


@pytest.mark.parametrize("reported_best", [None, "baseline"])
def test_inconclusive_flow_cannot_fall_back_to_a_numerically_attractive_anchor(reported_best):
    protocol = spec()
    protocol.evaluators.append(
        StudyEvaluator(
            kind="flow",
            settings={
                "density_kg_m3": 1.2,
                "dynamic_viscosity_pa_s": 0.000018,
                "velocity_m_s": [10, 0, 0],
                "reference_area_m2": 0.01,
                "reference_length_m": 0.1,
            },
        )
    )
    result = choose_search_anchor(
        protocol,
        [candidate(x=1, y=1, feasible=True)],
        {"conclusion": "inconclusive", "best_candidate_id": reported_best},
    )
    assert result["candidate_id"] is None
    assert "comparable" in result["reason"]


@pytest.fixture
def numerical_run(monkeypatch):
    protocol = spec()
    ledger = [candidate()]
    saved_runs = []
    evaluated = []
    interrupted = {"cancelled": False, "cancel_after_first": False}
    run = {"id": "run", "rounds_completed": 0}

    class Connection:
        def execute(self, *_):
            return self

        def fetchone(self):
            return None

    @contextmanager
    def connect():
        yield Connection()

    store = SimpleNamespace(connect=connect, event=lambda *_args, **_kwargs: None)
    workspace = SimpleNamespace(store=store)
    study = {"id": "study", "spec": protocol.model_dump()}
    monkeypatch.setattr(service, "_study", lambda *_: study)
    monkeypatch.setattr(service, "_decode", lambda *_: run)
    monkeypatch.setattr(service, "_save_run", lambda _, value: saved_runs.append(deepcopy(value)))
    monkeypatch.setattr(service, "_fresh", lambda *_: None)
    monkeypatch.setattr(service, "_candidates", lambda *_: ledger)
    monkeypatch.setattr(service, "_verify_evaluations", lambda *_: None)
    monkeypatch.setattr(service, "_confirm_flow_candidates", lambda *_: None)
    monkeypatch.setattr(
        service,
        "read_study",
        lambda *_: {
            "candidates": ledger,
            "comparison": service.compare_candidates(protocol, ledger, "baseline"),
        },
    )

    def add(_workspace, _sid, _owner, _study_id, request):
        result = {
            "id": f"candidate{len(ledger)}",
            "parameters": request.parameters,
            "status": "pending",
        }
        ledger.append(result)
        return result

    def evaluate(_workspace, _sid, _owner, _study, entry, _cancelled):
        values = entry["parameters"]
        evaluated.append(dict(values))
        entry.update(
            candidate(
                entry["id"],
                x=values["x"],
                y=values["y"],
                volume=10 + values["x"] + values["y"],
                feasible=values["x"] >= 1 and values["y"] >= 1,
            )
        )
        if interrupted["cancel_after_first"]:
            interrupted["cancelled"] = True

    monkeypatch.setattr(service, "add_candidate", add)
    monkeypatch.setattr(service, "_evaluate_candidate", evaluate)
    job = {
        "id": "job",
        "session": "session",
        "owner": "owner",
        "study_id": "study",
        "run_id": "run",
        "search": {"rounds": 1, "step_fraction": 0.5},
    }
    return workspace, job, ledger, evaluated, saved_runs, interrupted


def test_failed_baseline_advances_through_two_sequential_variable_repairs(numerical_run):
    workspace, job, ledger, evaluated, runs, interrupted = numerical_run
    service.run_study_job(workspace, job, lambda: interrupted["cancelled"])
    assert evaluated == [{"x": 1, "y": 0}, {"x": 1, "y": 1}]
    assert ledger[0]["feasible"] is False
    assert ledger[1]["feasible"] is False
    assert ledger[2]["feasible"] is True
    assert runs[-1]["state"] == "completed"
    assert runs[-1]["rounds_completed"] == 1
    assert runs[-1]["search_anchor"]["candidate_id"] == "candidate1"
    assert runs[-1]["search_anchor"]["status"] == "search_only"
    comparison = service.compare_candidates(spec(), ledger, "baseline")
    assert comparison["best_candidate_id"] == "candidate2"
    assert comparison["conclusion"] == "inconclusive"


def test_unavailable_anchor_stops_without_building_an_unmeasured_baseline(numerical_run):
    workspace, job, ledger, evaluated, runs, interrupted = numerical_run
    ledger[0]["metrics"] = {}
    service.run_study_job(workspace, job, lambda: interrupted["cancelled"])
    assert evaluated == []
    assert len(ledger) == 1
    assert runs[-1]["rounds_completed"] == 0
    assert runs[-1]["search_anchor"]["status"] == "unavailable"
    assert runs[-1]["stop_reason"] == runs[-1]["search_anchor"]["reason"]


def test_cancellation_after_first_repair_admits_no_second_coordinate(numerical_run):
    workspace, job, ledger, evaluated, runs, interrupted = numerical_run
    interrupted["cancel_after_first"] = True
    service.run_study_job(workspace, job, lambda: interrupted["cancelled"])
    assert evaluated == [{"x": 1, "y": 0}]
    assert len(ledger) == 2
    assert runs[-1]["state"] == "cancelled"
    assert not any(entry["feasible"] for entry in ledger)
