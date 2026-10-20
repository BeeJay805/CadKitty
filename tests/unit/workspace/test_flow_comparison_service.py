"""Shared flow validation and proposal gates at the study orchestration boundary."""

import copy
from types import SimpleNamespace

import pytest

from forge_mind.workspace import design_study_service as service_module
from forge_mind.workspace.design_study_models import AcceptStudyCandidate, DesignStudySpec
from forge_mind.workspace.flow_comparison import common_flow_validation
from forge_mind.workspace.store import WorkspaceError


def spec():
    return DesignStudySpec.model_validate(
        {
            "name": "Shared flow comparison",
            "objectives": [{"metric": "flow.drag_n", "direction": "minimize"}],
            "evaluators": [
                {"kind": "geometry"},
                {
                    "kind": "flow",
                    "settings": {
                        "density_kg_m3": 1,
                        "dynamic_viscosity_pa_s": 1,
                        "velocity_m_s": [1, 0, 0],
                        "reference_area_m2": 1,
                        "reference_length_m": 2,
                    },
                },
            ],
        }
    )


def candidate(letter, value, level):
    m, d = level
    return {
        "id": letter * 32,
        "label": letter,
        "revision_id": letter * 32,
        "status": "complete",
        "eligible": True,
        "feasible": True,
        "metrics": {"flow.drag_n": value},
        "metric_uncertainty": {"flow.drag_n": 0.01},
        "evaluations": [
            {
                "kind": "flow",
                "result": {
                    "final_validation": {
                        "confirmed": True,
                        "mesh_level": m,
                        "domain_level": d,
                        "protocol": "controlled_cross_v1",
                        "reference_length_m": 2,
                    },
                    "refinement_runs": [
                        {
                            "stage": f"m{x}d{y}",
                            "result": {
                                "metrics": {"drag_n": value},
                                "convergence": {"force_relative_spread": 0},
                            },
                        }
                        for x, y in [(m, d), (m - 1, d), (m - 2, d), (m, d - 1), (m, d - 2)]
                    ],
                },
            }
        ],
    }


def orchestration(monkeypatch, values, evaluate):
    records = copy.deepcopy(values)
    study = {"id": "study", "spec": spec().model_dump()}
    calls = []
    monkeypatch.setattr(service_module, "_candidates", lambda *_: copy.deepcopy(records))

    def run(_service, _sid, _owner, _study, current, cancelled, floor):
        calls.append((current["id"], floor))
        replacement = evaluate(current, floor)
        current.clear()
        current.update(replacement)
        index = next(i for i, item in enumerate(records) if item["id"] == current["id"])
        records[index] = copy.deepcopy(current)

    monkeypatch.setattr(service_module, "_evaluate_candidate", run)
    return (
        records,
        calls,
        lambda cancelled=lambda: False: service_module._confirm_flow_candidates(
            SimpleNamespace(), "session", "owner", study, cancelled
        ),
    )


def test_confirmation_recomputes_finalists_after_refinement_changes_scores(monkeypatch):
    values = [candidate("a", 100, (3, 3)), candidate("b", 80, (2, 2)), candidate("c", 85, (2, 2))]
    records, calls, confirm = orchestration(
        monkeypatch,
        values,
        lambda current, floor: candidate(
            current["label"],
            95 if current["label"] == "b" else current["metrics"]["flow.drag_n"],
            floor,
        ),
    )
    confirm()
    assert [item[0] for item in calls] == ["b" * 32, "c" * 32]
    assert common_flow_validation(records, "a" * 32)
    assert (
        service_module.compare_candidates(spec(), records, "a" * 32)["best_candidate_id"]
        == "c" * 32
    )


def test_shared_floor_repeats_when_one_finalist_needs_further_refinement(monkeypatch):
    values = [candidate("a", 100, (2, 4)), candidate("b", 80, (4, 2))]

    def evaluate(current, floor):
        return candidate(
            current["label"],
            current["metrics"]["flow.drag_n"],
            (5, 4) if current["label"] == "b" else floor,
        )

    records, calls, confirm = orchestration(monkeypatch, values, evaluate)
    confirm()
    assert calls == [("a" * 32, (4, 4)), ("b" * 32, (4, 4)), ("a" * 32, (5, 4))]
    assert common_flow_validation(records, "a" * 32)


def test_cancellation_then_resume_reuses_saved_confirmation_levels(monkeypatch):
    stopped = [False]

    def evaluate(current, floor):
        stopped[0] = True
        return candidate(current["label"], current["metrics"]["flow.drag_n"], floor)

    records, calls, confirm = orchestration(
        monkeypatch, [candidate("a", 100, (2, 4)), candidate("b", 80, (4, 2))], evaluate
    )
    confirm(lambda: stopped[0])
    assert len(calls) == 1
    assert (
        service_module.compare_candidates(spec(), records, "a" * 32)["conclusion"] == "inconclusive"
    )
    stopped[0] = False
    confirm(lambda: stopped[0])
    assert calls == [("a" * 32, (4, 4)), ("b" * 32, (4, 4))]
    assert common_flow_validation(records, "a" * 32)


def test_equal_levels_with_different_protocols_stop_instead_of_recalculating(monkeypatch):
    values = [candidate("a", 100, (2, 2)), candidate("b", 80, (2, 2))]
    values[1]["evaluations"][0]["result"]["final_validation"]["protocol"] = "other"
    records, calls, confirm = orchestration(
        monkeypatch, values, lambda *_: pytest.fail("Unexpected solve")
    )
    with pytest.raises(WorkspaceError, match="protocol"):
        confirm()
    assert calls == []
    assert not common_flow_validation(records, "a" * 32)
    assert (
        service_module.compare_candidates(spec(), records, "a" * 32)["conclusion"] == "inconclusive"
    )


def test_combined_sensitivity_prevents_improvement_claim_from_final_scores():
    values = [candidate("a", 100, (2, 2)), candidate("b", 97, (2, 2))]
    for value in values:
        value["metric_uncertainty"]["flow.drag_n"] = 1.8
    result = service_module.compare_candidates(spec(), values, "a" * 32)
    assert result["conclusion"] == "inconclusive"
    assert result["improved_candidate_ids"] == []


def test_resolved_cross_reversal_prevents_service_improvement_claim():
    values = [candidate("a", 100, (2, 2)), candidate("b", 80, (2, 2))]
    values[1]["evaluations"][0]["result"]["refinement_runs"][1]["result"]["metrics"]["drag_n"] = 110
    result = service_module.compare_candidates(spec(), values, "a" * 32)
    assert result["conclusion"] != "improved"
    assert not result.get("improved_candidate_ids")
    assert result["best_candidate_id"] is None
    assert result["pareto_candidate_ids"] == []


def test_unresolved_final_tie_cannot_hide_opposing_resolved_cross_rankings():
    values = [candidate("a", 100, (2, 2)), candidate("b", 100, (2, 2))]
    runs = values[1]["evaluations"][0]["result"]["refinement_runs"]
    # Each axis contracts monotonically and the combined sensitivity is below 2%.
    # The mesh and domain approaches nevertheless favor opposite candidates.
    for run, value in zip(runs, [100, 99.5, 99.4, 100.5, 100.6], strict=True):
        run["result"]["metrics"]["drag_n"] = value
    result = service_module.compare_candidates(spec(), values, "a" * 32)
    assert result["conclusion"] == "inconclusive"
    assert result["best_candidate_id"] is None


@pytest.mark.parametrize("invalid", ["unmatched", "unconfirmed", "infeasible_baseline"])
def test_screening_candidate_cannot_be_proposed_before_shared_validation(monkeypatch, invalid):
    baseline = candidate("a", 100, (2, 2))
    other = candidate("b", 80, (3, 2) if invalid == "unmatched" else (2, 2))
    if invalid == "unconfirmed":
        other["evaluations"][0]["result"]["final_validation"]["confirmed"] = False
    if invalid == "infeasible_baseline":
        baseline["feasible"] = False
    study = {
        "id": "study",
        "current": True,
        "spec": spec().model_dump(),
        "input_version": 1,
        "candidates": [baseline, other],
        "comparison": {"baseline_candidate_id": baseline["id"]},
    }
    monkeypatch.setattr(service_module, "read_study", lambda *_: study)
    service = SimpleNamespace(
        store=SimpleNamespace(enqueue=lambda *_: pytest.fail("Unconfirmed proposal queued")),
        _load_revision=lambda *_: {"id": "b" * 32, "document": {}, "contract": {}},
        start=lambda: None,
    )
    with pytest.raises(WorkspaceError):
        service_module.propose_candidate(
            service,
            "session",
            "owner",
            "study",
            AcceptStudyCandidate(candidate_id="b" * 32, expected_current_revision="a" * 32),
        )
