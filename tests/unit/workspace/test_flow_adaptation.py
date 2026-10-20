"""Numerical acceptance controls independent of the native solver implementation."""

import copy
import math
from pathlib import Path

import pytest

from forge_mind.workspace.flow_adaptation import adaptive_flow, assess_cross, effective_settings
from forge_mind.workspace.flow_comparison import common_flow_validation, rank_stable
from forge_mind.workspace.flow_models import FlowSettings


def settings(**extra):
    return FlowSettings.model_validate(
        {
            "density_kg_m3": 1,
            "dynamic_viscosity_pa_s": 1,
            "velocity_m_s": [1, 0, 0],
            "reference_area_m2": 1,
            "reference_length_m": 2,
            "mesh": {"fine_scale": 0.8},
            **extra,
        }
    )


def result(m, d, value, noise=0):
    return {
        "eligible": True,
        "status": "complete",
        "metrics": {"drag_n": value, "drag_coefficient": value / 2},
        "provenance": {"mesh_sha256": f"{m}-{d}"},
        "convergence": {"force_relative_spread": noise},
        "artifacts": {},
    }


def cross(value=lambda m, d: 100 + 0.2**m + 0.3**d, noise=0):
    return {
        (m, d): result(m, d, value(m, d), noise)
        for m, d in [(2, 2), (1, 2), (0, 2), (2, 1), (2, 0)]
    }


def test_cross_uses_two_controlled_changes_and_larger_change_per_axis():
    assessment = assess_cross(cross(), 2, 2, 0.02, confirmation=True)
    assert assessment["eligible"]
    assert assessment["drag_n"]["mesh_delta"] == pytest.approx(0.8)
    assert assessment["drag_n"]["domain_delta"] == pytest.approx(0.7)
    assert assessment["drag_n"]["relative_change"] == pytest.approx(1.5 / 100.13)


def test_small_last_pair_does_not_hide_large_preceding_change():
    runs = cross(lambda m, d: 100 + (10 if m == 0 else 0.01**m))
    assert assess_cross(runs, 2, 2, 0.02, confirmation=False)["eligible"]
    assert not assess_cross(runs, 2, 2, 0.02, confirmation=True)["eligible"]


@pytest.mark.parametrize("corruption", ["mesh", "missing", "noise", "nonfinite", "failed"])
def test_invalid_cross_never_passes(corruption):
    runs = cross()
    if corruption == "mesh":
        runs[0, 2]["provenance"] = runs[1, 2]["provenance"]
    elif corruption == "missing":
        del runs[0, 2]
    elif corruption == "noise":
        runs[0, 2]["convergence"]["force_relative_spread"] = 0.01
    elif corruption == "nonfinite":
        runs[0, 2]["metrics"]["drag_n"] = math.nan
    else:
        runs[0, 2]["eligible"] = False
    assert not assess_cross(runs, 2, 2, 0.02, confirmation=True)["eligible"]


def test_oscillating_cross_above_noise_remains_unverified():
    runs = cross(lambda m, d: [100.3, 100.0, 100.2][m])
    assessment = assess_cross(runs, 2, 2, 0.02, confirmation=True)
    assert assessment["drag_n"]["oscillating"]
    assert not assessment["eligible"]


@pytest.mark.parametrize("noise", [math.nan, -0.01])
def test_a_later_invalid_noise_cannot_hide_behind_a_valid_maximum(noise):
    runs = cross(noise=0.001)
    runs[0, 2]["convergence"]["force_relative_spread"] = noise
    assert not assess_cross(runs, 2, 2, 0.02, confirmation=True)["eligible"]


def test_effective_settings_hold_physics_and_physical_sizing_reference():
    base = settings()
    fine = effective_settings(base, 3, 4)
    assert fine.mesh.reference_length_m == 2
    assert fine.mesh.fine_scale == pytest.approx(0.8**3)
    assert fine.mesh.domain_padding_factor == 12 * 1.5**4
    assert fine.mesh.surface_size_factor == base.mesh.surface_size_factor
    assert fine.mesh.first_layer_size_factor == base.mesh.first_layer_size_factor
    assert fine.numerics == base.numerics
    assert fine.reference_area_m2 == base.reference_area_m2
    assert base.mesh.reference_length_m is None


def install_cache(monkeypatch, tmp_path, values, *, fail=None, cancel=None, grading_shift=0):
    calls = []

    class Cache:
        def __init__(self, *args):
            pass

        def update_progress(self, **kwargs):
            pass

        def run(self, effective, label):
            m = (
                0
                if effective.mesh.refinement == "coarse"
                else round(math.log(effective.mesh.fine_scale) / math.log(0.8))
            )
            d = round(math.log(effective.mesh.domain_padding_factor / 12) / math.log(1.5))
            calls.append((m, d))
            measured = result(m, d, values(m, d) + effective.mesh.grading_rate * grading_shift)
            if fail == (m, d):
                measured.update(eligible=False, errors=["explicit stage failure"], metrics={})
            if cancel is not None and len(calls) == 2:
                cancel[0] = True
            location = tmp_path / f"stage-{m}-{d}"
            location.mkdir(exist_ok=True)
            return measured, location, False

    monkeypatch.setattr("forge_mind.workspace.flow_stage_cache.FlowStageCache", Cache)
    return calls


def run(tmp_path, base=None, cancelled=lambda: False, floor=(0, 0)):
    return adaptive_flow(
        Path("unused.step"),
        tmp_path / "evaluation",
        base or settings(),
        0.02,
        cancelled,
        cache_root=tmp_path / "cache",
        identity={"study": "test"},
        minimum_level=floor,
    )


def test_initial_failure_refines_dominant_axis_then_confirms(monkeypatch, tmp_path):
    calls = install_cache(monkeypatch, tmp_path, lambda m, d: 100 + 2 * 0.2**m + 12 * 0.3**d)
    measured = run(tmp_path)
    assert measured["eligible"]
    assert measured["sensitivity_history"][0]["eligible"] is False
    assert measured["sensitivity_history"][0]["promoted_axis"] == "domain_delta"
    assert measured["final_validation"]["confirmed"]
    assert len(calls) == len(set(calls))
    m, d = measured["final_validation"]["mesh_level"], measured["final_validation"]["domain_level"]
    assert {(m, d), (m - 1, d), (m - 2, d), (m, d - 1), (m, d - 2)} <= set(calls)
    assert measured["metrics"]["drag_n"] == pytest.approx(100 + 2 * 0.2**m + 12 * 0.3**d)


def test_confirmation_failure_never_restores_earlier_passing_score(monkeypatch, tmp_path):
    install_cache(monkeypatch, tmp_path, lambda m, d: 100, fail=(0, 2))
    measured = run(tmp_path)
    assert not measured["eligible"]
    assert measured["final_validation"]["confirmed"] is False
    assert measured["metrics"] == {}
    assert measured["sensitivity"]["eligible"] is False
    assert "explicit stage failure" in measured["errors"][0]


def test_cancellation_preserves_no_eligible_claim(monkeypatch, tmp_path):
    cancel = [False]
    calls = install_cache(monkeypatch, tmp_path, lambda m, d: 100, cancel=cancel)
    measured = run(tmp_path, cancelled=lambda: cancel[0])
    assert len(calls) == 2
    assert not measured["eligible"]
    assert "cancelled" in measured["errors"][0]


def test_requested_common_floor_is_actually_confirmed(monkeypatch, tmp_path):
    calls = install_cache(monkeypatch, tmp_path, lambda m, d: 100 + 0.2**m + 0.3**d)
    measured = run(tmp_path, floor=(4, 3))
    assert measured["eligible"]
    assert measured["final_validation"]["mesh_level"] == 4
    assert measured["final_validation"]["domain_level"] == 3
    assert {(2, 3), (3, 3), (4, 3), (4, 2), (4, 1)} <= set(calls)


def test_noncontracting_sensitivity_stops_without_hidden_round_cap(monkeypatch, tmp_path):
    calls = install_cache(monkeypatch, tmp_path, lambda m, d: 100 + 10 * d)
    measured = run(tmp_path)
    assert not measured["eligible"]
    assert "not contracting" in measured["errors"][0]
    assert len(calls) < 20
    assert settings().adaptation.max_refinements is None


def test_fast_but_inaccurate_grading_is_rejected_before_refinement(monkeypatch, tmp_path):
    calls = install_cache(monkeypatch, tmp_path, lambda m, d: 100, grading_shift=50)
    measured = run(tmp_path, settings(mesh={"fine_scale": 0.8, "grading_rate": 0.1}))
    assert calls == [(0, 0), (0, 0)]
    assert not measured["eligible"]
    assert "grading perturbation" in measured["errors"][0]
    assert measured["metrics"] == {}


def test_accepted_grading_perturbation_still_consumes_sensitivity_budget(monkeypatch, tmp_path):
    install_cache(monkeypatch, tmp_path, lambda m, d: 100, grading_shift=5)
    measured = run(tmp_path, settings(mesh={"fine_scale": 0.8, "grading_rate": 0.1}))
    assert measured["eligible"]
    assert measured["grading_control"]["relative_allowance"]["drag_n"] == 0.005
    assert measured["sensitivity"]["drag_n"]["relative_change"] == 0.005
    assert measured["metric_uncertainty"]["drag_n"] == pytest.approx(0.5025)


def candidate(id, level=(2, 2), offset=0):
    m, d = level
    points = [(m, d), (m - 1, d), (m - 2, d), (m, d - 1), (m, d - 2)]
    return {
        "id": id,
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
                        {"stage": f"m{x}d{y}", "result": result(x, y, 100 + offset)}
                        for x, y in points
                    ],
                },
            }
        ],
    }


def test_common_comparison_rejects_independent_stopping_levels():
    baseline = candidate("baseline")
    other = candidate("candidate", (3, 2))
    assert not common_flow_validation([baseline, other], "baseline")
    assert common_flow_validation([baseline, candidate("candidate")], "baseline")
    assert not common_flow_validation([other], "baseline")


def test_resolved_ranking_reversal_cannot_claim_improvement():
    baseline = candidate("baseline")
    other = candidate("candidate", offset=-10)
    assert rank_stable(other, baseline)
    reversed_candidate = copy.deepcopy(other)
    reversed_candidate["evaluations"][0]["result"]["refinement_runs"][1]["result"]["metrics"][
        "drag_n"
    ] = 110
    assert not rank_stable(reversed_candidate, baseline)
