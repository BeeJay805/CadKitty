"""Known-model controls and deliberately misleading data exercise scientific claims."""

import csv
import io

import pytest
from pydantic import ValidationError

from forge_mind.workspace.experiment_models import (
    ExperimentCondition,
    ExperimentProtocol,
    Observation,
)
from forge_mind.workspace.experiment_service import synthetic_tutorial
from forge_mind.workspace.experiments import (
    compare_hypotheses,
    fit_hypothesis,
    parse_observations,
    predict,
    recommend_thickness,
    suggest_experiments,
)


def protocol():
    return ExperimentProtocol.model_validate(synthetic_tutorial()["protocol"])


def observations(hypothesis, spans=(60, 100, 160), forces=(5, 10)):
    p = protocol()
    inertia = p.width_mm * p.thickness_mm**3 / 12
    result = []
    for span in spans:
        for force in forces:
            if hypothesis == "material_stiffness":
                value = force * span**3 / (3 * 40000 * inertia)
            else:
                value = force * span**3 / (3 * p.reference_modulus_mpa * inertia)
                value += force * (
                    0.005 if hypothesis == "fixture_translation" else span**2 * 0.000001
                )
            result.append(
                Observation(
                    observation_id=f"row_{len(result)}",
                    span_mm=span,
                    force_n=force,
                    force_uncertainty_n=0.001,
                    displacement_mm=value,
                    displacement_uncertainty_mm=0.0001,
                )
            )
    return result


@pytest.mark.parametrize(
    "hypothesis,parameter",
    [("material_stiffness", 40000), ("fixture_translation", 0.005), ("fixture_rotation", 0.000001)],
)
def test_multispan_synthetic_controls_recover_and_discriminate(hypothesis, parameter):
    result = compare_hypotheses(protocol(), observations(hypothesis))
    selected = next(f for f in result["fits"] if f["hypothesis"] == hypothesis)
    assert selected["parameter"] == pytest.approx(parameter, rel=1e-9)
    assert result["selected_hypothesis"] == hypothesis
    assert result["status"] == "discriminated"
    assert result["physical_qualification"] == "unverified"


def test_single_span_force_sweep_cannot_identify_cause_even_with_exact_data():
    result = compare_hypotheses(
        protocol(), observations("fixture_translation", spans=(100,), forces=(2, 5, 10, 15))
    )
    assert result["status"] == "ambiguous"
    assert result["selected_hypothesis"] is None
    assert len(result["compatible_hypotheses"]) == 3
    assert result["suggested_experiments"]
    assert all(s["condition"]["span_mm"] != 100 for s in result["suggested_experiments"])
    assert (
        result["suggested_experiments"][0]["discrimination_score"]
        >= result["suggested_experiments"][-1]["discrimination_score"]
    )


def test_nonlinear_fixture_wrong_model_is_not_relabelled_as_valid_material():
    rows = observations("fixture_translation")
    for row in rows:
        row.displacement_mm += 0.003 * row.force_n**2
    result = compare_hypotheses(protocol(), rows)
    assert result["status"] == "no_model_fits"
    assert not result["compatible_hypotheses"]


def test_displacement_and_force_uncertainty_both_widen_prediction():
    rows = observations("material_stiffness")
    low = fit_hypothesis(protocol(), rows, "material_stiffness")
    for row in rows:
        row.displacement_uncertainty_mm *= 10
        row.force_uncertainty_n *= 10
    high = fit_hypothesis(protocol(), rows, "material_stiffness")
    condition = ExperimentCondition(span_mm=120, force_n=10, force_uncertainty_n=0.001)
    assert (
        predict(protocol(), high, condition)["standard_uncertainty_mm"]
        > predict(protocol(), low, condition)["standard_uncertainty_mm"]
    )
    larger = condition.model_copy(update={"force_uncertainty_n": 1})
    assert (
        predict(protocol(), low, larger)["standard_uncertainty_mm"]
        > predict(protocol(), low, condition)["standard_uncertainty_mm"]
    )


def test_csv_exact_units_roundtrip_and_no_extra_or_duplicate_columns():
    rows = observations("fixture_translation")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].model_dump()))
    writer.writeheader()
    writer.writerows(row.model_dump() for row in rows)
    assert parse_observations(output.getvalue()) == rows
    for bad in (
        output.getvalue().replace("force_n", "force_lbf"),
        output.getvalue().replace("force_n", "span_mm"),
        "",
        output.getvalue() + "extra,row\n",
    ):
        with pytest.raises(ValueError):
            parse_observations(bad)


@pytest.mark.parametrize(
    "key,value",
    [("force_uncertainty_n", 0), ("displacement_uncertainty_mm", -1), ("force_n", float("nan"))],
)
def test_missing_or_nonphysical_uncertainty_is_rejected(key, value):
    data = observations("material_stiffness")[0].model_dump()
    data[key] = value
    with pytest.raises(ValidationError):
        Observation.model_validate(data)


def test_model_scope_and_duplicate_observations_are_enforced():
    rows = observations("material_stiffness")
    with pytest.raises(ValueError, match="unique"):
        compare_hypotheses(protocol(), [*rows, rows[0]])
    rows[0].span_mm = 30
    with pytest.raises(ValueError, match="span"):
        compare_hypotheses(protocol(), rows)
    bad = protocol().model_dump()
    bad["min_span_mm"] = 20
    with pytest.raises(ValidationError, match="thicknesses"):
        ExperimentProtocol.model_validate(bad)


def test_tutorial_is_observably_synthetic_and_omits_future_outcome():
    data = synthetic_tutorial()
    assert data["provenance"] == "synthetic"
    assert "no instrument" in data["protocol"]["apparatus"]
    assert not any(row["span_mm"] == 160 for row in data["observations"])
    assert data["physical_qualification"] == "unverified"


@pytest.mark.parametrize(
    "hypothesis", ["material_stiffness", "fixture_translation", "fixture_rotation"]
)
def test_thickness_proposal_preserves_fixture_floor_and_meets_nominal_target(hypothesis):
    fit = fit_hypothesis(protocol(), observations(hypothesis), hypothesis)
    condition = ExperimentCondition(span_mm=120, force_n=10, force_uncertainty_n=0.001)
    original = protocol().model_dump()
    before = predict(protocol(), fit, condition)["displacement_mm"]
    target = before * 0.9
    result = recommend_thickness(protocol(), fit, condition, target, 4, 6)
    assert 4 < result["recommended_thickness_mm"] <= 6
    assert result["after"]["displacement_mm"] == pytest.approx(target)
    assert result["before"]["displacement_mm"] == before
    assert result["physical_qualification"] == "unverified"
    assert result["automatic_application"] is False
    assert protocol().model_dump() == original


def test_thickness_cannot_remove_fixture_compliance_or_bypass_slenderness():
    fit = fit_hypothesis(protocol(), observations("fixture_translation"), "fixture_translation")
    condition = ExperimentCondition(span_mm=120, force_n=10, force_uncertainty_n=0.001)
    with pytest.raises(ValueError, match="unattainable"):
        recommend_thickness(protocol(), fit, condition, 0.01, 4, 100)
    with pytest.raises(ValueError, match="No thickness"):
        recommend_thickness(protocol(), fit, condition, 0.051, 4, 100)
    with pytest.raises(ValueError, match="finite"):
        recommend_thickness(protocol(), fit, condition, float("nan"), 4, 6)


def test_experiment_ranking_changes_with_planned_measurement_uncertainty():
    rows = observations("fixture_translation", spans=(100,), forces=(2, 5, 10, 15))
    fits = [
        fit_hypothesis(protocol(), rows, h)
        for h in ("material_stiffness", "fixture_translation", "fixture_rotation")
    ]
    precise = [row.model_copy(update={"displacement_uncertainty_mm": 0.00001}) for row in rows]
    noisy = [row.model_copy(update={"displacement_uncertainty_mm": 0.001}) for row in rows]
    precise_rank = suggest_experiments(protocol(), precise, fits)
    noisy_rank = suggest_experiments(protocol(), noisy, fits)
    # Identical fitted models: only the assumed future instrument precision changes.
    assert precise_rank[0]["condition"]["span_mm"] == 60
    assert noisy_rank[0]["condition"]["span_mm"] == 180
    assert precise_rank[0]["discrimination_score"] > noisy_rank[0]["discrimination_score"]
    assert "not a formal information bound" in precise_rank[0]["scoring_method"]
    assert precise_rank[0]["planned_measurement_uncertainty_mm"] == 0.00001


def test_single_compatible_model_suggestions_are_challenges_not_rival_discrimination():
    result = compare_hypotheses(protocol(), observations("material_stiffness"))
    assert result["suggested_experiments"]
    for suggestion in result["suggested_experiments"]:
        assert suggestion["purpose"] == "selected_model_challenge"
        assert suggestion["considered_hypotheses"] == ["material_stiffness"]
        assert suggestion["predicted_separation_mm"] == 0
