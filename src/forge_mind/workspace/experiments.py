"""Small, inspectable hypothesis comparisons, conditional on declared beam assumptions."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from typing import Any

import numpy as np

from forge_mind.workspace.experiment_models import (
    ExperimentCondition,
    ExperimentProtocol,
    Hypothesis,
    Observation,
)

MODEL_VERSION = "cantilever-hypotheses-1"
HYPOTHESES: tuple[Hypothesis, ...] = (
    "material_stiffness",
    "fixture_translation",
    "fixture_rotation",
)
LIMITATIONS = [
    "Conditional Euler-Bernoulli rectangular slender-beam model with a transverse tip load.",
    "Geometry and reference modulus are treated as declared exact inputs; their uncertainty is not propagated.",
    "Independent Gaussian standard uncertainties and a linear elastic, small-displacement response are assumed.",
    "No contact, backlash, hysteresis, shear deformation, plasticity, fatigue or assembly qualification.",
    "Reported physical observations are owner claims, not authenticated instrument acquisition.",
]


def parse_observations(csv_text: str) -> list[Observation]:
    """Strict SI-derived column units; retain original text separately in the receipt."""
    if len(csv_text.encode("utf-8")) > 200000:
        raise ValueError("CSV exceeds 200000 bytes")
    reader = csv.DictReader(io.StringIO(csv_text))
    columns = set(Observation.model_fields)
    if (
        reader.fieldnames is None
        or set(reader.fieldnames) != columns
        or len(reader.fieldnames) != len(columns)
    ):
        raise ValueError("CSV columns must exactly match: " + ",".join(sorted(columns)))
    rows: list[Observation] = []
    for raw in reader:
        if len(rows) >= 200 or None in raw or any(value is None for value in raw.values()):
            raise ValueError("CSV contains too many rows or malformed columns")
        rows.append(Observation.model_validate(raw))
    if not rows:
        raise ValueError("CSV has no observations")
    return rows


def validate_condition(protocol: ExperimentProtocol, condition: ExperimentCondition) -> None:
    if not protocol.min_span_mm <= condition.span_mm <= protocol.max_span_mm:
        raise ValueError("Condition span is outside the declared protocol")
    if condition.force_n > protocol.max_force_n:
        raise ValueError("Condition force exceeds the protocol limit")
    stress = (
        6 * condition.force_n * condition.span_mm / (protocol.width_mm * protocol.thickness_mm**2)
    )
    if stress > protocol.max_elastic_stress_mpa:
        raise ValueError("Declared elastic stress applicability limit exceeded")
    if isinstance(condition, Observation) and condition.displacement_mm > 0.05 * condition.span_mm:
        raise ValueError("Observed displacement exceeds the small-displacement scope")


def validate_observations(protocol: ExperimentProtocol, observations: list[Observation]) -> None:
    ids = [row.observation_id for row in observations]
    if len(set(ids)) != len(ids):
        raise ValueError("Observation IDs must be unique")
    for row in observations:
        validate_condition(protocol, row)


def _terms(
    protocol: ExperimentProtocol, span: float, hypothesis: Hypothesis
) -> tuple[float, float]:
    inertia = protocol.width_mm * protocol.thickness_mm**3 / 12
    beam = span**3 / (3 * inertia)
    if hypothesis == "material_stiffness":
        return 0.0, beam
    return (
        beam / protocol.reference_modulus_mpa,
        1.0 if hypothesis == "fixture_translation" else span**2,
    )


def predict(
    protocol: ExperimentProtocol, fit: dict[str, Any], condition: ExperimentCondition
) -> dict[str, Any]:
    validate_condition(protocol, condition)
    base, coefficient = _terms(protocol, condition.span_mm, fit["hypothesis"])
    compliance = base + coefficient * fit["coefficient"]
    mean = condition.force_n * compliance
    if not math.isfinite(mean) or mean < 0 or mean > 0.05 * condition.span_mm:
        raise ValueError("Predicted displacement exceeds the model applicability scope")
    variance = (condition.force_n * coefficient) ** 2 * fit["coefficient_variance"] + (
        compliance * condition.force_uncertainty_n
    ) ** 2
    sigma = math.sqrt(variance)
    return {
        "displacement_mm": mean,
        "standard_uncertainty_mm": sigma,
        "interval_95_mm": [mean - 1.96 * sigma, mean + 1.96 * sigma],
        "interval_scope": "conditional approximate normal interval; excludes model and geometry uncertainty",
    }


def fit_hypothesis(
    protocol: ExperimentProtocol, rows: list[Observation], hypothesis: Hypothesis
) -> dict[str, Any]:
    if len(rows) < 3:
        raise ValueError("At least three observations are required")
    validate_observations(protocol, rows)
    terms = np.array([_terms(protocol, row.span_mm, hypothesis) for row in rows])
    forces = np.array([row.force_n for row in rows])
    force_sigma = np.array([row.force_uncertainty_n for row in rows])
    measured = np.array([row.displacement_mm for row in rows])
    measurement_sigma = np.array([row.displacement_uncertainty_mm for row in rows])
    offset, design = forces * terms[:, 0], forces * terms[:, 1]
    theta = float(np.dot(design, measured - offset) / np.dot(design, design))
    lower, upper = (1 / 2000000, 1000.0) if hypothesis == "material_stiffness" else (0.0, 1000.0)
    unconstrained = theta
    theta = min(upper, max(lower, theta))
    for _ in range(100):
        variance = measurement_sigma**2 + ((terms[:, 0] + terms[:, 1] * theta) * force_sigma) ** 2
        weights = 1 / variance
        candidate = float(
            np.sum(weights * design * (measured - offset)) / np.sum(weights * design**2)
        )
        updated = min(upper, max(lower, candidate))
        if abs(updated - theta) <= 1e-12 * max(abs(theta), 1e-12):
            theta = updated
            break
        theta = updated
    else:
        raise ValueError("Uncertainty-weighted fit did not converge")
    variance = measurement_sigma**2 + ((terms[:, 0] + terms[:, 1] * theta) * force_sigma) ** 2
    residual = measured - offset - design * theta
    z = residual / np.sqrt(variance)
    chi = float(np.dot(z, z))
    reduced = chi / (len(rows) - 1)
    at_boundary = theta in (lower, upper)
    compatible = not at_boundary and reduced <= 4 and float(np.max(np.abs(z))) <= 3
    coefficient_variance = float(max(1.0, reduced) / np.sum(design**2 / variance))
    parameter = 1 / theta if hypothesis == "material_stiffness" else theta
    return {
        "hypothesis": hypothesis,
        "model_version": MODEL_VERSION,
        "coefficient": theta,
        "coefficient_variance": coefficient_variance,
        "parameter": parameter,
        "parameter_name": {
            "material_stiffness": "elastic_modulus_mpa",
            "fixture_translation": "compliance_mm_per_n",
            "fixture_rotation": "rotational_compliance_per_n_mm",
        }[hypothesis],
        "unconstrained_initial_coefficient": unconstrained,
        "at_parameter_boundary": at_boundary,
        "compatible": compatible,
        "reduced_chi_squared": reduced,
        "max_standardized_residual": float(np.max(np.abs(z))),
        "residuals_mm": residual.tolist(),
        "training_prediction_mm": (offset + design * theta).tolist(),
        "coefficient_uncertainty_scope": "local linear approximation; not reliable at a parameter boundary",
    }


def suggest_experiments(
    protocol: ExperimentProtocol,
    rows: list[Observation],
    fits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bounded screening heuristic, not a Fisher-information or causal-discovery bound."""
    rivals = [fit for fit in fits if fit["compatible"]]
    if not rivals:
        return []
    spans = sorted({row.span_mm for row in rows})
    suggestions = []
    force = min(row.force_n for row in rows)
    measurement_sigma = max(row.displacement_uncertainty_mm for row in rows)
    for span in np.linspace(protocol.min_span_mm, protocol.max_span_mm, 9):
        if any(math.isclose(float(span), old, rel_tol=1e-6) for old in spans):
            continue
        condition = ExperimentCondition(
            span_mm=float(span),
            force_n=force,
            force_uncertainty_n=max(row.force_uncertainty_n for row in rows),
        )
        try:
            predictions = [predict(protocol, fit, condition) for fit in rivals]
            if len(rivals) == 1:
                nearest = min(spans, key=lambda old: abs(old - span))
                baseline = predict(
                    protocol, rivals[0], condition.model_copy(update={"span_mm": nearest})
                )
                pairs = [(predictions[0], baseline)]
            else:
                pairs = [(a, b) for i, a in enumerate(predictions) for b in predictions[i + 1 :]]
        except ValueError:
            continue
        means = [item["displacement_mm"] for item in predictions]
        scores = [
            abs(a["displacement_mm"] - b["displacement_mm"])
            / math.sqrt(
                a["standard_uncertainty_mm"] ** 2
                + b["standard_uncertainty_mm"] ** 2
                + 2 * measurement_sigma**2
            )
            for a, b in pairs
        ]
        challenge = len(rivals) == 1
        suggestions.append(
            {
                "condition": condition.model_dump(),
                "predicted_separation_mm": max(means) - min(means),
                "discrimination_score": min(scores),
                "planned_measurement_uncertainty_mm": measurement_sigma,
                "considered_hypotheses": [fit["hypothesis"] for fit in rivals],
                "purpose": "selected_model_challenge" if challenge else "rival_discrimination",
                "scoring_method": (
                    "Predicted response change from the nearest tested span, at the proposed force, divided by combined predicted and measurement standard uncertainty."
                    if challenge
                    else "Minimum pairwise separation of compatible hypotheses divided by sqrt(sigma_a^2 + sigma_b^2 + 2*measurement_sigma^2)."
                )
                + " Measurement sigma is the largest observed displacement standard uncertainty. Heuristic ranking ignores predictive covariance; not a formal information bound.",
                "reason": (
                    "Only one hypothesis is compatible. Challenge its conditional prediction at another span; rejected alternatives are not treated as equally plausible."
                    if challenge
                    else "Change span to distinguish compatible beam-stiffness and fixture-compliance explanations relative to their uncertainty. Measure fixture motion independently if ambiguity remains."
                ),
            }
        )
    suggestions.sort(key=lambda item: item["discrimination_score"], reverse=True)
    return suggestions[:3]


def compare_hypotheses(protocol: ExperimentProtocol, rows: list[Observation]) -> dict[str, Any]:
    fits = [fit_hypothesis(protocol, rows, name) for name in HYPOTHESES]
    compatible = [fit["hypothesis"] for fit in fits if fit["compatible"]]
    spans = sorted({row.span_mm for row in rows})
    identifiable = len(spans) > 1 and spans[-1] / spans[0] >= 1.2
    status = (
        "discriminated"
        if identifiable and len(compatible) == 1
        else "ambiguous"
        if compatible
        else "no_model_fits"
    )
    return {
        "model_version": MODEL_VERSION,
        "status": status,
        "fits": fits,
        "compatible_hypotheses": compatible,
        "selected_hypothesis": compatible[0] if status == "discriminated" else None,
        "span_diversity_sufficient": identifiable,
        "suggested_experiments": suggest_experiments(protocol, rows, fits),
        "selection_rule": "A unique compatible model plus at least 20 percent span diversity; residual compatibility is not proof of causal truth.",
        "limitations": LIMITATIONS,
        "physical_qualification": "unverified",
    }


def csv_sha256(csv_text: str) -> str:
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def recommend_thickness(
    protocol: ExperimentProtocol,
    fit: dict[str, Any],
    condition: ExperimentCondition,
    target_displacement_mm: float,
    min_thickness_mm: float,
    max_thickness_mm: float,
) -> dict[str, Any]:
    """Analytic monotone section proposal, conditional on unchanged fitted material/fixture."""
    values = (target_displacement_mm, min_thickness_mm, max_thickness_mm)
    if (
        not all(math.isfinite(value) and value > 0 for value in values)
        or min_thickness_mm > max_thickness_mm
    ):
        raise ValueError("Target and ordered thickness range must be finite and positive")
    if not fit.get("compatible"):
        raise ValueError("A design proposal requires a compatible calibrated model")
    validate_condition(protocol, condition)
    before = predict(protocol, fit, condition)
    hypothesis = fit["hypothesis"]
    modulus = (
        fit["parameter"] if hypothesis == "material_stiffness" else protocol.reference_modulus_mpa
    )
    fixture = (
        0.0
        if hypothesis == "material_stiffness"
        else fit["coefficient"]
        * (1.0 if hypothesis == "fixture_translation" else condition.span_mm**2)
    )
    floor = condition.force_n * fixture
    if target_displacement_mm <= floor:
        raise ValueError("Target is unattainable by thickness alone because of fixture compliance")
    # delta = 4 F L^3 / (E width thickness^3) + fixture contribution.
    needed = (
        4
        * condition.force_n
        * condition.span_mm**3
        / (modulus * protocol.width_mm * (target_displacement_mm - floor))
    ) ** (1 / 3)
    stress_minimum = math.sqrt(
        6
        * condition.force_n
        * condition.span_mm
        / (protocol.width_mm * protocol.max_elastic_stress_mpa)
    )
    recommended = max(min_thickness_mm, needed, stress_minimum)
    maximum = min(max_thickness_mm, protocol.min_span_mm / 10)
    if recommended > maximum * (1 + 1e-12):
        raise ValueError(
            "No thickness in the declared range satisfies deflection, elastic stress and slenderness limits"
        )
    candidate = ExperimentProtocol.model_validate(
        {**protocol.model_dump(), "thickness_mm": min(recommended, maximum)}
    )
    after = predict(candidate, fit, condition)
    return {
        "recommended_thickness_mm": candidate.thickness_mm,
        "before": before,
        "after": after,
        "fixture_floor_mm": floor,
        "target_displacement_mm": target_displacement_mm,
        "prediction_basis": "Minimum thickness satisfying nominal predicted response and declared stress/slenderness; uncertainty interval is shown separately and is not guaranteed below target.",
        "assumptions": "Fitted modulus or fixture compliance remains unchanged after the section change. This is a new design hypothesis requiring a new prediction and test.",
        "physical_qualification": "unverified",
        "automatic_application": False,
    }
