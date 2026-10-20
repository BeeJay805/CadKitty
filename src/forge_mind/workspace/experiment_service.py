"""Immutable experiment ancestry shared by HTTP and MCP; never invokes a model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.experiment_models import (
    CalibrationRequest,
    ExperimentProtocol,
    ExperimentRequest,
    Observation,
    OutcomeRequest,
    PredictionRequest,
)
from forge_mind.workspace.experiments import (
    LIMITATIONS,
    compare_hypotheses,
    csv_sha256,
    parse_observations,
    predict,
    validate_condition,
)
from forge_mind.workspace.lab_records import capture_binding, list_records, read_record, save_record
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def _current(
    service: WorkspaceService, sid: str, owner: str, identifier: str, kind: str
) -> dict[str, Any]:
    record = read_record(service, sid, owner, identifier, kind)
    if record["stale"]:
        raise WorkspaceError("Experiment evidence is stale; create a current investigation")
    return record


def create_experiment(
    service: WorkspaceService, sid: str, owner: str, request: ExperimentRequest
) -> dict[str, Any]:
    binding = capture_binding(
        service, sid, owner, request.revision_id, request.expected_input_version
    )
    rows = (
        parse_observations(request.csv_text)
        if request.csv_text is not None
        else request.observations
    )
    if len({row.observation_id for row in rows}) != len(rows):
        raise ValueError("Observation IDs must be unique")
    assessments = []
    for row in rows:
        deviations = []
        try:
            validate_condition(request.protocol, row)
        except ValueError as exc:
            deviations.append(str(exc))
        assessments.append(
            {
                "observation_id": row.observation_id,
                "status": "protocol_deviation" if deviations else "within_model_scope",
                "protocol_deviations": deviations,
            }
        )
    return save_record(
        service,
        sid,
        owner,
        "experiment",
        binding,
        {
            "title": request.title,
            "provenance": request.provenance,
            "protocol": request.protocol.model_dump(),
            "observations": [row.model_dump() for row in rows],
            "observation_assessments": assessments,
            "status": "recorded_with_protocol_deviations"
            if any(a["protocol_deviations"] for a in assessments)
            else "recorded",
            "csv_text": request.csv_text,
            "csv_sha256": csv_sha256(request.csv_text) if request.csv_text is not None else None,
            "physical_qualification": "unverified",
            "limitations": LIMITATIONS,
            "geometry_scope": "Owner-declared specimen dimensions and spans, not measurements derived from the bound CAD.",
        },
    )


def calibrate_experiment(
    service: WorkspaceService, sid: str, owner: str, request: CalibrationRequest
) -> dict[str, Any]:
    experiment = _current(service, sid, owner, request.experiment_id, "experiment")
    rows = {row["observation_id"]: row for row in experiment["observations"]}
    if not set(request.observation_ids) <= rows.keys():
        raise WorkspaceError("Calibration contains an unknown observation ID")
    selected = [Observation.model_validate(rows[key]) for key in request.observation_ids]
    comparison = compare_hypotheses(
        ExperimentProtocol.model_validate(experiment["protocol"]), selected
    )
    return save_record(
        service,
        sid,
        owner,
        "calibration",
        experiment["binding"],
        {
            "experiment_id": experiment["id"],
            "protocol": experiment["protocol"],
            "provenance": experiment["provenance"],
            "observation_ids": request.observation_ids,
            "training_observations": [row.model_dump() for row in selected],
            **comparison,
        },
        parents=(experiment["id"],),
    )


def _same_condition(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(
        math.isclose(a[key], b[key], rel_tol=1e-9, abs_tol=1e-9) for key in ("span_mm", "force_n")
    )


def _same_specimen(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(
        a.get(key) == b.get(key)
        for key in (
            "specimen_id",
            "material_id",
            "fixture_id",
            "width_mm",
            "thickness_mm",
        )
    )


def commit_prediction(
    service: WorkspaceService, sid: str, owner: str, request: PredictionRequest
) -> dict[str, Any]:
    calibration = _current(service, sid, owner, request.calibration_id, "calibration")
    protocol = ExperimentProtocol.model_validate(calibration["protocol"])
    fit = next(item for item in calibration["fits"] if item["hypothesis"] == request.hypothesis)
    if not fit["compatible"]:
        raise WorkspaceError(
            "Cannot preregister from a model incompatible with its calibration observations"
        )
    prediction = predict(protocol, fit, request.condition)
    history = list_records(service, sid, owner, limit=200)
    if len(history) >= 200:
        raise WorkspaceError(
            "Preregistration history exceeds the bounded 199-record audit; start a new project"
        )
    for record in history:
        if record["kind"] not in {"experiment", "outcome"} or not _same_specimen(
            record["protocol"], calibration["protocol"]
        ):
            continue
        rows = record.get("observations", [record.get("observation")])
        if any(
            row is not None and _same_condition(row, request.condition.model_dump()) for row in rows
        ):
            raise WorkspaceError(
                "This condition already has a recorded observation; it is not a held-out prediction"
            )
    spans = [row["span_mm"] for row in calibration["training_observations"]]
    new_span = not any(
        math.isclose(request.condition.span_mm, span, rel_tol=1e-9) for span in spans
    )
    scope = (
        "new_span_conditional_prediction" if new_span else "same_geometry_load_scaling_validation"
    )
    return save_record(
        service,
        sid,
        owner,
        "prediction",
        calibration["binding"],
        {
            "calibration_id": calibration["id"],
            "protocol": calibration["protocol"],
            "provenance": calibration["provenance"],
            "hypothesis": request.hypothesis,
            "condition": request.condition.model_dump(),
            "prediction": prediction,
            "comparison_tolerance_mm": request.comparison_tolerance_mm,
            "planned_displacement_uncertainty_mm": request.planned_displacement_uncertainty_mm,
            "comparison_rule": "Within tolerance only if absolute error plus 1.96 combined standard uncertainty is within the frozen tolerance; overlapping boundaries are inconclusive.",
            "rationale": request.rationale,
            "validation_scope": scope,
            "physical_qualification": "unverified",
            "limitations": [
                *LIMITATIONS,
                "Server order proves commitment before outcome submission, not that the operator has never seen an outcome.",
                "A new load at the same geometry tests load scaling and is not generalization.",
            ],
        },
        parents=(calibration["id"],),
        expected_sequence=max((row["sequence"] for row in history), default=0),
    )


def record_outcome(
    service: WorkspaceService, sid: str, owner: str, request: OutcomeRequest
) -> dict[str, Any]:
    commitment = _current(service, sid, owner, request.prediction_id, "prediction")
    if request.provenance != commitment["provenance"]:
        raise WorkspaceError(
            "Outcome provenance must match the committed experiment; synthetic evidence cannot become physical"
        )
    condition = commitment["condition"]
    actual = request.observation.model_dump()
    deviations = []
    if (
        not _same_condition(condition, actual)
        or condition["force_uncertainty_n"] != actual["force_uncertainty_n"]
    ):
        deviations.append("Observed span, force or force uncertainty differs from preregistration")
    if actual["displacement_uncertainty_mm"] != commitment["planned_displacement_uncertainty_mm"]:
        deviations.append(
            "Observed displacement uncertainty differs from the frozen measurement plan"
        )
    try:
        validate_condition(
            ExperimentProtocol.model_validate(commitment["protocol"]), request.observation
        )
    except ValueError as exc:
        deviations.append(str(exc))
    prediction = commitment["prediction"]
    error = actual["displacement_mm"] - prediction["displacement_mm"]
    sigma = math.hypot(prediction["standard_uncertainty_mm"], actual["displacement_uncertainty_mm"])
    tolerance = commitment["comparison_tolerance_mm"]
    if deviations:
        status = "protocol_deviation"
    elif abs(error) + 1.96 * sigma <= tolerance:
        status = "within_tolerance"
    elif abs(error) - 1.96 * sigma > tolerance:
        status = "outside_tolerance"
    else:
        status = "inconclusive"
    return save_record(
        service,
        sid,
        owner,
        "outcome",
        commitment["binding"],
        {
            "prediction_id": commitment["id"],
            "protocol": commitment["protocol"],
            "provenance": request.provenance,
            "observation": actual,
            "acquisition_note": request.acquisition_note,
            "status": status,
            "protocol_deviations": deviations,
            "comparison_valid": not deviations,
            "frozen_prediction": prediction,
            "planned_condition": condition,
            "planned_displacement_uncertainty_mm": commitment[
                "planned_displacement_uncertainty_mm"
            ],
            "error_mm": error,
            "combined_standard_uncertainty_mm": sigma,
            "comparison_tolerance_mm": tolerance,
            "validation_scope": commitment["validation_scope"],
            "physical_qualification": "unverified",
            "limitations": commitment["limitations"],
        },
        parents=(commitment["id"],),
        unique_key=commitment["id"],
    )


def synthetic_tutorial() -> dict[str, Any]:
    """Data only: callers must explicitly import, and cannot mistake this for a test run."""
    protocol = ExperimentProtocol(
        specimen_id="synthetic_beam",
        material_id="synthetic_aluminium",
        fixture_id="synthetic_fixture",
        apparatus="Synthetic tutorial; no instrument or physical specimen was used",
        reference_modulus_mpa=70000,
        width_mm=20,
        thickness_mm=4,
        min_span_mm=60,
        max_span_mm=180,
        max_force_n=20,
        max_elastic_stress_mpa=100,
        geometry_basis="Declared synthetic rectangular beam dimensions, not CAD-derived",
        assumptions="Small elastic deflection with an added translational fixture compliance",
        uncertainty_basis="Illustrative independent standard uncertainties, not measured instrument calibration",
    )
    rows = []
    for index, (span, force) in enumerate(((80, 5), (80, 10), (120, 5), (120, 10))):
        inertia = protocol.width_mm * protocol.thickness_mm**3 / 12
        value = force * (span**3 / (3 * protocol.reference_modulus_mpa * inertia) + 0.005)
        rows.append(
            Observation(
                observation_id=f"synthetic_{index + 1}",
                span_mm=span,
                force_n=force,
                force_uncertainty_n=0.01,
                displacement_mm=value,
                displacement_uncertainty_mm=0.001,
            ).model_dump()
        )
    return {
        "title": "Synthetic cantilever hypothesis tutorial",
        "provenance": "synthetic",
        "protocol": protocol.model_dump(),
        "observations": rows,
        "instructions": "Supply current revision_id and expected_input_version, then explicitly import. Reserve span 160 mm at 10 N for a prospective prediction; no outcome is supplied.",
        "physical_qualification": "unverified",
    }
