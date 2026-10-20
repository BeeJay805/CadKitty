"""Real SQLite record ancestry, ordering and ownership; no solver or model calls."""

import copy
import json

import pytest

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.experiment_models import (
    CalibrationRequest,
    ExperimentRequest,
    OutcomeRequest,
    PredictionRequest,
)
from forge_mind.workspace.experiment_service import (
    calibrate_experiment,
    commit_prediction,
    create_experiment,
    record_outcome,
    synthetic_tutorial,
)
from forge_mind.workspace.lab_records import read_record
from forge_mind.workspace.store import Store, WorkspaceError


@pytest.fixture
def lab(tmp_path, monkeypatch):
    from forge_mind.workspace import lab_records

    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    sid = service.store.create("owner", "Experiment", {"requirements": []})["id"]
    rid = "a" * 32
    service.store.add_revision(
        sid,
        {
            "id": rid,
            "document": {"variables": {"thickness": 4}},
            "contract": {"requirements": []},
            "measurement": {"geometry_valid": True, "artifact_hashes": {"part.step": "step"}},
        },
    )
    service.store.update(sid, proposal={"revision_id": rid, "based_on": None, "eligible": True})
    monkeypatch.setattr(service, "_load_revision", service.store.revision)
    monkeypatch.setattr(lab_records, "engine_fingerprint", lambda: "engine")
    tutorial = synthetic_tutorial()
    request = ExperimentRequest(
        revision_id=rid,
        expected_input_version=0,
        **{key: tutorial[key] for key in ("title", "protocol", "provenance", "observations")},
    )
    return service, sid, rid, request


def calibration(lab):
    service, sid, _, request = lab
    experiment = create_experiment(service, sid, "owner", request)
    return calibrate_experiment(
        service,
        sid,
        "owner",
        CalibrationRequest(
            experiment_id=experiment["id"],
            observation_ids=[row.observation_id for row in request.observations],
        ),
    )


def prediction_request(calibration, span=160, force=10):
    return PredictionRequest(
        calibration_id=calibration["id"],
        hypothesis="fixture_translation",
        condition={"span_mm": span, "force_n": force, "force_uncertainty_n": 0.01},
        comparison_tolerance_mm=0.01,
        planned_displacement_uncertainty_mm=0.001,
        rationale="Prospective span change",
    )


def outcome_request(prediction):
    return OutcomeRequest(
        prediction_id=prediction["id"],
        provenance="synthetic",
        observation={
            **prediction["condition"],
            "observation_id": "withheld",
            "displacement_mm": prediction["prediction"]["displacement_mm"],
            "displacement_uncertainty_mm": 0.001,
        },
        acquisition_note="Synthetic model control, no physical measurement",
    )


def test_commit_before_outcome_preserves_source_and_cannot_double_record(lab):
    service, sid, rid, _ = lab
    before = copy.deepcopy(service.store.session_record(sid, "owner"))
    fitted = calibration(lab)
    predicted = commit_prediction(service, sid, "owner", prediction_request(fitted))
    frozen = copy.deepcopy(predicted)
    outcome = record_outcome(service, sid, "owner", outcome_request(predicted))
    assert fitted["sequence"] < predicted["sequence"] < outcome["sequence"]
    assert outcome["status"] == "within_tolerance"
    assert outcome["physical_qualification"] == "unverified"
    assert outcome["validation_scope"] == "new_span_conditional_prediction"
    assert read_record(service, sid, "owner", predicted["id"]) == frozen
    assert before == service.store.session_record(sid, "owner")
    assert outcome["binding"]["revision_id"] == rid
    with pytest.raises(WorkspaceError, match="already"):
        record_outcome(service, sid, "owner", outcome_request(predicted))


def test_known_outcome_and_same_geometry_scaling_are_not_generalization(lab):
    service, sid, _, _ = lab
    fitted = calibration(lab)
    with pytest.raises(WorkspaceError, match="already has"):
        commit_prediction(service, sid, "owner", prediction_request(fitted, span=80))
    prediction = commit_prediction(
        service, sid, "owner", prediction_request(fitted, span=80, force=7)
    )
    assert prediction["validation_scope"] == "same_geometry_load_scaling_validation"


def test_existing_nontraining_observation_still_blocks_preregistration(lab):
    service, sid, _, request = lab
    extra = request.observations[-1].model_copy(update={"observation_id": "known", "span_mm": 160})
    request.observations.append(extra)
    experiment = create_experiment(service, sid, "owner", request)
    fitted = calibrate_experiment(
        service,
        sid,
        "owner",
        CalibrationRequest(
            experiment_id=experiment["id"],
            observation_ids=[row.observation_id for row in request.observations[:-1]],
        ),
    )
    with pytest.raises(WorkspaceError, match="already has"):
        commit_prediction(service, sid, "owner", prediction_request(fitted))


def test_frozen_tolerance_uncertainty_and_provenance_cannot_be_rewritten(lab):
    service, sid, _, _ = lab
    predicted = commit_prediction(service, sid, "owner", prediction_request(calibration(lab)))
    request = outcome_request(predicted)
    request.provenance = "physical_reported"
    with pytest.raises(WorkspaceError, match="provenance"):
        record_outcome(service, sid, "owner", request)
    request.provenance = "synthetic"
    request.observation.displacement_mm += 0.1
    assert record_outcome(service, sid, "owner", request)["status"] == "outside_tolerance"


def test_owner_stale_engine_and_tampered_parent_are_rejected(lab, monkeypatch):
    from forge_mind.workspace import lab_records

    service, sid, _, _ = lab
    fitted = calibration(lab)
    with pytest.raises(WorkspaceError):
        commit_prediction(service, sid, "different-owner", prediction_request(fitted))
    monkeypatch.setattr(lab_records, "engine_fingerprint", lambda: "changed")
    with pytest.raises(WorkspaceError, match="stale"):
        commit_prediction(service, sid, "owner", prediction_request(fitted))
    monkeypatch.setattr(lab_records, "engine_fingerprint", lambda: "engine")
    with service.store.connect() as con:
        row = con.execute(
            "SELECT document FROM lab_records WHERE id=?", (fitted["experiment_id"],)
        ).fetchone()
        value = json.loads(row[0])
        value["observations"][0]["displacement_mm"] += 1
        con.execute(
            "UPDATE lab_records SET document=? WHERE id=?",
            (json.dumps(value), fitted["experiment_id"]),
        )
    with pytest.raises(WorkspaceError, match="integrity"):
        commit_prediction(service, sid, "owner", prediction_request(fitted))


def test_concurrent_observation_append_invalidates_preregistration(lab, monkeypatch):
    from forge_mind.workspace import experiment_service

    service, sid, _, request = lab
    fitted = calibration(lab)
    original = experiment_service.save_record

    def raced(*args, **kwargs):
        if args[3] == "prediction":
            create_experiment(service, sid, "owner", request)
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment_service, "save_record", raced)
    with pytest.raises(WorkspaceError, match="changed while computing"):
        commit_prediction(service, sid, "owner", prediction_request(fitted))


def test_outcome_overlapping_frozen_tolerance_is_inconclusive(lab):
    service, sid, _, _ = lab
    predicted = commit_prediction(service, sid, "owner", prediction_request(calibration(lab)))
    request = outcome_request(predicted)
    request.observation.displacement_mm += predicted["comparison_tolerance_mm"]
    assert record_outcome(service, sid, "owner", request)["status"] == "inconclusive"


def test_csv_receipt_retains_original_bytes_and_reports_physical_as_claim(lab):
    import csv
    import hashlib
    import io

    service, sid, _, request = lab
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(request.observations[0].model_dump()))
    writer.writeheader()
    writer.writerows(row.model_dump() for row in request.observations)
    request.csv_text = output.getvalue()
    request.observations = []
    request.provenance = "physical_reported"
    receipt = create_experiment(service, sid, "owner", request)
    assert receipt["csv_text"] == output.getvalue()
    assert receipt["csv_sha256"] == hashlib.sha256(output.getvalue().encode()).hexdigest()
    assert receipt["provenance"] == "physical_reported"
    assert receipt["physical_qualification"] == "unverified"
    assert any("owner claims" in limitation for limitation in receipt["limitations"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("displacement_mm", 30),
        ("force_n", 25),
        ("force_uncertainty_n", 0.4),
        ("displacement_uncertainty_mm", 0.2),
    ],
)
def test_actual_protocol_deviations_are_retained_without_validation_credit(lab, key, value):
    service, sid, _, _ = lab
    predicted = commit_prediction(service, sid, "owner", prediction_request(calibration(lab)))
    frozen = copy.deepcopy(predicted)
    request = outcome_request(predicted)
    setattr(request.observation, key, value)
    receipt = record_outcome(service, sid, "owner", request)
    assert receipt["observation"][key] == value
    assert receipt["status"] == "protocol_deviation"
    assert receipt["comparison_valid"] is False
    assert receipt["protocol_deviations"]
    assert receipt["frozen_prediction"] == frozen["prediction"]
    assert receipt["planned_condition"] == frozen["condition"]
    assert receipt["comparison_tolerance_mm"] == frozen["comparison_tolerance_mm"]
    assert read_record(service, sid, "owner", predicted["id"]) == frozen


@pytest.mark.parametrize("bad", ["../outside", "bad-id", "a" * 31, "A" * 32])
def test_record_ids_are_lowercase_uuid_hex_only(lab, bad):
    from pydantic import ValidationError

    _, _, _, request = lab
    with pytest.raises(ValidationError):
        ExperimentRequest.model_validate({**request.model_dump(), "revision_id": bad})
    with pytest.raises(ValidationError):
        CalibrationRequest(experiment_id=bad, observation_ids=["one", "two", "three"])


@pytest.mark.parametrize("key,value", [("displacement_mm", 30), ("force_n", 25), ("span_mm", 30)])
def test_out_of_scope_import_is_retained_but_cannot_enter_calibration(lab, key, value):
    service, sid, _, request = lab
    setattr(request.observations[0], key, value)
    experiment = create_experiment(service, sid, "owner", request)
    assert experiment["observations"][0][key] == value
    assert experiment["status"] == "recorded_with_protocol_deviations"
    assessment = experiment["observation_assessments"][0]
    assert assessment["observation_id"] == request.observations[0].observation_id
    assert assessment["status"] == "protocol_deviation"
    assert assessment["protocol_deviations"]
    with pytest.raises(ValueError):
        calibrate_experiment(
            service,
            sid,
            "owner",
            CalibrationRequest(
                experiment_id=experiment["id"],
                observation_ids=[r.observation_id for r in request.observations],
            ),
        )
    fitted = calibrate_experiment(
        service,
        sid,
        "owner",
        CalibrationRequest(
            experiment_id=experiment["id"],
            observation_ids=[r.observation_id for r in request.observations[1:]],
        ),
    )
    assert len(fitted["observation_ids"]) == 3
    assert read_record(service, sid, "owner", experiment["id"])["observations"][0][key] == value
