"""A retained diagnostic field must not promote an incomplete flow comparison."""

import json

import pytest

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.design_evaluators import flow_worker


def run_flow(
    tmp_path, monkeypatch, *, failed_stage=None, fields=("coarse", "fine", "domain"), corrupt=None
):
    source = tmp_path / "source.step"
    source.write_text("current candidate STEP")
    executable = tmp_path / "solver.exe"
    executable.write_text("fixed operator solver")
    attempt = tmp_path / "current-attempt"
    attempt.mkdir()
    previous = tmp_path / "previous-attempt" / "coarse"
    previous.mkdir(parents=True)
    (previous / "surface_field.json").write_text("previous candidate field")
    monkeypatch.setattr("forge_mind.workspace.flow_solver.solver_executable", lambda: executable)

    def evaluate(_source, directory, _settings, _cancelled):
        stage = directory.name
        directory.mkdir()
        result = {
            "status": "unverified" if stage == failed_stage else "complete",
            "eligible": stage != failed_stage,
            "metrics": {} if stage == failed_stage else {"drag_n": 10, "drag_coefficient": 1},
            "errors": ["explicit resource stop"] if stage == failed_stage else [],
            "provenance": {
                "solver_sha256": file_sha256(executable),
                "source_step_sha256": file_sha256(source),
                "mesh_sha256": stage,
            },
            "artifacts": {},
        }
        if stage in fields:
            field = directory / "surface_field.json"
            field.write_text(json.dumps({"stage": stage, "current_candidate": True}))
            result["artifacts"][field.name] = {"sha256": file_sha256(field)}
            if corrupt == "hash":
                field.write_text("field replaced after receipt")
            elif corrupt == "source":
                result["provenance"]["source_step_sha256"] = "another candidate"
            elif corrupt == "unreceipted":
                result["artifacts"] = {}
        return result

    monkeypatch.setattr("forge_mind.workspace.flow_solver.evaluate_flow", evaluate)
    payload = {
        "source_step": str(source),
        "settings": {
            "density_kg_m3": 1,
            "dynamic_viscosity_pa_s": 1,
            "velocity_m_s": [1, 0, 0],
            "reference_area_m2": 1,
            "reference_length_m": 1,
        },
        "sensitivity_tolerance": 0.02,
    }
    return flow_worker(attempt, payload), attempt


@pytest.mark.parametrize(
    "failed_stage,fields,expected_stage",
    [
        ("fine", ("coarse",), "coarse"),
        ("domain", ("coarse", "fine"), "fine"),
        ("fine", ("coarse", "fine"), "fine"),
    ],
)
def test_incomplete_refinement_retains_latest_receipted_field_without_score(
    tmp_path, monkeypatch, failed_stage, fields, expected_stage
):
    result, attempt = run_flow(tmp_path, monkeypatch, failed_stage=failed_stage, fields=fields)
    assert result["field_stage"] == expected_stage
    assert result["field_scope"] == f"diagnostic {expected_stage} field; refinement incomplete"
    assert json.loads((attempt / "surface_field.json").read_text())["stage"] == expected_stage
    assert result["metrics"] == {}
    assert result["status"] == "unverified"
    assert result["eligible"] is False
    assert result["sensitivity"]["eligible"] is False
    assert result["metric_uncertainty"] == {}
    assert result["errors"] == ["explicit resource stop"]


def test_completed_comparison_uses_domain_field_and_preserves_metrics(tmp_path, monkeypatch):
    result, attempt = run_flow(tmp_path, monkeypatch)
    assert result["field_stage"] == "domain"
    assert "physical validity unverified" in result["field_scope"]
    assert json.loads((attempt / "surface_field.json").read_text())["stage"] == "domain"
    assert result["status"] == "complete"
    assert result["eligible"] is True
    assert result["metrics"] == {"drag_n": 10, "drag_coefficient": 1}
    assert result["metric_uncertainty"] == {"drag_n": 0, "drag_coefficient": 0}


@pytest.mark.parametrize("corrupt", ["hash", "source", "unreceipted"])
def test_fallback_rejects_tampered_other_candidate_or_unreceipted_fields(
    tmp_path, monkeypatch, corrupt
):
    result, attempt = run_flow(
        tmp_path, monkeypatch, failed_stage="fine", fields=("coarse",), corrupt=corrupt
    )
    assert result["field_stage"] is None
    assert result["field_scope"] is None
    assert not (attempt / "surface_field.json").exists()
    assert result["eligible"] is False
    assert result["metrics"] == {}


def test_no_fallback_to_another_attempt_when_current_attempt_has_no_field(tmp_path, monkeypatch):
    result, attempt = run_flow(tmp_path, monkeypatch, failed_stage="coarse", fields=())
    assert result["field_stage"] is None
    assert not (attempt / "surface_field.json").exists()
