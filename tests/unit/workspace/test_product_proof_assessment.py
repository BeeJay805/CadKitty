"""Controlled synthetic fixtures test accounting rules, never real productivity claims."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "product_proof_assessment.py"
SPEC = importlib.util.spec_from_file_location("product_proof_assessment", SCRIPT)
assert SPEC and SPEC.loader
assessment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assessment
SPEC.loader.exec_module(assessment)


def reference(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def trial(tmp_path):
    task = save(tmp_path / "task.json", {"task": "Controlled test fixture only"})
    acceptance = save(
        tmp_path / "acceptance.json",
        {"criteria": [{"id": "fit", "description": "Independent fit assessment"}]},
    )
    inputs = save(tmp_path / "inputs.json", {"initial_state": "controlled"})
    observed = save(
        tmp_path / "observation.json", {"note": "Synthetic test evidence, not a real trial"}
    )
    provenance = {
        "actor_id": "reviewer",
        "method": "observed",
        "evidence": ["observation"],
        "note": "Fixture asserts complete scoped observation",
    }
    coverage = {"status": "complete", "provenance": provenance, "gaps": [], "empty_confirmed": True}
    interval = {
        "actor_id": "operator",
        "activity": "task_work",
        "start_seconds": 0,
        "end_seconds": 120,
        "provenance": provenance,
    }
    value = {
        "assessment_id": "fixture-baseline",
        "attempt_id": "baseline",
        "data_origin": "observed",
        "method": "Controlled fixture",
        "tools_and_resources": ["test tool"],
        "evidence": {
            "task": reference(task),
            "acceptance": reference(acceptance),
            "inputs": reference(inputs),
            "observation": reference(observed),
        },
        "task_evidence": "task",
        "acceptance_evidence": "acceptance",
        "starting_inputs_evidence": "inputs",
        "actors": [
            {"id": "operator", "role": "task operator", "kind": "human"},
            {"id": "reviewer", "role": "observer", "kind": "human"},
        ],
        "criteria": [{"id": "fit", "outcome": "passed", "assessment": provenance}],
        "accounting": {
            "boundary": "Entire task including preparation, review and corrections",
            "included_activities": ["task_work"],
            "excluded_activities": ["unattended_compute"],
        },
        "intervals": [interval],
        "human_time_coverage": coverage,
        "interventions": [],
        "intervention_coverage": coverage,
        "machine_runtime_seconds": 10000,
    }
    return value


def pair(tmp_path, trial, candidate=None):
    candidate = copy.deepcopy(trial) if candidate is None else candidate
    candidate["attempt_id"] = "candidate"
    candidate["assessment_id"] = "fixture-candidate"
    before = save(tmp_path / "baseline.json", trial)
    after = save(tmp_path / "candidate.json", candidate)
    return save(
        tmp_path / "pair.json",
        {
            "baseline": reference(before),
            "candidate": reference(after),
            "comparable": True,
            "assessor": "Fixture reviewer",
            "method": "observed",
            "rationale": "Same task and full accounting boundary; resource and operator differences reviewed",
            "evidence": [trial["evidence"]["observation"]],
        },
    )


def test_absent_baseline_never_estimates_labor_from_runtime(tmp_path, trial):
    result = assessment.summarize(assessment.load_assessment(save(tmp_path / "trial.json", trial)))
    assert result["recorded_human_seconds"] == 120
    assert result["machine_runtime_seconds"] == 10000
    assert result["labor_savings"] is None
    assert result["labor_savings_reasons"] == ["No matched baseline comparison supplied"]


def test_overlap_merge_and_simultaneous_people_sum_labor(tmp_path, trial):
    trial["intervals"] = [
        {**trial["intervals"][0], "end_seconds": 100},
        {**trial["intervals"][0], "start_seconds": 50, "end_seconds": 150},
        {**trial["intervals"][0], "actor_id": "reviewer", "end_seconds": 100},
    ]
    result = assessment.summarize(assessment.load_assessment(save(tmp_path / "trial.json", trial)))
    assert result["recorded_human_seconds"] == 250
    assert result["human_seconds_by_actor"] == {"operator": 150, "reviewer": 100}
    assert result["overlaps_merged_for_actors"] == ["operator"]


@pytest.mark.parametrize("duration,savings", [(60, 60), (180, -60)])
def test_matched_comparison_preserves_positive_and_negative_savings(
    tmp_path, trial, duration, savings
):
    candidate = copy.deepcopy(trial)
    candidate["intervals"][0]["end_seconds"] = duration
    result = assessment.compare(pair(tmp_path, trial, candidate))
    assert result["labor_savings"] == {
        "human_seconds": savings,
        "fraction_of_baseline": savings / 120,
    }
    assert result["labor_savings_reasons"] == []


@pytest.mark.parametrize(
    "change",
    [
        "synthetic",
        "self_reported",
        "timing_self_report",
        "model_acceptance",
        "failed",
        "unassessed",
        "partial",
        "unknown",
        "unconfirmed_empty",
        "intervention_gap",
        "accounting",
        "frozen_task",
    ],
)
def test_incomparable_or_incomplete_records_withhold_savings(tmp_path, trial, change):
    candidate = copy.deepcopy(trial)
    if change in {"synthetic", "self_reported"}:
        candidate["data_origin"] = change
    elif change == "timing_self_report":
        candidate["intervals"][0]["provenance"]["method"] = "self_reported"
    elif change == "model_acceptance":
        candidate["criteria"][0]["assessment"]["method"] = "model_inference"
    elif change in {"failed", "unassessed"}:
        candidate["criteria"][0]["outcome"] = change
    elif change in {"partial", "unknown"}:
        candidate["human_time_coverage"]["status"] = change
    elif change == "unconfirmed_empty":
        candidate["intervals"] = []
        candidate["human_time_coverage"]["empty_confirmed"] = False
    elif change == "intervention_gap":
        candidate["intervention_coverage"]["status"] = "partial"
    elif change == "accounting":
        candidate["accounting"]["boundary"] = "Only final correction"
    else:
        candidate["evidence"]["task"] = reference(
            save(tmp_path / "changed-task.json", {"task": "different"})
        )
    result = assessment.compare(pair(tmp_path, trial, candidate))
    assert result["labor_savings"] is None
    assert result["labor_savings_reasons"]


@pytest.mark.parametrize(
    "start,end", [(-1, 1), (2, 1), (1, 1), (0, float("inf")), (float("nan"), 1), (False, 1)]
)
def test_invalid_times_are_rejected(tmp_path, trial, start, end):
    trial["intervals"][0].update(start_seconds=start, end_seconds=end)
    with pytest.raises(ValueError):
        assessment.load_assessment(save(tmp_path / "trial.json", trial))


def test_changed_evidence_and_assessment_hashes_fail_closed(tmp_path, trial):
    comparison = pair(tmp_path, trial)
    Path(trial["evidence"]["observation"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        assessment.compare(comparison)
    Path(tmp_path / "candidate.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        assessment.compare(comparison)


def test_missing_criteria_cannot_pass_partial_acceptance(tmp_path, trial):
    manifest_path = Path(trial["evidence"]["acceptance"]["path"])
    save(
        manifest_path,
        {
            "criteria": [
                {"id": "fit", "description": "Fit"},
                {"id": "strength", "description": "Strength"},
            ]
        },
    )
    trial["evidence"]["acceptance"] = reference(manifest_path)
    with pytest.raises(ValueError, match="entire frozen acceptance"):
        assessment.load_assessment(save(tmp_path / "trial.json", trial))


def test_absent_interventions_are_unknown_without_confirmed_observation(tmp_path, trial):
    trial["intervention_coverage"]["empty_confirmed"] = False
    result = assessment.summarize(assessment.load_assessment(save(tmp_path / "trial.json", trial)))
    assert result["recorded_interventions"] == 0
    assert result["intervention_count"] is None


@pytest.mark.parametrize("mismatch", ["empty", "actor", "activity"])
def test_intervention_without_corresponding_human_timing_blocks_savings(tmp_path, trial, mismatch):
    candidate = copy.deepcopy(trial)
    candidate["interventions"] = [
        {
            "actor_id": "operator",
            "activity": "task_work",
            "reason": "Manual correction",
            "provenance": candidate["human_time_coverage"]["provenance"],
        }
    ]
    if mismatch == "empty":
        candidate["intervals"] = []
    elif mismatch == "actor":
        candidate["intervals"][0]["actor_id"] = "reviewer"
    else:
        candidate["accounting"]["included_activities"].append("setup")
        trial["accounting"]["included_activities"].append("setup")
        candidate["intervals"][0]["activity"] = "setup"
    result = assessment.compare(pair(tmp_path, trial, candidate))
    assert result["labor_savings"] is None
    assert any(
        "intervention has no timed interval" in reason for reason in result["labor_savings_reasons"]
    )


def test_human_time_overflow_returns_controlled_error(tmp_path, trial, capsys):
    trial["intervals"][0]["end_seconds"] = 1e308
    trial["intervals"].append({**trial["intervals"][0], "actor_id": "reviewer"})
    assert assessment.main(["assess", str(save(tmp_path / "trial.json", trial))]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_readable_report_contains_resolvable_acceptance_and_measurement_evidence(tmp_path, trial):
    trial["evidence"]["observation"]["path"] = "observation.json"
    report = assessment.summarize(assessment.load_assessment(save(tmp_path / "trial.json", trial)))
    assert report["criteria"] == trial["criteria"]
    assert report["intervals"] == trial["intervals"]
    assert report["method"] == trial["method"]
    assert report["tools_and_resources"] == trial["tools_and_resources"]
    for reference in report["evidence"].values():
        assert Path(reference["path"]).is_absolute()
    assert report["frozen_evidence"]["acceptance_evidence"] == trial["evidence"]["acceptance"]


def test_comparability_is_explicit_and_pair_needs_distinct_attempts(tmp_path, trial):
    comparison_path = pair(tmp_path, trial)
    comparison = json.loads(comparison_path.read_text())
    comparison["comparable"] = False
    assert assessment.compare(save(comparison_path, comparison))["labor_savings"] is None
    comparison["comparable"] = True
    comparison["candidate"] = comparison["baseline"]
    assert (
        "Comparison requires distinct attempts"
        in assessment.compare(save(comparison_path, comparison))["labor_savings_reasons"]
    )


def test_zero_measured_baseline_withholds_fraction(tmp_path, trial):
    trial["intervals"] = []
    result = assessment.compare(pair(tmp_path, trial))
    assert result["labor_savings"] is None
    assert "Baseline human labor must be positive" in result["labor_savings_reasons"]


def test_schema_cli_and_exclusive_record_with_correction(tmp_path, trial, capsys):
    assert assessment.main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["additionalProperties"] is False
    source = save(tmp_path / "source.json", trial)
    output = tmp_path / "record.json"
    assert assessment.main(["record", str(source), "--output", str(output)]) == 0
    original = output.read_bytes()
    assert assessment.main(["record", str(source), "--output", str(output)]) == 1
    assert output.read_bytes() == original
    trial["supersedes"] = reference(output)
    trial["assessment_id"] = "corrected-fixture"
    assert (
        assessment.main(
            ["record", str(save(source, trial)), "--output", str(tmp_path / "correction.json")]
        )
        == 0
    )
    assert output.read_bytes() == original


def test_report_output_deterministic_and_no_overwrite(tmp_path, trial):
    path = pair(tmp_path, trial)
    first = assessment.compare(path)
    assert first == assessment.compare(path)
    output = tmp_path / "report.json"
    assert assessment.main(["compare", str(path), "--output", str(output)]) == 0
    original = output.read_bytes()
    assert assessment.main(["compare", str(path), "--output", str(output)]) == 1
    assert output.read_bytes() == original


def test_relocated_comparison_report_preserves_all_provenance_paths(tmp_path, trial):
    source = pair(tmp_path, trial)
    value = json.loads(source.read_text())
    for item in [value["baseline"], value["candidate"], *value["evidence"]]:
        item["path"] = Path(item["path"]).name
    save(source, value)
    destination = tmp_path / "reports"
    destination.mkdir()
    output = destination / "comparison.json"
    assert assessment.main(["compare", str(source), "--output", str(output)]) == 0
    reported = json.loads(output.read_text())["comparability"]
    for item in [reported["baseline"], reported["candidate"], *reported["evidence"]]:
        assert Path(item["path"]).is_absolute()
        assert assessment.verified_path(
            assessment.Evidence.model_validate(item), output.parent
        ).exists()


@pytest.mark.parametrize("entrypoint", ["assessment", "acceptance", "comparison"])
@pytest.mark.parametrize("ambiguity", ["duplicate", "NaN", "Infinity", "-Infinity", "1e309"])
def test_ambiguous_and_nonfinite_json_is_rejected_before_schema_validation(
    tmp_path, trial, entrypoint, ambiguity
):
    if entrypoint == "comparison":
        path = pair(tmp_path, trial)
        operation = assessment.compare
        key, first = "comparable", "false"
    elif entrypoint == "acceptance":
        path = Path(trial["evidence"]["acceptance"]["path"])
        key, first = "criteria", "[]"
    else:
        path = save(tmp_path / "assessment.json", trial)
        key, first = "data_origin", '"synthetic"'
    original = path.read_text(encoding="utf-8")
    prefix = f'"{key}":{first}' if ambiguity == "duplicate" else f'"nonfinite":{ambiguity}'
    path.write_text("{" + prefix + "," + original[1:], encoding="utf-8")
    if entrypoint == "acceptance":
        trial["evidence"]["acceptance"] = reference(path)
        path = save(tmp_path / "assessment.json", trial)
    if entrypoint != "comparison":
        operation = assessment.load_assessment
    expected = "Duplicate JSON key" if ambiguity == "duplicate" else "Non-finite JSON number"
    with pytest.raises(ValueError, match=expected):
        operation(path)


def test_nested_duplicate_outcome_cannot_replace_failed_acceptance(tmp_path, trial):
    path = save(tmp_path / "assessment.json", trial)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"outcome": "passed"', '"outcome": "failed", "outcome": "passed"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON key: outcome"):
        assessment.load_assessment(path)
