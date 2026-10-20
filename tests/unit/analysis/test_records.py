"""Unit tests for record/dataset construction + serialization (no LLM)."""

from __future__ import annotations

import json

from forge_mind.analysis.records import TrialRecord, aggregate, dataset_to_dict
from forge_mind.analysis.report import render_markdown, write_dataset


def _record(
    idx: int,
    *,
    passed: bool,
    label: str,
    stage: str | None,
    checks: list[str],
    gen_path: str = "direct-dsl",
) -> TrialRecord:
    return TrialRecord(
        trial_index=idx,
        passed=passed,
        first_failing_stage=stage,
        primary_label=label,
        primary_check_id=(checks[0] if checks else None),
        validator_message="msg" if checks else None,
        op_id=None,
        failed_check_ids=checks,
        cost_usd=0.02,
        duration_ms=12000.0,
        total_tokens=9000,
        artifacts={"dir": f"/runs/trial-{idx}", "stl": f"/runs/trial-{idx}/part.stl"},
        gen_path=gen_path,
    )


def test_trial_record_json_round_trip() -> None:
    rec = _record(
        0, passed=False, label="envelope_failed", stage="validate", checks=["mesh.bbox_within_max"]
    )
    from dataclasses import asdict

    reloaded = json.loads(json.dumps(asdict(rec)))
    assert reloaded["primary_label"] == "envelope_failed"
    assert reloaded["failed_check_ids"] == ["mesh.bbox_within_max"]
    assert reloaded["artifacts"]["stl"].endswith("part.stl")


def test_aggregate_counts() -> None:
    records = [
        _record(0, passed=True, label="passed", stage=None, checks=[]),
        _record(
            1,
            passed=False,
            label="envelope_failed",
            stage="validate",
            checks=["mesh.bbox_within_max"],
        ),
        _record(
            2,
            passed=False,
            label="envelope_failed",
            stage="validate",
            checks=["mesh.bbox_within_max", "printability.overhang_max_angle"],
        ),
        _record(
            3, passed=False, label="topology_wrong", stage="validate", checks=["mesh.single_body"]
        ),
    ]
    ds = aggregate("bench-001", "run1", records)
    assert ds.n_trials == 4
    assert ds.n_passed == 1
    assert ds.pass_rate == 0.25
    assert ds.label_counts["envelope_failed"] == 2
    assert ds.stage_counts["validate"] == 3  # only failed trials counted
    assert ds.gen_path_counts["direct-dsl"] == 4  # every trial's generation path counted
    # top_check_ids is sorted by count; mesh.bbox_within_max appears twice.
    top = dict(ds.top_check_ids)
    assert top["mesh.bbox_within_max"] == 2
    assert top["mesh.single_body"] == 1


def test_dataset_serialization_and_markdown(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [
        _record(0, passed=True, label="passed", stage=None, checks=[]),
        _record(
            1, passed=False, label="topology_wrong", stage="validate", checks=["mesh.single_body"]
        ),
    ]
    ds = aggregate("bench-001", "run1", records)

    payload = dataset_to_dict(ds)
    assert json.loads(json.dumps(payload))["bench_id"] == "bench-001"

    md = render_markdown(ds)
    assert "Failure analysis: bench-001" in md
    assert "topology_wrong" in md
    assert "mesh.single_body" in md

    paths = write_dataset(ds, tmp_path)
    assert paths["markdown"].exists()
    assert paths["json"].exists()
    # records.jsonl has one line per trial.
    lines = paths["jsonl"].read_text().strip().splitlines()
    assert len(lines) == 2
    # The compact json report omits the heavy per-record list.
    assert "records" not in json.loads(paths["json"].read_text())
