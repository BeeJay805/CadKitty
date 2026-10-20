"""Portable evidence controls. Test fixture outcomes cannot inflate quality claims."""

from __future__ import annotations

import json

import pytest
from scripts import customer_outcome_bench as bench
from scripts.customer_outcome_bench import DEFAULT_CORPUS, FixtureAdapter, Journal, run_benchmark
from scripts.customer_outcome_report import build_report, export_report, html_report, public_result

from tests.unit.test_customer_outcome_bench import observation


@pytest.fixture(autouse=True)
def synthetic_git_identity(monkeypatch, tmp_path):
    """Report fixtures exercise retained data; they must not require a host Git install.

    Even adapters named customer_local below are inert protocol fixtures, not real
    customer executions. Keep their provenance visibly synthetic, never a made-up SHA.
    Production metadata still refuses broken repositories and missing Git tools.
    """
    original = bench.source_git_metadata
    monkeypatch.setenv("PATH", str(tmp_path / "no-external-commands"))

    def fixture_head(root):
        assert root == bench.ROOT
        return {
            "git_head": "synthetic-report-fixture-not-a-git-commit",
            "git_metadata": {"status": "synthetic_fixture", "worktree_cleanliness": "not_assessed"},
        }

    monkeypatch.setattr(bench, "source_git_metadata", fixture_head)
    return original


def test_actual_archive_refuses_empty_source_identity_before_adapter(
    tmp_path, monkeypatch, synthetic_git_identity
):
    # Restore real provenance checks in a true archive with no usable source hashes.
    monkeypatch.setattr(bench, "source_git_metadata", synthetic_git_identity)
    archive = tmp_path / "source-archive"
    archive.mkdir()
    monkeypatch.setattr(bench, "ROOT", archive)

    class NeverRun:
        name = "customer_local"

        def run(self, case, directory, emit):
            pytest.fail("Missing source provenance must refuse before actual adapter entry")

    run = tmp_path / "missing-git"
    with pytest.raises(ValueError, match="complete exact file-hash inventory"):
        run_benchmark(run, DEFAULT_CORPUS, ["block_stop"], NeverRun(), inventory=lambda: {})
    assert not (run / "events.jsonl").exists()
    assert not (run / ".runner.lock").exists()


def test_report_fixture_records_synthetic_source_identity_without_git(tmp_path):
    run = tmp_path / "run"
    run_benchmark(
        run,
        DEFAULT_CORPUS,
        ["block_stop"],
        FixtureAdapter({"block_stop": observation()}),
        inventory=lambda: {},
    )
    assert (
        Journal(run / "events.jsonl").rows[0]["git_head"]
        == "synthetic-report-fixture-not-a-git-commit"
    )


def test_synthetic_report_has_no_actual_pass_rate_and_no_private_text(tmp_path):
    private = "Bearer super-secret C:\\Users\\owner\\private /home/owner/private"
    adapter = FixtureAdapter(
        {"block_stop": observation(error=private, diagnostics=private, literal=private)}
    )
    run = tmp_path / "run"
    run_benchmark(
        run, DEFAULT_CORPUS, ["block_stop"], adapter, inventory=lambda: {"source": "fixed"}
    )
    output = tmp_path / "public"
    report = export_report(run, output)
    assert report["synthetic_attempts"] == 1
    assert report["groups"] == []
    assert not report["attempts"][0]["nominal_digital_pass"]
    assert {p.name for p in output.iterdir()} == {
        "evidence.json",
        "evidence.md",
        "index.html",
        "SHA256.json",
    }
    for path in output.iterdir():
        text = path.read_text()
        assert private not in text
        assert "super-secret" not in text
        assert "C:\\\\Users" not in text
    with pytest.raises(FileExistsError):
        export_report(run, output)


def test_result_tampering_is_refused(tmp_path):
    run = tmp_path / "run"
    rows = run_benchmark(
        run,
        DEFAULT_CORPUS,
        ["block_stop"],
        FixtureAdapter({"block_stop": observation()}),
        inventory=lambda: {},
    )
    (run / rows[0]["attempt_id"] / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        build_report(run)


def test_public_result_allows_only_known_model_and_measurements():
    row = {
        "attempt_id": "a" * 32,
        "condition": "cold",
        "observation": observation(
            actual_model="Bearer private-model", state="C:/private/path", total_seconds=-1
        ),
        "assessment": {},
        "result_sha256": "0" * 64,
    }
    public = public_result(row, {"id": "block_stop", "family": "block", "split": "development"})
    assert public["actual_model"] == "unverified"
    assert public["state"] == "other_or_missing"
    assert public["total_seconds"] is None
    assert "Bearer" not in json.dumps(public)
    assert public["full_task_success"] is None
    assert public["physical_readiness"] == "unverified"


def test_standalone_html_has_no_scripts_or_remote_dependencies(tmp_path):
    run = tmp_path / "run"
    run_benchmark(
        run,
        DEFAULT_CORPUS,
        ["block_stop"],
        FixtureAdapter({"block_stop": observation()}),
        inventory=lambda: {},
    )
    markup = html_report(build_report(run))
    assert "<script" not in markup
    assert "https://" not in markup
    assert "<table>" in markup
    assert "physical" in markup


def test_interrupted_and_abandoned_actual_attempts_stay_in_denominator(tmp_path):
    class Interrupted:
        name = "customer_local"

        def run(self, case, directory, emit):
            raise KeyboardInterrupt

    run = tmp_path / "run"
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            run_benchmark(run, DEFAULT_CORPUS, ["block_stop"], Interrupted(), inventory=lambda: {})
    report = build_report(run)
    assert report["completed_attempts"] == 0
    assert report["abandoned_attempts"] == 1
    assert report["unfinished_attempts"] == 1
    assert report["groups"][0]["actual_attempts"] == 2
    assert report["groups"][0]["first_attempt_denominator"] == 1
    assert report["groups"][0]["all_attempt_nominal_digital_passes"] == 0


def test_pre_generation_adapter_failure_is_an_actual_attempt(tmp_path):
    class FailedQualification:
        name = "customer_local"

        def run(self, case, directory, emit):
            raise ValueError("worker qualification failed before generation")

    run = tmp_path / "run"
    run_benchmark(run, DEFAULT_CORPUS, ["block_stop"], FailedQualification(), inventory=lambda: {})
    report = build_report(run)
    assert report["groups"][0]["actual_attempts"] == 1
    assert report["groups"][0]["all_attempt_nominal_digital_passes"] == 0


def test_export_refuses_rechained_journal_divergence_from_retained_result(tmp_path):
    run = tmp_path / "run"
    rows = run_benchmark(
        run,
        DEFAULT_CORPUS,
        ["block_stop"],
        FixtureAdapter({"block_stop": observation()}),
        inventory=lambda: {},
    )
    journal = Journal(run / "events.jsonl")
    original = list(journal.rows)
    (run / "events.jsonl").unlink()
    forged = Journal(run / "events.jsonl")
    for row in original:
        values = {
            k: v
            for k, v in row.items()
            if k not in {"kind", "sequence", "previous_hash", "event_hash", "at"}
        }
        if row["kind"] == "attempt_finished":
            values["assessment"] = {**values["assessment"], "nominal_digital_pass": True}
        forged.append(row["kind"], **values)
    assert len(rows) == 1
    with pytest.raises(ValueError, match="Journal result differs"):
        build_report(run)


@pytest.mark.parametrize("error,unchanged", [("failure", True), (None, False)])
def test_public_result_suppresses_pass_on_error_or_source_change(error, unchanged):
    row = {
        "attempt_id": "a" * 32,
        "condition": "cold",
        "observation": observation(),
        "assessment": {"nominal_digital_pass": True, "clarification_pass": True},
        "result_sha256": "0" * 64,
        "error": error,
        "source_unchanged": unchanged,
    }
    result = public_result(row, {"id": "block_stop", "family": "block", "split": "development"})
    assert not result["nominal_digital_pass"]
    assert not result["clarification_pass"]
