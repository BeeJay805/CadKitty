"""Adversarial evidence binding, recovery and comparison protocol regression tests."""

from __future__ import annotations

import copy
import sqlite3
from types import SimpleNamespace

import build123d as b
import pytest
from tests.unit.workspace.test_assembly_generation import simple
from tests.unit.workspace.test_design_studies import (
    claim_and_run,
    create,
    study_spec,
)
from tests.unit.workspace.test_design_studies import (
    studies as shared_studies,
)
from tests.unit.workspace.test_structural_studies import study as structural_settings

from forge_mind.workspace import design_evaluators as evaluators
from forge_mind.workspace import design_study_identity as identity
from forge_mind.workspace import design_study_service as studies_service
from forge_mind.workspace.design_study_models import AddStudyCandidate, StudyEvaluator
from forge_mind.workspace.document import DesignDocument, Occurrence
from forge_mind.workspace.store import WorkspaceError

studies = shared_studies


def structural_result():
    fine = {"max_displacement_mm": 0.05, "peak_von_mises_mpa": 50}
    coarse = {"max_displacement_mm": 0.052, "peak_von_mises_mpa": 52}
    return {
        "status": "computed",
        "summary": fine,
        "numerical_quality": {
            "status": "pass",
            "coarse": {"summary": coarse},
            "fine": {"summary": fine},
        },
        "comparisons": [
            {"id": "deflection", "status": "pass"},
            {"id": "yield_ratio", "status": "pass"},
        ],
    }


@pytest.mark.parametrize("failure", [None, "fail", "unverified", "missing", "no_coarse"])
def test_structural_declared_limits_and_mesh_noise_are_required(tmp_path, monkeypatch, failure):
    result = structural_result()
    if failure in {"fail", "unverified"}:
        result["comparisons"][0]["status"] = failure
    elif failure == "missing":
        result["comparisons"].clear()
    elif failure == "no_coarse":
        result["numerical_quality"].pop("coarse")
    monkeypatch.setattr(evaluators, "process_revision", lambda *a, **k: copy.deepcopy(result))
    service = SimpleNamespace(store=SimpleNamespace(directory=lambda *a: tmp_path))
    evaluated = evaluators.evaluate_design(
        service,
        "s",
        {"id": "r"},
        StudyEvaluator(kind="structure", settings=structural_settings()),
        tmp_path / "instrument",
        lambda: False,
    )
    assert evaluated["eligible"] is (failure is None)
    if failure is None:
        assert evaluated["metric_uncertainty"]["peak_von_mises_mpa"] == 2
        assert evaluated["metric_uncertainty"]["max_displacement_mm"] == pytest.approx(0.002)


def test_frozen_occurrence_survives_real_export_reordering(tmp_path):
    from forge_mind.workspace.kernel import compile_document

    document = simple()
    path = tmp_path / "before.step"
    b.export_step(compile_document(document, {})[0], path)
    solids, mapping = identity._components(document, path)
    before = mapping["right/b"][0]
    spec = study_spec()
    spec.evaluators.append(
        StudyEvaluator(
            kind="structure",
            settings={**structural_settings(), "target": {"body_index": before}},
        )
    )
    bindings = identity.bind_targets(document, path, spec)
    assert bindings["structure"]["path"] == "right/b"
    assert len(solids) == 4
    document.assembly.roots.insert(0, Occurrence(id="new", definition="block", position=(0, 60, 0)))
    changed_path = tmp_path / "after.step"
    b.export_step(compile_document(document, {})[0], changed_path)
    _, changed = identity._components(document, changed_path)
    after = changed["right/b"][0]
    assert after != before, "Fixture must actually reorder exported solid indices"
    effective = identity.remap_evaluators(document, changed_path, spec, bindings)[-1]
    assert effective.settings["target"]["body_index"] == after
    assert spec.evaluators[-1].settings["target"]["body_index"] == before
    bindings["structure"]["definition"] = "replacement"
    with pytest.raises(WorkspaceError, match="changed or removed"):
        identity.remap_evaluators(document, changed_path, spec, bindings)


def test_unnamed_multibody_target_cannot_be_frozen(tmp_path):
    from forge_mind.workspace.kernel import compile_document

    document = DesignDocument.model_validate(
        {
            "features": [
                {"id": "a", "name": "A", "kind": "box", "params": {"x": 3, "y": 4, "z": 5}},
                {"id": "b", "name": "B", "kind": "box", "params": {"x": 7, "y": 8, "z": 9}},
            ],
            "outputs": ["a", "b"],
        }
    )
    path = tmp_path / "ambiguous.step"
    b.export_step(compile_document(document, {})[0], path)
    spec = study_spec()
    spec.evaluators.append(StudyEvaluator(kind="structure", settings=structural_settings()))
    with pytest.raises(WorkspaceError, match="ambiguous"):
        identity.bind_targets(document, path, spec)


def test_flow_solver_file_replacement_changes_frozen_identity(tmp_path, monkeypatch):
    executable = tmp_path / "solver.exe"
    executable.write_bytes(b"solver version one")
    monkeypatch.setattr("forge_mind.workspace.flow_solver.solver_executable", lambda: executable)
    spec = SimpleNamespace(evaluators=[SimpleNamespace(kind="flow")])
    original = identity.instrument_identity(spec)
    executable.write_bytes(b"solver version two")
    assert identity.instrument_identity(spec) != original


def test_baseline_insertion_failure_rolls_back_study(studies):
    service = studies[0]
    studies_service._tables(service)
    with service.store.connect() as con:
        con.execute(
            "CREATE TRIGGER reject_baseline BEFORE INSERT ON design_candidates "
            "BEGIN SELECT RAISE(ABORT, 'forced baseline failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="forced baseline failure"):
        create(studies)
    with service.store.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM design_studies").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM design_candidates").fetchone()[0] == 0


def test_rejected_second_enqueue_preserves_live_run(studies):
    service, owner, sid, _, _ = studies
    study = create(studies)
    first = studies_service.enqueue_study(service, sid, owner, study["id"])
    job = service.store.claim()
    assert job["id"] == first["job_id"]
    with pytest.raises(WorkspaceError):
        studies_service.enqueue_study(service, sid, owner, study["id"])
    actual = studies_service.read_study(service, sid, owner, study["id"])
    assert actual["job_id"] == first["job_id"]
    assert actual["status"] in {"queued", "running"}
    with service.store.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM design_study_runs").fetchone()[0] == 1


@pytest.mark.parametrize("cancel_build", [False, True])
def test_failed_generated_build_can_retry_without_reusing_invalid_revision(
    studies, monkeypatch, cancel_build
):
    service, owner, sid, _, _ = studies
    study = claim_and_run(studies, create(studies))
    candidate = studies_service.add_candidate(
        service,
        sid,
        owner,
        study["id"],
        AddStudyCandidate(parameters={"width": 6}),
    )
    from forge_mind.workspace import document_service

    real_build = document_service.build_document
    failed_id = "f" * 32
    cancellation = [False]

    def interrupted_build(*args, **kwargs):
        cancellation[0] = cancel_build
        return {"id": failed_id, "measurement": {"geometry_valid": False}}

    monkeypatch.setattr(document_service, "build_document", interrupted_build)
    studies_service._evaluate_candidate(
        service, sid, owner, study, candidate, lambda: cancellation[0]
    )
    assert candidate["status"] == ("cancelled" if cancel_build else "failed")
    assert candidate["revision_id"] is None
    assert candidate["attempts"][0]["built_revision_id"] == failed_id
    monkeypatch.setattr(document_service, "build_document", real_build)
    studies_service._evaluate_candidate(service, sid, owner, study, candidate, lambda: False)
    assert candidate["status"] == "complete", candidate.get("error")
    assert candidate["revision_id"] != failed_id
    assert len(candidate["attempts"]) == 2


def test_previous_attempt_artifact_tampering_blocks_recovered_study(studies):
    service, owner, sid, _, _ = studies
    study = claim_and_run(studies, create(studies))
    candidate = study["candidates"][0]
    old_receipt = candidate["evaluations"][0]
    studies_service._evaluate_candidate(service, sid, owner, study, candidate, lambda: False)
    assert len(candidate["attempts"]) == 2
    artifact = (
        studies_service.candidate_directory(service, sid, study["id"], candidate["id"])
        / old_receipt["attempt"]
        / "geometry"
        / "measurement.json"
    )
    artifact.write_text("tampered history")
    with pytest.raises(WorkspaceError, match="artifact changed"):
        studies_service.read_study(service, sid, owner, study["id"])


def test_multiobjective_tradeoff_does_not_hide_dominating_improvement():
    spec = study_spec()
    spec.objectives.append(
        type(spec.objectives[0])(
            metric="geometry.surface_area_mm2",
            direction="minimize",
        )
    )

    def candidate(name, volume, area, noise=0):
        return {
            "id": name,
            "status": "complete",
            "eligible": True,
            "feasible": True,
            "metrics": {"geometry.volume_mm3": volume, "geometry.surface_area_mm2": area},
            "metric_uncertainty": {"geometry.volume_mm3": noise},
        }

    baseline = candidate("baseline", 100, 100)
    tradeoff = candidate("tradeoff", 1, 1000)
    improvement = candidate("improvement", 90, 90)
    result = studies_service.compare_candidates(spec, [baseline, tradeoff, improvement], "baseline")
    assert result["conclusion"] == "improved"
    assert result["best_candidate_id"] == "improvement"
    assert set(result["pareto_candidate_ids"]) == {"tradeoff", "improvement"}
    noisy = candidate("noisy", 90, 100, noise=20)
    result = studies_service.compare_candidates(spec, [baseline, noisy], "baseline")
    assert result["conclusion"] == "inconclusive"


def test_hosted_typed_build_rejects_python_before_execution(studies, monkeypatch):
    from forge_mind.workspace.document_service import build_document
    from forge_mind.workspace.models import Contract

    service, owner, sid, _, _ = studies
    service.executor.mode = "docker"
    monkeypatch.setattr(
        "forge_mind.workspace.program_feature.Executor.build",
        lambda *a, **k: pytest.fail("Public typed feature reached local execution"),
    )
    document = DesignDocument.model_validate(
        {
            "features": [
                {"id": "code", "name": "Program", "kind": "python", "code": "result=b.Box(1,2,3)"}
            ],
            "outputs": ["code"],
        }
    )
    with pytest.raises(WorkspaceError, match="private-local"):
        build_document(service, sid, owner, document, Contract(), None, "test", None, lambda: False)


def test_large_search_stops_at_float_resolution_and_checkpoints(studies, monkeypatch):
    from forge_mind.workspace.design_study_models import SearchDesignStudy

    service, owner, sid, _, _ = studies
    study = create(studies)
    study["spec"]["variables"] = {"width": [-1e308, 1e308]}
    baseline = study["candidates"][0]
    baseline["parameters"]["width"] = 0.0
    baseline["status"] = "complete"
    baseline.update(
        eligible=True,
        feasible=True,
        original_requirement_failures=[],
        metrics={"geometry.volume_mm3": 6400, "geometry.extent_x_mm": 100},
    )
    study["comparison"]["best_candidate_id"] = baseline["id"]
    monkeypatch.setattr(studies_service, "_study", lambda *a: study)
    monkeypatch.setattr(studies_service, "_fresh", lambda *a: None)
    monkeypatch.setattr(studies_service, "_verify_evaluations", lambda *a: None)
    monkeypatch.setattr(studies_service, "read_study", lambda *a: study)
    monkeypatch.setattr(studies_service, "_candidates", lambda *a: [baseline])
    sampled = []

    def add(*args):
        value = args[-1].parameters["width"]
        assert value != 0 and abs(value) <= 1e308
        sampled.append(value)
        return {"status": "complete"}

    monkeypatch.setattr(studies_service, "add_candidate", add)
    queued = studies_service.enqueue_study(
        service,
        sid,
        owner,
        study["id"],
        search=SearchDesignStudy(rounds=10000),
    )
    job = service.store.claim()
    assert job["id"] == queued["job_id"]
    studies_service.run_study_job(service, job, lambda: False)
    with service.store.connect() as con:
        run = studies_service._decode(
            con.execute(
                "SELECT document,sha256 FROM design_study_runs WHERE id=?",
                (job["run_id"],),
            ).fetchone()
        )
    assert run["state"] == "completed"
    assert 1024 < run["rounds_completed"] < 10000
    assert "representable" in run["stop_reason"]
    assert len(sampled) > 2048


def test_cancellation_at_last_evaluator_return_cannot_publish_feasibility(studies, monkeypatch):
    study = create(studies)
    cancelled = False
    original = studies_service.evaluate_design

    def evaluate(*args, **kwargs):
        nonlocal cancelled
        result = original(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(studies_service, "evaluate_design", evaluate)
    result = claim_and_run(studies, study, cancelled=lambda: cancelled)
    candidate = result["candidates"][0]
    assert candidate["status"] == "cancelled"
    assert candidate["eligible"] is False
    assert candidate["feasible"] is None
    assert result["comparison"]["best_candidate_id"] is None


def test_replaced_flow_binary_makes_study_stale_and_blocks_execution(
    studies, tmp_path, monkeypatch
):
    from forge_mind.workspace.design_study_models import CreateDesignStudy

    service, owner, sid, rid, _ = studies
    executable = tmp_path / "owned-solver.exe"
    executable.write_bytes(b"first instrument")
    monkeypatch.setattr("forge_mind.workspace.flow_solver.solver_executable", lambda: executable)
    spec = study_spec()
    spec.evaluators.append(
        StudyEvaluator(
            kind="flow",
            settings={
                "density_kg_m3": 1.2,
                "dynamic_viscosity_pa_s": 1.8e-5,
                "velocity_m_s": [0.01, 0, 0],
                "reference_area_m2": 0.001,
                "reference_length_m": 0.1,
            },
        )
    )
    study = studies_service.create_study(
        service,
        sid,
        owner,
        CreateDesignStudy(
            revision_id=rid,
            input_version=0,
            spec=spec,
        ),
    )
    assert study["current"] is True
    executable.write_bytes(b"replacement instrument")
    actual = studies_service.read_study(service, sid, owner, study["id"])
    assert actual["current"] is False
    assert actual["comparison"]["conclusion"] == "inconclusive"
    with pytest.raises(WorkspaceError, match="external instruments changed"):
        studies_service.enqueue_study(service, sid, owner, study["id"])


@pytest.mark.parametrize("fingerprint", ["historical-engine", None])
def test_historical_candidate_cannot_enter_current_study(studies, fingerprint):
    service, owner, sid, rid, _ = studies
    study = create(studies)
    historical = copy.deepcopy(service.store.revision(sid, rid))
    historical["id"] = "d" * 32
    historical["document"]["variables"]["width"] = 6
    historical["measurement"]["engine_fingerprint"] = fingerprint
    service.store.add_revision(sid, historical)
    with pytest.raises(WorkspaceError, match="different instruments"):
        studies_service.add_candidate(
            service,
            sid,
            owner,
            study["id"],
            AddStudyCandidate(revision_id=historical["id"]),
        )
    assert len(studies_service.read_study(service, sid, owner, study["id"])["candidates"]) == 1


def test_previously_admitted_historical_candidate_is_rejected_on_resume_and_read(
    studies, monkeypatch
):
    service, owner, sid, rid, _ = studies
    study = create(studies)
    historical = copy.deepcopy(service.store.revision(sid, rid))
    historical["id"] = "d" * 32
    historical["document"]["variables"]["width"] = 6
    historical["measurement"]["engine_fingerprint"] = "historical-engine"
    service.store.add_revision(sid, historical)
    candidate = studies_service.add_candidate(
        service,
        sid,
        owner,
        study["id"],
        AddStudyCandidate(parameters={"width": 6}),
    )
    # Simulate an existing persisted record admitted before the freshness guard.
    candidate.update(
        revision_id=historical["id"],
        status="complete",
        eligible=True,
        feasible=True,
        metrics={"geometry.volume_mm3": 4000},
    )
    studies_service._save_candidate(service, candidate)
    with pytest.raises(WorkspaceError, match="different instruments"):
        studies_service.read_study(service, sid, owner, study["id"])
    monkeypatch.setattr(
        studies_service,
        "evaluate_design",
        lambda *a, **k: pytest.fail("Stale contract evidence reached evaluator"),
    )
    studies_service._evaluate_candidate(service, sid, owner, study, candidate, lambda: False)
    assert candidate["status"] == "failed"
    assert candidate["eligible"] is False
    assert "different instruments" in candidate["error"]
    assert candidate["evaluations"] == []
