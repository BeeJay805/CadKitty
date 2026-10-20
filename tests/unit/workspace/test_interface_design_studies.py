"""Saved native CAD, real worker measurements and the existing study lifecycle."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.design_evaluators import evaluator_discovery, metric_catalog
from forge_mind.workspace.design_study_models import (
    AddStudyCandidate,
    CreateDesignStudy,
    DesignStudySpec,
    SearchDesignStudy,
)
from forge_mind.workspace.design_study_service import (
    add_candidate,
    create_study,
    enqueue_study,
    read_study,
    run_study_job,
)
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.document_service import build_document
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import (
    Contract,
    DesignPlan,
    OperationPlan,
    OperationStep,
    Requirement,
)
from forge_mind.workspace.store import Store


def clearance_document() -> DesignDocument:
    """Two independent interfaces require two changes before either design is feasible."""
    return DesignDocument.model_validate(
        {
            "name": "Two fixture interfaces",
            "variables": {"upper_x": 9.0, "lower_x": 9.0},
            "assembly": {
                "parts": [
                    {
                        "id": identifier,
                        "name": identifier,
                        "production": {"role": role},
                        "features": [
                            {
                                "id": "body",
                                "name": "Body",
                                "kind": "box",
                                "params": {"x": 10, "y": 10, "z": 10},
                            },
                        ],
                        "output": "body",
                    }
                    for identifier, role in (("workpiece", "reference"), ("jaw", "make"))
                ],
                "roots": [
                    {"id": "upper_reference", "definition": "workpiece", "position": [0, 30, 0]},
                    {"id": "lower_reference", "definition": "workpiece"},
                    {"id": "upper_jaw", "definition": "jaw", "position": ["upper_x", 30, 0]},
                    {"id": "lower_jaw", "definition": "jaw", "position": ["lower_x", 0, 0]},
                ],
            },
        }
    )


def clearance_spec() -> DesignStudySpec:
    return DesignStudySpec.model_validate(
        {
            "name": "Nominal interface clearance experiment",
            "objectives": [{"metric": "geometry.extent_x_mm", "direction": "minimize"}],
            "constraints": [
                constraint
                for identifier in ("upper", "lower")
                for constraint in (
                    {
                        "metric": f"interface_geometry.{identifier}_overlap_mm3",
                        "operator": "le",
                        "value": 1e-5,
                    },
                    {
                        "metric": f"interface_geometry.{identifier}_distance_mm",
                        "operator": "ge",
                        "value": 0.5,
                    },
                    {
                        "metric": f"interface_geometry.{identifier}_distance_mm",
                        "operator": "le",
                        "value": 2,
                    },
                )
            ],
            "variables": {"upper_x": [9, 15], "lower_x": [9, 15]},
            "evaluators": [
                {"kind": "geometry"},
                {
                    "kind": "interface_geometry",
                    "settings": {
                        "pairs": [
                            {
                                "id": name,
                                "a": name + "_reference",
                                "b": name + "_jaw",
                                "basis": "Explicit fixture experiment target: 0.5 to 2 mm nominal gap; not a manufacturing allowance",
                            }
                            for name in ("upper", "lower")
                        ]
                    },
                },
            ],
        }
    )


def native_fixture(root: Path) -> tuple[WorkspaceService, str, str, str]:
    service = WorkspaceService(
        Store(root),
        Executor("trusted_local"),
        client_factory=lambda **_: pytest.fail("Deterministic repair must not call a model"),
    )
    owner = "fixture-owner"
    contract = Contract(
        requirements=[
            Requirement(
                id="bodies",
                text="Four fixture bodies",
                source_quote="four fixture bodies",
                metric="solid_count",
                value=4,
            )
        ]
    )
    sid = service.store.create(
        owner,
        "Two independently controlled clearances with four fixture bodies",
        None,
        "native_operations",
    )["id"]
    document = clearance_document()
    plan = DesignPlan(
        objective="Two independent fixture interfaces",
        operation_plan=OperationPlan(
            part_ids=["workpiece", "jaw"],
            steps=[
                OperationStep(
                    id=name,
                    part_id=name,
                    feature_ids=["body"],
                    requirement_ids=["bodies"],
                    expected_effect="A fixture interface body",
                )
                for name in ("workpiece", "jaw")
            ],
        ),
    )
    service.store.set_design_plan(sid, owner, plan, 0, contract)
    revision = build_document(
        service,
        sid,
        owner,
        document,
        contract,
        None,
        "Seed an infeasible experiment",
        None,
        lambda: False,
    )
    assert revision["measurement"]["geometry_valid"], revision["measurement"]["errors"]
    service.store.update(sid, selected_revision=revision["id"])
    return service, owner, sid, revision["id"]


def test_interface_metric_registration_is_exact_and_discoverable():
    spec = clearance_spec()
    metrics = metric_catalog("interface_geometry", spec.evaluators[1].settings)
    assert {metric["key"] for metric in metrics} == {
        f"interface_geometry.{name}_{suffix}"
        for name in ("upper", "lower")
        for suffix in ("distance_mm", "overlap_mm3")
    }
    assert any(row["kind"] == "interface_geometry" for row in evaluator_discovery())
    bad = spec.model_dump()
    bad["constraints"][0]["metric"] = "interface_geometry.undeclared_overlap_mm3"
    with pytest.raises(ValueError, match="not measured"):
        DesignStudySpec.model_validate(bad)


def test_actual_native_search_repairs_two_interfaces_without_acceptance(tmp_path, monkeypatch):
    service, owner, sid, rid = native_fixture(tmp_path)
    monkeypatch.setattr(service, "start", lambda: None)
    before = service.store.get(sid, owner)
    study = create_study(
        service,
        sid,
        owner,
        CreateDesignStudy(revision_id=rid, input_version=0, spec=clearance_spec()),
    )
    queued = enqueue_study(service, sid, owner, study["id"], search=SearchDesignStudy(rounds=1))
    job = service.store.claim()
    assert job is not None and job["id"] == queued["job_id"]
    run_study_job(service, job, lambda: False)
    service.store.finish(job["id"], sid, "needs_review")
    result = read_study(service, sid, owner, study["id"])
    baseline = result["candidates"][0]
    assert baseline["eligible"] and baseline["feasible"] is False
    assert baseline["metrics"]["interface_geometry.upper_overlap_mm3"] == pytest.approx(100)
    best = next(
        c for c in result["candidates"] if c["id"] == result["comparison"]["best_candidate_id"]
    )
    assert best["feasible"] is True
    assert best["parameters"] == {"upper_x": 10.5, "lower_x": 10.5}
    assert any(
        c["parameters"] == {"upper_x": 10.5, "lower_x": 9} and c["feasible"] is False
        for c in result["candidates"]
    )
    # Independent analytic answers and fresh native Boolean readback, not the cached metric.
    import build123d as cad

    from forge_mind.workspace.assembly import named_solids, occurrence_label
    from forge_mind.workspace.geometry_results import boolean_volume

    shape = cad.import_step(service.store.directory(sid, best["revision_id"]) / "part.step")
    named = named_solids(shape)
    for name in ("upper", "lower"):
        reference = named[occurrence_label(name + "_reference")]
        jaw = named[occurrence_label(name + "_jaw")]
        assert boolean_volume(reference & jaw) == pytest.approx(0)
        assert pytest.approx(0.5) == jaw.bounding_box().min.X - reference.bounding_box().max.X
        assert reference.volume == pytest.approx(1000)
        assert best["metrics"][f"interface_geometry.{name}_distance_mm"] == pytest.approx(0.5)
    after = service.store.get(sid, owner)
    assert after.get("accepted_revision") == before.get("accepted_revision") is None
    assert after["selected_revision"] == rid


def test_original_reference_cannot_be_changed_by_a_candidate(tmp_path, monkeypatch):
    service, owner, sid, rid = native_fixture(tmp_path)
    monkeypatch.setattr(service, "start", lambda: None)
    study = create_study(
        service,
        sid,
        owner,
        CreateDesignStudy(revision_id=rid, input_version=0, spec=clearance_spec()),
    )
    raw = deepcopy(service._load_revision(sid, rid)["document"])
    raw["assembly"]["parts"][0]["features"][0]["params"]["x"] = 1
    changed = build_document(
        service,
        sid,
        owner,
        DesignDocument.model_validate(raw),
        Contract.model_validate(service._load_revision(sid, rid)["contract"]),
        rid,
        "Try to shrink the test object",
        None,
        lambda: False,
    )
    assert changed["measurement"]["geometry_valid"]
    candidate = add_candidate(
        service, sid, owner, study["id"], AddStudyCandidate(revision_id=changed["id"])
    )
    queued = enqueue_study(service, sid, owner, study["id"])
    job = service.store.claim()
    assert job is not None and job["id"] == queued["job_id"]
    run_study_job(service, job, lambda: False)
    service.store.finish(job["id"], sid, "needs_review")
    result = read_study(service, sid, owner, study["id"])
    rejected = next(c for c in result["candidates"] if c["id"] == candidate["id"])
    assert rejected["eligible"] is False
    assert rejected["feasible"] is None
    assert rejected["status"] == "failed"
    assert "fixed" in rejected["error"].lower() or "reference" in rejected["error"].lower()
    assert result["comparison"]["best_candidate_id"] is None
