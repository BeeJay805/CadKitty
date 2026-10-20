"""Actual annular CAD study; nominal geometry only, without model calls."""

from math import pi

import pytest

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.design_study_models import (
    CreateDesignStudy,
    DesignStudySpec,
    SearchDesignStudy,
)
from forge_mind.workspace.design_study_service import (
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


def test_actual_annular_search_repairs_clearance_without_acceptance(tmp_path, monkeypatch):
    service = WorkspaceService(
        Store(tmp_path),
        Executor("trusted_local"),
        client_factory=lambda **_: pytest.fail("Annular study must not call a model"),
    )
    # Only background scheduling is suppressed; claim, worker, CAD and search are real.
    monkeypatch.setattr(service, "start", lambda: None)
    owner = "annular-owner"
    contract = Contract(
        requirements=[
            Requirement(
                id="bodies",
                text="Two annular fixture bodies",
                source_quote="two fixture bodies",
                metric="solid_count",
                value=2,
            )
        ]
    )
    sid = service.store.create(owner, "two fixture bodies", None, "native_operations")["id"]
    document = DesignDocument.model_validate(
        {
            "name": "Annular clearance fixture",
            "variables": {"bore_radius": 4.75},
            "assembly": {
                "parts": [
                    {
                        "id": "shaft",
                        "name": "Reference shaft",
                        "production": {"role": "reference"},
                        "features": [
                            {
                                "id": "body",
                                "name": "Shaft",
                                "kind": "cylinder",
                                "params": {"radius": 5, "height": 12},
                            }
                        ],
                        "output": "body",
                    },
                    {
                        "id": "sleeve",
                        "name": "Made sleeve",
                        "production": {"role": "make"},
                        "features": [
                            {
                                "id": "blank",
                                "name": "Blank",
                                "kind": "cylinder",
                                "params": {"radius": 8, "height": 12},
                            },
                            {
                                "id": "bore",
                                "name": "Through bore",
                                "kind": "cylinder",
                                "params": {"height": 14},
                                "bindings": {"radius": "bore_radius"},
                            },
                            {
                                "id": "body",
                                "name": "Sleeve",
                                "kind": "cut",
                                "inputs": ["blank", "bore"],
                            },
                        ],
                        "output": "body",
                    },
                ],
                "roots": [
                    {"id": name, "definition": name, "position": [13, -7, 3]}
                    for name in ("shaft", "sleeve")
                ],
            },
        }
    )
    plan = DesignPlan(
        objective="Nominal annular clearance",
        operation_plan=OperationPlan(
            part_ids=["shaft", "sleeve"],
            steps=[
                OperationStep(
                    id=name,
                    part_id=name,
                    feature_ids=features,
                    requirement_ids=["bodies"],
                    expected_effect="An annular fixture body",
                )
                for name, features in (("shaft", ["body"]), ("sleeve", ["blank", "bore", "body"]))
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
        "Seed annular interference",
        None,
        lambda: False,
    )
    assert revision["measurement"]["geometry_valid"], revision["measurement"]["errors"]
    rid = revision["id"]
    service.store.update(sid, selected_revision=rid)
    before = service.store.get(sid, owner)
    spec = DesignStudySpec.model_validate(
        {
            "name": "Annular nominal experiment",
            "objectives": [{"metric": "geometry.volume_mm3", "direction": "minimize"}],
            "constraints": [
                {
                    "metric": "interface_geometry.radial_overlap_mm3",
                    "operator": "le",
                    "value": 1e-5,
                },
                {
                    "metric": "interface_geometry.radial_distance_mm",
                    "operator": "ge",
                    "value": 0.25,
                },
                {
                    "metric": "interface_geometry.radial_distance_mm",
                    "operator": "le",
                    "value": 0.60,
                },
            ],
            "variables": {"bore_radius": [4.75, 7.75]},
            "evaluators": [
                {"kind": "geometry"},
                {
                    "kind": "interface_geometry",
                    "settings": {
                        "fixed_occurrence_paths": ["shaft"],
                        "pairs": [
                            {
                                "id": "radial",
                                "a": "shaft",
                                "b": "sleeve",
                                "basis": "Experimental 0.25 to 0.60 mm radial gap, not a print-fit allowance",
                            }
                        ],
                    },
                },
            ],
        }
    )
    study = create_study(
        service,
        sid,
        owner,
        CreateDesignStudy(
            revision_id=rid,
            input_version=0,
            spec=spec,
        ),
    )
    queued = enqueue_study(service, sid, owner, study["id"], search=SearchDesignStudy(rounds=1))
    job = service.store.claim()
    assert job is not None and job["id"] == queued["job_id"]
    run_study_job(service, job, lambda: False)
    service.store.finish(job["id"], sid, "needs_review")
    result = read_study(service, sid, owner, study["id"])
    baseline = result["candidates"][0]
    best = next(
        c for c in result["candidates"] if c["id"] == result["comparison"]["best_candidate_id"]
    )
    assert baseline["eligible"] is True and baseline["feasible"] is False
    assert best["eligible"] is True and best["feasible"] is True
    assert best["parameters"] == {"bore_radius": 5.5}
    assert len(result["candidates"]) == 2

    # Fresh STEP import, analytic cylindrical radii and a native Boolean are independent
    # of interface_geometry's distance/overlap result and cached study metrics.
    import build123d as cad
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.GeomAbs import GeomAbs_Cylinder

    from forge_mind.workspace.assembly import named_solids, occurrence_label
    from forge_mind.workspace.geometry_results import boolean_volume

    fixed_bounds = []
    for candidate, radius in ((baseline, 4.75), (best, 5.5)):
        shape = cad.import_step(
            service.store.directory(sid, candidate["revision_id"]) / "part.step"
        )
        solids = named_solids(shape)
        shaft, sleeve = (solids[occurrence_label(name)] for name in ("shaft", "sleeve"))
        shaft_box, sleeve_box = shaft.bounding_box(), sleeve.bounding_box()
        fixed_bounds.append([tuple(shaft_box.min), tuple(shaft_box.max)])
        assert pytest.approx(10) == shaft_box.max.X - shaft_box.min.X
        assert pytest.approx(12) == shaft_box.max.Z - shaft_box.min.Z
        assert shaft.volume == pytest.approx(pi * 5**2 * 12)
        assert sleeve.volume == pytest.approx(pi * (8**2 - radius**2) * 12)
        radii = sorted(
            {
                BRepAdaptor_Surface(face.wrapped).Cylinder().Radius()
                for face in sleeve.faces()
                if BRepAdaptor_Surface(face.wrapped).GetType() == GeomAbs_Cylinder
            }
        )
        assert radii == pytest.approx([radius, 8])
        assert pytest.approx(16) == sleeve_box.max.X - sleeve_box.min.X
        extrema = BRepExtrema_DistShapeShape()
        extrema.LoadS1(shaft.wrapped)
        extrema.LoadS2(sleeve.wrapped)
        extrema.Perform()
        assert extrema.IsDone() is True
        assert extrema.Value() == pytest.approx(max(radius - 5, 0), abs=1e-7)
        overlap = boolean_volume(shaft & sleeve)
        assert overlap == pytest.approx(pi * max(5**2 - radius**2, 0) * 12, abs=1e-7)
        assert candidate["metrics"]["interface_geometry.radial_overlap_mm3"] == pytest.approx(
            overlap, abs=1e-7
        )
        assert candidate["metrics"]["interface_geometry.radial_distance_mm"] == pytest.approx(
            max(radius - 5, 0), abs=1e-7
        )
    assert fixed_bounds[0] == fixed_bounds[1]
    after = service.store.get(sid, owner)
    assert after["selected_revision"] == before["selected_revision"] == rid
    assert after.get("accepted_revision") == before.get("accepted_revision") is None
    assert ("accepted_revision" in after) == ("accepted_revision" in before)
