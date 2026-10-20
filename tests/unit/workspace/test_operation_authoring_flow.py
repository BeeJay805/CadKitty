"""Exercise native operation batches through real saved CAD, not source-only mocks."""

from pathlib import Path

import build123d as b
import pytest

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.author_guidance import build_author_guidance
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.document_service import build_document
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.measure import verify_artifacts
from forge_mind.workspace.models import Action, Contract, DesignPlan, Measurement
from forge_mind.workspace.store import Store, WorkspaceError


def contract():
    return Contract.model_validate(
        {
            "requirements": [
                {
                    "id": "opening",
                    "text": "A 4 mm through mounting bore",
                    "source_quote": "4 mm bore",
                    "metric": "hole_diameter",
                    "value": 4,
                    "tolerance": 0.001,
                    "center": [0, 0],
                },
                {
                    "id": "thickness",
                    "text": "6 mm plate thickness",
                    "source_quote": "6 mm plate",
                    "metric": "extent_z",
                    "value": 6,
                    "tolerance": 0.001,
                },
            ]
        }
    )


def plan():
    return DesignPlan.model_validate(
        {
            "objective": "Mount with a clear through bore",
            "operation_plan": {
                "part_ids": ["mount"],
                "frames": [
                    {
                        "id": "local",
                        "part_id": "mount",
                        "description": "Centered plate; bore along Z",
                    }
                ],
                "steps": [
                    {
                        "id": "base",
                        "part_id": "mount",
                        "feature_ids": ["base"],
                        "requirement_ids": ["thickness"],
                        "expected_effect": "Plate stock",
                    },
                    {
                        "id": "bore",
                        "part_id": "mount",
                        "depends_on": ["base"],
                        "feature_ids": ["tool", "bored"],
                        "requirement_ids": ["opening"],
                        "expected_effect": "Clear mounting bore",
                    },
                ],
            },
        }
    )


def document(width=40, *, include_bore=True):
    features = [
        {
            "id": "base",
            "name": "Plate",
            "kind": "box",
            "params": {"x": 40, "y": 24, "z": 6},
            "bindings": {"x": "width"},
        }
    ]
    if include_bore:
        features += [
            {
                "id": "tool",
                "name": "Bore tool",
                "kind": "cylinder",
                "params": {"radius": 2, "height": 12},
            },
            {"id": "bored", "name": "Bored plate", "kind": "cut", "inputs": ["base", "tool"]},
        ]
    return DesignDocument.model_validate(
        {
            "variables": {"width": width},
            "features": features,
            "outputs": ["bored" if include_bore else "base"],
        }
    )


@pytest.mark.slow
def test_real_native_checkpoint_repair_dimension_edit_and_saved_geometry(tmp_path: Path):
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    sid = service.store.create(
        "owner", "4 mm bore in a 6 mm plate", None, authoring_mode="native_operations"
    )["id"]
    binding = service.store.set_design_plan(
        sid, "owner", plan(), expected_plan_version=0, contract=contract()
    )
    # The first useful checkpoint is valid stock, with the unmet bore retained.
    base = build_document(
        service,
        sid,
        "owner",
        document(include_bore=False),
        contract(),
        None,
        "Stock checkpoint",
        90,
        lambda: False,
    )
    assert base["measurement"]["geometry_valid"]
    assert not base["operation_record"]["graph_complete"]
    assert any(
        f["status"] != "pass"
        for f in base["measurement"]["findings"]
        if f["requirement_id"] == "opening"
    )
    made = build_document(
        service,
        sid,
        "owner",
        document(),
        contract(),
        base["id"],
        "Add through bore",
        90,
        lambda: False,
        plan_snapshot=binding,
    )
    assert made["operation_record"]["graph_complete"]
    assert made["operation_record"]["source"]["python_feature_count"] == 0
    assert all(f["status"] == "pass" for f in made["measurement"]["findings"])
    assert all(not step["effect_verified"] for step in made["operation_record"]["steps"])
    service.store.update(sid, accepted_revision=made["id"])
    edited = build_document(
        service,
        sid,
        "owner",
        document(44),
        contract(),
        made["id"],
        "Widen the mounting plate",
        90,
        lambda: False,
    )
    assert all(f["status"] == "pass" for f in edited["measurement"]["findings"])
    assert service.store.get(sid, "owner")["accepted_revision"] == made["id"]
    shape = b.import_step(service.store.directory(sid, edited["id"]) / "part.step")
    assert shape.is_valid and len(shape.solids()) == 1
    assert pytest.approx(44) == shape.bounding_box().size.X
    assert shape.volume == pytest.approx(44 * 24 * 6 - 3.141592653589793 * 2**2 * 6)
    assert verify_artifacts(
        service.store.directory(sid, edited["id"]),
        Measurement.model_validate(edited["measurement"]),
    )


def test_native_service_refuses_code_before_creating_artifacts(tmp_path):
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    sid = service.store.create("owner", "Mount", None, authoring_mode="native_operations")["id"]
    with pytest.raises(WorkspaceError):
        service.build(sid, "print('not CAD')", contract(), None, "Forbidden", 1, lambda: False)
    assert service.store.revisions(sid) == []
    assert not (tmp_path / "sessions").exists()


def test_guidance_uses_full_native_vocabulary_without_python_preference():
    text = build_author_guidance({"authoring_mode": "native_operations"}, "trusted_local")
    assert "Prefer concise Python" not in text
    assert "operation_plan" in text
    assert "no fixed feature cap" in text


@pytest.mark.slow
def test_internal_author_plan_dispatch_build_and_inspection(tmp_path):
    actions = iter(
        [
            Action(kind="plan", explanation="Plan the mounting part", plan=plan()),
            Action(
                kind="document",
                explanation="Construct native operations",
                document_json=document().model_dump_json(),
            ),
            Action(kind="inspect", explanation="Inspect saved geometry"),
            Action(
                kind="finish",
                explanation="Deliver the nominal checked part; physical fit remains unknown",
            ),
        ]
    )

    class ScriptedAuthor:
        model = "scripted-contract-check"

        def generate_structured(self, **kwargs):
            assert "Native operation mode" in kwargs["system"]
            return next(actions), LLMUsage(self.model, 0, 0)

        generate_structured_with_image = generate_structured

    service = WorkspaceService(
        Store(tmp_path), Executor("trusted_local"), lambda **kwargs: ScriptedAuthor()
    )
    sid = service.store.create(
        "owner",
        "4 mm bore in a 6 mm plate",
        contract().model_dump(),
        authoring_mode="native_operations",
    )["id"]
    service.store.enqueue(sid, "owner", {"max_actions": 4, "max_seconds": 90, "preview_only": True})
    job = service.store.claim()
    assert job
    service.run_job(job)
    result = service.store.get(sid, "owner")
    assert result["state"] == "contract_satisfied", result["events"]
    assert len(result["revisions"]) == 1
    revision = result["revisions"][0]
    assert revision["measurement"]["geometry_valid"]
    assert revision["operation_record"]["graph_complete"]
    assert revision["operation_record"]["plan_version"] == 1
    assert "operation-plan.json" in revision["measurement"]["artifact_hashes"]
    assert "operation-record.json" in revision["measurement"]["artifact_hashes"]
    assert result.get("accepted_revision") is None
