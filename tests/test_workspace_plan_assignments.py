"""Explicit ownership additions retain duties and use the existing atomic publication."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from forge_mind.workspace.construction_transaction import prepare_construction
from forge_mind.workspace.document import Feature
from forge_mind.workspace.models import DesignPlan, OperationStep
from forge_mind.workspace.operation_plan import coverage, snapshot
from forge_mind.workspace.plan_patch import PlanPatch, apply_plan_patch
from forge_mind.workspace.store import WorkspaceError
from tests.unit.workspace.test_construction_transaction import (
    BASE,
    RESULT,
    candidate,
    canonical,
)
from tests.unit.workspace.test_construction_transaction import construction as construction
from tests.unit.workspace.test_plan_patch import basis, contract, document, plan


def test_assignment_preserves_every_other_field_and_does_not_mutate_inputs():
    original = plan()
    before = original.model_dump(mode="json")
    patch = PlanPatch.model_validate(
        {
            "basis": basis(original),
            "construction_assignments": [
                {"step_id": "blank", "feature_ids": ["body", "bore", "finished"]}
            ],
        }
    )
    patch_before = patch.model_dump(mode="json")
    result = apply_plan_patch(original, patch, contract())
    expected = deepcopy(before)
    expected["operation_plan"]["steps"][0]["feature_ids"] = ["body", "bore", "finished"]
    assert result.model_dump(mode="json") == expected
    assert original.model_dump(mode="json") == before
    assert patch.model_dump(mode="json") == patch_before
    # Repeating an already-owned ID is an idempotent union, not duplicated ownership.
    result.operation_plan.steps[0].feature_ids.append("independent")
    assert original.operation_plan.steps[0].feature_ids == ["body"]


@pytest.mark.parametrize(
    "assignments",
    [
        [{"step_id": "blank", "feature_ids": []}],
        [{"step_id": "blank", "feature_ids": ["bore", "bore"]}],
        [
            {"step_id": "blank", "feature_ids": ["bore"]},
            {"step_id": "blank", "feature_ids": ["finished"]},
        ],
        [{"step_id": "blank", "feature_ids": ["bore"], "requirement_ids": []}],
        [{"step_id": "blank", "feature_ids": ["bore"], "status": "complete"}],
        [{"step_id": "blank", "feature_ids": ["bore"], "part_id": "other"}],
        [{"step_id": "blank", "feature_ids": ["bore"], "target": 20}],
    ],
)
def test_closed_assignment_schema_rejects_ambiguous_or_authority_fields(assignments):
    with pytest.raises(ValidationError):
        PlanPatch.model_validate({"basis": basis(plan()), "construction_assignments": assignments})


def test_full_step_update_and_assignment_cannot_overlap():
    original = plan()
    with pytest.raises(ValidationError, match="cannot overlap"):
        PlanPatch.model_validate(
            {
                "basis": basis(original),
                "operation_steps": [original.operation_plan.steps[0]],
                "construction_assignments": [{"step_id": "blank", "feature_ids": ["bore"]}],
            }
        )


@pytest.mark.parametrize("step_id", ["unknown", "inspect"])
def test_assignment_requires_an_existing_construction_step(step_id):
    original = plan()
    before = original.model_dump_json()
    patch = PlanPatch.model_validate(
        {
            "basis": basis(original),
            "construction_assignments": [{"step_id": step_id, "feature_ids": ["bore"]}],
        }
    )
    with pytest.raises(ValueError, match="existing construction step"):
        apply_plan_patch(original, patch, contract())
    assert original.model_dump_json() == before


def test_assignment_cannot_create_a_step_or_an_operation_plan():
    original = DesignPlan(objective="Future work")
    patch = PlanPatch.model_validate(
        {
            "basis": basis(original),
            "construction_assignments": [{"step_id": "build", "feature_ids": ["body"]}],
        }
    )
    with pytest.raises(ValueError, match="existing operation plan"):
        apply_plan_patch(original, patch, contract())


@pytest.mark.parametrize("part_id", ["main", "other"])
def test_existing_validation_keeps_feature_ownership_scoped_to_its_part(part_id):
    original = plan()
    if part_id == "other":
        original.operation_plan.part_ids.append("other")
    original.operation_plan.steps.append(
        OperationStep(id="drill", part_id=part_id, expected_effect="Drill the plate")
    )
    before = original.model_dump_json()
    patch = PlanPatch.model_validate(
        {
            "basis": basis(original),
            "construction_assignments": [
                {"step_id": "blank", "feature_ids": ["bore"]},
                {"step_id": "drill", "feature_ids": ["bore"]},
            ],
        }
    )
    if part_id == "main":
        with pytest.raises(ValueError, match="belongs to more than one"):
            apply_plan_patch(original, patch, contract())
    else:
        result = apply_plan_patch(original, patch, contract())
        assert result.operation_plan.steps[0].feature_ids == ["body", "bore"]
        assert result.operation_plan.steps[-1].feature_ids == ["bore"]
    assert original.model_dump_json() == before


def test_missing_features_remain_unverified_intent_not_a_claimed_cad_effect():
    original, doc = plan(), document()
    before = doc.model_dump_json()
    result = apply_plan_patch(
        original,
        PlanPatch.model_validate(
            {
                "basis": basis(original, doc),
                "construction_assignments": [{"step_id": "blank", "feature_ids": ["bore"]}],
            }
        ),
        contract(),
        doc,
    )
    report = coverage(snapshot(result, 3, contract()), doc)
    row = next(row for row in report["steps"] if row["step_id"] == "blank")
    assert row["missing_feature_ids"] == ["bore"]
    assert not row["graph_present"] and not row["effect_verified"]
    assert not report["graph_complete"]
    assert doc.model_dump_json() == before


@pytest.mark.parametrize("stale", [False, True])
def test_assignment_and_cad_publish_together_under_existing_cas(construction, stale):
    service, sid, job, source = construction
    initial = canonical(construction)
    previous = deepcopy(initial["design_plan"])
    edited = source.model_copy(deep=True)
    edited.features.append(
        Feature(id="cap", name="Authored cap", kind="box", params={"x": 10, "y": 9, "z": 4})
    )
    edited.outputs = ["cap"]
    patch = PlanPatch.model_validate(
        {
            "basis": service.store.edit_basis(sid, "owner"),
            "construction_assignments": [{"step_id": "base", "feature_ids": ["cap"]}],
        }
    )
    prepared = prepare_construction(
        service, sid, "owner", job["id"], edited, patch, author_intent="10 mm wide"
    )
    expected = deepcopy(previous)
    expected["operation_plan"]["steps"][0]["feature_ids"] = ["body", "cap"]
    assert prepared.successor_snapshot["plan"] == expected
    assert canonical(construction) == initial
    candidate(construction, prepared)
    if stale:
        service.store.update(sid, plan_version=9)
        before = canonical(construction)
        with pytest.raises(WorkspaceError):
            service.store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
        assert canonical(construction) == before
        assert before["selected_revision"] == BASE
        assert before["design_plan"] == previous
    else:
        receipt = service.store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
        current = canonical(construction)
        assert receipt["applied"] and receipt["plan_applied"] and receipt["cad_applied"]
        assert current["design_plan"] == expected
        assert current["selected_revision"] == RESULT and current["plan_version"] == 2
        assert current["accepted_revision"] is None
        assert service.store.revision(sid, RESULT)["operation_record"]["plan_snapshot"] == (
            prepared.successor_snapshot
        )


def test_legacy_patch_serialization_does_not_gain_assignment_defaults():
    patch = PlanPatch(basis=basis(plan()), verification=["Inspect geometry"])
    assert set(patch.model_dump(exclude_unset=True)) == {"basis", "verification"}
