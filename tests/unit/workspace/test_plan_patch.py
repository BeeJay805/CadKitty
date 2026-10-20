"""Focused plan repairs retain authority and cannot claim an unapplied CAD effect."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge_mind.workspace.completion_models import (
    FunctionalInterface,
    InterfaceDuty,
    InterfaceFeature,
    InterfaceState,
    PreparedCheck,
    RequirementDisposition,
    Subsystem,
)
from forge_mind.workspace.document import DesignDocument, Feature
from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.inspection_models import InspectionCharacteristic
from forge_mind.workspace.models import (
    Contract,
    DesignPlan,
    DimensionFact,
    OperationFrame,
    OperationPlan,
    OperationStep,
    Requirement,
)
from forge_mind.workspace.operation_plan import coverage, snapshot
from forge_mind.workspace.plan_patch import PlanPatch, apply_plan_patch


def contract() -> Contract:
    return Contract(
        requirements=[
            Requirement(
                id="fit", text="Fit the plate", source_quote="Fit the plate", part_id="main"
            )
        ]
    )


def document() -> DesignDocument:
    return DesignDocument(
        variables={"width": 30},
        features=[
            Feature(
                id="body",
                name="Blank",
                kind="box",
                params={"y": 12, "z": 5},
                bindings={"x": "width"},
            )
        ],
        outputs=["body"],
    )


def check(identifier: str = "grip") -> PreparedCheck:
    return PreparedCheck.model_validate(
        {
            "id": identifier,
            "request": {
                "tool": "fastener_stack",
                "units": "mm",
                "screw_length": {"low": 12.0, "high": 12.0},
                "grip_layers": [{"low": 5.0, "high": 5.0}],
                "required_engagement": 3.0,
                "available_depth": {"low": 8.0, "high": 8.0},
            },
        }
    )


def interface(identifier: str = "connection") -> FunctionalInterface:
    return FunctionalInterface(
        id=identifier,
        requirement_ids=["fit"],
        participant_paths=["plate", "pin"],
        installed_occurrence_paths=["pin"],
        required_features=[InterfaceFeature(occurrence_path="plate", feature_id="body")],
        states=[InterfaceState(id="rest")],
        duties=[
            InterfaceDuty(
                id="retain",
                kind="retention",
                proof_scope="physical_function",
                state_ids=["rest"],
                missing_inputs=["Retention method and load are unknown"],
            )
        ],
    )


def plan() -> DesignPlan:
    return DesignPlan(
        objective="Make an editable plate",
        parts=["Plate", "Pin"],
        interfaces=["Pin connection"],
        assembly_sequence=["Place the pin"],
        verification=["Inspect the exported geometry"],
        unresolved=["Physical fit remains unknown"],
        operation_plan=OperationPlan(
            part_ids=["main"],
            checkpoint="Plan retained",
            unresolved=["Holding remains unverified"],
            frames=[OperationFrame(id="datum", description="Plate datum", part_id="main")],
            dimensions=[
                DimensionFact(
                    id="width_original",
                    value=30,
                    source="customer",
                    source_quote="Use 30 mm width",
                    binding_name="width",
                    part_id="main",
                    requirement_ids=["fit"],
                )
            ],
            steps=[
                OperationStep(
                    id="blank",
                    part_id="main",
                    feature_ids=["body"],
                    requirement_ids=["fit"],
                    binding_names=["width"],
                    frame_id="datum",
                    expected_effect="Build the blank",
                ),
                OperationStep(
                    id="inspect",
                    part_id="main",
                    kind="inspection",
                    depends_on=["blank"],
                    referenced_feature_ids=["body"],
                    requirement_ids=["fit"],
                    expected_effect="Inspect the blank",
                ),
            ],
        ),
        subsystems=[Subsystem(id="mount", name="Mount", requirement_ids=["fit"], checks=[check()])],
        requirement_dispositions=[RequirementDisposition(requirement_id="fit", check_ids=["grip"])],
        inspection=[
            InspectionCharacteristic(
                id="width",
                label="Plate width",
                requirement_ids=["fit"],
                characteristic="Width of the plate",
                datum="Declared plate datum",
                nominal_mm=30.0,
                basis="Customer dimension",
                physical_method="Measure after manufacture",
                missing_evidence=["Physical width is unknown"],
            )
        ],
        functional_interfaces=[interface()],
    )


def basis(value: DesignPlan, doc: DesignDocument | None = None) -> EditBasis:
    return EditBasis(
        revision_id="a" * 32 if doc is not None else None,
        document_hash=doc.fingerprint() if doc is not None else None,
        plan_version=2,
        plan_hash=snapshot(value, 2, contract())["plan_hash"],
        contract_hash=contract().fingerprint(),
        input_version=0,
        source_fingerprint="d" * 64,
    )


def test_provided_prose_changes_retain_every_other_plan_record():
    original = plan()
    before = original.model_dump()
    result = apply_plan_patch(
        original,
        PlanPatch(basis=basis(original), objective="Make the corrected plate", unresolved=[]),
        contract(),
    )
    expected = original.model_dump()
    expected.update(objective="Make the corrected plate", unresolved=[])
    assert result.model_dump() == expected
    assert original.model_dump() == before
    result.parts.append("Independent result change")
    assert original.parts == ["Plate", "Pin"]


def test_typed_upserts_preserve_order_and_unrelated_work():
    original = plan()
    inspection = original.operation_plan.steps[1].model_copy(
        update={"expected_effect": "Inspect the corrected local Feature references"}
    )
    frame = original.operation_plan.frames[0].model_copy(update={"description": "Corrected datum"})
    subsystem = original.subsystems[0].model_copy(update={"checks": [check(), check("tip")]})
    result = apply_plan_patch(
        original,
        PlanPatch(
            basis=basis(original),
            operation_steps=[inspection],
            operation_frames=[frame],
            subsystems=[subsystem],
            functional_interfaces=[interface("secondary")],
        ),
        contract(),
    )
    assert [step.id for step in result.operation_plan.steps] == ["blank", "inspect"]
    assert (
        result.operation_plan.steps[0].model_dump() == original.operation_plan.steps[0].model_dump()
    )
    assert result.operation_plan.frames[0].description == "Corrected datum"
    assert [item.id for item in result.subsystems[0].checks] == ["grip", "tip"]
    assert [item.id for item in result.functional_interfaces] == ["connection", "secondary"]
    assert result.operation_plan.checkpoint == original.operation_plan.checkpoint
    assert result.operation_plan.unresolved == original.operation_plan.unresolved


def test_plan_only_repair_cannot_create_missing_features_or_complete_them():
    original, doc = plan(), document()
    saved_doc = doc.model_dump_json()
    result = apply_plan_patch(
        original,
        PlanPatch(
            basis=basis(original, doc),
            operation_steps=[
                OperationStep(
                    id="drill",
                    part_id="main",
                    depends_on=["blank"],
                    feature_ids=["hole", "drilled"],
                    requirement_ids=["fit"],
                    expected_effect="Drill the mounting hole",
                )
            ],
        ),
        contract(),
        doc,
    )
    assert doc.model_dump_json() == saved_doc
    report = coverage(snapshot(result, 3, contract()), doc)
    drill = next(row for row in report["steps"] if row["step_id"] == "drill")
    assert drill["missing_feature_ids"] == ["drilled", "hole"]
    assert not drill["graph_present"]
    assert not drill["effect_verified"]
    assert not report["graph_complete"]


def test_dimension_successor_retains_original_source_and_can_precede_cad_edit():
    original, doc = plan(), document()
    successor = DimensionFact(
        id="width_updated",
        value=32,
        source="customer",
        source_quote="Change the width to 32 mm",
        part_id="main",
        binding_name="width",
        requirement_ids=["fit"],
        supersedes_fact_id="width_original",
    )
    result = apply_plan_patch(
        original, PlanPatch(basis=basis(original, doc), dimensions=[successor]), contract(), doc
    )
    assert result.operation_plan.dimensions[0].model_dump() == (
        original.operation_plan.dimensions[0].model_dump()
    )
    assert result.operation_plan.dimensions[-1].value == 32
    assert doc.variables["width"] == 30


@pytest.mark.parametrize(
    "problem", ["rewrite_source", "parallel_fact", "cycle", "unknown_requirement"]
)
def test_invalid_dimension_update_is_atomic(problem):
    original = plan()
    old = original.operation_plan.dimensions[0]
    if problem == "rewrite_source":
        fact = old.model_copy(update={"value": 32})
    elif problem == "parallel_fact":
        fact = old.model_copy(update={"id": "parallel", "value": 32})
    elif problem == "cycle":
        fact = old.model_copy(update={"id": "cycle", "supersedes_fact_id": "cycle"})
    else:
        fact = old.model_copy(
            update={"id": "new_width", "requirement_ids": ["unknown"], "binding_name": "other"}
        )
    before = original.model_dump_json()
    with pytest.raises(ValueError):
        apply_plan_patch(original, PlanPatch(basis=basis(original), dimensions=[fact]), contract())
    assert original.model_dump_json() == before


@pytest.mark.parametrize(
    "field", ["operation_steps", "operation_frames", "subsystems", "inspection"]
)
def test_duplicate_patch_ids_are_rejected_without_mutation(field):
    original = plan()
    record = {
        "operation_steps": original.operation_plan.steps[0],
        "operation_frames": original.operation_plan.frames[0],
        "subsystems": original.subsystems[0],
        "inspection": original.inspection[0],
    }[field]
    before = original.model_dump_json()
    with pytest.raises(ValueError, match="must be unique"):
        apply_plan_patch(
            original,
            PlanPatch.model_validate({"basis": basis(original), field: [record, record]}),
            contract(),
        )
    assert original.model_dump_json() == before


@pytest.mark.parametrize(
    "problem", ["owned_features", "operation_requirements", "check_ids", "prepared_checks"]
)
def test_patch_cannot_remove_old_obligations(problem):
    original = plan()
    changes = {}
    if problem in {"owned_features", "operation_requirements"}:
        field = "feature_ids" if problem == "owned_features" else "requirement_ids"
        changes["operation_steps"] = [
            original.operation_plan.steps[0].model_copy(update={field: []})
        ]
    elif problem == "check_ids":
        changes["requirement_dispositions"] = [RequirementDisposition(requirement_id="fit")]
    else:
        changes["subsystems"] = [original.subsystems[0].model_copy(update={"checks": []})]
    before = original.model_dump_json()
    with pytest.raises(ValueError):
        apply_plan_patch(
            original, PlanPatch.model_validate({"basis": basis(original), **changes}), contract()
        )
    assert original.model_dump_json() == before


def test_part_scope_correction_cannot_move_a_stable_operation():
    original = plan()
    change = original.operation_plan.steps[0].model_copy(update={"part_id": "other"})
    with pytest.raises(ValueError):
        apply_plan_patch(
            original, PlanPatch(basis=basis(original), operation_steps=[change]), contract()
        )


@pytest.mark.parametrize("field", ["plan_hash", "contract_hash", "document_hash"])
def test_stale_basis_rejection_preserves_plan_contract_and_document(field):
    original, frozen, doc = plan(), contract(), document()
    stale = basis(original, doc).model_copy(update={field: "0" * 64})
    before = (original.model_dump_json(), frozen.model_dump_json(), doc.model_dump_json())
    with pytest.raises(ValueError, match="basis changed"):
        apply_plan_patch(
            original, PlanPatch(basis=stale, verification=["Updated check"]), frozen, doc
        )
    assert (original.model_dump_json(), frozen.model_dump_json(), doc.model_dump_json()) == before


def test_operation_patch_requires_an_existing_operation_plan():
    original = DesignPlan(objective="Future work")
    with pytest.raises(ValueError, match="existing operation plan"):
        apply_plan_patch(
            original,
            PlanPatch(
                basis=basis(original),
                operation_steps=[
                    OperationStep(id="build", part_id="main", expected_effect="Build")
                ],
            ),
            contract(),
        )


@pytest.mark.parametrize(
    "payload", [{}, {"objective": None}, {"parts": None}, {"status": "complete"}]
)
def test_empty_null_and_authored_status_patches_are_rejected(payload):
    with pytest.raises(ValidationError):
        PlanPatch.model_validate({"basis": basis(plan()), **payload})


def test_arbitrary_json_pointer_and_contract_replacement_are_rejected():
    for extra in ({"path": "/operation_plan/steps/0"}, {"contract": contract().model_dump()}):
        with pytest.raises(ValidationError):
            PlanPatch.model_validate({"basis": basis(plan()), "verification": [], **extra})
