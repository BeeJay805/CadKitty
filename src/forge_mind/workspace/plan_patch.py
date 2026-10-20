"""Typed plan upserts preserve frozen work; a plan change never applies CAD."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypeVar

from pydantic import BaseModel, Field, StrictStr, model_validator

from forge_mind.workspace.check_amendment import request_hash
from forge_mind.workspace.completion_models import (
    FunctionalInterface,
    RequirementDisposition,
    Subsystem,
)
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.inspection_models import InspectionCharacteristic
from forge_mind.workspace.models import (
    Contract,
    DesignPlan,
    DimensionFact,
    OperationFrame,
    OperationStep,
    PlanName,
    Record,
)
from forge_mind.workspace.operation_plan import snapshot, validate_plan
from forge_mind.workspace.verification_models import VerificationSubjectBinding

_PROSE = ("objective", "parts", "interfaces", "assembly_sequence", "verification", "unresolved")
_OPERATIONS = (
    "part_ids",
    "operation_steps",
    "operation_frames",
    "dimensions",
    "construction_assignments",
)
_COLLECTIONS = (
    "subsystems",
    "inspection",
    "requirement_dispositions",
    "functional_interfaces",
    "verification_bindings",
)
_RecordType = TypeVar("_RecordType", bound=BaseModel)


class ConstructionAssignment(Record):
    """Add explicit Feature ownership without rewriting an existing step's duties."""

    step_id: PlanName
    feature_ids: list[PlanName] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_features(self) -> ConstructionAssignment:
        if len(self.feature_ids) != len(set(self.feature_ids)):
            raise ValueError("Construction assignment Feature IDs must be unique")
        return self


class PlanPatch(Record):
    basis: EditBasis
    objective: StrictStr | None = Field(default=None, min_length=1, max_length=2000)
    parts: list[StrictStr] | None = None
    interfaces: list[StrictStr] | None = None
    assembly_sequence: list[StrictStr] | None = None
    verification: list[StrictStr] | None = None
    unresolved: list[StrictStr] | None = None
    part_ids: list[PlanName] = Field(
        default_factory=list,
        description="Add declared operation part IDs; existing part IDs remain unchanged.",
    )
    operation_steps: list[OperationStep] = Field(default_factory=list)
    construction_assignments: list[ConstructionAssignment] = Field(
        default_factory=list,
        description=(
            "Union explicit Feature IDs into an existing construction step. All other step "
            "fields and obligations remain unchanged. Cannot overlap operation_steps updates."
        ),
    )
    operation_frames: list[OperationFrame] = Field(default_factory=list)
    dimensions: list[DimensionFact] = Field(default_factory=list)
    subsystems: list[Subsystem] = Field(default_factory=list)
    inspection: list[InspectionCharacteristic] = Field(default_factory=list)
    requirement_dispositions: list[RequirementDisposition] = Field(default_factory=list)
    functional_interfaces: list[FunctionalInterface] = Field(default_factory=list)
    verification_bindings: list[VerificationSubjectBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def explicit_changes(self) -> PlanPatch:
        if len(self.part_ids) != len(set(self.part_ids)):
            raise ValueError("Added operation part IDs must be unique")
        assigned_steps = [assignment.step_id for assignment in self.construction_assignments]
        if len(assigned_steps) != len(set(assigned_steps)):
            raise ValueError("Construction assignment step IDs must be unique")
        if set(assigned_steps) & {step.id for step in self.operation_steps}:
            raise ValueError("Construction assignments cannot overlap full operation step updates")
        for field in _PROSE:
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"Omit {field} to retain it; explicit null is not a plan change")
        if not any(field in self.model_fields_set for field in _PROSE) and not any(
            getattr(self, field) for field in (*_OPERATIONS, *_COLLECTIONS)
        ):
            raise ValueError("A plan patch must contain at least one change")
        return self


def _upsert(
    previous: list[_RecordType], changes: list[_RecordType], key: str = "id"
) -> list[_RecordType]:
    ids = [str(getattr(record, key)) for record in changes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Plan patch {key} values must be unique")
    result = {str(getattr(record, key)): record.model_copy(deep=True) for record in previous}
    result.update({str(getattr(record, key)): record.model_copy(deep=True) for record in changes})
    return list(result.values())


def _retain_obligations(previous: DesignPlan, result: DesignPlan) -> None:
    if previous.operation_plan is not None and result.operation_plan is not None:
        steps = {step.id: step for step in result.operation_plan.steps}
        for old in previous.operation_plan.steps:
            current = steps[old.id]
            if set(old.feature_ids) - set(current.feature_ids):
                raise ValueError("A plan patch cannot remove owned construction Features")
            if set(old.requirement_ids) - set(current.requirement_ids):
                raise ValueError("A plan patch cannot remove operation requirement obligations")
    inspections = {row.id: row for row in result.inspection}
    for old_inspection in previous.inspection:
        if set(old_inspection.requirement_ids) - set(
            inspections[old_inspection.id].requirement_ids
        ):
            raise ValueError("A plan patch cannot remove inspection requirement obligations")
    dispositions = {row.requirement_id: row for row in result.requirement_dispositions}
    for old_disposition in previous.requirement_dispositions:
        for field in ("check_ids", "missing_inputs", "unsupported_checks", "physical_validation"):
            if set(getattr(old_disposition, field)) - set(
                getattr(dispositions[old_disposition.requirement_id], field)
            ):
                raise ValueError("A plan patch cannot remove requirement check obligations")


def _unambiguous_dimensions(plan: DesignPlan) -> None:
    if plan.operation_plan is None:
        return
    dimensions = plan.operation_plan.dimensions
    superseded = {fact.supersedes_fact_id for fact in dimensions}
    identities: set[tuple[str, str | None, str]] = set()
    for fact in dimensions:
        if fact.id in superseded or fact.binding_name is None:
            continue
        identity = (fact.binding_name, fact.part_id, fact.units)
        if identity in identities:
            raise ValueError("Dimension facts with the same binding require explicit supersession")
        identities.add(identity)


def apply_plan_patch(
    plan: DesignPlan,
    patch: PlanPatch,
    contract: Contract,
    document: DesignDocument | None = None,
    *,
    previous_raw: dict[str, Any] | None = None,
) -> DesignPlan:
    """Validate an in-memory candidate; the Store compares live versions and publishes."""
    original = plan.model_copy(deep=True)
    patch = PlanPatch.model_validate(patch.model_dump(exclude_unset=True))
    contract = contract.model_copy(deep=True)
    captured = snapshot(original, patch.basis.plan_version, contract)
    previous: Any = original
    if previous_raw is not None:
        previous = deepcopy(previous_raw)
        if DesignPlan.model_validate(previous).model_dump(mode="json") != original.model_dump(
            mode="json"
        ):
            raise ValueError("Raw previous plan differs from the typed patch baseline")
    previous_hash = request_hash(previous) if previous_raw is not None else captured["plan_hash"]
    if patch.basis.plan_hash != previous_hash:
        raise ValueError("Plan patch basis changed; refresh the current plan")
    if patch.basis.contract_hash != contract.fingerprint():
        raise ValueError("Plan patch contract basis changed; refresh the frozen requirements")
    if document is not None:
        document = document.model_copy(deep=True)
        if patch.basis.document_hash != document.fingerprint():
            raise ValueError("Plan patch document basis changed; refresh the source revision")
    result = original.model_copy(deep=True)
    for field in _PROSE:
        if field in patch.model_fields_set:
            setattr(result, field, getattr(patch, field))
    if any(getattr(patch, field) for field in _OPERATIONS):
        if result.operation_plan is None or original.operation_plan is None:
            raise ValueError("Operation upserts require an existing operation plan")
        result.operation_plan.part_ids = list(
            dict.fromkeys([*result.operation_plan.part_ids, *patch.part_ids])
        )
        result.operation_plan.steps = _upsert(result.operation_plan.steps, patch.operation_steps)
        result.operation_plan.frames = _upsert(result.operation_plan.frames, patch.operation_frames)
        result.operation_plan.dimensions = _upsert(
            result.operation_plan.dimensions, patch.dimensions
        )
        for assignment in patch.construction_assignments:
            # Only an admitted predecessor step can receive this narrow update.
            previous_steps = [
                step for step in original.operation_plan.steps if step.id == assignment.step_id
            ]
            if len(previous_steps) != 1 or previous_steps[0].kind != "construction":
                raise ValueError("Construction assignment requires one existing construction step")
            step = next(
                step for step in result.operation_plan.steps if step.id == assignment.step_id
            )
            step.feature_ids = list(dict.fromkeys([*step.feature_ids, *assignment.feature_ids]))
    result.subsystems = _upsert(result.subsystems, patch.subsystems)
    result.inspection = _upsert(result.inspection, patch.inspection)
    result.requirement_dispositions = _upsert(
        result.requirement_dispositions, patch.requirement_dispositions, "requirement_id"
    )
    result.functional_interfaces = _upsert(
        result.functional_interfaces, patch.functional_interfaces
    )
    result.verification_bindings = _upsert(
        result.verification_bindings, patch.verification_bindings, "subject_id"
    )
    result = DesignPlan.model_validate(result.model_dump())
    validate_plan(result, contract, previous=previous, document=document)
    _retain_obligations(original, result)
    _unambiguous_dimensions(result)
    return result
