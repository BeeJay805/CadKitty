"""Lower startup part intent into canonical, still-pending construction work."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from forge_mind.workspace.completion_models import Subsystem
from forge_mind.workspace.intent_attribution import IntentIssue
from forge_mind.workspace.models import (
    Contract,
    DesignPlan,
    OperationPlan,
    OperationStep,
    PlanName,
    Record,
)
from forge_mind.workspace.operation_plan import validate_plan

RequirementId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class IntentPart(Record):
    """An authored part role does not verify procurement or manufacture."""

    id: PlanName
    role: Literal["make", "buy", "reference"]
    purpose: str = Field(min_length=1, max_length=2000)
    requirement_ids: list[RequirementId] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_requirement_references(self) -> IntentPart:
        if len(self.requirement_ids) != len(set(self.requirement_ids)):
            raise ValueError("Intent part requirement references must be unique")
        return self


class PlanIntent(Record):
    objective: str = Field(min_length=1, max_length=2000)
    parts: list[IntentPart] = Field(min_length=1)
    first_part_id: PlanName
    unresolved: list[str] = Field(default_factory=list)
    attribution_issues: list[IntentIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_parts_and_make_target(self) -> PlanIntent:
        parts = {part.id: part for part in self.parts}
        if len(parts) != len(self.parts):
            raise ValueError("Intent part IDs must be unique")
        first = parts.get(self.first_part_id)
        if first is None:
            raise ValueError("Intent first_part_id must reference an existing part")
        if first.role != "make":
            raise ValueError("Intent first_part_id must reference a make part")
        return self


def lower_intent(intent: PlanIntent, contract: Contract) -> DesignPlan:
    """Return canonical intent without inventing geometry, checks or evidence."""
    intent = PlanIntent.model_validate(intent.model_dump())
    contract = Contract.model_validate(contract.model_dump())
    if intent.attribution_issues:
        raise ValueError("Resolve material intent questions before admitting a construction plan")
    prose = [f"{part.id} ({part.role}): {part.purpose}" for part in intent.parts]
    part_ids = [part.id for part in intent.parts]
    plan = DesignPlan(
        objective=intent.objective,
        parts=prose,
        unresolved=list(intent.unresolved),
        operation_plan=OperationPlan(
            part_ids=part_ids,
            steps=[
                OperationStep(
                    id=part.id,
                    part_id=part.id,
                    expected_effect=description,
                    requirement_ids=list(part.requirement_ids),
                )
                for part, description in zip(intent.parts, prose, strict=True)
            ],
            checkpoint=f"Construct the first meaningful make part: {intent.first_part_id}",
            unresolved=list(intent.unresolved),
        ),
        subsystems=[
            Subsystem(
                id=f"startup_part_{index}",
                name=part.purpose[:160],
                part_ids=[part.id],
                requirement_ids=list(part.requirement_ids),
            )
            for index, part in enumerate(intent.parts)
        ]
        + [
            Subsystem(
                id="request_scope",
                name="Request scope",
                requirement_ids=[requirement.id for requirement in contract.requirements],
                part_ids=part_ids,
            )
        ],
    )
    validate_plan(plan, contract)
    return plan
