"""Frozen, domain-independent design comparisons. Objectives are not acceptance claims."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from forge_mind.workspace.document import Name
from forge_mind.workspace.models import Record


class StudyObjective(Record):
    metric: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    direction: Literal["minimize", "maximize"]


class StudyConstraint(Record):
    metric: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    operator: Literal["ge", "le"]
    value: float


class StudyEvaluator(Record):
    kind: Literal["geometry", "structure", "flow", "interface_geometry"]
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def settings_schema(self) -> StudyEvaluator:
        if self.kind == "geometry" and self.settings:
            raise ValueError("Geometry measurements have no adjustable physical assumptions")
        if self.kind == "interface_geometry":
            from forge_mind.workspace.interface_geometry import InterfaceGeometrySettings

            self.settings = InterfaceGeometrySettings.model_validate(self.settings).model_dump()
        if self.kind == "structure":
            from forge_mind.workspace.structural_models import StructuralStudy

            self.settings = StructuralStudy.model_validate(self.settings).model_dump()
        if self.kind == "flow":
            from forge_mind.workspace.flow_models import FlowSettings

            self.settings = FlowSettings.model_validate(self.settings).model_dump()
        return self


class DesignStudySpec(Record):
    name: str = Field(min_length=1, max_length=160)
    objectives: list[StudyObjective] = Field(min_length=1)
    constraints: list[StudyConstraint] = Field(default_factory=list)
    variables: dict[Name, tuple[float, float]] = Field(default_factory=dict)
    evaluators: list[StudyEvaluator] = Field(min_length=1)
    comparison_tolerance: float = Field(default=0.02, ge=0, lt=1)

    @model_validator(mode="after")
    def consistent(self) -> DesignStudySpec:
        if not self.name.strip():
            raise ValueError("Study name cannot be blank")
        kinds = [e.kind for e in self.evaluators]
        metrics = [o.metric for o in self.objectives]
        if len(set(kinds)) != len(kinds) or len(set(metrics)) != len(metrics):
            raise ValueError("Evaluator kinds and objective metrics must be distinct")
        if "geometry" not in kinds:
            raise ValueError("Every study requires exported geometry measurements")
        from forge_mind.workspace.design_evaluators import metric_catalog

        available = {
            m["key"]
            for evaluator in self.evaluators
            for m in metric_catalog(evaluator.kind, evaluator.settings)
        }
        if any(m not in available for m in [*metrics, *(c.metric for c in self.constraints)]):
            raise ValueError("Objective or constraint is not measured by the selected evaluators")
        if any(lo >= hi for lo, hi in self.variables.values()):
            raise ValueError("Search variables require finite increasing lower and upper bounds")
        return self


class CreateDesignStudy(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    input_version: int = Field(ge=0)
    spec: DesignStudySpec


class AddStudyCandidate(Record):
    label: str = Field(default="Candidate", min_length=1, max_length=160)
    parameters: dict[Name, float] | None = None
    revision_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")

    @model_validator(mode="after")
    def one_source(self) -> AddStudyCandidate:
        if (self.parameters is None) == (self.revision_id is None):
            raise ValueError("Supply parameter changes or one existing candidate revision")
        return self


class RunDesignStudy(Record):
    retry_failed: bool = False


class SearchDesignStudy(Record):
    rounds: int = Field(default=2, ge=1)
    step_fraction: float = Field(default=0.25, gt=0, le=0.5)


class AcceptStudyCandidate(Record):
    candidate_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    expected_current_revision: str = Field(pattern=r"^[a-f0-9]{32}$")
